"""E1 neural arms: one-hot GRU and a small causal transformer, plus their per-token stream losses.

A stream loss is -log p(x_t | x_<t) for t = 1..T-1, the same tokens every E0 arm scores.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import V


class GRUNet(nn.Module):
    """1-layer GRU on one-hot tokens (no embedding table), linear head."""

    def __init__(self, hidden):
        super().__init__()
        self.gru = nn.GRU(V, hidden, batch_first=True)
        self.head = nn.Linear(hidden, V)

    def forward(self, x, h=None):
        y, h = self.gru(F.one_hot(x, V).float(), h)
        return self.head(y), h


class TFNet(nn.Module):
    """Pre-norm causal transformer: 4 layers, d = 128, 4 heads, MLP 4d, learned positions, context 512."""

    def __init__(self, d=128, layers=4, heads=4, ctx=512):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, x, h=None):
        L = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        z = self.tok(x) + self.pos(torch.arange(L, device=x.device))
        return self.head(self.norm(self.body(z, mask=mask, is_causal=True))), None


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def gru_stream_loss(model, x, device):
    """State carried over the whole stream."""
    xt = torch.as_tensor(x, device=device).long()[None]
    logits, _ = model(xt[:, :-1])
    return F.cross_entropy(logits[0], xt[0, 1:], reduction="none").cpu().numpy()


@torch.no_grad()
def tf_stream_loss(model, x, device, stride=256, batch=64):
    """Sliding window: windows of ctx inputs every `stride` tokens; the first window scores all its targets,
    each later one only its last `stride`, so every scored token sees >= ctx - stride context."""
    xt = torch.as_tensor(x, device=device).long()
    n, ctx = len(x) - 1, model.ctx
    starts = list(range(0, max(n - ctx, 0) + 1, stride))
    if starts[-1] + ctx < n:
        starts.append(n - ctx)
    out = torch.empty(n, device=device)
    done = 0
    for i in range(0, len(starts), batch):
        S = starts[i:i + batch]
        L = min(ctx, n)
        inp = torch.stack([xt[s:s + L] for s in S])
        tgt = torch.stack([xt[s + 1:s + 1 + L] for s in S])
        logits, _ = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none").view(len(S), L)
        for s, row in zip(S, loss):
            out[done:s + L] = row[done - s:]
            done = s + L
    assert done == n
    return out.cpu().numpy()


def stream_loss(arch, model, x, device):
    return (gru_stream_loss if arch == "GRU-META" else tf_stream_loss)(model, x, device)


def cosine(step, total, lr):
    return 0.5 * lr * (1 + math.cos(math.pi * min(step, total) / total))

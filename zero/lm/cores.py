"""LM cores for M0. Contract: model(tokens [B,T] long) -> dict(logits [B,T,V] (logits[:,t] predicts token t+1)
and optionally nll [B,T] with nll[:,t] = -log p(tokens[:,t] | tokens[:,:t]) for t >= 1; entry 0 ignored).
If nll is present the harness uses it directly, else cross-entropy on logits.
External cores: `pkg.module:Class`, built as Class(vocab=V, d=d, **kw); must expose n_params() and
active_flops_per_token()."""
import importlib
import math

import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(a)
        return x + self.mlp(self.ln2(x))


class GPTCore(nn.Module):
    """Pre-LN causal transformer, learned positional embedding. [B,T,d] -> [B,T,d]."""

    def __init__(self, d, layers=4, heads=None, max_seq=257):
        super().__init__()
        heads = heads or next((h for h in (8, 4, 2) if d % h == 0 and d // h >= 16), 1)
        self.d, self.layers_n, self.max_seq = d, layers, max_seq
        self.pos = nn.Parameter(torch.randn(max_seq, d) * 0.02)
        self.layers = nn.ModuleList(Block(d, heads) for _ in range(layers))

    def forward(self, x):
        x = x + self.pos[: x.shape[1]]
        for b in self.layers:
            x = b(x)
        return x

    def flops_per_token(self, seq):
        # MACs/layer: qkv+proj 4d^2, MLP 8d^2, QK+AV ~ seq*d (mean context seq/2, two matmuls); FLOPs = 2*MACs
        return 2 * self.layers_n * (12 * self.d ** 2 + seq * self.d)


class LM(nn.Module):
    """Token embedding (tied head) -> core -> final LN -> logits."""

    def __init__(self, core, vocab, d):
        super().__init__()
        self.vocab, self.d, self.core = vocab, d, core
        self.emb, self.ln = nn.Embedding(vocab, d), nn.LayerNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, tokens):
        return {"logits": self.ln(self.core(self.emb(tokens))) @ self.emb.weight.T}

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def active_flops_per_token(self, seq=256):
        return self.core.flops_per_token(seq) + 2 * self.d * self.vocab


def _asmo(**kw):
    raise NotImplementedError("superseded by QF-LM, PLAN.md item 3")


CORES = {"gpt": GPTCore, "asmo": _asmo}


def build(core, vocab, d, **kw):
    """Registry name or 'pkg.module:Class' -> model obeying the contract above."""
    if ":" in core:
        mod, cls = core.split(":")
        return getattr(importlib.import_module(mod), cls)(vocab=vocab, d=d, **kw)
    return LM(CORES[core](d=d, **kw), vocab, d)


def match_params(core, target, vocab, **fixed):
    """Bisect width d (multiple of 8) so n_params is within 5% of target -> (d, n_params)."""
    n = lambda d: build(core, vocab, d, **fixed).n_params()
    lo, hi = 1, 2
    while n(8 * hi) < target:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if n(8 * mid) < target else (lo, mid)
    d = min((8 * lo, 8 * hi), key=lambda d: abs(n(d) - target))
    got = n(d)
    if abs(got / target - 1) > 0.05:
        raise ValueError(f"cannot match {target:.3g} params within 5% ({core}: d={d} gives {got})")
    return d, got

"""The TT network: the softmax that is smart alone. A stack of causal blocks [short conv -> minGRU] + SwiGLU, no
attention, so its state is flat (fixed size per stream) and it streams forward storing only results of history.

minGRU (Feng et al. 2024): h_t = (1 - z_t) h_{t-1} + z_t g(c_t), with z, c linear in the input only, so the whole
sequence is one linear recurrence, run as a parallel scan in log space (Heinsen 2023):
    log h_t = a*_t + logcumsumexp(log b_s - a*_s),  a*_t = sum_{s<=t} log(1 - z_s),  b_s = z_s g(c_s) > 0.
Training is parallel over time; generation is O(1) per token from the carried state.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


def _log_g(x):
    """log of g(x) = x + 0.5 (x >= 0), sigmoid(x) (x < 0): positive, continuous, linear growth."""
    return torch.where(x >= 0, torch.log(F.relu(x) + 0.5), -F.softplus(-x))


def _g(x):
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.w = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class MinGRU(nn.Module):
    def __init__(self, d, expand=1.5, conv=4):
        super().__init__()
        self.h = int(expand * d)
        self.conv_k = conv
        self.conv = nn.Conv1d(d, d, conv, groups=d, padding=0)
        self.zc = nn.Linear(d, 2 * self.h, bias=False)
        self.out = nn.Linear(self.h, d, bias=False)

    def forward(self, x, state=None):
        """x [B, T, d]; state = (conv_buf [B, d, k-1], h [B, H]) or None -> (y [B, T, d], new state)."""
        B, T, d = x.shape
        xc = x.transpose(1, 2)
        buf = state[0] if state is not None else xc.new_zeros(B, d, self.conv_k - 1)
        xc = torch.cat([buf, xc], 2)
        new_buf = xc[:, :, -(self.conv_k - 1):]
        u = F.silu(self.conv(xc)).transpose(1, 2)                  # causal depthwise conv
        k, c = self.zc(u).float().chunk(2, -1)
        log_a = -F.softplus(k)                                     # log(1 - z)
        log_b = -F.softplus(-k) + _log_g(c)                        # log(z g(c))
        if state is not None:
            log_h0 = torch.log(state[1].float().clamp_min(1e-30))[:, None]
        else:
            log_h0 = torch.full((B, 1, self.h), -30.0, device=x.device)    # h_0 = 0 (numerically)
        a_star = torch.cumsum(log_a, 1)
        lb = torch.cat([log_h0, log_b - a_star], 1)
        log_h = a_star + torch.logcumsumexp(lb, 1)[:, 1:]
        h = torch.exp(log_h)
        return self.out(h.to(x.dtype)), (new_buf, h[:, -1])


class SwiGLU(nn.Module):
    def __init__(self, d, mult=8 / 3):
        super().__init__()
        m = int(mult * d + 63) // 64 * 64
        self.wi = nn.Linear(d, 2 * m, bias=False)
        self.wo = nn.Linear(m, d, bias=False)

    def forward(self, x):
        a, b = self.wi(x).chunk(2, -1)
        return self.wo(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, d, expand, conv):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.mix, self.mlp = MinGRU(d, expand, conv), SwiGLU(d)

    def forward(self, x, state=None):
        y, st = self.mix(self.n1(x), state)
        x = x + y
        return x + self.mlp(self.n2(x)), st


class TT(nn.Module):
    """tokens [B, T] -> features h [B, T, d] (final-normed), logits = h @ E^T (tied embedding)."""

    def __init__(self, vocab, d=384, layers=6, expand=1.5, conv=4, grad_ckpt=False):
        super().__init__()
        self.vocab, self.d, self.n_layers, self.expand, self.conv = vocab, d, layers, expand, conv
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, expand, conv) for _ in range(layers))
        self.norm = RMSNorm(d)
        self.grad_ckpt = grad_ckpt
        nn.init.normal_(self.emb.weight, std=0.02)
        for b in self.blocks:
            for lin in (b.mix.zc, b.mlp.wi):
                nn.init.normal_(lin.weight, std=0.02)
            for lin in (b.mix.out, b.mlp.wo):                      # GPT-2 style residual scaling
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * layers))

    def features(self, tokens, state=None):
        x = self.emb(tokens)
        new = []
        for i, b in enumerate(self.blocks):
            st = state[i] if state is not None else None
            if self.grad_ckpt and self.training and state is None:
                x, s = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            else:
                x, s = b(x, st)
            new.append(s)
        return self.norm(x), new

    def logits(self, h):
        return h @ self.emb.weight.T

    def macs_per_token(self):
        """Forward multiply-accumulates per token (embedding lookup free; head d*V)."""
        b = self.blocks[0]
        H, m = b.mix.h, b.mlp.wo.in_features
        per = self.d * self.conv + self.d * 2 * H + H * self.d + self.d * 2 * m + m * self.d
        return self.n_layers * per + self.d * self.vocab

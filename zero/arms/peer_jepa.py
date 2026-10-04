"""PEER-JEPA: SingleJEPA whose encoder and predictor MLPs carry a PEER layer (He, arXiv 2407.04153).

PEER: query q(x) (BatchNorm'd), split in halves; product keys from sub-key sets C, C' (sqrt(N) each);
top-k per half -> k^2 candidates -> top-k; expert e_i(x) = gelu(u_i . x) v_i; gates softmax over the top-k;
h heads share the expert pool, outputs summed. Params scale with N, compute with h*k: the owned route to
"params grow, active compute fixed" (T3). Sub-keys are shared across heads (paper: h query networks
sharing one pool; PKM-style per-head keys would add params only).
MLP block: Linear(i,h) -> GELU -> x + PEER(x) -> Linear(h,o)  (PEER replaces the dense hidden layer).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from arms.single_jepa import SingleJEPA


class PEER(nn.Module):
    def __init__(self, dim: int, n_experts: int = 1024, heads: int = 4, topk: int = 8, d_key: int = 32):
        super().__init__()
        n = math.isqrt(n_experts)
        assert n * n == n_experts and d_key % 2 == 0 and topk <= n
        self.n, self.h, self.k, self.dk = n, heads, topk, d_key
        self.q = nn.Linear(dim, heads * d_key, bias=False)
        self.bn = nn.BatchNorm1d(heads * d_key)
        self.c1 = nn.Parameter(torch.randn(n, d_key // 2) / math.sqrt(d_key // 2))
        self.c2 = nn.Parameter(torch.randn(n, d_key // 2) / math.sqrt(d_key // 2))
        self.u = nn.Embedding(n_experts, dim)
        self.v = nn.Embedding(n_experts, dim)
        nn.init.normal_(self.u.weight, std=dim ** -0.5)
        nn.init.normal_(self.v.weight, std=(heads * topk) ** -0.5)

    def forward(self, x):
        shp = x.shape
        x = x.reshape(-1, shp[-1])
        q = self.bn(self.q(x)).view(len(x), self.h, 2, self.dk // 2)
        s1, i1 = (q[:, :, 0] @ self.c1.T).topk(self.k, -1)  # [N, h, k]
        s2, i2 = (q[:, :, 1] @ self.c2.T).topk(self.k, -1)
        cand = (s1[..., :, None] + s2[..., None, :]).flatten(-2)  # [N, h, k*k]
        s, j = cand.topk(self.k, -1)
        idx = i1.gather(-1, j // self.k) * self.n + i2.gather(-1, j % self.k)  # [N, h, k]
        a = F.gelu(torch.einsum("nd,nhkd->nhk", x, self.u(idx))) * s.softmax(-1)
        return torch.einsum("nhk,nhkd->nd", a, self.v(idx)).view(shp)

    def extra_macs(self) -> int:  # q is an nn.Linear, counted by macs(); this adds sub-key scoring + experts
        return self.h * self.n * self.dk + self.h * self.k * 2 * self.u.weight.shape[1]


class PEERMLP(nn.Module):
    def __init__(self, i, h, o, **peer):
        super().__init__()
        self.inp, self.peer, self.out = nn.Linear(i, h), PEER(h, **peer), nn.Linear(h, o)

    def forward(self, x):
        x = F.gelu(self.inp(x))
        return self.out(x + self.peer(x))


class PEERJEPA(SingleJEPA):
    def __init__(self, D: int, A: int, d: int, hidden: int = 128, n_experts: int = 1024, heads: int = 4,
                 topk: int = 8, d_key: int = 32, **kw):
        self._peer = dict(n_experts=n_experts, heads=heads, topk=topk, d_key=d_key)
        super().__init__(D, A, d, hidden=hidden, **kw)

    def _mlp(self, i, h, o, depth):
        return PEERMLP(i, h, o, **self._peer)

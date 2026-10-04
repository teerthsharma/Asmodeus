"""Shared pieces for the owned arms (contract: zero/common.py).

Conventions (same as OBSERVER-ATLAS):
  active_flops = multiply-adds per time step of the inference path `observe` (encode o_t, route, predict
                 o_{t+1}); Linear counts in*out, biases/activations/top-k ignored. EMA target excluded.
  n_params     = learned parameters; EMA copies (modules named tgt_*) excluded.
  rollout act[:, h] is the action taken at step h; output[:, h] predicts g_{h+1}.
"""
import copy
import math
import pathlib
import sys

import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import common  # noqa: E402,F401  (pins TF32 off)
from sigreg import LAM, LAM_BAL, sigreg  # noqa: E402,F401  (re-exported: one lambda table for every arm)


def mlp(i: int, h: int, o: int, depth: int = 2, act=nn.GELU) -> nn.Sequential:
    """depth hidden layers of width h."""
    dims = [i] + [h] * depth
    layers = []
    for a, b in zip(dims, dims[1:]):
        layers += [nn.Linear(a, b), act()]
    return nn.Sequential(*layers, nn.Linear(dims[-1], o))


def macs(m: nn.Module) -> int:
    """Linear in*out over the module tree plus any module's own extra_macs() (e.g. PEER retrieval)."""
    return sum(x.in_features * x.out_features if isinstance(x, nn.Linear) else
               (x.extra_macs() if hasattr(x, "extra_macs") else 0) for x in m.modules())


class Heads(nn.Module):
    """K independent one-hidden-layer ReLU MLPs, stacked so all-K is one einsum and top-1 is one gather."""

    def __init__(self, K: int, i: int, h: int, o: int):
        super().__init__()
        u = lambda fan, *s: nn.Parameter(torch.empty(*s).uniform_(-1 / math.sqrt(fan), 1 / math.sqrt(fan)))
        self.W1, self.b1 = u(i, K, i, h), u(i, K, h)
        self.W2, self.b2 = u(h, K, h, o), u(h, K, o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [N, i] (shared input) or [N, K, i] (per-head input) -> [N, K, o]."""
        if x.dim() == 2:
            x = x[:, None].expand(-1, self.W1.shape[0], -1)
        h = torch.relu(torch.einsum("nki,kih->nkh", x, self.W1) + self.b1)
        return torch.einsum("nkh,kho->nko", h, self.W2) + self.b2

    def one(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """x [N, i], idx [N] -> head idx[n] applied to x[n]: [N, o]. Only the chosen head computes."""
        h = torch.relu(torch.bmm(x[:, None], self.W1[idx])[:, 0] + self.b1[idx])
        return torch.bmm(h[:, None], self.W2[idx])[:, 0] + self.b2[idx]

    def head_macs(self) -> int:
        _, i, h = self.W1.shape
        return i * h + h * self.W2.shape[-1]


def frozen_copy(m: nn.Module) -> nn.Module:
    return copy.deepcopy(m).requires_grad_(False)


class Arm(nn.Module):
    tau = 0.996  # EMA momentum (MoP-JEPA and BYOL default)

    def n_params(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("tgt_"))

    @torch.no_grad()
    def ema_(self):
        """target <- tau*target + (1-tau)*online for every child tgt_X of child X; buffers copied."""
        for name, t in self.named_children():
            if name.startswith("tgt_"):
                o = getattr(self, name[4:])
                for pt, po in zip(t.parameters(), o.parameters()):
                    pt.lerp_(po, 1 - self.tau)
                for bt, bo in zip(t.buffers(), o.buffers()):
                    bt.copy_(bo)

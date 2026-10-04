"""Dynamics telemetry for QF-LM: eval readouts only, never training signals."""
import math
import sys
from pathlib import Path

import pytest
import torch

torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qflm import telemetry as T  # noqa: E402
from qflm.model import QFLM  # noqa: E402


def phased(v):
    return v * torch.polar(torch.ones(()), torch.rand(()) * 6.28)


def unit(v):
    return v / v.abs().pow(2).sum().sqrt()


def test_pr_endpoints_and_scale():
    """Detects PR not normalising |amp|^2 (scale-dependent) or wrong endpoints (one-hot=1, uniform=N)."""
    e = torch.zeros(64, dtype=torch.complex64); e[3] = 1
    assert abs(float(T.participation_ratio(e)) - 1.0) < 1e-5
    u = torch.ones(64, dtype=torch.complex64) / 8
    assert abs(float(T.participation_ratio(u)) - 64.0) < 1e-4
    assert abs(float(T.participation_ratio(u * 7.0)) - 64.0) < 1e-4  # scale invariant
    assert T.participation_ratio(u[None].expand(5, 64)).shape == (5,)


def test_traj_erank():
    """Detects erank that counts phase/copies as dimensions (should be 1) or collapses random states."""
    g = torch.Generator().manual_seed(0)
    v = torch.randn(64, dtype=torch.complex64, generator=g)
    same = torch.stack([phased(v) for _ in range(50)])
    assert abs(float(T.traj_erank(same)) - 1.0) < 1e-4
    rnd = torch.randn(50, 64, dtype=torch.complex64, generator=g)
    assert float(T.traj_erank(rnd)) > 30


def test_cap_hit_rate():
    """Detects wrong cap fraction or division/zero error when s_max == 0."""
    s = torch.tensor([[0, 4, 4], [1, 2, 4]])
    assert T.cap_hit_rate(s, 4) == pytest.approx(3 / 6)
    assert T.cap_hit_rate(s, 0) == 0.0


def _loop(seqs):  # list over b of list over k of vectors -> (K,B,N)
    return torch.stack([torch.stack(s) for s in seqs], 1)


def test_period2():
    """Detects flip detector flagging fixed points, missing real a,b,a,b flips, or being phase sensitive."""
    N = 64
    a = torch.zeros(N, dtype=torch.complex64); a[0] = 1
    b = torch.zeros(N, dtype=torch.complex64); b[1] = 1
    flip = [phased(x) for x in (a, b, a, b, a)]
    fixed = [phased(a) for _ in range(5)]
    assert T.period2_flip_rate(_loop([flip])) == 1.0
    assert T.period2_flip_rate(_loop([fixed])) == 0.0
    rnd = torch.randn(5, N, dtype=torch.complex64)
    assert T.period2_flip_rate(_loop([list(rnd)])) == 0.0
    assert T.period2_flip_rate(_loop([flip, flip, fixed, fixed])) == 0.5


def tiny(**kw):
    return QFLM(16, 32, s_max=4, arm="qflm", seed=0, **kw)


def test_generate_trace_bit_identical():
    """Detects trace instrumentation changing sampling (tokens/steps) or recording wrong shapes."""
    m = tiny()
    g0 = torch.Generator().manual_seed(5)
    x0, s0 = m.generate(20, B=2, generator=g0, return_steps=True)
    g1 = torch.Generator().manual_seed(5)
    tr = []
    x1, s1 = m.generate(20, B=2, generator=g1, return_steps=True, trace=tr)
    assert torch.equal(x0, x1) and torch.equal(s0, s1)
    assert len(tr) == 20 and tr[0]["post"].shape == (2, 16) and tr[0]["loop"].shape == (5, 2, 16)
    assert tr[0]["steps"].shape == (2,)


def test_dynamics_tiny():
    """Detects missing/non-finite keys or values outside their theoretical ranges."""
    m = tiny()
    d = T.dynamics(m, n_tokens=32, B=2, seed=0)
    keys = {"pr_mean", "pr_min", "traj_erank", "cap_hit_rate", "period2_flip_rate", "n_tokens", "N"}
    assert keys <= set(d) and all(math.isfinite(d[k]) for k in keys)
    assert 1 - 1e-4 <= d["pr_min"] <= d["pr_mean"] <= 16 + 1e-4
    assert 0 <= d["cap_hit_rate"] <= 1 and 0 <= d["period2_flip_rate"] <= 1
    assert 1 - 1e-4 <= d["traj_erank"] <= 16 + 1e-4

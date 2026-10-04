import math

import pytest
import torch

from qflm.stability import (butterfly_apply, log_born, null_record_logp,
                            orth_error, renorm_psi)


def rand_angles(N, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(int(math.log2(N)), N // 2, generator=g) * 2 * math.pi


def make_instrument(N, V, seed=0):
    """Reference instrument: M row-stochastic (softmax of randn rows), Q from random butterfly."""
    g = torch.Generator().manual_seed(seed)
    M = torch.softmax(torch.randn(N, V, generator=g), dim=1)
    ang = rand_angles(N, seed + 1)
    return M, ang


def test_psi_norm_no_drift_2k_tokens():
    """Failure mode: psi norm drifts over a long think/Luders chain without renorm."""
    torch.manual_seed(0)
    N, V = 256, 512
    M, ang = make_instrument(N, V)
    sM = M.sqrt()
    psi = torch.randn(N, dtype=torch.complex64)
    psi = renorm_psi(psi)
    worst = 0.0
    for _ in range(2000):
        psi = butterfly_apply(butterfly_apply(psi, ang), ang)  # think x2
        q = butterfly_apply(psi, ang)                           # Q psi
        p = (M * (q.abs() ** 2)[:, None]).sum(0)
        x = torch.multinomial(p / p.sum(), 1).item()
        psi = renorm_psi(sM[:, x] * q)                          # Luders write
        assert psi.dtype == torch.complex64
        worst = max(worst, abs(psi.abs().pow(2).sum().sqrt().item() - 1))
        assert abs(torch.linalg.vector_norm(psi).item() - 1) < 1e-5
    print("guard worst norm drift", worst)
    # control: fp16 think x2000 without renorm drifts
    x = torch.randn(N)
    x = (x / x.norm()).half()
    a16 = ang.half()
    for _ in range(2000):
        x = butterfly_apply(x, a16)
    drift = abs(x.float().norm().item() - 1)
    print("fp16 control drift", drift)
    assert drift > 1e-3


def test_fp16_fp32_nll_parity():
    """Failure mode: squaring fp16 amplitudes underflows/rounds, corrupting nll."""
    torch.manual_seed(1)
    N, V = 2048, 4096
    M, ang = make_instrument(N, V)
    psi = renorm_psi(torch.randn(N))
    q = butterfly_apply(psi, ang)
    vs = torch.randint(0, V, (64,))
    a = M.sqrt()[:, vs].T * q                      # (64, N)
    ref = -torch.log((a.double() ** 2).sum(-1))
    naive = -torch.log((a.half() ** 2).sum(-1)).float()
    bad = ~torch.isfinite(naive) | ((naive - ref).abs() > 1e-2)
    print("naive fp16 max nll diff", (naive - ref).abs()[torch.isfinite(naive)].max().item(),
          "nonfinite", (~torch.isfinite(naive)).sum().item())
    assert bad.all()
    guard = -log_born(a.half())
    err = (guard - ref).abs().max().item()
    print("guard max nll diff", err)
    assert err < 1e-3


def test_null_record_no_underflow():
    """Failure mode: product of tiny probabilities across null records underflows to 0."""
    torch.manual_seed(2)
    B, S = 32, 4
    lpn = -20 - 10 * torch.rand(B, S)
    lpx = -25 - 10 * torch.rand(B, S + 1)
    out = null_record_logp(lpn, lpx)
    ref = torch.logsumexp(
        torch.cat([torch.zeros(B, 1, dtype=torch.float64), lpn.double().cumsum(-1)], -1)
        + lpx.double(), -1)
    assert torch.isfinite(out).all()
    assert ((out.double() - ref).abs() / ref.abs()).max() < 1e-4
    # naive in half precision storage (|V|=4096 probabilities): underflows to 0
    pn, px = lpn.exp().half(), lpx.exp().half()
    naive = (torch.cat([torch.ones(B, 1).half(), pn.cumprod(-1)], -1) * px).sum(-1).float().log()
    assert torch.isinf(naive).all()
    # deep fp32 case: probabilities ~e-100 underflow in fp32 even though fp32 logs are fine
    lpn2, lpx2 = lpn * 8, lpx * 8
    naive32 = (torch.cat([torch.ones(B, 1), lpn2.exp().cumprod(-1)], -1) * lpx2.exp()).sum(-1).log()
    assert torch.isinf(naive32).all()
    out2 = null_record_logp(lpn2, lpx2)
    ref2 = torch.logsumexp(
        torch.cat([torch.zeros(B, 1, dtype=torch.float64), lpn2.double().cumsum(-1)], -1)
        + lpx2.double(), -1)
    assert torch.isfinite(out2).all()
    assert ((out2.double() - ref2).abs() / ref2.abs()).max() < 1e-4
    # S=0 path
    assert torch.allclose(null_record_logp(lpn[:, :0], lpx[:, :1]), lpx[:, 0])
    # uniform null 0.5 vs direct float64 sum
    lpn3 = torch.full((B, S), math.log(0.5))
    direct = sum(0.5 ** s * lpx.double()[:, s].exp() for s in range(S + 1)).log()
    assert (null_record_logp(lpn3, lpx).double() - direct).abs().max() < 1e-4


def test_instrument_completeness_fp32():
    """Failure mode: sum_v p(v) != 1 (lost probability mass) at fp32 with large N,V."""
    torch.manual_seed(3)
    N, V = 2048, 4096
    M, ang = make_instrument(N, V)
    B = butterfly_apply(torch.eye(N), ang)          # rows = Q e_i, so Q psi = psi @ B
    for _ in range(8):
        psi = renorm_psi(torch.randn(N))
        p = (psi @ B).pow(2) @ M
        assert abs(p.sum().item() - 1) < 1e-5


def test_butterfly_orthogonality():
    """Failure mode: butterfly loses orthogonality; stored fp16 Givens products do."""
    B = butterfly_apply(torch.eye(1024), rand_angles(1024))
    e = orth_error(B).item()
    print("orth_error N=1024", e)
    assert e < 1e-5
    N = 256
    ang = rand_angles(N, 5)
    x = renorm_psi(torch.randn(N))
    for _ in range(10_000):
        x = butterfly_apply(x, ang)
    d = abs(x.norm().item() - 1)
    print("10k fp32 norm drift", d)
    assert d < 1e-4
    G = butterfly_apply(torch.eye(32), rand_angles(32, 6)).half()
    W = torch.eye(32).half()
    for _ in range(10_000):
        W = W @ G
    oe = orth_error(W.float()).item()
    print("fp16 stored-product orth_error", oe)
    assert not (oe <= 1e-3)  # also true for nan


def test_log_born_zero_amplitude():
    """Failure mode: log(0)/nan from zero amplitudes; gradient blowup at tiny amplitude."""
    a = torch.zeros(16, dtype=torch.complex64)
    assert log_born(a).item() == -math.inf
    a[3] = 0.5j
    assert abs(log_born(a).item() - math.log(0.25)) < 1e-6
    z = torch.zeros(4, 16, dtype=torch.complex64)
    assert (log_born(z) == -math.inf).all() and not torch.isnan(log_born(z)).any()
    t = torch.full((8,), 1e-30, dtype=torch.complex64).requires_grad_()
    log_born(t).backward()
    assert torch.isfinite(t.grad).all()

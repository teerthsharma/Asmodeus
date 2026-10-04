import math
import pathlib
import cmath

import pytest
import torch

from qflm import (butterfly, butterfly_matrix, probs, luders_update, think_step,
                  nll, max_reachable_fraction, capacity_M)


def setup(N, V, B=1, seed=0, dtype=torch.complex64, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    L = int(math.log2(N))
    rd = torch.float32 if dtype == torch.complex64 else torch.float64
    M = torch.softmax(torch.randn(N, V + 1, generator=g, dtype=rd) * scale, dim=1)
    theta = torch.randn(L, N // 2, generator=g, dtype=rd)
    phi = torch.randn(L, N // 2, generator=g, dtype=rd)
    psi = torch.randn(B, N, generator=g, dtype=rd) + 1j * torch.randn(B, N, generator=g, dtype=rd)
    psi = (psi / psi.norm(dim=1, keepdim=True)).to(dtype)
    tp = (0.1 * torch.randn(L, N // 2, generator=g, dtype=rd), 0.1 * torch.randn(L, N // 2, generator=g, dtype=rd),
          0.1 * torch.randn(L * N // 2, N, generator=g, dtype=rd), 0.1 * torch.randn(L * N // 2, N, generator=g, dtype=rd))
    return psi, M, theta, phi, tp


def dense_K(M, theta, phi, v):
    N = M.shape[0]
    return torch.diag(M[:, v].sqrt()).to(torch.complex64) @ butterfly_matrix(theta, phi, N)


def test_bound():
    assert max_reachable_fraction(capacity_M(2048, 4096)) >= 0.95
    N, V = 16, 4096
    assert max_reachable_fraction(capacity_M(N, V)) < 0.95
    for s in range(5):
        M = setup(N, V, seed=s, scale=8.0)[1]
        assert max_reachable_fraction(M) < 0.95
    N, V = 8, 20
    _, M, th, ph, _ = setup(N, V)
    tot = 0.0
    for v in range(V):
        K = dense_K(M, th, ph, v)
        ev = torch.linalg.eigvalsh(K.conj().T @ K).max().item()
        assert abs(ev - M[:, v].max().item()) < 1e-5
        tot += ev
    assert tot <= N


def test_completeness():
    psi, M, th, ph, _ = setup(64, 100, B=8)
    assert (probs(psi, M, th, ph).sum(1) - 1).abs().max() < 1e-6
    N, V = 16, 10
    _, M, th, ph, _ = setup(N, V)
    S = sum(dense_K(M, th, ph, v).conj().T @ dense_K(M, th, ph, v) for v in range(V + 1))
    assert (S - torch.eye(N)).abs().max() < 1e-5


def test_norm_rollout():
    N, V = 16, 12
    psi, M, th, ph, tp = setup(N, V, seed=3)
    for _ in range(50):
        p = probs(psi, M, th, ph)
        v = torch.multinomial(p, 1)[:, 0]
        psi = luders_update(psi, v, M, th, ph)
        assert abs(psi.norm().item() - 1) < 1e-6
        psi = think_step(psi, tp)
        assert abs(psi.norm().item() - 1) < 1e-6


def test_phase_invariance():
    N, V = 16, 10
    psi, M, th, ph, tp = setup(N, V, B=3, seed=4)
    z = cmath.exp(0.77j)
    assert (probs(z * psi, M, th, ph) - probs(psi, M, th, ph)).abs().max() < 1e-6
    assert (think_step(z * psi, tp) - z * think_step(psi, tp)).abs().max() < 1e-5
    toks = torch.randint(0, V, (3, 6))
    psi0 = torch.zeros(N, dtype=torch.complex64); psi0[0] = 1
    a = nll(toks, M, th, ph, tp, psi0=psi0)
    b = nll(toks, M, th, ph, tp, psi0=z * psi0)
    assert abs(a.item() - b.item()) < 1e-5


def test_butterfly_unitary():
    N = 32
    psi, _, th, ph, _ = setup(N, 4, B=2)
    Q = butterfly_matrix(th, ph, N)
    assert (Q.conj().T @ Q - torch.eye(N)).abs().max() < 1e-5
    assert (butterfly(psi, th, ph) - psi @ Q.T).abs().max() < 1e-5


def test_no_embedding():
    for f in pathlib.Path(__file__).parent.parent.joinpath("qflm").glob("*.py"):
        s = f.read_text().lower()
        for bad in ("embedding", "one_hot"):
            assert bad not in s, (f, bad)


def test_nll_grads_and_smax0():
    N, V, T, B = 16, 30, 12, 4
    _, M0, th, ph, tp = setup(N, V)
    logits = torch.randn(N, V + 1, requires_grad=True)
    th, ph = th.clone().requires_grad_(), ph.clone().requires_grad_()
    tp = tuple(t.clone().requires_grad_() for t in tp)
    toks = torch.randint(0, V, (B, T))
    loss = nll(toks, torch.softmax(logits, 1), th, ph, tp)
    assert torch.isfinite(loss)
    loss.backward()
    for t in (logits, th, ph, *tp):
        assert t.grad is not None and torch.isfinite(t.grad).all()
    with torch.no_grad():
        M = torch.softmax(logits, 1)
        psi = torch.zeros(B, N, dtype=torch.complex64); psi[:, 0] = 1
        tot = 0.0
        for t in range(T):
            p = probs(psi, M, th, ph)
            x = toks[:, t]
            tot += -(p.gather(1, x[:, None])[:, 0] / (1 - p[:, V])).log().sum().item()
            psi = luders_update(psi, x, M, th, ph)
        ref = tot / (B * T)
        assert abs(nll(toks, M, th, ph, tp, s_max=0).item() - ref) < 1e-4

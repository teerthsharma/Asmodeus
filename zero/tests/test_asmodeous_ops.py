import math

import pytest
import torch

from atlas.asmodeous import (cocycle_loss, cptp_fixed_point, fidelity, haar_cp_stat, link_cayley, povm_probs,
                             povm_probs_subset, stiefel)

torch.set_num_threads(4)
C128 = torch.complex128


def cgauss(*shape, dtype=C128, gen=None):
    re = torch.randn(*shape, dtype=torch.float64, generator=gen)
    im = torch.randn(*shape, dtype=torch.float64, generator=gen)
    return torch.complex(re, im).to(dtype)


def unit(z):
    return z / z.abs().pow(2).sum(-1, keepdim=True).sqrt()


# ---------------- stiefel ----------------
@pytest.mark.parametrize("K", [16, 65, 257])
def test_stiefel_isometry(K):
    torch.manual_seed(0)
    V = stiefel(cgauss(K, 16))
    err = (V.conj().T @ V - torch.eye(16, dtype=C128)).abs().max()
    assert err < 1e-10


def test_stiefel_real_input_is_real_isometry():
    torch.manual_seed(0)
    V = stiefel(torch.randn(65, 16, dtype=torch.float64))
    assert not V.is_complex()
    assert (V.T @ V - torch.eye(16, dtype=torch.float64)).abs().max() < 1e-10


def test_stiefel_gradient_flows():
    torch.manual_seed(0)
    W = cgauss(65, 16).requires_grad_(True)
    V = stiefel(W)
    (V.abs() ** 2 * torch.arange(65.0, dtype=torch.float64)[:, None]).sum().backward()
    assert W.grad is not None and torch.isfinite(W.grad).all() and W.grad.abs().max() > 0


def test_stiefel_phase_fixed():
    # sign/phase gauge: V^H W is upper triangular with a real positive diagonal
    torch.manual_seed(0)
    W = cgauss(40, 8)
    R = stiefel(W).conj().T @ W
    d = torch.diagonal(R)
    assert d.imag.abs().max() < 1e-10 and d.real.min() > 0


# ---------------- povm ----------------
def test_povm_probs_tight_frame():
    torch.manual_seed(0)
    V = stiefel(cgauss(65, 16))
    psi = unit(cgauss(7, 3, 16))
    p = povm_probs(V, psi)
    assert p.shape == (7, 3, 65)
    assert p.min() >= 0
    assert (p.sum(-1) - 1).abs().max() < 1e-10


def test_povm_probs_subset_equals_gather():
    torch.manual_seed(0)
    V = stiefel(cgauss(65, 16))
    psi = unit(cgauss(7, 3, 16))
    idx = torch.randint(0, 65, (7, 3, 5))
    ps = povm_probs_subset(V, psi, idx)
    assert ps.shape == (7, 3, 5)
    assert (ps - povm_probs(V, psi).gather(-1, idx)).abs().max() < 1e-12


# ---------------- cptp fixed point ----------------
def _stochastic(n, K=8, conc=0.05):
    # near one-hot columns: random functional graphs, usually reducible, so uniqueness needs the damping
    return torch.distributions.Dirichlet(torch.full((K,), conc, dtype=torch.float64)).sample((n, K)).transpose(1, 2)


def _F(P):
    return lambda w: torch.einsum("bkj,bj->bk", P, w)


def _rand_simplex(n, K=8):
    return torch.distributions.Dirichlet(torch.ones(K, dtype=torch.float64)).sample((n,))


def test_cptp_converges_geometrically():
    torch.manual_seed(0)
    P = _stochastic(1000)
    w, iters, res = cptp_fixed_point(_F(P), _rand_simplex(1000), eps=0.1, tol=1e-9, max_iter=500)
    assert res.max() < 1e-8
    assert iters <= math.ceil(math.log(1e-9) / math.log(0.9)) + 5
    assert (w.sum(-1) - 1).abs().max() < 1e-8


def test_cptp_unique_from_10_starts():
    torch.manual_seed(1)
    P = _stochastic(1000)
    sols = [cptp_fixed_point(_F(P), _rand_simplex(1000), eps=0.1, tol=1e-9, max_iter=500)[0] for _ in range(10)]
    for s in sols[1:]:
        assert (s - sols[0]).abs().max() < 1e-7


def test_cptp_undamped_cycle_does_not_converge():
    K = 8
    P = torch.roll(torch.eye(K, dtype=torch.float64), 1, 0)[None]
    w0 = torch.zeros(1, K, dtype=torch.float64)
    w0[0, 0] = 1.0
    _, iters, res = cptp_fixed_point(_F(P), w0, eps=0.0, tol=1e-9, max_iter=200)
    assert iters == 200 and res.max() > 0.5
    _, _, res_d = cptp_fixed_point(_F(P), w0, eps=0.1, tol=1e-9, max_iter=500)
    assert res_d.max() < 1e-8


# ---------------- links ----------------
def test_link_cayley_unitary_and_cocycle():
    torch.manual_seed(0)
    d = 3
    zj, zk = cgauss(6, d), cgauss(6, d)
    W = torch.randn(4 * d, 2 * d * d, dtype=torch.float64) * 0.3
    G = link_cayley(zj, zk, W)
    assert G.shape == (6, d, d) and G.dtype == C128
    eye = torch.eye(d, dtype=C128)
    assert (G @ G.conj().transpose(-1, -2) - eye).abs().max() < 1e-10
    zk_exact = torch.einsum("bij,bj->bi", G, zj)
    assert cocycle_loss(G, zj, zk_exact) < 1e-24
    assert cocycle_loss(G, zj, zk) > 1e-3


def test_cocycle_loss_gradient_to_W():
    torch.manual_seed(0)
    d = 2
    zj, zk = cgauss(5, d), cgauss(5, d)
    W = (torch.randn(4 * d, 2 * d * d, dtype=torch.float64) * 0.3).requires_grad_(True)
    cocycle_loss(link_cayley(zj, zk, W), zj, zk).backward()
    assert W.grad is not None and W.grad.abs().max() > 0


# ---------------- fidelity ----------------
def test_fidelity():
    torch.manual_seed(0)
    a, b = unit(cgauss(9, 16)), unit(cgauss(9, 16))
    assert fidelity(a, a).abs().max() < 1e-12
    ph = lambda t: torch.polar(torch.ones(9, 1, dtype=torch.float64), torch.randn(9, 1, dtype=torch.float64)) * t
    f = fidelity(a, b)
    assert (fidelity(ph(a), b) - f).abs().max() < 1e-12
    assert (fidelity(a, ph(b)) - f).abs().max() < 1e-12
    e0 = torch.tensor([1, 0, 0, 0], dtype=C128)
    e1 = torch.tensor([0, 1j, 0, 0], dtype=C128)
    assert abs(float(fidelity(e0, e1)) - 1) < 1e-12


# ---------------- haar ----------------
def test_haar_stat_haar_vs_collapsed():
    gen = torch.Generator().manual_seed(0)
    psi = unit(cgauss(4000, 4, gen=gen))
    haar = float(haar_cp_stat(psi, generator=torch.Generator().manual_seed(1)))
    assert 0 <= haar < 0.002
    collapsed = psi[:1].expand(4000, 4).contiguous()
    assert float(haar_cp_stat(collapsed, generator=torch.Generator().manual_seed(1))) > 0.02


def test_haar_stat_gradient_flows():
    gen = torch.Generator().manual_seed(0)
    psi = unit(cgauss(500, 4, gen=gen))
    raw = psi.clone().requires_grad_(True)
    haar_cp_stat(raw / raw.abs().pow(2).sum(-1, keepdim=True).sqrt(), generator=torch.Generator().manual_seed(1)).backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all() and raw.grad.abs().max() > 0

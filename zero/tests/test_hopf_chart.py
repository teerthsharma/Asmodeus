"""HopfChart: single global chart on CP^{N-1}. Chern/icosphere fixtures are evaluation-only, never given to the model."""
import inspect
import io
import re
import tokenize

import numpy as np
import pytest
import torch

import beds
import readouts
from atlas import hopf_chart
from atlas.hopf_chart import HopfChart, bloch, fidelity_loss
from ref.berry_check import chern_number, icosphere as ico_ref, spin_state

torch.set_num_threads(4)


def _psi(n, N, seed=0, dtype=torch.complex64):
    g = torch.Generator().manual_seed(seed)
    z = torch.complex(torch.randn(n, N, generator=g), torch.randn(n, N, generator=g)).to(dtype)
    return z / z.abs().pow(2).sum(-1, keepdim=True).sqrt()


@pytest.mark.parametrize("N,Dg", [(2, 3), (4, 15)])
def test_contract(N, Dg):
    torch.manual_seed(0)
    B, T, D, A, H = 3, 5, 16, 2, 4
    m = HopfChart(D, A, N=N, hidden=16)
    out = m(torch.randn(B, T, D), torch.randn(B, T, A))
    assert out["g"].shape == (B, T, Dg) and out["ghat"].shape == (B, T - 1, Dg)
    assert out["psi"].is_complex() and out["psi"].shape == (B, T, N)
    assert torch.allclose(out["psi"].abs().pow(2).sum(-1), torch.ones(B, T), atol=1e-6)
    assert out["w"] is None and out["loss"].dim() == 0 and set(out["terms"]) == {"fid", "sig", "task"}
    assert m.D_g == Dg
    assert m.rollout(torch.randn(B, D), torch.randn(B, H, A)).shape == (B, H, Dg)


def test_hopf_map_n2():
    psi = _psi(500, 2, dtype=torch.complex128)
    g = bloch(psi)
    assert torch.allclose(g.norm(dim=-1), torch.ones(500, dtype=torch.float64), atol=1e-6)
    z0, z1 = psi[:, 0], psi[:, 1]
    w = z0.conj() * z1  # standard Bloch vector (matches spin_state(n, 1/2) -> n)
    ref = torch.stack([2 * w.real, 2 * w.imag, z0.abs() ** 2 - z1.abs() ** 2], -1)
    assert torch.allclose(g, ref, atol=1e-6)


@pytest.mark.parametrize("N", [2, 3, 4, 5])
def test_pure_state_norm_hilbert_schmidt(N):
    # g_a = tr(rho L_a) with tr(L_a L_b) = 2 delta_ab  =>  ||g||^2 = 2 * ||rho - I/N||_HS^2 = 2 (1 - 1/N)
    g = bloch(_psi(300, N, dtype=torch.complex128))
    assert g.shape[-1] == N * N - 1
    assert torch.allclose(g.pow(2).sum(-1), torch.full((300,), 2 * (1 - 1 / N), dtype=torch.float64), atol=1e-9)


@pytest.mark.parametrize("N", [2, 4])
def test_gauge_invariance(N):
    psi = _psi(50, N, seed=1)
    ph = torch.exp(1j * torch.rand(50, 1) * 6.28).to(psi.dtype)
    assert torch.allclose(bloch(psi), bloch(psi * ph), atol=1e-6)
    tgt = _psi(50, N, seed=2)
    assert torch.allclose(fidelity_loss(psi, tgt), fidelity_loss(psi, tgt * ph), atol=1e-6)
    assert float(fidelity_loss(psi, psi * ph).abs().max()) < 1e-6


@pytest.mark.parametrize("N", [2, 4])
def test_lipschitz(N):
    p1, p2 = _psi(2000, N, seed=3, dtype=torch.complex128), _psi(2000, N, seed=4, dtype=torch.complex128)
    # include near pairs, where the ratio is largest
    p3 = p1 + 1e-3 * _psi(2000, N, seed=5, dtype=torch.complex128)
    p3 = p3 / p3.abs().pow(2).sum(-1, keepdim=True).sqrt()
    for a, b in ((p1, p2), (p1, p3)):
        lhs = (bloch(a) - bloch(b)).norm(dim=-1)
        rhs = (a - b).abs().pow(2).sum(-1).sqrt()
        assert bool((lhs <= 2 * rhs + 1e-9).all())


def test_unitary_step():
    torch.manual_seed(0)
    m = HopfChart(16, 2, N=4, hidden=16)
    psi, a = _psi(100, 4), torch.randn(100, 2) * 3
    nxt = m.step(a, psi)
    assert torch.allclose(nxt.abs().pow(2).sum(-1), torch.ones(100), atol=1e-6)


def test_surjective_s2():
    g = bloch(_psi(2000, 2, seed=7)).numpy()
    d = np.random.default_rng(0).normal(size=(20, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    assert (np.linalg.norm(g[None] - d[:, None], axis=-1).min(1) < 0.3).all()


def test_chern_plumbing_fixture():
    V, F = ico_ref(2)
    psi = torch.as_tensor(np.stack([spin_state(v, 0.5) for v in V]).astype(np.complex128))
    assert np.allclose(bloch(psi).numpy(), V, atol=1e-6)
    assert abs(round(chern_number(V, F, lambda v: spin_state(v, 0.5)))) == 1


def test_chern_of_arm_runs():
    act_dim = beds.make_bed("berry0", 2, 4)["act"].shape[-1]
    torch.manual_seed(0)
    arm = HopfChart(D=256, A=act_dim, N=2)
    r = readouts.chern_of_arm(arm, "berry0", level=1)
    assert r is not None and np.isfinite(r["chern"])


def test_training_smoke():
    torch.manual_seed(0)
    bed = beds.make_bed("s2", 16, 16, seed=3)
    obs, act = torch.as_tensor(bed["obs"]), torch.as_tensor(bed["act"])
    m = HopfChart(obs.shape[-1], act.shape[-1], N=2, hidden=32).train()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    losses = []
    for _ in range(30):
        loss = m(obs, act)["loss"]
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] and not np.isnan(losses).any()


def test_action_changes_rollout():
    torch.manual_seed(0)
    m = HopfChart(16, 2, N=2, hidden=16)
    o = torch.randn(4, 16)
    r0 = m.rollout(o, torch.zeros(4, 3, 2))
    r1 = m.rollout(o, torch.ones(4, 3, 2))
    assert not torch.allclose(r0, r1, atol=1e-5)


def test_l_given_no_state_token():
    src = inspect.getsource(hopf_chart)
    names = [t.string for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type == tokenize.NAME]
    assert not [n for n in names if re.search(r"\bstate\b", n)]
    assert list(inspect.signature(HopfChart.forward).parameters) == ["self", "obs", "act"]


def test_flops_and_params():
    a, b = HopfChart(16, 2, N=2, hidden=16), HopfChart(16, 2, N=2, hidden=64)
    assert 0 < a.active_flops() < b.active_flops()
    assert any(n.startswith("tgt_") for n, _ in a.named_parameters())
    assert a.n_params() == sum(p.numel() for n, p in a.named_parameters() if not n.startswith("tgt_"))
    assert a.n_params() < sum(p.numel() for p in a.parameters())


def test_single_frame_loss_finite():
    torch.manual_seed(0)
    m = HopfChart(16, 2, N=2, hidden=16).train()
    out = m(torch.randn(3, 1, 16), torch.randn(3, 1, 2))
    assert out["ghat"].shape == (3, 0, 3)
    assert torch.isfinite(out["loss"]) and float(out["terms"]["fid"]) == 0 and float(out["terms"]["task"]) == 0
    out["loss"].backward()

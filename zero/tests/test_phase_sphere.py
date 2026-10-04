"""PHASE-SPHERE (ORIGIN.md §3, TO1-TO7): RED-first contracts for the primitives and the layer.
Ground truth here is analytic (monopole Chern numbers, Klein orientation holonomy); the hand-built frame
fields are test fixtures only and are never given to a model (L-GIVEN)."""
import math

import numpy as np
import pytest
import torch

from atlas.phase_sphere import (born, fixed_point, cayley, skew_hermitian, holonomy, sign_holonomy,
                                chern_fhs, PhaseSphere)
from beds import klein_embed, make_bed

C = torch.complex128


# ---------------- decider = Born rule ----------------
def test_born_is_softmax_of_2log_abs():
    g = torch.Generator().manual_seed(0)
    a = torch.complex(torch.randn(100, 9, generator=g, dtype=torch.float64),
                      torch.randn(100, 9, generator=g, dtype=torch.float64))
    p = born(a)
    assert torch.allclose(p, torch.softmax(2 * torch.log(a.abs()), -1), atol=1e-12)
    assert torch.allclose(born(a * torch.exp(1j * torch.rand(100, 1, dtype=torch.float64))), p, atol=1e-12)
    null = torch.complex(torch.randn(100, generator=g, dtype=torch.float64), torch.zeros(100, dtype=torch.float64))
    q = born(a, null=null)                                  # "observe nothing" is the last outcome
    assert q.shape == (100, 10) and torch.allclose(q.sum(-1), torch.ones(100, dtype=torch.float64))
    assert torch.allclose(q[:, :9] / q[:, :9].sum(-1, keepdim=True), p, atol=1e-12)


# ---------------- CTC observer = self-consistent fixed point ----------------
def test_fixed_point_on_1000_column_stochastic_matrices():
    rng = np.random.default_rng(0)
    P = torch.as_tensor(rng.random((1000, 8, 8)))
    P = P / P.sum(1, keepdim=True)                          # columns sum to 1
    f = lambda x: (P @ x[..., None])[..., 0]
    x0 = torch.full((1000, 8), 1 / 8, dtype=torch.float64)
    x, iters, res = fixed_point(f, x0, tol=1e-10, max_iter=500)
    assert res.shape == (1000,) and float(res.max()) < 1e-8
    assert float((f(x) - x).abs().max()) < 1e-8
    assert (x >= 0).all() and torch.allclose(x.sum(-1), torch.ones(1000, dtype=torch.float64))
    assert int(iters) <= 500


# ---------------- unitary local steps ----------------
def test_cayley_unitary_and_matches_coherent_observer():
    g = torch.Generator().manual_seed(1)
    X = torch.complex(torch.randn(50, 4, 4, generator=g, dtype=torch.float64),
                      torch.randn(50, 4, 4, generator=g, dtype=torch.float64))
    U = cayley(skew_hermitian(X))
    eye = torch.eye(4, dtype=C).expand(50, 4, 4)
    assert torch.allclose(U.conj().transpose(-1, -2) @ U, eye, atol=1e-10)
    # real skew A = P Q^T - Q P^T: same operator as Kutner's atlas/coherent_observer.cayley_apply
    from atlas.coherent_observer import cayley_apply
    Pm, Qm = torch.randn(6, 2, dtype=torch.float64), torch.randn(6, 2, dtype=torch.float64)
    A = Pm @ Qm.T - Qm @ Pm.T
    x = torch.randn(5, 6, dtype=torch.float64)
    ours = (cayley(A.to(C)) @ x.to(C).T).T.real
    assert torch.allclose(ours, cayley_apply(x, Pm, Qm), atol=1e-10)


# ---------------- Chern number (Fukui-Hatsugai-Suzuki) ----------------
def icosphere(level):
    t = (1 + 5 ** .5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
         (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = [np.array(v) / np.linalg.norm(v) for v in V]
    for _ in range(level):
        mid, F2 = {}, []
        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                v = V[a] + V[b]; V.append(v / np.linalg.norm(v)); mid[k] = len(V) - 1
            return mid[k]
        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    V, F = np.array(V), np.array(F)
    cen = V[F].mean(1)                                       # orient every triangle outward (CCW from outside)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = (nrm * cen).sum(1) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return V, torch.as_tensor(F)


def coherent(V, spin2):
    th, ph = np.arccos(np.clip(V[:, 2], -1, 1)), np.arctan2(V[:, 1], V[:, 0])
    c, s = np.cos(th / 2), np.sin(th / 2)
    if spin2 == 1:
        psi = np.stack([c, s * np.exp(1j * ph)], 1)
    else:                                                   # spin 1
        psi = np.stack([c * c, 2 ** .5 * c * s * np.exp(1j * ph), s * s * np.exp(2j * ph)], 1)
    return torch.as_tensor(psi, dtype=C)


def test_chern_of_monopoles():
    V, F = icosphere(2)
    half = coherent(V, 1)
    assert abs(chern_fhs(half, F) - 1) < 1e-9                       # Hopf / Dirac monopole, charge 1
    assert abs(chern_fhs(half, F[:, [0, 2, 1]]) + 1) < 1e-9         # orientation flip
    assert abs(chern_fhs(coherent(V, 2), F) - 2) < 1e-9             # spin 1: charge 2
    gauge = torch.exp(1j * torch.rand(len(V), 1, dtype=torch.float64) * 2 * math.pi)
    assert abs(chern_fhs(half * gauge, F) - 1) < 1e-9               # gauge invariant
    const = torch.zeros(len(V), 2, dtype=C); const[:, 0] = 1
    assert abs(chern_fhs(const, F)) < 1e-12                          # trivial bundle


# ---------------- holonomy: Klein generator reverses orientation ----------------
def frame_loop(kind, L=400, u0=0.3, v0=0.0):
    """Hand-built tangent frame along a closed loop (fixture only). klein-a: u -> u + 2pi at v = v0
    (identified with (u, -v)), frame d/dv; klein-b: v -> v + 2pi, frame d/du; torus: theta1 loop, d/dtheta2."""
    t = np.linspace(0, 2 * np.pi, L, endpoint=False)
    h = 1e-5
    if kind == "klein-a":
        f = (klein_embed(u0 + t, v0 + h) - klein_embed(u0 + t, v0 - h)) / (2 * h)
    elif kind == "klein-b":
        f = (klein_embed(u0 + h + 0 * t, v0 + t) - klein_embed(u0 - h + 0 * t, v0 + t)) / (2 * h)
    else:
        th1, th2 = u0 + t, v0 + 0 * t
        f = np.stack([0 * t, 0 * t, -np.sin(th2), np.cos(th2)], 1)
    return torch.as_tensor(f / np.linalg.norm(f, axis=1, keepdims=True), dtype=C)


def test_holonomy_klein_generator_is_minus_one():
    assert sign_holonomy(frame_loop("klein-a")) == -1
    assert abs(abs(holonomy(frame_loop("klein-a"))) - math.pi) < 1e-6
    assert sign_holonomy(frame_loop("klein-b")) == 1
    assert sign_holonomy(frame_loop("torus")) == 1
    loops = torch.stack([frame_loop("klein-a"), frame_loop("torus")])   # batched over leading dims
    assert sign_holonomy(loops).tolist() == [-1, 1]


def test_holonomy_of_berry_loop_is_half_solid_angle():
    # spin-1/2 around a latitude circle at polar angle a: Berry phase = half the enclosed solid angle
    a = 1.0
    ph = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    V = np.stack([np.sin(a) * np.cos(ph), np.sin(a) * np.sin(ph), np.cos(a) + 0 * ph], 1)
    omega = 2 * np.pi * (1 - np.cos(a))
    assert abs(holonomy(coherent(V, 1)) - omega / 2) < 1e-4


# ---------------- the layer: common.py contract ----------------
@pytest.fixture(scope="module")
def small():
    bed = make_bed("s2", n_traj=8, T=12, obs="lift", seed=0)
    torch.manual_seed(0)
    arm = PhaseSphere(D=256, A=3, K=6, m=2, N=8, d_c=2, D_g=6)
    return arm, torch.as_tensor(bed["obs"]), torch.as_tensor(bed["act"])


def test_contract_shapes_and_invariants(small):
    arm, o, a = small
    out = arm(o, a)
    B, T = o.shape[:2]
    assert out["g"].shape == (B, T, 6) and out["ghat"].shape == (B, T - 1, 6)
    assert out["w"].shape == (B, T, 6) and out["w_soft"].shape == (B, T, 6)
    assert ((out["w"] > 0).sum(-1) <= 2).all()
    assert torch.allclose(out["w"].sum(-1), torch.ones(B, T), atol=1e-5)
    assert torch.allclose(out["w_soft"].sum(-1), torch.ones(B, T), atol=1e-5)
    assert ((out["gate"] >= 0) & (out["gate"] <= 1)).all()                   # 1 - p(nothing)
    assert out["psi"].dtype == torch.complex64 and torch.allclose(out["psi"].abs().pow(2).sum(-1), torch.ones(B, T), atol=1e-5)
    assert out["z"].shape == (B, T, 2, 4) and out["zidx"].shape == (B, T, 2)  # real view of d_c complex
    assert out["chi"].shape == (B, T, 2, 2)                                     # transition phases, top-m pairs
    assert torch.allclose(out["chi"], -out["chi"].transpose(-1, -2), atol=1e-5)
    assert int(out["ctc_iters"]) <= arm.loop_iters and torch.isfinite(out["ctc_residual"]).all()
    out["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in arm.parameters() if p.grad is not None)
    assert arm.rollout(o[:, 0], a[:, :5]).shape == (B, 5, 6)
    assert isinstance(arm.active_flops(), int) and isinstance(arm.n_params(), int)


def test_global_phase_of_psi_does_not_change_g(small):
    arm, o, a = small
    arm.eval()
    with torch.no_grad():
        r1 = arm.observe(o, a)
        r2 = arm.observe(o, a, phase=torch.exp(1j * torch.rand(o.shape[0], o.shape[1])))
    assert torch.allclose(r1["g"], r2["g"], atol=1e-5)                          # charts gauge-fix by arg alpha_k


def test_amortised_head_and_lazy_flops(small):
    arm, o, a = small
    with torch.no_grad():
        oa = arm(o, a, mode="amortised")
    assert oa["g"].shape == (o.shape[0], o.shape[1], 6) and int(oa["ctc_iters"]) == 0
    f = lambda **kw: PhaseSphere(D=256, A=3, N=8, d_c=2, D_g=6, **kw).active_flops()
    assert f(K=8, m=1) < f(K=8, m=2)
    assert f(K=64, m=2) - f(K=8, m=2) <= 56 * 8 * 8          # K only enters through the K x N Born scores

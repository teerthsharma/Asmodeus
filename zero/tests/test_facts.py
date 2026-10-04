"""Binds the computable facts in PREREG_DRAFT_facts.md / PREREG.md §0 (Wilson).

Independent Betti implementation: ref/betti_check.py (exact Q via Fractions, Z2 via GF(2)).
Cross-checked against metrics.betti_z2. Facts bound here:
  F§1.4  Z2 and Q Betti of every ref/triangulations.json entry (T^2 vs Klein separated over Q only)
  F§3a   Betti-only minimal vertex counts: S^2 4, T^2/Klein 5, RP^2 5, RP^3 6 (explicit complexes;
         smaller counts excluded by exhaustive enumeration of all complexes on <= 5 vertices)
  F§5    top-m co-selection nerve has dim <= m-1; PoU tau-nerve has dim <= floor(1/tau)-1
  F§1.3  T^2 and Klein bottle are each covered by two open annuli (planar charts), checked numerically
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from metrics import betti_z2, closure
from ref.betti_check import betti, faces_of

TRI = json.loads((Path(__file__).resolve().parents[1] / "ref" / "triangulations.json").read_text())["triangulations"]

EXPECTED = {  # name: (Z2, Q)
    "S2_tetrahedron_boundary": ((1, 0, 1), (1, 0, 1)),
    "S2_octahedron": ((1, 0, 1), (1, 0, 1)),
    "T2_Moebius_Csaszar_7": ((1, 2, 1), (1, 2, 1)),
    "RP2_6": ((1, 1, 1), (1, 0, 0)),
    "Klein_8": ((1, 2, 1), (1, 1, 0)),
    "RP3_Walkup_11": ((1, 1, 1, 1), (1, 0, 0, 1)),
}


# ---------------- F§1.4: triangulations ----------------
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_triangulation_betti_z2_and_q(name):
    t = TRI[name]
    z2, q = EXPECTED[name]
    assert betti(t["facets"], "Z2") == z2
    assert betti(t["facets"], "Q") == q
    assert list(z2) == t["betti_Z2"] and list(q) == t["betti_Q"]
    f = [sum(len(s) == k + 1 for s in faces_of(t["facets"])) for k in range(t["dim"] + 1)]
    assert f == t["f_vector"]
    assert sum((-1) ** k * x for k, x in enumerate(f)) == t["euler_characteristic"] == sum((-1) ** k * b for k, b in enumerate(q))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_independent_impl_matches_metrics(name):
    F = TRI[name]["facets"]
    assert betti(F, "Z2") == tuple(betti_z2(closure([tuple(sorted(f)) for f in F])))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_closed_pseudomanifold_with_sphere_links(name):
    F = TRI[name]["facets"]
    d = len(F[0]) - 1
    ridges = [r for r in faces_of(F) if len(r) == d]
    assert all(sum(set(r) <= set(f) for f in F) == 2 for r in ridges)
    sphere = (1,) + (0,) * (d - 2) + (1,)
    for v in {x for f in F for x in f}:
        assert betti([[x for x in f if x != v] for f in F if v in f], "Q") == sphere


def test_z2_does_not_separate_torus_from_klein_but_q_does():
    a, b = TRI["T2_Moebius_Csaszar_7"]["facets"], TRI["Klein_8"]["facets"]
    assert betti(a, "Z2") == betti(b, "Z2")
    assert betti(a, "Q") != betti(b, "Q")


def test_betti_impl_known_small_cases():
    assert betti([[1, 2, 3]], "Q") == (1, 0, 0)                      # filled triangle
    assert betti([[1, 2], [2, 3], [1, 3]], "Z2") == (1, 1)           # circle
    assert betti([[1], [2]], "Q") == (2,)                            # two points
    # Mutation guard: a facet dropped from RP^3 must change the answer
    F = TRI["RP3_Walkup_11"]["facets"]
    assert betti(F[1:], "Z2") != (1, 1, 1, 1)


# ---------------- F§3a: Betti-only minimal vertex counts ----------------
SMALL = {
    "S2": ((1, 0, 1), 4, [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]),
    "T2/Klein": ((1, 2, 1), 5, [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4], [5, 1], [5, 2], [5, 3]]),
    "RP2": ((1, 1, 1), 5, [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4], [5, 1], [5, 2]]),
    "RP3": ((1, 1, 1, 1), 6, [list(c) for c in itertools.combinations([1, 2, 3, 4, 5], 4)]
            + [[6, 1, 2], [6, 1, 3], [6, 2, 3], [6, 4]]),
}


@pytest.mark.parametrize("name", sorted(SMALL))
def test_explicit_min_vertex_complex(name):
    target, n, F = SMALL[name]
    assert len({v for f in F for v in f}) == n
    assert betti(F, "Z2") == target


def _all_complexes(n):
    """Every simplicial complex on exactly the vertex set range(n) (as lists of maximal faces + vertices)."""
    faces = sorted([s for k in range(2, n + 1) for s in itertools.combinations(range(n), k)], key=len, reverse=True)

    def rec(i, chosen):
        if i == len(faces):
            yield [list(c) for c in chosen] + [[v] for v in range(n) if not any(v in c for c in chosen)]
            return
        yield from rec(i + 1, chosen)
        if not any(set(faces[i]) <= set(c) for c in chosen):
            yield from rec(i + 1, chosen + [faces[i]])

    yield from rec(0, [])


def _strip(b):
    b = list(b)
    while b and b[-1] == 0:
        b.pop()
    return tuple(b)


def test_min_vertex_counts_are_minimal_exhaustive():
    """No complex on fewer vertices than SMALL[...] attains the target Z2 Betti vector.
    Counts 9/114/6894 = OEIS A307249 (simplicial complexes on n labelled nodes, all used)."""
    seen = {}
    for n, count in ((3, 9), (4, 114), (5, 6894)):
        cx = list(_all_complexes(n))
        assert len(cx) == count
        for F in cx:
            seen.setdefault(_strip(betti(F, "Z2")), set()).add(n)
    for target, n, _ in SMALL.values():
        pad = _strip(target)
        assert all(k >= n for k in seen.get(pad, set())), (target, seen.get(pad))
        if n <= 5:
            assert n in seen[pad]


# ---------------- F§5: nerve dimension bounds ----------------
def _nerve(W, tau):
    out = set()
    for row in W:
        S = tuple(np.flatnonzero(row >= tau))
        for r in range(1, len(S) + 1):
            out.update(itertools.combinations(S, r))
    return out


def _random_pou(rng, N, K, temp):
    x = rng.normal(size=(N, K)) * temp
    w = np.exp(x - x.max(1, keepdims=True))
    return w / w.sum(1, keepdims=True)


@pytest.mark.parametrize("seed", range(20))
def test_top_m_nerve_dim_le_m_minus_1(seed):
    rng = np.random.default_rng(seed)
    K, m = int(rng.integers(3, 12)), int(rng.integers(1, 5))
    w = _random_pou(rng, 400, K, rng.uniform(0.1, 3))
    top = np.argsort(-w, 1)[:, :m]
    wm = np.zeros_like(w)
    np.put_along_axis(wm, top, np.take_along_axis(w, top, 1), 1)
    wm /= wm.sum(1, keepdims=True)
    for tau in (1e-9, 0.05, 0.2):
        cx = _nerve(wm, tau)
        assert max(len(s) for s in cx) - 1 <= m - 1
        b = betti([list(s) for s in cx], "Z2")
        assert all(x == 0 for x in b[m:])                     # H_k = 0 for k >= m
    if m <= 2:
        b = betti([list(s) for s in _nerve(wm, 1e-9)], "Z2")
        assert b[2:] == () or all(x == 0 for x in b[2:])      # beta2 = beta3 = 0


@pytest.mark.parametrize("seed", range(20))
def test_pou_tau_nerve_dim_le_floor_inv_tau_minus_1(seed):
    rng = np.random.default_rng(100 + seed)
    K = int(rng.integers(3, 14))
    w = _random_pou(rng, 400, K, rng.uniform(0.01, 3))
    assert np.allclose(w.sum(1), 1)
    for tau in (1 / 2, 1 / 3, 0.3, 1 / 4, 0.2, 1 / 6, 0.12):
        cx = _nerve(w, tau)
        if cx:
            assert max(len(s) for s in cx) - 1 <= math.floor(1 / tau) - 1


def test_softmax_small_tau_nerve_is_full_simplex():
    rng = np.random.default_rng(7)
    w = _random_pou(rng, 50, 6, 1.0)
    assert (w > 0).all()
    tau = w.min() / 2
    cx = _nerve(w, tau)
    assert tuple(range(6)) in cx
    assert betti([list(s) for s in cx], "Q") == (1, 0, 0, 0, 0, 0)


# ---------------- F§1.3: two-annulus covers ----------------
TWO_PI = 2 * np.pi


def _wrap(a):
    return np.mod(a, TWO_PI)


def _torus_charts(p=0.0, q=np.pi):
    """T^2 = S^1 x S^1, angles (th, ph). U = {ph != p}, V = {ph != q}.
    Chart: (th, ph) -> exp(t)(cos th, sin th), t = signed distance of ph from the removed point in (0, 2pi)."""
    def chart(c):
        def f(th, ph):
            t = _wrap(ph - c) - np.pi          # in (-pi, pi) on the domain
            r = np.exp(t)
            return np.stack([r * np.cos(th), r * np.sin(th)], -1)
        return f
    dom_U = lambda th, ph: np.abs(np.angle(np.exp(1j * (ph - p)))) > 0   # noqa: E731
    dom_V = lambda th, ph: np.abs(np.angle(np.exp(1j * (ph - q)))) > 0   # noqa: E731
    return (dom_U, chart(p)), (dom_V, chart(q))


def _injective(points_xy, ids, tol=1e-9):
    """Distinct manifold points must map to distinct chart points (checked on a grid with known ids)."""
    _, inv, cnt = np.unique(np.round(points_xy / tol).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    for k in np.flatnonzero(cnt > 1):
        assert len(set(ids[inv.ravel() == k])) == 1


def test_torus_two_annulus_cover():
    n = 120
    th, ph = np.meshgrid(np.arange(n) * TWO_PI / n + 0.013, np.arange(n) * TWO_PI / n + 0.007)
    th, ph = th.ravel(), ph.ravel()
    ids = np.arange(th.size)
    (dU, fU), (dV, fV) = _torus_charts(0.0, np.pi)
    assert (dU(th, ph) | dV(th, ph)).all()                       # cover
    for d, f in ((dU, fU), (dV, fV)):
        m = d(th, ph)
        xy = f(th[m], ph[m])
        assert np.isfinite(xy).all()
        _injective(xy, ids[m], tol=1e-6)
        r = np.linalg.norm(xy, axis=1)
        assert (r > 0).all()                                    # image in R^2 \ {0}: open annulus


def _klein_chart_U(t, y):
    """K = [0,1] x S^1 / (0,y) ~ (1,-y), y in [0,1) mod 1.  U = (0,1) x S^1 -> annulus."""
    r = 1.0 + t
    return np.stack([r * np.cos(TWO_PI * y), r * np.sin(TWO_PI * y)], -1)


def _klein_chart_V(t, y, eps):
    """V = {t in [0,eps) or (1-eps,1]}; phi^{-1}: s = t (t<eps), s = t-1 with y -> -y (t>1-eps); annulus (s, y)."""
    s = np.where(t < eps, t, t - 1.0)
    yy = np.where(t < eps, y, -y)
    r = 2.0 + s
    return np.stack([r * np.cos(TWO_PI * yy), r * np.sin(TWO_PI * yy)], -1)


def test_klein_two_annulus_cover_and_gluing_consistency():
    eps = 0.2
    n = 150
    t, y = np.meshgrid(np.arange(n + 1) / n, (np.arange(n) + 0.37) / n)
    t, y = t.ravel(), y.ravel()
    # canonical representative: identify (0, y) with (1, -y); keep t=1 points as their t=0 twins
    key = np.where(t == 1.0, np.stack([np.zeros_like(t), np.mod(-y, 1)], 1).T, np.stack([t, y], 1).T).T
    _, ids = np.unique(np.round(key * 1e6).astype(np.int64), axis=0, return_inverse=True)
    ids = ids.ravel()
    inU = (t > 0) & (t < 1)
    inV = (t < eps) | (t > 1 - eps)
    assert (inU | inV).all()                                     # cover
    xyU = _klein_chart_U(t[inU], y[inU])
    _injective(xyU, ids[inU], tol=1e-6)
    xyV = _klein_chart_V(t[inV], np.mod(y[inV], 1), eps)
    _injective(xyV, ids[inV], tol=1e-6)
    # the two representatives of a glued point land on the same chart point (V well defined on the quotient)
    a = _klein_chart_V(np.zeros(n), (np.arange(n) + 0.37) / n, eps)
    b = _klein_chart_V(np.ones(n), np.mod(-(np.arange(n) + 0.37) / n, 1), eps)
    assert np.allclose(a, b)
    # images are open annuli: radii in (1, 2) for U and (2 - eps, 2 + eps) for V
    rU, rV = np.linalg.norm(xyU, axis=1), np.linalg.norm(xyV, axis=1)
    assert ((rU > 1) & (rU < 2)).all() and ((rV > 2 - eps) & (rV < 2 + eps)).all()


# ---------------- Origin TO1 facts: Berry phase and Hopf-bundle Chern number (Wilson) ----------------
# Convention (stated once): A = i<psi|d psi>, Berry phase gamma = i * loop-integral <psi|d psi>,
# discretised gauge-invariantly as gamma = -arg prod_j <psi_j|psi_{j+1}> (Pancharatnam / Bargmann).
# Surfaces are oriented by the OUTWARD normal; loops run counter-clockwise seen from outside.
# psi(n) = spin-up along n: (cos(t/2), e^{i p} sin(t/2)); spin-1: its symmetric square.
from ref.berry_check import berry_phase, chern_number, icosphere, spin_state, solid_angle


@pytest.mark.parametrize("theta0", [0.3, 0.8, 1.2, 1.5, 2.0])
def test_spin_half_berry_phase_is_minus_half_solid_angle(theta0):
    phi = np.linspace(0, 2 * np.pi, 4001)[:-1]
    loop = [spin_state(np.array([np.sin(theta0) * np.cos(p), np.sin(theta0) * np.sin(p), np.cos(theta0)]), 0.5)
            for p in phi]
    omega = 2 * np.pi * (1 - np.cos(theta0))          # cap solid angle, loop counter-clockwise about +z
    g = berry_phase(loop)
    target = np.angle(np.exp(-1j * omega / 2))          # -Omega/2, wrapped to (-pi, pi]
    assert abs(np.angle(np.exp(1j * (g - target)))) < 1e-3


def test_spin_half_triangle_bargmann_phase_is_exactly_minus_half_solid_angle():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b, c = (v / np.linalg.norm(v) for v in rng.normal(size=(3, 3)))
        if np.dot(a, np.cross(b, c)) < 0:
            b, c = c, b                                   # counter-clockwise seen from outside
        g = berry_phase([spin_state(v, 0.5) for v in (a, b, c)])
        assert abs(np.angle(np.exp(1j * (g + solid_angle(a, b, c) / 2)))) < 1e-9


@pytest.mark.parametrize("level", [2, 3, 4])
def test_hopf_bundle_chern_number_is_one(level):
    V, F = icosphere(level)
    c_half = chern_number(V, F, lambda n: spin_state(n, 0.5))
    assert abs(c_half - round(c_half)) < 1e-9 and round(c_half) == -1   # |c1| = 1; sign -1 in this convention
    c_one = chern_number(V, F, lambda n: spin_state(n, 1.0))
    assert abs(c_one - round(c_one)) < 1e-9 and round(c_one) == -2      # spin-1 control: |c1| = 2
    c_triv = chern_number(V, F, lambda n: np.array([1.0 + 0j, 0.0]))
    assert abs(c_triv) < 1e-12                                           # trivial bundle control: 0


def test_icosphere_is_outward_oriented_closed_surface():
    V, F = icosphere(3)
    assert all(np.dot(V[i], np.cross(V[j], V[k])) > 0 for i, j, k in F)
    assert betti([list(f) for f in F], "Q") == (1, 0, 1)


# ---------------- PLAN.md bound: complete instruments cannot make many tokens likely (Wilson) ----------------
# Fact: for any instrument {K_v} on C^N with sum_v K_v^H K_v = I,
#   sum_v max_{|psi|=1} |K_v psi|^2 = sum_v sigma_max(K_v)^2 <= sum_v ||K_v||_F^2 = tr(I) = N,
# so at most 2N outcomes can ever reach p >= 1/2. Equality iff every K_v has rank 1.
from ref.instrument_check import (diag_q_instrument, instrument_max_mass, is_complete, random_instrument,
                                  tokens_reachable)


@pytest.mark.parametrize("N,V,rank", [(4, 9, 1), (4, 9, 3), (8, 40, 2), (16, 200, 1), (16, 200, 4)])
def test_instrument_max_mass_bounded_by_dimension(N, V, rank):
    rng = np.random.default_rng(N * 1000 + V + rank)
    K = random_instrument(N, V, rank, rng)
    assert is_complete(K)
    total = instrument_max_mass(K)
    assert total <= N + 1e-9
    if rank == 1:
        assert abs(total - N) < 1e-9                     # rank-1 (Parseval) heads saturate the bound
    assert tokens_reachable(K, 0.5) <= 2 * N


def test_plan_diag_q_head_bound_and_talk_threshold():
    """K_v = diag(sqrt(M[:, v])) Q with M row-stochastic N x V and Q unitary (PLAN.md item 3)."""
    rng = np.random.default_rng(3)
    for N, V in [(16, 4096), (64, 4096)]:
        K = diag_q_instrument(N, V, rng)
        assert is_complete(K)
        assert instrument_max_mass(K) <= N + 1e-9
        assert tokens_reachable(K, 0.5) <= 2 * N < V     # N = 16 or 64 cannot make most of |V| = 4096 likely


def test_diag_q_head_effects_commute():
    """Foreman's finding (Amendment 2): effects E_v = K_v^H K_v = Q^H diag(M_v) Q all commute, so a single-step
    readout of this head is classical; any quantum advantage must come from coherence carried across steps."""
    rng = np.random.default_rng(5)
    K = diag_q_instrument(8, 50, rng)
    E = [K[v].conj().T @ K[v] for v in range(len(K))]
    assert max(np.abs(a @ b - b @ a).max() for a in E[:10] for b in E[:10]) < 1e-12
    d = K.Q @ E[3] @ K.Q.conj().T                       # diagonal in the Q basis
    assert np.abs(d - np.diag(np.diag(d))).max() < 1e-12


@pytest.mark.parametrize("N,V,X", [(2, 6, 40), (4, 30, 200), (8, 100, 300)])
def test_holevo_one_instrument_carries_at_most_ln_N_nats(N, V, X):
    """Amendment 3 / Park: with past x -> pure psi_x in C^N and the future read by one complete instrument
    (any composition of instruments is again one), I(x; outcome) <= ln N nats (Holevo; tight only for
    N orthogonal states read in their basis)."""
    from ref.instrument_check import mutual_information_nats
    rng = np.random.default_rng(N + V)
    K = random_instrument(N, V, 1, rng)
    psi = rng.normal(size=(X, N)) + 1j * rng.normal(size=(X, N))
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    assert mutual_information_nats(K, psi) <= np.log(N) + 1e-9
    basis = [np.diag(np.eye(N)[i]).astype(complex) for i in range(N)]       # projective, saturating case
    assert abs(mutual_information_nats(basis, np.eye(N, dtype=complex)) - np.log(N)) < 1e-9

"""L1-L4, L23: observer (Parseval frame, Born, residual, tail certificate, post-selection, null split)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa.observer import (  # noqa: E402
    born,
    birth_direction,
    captured_mass,
    effects_from_frame,
    parseval_frame,
    post_select,
    residual,
    split_null,
    tail_certificate,
)


def _psi(rng, n, b=None):
    shape = (n,) if b is None else (b, n)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _dirichlet_p(rng, k):
    p = rng.dirichlet(np.ones(k) * 0.5)
    return p


# ---- L1 -------------------------------------------------------------------
@pytest.mark.parametrize("N,K", [(1, 1), (2, 3), (4, 4), (4, 15), (8, 40)])
def test_L1_born_exactness(N, K):
    rng = np.random.default_rng(0)
    V = parseval_frame(N, K, rng)
    assert V.shape == (K + 1, N)
    assert np.allclose(V.conj().T @ V, np.eye(N), atol=1e-12)
    psi = _psi(rng, N, 50)
    p = born(V, psi)
    assert p.shape == (50, K + 1)
    assert np.allclose(p.sum(-1), 1.0, atol=1e-12)
    assert (p >= 0).all()
    # single state, and effects agree with the rows: sum E_k = I, <psi|E_k|psi> = p_k
    E = effects_from_frame(V)
    assert np.allclose(sum(E), np.eye(N), atol=1e-12)
    p1 = born(V, psi[0])
    assert p1.shape == (K + 1,)
    assert np.allclose(p1, [np.vdot(psi[0], e @ psi[0]).real for e in E], atol=1e-12)


def test_L1_frame_rejects_underdetermined_and_deterministic():
    with pytest.raises(ValueError):
        parseval_frame(5, 3, np.random.default_rng(0))  # K+1 < N: V^H V = I impossible
    a = parseval_frame(3, 5, 7)
    b = parseval_frame(3, 5, 7)
    assert np.array_equal(a, b)


def test_L1_born_rejects_non_unit_state():
    V = parseval_frame(3, 4, 0)
    with pytest.raises(ValueError):
        born(V, 2.0 * _psi(np.random.default_rng(1), 3))


# ---- L2 -------------------------------------------------------------------
def test_L2_exact_residual():
    rng = np.random.default_rng(1)
    for _ in range(200):
        k = int(rng.integers(2, 30))
        p = _dirichlet_p(rng, k)
        T = rng.choice(k, size=int(rng.integers(0, k + 1)), replace=False)
        r = residual(p, T)
        out = np.setdiff1d(np.arange(k), T)
        assert r >= 0
        assert r == pytest.approx(p[out].sum(), abs=1e-15)  # exact mass outside T
        assert r == pytest.approx(1 - p[T].sum(), abs=1e-12)
    assert residual(np.array([0.5, 0.5]), []) == pytest.approx(1.0)
    assert residual(np.array([0.5, 0.5]), [0, 1]) == 0.0


# ---- L3 -------------------------------------------------------------------
def test_L3_tail_certificate_holds_for_temp_le_1():
    rng = np.random.default_rng(2)
    for _ in range(500):
        k = int(rng.integers(2, 40))
        p = _dirichlet_p(rng, k)
        T = rng.choice(k, size=int(rng.integers(1, k)), replace=False)
        out = np.setdiff1d(np.arange(k), T)
        for temp in (1.0, 0.7, 0.3, 0.05):
            lhs = (p[out] ** (1 / temp)).sum()
            bound = tail_certificate(p, T, temp)
            assert lhs <= bound * (1 + 1e-12) + 1e-300
    # edges: empty tail, zero tail
    assert tail_certificate(np.array([0.6, 0.4]), [0, 1], 0.5) == 0.0
    assert tail_certificate(np.array([1.0, 0.0]), [0], 0.5) == 0.0
    # temp = 1 is the identity: bound = r_T
    p = np.array([0.5, 0.3, 0.2])
    assert tail_certificate(p, [0], 1.0) == pytest.approx(0.5)


def test_L3_prime_tail_certificate_for_temp_gt_1():
    """Amendment B, L3': for q = 1/T < 1, sum_tail p^q <= m_out^{1-q} r^q (Holder), tight at a uniform tail."""
    rng = np.random.default_rng(12)
    for _ in range(500):
        k = int(rng.integers(2, 40))
        p = _dirichlet_p(rng, k)
        T = rng.choice(k, size=int(rng.integers(1, k)), replace=False)
        out = np.setdiff1d(np.arange(k), T)
        for temp in (1.0001, 1.5, 2.0, 5.0, 50.0):
            q = 1 / temp
            lhs = (p[out] ** q).sum()
            bound = tail_certificate(p, T, temp)
            assert lhs <= bound * (1 + 1e-12) + 1e-300
            assert bound == pytest.approx(len(out) ** (1 - q) * residual(p, T) ** q, rel=1e-12)
    # tight at the uniform tail
    p = np.concatenate([[0.5], np.full(100, 0.005)])
    for temp in (1.5, 2.0, 4.0):
        assert (p[1:] ** (1 / temp)).sum() == pytest.approx(tail_certificate(p, [0], temp), rel=1e-12)
    # the temp<=1 expression is NOT valid for temp>1 (why the formula switches), and the K-dependence is real
    tail = np.concatenate([[0.4], np.full(1000, 1e-4)])
    assert (tail ** 0.5).sum() > 10 * tail.max() ** (0.5 - 1) * tail.sum()
    sums = [tail_certificate(np.concatenate([[0.5], np.full(K, 0.5 / K)]), [0], 2.0) for K in (10, 1000, 100000)]
    assert sums[0] < sums[1] < sums[2] and sums[2] > 200
    # edges
    assert tail_certificate(np.array([0.6, 0.4]), [0, 1], 2.0) == 0.0
    assert tail_certificate(np.array([1.0, 0.0]), [0], 2.0) == 0.0


def test_L3_nonpositive_temperature_raises():
    p = np.array([0.5, 0.25, 0.25])
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            tail_certificate(p, [0], bad)


# ---- L4 -------------------------------------------------------------------
def test_L4_post_selection_temperature():
    rng = np.random.default_rng(3)
    for _ in range(300):
        k = int(rng.integers(2, 30))
        p = _dirichlet_p(rng, k)
        T = rng.choice(k, size=int(rng.integers(1, k + 1)), replace=False)
        for temp in (2.0, 1.0, 0.5, 0.1):
            if p[T].sum() <= 0:
                continue
            w = post_select(p, T, temp)
            assert w.shape == (len(T) + 1,)
            assert (w >= 0).all()
            assert w.sum() == pytest.approx(1.0, abs=1e-12)
            assert w[-1] == pytest.approx(residual(p, T), abs=1e-12)
            # order on T preserved (strictly where p strictly ordered)
            pT = p[T]
            for i in range(len(T)):
                for j in range(len(T)):
                    if pT[i] > pT[j]:
                        assert w[i] > w[j]
    # T = all outcomes: other mass is 0; empty T: all mass is "other"
    assert np.allclose(post_select(np.array([0.7, 0.3]), [0, 1], 1.0), [0.7, 0.3, 0.0])
    # Amendment C: L4 needs sum_{k in T} p_k > 0
    with pytest.raises(ValueError):
        post_select(np.array([0.7, 0.3]), [], 1.0)
    with pytest.raises(ValueError):
        post_select(np.array([0.0, 0.0, 1.0]), [0, 1], 1.0)
    # extreme cold temperature does not overflow
    w = post_select(np.array([0.9, 0.05, 0.05]), [0, 1], 0.001)
    assert np.isfinite(w).all() and w.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        post_select(np.array([0.5, 0.5]), [0], 0.0)


# ---- L23-prime (Amendment A: E_new = E^{1/2} Q E^{1/2}, 0 <= Q <= I) -------------
def _null_rich_effects(rng, N, K, n_null):
    """K rank-one era effects plus a null effect that merges n_null frame rows (rank <= n_null)."""
    V = parseval_frame(N, K + n_null - 1, rng)
    rows = effects_from_frame(V)
    return rows[:K] + [sum(rows[K:])]


def _psd(M, tol=1e-12):
    return np.linalg.eigvalsh((M + M.conj().T) / 2).min() >= -tol


def _random_Q(rng, N):
    U = np.linalg.qr(rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))[0]
    return (U * rng.uniform(0, 1, N)) @ U.conj().T  # Hermitian, spectrum in [0, 1]


def _sqrtm_psd(M):
    w, U = np.linalg.eigh(M)
    return (U * np.sqrt(np.clip(w, 0, None))) @ U.conj().T


@pytest.mark.parametrize("n_null", [1, 2, 4])  # null effect of rank 1, 2, 4 (full rank at N=4)
def test_L23_split_null_nondestructive(n_null):
    rng = np.random.default_rng(4)
    N, K = 4, 3
    E = _null_rich_effects(rng, N, K, n_null)
    Enull = E[-1]
    S = _sqrtm_psd(Enull)
    Qs = [0.37 * np.eye(N), np.zeros((N, N)), np.eye(N), _random_Q(rng, N), _random_Q(rng, N)]
    for Q in Qs:
        E2 = split_null(E, Q)
        assert len(E2) == len(E) + 1
        assert np.allclose(sum(E2), np.eye(N), atol=1e-12)
        assert all(_psd(e) for e in E2)
        assert np.allclose(E2[-2], S @ Q @ S, atol=1e-12)
        assert np.allclose(E2[-1], S @ (np.eye(N) - Q) @ S, atol=1e-12)
        assert np.allclose(E2[-2] + E2[-1], Enull, atol=1e-12)
        psis = _psi(rng, N, 200)
        for k in range(K):  # existing-era probabilities unchanged to 1e-12
            for psi in psis:
                a = np.vdot(psi, E[k] @ psi).real
                b = np.vdot(psi, E2[k] @ psi).real
                assert abs(a - b) <= 1e-12
            assert np.array_equal(E2[k], E[k])  # not even touched
        pv = np.array([np.vdot(psis[0], e @ psis[0]).real for e in E2])
        assert pv.sum() == pytest.approx(1.0, abs=1e-12)


def test_L23_split_null_rejects_illegal_Q():
    rng = np.random.default_rng(5)
    N = 4
    E = _null_rich_effects(rng, N, 3, 2)
    with pytest.raises(ValueError):
        split_null(E, 1.5 * np.eye(N))  # Q not <= I
    with pytest.raises(ValueError):
        split_null(E, -0.1 * np.eye(N))  # Q not PSD
    nonherm = np.eye(N, dtype=complex) * 0.5
    nonherm[0, 1] = 0.3
    with pytest.raises(ValueError):
        split_null(E, nonherm)  # not Hermitian
    with pytest.raises(ValueError):
        split_null(E[:-1], 0.5 * np.eye(N))  # effects do not sum to I
    with pytest.raises(ValueError):
        split_null(E, 0.5 * np.eye(N - 1))  # shape mismatch


# ---- L23'' (optimal birth direction, Ky Fan) ---------------------------------
def _random_admissible_Q(rng, N, r):
    """0 <= Q <= I with rank <= r: U diag(d) U^H, d in [0, 1]."""
    U = np.linalg.qr(rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))[0][:, :r]
    return (U * rng.uniform(0, 1, r)) @ U.conj().T


@pytest.mark.parametrize("N,r", [(3, 1), (4, 2), (6, 3)])
def test_L23_double_prime_top_eigenprojector_is_optimal(N, r):
    rng = np.random.default_rng(13)
    E = _null_rich_effects(rng, N, 2, N)  # null of full rank
    Enull = E[-1]
    psis = _psi(rng, N, 40)
    C = sum(np.outer(p, p.conj()) for p in psis)
    S = _sqrtm_psd(Enull)
    lam = np.linalg.eigvalsh(S @ C @ S)[::-1]
    Q = birth_direction(Enull, C, r)
    assert np.allclose(Q @ Q, Q, atol=1e-10) and np.allclose(Q, Q.conj().T) and np.trace(Q).real == pytest.approx(r)
    best = captured_mass(Enull, C, Q)
    assert best == pytest.approx(lam[:r].sum(), rel=1e-10)  # equality in Ky Fan at the top-r projector
    for _ in range(10_000):  # no admissible Q of rank <= r does better
        rr = int(rng.integers(1, r + 1))
        assert captured_mass(Enull, C, _random_admissible_Q(rng, N, rr)) <= best + 1e-10
    split_null(E, Q)  # and the optimum is itself a legal split
    assert np.allclose(birth_direction(Enull, C, 0), 0)
    assert np.allclose(birth_direction(Enull, C, 10 * N), np.eye(N))  # r >= N clips to N

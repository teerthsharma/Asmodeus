"""L22 damped_loop_depth: F = (1-eps) G + eps/M on the simplex; dist(F^n x, w*) <= dist(x,Fx) K^n/(1-K), K=(1-eps)L."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa.ctc import contraction, damped_loop, iteration_bound, km_budget, post_lipschitz  # noqa: E402


def _chain(M=5, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(M), size=M)  # row-stochastic; w -> P^T w is an l1 non-expansion (L = 1)
    return (lambda w: P.T @ w), M


def _F(G, eps, M):
    return lambda w: (1 - eps) * G(w) + eps / M


def test_L22_apriori_distance_bound():
    G, M = _chain()
    eps, K = 0.2, 0.8
    F = _F(G, eps, M)
    w0 = np.eye(M)[0]
    wstar = w0
    for _ in range(2000):
        wstar = F(wstar)
    d0 = np.abs(F(w0) - w0).sum()
    w = w0
    for n in range(0, 40):
        assert np.abs(w - wstar).sum() <= d0 * K**n / (1 - K) + 1e-12
        w = F(w)


def test_L22_iteration_count_obeys_bound():
    G, M = _chain(seed=1)
    eps, tol = 0.2, 1e-8
    w0 = np.eye(M)[2]
    res = damped_loop(G, w0, eps, L=1.0, tol=tol)
    w, n, ok = res
    d0 = np.abs(_F(G, eps, M)(w0) - w0).sum()
    assert ok and res.mode == "DAMPED" and res.c_loc <= res.c_worst == pytest.approx(0.8)
    assert n <= iteration_bound(d0, res.c_loc, tol) <= iteration_bound(d0, 0.8, tol)
    assert abs(w.sum() - 1) < 1e-12 and np.all(w >= 0)


def test_L22_fixed_point_seed_converges_in_one_step():  # faraday style
    G, M = _chain(seed=2)
    eps = 0.3
    w, _, _ = damped_loop(G, np.ones(M) / M, eps, L=1.0, tol=1e-12)
    w2, n, ok = damped_loop(G, w, eps, L=1.0, tol=1e-9)
    assert ok and n == 1 and np.allclose(w, w2)


def test_L22dagger_refuses_expansive_G():
    sharpen = lambda w: w**3 / (w**3).sum()  # tangent l1 norm > 1 away from the uniform point
    w0 = np.array([.5, .3, .2])
    res = damped_loop(sharpen, w0, eps=0.1, L=None, tol=1e-9)
    w, n, ok = res
    assert res.mode == "REFUSE" and not ok and n == 0 and np.allclose(w, w0)
    assert res.c_loc > 1 and res.c_worst == np.inf


def test_L22dagger_km_for_nonexpansive_undamped():
    roll = lambda w: np.roll(w, 1)  # l1 isometry: not a contraction, nonexpansive; eps = 0
    w0 = np.array([.7, .2, .1])
    res = damped_loop(roll, w0, eps=0.0, L=1.0, tol=1e-9)
    assert res.mode == "KM"
    w, n, ok = res
    assert ok and n <= km_budget(1e-9) and np.allclose(w, 1 / 3)
    assert np.abs(roll(w) - w).sum() <= 1e-8  # honest: the returned point IS a fixed point of G


def test_L22dagger_local_beats_worst_case():
    rng = np.random.default_rng(0)
    M, beta, eps = 5, 1.5, 0.2
    A = rng.uniform(-1, 1, size=(M, M))
    G = lambda w: np.exp(beta * (A @ w)) / np.exp(beta * (A @ w)).sum()
    L_worst = beta * np.abs(A).max()  # |s(z)-s(z')|_1 <= |z-z'|_inf, |A v|_inf <= max|A| |v|_1
    res = damped_loop(G, np.ones(M) / M, eps, L=L_worst, tol=1e-10)
    print(f"c_loc={res.c_loc:.4f} c_worst={res.c_worst:.4f} ratio={res.c_loc / res.c_worst:.4f} n={res.n_iters if hasattr(res, 'n_iters') else res[1]}")
    assert res.c_worst >= 1 > res.c_loc  # worst-case certificate is vacuous, orbit-local one certifies
    assert res.mode == "DAMPED" and res[2]
    assert np.abs(0.8 * G(res[0]) + 0.2 / M - res[0]).sum() <= 1e-9


def test_L22dagger_uncertified_T_refuses_and_budget_honest():
    G, M = _chain(seed=3)
    _, n, ok = damped_loop(G, np.eye(M)[0], 0.2, 1.0, 1e-14, max_iter=3)
    assert not ok and n == 3  # budget too small: reported, not hidden
    res = damped_loop(G, np.eye(M)[0], 0.2, 1.0, 1e-6, certified=False)
    assert res.mode == "REFUSE" and not res[2] and res[1] == 0


def test_L22_rejects_non_simplex_start():
    with pytest.raises(ValueError):
        damped_loop(lambda w: w, np.array([2.0, -1.0]), 0.1, 1.0, 1e-6)


def _lift(w, T, M):
    out = np.zeros(M + 1)
    out[list(T)] = w[:-1]
    out[M] = w[-1]
    return out


def test_L22star_post_selection_not_lipschitz_across_flip():
    ob = pytest.importorskip("fcpa.observer")
    d = 1e-5
    p, q = np.array([.5 + d, .5 - d, 0.0]), np.array([.5 - d, .5 + d, 0.0])  # m = 1, the top-1 set flips
    wp, wq = _lift(ob.post_select(p, [0], 1.0), [0], 3), _lift(ob.post_select(q, [1], 1.0), [1], 3)
    ratio = np.abs(wp - wq).sum() / np.abs(p - q).sum()
    assert ratio > 100 * post_lipschitz(1.0, 0.25)  # no finite L covers the flip


def test_L22star_post_lipschitz_bound_for_fixed_T():
    ob = pytest.importorskip("fcpa.observer")
    rng = np.random.default_rng(7)
    delta, T = 0.05, [0, 1]
    for temp in (0.5, 1.0):
        worst = 0.0
        for _ in range(3000):
            p = rng.dirichlet(np.ones(4))
            if p[T].min() < delta:
                continue
            q = np.abs(p + 0.02 * rng.standard_normal(4))
            q /= q.sum()
            if q[T].min() < delta:
                continue
            worst = max(worst, np.abs(ob.post_select(p, T, temp) - ob.post_select(q, T, temp)).sum() / np.abs(p - q).sum())
        assert worst <= post_lipschitz(temp, delta)


def test_L22star_contraction_constant():
    assert contraction(0.2, 0.5, 1.0) == pytest.approx(0.8 * 0.5 * 1.0)
    assert post_lipschitz(0.5, 0.1) == pytest.approx(2 + 2 / (0.5 * 0.1))


# ---- L22* inequalities (the claims behind "c < 1"), not the formulas ----------------------------------
def _softmax_map(seed, M=5, beta=0.8):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1, 1, size=(M, M))
    G = lambda w: np.exp(beta * (A @ w)) / np.exp(beta * (A @ w)).sum()
    return G, beta * np.abs(A).max(), M  # |s(z)-s(z')|_1 <= |z-z'|_inf, so L_G = beta max|A|


def _fixed(G, w0, n=5000):
    w = w0
    for _ in range(n):
        w = G(w)
    return w


def test_L22star_residual_at_damped_fixed_point_is_at_most_2eps():
    from fcpa.ctc import residual_bound

    worst = 0.0
    for seed in range(6):
        G, L, M = _softmax_map(seed, beta=1.5)
        for eps in (0.05, 0.2, 0.5):
            w, _, ok = damped_loop(G, np.ones(M) / M, eps, L=None, tol=1e-12, max_iter=100_000)
            assert ok
            res = np.abs(G(w) - w).sum()  # undamped residual at the damped fixed point
            assert res <= residual_bound(eps) + 1e-12
            worst = max(worst, res / residual_bound(eps))
    assert worst > 0.05  # the bound is exercised, not vacuous


def test_L22star_bias_to_undamped_fixed_point_is_at_most_2eps_over_1_minus_c():
    from fcpa.ctc import bias_bound

    worst = 0.0
    for seed in range(6):
        G, L, M = _softmax_map(seed, beta=0.8)  # L_G < 1 so the undamped fixed point is the limit of G^n
        assert L < 1
        wbar = _fixed(G, np.ones(M) / M)
        assert np.abs(G(wbar) - wbar).sum() < 1e-12
        for eps in (0.05, 0.2, 0.5):
            c = contraction(eps, L)
            w, _, ok = damped_loop(G, np.ones(M) / M, eps, L=L, tol=1e-12, max_iter=100_000)
            bias = np.abs(w - wbar).sum()
            assert ok and bias <= bias_bound(eps, c) + 1e-12
            worst = max(worst, bias / bias_bound(eps, c))
    G, M = _chain(seed=4)  # linear chain, declared L = 1: wbar is the stationary law
    ev, U = np.linalg.eig(np.array([G(e) for e in np.eye(M)]).T)
    wbar = np.real(U[:, np.argmax(ev.real)])
    wbar /= wbar.sum()
    for eps in (0.05, 0.3):
        w, _, ok = damped_loop(G, np.eye(M)[0], eps, L=1.0, tol=1e-14)
        assert ok and np.abs(w - wbar).sum() <= bias_bound(eps, contraction(eps, 1.0)) + 1e-12
    assert worst > 0.01


def test_L22star_support_T_constant_on_orbit_when_certified_on_l1_ball():
    from fcpa.ctc import orbit_radius
    from fcpa.router import certified_topm

    rng = np.random.default_rng(0)
    n_cert = 0
    for trial in range(120):
        G0, L0, M = _softmax_map(trial % 6, beta=1.5)
        s = rng.dirichlet(0.3 * np.ones(M))  # a peaked anchor, so the top-m margins are real
        lam = 0.3
        G, L = (lambda w, G0=G0, s=s: (1 - lam) * s + lam * G0(w)), lam * L0
        eps, m = 0.1, int(rng.integers(1, 3))
        c = contraction(eps, L)
        F = _F(G, eps, M)
        wstar = _fixed(F, np.ones(M) / M, n=200)
        w0 = np.abs(wstar + 0.005 * rng.standard_normal(M)) + 1e-9
        w0 /= w0.sum()
        d0 = np.abs(F(w0) - w0).sum()
        T, _ = certified_topm(w0, orbit_radius(d0, c), m)  # l1 ball radius => per-coordinate radius
        if T is None:
            continue
        n_cert += 1
        w = w0
        for _ in range(100):
            assert set(np.argsort(-w, kind="stable")[:m].tolist()) == set(T)
            w = F(w)
    assert n_cert > 30
    # necessity: a non-contractive-looking start (G constant, c = 0) whose top-1 flips is NOT certified
    target = np.array([.1, .7, .2])
    F2 = _F(lambda w: target, 0.0, 3)
    w0 = np.array([.5, .3, .2])
    d0 = np.abs(F2(w0) - w0).sum()
    T, fr = certified_topm(w0, orbit_radius(d0, 0.0), 1)
    assert T is None and fr is not None
    assert int(np.argmax(F2(w0))) != int(np.argmax(w0))  # the support really changes


def test_local_norm_batched_equals_pointwise():
    """The batched secant evaluation (G.batch) must agree with calling G point by point."""
    from fcpa.ctc import local_norm

    rng = np.random.default_rng(0)
    A = rng.uniform(-1, 1, size=(6, 6))
    G = lambda w: np.exp(A @ w) / np.exp(A @ w).sum()
    Gb = lambda W: np.exp(W @ A.T) / np.exp(W @ A.T).sum(1, keepdims=True)
    w = rng.dirichlet(np.ones(6))
    plain = local_norm(G, w)
    G.batch = Gb
    assert local_norm(G, w) == pytest.approx(plain, rel=1e-6)

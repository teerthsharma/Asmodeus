"""H4: Hilbert projective metric for the CTC loop. G(w) = normalise(mu * exp(beta A w / T)) on the simplex.

d_H(x,y) = osc(log x - log y) is a metric on positive vectors modulo scale; normalisation and a fixed positive mask
mu are d_H isometries, the power 1/T scales d_H by exactly 1/T, and the log-linear kernel exp(beta A w) has
d_H-Lipschitz constant kappa = beta*D/4 (D = max_{a,b,i,j} A_ai - A_bi - A_aj + A_bj) on the simplex.
Top-m is handled by a SOFT POSITIVE MASK mu (fixed, > 0), not a hard support: hard top-m flips are outside the claim.
Config = test_ctc.test_L22dagger_local_beats_worst_case (M=5, beta=1.5, A ~ U(-1,1) seed 0, orbit-local 0.418).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa.hilbert import birkhoff_coeff, dH, iterate, loglinear_kappa, oscillation, tt_map  # noqa: E402

M, BETA, N_PAIRS, TOL = 5, 1.5, 10_000, 1e-9


def _A(seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, size=(M, M))  # same draw as test_ctc (seed 0)


def _pairs(A, beta, T, n=N_PAIRS, seed=1):
    """n pairs on the simplex: local, far, min coordinate 1e-12 in both/one, and kernel-adversarial (mass 1/2 on i,j)."""
    rng = np.random.default_rng(seed)
    norm = lambda w: w / w.sum()
    out = []
    for _ in range(3000):  # local, log-uniform scale, d_H from ~1e-4 to ~10
        w = np.exp(rng.uniform(-6, 0, M))
        out.append((norm(w), norm(w * np.exp(10 ** rng.uniform(-4, 0.7) * rng.standard_normal(M)))))
    for _ in range(2000):  # shared coordinate at 1e-12 (min coordinate), both sides
        w = np.exp(rng.uniform(-3, 0, M))
        v = w * np.exp(10 ** rng.uniform(-4, 0.3) * rng.standard_normal(M))
        k = rng.integers(M)
        w[k], v[k] = 1e-12, 1e-12 * np.exp(10 ** rng.uniform(-4, 0) * rng.standard_normal())
        out.append((norm(w), norm(v)))
    for _ in range(2000):  # far apart, 1e-12 on one side only
        w, v = np.exp(rng.uniform(-5, 0, M)), np.exp(rng.uniform(-5, 0, M))
        w[rng.integers(M)] = 1e-12
        out.append((norm(w), norm(v)))
    D = A[:, None, :] - A[None, :, :]  # D[a,b,i] = A_ai - A_bi
    a, b = np.unravel_index(np.argmax(D.max(-1) - D.min(-1)), (M, M))
    i, j = int(np.argmax(D[a, b])), int(np.argmin(D[a, b]))
    for e in 10 ** rng.uniform(-4, -1, n - len(out)):  # tight pairs: move mass e from i to j
        w = np.full(M, 1e-12)
        w[[i, j]] = 0.5
        v = w.copy()
        v[i] -= e
        v[j] += e
        out.append((norm(w), norm(v)))
    assert len(out) == n
    return out


def _ratios(A, beta, T, mu=None):
    G = tt_map(A, beta, T, mu)
    return np.array([dH(G(w), G(v)) / dH(w, v) for w, v in _pairs(A, beta, T)])


def test_H4_dH_is_a_projective_metric():
    rng = np.random.default_rng(3)
    x, y, z = (np.exp(rng.standard_normal(M)) for _ in range(3))
    assert dH(x, x) == 0 and dH(x, 7.0 * x) == pytest.approx(0, abs=1e-12)  # scale-blind
    assert dH(x, y) == pytest.approx(dH(y, x)) and dH(x, z) <= dH(x, y) + dH(y, z) + 1e-12
    assert dH(x, y) == pytest.approx(oscillation(np.log(x) - np.log(y)))
    mu = np.exp(rng.standard_normal(M))
    assert dH(mu * x, mu * y) == pytest.approx(dH(x, y))  # a fixed positive mask is an isometry
    assert dH(x ** (1 / 2.5), y ** (1 / 2.5)) == pytest.approx(dH(x, y) / 2.5)  # power 1/T: Lipschitz exactly 1/T


def test_H4_birkhoff_coefficient_bounds_positive_linear_maps():
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(40):
        K = np.exp(rng.uniform(-1, 1, (M, M)))
        c = birkhoff_coeff(K)
        assert 0 < c < 1
        for _ in range(200):
            x, y = np.exp(rng.uniform(-4, 4, M)), np.exp(rng.uniform(-4, 4, M))
            worst = max(worst, dH(K @ x, K @ y) / dH(x, y) / c)
    assert worst <= 1 + TOL and worst > 0.2  # Birkhoff holds and is exercised


T_CERT = 1.25  # smallest grid temperature above kappa


def test_H4_bar_kappa_over_T_below_one():
    """Pre-registered bar kappa/T < 1. At the config's T = 1 it FAILS and the failure is real (the measured d_H
    ratio exceeds 1 on 3002 pairs, below), so the bar holds only once the post-selection power has T > kappa."""
    k = loglinear_kappa(_A(), BETA)
    assert k / 1.0 > 1 and _ratios(_A(), BETA, 1.0).max() > 1  # T = 1: d_H-expansive, no certificate
    assert k / T_CERT < 1  # T = 1.25: certified


@pytest.mark.parametrize("T", [1.0, T_CERT])
def test_H4_measured_dH_ratios_never_exceed_kappa_over_T(T):
    A = _A()
    kT = loglinear_kappa(A, BETA) / T
    r = _ratios(A, BETA, T)
    assert len(r) == N_PAIRS and np.isfinite(r).all()
    assert r.max() <= kT + TOL, f"max ratio {r.max():.6f} > kappa/T {kT:.6f}"
    assert r.max() >= 0.9 * kT  # the adversarial pairs make the bound tight, not slack


def test_H4_soft_positive_mask_changes_nothing():
    A = _A()
    mu = np.where(np.arange(M) < 2, 1.0, 0.05)  # soft top-2 mask: dropped coordinates keep weight 0.05 > 0
    kT = loglinear_kappa(A, BETA) / T_CERT
    assert _ratios(A, BETA, T_CERT, mu).max() <= kT + TOL


def test_H4_undamped_iteration_converges_geometrically_at_kappa_over_T():
    A = _A()
    q = loglinear_kappa(A, BETA) / T_CERT
    assert q < 1
    mu = np.where(np.arange(M) < 2, 1.0, 0.05)
    rng = np.random.default_rng(9)
    starts = [np.ones(M) / M, np.eye(M)[0] * (1 - 4e-12) + 1e-12, np.exp(rng.uniform(-8, 0, M))]
    for m in (None, mu):
        G = tt_map(A, BETA, T_CERT, m)
        for w0 in starts:
            w0 = w0 / w0.sum()
            w, n, cert, steps = iterate(G, w0, q, 1e-10)  # eps = 0: no damping, no mixing
            assert cert <= 1e-10 and n <= math.ceil(math.log(1e-10 * (1 - q) / (q * steps[0])) / math.log(q)) + 1
            rho = [s1 / s0 for s0, s1 in zip(steps, steps[1:]) if s0 > 1e-13]
            assert max(rho) <= q + TOL  # every step ratio, not just the average
            assert dH(G(w), w) <= (1 - q) / q * 1e-10 + 1e-12  # w is a fixed point of the UNdamped G
            assert abs(w.sum() - 1) < 1e-12

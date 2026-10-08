"""Contracts for atlas/nerve_persistence.py. Fixture beta=4.0 (octahedral cover of S^2)."""
import numpy as np
import pytest

import metrics
from atlas.nerve_persistence import betti_at, certified_betti, nerve_filtration, persistence

BETA = 4.0


def _softmax(z):
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def octa(n=3000, beta=BETA, seed=0):
    p = np.random.default_rng(seed).normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    e = np.vstack([np.eye(3), -np.eye(3)])
    return _softmax(beta * p @ e.T)


@pytest.fixture(scope="module")
def w():
    return octa()


@pytest.fixture(scope="module")
def clean(w):
    filt = nerve_filtration(w, max_dim=3)
    return filt, persistence(filt, 2)


def test_betti_at_matches_metrics(clean):
    filt, bars = clean
    for tau in (1 / 4, 1 / 5, 1 / 6, 1 / 8):
        ref = metrics.betti_z2({s for s, f in filt.items() if f >= tau}, 2)
        assert betti_at(bars, tau, 2) == tuple(ref)


def test_octahedral_s2_certified(clean):
    v, length, (lo, hi) = certified_betti(clean[1], 0.05, 2)
    print("octa clean interval", v, length, lo, hi)
    assert v == (1, 0, 1) and length > 0.05


def test_disjoint_charts_window_does_not_win():
    # two disjoint charts: window (2,0,..) over [tau_min, 0.5) is short; (1,..) never exists -> no wrong win
    w2 = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5)
    bars = persistence(nerve_filtration(w2, tau_min=0.1), 1)
    assert certified_betti(bars, 0.05, 1, tau_min=0.1)[0] is None
    # finite H0 bars: six charts merging at 0.5 then staying connected: the connected window must win
    bars = [(0, 0.9, 0.5)] * 5 + [(0, 0.9, 0.0)]
    v, length, _ = certified_betti(bars, 0.05, 0)
    assert v == (1,) and length > 0.4


def test_filtration_monotone(clean):
    filt, _ = clean
    for s, f in filt.items():
        for i in range(len(s)) if len(s) > 1 else ():
            face = s[:i] + s[i + 1:]
            assert face in filt and filt[face] >= f


def test_stability_under_sup_noise(w, clean):
    eps = 0.01
    noisy = w + np.random.default_rng(1).uniform(-eps, eps, w.shape)
    bars_n = persistence(nerve_filtration(noisy, max_dim=3), 2)
    vc, lc, _ = certified_betti(clean[1], 0.05, 2)
    vn, ln, bn = certified_betti(bars_n, 0.05, 2)
    print("octa noisy interval", vn, ln, bn)
    assert vn == vc and abs(ln - lc) <= 2 * eps


def _outliers(w, frac):
    return np.vstack([w, np.random.default_rng(2).dirichlet(np.ones(6), size=int(frac * len(w)))])


def test_outlier_rows_need_support(w):
    wo = _outliers(w, 0.005)  # 0.5%: naive already breaks
    naive = certified_betti(persistence(nerve_filtration(wo, max_dim=3, n_min=1), 2), 0.05, 2)
    assert naive[0] != (1, 0, 1)
    nm = max(3, int(0.006 * len(wo)))
    v, length, bd = certified_betti(persistence(nerve_filtration(wo, max_dim=3, n_min=nm), 2), 0.05, 2)
    print("octa 0.5% outlier interval", v, length, bd)
    assert v == (1, 0, 1)


def test_outlier_ceiling_documented(w):
    wo = _outliers(w, 0.05)  # 5% > n_min/N: support guard no longer covers it
    v = certified_betti(persistence(nerve_filtration(wo, max_dim=3, n_min=30), 2), 0.05, 2)[0]
    assert v != (1, 0, 1)


def test_two_charts_on_s2_is_contractible():
    p = np.random.default_rng(0).normal(size=(3000, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    w2 = _softmax(BETA * p @ np.array([[0, 0, 1], [0, 0, -1]]).T)
    assert certified_betti(persistence(nerve_filtration(w2, max_dim=3), 2), 0.05, 2)[0] == (1, 0, 0)


def test_essential_component():
    bars = persistence(nerve_filtration(np.ones((10, 1))), 3)
    assert bars == [(0, 1.0, 0.0)]

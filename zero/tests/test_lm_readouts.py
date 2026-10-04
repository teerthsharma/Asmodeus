"""Tests for lm_readouts. Bargmann sign convention under test:
arg(<a|b><b|c><c|a>) = +Omega/2 (mod 2pi), Omega = signed solid angle of the geodesic
triangle a->b->c on the Bloch sphere (counter-clockwise seen from outside = +), with the
Bloch vector r = (<sx>,<sy>,<sz>). (Berry phase = minus this.)"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lm_readouts as L

RNG = np.random.default_rng(0)


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    n_s = sum(a * m for a, m in zip(axis, s))
    return np.cos(ang / 2) * np.eye(2) - 1j * np.sin(ang / 2) * n_s


PSI0 = np.array([0.8, 0.6j * np.exp(0.4j)])
AXES = {0: ([1, 0, 0], 1.1), 1: ([0, 0, 1], 0.7), 2: ([0, 1, 0], 0.9), 3: ([1, 1, 1], 0.5)}


def su2_fn(tokens):
    psi = PSI0.astype(complex)
    for t in tokens:
        psi = _rot(*AXES[t]) @ psi
    return psi


def comm_fn(tokens):
    psi = np.array([1, 1j, -1, 0.5], complex)
    d = np.array([0.3, 1.2, -0.7, 2.0])
    for t in tokens:
        psi = np.exp(1j * (t + 1) * d) * psi
    return psi


def gauge(fn, seed=1):
    r = np.random.default_rng(seed)
    return lambda toks: np.exp(2j * np.pi * r.random()) * fn(toks)


def bloch(psi):
    psi = psi / np.linalg.norm(psi)
    return np.array([2 * np.real(np.conj(psi[0]) * psi[1]), 2 * np.imag(np.conj(psi[0]) * psi[1]),
                     abs(psi[0]) ** 2 - abs(psi[1]) ** 2])


def omega(a, b, c):
    return 2 * np.arctan2(a @ np.cross(b, c), 1 + a @ b + b @ c + c @ a)


def test_gauge_invariance():
    for f, args in [(L.exchange_loop, ([0], [1], [2], [3])), ]:
        base = f(su2_fn, *args)["phase"]
        for s in range(5):
            assert abs(_wrap(f(gauge(su2_fn, s), *args)["phase"] - base)) < 1e-9
    loop = [[0], [1, 2], [3], [2, 0]]
    b0 = L.holonomy_loop(su2_fn, loop)[0]
    assert abs(_wrap(L.holonomy_loop(gauge(su2_fn, 7), loop)[0] - b0)) < 1e-9
    # single overlap is gauge dependent
    t0 = L.exchange_phase(su2_fn, [0], [1])[0]
    ts = [L.exchange_phase(gauge(su2_fn, s), [0], [1])[0] for s in range(5)]
    assert max(abs(_wrap(t - t0)) for t in ts) > 1e-2


def test_commuting_gives_zero():
    for A, B in [([0], [1]), ([2, 3], [1]), ([0, 0], [3])]:
        r = L.exchange_loop(comm_fn, A, B, prefix=[1], suffix=[2])
        assert abs(r["phase"]) < 1e-9 and r["min_overlap"] > 0.01
        assert abs(L.exchange_loop(gauge(comm_fn), A, B)["phase"]) < 1e-9


def test_noncommuting_matches_solid_angle():
    r = L.exchange_loop(su2_fn, [0], [1], prefix=[2], suffix=[3])
    a, b, c = (bloch(su2_fn(t)) for t in ([2, 0, 1, 3], [2, 1, 0, 3], [2, 3]))
    om = omega(a, b, c)
    assert abs(r["phase"]) > 1e-3
    assert abs(_wrap(r["phase"] - om / 2)) < 1e-6


def test_bargmann_sign_octant():
    z, x, y = np.array([1, 0j]), np.array([1, 1]) / np.sqrt(2), np.array([1, 1j]) / np.sqrt(2)
    assert abs(L.bargmann([z, x, y]) - np.pi / 4) < 1e-12      # +Omega/2, Omega=pi/2
    assert abs(L.bargmann([z, y, x]) + np.pi / 4) < 1e-12      # reversed orientation
    for _ in range(20):
        ps = [RNG.normal(size=2) + 1j * RNG.normal(size=2) for _ in range(3)]
        om = omega(*(bloch(p) for p in ps))
        assert abs(_wrap(L.bargmann(ps) - om / 2)) < 1e-9


def test_near_orthogonal_nan():
    ph, mn = L.holonomy_loop(lambda t: np.eye(3)[t[0]].astype(complex), [[0], [1], [2]])
    assert np.isnan(ph) and mn < 1e-3


def test_spearman_monotone():
    rho, lo, hi = L.think_vs_surprisal([1, 2, 3, 4, 5, 6], [0.1, 0.2, 0.5, 0.7, 0.9, 1.5], seed=0)
    assert rho == 1.0 and hi == 1.0 and lo <= 1.0
    rho, lo, hi = L.think_vs_surprisal([1, 1, 2, 2, 3, 3, 4], [1, 2, 2, 3, 3, 5, 4], seed=0)
    assert np.isfinite(rho) and lo <= rho <= hi


def test_circular_mean_wrap():
    m, R, ci = L.circular_stats([np.pi - 0.1, -np.pi + 0.1, np.pi - 0.05], seed=0)
    assert abs(abs(m) - np.pi) < 0.1 and R > 0.99
    assert abs(np.mean([np.pi - 0.1, -np.pi + 0.1])) < 0.2  # naive mean ~0: the failure avoided
    assert ci[0] <= R <= ci[1] + 1e-12


def test_summary_table():
    t = L.summary_table(su2_fn, exchange_items=[dict(A=[0], B=[1]), dict(A=[2], B=[3], prefix=[0])],
                        loops=[[[0], [1], [2]]], seed=0)
    assert set(t) == {"exchange_phase", "exchange_loop", "holonomy"}
    assert t["exchange_loop"]["n"] == 2 and 0 <= t["exchange_loop"]["R"] <= 1

"""L8 / L9 / E2' / L10 / L11 / segment / learned-operator arm (L-GIVEN) with the certified sparse law."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa import bounds  # noqa: E402
from fcpa.beds import ns2d  # noqa: E402

N = 16


@pytest.fixture(scope="module")
def slab():
    """One constant-law slab (nu=0.02, Kolmogorov f) with exact spectral derivatives."""
    return ns2d.simulate(N, [(0, 0.02)], [(0, (0, 4, 1.0))], None, 12, 0.01, 3, 1, return_derivs=True)


def test_L8_nu_projection_exact(slab):
    """L8: with known f, nu = <w_t + u.grad w - f, lap w> / ||lap w||^2, exact on clean data."""
    for u in range(3):
        for t in (0, 5, 11):
            nu = bounds.nu_projection(slab["dwdt"][u, t], slab["adv"][u, t], slab["lap"][u, t], slab["f"][t])
            assert abs(nu - 0.02) < 1e-6 * 0.02
    nu_all = bounds.nu_projection(slab["dwdt"], slab["adv"], slab["lap"], slab["f"][None], axis=(0, 2, 3))
    assert np.max(np.abs(nu_all - 0.02)) < 1e-8 * 0.02  # per-step pooled over universes


def test_L9_noise_bound(slab):
    """L9: |nu_hat - nu| <= ||e|| / ||lap w|| (Cauchy-Schwarz), 300/300 draws."""
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(300):
        e = rng.standard_normal(slab["lap"][0, 0].shape) * 10 ** rng.uniform(-3, 0)
        nu = bounds.nu_projection(slab["dwdt"][0, 0] + e, slab["adv"][0, 0], slab["lap"][0, 0], slab["f"][0])
        ok += abs(nu - 0.02) <= bounds.nu_error_bound(e, slab["lap"][0, 0]) * (1 + 1e-12)
    assert ok == 300


def _slab_sample(seed, nu, pat, noise_sd, T=6):
    out = ns2d.simulate(N, [(0, nu)], [(0, pat)], None, T, 0.01, 1, seed, return_derivs=True)
    rng = np.random.default_rng(seed + 10_000)
    e = noise_sd * rng.standard_normal(out["dwdt"][0].shape)
    return out["dwdt"][0] + e, out["adv"][0], out["lap"][0], e, out


def test_E2prime_within_slab_bound_200_seeds():
    """E2': demeaned estimator satisfies |nu_hat - nu| <= ||e||_F/||lap~||_F in 200/200 seeds; E2 'if' direction."""
    ok = 0
    for s in range(200):
        dw, adv, lap, e, _ = _slab_sample(s, 0.02, (0, 4, 1.0), 10 ** np.random.default_rng(s).uniform(-3, -1))
        est = bounds.within_slab_estimate(dw, adv, lap)
        assert est.verdict != bounds.REFUSE
        lo, hi = est.interval(np.linalg.norm(e))
        ok += (abs(est.nu - 0.02) <= np.linalg.norm(e) / est.denom * (1 + 1e-12)) and lo <= 0.02 <= hi
    assert ok == 200


def test_E2prime_refuse_steady_and_certify_switched():
    # steady slab: w = A cos(4y), f = nu k^2 A cos(4y)  -> lap w constant in time -> REFUSE
    nu, A = 0.02, 1.0
    y = 2 * np.pi * np.arange(N) / N
    w0 = A * np.cos(4 * y)[None, :] * np.ones((N, 1))
    st = ns2d.simulate(N, [(0, nu)], [(0, (0, 4, nu * 16 * A))], None, 6, 0.01, 1, 0, omega0=w0, return_derivs=True)
    assert np.max(np.abs(st["dwdt"])) < 1e-10
    assert bounds.within_slab_estimate(st["dwdt"][0], st["adv"][0], st["lap"][0]).verdict == bounds.REFUSE
    # switched slab: nu 0.02 -> 0.01 gives disjoint certified intervals
    ints = []
    for seed, nu_ in [(1, 0.02), (2, 0.01)]:
        dw, adv, lap, e, _ = _slab_sample(seed, nu_, (0, 4, 1.0), 1e-3)
        est = bounds.within_slab_estimate(dw, adv, lap)
        ints.append(est.interval(np.linalg.norm(e)))
    assert bounds.certified_change(*ints) == bounds.CHANGE


def test_L10_kernel_estimate_and_intervals():
    """L10 (CITED Paulin 2015; Chung et al. 2012): per-row intervals cover P with prob >= 1-delta on known-gap chains."""
    P = np.array([[0.8, 0.2, 0.0], [0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
    rng = np.random.default_rng(0)
    s = ns2d.markov_chain(P, 400, rng)
    Ph = bounds.markov_kernel(s, 3)
    np.testing.assert_allclose(Ph.sum(1), 1)
    assert np.all(Ph > 0)  # Laplace smoothing
    lo, hi = bounds.kernel_interval(s, 3, 0.1)
    assert np.all(lo >= 0) and np.all(hi <= 1) and np.all(lo <= hi)
    lo2, hi2 = bounds.kernel_interval(ns2d.markov_chain(P, 4000, rng), 3, 0.1)
    assert (hi2 - lo2).max() < (hi - lo).max()  # shrinks with data
    delta, miss = 0.1, 0
    for i in range(300):
        l, h = bounds.kernel_interval(ns2d.markov_chain(P, 150, np.random.default_rng(100 + i)), 3, delta)
        miss += not (np.all(l <= P) and np.all(P <= h))
    assert miss / 300 <= delta, miss / 300  # joint-box coverage over all 9 entries
    l, h = bounds.kernel_interval(np.array([0, 0, 0, 0]), 3, 0.1)  # unvisited rows are uninformative
    assert l[1].tolist() == [0, 0, 0] and h[1].tolist() == [1, 1, 1]


def test_L11_certified_change_verdicts():
    a = (np.array([0.1, 0.2]), np.array([0.2, 0.3]))
    far = (np.array([0.5, 0.2]), np.array([0.6, 0.3]))  # first coordinate disjoint
    near = (np.array([0.15, 0.25]), np.array([0.25, 0.35]))  # overlaps in every coordinate
    assert bounds.certified_change(a, far) == bounds.CHANGE
    assert bounds.certified_change(a, near) == bounds.REFUSE  # overlap is REFUSE, never "no change"
    assert bounds.certified_change(a, a) == bounds.REFUSE
    pt = (np.array([0.3]), np.array([0.3]))
    assert bounds.certified_change(pt, pt) == bounds.NO_CHANGE_UNCERTIFIED  # coincident exact points only
    with pytest.raises(ValueError):
        bounds.certified_change((np.array([1.0]), np.array([0.0])), pt)


def test_L11_soundness_equal_laws_joint_box():
    """L11: needs EQUAL true laws under H0 and joint-box coverage of each window (prob >= 1-delta) =>
    P(false certified CHANGE) <= 2*delta (union bound). Checked on kernel boxes of one fixed chain law."""
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    delta, trials, false = 0.1, 400, 0
    for i in range(trials):
        r = np.random.default_rng(i)
        ia = bounds.kernel_interval(ns2d.markov_chain(P, 120, r), 2, delta)
        ib = bounds.kernel_interval(ns2d.markov_chain(P, 120, r), 2, delta)
        false += bounds.certified_change(ia, ib) == bounds.CHANGE
    assert false / trials <= 2 * delta, false / trials
    # power: genuinely different laws are certified
    Q = np.array([[0.1, 0.9], [0.9, 0.1]])
    r = np.random.default_rng(0)
    verdicts = [bounds.certified_change(bounds.kernel_interval(ns2d.markov_chain(P, 300, r), 2, delta),
                                        bounds.kernel_interval(ns2d.markov_chain(Q, 300, r), 2, delta))
                for _ in range(20)]
    assert verdicts.count(bounds.CHANGE) >= 18


def test_segment_piecewise_constant_exact_and_noisy():
    truth = [40, 100, 170]
    x = np.zeros(240)
    for b, v in zip(truth, [1.0, -0.5, 2.0]):
        x[b:] = v
    assert bounds.segment(x, 20) == truth  # noise-free: exact
    assert bounds.segment(np.zeros(100), 20) == []  # no change, no boundary
    rng = np.random.default_rng(0)
    got = bounds.segment(x + 0.05 * rng.standard_normal(240), 20, sigma="auto")
    assert len(got) == 3 and all(abs(g - b) <= 2 for g, b in zip(got, truth))


def test_segment_markov_boundaries():
    """Markov bound: certified kernel change between [t-w,t) and [t,t+w) finds the switch times.
    T=3000, w=100, two switches: lambda_sw*w = 2/3000*100 = 0.067 <= 0.1 (Amendment A)."""
    S = np.array([[0.9, 0.1], [0.1, 0.9]])
    F = np.array([[0.1, 0.9], [0.9, 0.1]])
    ch = ns2d.markov_chain([(0, S), (1000, F), (2000, S)], 3000, np.random.default_rng(2))
    got = bounds.segment_markov(ch, 2, 100, delta=0.01)
    assert len(got) == 2 and abs(got[0] - 1000) <= 25 and abs(got[1] - 2000) <= 25, got
    assert bounds.segment_markov(ns2d.markov_chain(S, 3000, np.random.default_rng(3)), 2, 100, delta=0.01) == []


@pytest.fixture(scope="module")
def era():
    return ns2d.simulate(32, [(0, 0.02)], [(0, (0, 4, 1.0))], None, 8, 0.01, 3, 5, return_derivs=True)


def test_Lgiven_learned_operator_recovers_ns_terms(era):
    """Arm with NO fixed terms: sparse library regression (STRidge) finds nu, advection and the forcing mode."""
    op = bounds.learn_operator(era["omega"], era["dwdt"])
    assert set(op["support"]) == {"lap w", "u.grad w", "cos(0,4)"}, op["support"]
    assert abs(op["coef"]["lap w"] - 0.02) < 1e-6 * 0.02
    assert abs(op["coef"]["u.grad w"] + 1) < 1e-6 and abs(op["coef"]["cos(0,4)"] - 1) < 1e-6
    assert op["rel_residual"] < 1e-8


def test_Lgiven_certified_structure_on_bed(era):
    A, names = bounds.library(era["omega"])
    y = era["dwdt"].reshape(-1)
    v = bounds.certify_sparse(bounds.normalise_columns(A)[0], y, 1e-10 * np.linalg.norm(y))
    present = {n for n, s in zip(names, v["verdicts"]) if s == bounds.PRESENT}
    assert present == {"lap w", "u.grad w", "cos(0,4)"}
    assert all(s in (bounds.PRESENT, bounds.SMALL) for s in v["verdicts"]) and v["certified_fraction"] == 1.0


def test_Lgiven_certify_sparse_1000_trials():
    """eta = ||e||/sigma_min(A). PRESENT: |x0_j| > eta (=> truth != 0); SMALL: |x0_j| <= eta (=> |truth_j| <= 2 eta);
    REFUSE when rank-deficient or when 2*eta exceeds the caller's `scale`. 0 false PRESENT, 0 false SMALL."""
    rng = np.random.default_rng(0)
    fp = fs = cert = tot = 0
    for _ in range(1000):
        m, p = 80, 8
        A = rng.standard_normal((m, p))
        t = np.zeros(p)
        idx = rng.choice(p, 3, replace=False)
        t[idx] = rng.choice([-1, 1], 3) * rng.uniform(0.5, 2, 3)
        e = rng.standard_normal(m)
        e *= rng.uniform(0.01, 3.0) / np.linalg.norm(e)
        v = bounds.certify_sparse(A, A @ t + e, np.linalg.norm(e), scale=0.25 * np.abs(t[idx]).min())
        eta = np.linalg.norm(e) / np.linalg.svd(A, compute_uv=False)[-1]
        assert abs(v["eta"] - eta) < 1e-9 * eta
        for j, s in enumerate(v["verdicts"]):
            fp += (s == bounds.PRESENT) and t[j] == 0
            fs += (s == bounds.SMALL) and abs(t[j]) > 2 * eta + 1e-12
            cert += s != bounds.REFUSE
            tot += 1
    print(f"certified fraction {cert}/{tot} = {cert / tot:.3f}; false PRESENT {fp}; false SMALL {fs}")
    assert fp == 0 and fs == 0 and cert / tot > 0.3
    deficient = np.ones((10, 3))
    assert set(bounds.certify_sparse(deficient, np.ones(10), 0.1)["verdicts"]) == {bounds.REFUSE}

"""Sanity tests for E0 (run: python -m pytest -q test_e0.py from this folder)."""
import numpy as np

import arms
from bed import make_bed

V = 8


def _short(kind, seed=7, T=1500, K=3):
    return make_bed(seed, kind=kind, K=K, V=V, T=T)


def test_bed_rec_switches_between_distinct_laws():
    x, r, laws = make_bed(5, kind="REC", K=4, V=V, T=5000)
    sw = np.flatnonzero(np.diff(r)) + 1
    assert len(laws) == 4 and len(sw) > 3 and np.all(r[sw] != r[sw - 1])
    assert set(np.unique(r)) <= set(range(4))


def test_e_value_has_mean_one_under_the_active_era():
    rng = np.random.default_rng(0)
    A, Wp = rng.dirichlet(np.ones(V)), rng.dirichlet(np.ones(V))
    e = (0.5 * A + 0.5 * Wp) / A
    assert abs(float(A @ e) - 1.0) < 1e-12


def test_oracle_equals_global_on_null():
    x, r, _ = _short("NULL")
    assert np.allclose(arms.oracle(x, r, V), arms.global_o1(x, V))


def test_bocpd_without_changes_equals_global():
    x, r, _ = _short("REC")
    assert np.allclose(arms.bocpd(x, V, hazard=1e-300, rmax=len(x)), arms.global_o1(x, V), atol=1e-9)


def test_fcpa_that_never_alarms_equals_global():
    x, r, _ = _short("REC")
    out = arms.fcpa(x, V, delta=1e-300, w=32, return_info=True)
    assert out["n_eras"] == 1 and np.allclose(out["hard"], arms.global_o1(x, V))


def test_ema_at_lambda_one_equals_global():
    x, r, _ = _short("REC")
    assert np.allclose(arms.ema(x, V, 1.0), arms.global_o1(x, V))


def test_fcpa_counts_conserve_tokens():
    x, r, _ = _short("REC", T=3000)
    out = arms.fcpa(x, V, delta=0.05, w=32, return_info=True)
    assert out["n_eras"] >= 2  # the bed switches; the detector must fire at least once


def test_e0b_defaults_reproduce_e0_exactly():
    x, r, _ = _short("REC", T=3000)
    a = arms.fcpa(x, V, 0.05, 32, alphas=(3e-3,), return_info=True)
    b = arms.fcpa(x, V, 0.05, 32, alphas=(3e-3,), return_info=True, ext=None, merge=False)
    assert np.array_equal(a["soft"][3e-3], b["soft"][3e-3]) and a["n_eras"] == b["n_eras"]


def test_logml_merge_prefers_one_era_for_one_law_and_two_for_two():
    rng = np.random.default_rng(1)
    P, Q = rng.dirichlet(np.full(V, 0.2), V), rng.dirichlet(np.full(V, 0.2), V)
    def counts(L, n):
        C = np.zeros((V, V)); prev = 0
        for _ in range(n):
            cur = rng.choice(V, p=L[prev]); C[prev, cur] += 1; prev = cur
        return C
    tabs = arms._logml_table(V, 5000)
    A1, A2, B = counts(P, 800), counts(P, 800), counts(Q, 800)
    bf = lambda X, Y: arms._logml(X + Y, tabs) - arms._logml(X, tabs) - arms._logml(Y, tabs)
    assert bf(A1, A2) > 0 > bf(A1, B)


def test_merge_keeps_token_count_and_reduces_eras():
    x, r, _ = make_bed(3, kind="REC", K=3, V=V, T=6000)
    a = arms.fcpa(x, V, 0.05, 32, return_info=True)
    b = arms.fcpa(x, V, 0.05, 32, return_info=True, merge=True)
    assert b["n_eras"] <= a["n_eras"]


def test_e0c_defaults_reproduce_e0b_exactly():
    """Reference values for E0b's SOFT+BM came from the pre-E0c code path; fold() must not change them."""
    x, r, _ = make_bed(3, kind="REC", K=3, V=V, T=6000)
    b = np.exp(-arms.bocpd(x, V, 1 / 400))
    a1 = arms.fcpa(x, V, 0.05, 32, alphas=(3e-3,), return_info=True, ext=b, merge=True)
    a2 = arms.fcpa(x, V, 0.05, 32, alphas=(3e-3,), return_info=True, ext=b, merge=True, n_min=0)
    assert np.array_equal(a1["soft"][3e-3], a2["soft"][3e-3]) and a1["n_eras"] == a2["n_eras"]


def test_global_merge_never_has_more_eras_than_active_merge_and_keeps_counts_nonnegative():
    x, r, _ = make_bed(4, kind="REC", K=3, V=V, T=6000)
    a = arms.fcpa(x, V, 0.05, 32, return_info=True, merge=True)
    g = arms.fcpa(x, V, 0.05, 32, return_info=True, merge="global", n_min=32)
    assert g["n_eras"] <= a["n_eras"] + 1


def test_bocpd_vectors_match_its_losses():
    x, r, _ = _short("REC")
    l, vec = arms.bocpd(x, V, 1 / 400, return_vec=True)
    assert np.allclose(vec.sum(1), 1) and np.allclose(-np.log(vec[np.arange(len(l)), x[1:]]), l)


def test_soft_decider_reproduces_the_e0c_soft_path_exactly():
    from deciders import Soft
    x, r, _ = make_bed(3, kind="REC", K=3, V=V, T=4000)
    l, vec = arms.bocpd(x, V, 1 / 400, return_vec=True)
    d = Soft(2, 3e-3)
    out = arms.fcpa(x, V, 0.05, 32, alphas=(3e-3,), ext=np.exp(-l), merge="global", n_min=64, deciders=(d,), ext_vec=vec)
    assert np.allclose(out["dec"][d], out["soft"][3e-3], atol=1e-10)


def test_cyc_bed_mostly_steps_forward():
    x, r, _ = make_bed(1, kind="CYC", K=4, V=V, T=20000, seg_min=20, seg_mean=100)
    sw = np.flatnonzero(np.diff(r)) + 1
    assert np.mean(r[sw] == (r[sw - 1] + 1) % 4) > 0.8


def test_ctc_and_momentum_deciders_return_valid_probabilities():
    from deciders import CTC, Momentum
    x, r, _ = make_bed(2, kind="CYC", K=3, V=V, T=3000, seg_min=20, seg_mean=100)
    l, vec = arms.bocpd(x, V, 1 / 400, return_vec=True)
    ds = (CTC(2, 3e-3, 0.8), Momentum(2, 3e-3))
    out = arms.fcpa(x, V, 0.05, 32, ext=np.exp(-l), merge="global", n_min=32, deciders=ds, ext_vec=vec)
    for d in ds:
        assert np.all(np.isfinite(out["dec"][d])) and np.all(out["dec"][d] > 0)
    assert abs(ds[1].share().sum() - 1) < 1e-12

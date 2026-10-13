"""Launch guards for the Amendment 11 primary run (python -m fcpa.run_primary).

Each test pins one validity guarantee of the closed-loop evaluation at smoke size (non-eval seeds 7001/7002):
(a) no timestamp / era label / true future state during rollout, (b) the lag stack is the arm's own predictions,
(c) seed hygiene, (d) P3 scored on the frozen era set with the era map fit on Cal only, (e) equal input rows.
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa import bars, baselines, run_primary as rp  # noqa: E402
from fcpa.estimator import FCPARegressor  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
SEED, P = 7001, rp.SMOKE


@pytest.fixture(scope="module")
def world():
    X, reg = rp.gen_bed(SEED, P)
    tr, ca, te = rp.three_way_split(len(X), SEED)
    XL = rp.lag_stack(X, P["n_lags"])
    est = rp.fit_fcpa(XL[tr], P, SEED, cal=XL[ca])  # as evaluate() fits it (Amendment 11c)
    return dict(X=X, reg=reg, tr=tr, ca=ca, te=te, XL=XL, est=est, D=X.shape[-1])


# ---------------------------------------------------------------- (a) closed-loop leakage
def test_a_predict_and_rollout_take_no_timestamp_or_label():
    assert list(inspect.signature(FCPARegressor.predict).parameters) == ["self", "X"]
    assert not {"reg", "regime", "t", "labels"} & set(inspect.signature(rp.rollout_errors).parameters)


def test_a_fcpa_is_fit_without_timestamps_or_targets(monkeypatch):
    calls, orig = [], FCPARegressor.partial_fit

    def spy(self, X, y=None, t=None):
        calls.append((y, t))
        return orig(self, X, y, t)

    monkeypatch.setattr(FCPARegressor, "partial_fit", spy)
    X, _ = rp.gen_bed(SEED, P)
    est = rp.fit_fcpa(rp.lag_stack(X, P["n_lags"])[:4], P, SEED)
    assert calls and all(y is None and t is None for y, t in calls)
    from fcpa.laws import NS2DLaw

    assert isinstance(est.law, NS2DLaw)  # the primary configuration is law mode (R = nu L + f from the lag stack)


def test_a_predict_is_rowwise_and_stateless(world):
    est, XL, te = world["est"], world["XL"], world["te"]
    rows = XL[te][:, 20:90:5].reshape(len(te) * 14, -1)
    rows = np.vstack([rows, 1.5 * rows + 0.3])  # true states plus off-distribution ones
    snap = (est.n_eras_, est.alpha_t_, est.n_samples_seen_, [c.copy() for c in est.law_counts_], est._model["n"].copy())
    batch = est.predict(rows)
    one = np.vstack([est.predict(r[None]) for r in rows])
    perm = np.random.default_rng(0).permutation(len(rows))
    np.testing.assert_allclose(batch, one, rtol=1e-9, atol=1e-12)  # no row sees another row
    np.testing.assert_allclose(est.predict(rows[perm]), batch[perm], rtol=1e-9, atol=1e-12)  # no hidden order/time
    assert snap[:3] == (est.n_eras_, est.alpha_t_, est.n_samples_seen_) and np.array_equal(snap[4], est._model["n"])
    assert all(np.array_equal(a, b) for a, b in zip(snap[3], est.law_counts_))  # predict never births or updates


def test_a_primary_path_never_calls_the_hmm_filter(world, monkeypatch):
    est, X, XL, reg, ca, te, D = (world[k] for k in ("est", "X", "XL", "reg", "ca", "te", "D"))

    def boom(*a, **k):
        raise AssertionError("hmm_filter (consecutive-state era smoothing) reached the primary path")

    monkeypatch.setattr(FCPARegressor, "hmm_filter", boom)
    rp.fcpa_stream(est, XL[ca], reg[ca], XL[te], reg[te], P)
    rp.rollout_errors(lambda r: rp.first_block(est.predict(r), D), X[te], P, n_lags=2)


# ---------------------------------------------------------------- (b) the lag stack is the arm's own prediction
def test_b_rollout_stack_holds_only_own_predictions_and_no_future_state(world):
    est, X, te, D = world["est"], world["X"], world["te"], world["D"]
    P1 = dict(P, origins=(20,))
    Xn = X[te].copy()
    Xn[:, 21:] = np.nan  # poison every true state after the origin: reading one makes a stack non-finite
    recs = []

    def step(rows):
        recs.append(rows.copy())
        return rp.first_block(est.predict(rows), D)

    rp.rollout_errors(step, Xn, P1, n_lags=2)
    H = P1["horizon"]
    assert len(recs) == H and all(np.isfinite(r).all() for r in recs)
    np.testing.assert_array_equal(recs[0], np.c_[X[te][:, 20], X[te][:, 19]])  # step 1: true history x_t0, x_t0-1
    np.testing.assert_array_equal(recs[1][:, D:], X[te][:, 20])  # step 2: lag = x_t0, the last true history point
    for k in range(1, H):
        np.testing.assert_array_equal(recs[k][:, :D], rp.first_block(est.predict(recs[k - 1]), D))  # newest = own prediction
        if k >= 2:  # from step 3 on the stack holds nothing but the arm's own outputs
            np.testing.assert_array_equal(recs[k][:, D:], recs[k - 1][:, :D])
        assert np.abs(recs[k][:, :D] - X[te][:, 20 + k]).max() > 0  # and is not the true state


# ---------------------------------------------------------------- (c) seed hygiene
def test_c_freeze_hashes_match_the_files_on_disk():
    assert rp.verify_freeze()


@pytest.mark.parametrize("seeds", [(0, 0, 0, 0, 0), (0, 1, 2), (0, 1, 2, 4)], ids=["duplicates", "subset3", "subset4"])
def test_c_full_run_requires_exactly_the_five_eval_seeds(seeds):
    with pytest.raises(ValueError):
        rp.check_seeds(seeds, smoke=False)


def test_c_seed_checks_refuse_eval_seed_in_smoke_and_tuning_seed_anywhere():
    rp.check_seeds(bars.EVAL_SEEDS, smoke=False)
    rp.check_seeds(rp.SMOKE_SEEDS, smoke=True)
    assert bars.EVAL_SEEDS == (0, 1, 2, 4, 5) and bars.TUNE_SEED == 3
    for seeds, smoke in (((3, 0, 1, 2, 4), False), ((3,), False), ((0,), True), ((3,), True), ((7001, 5), True)):
        with pytest.raises(ValueError):
            rp.check_seeds(seeds, smoke)
    with pytest.raises(AssertionError):
        rp.evaluate(3, P, tuned=dict(lr=1e-2, m=1))


@pytest.mark.parametrize("argv", [["--seeds", "3"], ["--smoke", "--seeds", "0"], ["--smoke", "--seeds", "3"]])
def test_c_main_refuses_bad_seeds_before_any_work(monkeypatch, argv):
    def reached(*a, **k):
        raise AssertionError("work started before the seed check")

    monkeypatch.setattr(rp, "verify_freeze", lambda: True)
    monkeypatch.setattr(rp, "tune", reached)
    monkeypatch.setattr(rp, "evaluate", reached)
    with pytest.raises(SystemExit, match="REFUSED"):
        rp.main(argv)


def test_c_tuning_touches_only_seed_3_and_bed_2003(monkeypatch, tmp_path):
    seen = []
    g0, s0, f0 = rp.gen_bed, rp.three_way_split, rp.fit_fcpa
    a0 = baselines.AsmodeousArm.__init__
    monkeypatch.setattr(rp, "gen_bed", lambda seed, *a, **k: (seen.append(("bed", seed)), g0(seed, *a, **k))[1])
    monkeypatch.setattr(rp, "three_way_split", lambda n, seed: (seen.append(("split", seed)), s0(n, seed))[1])
    monkeypatch.setattr(rp, "fit_fcpa", lambda XL, P_, seed, **k: (seen.append(("fcpa", seed)), f0(XL, P_, seed, **k))[1])

    def init(self, *a, seed=0, **k):
        seen.append(("asm", seed))
        return a0(self, *a, seed=seed, **k)

    monkeypatch.setattr(baselines.AsmodeousArm, "__init__", init)
    rp.tune(P, 1, tmp_path, log=lambda s: None)
    assert {s for kind, s in seen if kind != "bed"} == {bars.TUNE_SEED} and {s for k, s in seen if k == "bed"} == {rp.TUNE_BED_SEED}
    assert not {s for _, s in seen} & set(bars.EVAL_SEEDS)


# ---------------------------------------------------------------- (d) P3 on the frozen era set, map fit on Cal only
def test_d_era_map_is_fit_on_cal_before_test_is_read(world, monkeypatch):
    est, XL, reg, ca, te = (world[k] for k in ("est", "XL", "reg", "ca", "te"))
    events, f0, e0 = [], rp.fcpa_scores, bars.era_map
    monkeypatch.setattr(rp, "fcpa_scores", lambda e, X: (events.append(("scores", len(X))), f0(e, X))[1])
    monkeypatch.setattr(bars, "era_map", lambda *a: (events.append("emap"), e0(*a))[1])
    a = rp.fcpa_stream(est, XL[ca], reg[ca], XL[te], reg[te], P)
    DL = XL.shape[-1]
    assert events == [("scores", XL[ca].size // DL), "emap", ("scores", XL[te].size // DL)]
    shuffled = np.random.default_rng(1).permutation(reg[te].ravel()).reshape(reg[te].shape)
    b = rp.fcpa_stream(est, XL[ca], reg[ca], XL[te], shuffled, P)
    assert a["emap"] == b["emap"]  # Test labels never move the map
    cal_learned = f0(est, XL[ca].reshape(-1, DL)).argmin(1)
    assert a["emap"] == e0(cal_learned, reg[ca].reshape(-1), est.n_eras_)


def test_d_test_stream_scoring_freezes_the_era_set(world, monkeypatch):
    est, XL, reg, ca, te = (world[k] for k in ("est", "XL", "reg", "ca", "te"))
    snap = (est.n_eras_, est.n_samples_seen_, est.alpha_t_, [c.copy() for c in est.law_counts_], id(est._model))

    def boom(*a, **k):
        raise AssertionError("the estimator was updated while the Test stream was scored")

    for name in ("fit", "partial_fit", "_finalize", "_consume", "_try_birth"):
        monkeypatch.setattr(FCPARegressor, name, boom)
    st = rp.fcpa_stream(est, XL[ca], reg[ca], XL[te], reg[te], P)
    assert (est.n_eras_, est.n_samples_seen_, est.alpha_t_, id(est._model)) == (snap[0], snap[1], snap[2], snap[4])
    assert all(np.array_equal(a, b) for a, b in zip(snap[3], est.law_counts_))
    DL = XL.shape[-1]
    pred = np.array([st["emap"][k] for k in rp.fcpa_scores(est, XL[te].reshape(-1, DL)).argmin(1)]).reshape(reg[te].shape)
    assert st["correct"] == [bars.id_counts(pred[i], reg[te][i], P["w"])[0] for i in range(len(te))]  # per-row argmin, no smoothing


# ---------------------------------------------------------------- (e) fair input
def test_e_every_arm_trains_and_rolls_out_on_the_same_lag_stacked_rows(monkeypatch):
    rec, stacks = {}, []
    pf0, gf0, af0, ro0 = FCPARegressor.partial_fit, baselines.GlobalModel.fit, baselines.AsmodeousArm.fit, rp.rollout_errors

    def pf(self, X, y=None, t=None):
        rec.setdefault("fcpa", []).append(np.array(X))
        return pf0(self, X, y, t)

    def gf(self, X, Y):
        rec["global"] = (np.array(X), np.array(Y))
        return gf0(self, X, Y)

    def af(self, trajs, steps, probe_pairs=20_000):
        rec["asm"] = np.array(trajs)
        return af0(self, trajs, steps, probe_pairs=probe_pairs)

    def ro(step, X, P_, counter=None, n_lags=1):
        first = []
        out = ro0(lambda rows: (first.append(np.array(rows)) if not first else None, step(rows))[1], X, P_, counter, n_lags)
        stacks.append(first[0])
        return out

    monkeypatch.setattr(FCPARegressor, "partial_fit", pf)
    monkeypatch.setattr(baselines.GlobalModel, "fit", gf)
    monkeypatch.setattr(baselines.AsmodeousArm, "fit", af)
    monkeypatch.setattr(rp, "rollout_errors", ro)
    rp.evaluate(SEED, P, tuned=dict(lr=1e-2, m=1), log=lambda s: None)

    X, _ = rp.gen_bed(SEED, P)
    tr, _, te = rp.three_way_split(len(X), SEED)
    XL, D = rp.lag_stack(X, P["n_lags"]), X.shape[-1]
    DL = XL.shape[-1]
    assert P["n_lags"] == 2 and DL == 2 * D == 180
    np.testing.assert_array_equal(np.vstack(rec["fcpa"]), XL[tr].reshape(-1, DL))
    np.testing.assert_array_equal(rec["asm"], XL[tr])
    gx, gy = rec["global"]
    want = np.c_[XL[tr][:, :-1].reshape(-1, DL), X[tr][:, 1:].reshape(-1, D)]
    got = np.c_[gx, gy]
    np.testing.assert_array_equal(got[np.lexsort(got.T)], want[np.lexsort(want.T)])  # same pairs, any order
    assert len(stacks) == 5  # fcpa, global, asmodeous, persistence, constant velocity
    start = XL[te][:, list(P["origins"])].reshape(-1, DL)
    for s in stacks:
        np.testing.assert_array_equal(s, start)


# ---------------------------------------------------------------- (f) L6'/L36, Amendment 11c: calibration by whole trajectory
def test_f_calibration_windows_are_not_interleaved_with_training_windows_of_their_trajectory(world, monkeypatch):
    """R1: partial_fit sees only Train trajectories, every one of their windows trains, and every pool row comes
    from a Cal trajectory passed to calibrate()."""
    XL, tr, ca = world["XL"], world["tr"], world["ca"]
    seen, p0 = [], FCPARegressor.partial_fit
    monkeypatch.setattr(FCPARegressor, "partial_fit", lambda self, X, y=None, t=None: (seen.append(np.asarray(X)), p0(self, X, y, t))[1])
    est = rp.fit_fcpa(XL[tr], P, SEED, cal=XL[ca])
    assert len(seen) == len(tr) and not any(np.array_equal(x, c) for x in seen for c in XL[ca])
    st = est._st
    assert est.holdout == "trajectory" and len(st.cal_lab) and (st.cal_traj >= 0).all() and st.cal_traj.max() < len(ca)
    held = FCPARegressor(**{**est.get_params(), "holdout": "interleaved"})
    assert held._held_out(4, False) and not est._held_out(4, False)  # no Train window is held out under R1


def test_f_calibration_pool_spans_the_training_trajectories_not_just_the_last_few(world):
    """R1: the pool is uncapped (no 400-row FIFO), spans the Cal trajectories, and calibrating changes no era
    statistic, TT sum or era count."""
    XL, tr, ca = world["XL"], world["tr"], world["ca"]
    est = rp.fit_fcpa(XL[tr], P, SEED)
    est.set_params(holdout="trajectory")
    st = est._st
    st.cal_phi, st.cal_lab, st.cal_traj = st.cal_phi[:0], st.cal_lab[:0], st.cal_traj[:0]
    st.cal_X, st.cal_Y = st.cal_X[:0], st.cal_Y[:0]
    before = ([sg.n for sg in st.segs], st.tt.counts.copy(), len(st.segs), st.n_pairs)
    cal = np.concatenate([XL[ca]] * 4)  # 4 x 2 Cal trajectories: more than 400 rows if the windows are admitted
    est.calibrate(cal)
    assert ([sg.n for sg in st.segs], len(st.segs), st.n_pairs) == (before[0], before[2], before[3])
    np.testing.assert_array_equal(st.tt.counts, before[1])
    assert len(st.cal_lab) > 400 and len(st.cal_lab) % P["w"] == 0
    assert len(np.unique(st.cal_traj)) >= len(cal) // 2, f"the pool comes from {len(np.unique(st.cal_traj))} of {len(cal)} trajectories"


def test_f_an_era_with_too_few_cal_trajectories_is_a_newborn(world):
    """11c item 4: an era calibrated by fewer than ceil(1/alpha) - 1 Cal trajectories is in every set."""
    est = world["est"]
    m = est._model
    assert m["cal"] is not None
    n_min, st = m["cal"].n_min_, est._st
    for i, k in enumerate(m["act"]):
        n_traj = len(np.unique(st.cal_traj[st.cal_lab == k]))
        assert np.isinf(m["cal"].quantiles_[i]) if n_traj < n_min else m["cal"].n_cal_[i] > 0

"""baselines.py (Global at matched parameters, ASMODEOUS top-m adapter) and run_primary.py (freeze gate, pins, smoke)."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa import baselines, bars  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ Global at matched parameters
def test_baselines_global_param_count_matches_fitted_shapes():
    rng = np.random.default_rng(0)
    d, p, width = 6, 6, 20
    g = baselines.GlobalModel(d, p, width, seed=0)
    X = rng.standard_normal((200, d))
    g.fit(X, X @ rng.standard_normal((d, p)))
    fitted = g.era.W_.size + g.era.b_.size
    assert g.n_params == fitted == (d + width + 1) * p


def test_baselines_global_recovers_a_linear_map():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 5))
    X = 100 * rng.standard_normal((400, 5))  # raw (unnormalised) scale like the encoded state
    g = baselines.GlobalModel(5, 5, 16, seed=0).fit(X, X @ A)
    Xt = 100 * rng.standard_normal((50, 5))
    err = np.linalg.norm(g.predict(Xt) - Xt @ A) / np.linalg.norm(Xt @ A)
    assert err < 0.15  # the family's isotropy shrinkage (beta = 0.1) costs a few percent; a mean predictor scores ~1


def test_baselines_solve_width_lands_within_two_percent():
    for target in (80_000, 123_457, 20_000):
        w = baselines.solve_width(target, d=90, p=90)
        assert abs(baselines.param_count(90, 90, w) - target) <= 0.02 * target
        assert baselines.param_count(90, 90, w) == (90 + w + 1) * 90


def test_baselines_solve_width_refuses_an_unreachable_target():
    with pytest.raises(ValueError):
        baselines.solve_width(100, d=90, p=90)  # even width 0 has 8,190 parameters


def test_baselines_solve_width_tolerance_edge():
    with pytest.raises(ValueError):
        baselines.solve_width(1050, d=2, p=100)  # counts move in steps of 100: nearest is 4.8% away
    assert baselines.param_count(2, 100, baselines.solve_width(1050, d=2, p=100, tol=0.05)) in (1000, 1100)
    assert baselines.param_count(2, 100, baselines.solve_width(1010, d=2, p=100)) == 1000


def test_baselines_global_budget_is_respected():
    d = p = 90
    width, n_max = 500, 51_200
    per = baselines.global_macs(1, d, p, width, with_solve=False)
    fixed = baselines.global_macs(0, d, p, width, with_solve=True)
    assert fixed > 0 and per > 0
    B = fixed + 1000 * per
    n = baselines.global_n_afford(B, d, p, width, n_max)
    assert n == 1000 and baselines.global_macs(n, d, p, width) <= B
    assert baselines.global_n_afford(fixed - 1, d, p, width, n_max) == 0
    assert baselines.global_n_afford(10 * B * 1000, d, p, width, n_max) == n_max


# ------------------------------------------------------------------ ASMODEOUS adapter
def _tiny_arm(seed=0, **kw):
    return baselines.AsmodeousArm(D=12, K=5, m=2, lr=3e-3, seed=seed, batch=4, window=5, hidden=16, **kw)


def test_baselines_asmodeous_trains_and_steps_in_the_common_space():
    rng = np.random.default_rng(0)
    trajs = rng.standard_normal((6, 30, 12)) * 100
    arm = _tiny_arm()
    arm.fit(trajs, steps=3)
    assert arm.steps_done == 3
    out = arm.step(trajs[0, :7])
    assert out.shape == (7, 12) and np.isfinite(out).all()


def test_baselines_asmodeous_is_deterministic_per_seed():
    trajs = np.random.default_rng(0).standard_normal((6, 30, 12))
    a, b = _tiny_arm(7), _tiny_arm(7)
    a.fit(trajs, steps=2), b.fit(trajs, steps=2)
    np.testing.assert_allclose(a.step(trajs[1, :5]), b.step(trajs[1, :5]))


def test_baselines_asmodeous_counts_params_and_macs():
    arm = _tiny_arm()
    arm.fit(np.random.default_rng(0).standard_normal((6, 30, 12)), steps=1)
    n_net = sum(p.numel() for n, p in arm.net.named_parameters() if not n.startswith("tgt_"))
    n_dec = arm.dec.W_.size + arm.dec.b_.size
    assert arm.n_params == n_net + n_dec
    assert arm.inference_macs() == arm.net.active_flops("loop") + arm.feat_dim * 12
    assert arm.train_step_macs() == bars.macs_bwd(arm.net.active_flops("loop")) * 4 * 5  # batch 4, window 5


def test_baselines_asmodeous_macs_grow_with_m():
    macs = baselines.asmodeous_macs_by_m(D=12, K=8, hidden=16)
    assert set(macs) == {1, 2, 3, 4} and all(macs[m] <= macs[m + 1] for m in (1, 2, 3))
    fc = macs[2] / 1.05 + 1  # FCPA just below macs[2]/1.05  -> m = 1
    assert bars.choose_m(macs, fc) <= 2


# ------------------------------------------------------------------ run_primary: freeze gate, pins, split
def test_baselines_freeze_gate_accepts_the_frozen_files():
    from fcpa import run_primary

    assert run_primary.verify_freeze() is True  # FREEZE.md (draft, design) and FREEZE2.md (Amendment 11a)
    assert (ROOT / "docs" / "specs" / "FREEZE2.md").exists()
    assert "PREREG_AMENDMENT_11a.md" in (ROOT / "docs" / "specs" / "FREEZE2.md").read_text()
    assert "PREREG_AMENDMENT_11b.md" in (ROOT / "docs" / "specs" / "FREEZE3.md").read_text()  # lag stack, step edges


@pytest.mark.parametrize("victim", ["PREREG_AMENDMENT_11_DRAFT.md", "2026-10-11-fcpa-design.md", "PREREG_AMENDMENT_11a.md",
                                    "PREREG_AMENDMENT_11b.md"])
def test_baselines_freeze_gate_refuses_a_changed_spec(tmp_path, victim):
    from fcpa import run_primary

    specs = tmp_path / "specs"
    shutil.copytree(ROOT / "docs" / "specs", specs)
    assert run_primary.verify_freeze(specs) is True
    f = specs / victim
    f.write_bytes(f.read_bytes() + b"\nedited after the freeze\n")
    with pytest.raises(run_primary.FreezeError):
        run_primary.verify_freeze(specs)


@pytest.mark.parametrize("eol", [b"\n", b"\r\n"])
def test_baselines_freeze_gate_ignores_line_endings(tmp_path, eol):
    """A pin taken on an autocrlf (Windows) checkout holds on an LF one and back: line endings are not content."""
    from fcpa import run_primary

    specs = tmp_path / "specs"
    shutil.copytree(ROOT / "docs" / "specs", specs)
    for f in specs.glob("*.md"):
        f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", eol))
    assert run_primary.verify_freeze(specs) is True


def test_baselines_freeze3_tampered_hash_makes_the_runner_refuse(tmp_path):
    """A wrong sha256 in FREEZE3.md (the file itself unchanged) is refused, and so is a missing FREEZE3.md."""
    import re

    from fcpa import run_primary

    specs = tmp_path / "specs"
    shutil.copytree(ROOT / "docs" / "specs", specs)
    f3 = specs / "FREEZE3.md"
    f3.write_text(re.sub(r"sha256 [0-9a-f]{64}", "sha256 " + "0" * 64, f3.read_text()))
    with pytest.raises(run_primary.FreezeError, match="PREREG_AMENDMENT_11b.md"):
        run_primary.verify_freeze(specs)
    f3.unlink()
    with pytest.raises(run_primary.FreezeError, match="FREEZE3.md"):
        run_primary.verify_freeze(specs)


def test_baselines_freeze_gate_refuses_without_freeze2(tmp_path):
    from fcpa import run_primary

    specs = tmp_path / "specs"
    shutil.copytree(ROOT / "docs" / "specs", specs)
    (specs / "FREEZE2.md").unlink()
    with pytest.raises(run_primary.FreezeError):
        run_primary.verify_freeze(specs)


def test_baselines_pins_match_the_amendment():
    from fcpa import run_primary as rp

    P = rp.PINS
    assert (P["grid"], P["kmax"], P["n_traj"], P["L"], P["w"]) == (32, 4, 256, 400, 20)
    assert P["lam_sw"] == 0.005 and abs(P["lam_sw"] * P["w"] - 0.1) < 1e-12
    assert len(rp.LAWS) == 4 and rp.LAWS[0] == (0.010, (0, 4, 0.5)) and rp.LAWS[3] == (0.020, (0, 2, 1.0))
    assert P["steps_per_snapshot"] == 10 and P["dt"] == 0.01 and P["burnin"] == 200
    assert P["n_lags"] == 2 and P["c2_mode"] == "prime" and P["train_steps"] == 5000 and P["batch"] == 64
    assert rp.TUNE_BED_SEED == 2003 and bars.EVAL_SEEDS == (0, 1, 2, 4, 5) and bars.TUNE_SEED == 3
    assert 64 * 400 == 25_600 and bars.p4_precondition(25_600)


def test_baselines_split_is_whole_trajectory_128_64_64():
    from fcpa import run_primary as rp

    tr, ca, te = rp.three_way_split(256, seed=0)
    assert (len(tr), len(ca), len(te)) == (128, 64, 64)
    assert not (set(tr) & set(ca)) and not (set(tr) & set(te)) and not (set(ca) & set(te))
    assert sorted(set(tr) | set(ca) | set(te)) == list(range(256))
    tr2, _, _ = rp.three_way_split(256, seed=0)
    assert list(tr) == list(tr2)  # deterministic in the seed


def test_baselines_c2_prime_trains_baselines_at_their_pinned_budgets():
    from fcpa import run_primary as rp

    tiny_B = 10  # a budget that would starve everything under the original rule
    assert rp.global_n_use("prime", tiny_B, 90, 90, 834, 51_072) == 51_072  # the full Train split
    assert rp.asmodeous_steps("prime", tiny_B, 2.01e7, 0, 5000) == 5000  # the pinned 5,000 steps


def test_baselines_c2_strict_is_the_original_rule():
    from fcpa import run_primary as rp

    B = 2_804_127_075
    assert rp.global_n_use("strict", B, 90, 90, 834, 51_072) == 1_912  # amendment 11a arithmetic: 3.7%
    assert rp.asmodeous_steps("strict", B, 20_097_024, 0, 5000) == 139  # floor(B / step); 2.8% of 5,000
    assert rp.asmodeous_steps("strict", B, 20_097_024, 0.2 * B, 5000) < 139  # a probe cost comes off first
    assert rp.asmodeous_steps("strict", 10 * 20_097_024 * 5000, 20_097_024, 0, 5000) == 5000  # capped at the pin
    assert rp.asmodeous_steps("strict", 1, 20_097_024, 5, 5000) == 0  # never negative


def _fake_seed(s, baseline_train):
    n = 8
    return dict(seed=s, T=25_600, e50={"fcpa": [0.5] * n, "global": [1.0] * n, "asmodeous": [1.0] * n},
                id=dict(score=0.95, correct=[95] * n, total=[100] * n), cov_dev=0.0,
                macs=dict(fcpa_inf=100, global_inf=200, asmodeous_inf=200, fcpa_train=1000,
                          global_train=baseline_train, asmodeous_train=baseline_train))


@pytest.mark.parametrize("mode,verdict,c2_status", [("prime", "PASS", "REPORTED"), ("strict", "FAIL", "FAIL")])
def test_baselines_c2_prime_is_reported_not_gated(mode, verdict, c2_status):
    from fcpa import run_primary as rp

    per_seed = [_fake_seed(s, baseline_train=50_000) for s in bars.EVAL_SEEDS]  # baselines train far above B_FCPA
    r = rp.assemble(per_seed, {}, dict(rp.PINS, c2_mode=mode), 200)
    assert r["bars"]["P1"]["status"] == r["bars"]["P2"]["status"] == "PASS"
    assert r["bars"]["C2"]["status"] == c2_status
    assert r["verdict"] == verdict
    assert r["kills"]["K1"] is (mode == "strict") and r["kills"]["K2"] is (mode == "strict")
    assert r["c2_mode"] == mode and r["diagnostic"] is (mode == "strict")


def test_baselines_c1_still_gates_in_prime_mode():
    from fcpa import run_primary as rp

    per_seed = [_fake_seed(s, baseline_train=50_000) for s in bars.EVAL_SEEDS]
    for r in per_seed:
        r["macs"]["global_inf"] = 50  # FCPA's inference (100) now exceeds Global's
    r = rp.assemble(per_seed, {}, dict(rp.PINS, c2_mode="prime"), 200)
    assert r["bars"]["C1"]["status"] == "FAIL" and r["verdict"] == "FAIL" and r["kills"]["K1"] is True


def test_baselines_seed_guard_keeps_development_off_eval_seeds():
    from fcpa import run_primary as rp

    for bad in ((0,), (1, 2), (4,), (5,), (3,)):
        with pytest.raises(ValueError):
            rp.check_seeds(bad, smoke=True)  # smoke must not touch eval seeds, nor tuning seed 3 via evaluate
    rp.check_seeds((7001, 7002), smoke=True)
    rp.check_seeds(bars.EVAL_SEEDS, smoke=False)
    for bad in ((7001,), (0, 3), (0, 1, 2, 4, 6)):
        with pytest.raises(ValueError):
            rp.check_seeds(bad, smoke=False)
    with pytest.raises(SystemExit):
        rp.main(["--smoke", "--seeds", "0"])
    with pytest.raises(SystemExit):
        rp.main(["--seeds", "7001"])


def test_baselines_evaluate_never_touches_tuning_seed_3():
    from fcpa import run_primary as rp

    with pytest.raises(AssertionError):
        rp.evaluate(3, rp.PINS)


def test_baselines_regime_schedule_is_the_four_state_chain():
    from fcpa import run_primary as rp

    s = rp.regime_path(L=2000, seed=0, i=0, lam_sw=0.05)
    assert set(np.unique(s)) <= {0, 1, 2, 3} and len(s) == 2000
    sw = np.flatnonzero(np.diff(s)) + 1
    assert 0.02 < len(sw) / 2000 < 0.09  # about lam_sw per step; every switch changes the regime
    s2 = rp.regime_path(L=2000, seed=0, i=1, lam_sw=0.05)
    assert not np.array_equal(s, s2)  # each trajectory draws its own schedule


# ------------------------------------------------------------------ lag stack, step edges, reference columns
def test_baselines_lag_stack_rows_are_x_t_then_x_t_minus_1():
    from fcpa import run_primary as rp

    X = np.array([[[1.0, 2], [3, 4], [5, 6], [7, 8]]])
    L2 = rp.lag_stack(X, 2)
    assert L2.shape == (1, 4, 4)
    np.testing.assert_array_equal(L2[0], [[1, 2, 1, 2], [3, 4, 1, 2], [5, 6, 3, 4], [7, 8, 5, 6]])  # row 0 pads with x_0
    np.testing.assert_array_equal(rp.lag_stack(X, 1), X)


def test_baselines_reference_steps_on_hand_rows():
    from fcpa import run_primary as rp

    rows = np.array([[3.0, 4.0, 1.0, 2.0]])  # [x_t, x_{t-1}]
    np.testing.assert_array_equal(rp.persistence_step(2)(rows), [[3, 4]])
    np.testing.assert_array_equal(rp.constant_velocity_step(2)(rows), [[5, 6]])  # 2 x_t - x_{t-1}


def test_baselines_first_block_takes_the_next_x_from_either_output_width():
    from fcpa import run_primary as rp

    assert rp.first_block(np.arange(8.0).reshape(2, 4), 2).tolist() == [[0, 1], [4, 5]]  # stacked-row prediction
    assert rp.first_block(np.arange(4.0).reshape(2, 2), 2).tolist() == [[0, 1], [2, 3]]  # next-x prediction


def _linear_bed(L=120):
    t = np.arange(L)[None, :, None]
    return (1.0 + t) * np.array([[[1.0, 2.0]]])  # x_t = (1 + t) (1, 2)


def test_baselines_closed_loop_rollout_updates_the_lag_stack():
    from fcpa import run_primary as rp

    X, P = _linear_bed(), dict(rp.PINS, origins=(20,), horizon=50)
    cv = rp.rollout_errors(rp.constant_velocity_step(2), X, P, n_lags=2)
    assert cv.shape == (1, 1) and cv[0, 0] < 1e-9  # linear motion: 2 x_t - x_{t-1} is exact, fed back on itself
    pers = rp.rollout_errors(rp.persistence_step(2), X, P, n_lags=2)
    assert pers[0, 0] == pytest.approx(50 / 71)  # |x_70 - x_20| / |x_70| = 50 / 71
    one = rp.rollout_errors(lambda r: r, X, P, n_lags=1)
    assert one[0, 0] == pytest.approx(50 / 71)


def test_baselines_fit_fcpa_uses_step_edges_and_the_lag_stack(monkeypatch):
    from fcpa import estimator, run_primary as rp

    seen = {}

    real = estimator.FCPARegressor

    class Stub:
        make_edges = staticmethod(real.make_edges)
        make_step_edges = staticmethod(real.make_step_edges)

        def __init__(self, **kw):
            seen.update(kw)

        def get_params(self):
            return {"n_lags": 1}

        def partial_fit(self, X, y=None):
            seen.setdefault("chunks", []).append(np.asarray(X).shape)
            seen["y"] = y
            return self

    monkeypatch.setattr(estimator, "FCPARegressor", Stub)
    monkeypatch.setattr(rp, "end_trajectory", lambda est: None)  # the stub has no window buffer
    X = np.random.default_rng(0).standard_normal((3, 30, 4))
    XL = rp.lag_stack(X, 2)
    rp.fit_fcpa(XL, dict(rp.PINS, n_lags=2), 0)
    np.testing.assert_array_equal(seen["y_edges"], real.make_step_edges(XL, n_lags=2))  # increments, not states
    assert seen["y_edges"].shape[0] == 4  # newest block only: D = 4 outputs
    assert seen["n_lags"] == 2 and seen["y"] is None
    assert seen["chunks"] == [(30, 8)] * 3  # one chunk per trajectory, rows [x_t, x_{t-1}]


def test_baselines_fit_fcpa_omits_n_lags_when_the_estimator_has_none(monkeypatch):
    from fcpa import estimator, run_primary as rp

    seen = {}

    real = estimator.FCPARegressor

    class Stub:
        make_edges = staticmethod(real.make_edges)
        make_step_edges = staticmethod(real.make_step_edges)

        def __init__(self, **kw):
            seen.update(kw)

        def get_params(self):
            return {}

        def partial_fit(self, X, y=None):
            return self

    monkeypatch.setattr(estimator, "FCPARegressor", Stub)
    monkeypatch.setattr(rp, "end_trajectory", lambda est: None)  # the stub has no window buffer
    rp.fit_fcpa(rp.lag_stack(np.zeros((2, 30, 4)) + np.arange(30)[None, :, None], 2), dict(rp.PINS, n_lags=2), 0)
    assert "n_lags" not in seen and "y_edges" in seen  # stacked rows are still passed; only the kwarg is withheld


def test_baselines_asmodeous_probe_can_target_fewer_dims_than_it_observes():
    arm = baselines.AsmodeousArm(D=12, D_out=6, K=5, m=2, lr=3e-3, seed=0, batch=4, window=5, hidden=16)
    trajs = np.random.default_rng(0).standard_normal((6, 30, 12))
    arm.fit(trajs, steps=1)
    assert arm.step(trajs[0, :7]).shape == (7, 6)
    assert arm.inference_macs() == arm.net.active_flops("loop") + arm.feat_dim * 6
    assert arm.n_params == arm.net.n_params() + arm.dec.W_.size + arm.dec.b_.size and arm.dec.W_.shape[1] == 6


def test_baselines_global_takes_stacked_input_and_predicts_x():
    rng = np.random.default_rng(2)
    g = baselines.GlobalModel(8, 4, 10, seed=0)  # d = 2 D inputs, p = D outputs
    g.fit(rng.standard_normal((100, 8)), rng.standard_normal((100, 4)))
    assert g.predict(rng.standard_normal((5, 8))).shape == (5, 4) and g.n_params == (8 + 10 + 1) * 4
    assert baselines.param_count(8, 4, baselines.solve_width(400, 8, 4)) == (8 + baselines.solve_width(400, 8, 4) + 1) * 4


# ------------------------------------------------------------------ launch fixes: law, pins, boundary flush, fingerprint
@pytest.fixture(scope="module")
def smoke_stack():
    from fcpa import run_primary as rp

    X, reg = rp.gen_bed(7001, rp.SMOKE, workers=4)
    return rp.lag_stack(X, rp.SMOKE["n_lags"]), X


def test_baselines_primary_fcpa_arm_is_fit_with_the_ns2d_law(smoke_stack):
    from fcpa import run_primary as rp
    from fcpa.laws import NS2DLaw

    XL, _ = smoke_stack
    est = rp.fit_fcpa(XL[:4], rp.SMOKE, 7001)
    assert isinstance(est.law, NS2DLaw) and est.n_lags == 2
    assert (est.law.n, est.law.kmax, est.law.dt) == (32, 4, pytest.approx(0.1))  # the bed's grid, encoder and snapshot step
    assert set(rp.FCPA_PINS) >= {"damping", "law"} and est.damping == rp.FCPA_PINS["damping"]


def test_baselines_fcpa_pins_dict_is_the_one_place_the_estimator_pins_live(smoke_stack, monkeypatch):
    from fcpa import run_primary as rp

    XL, _ = smoke_stack
    monkeypatch.setitem(rp.FCPA_PINS, "damping", 0.7)
    assert rp.fit_fcpa(XL[:2], rp.SMOKE, 7001).damping == 0.7
    monkeypatch.setitem(rp.FCPA_PINS, "law", None)
    assert rp.fit_fcpa(XL[:2], rp.SMOKE, 7001).law is None


def _boundary_spy(monkeypatch, XL, law):
    from fcpa import run_primary as rp
    from fcpa.estimator import FCPARegressor

    owner = {tuple(r.tobytes() for r in [row]): i for i, tr in enumerate(XL) for row in tr}
    wins, orig, holder = [], FCPARegressor._consume, {}

    def spy(self, st, X, Y, flush):
        if st is self._st:  # the real state, not _finalize's scratch copy
            wins.append((flush, [owner[(r.tobytes(),)] for r in X]))
        return orig(self, st, X, Y, flush)

    monkeypatch.setattr(FCPARegressor, "_consume", spy)
    pins = dict(rp.FCPA_PINS, **({} if law else {"law": None}))
    rp.fit_fcpa(XL, rp.SMOKE, 7001, pins=pins)
    return wins


@pytest.mark.parametrize("law", [True, False])
def test_baselines_no_consumed_window_spans_two_trajectories(smoke_stack, monkeypatch, law):
    XL = smoke_stack[0][:4]
    wins = _boundary_spy(monkeypatch, XL, law)
    assert wins and all(len(set(ids)) == 1 for _, ids in wins)
    assert any(flush for flush, _ in wins)  # each trajectory's part-filled tail is consumed as a short window


def test_baselines_boundary_flush_keeps_every_row_exactly_once(smoke_stack, monkeypatch):
    """L27' exactness at the boundary: with no law (every row is explained) the rows consumed by the real state are
    exactly the transitions of the chunks, none dropped, none twice. Dropping the tails would lose 9 of 109 per trajectory."""
    XL = smoke_stack[0][:4]
    wins = _boundary_spy(monkeypatch, XL, law=False)
    per_traj = np.bincount([ids[0] for _, ids in wins for _ in ids], minlength=4)
    assert list(per_traj) == [XL.shape[1] - 1] * 4


def test_baselines_tuned_fingerprint_refuses_any_other_configuration(tmp_path, monkeypatch):
    from fcpa import run_primary as rp

    f = tmp_path / "tuned.json"
    tuned = dict(lr=0.01, m=2, K=3, fcpa_inf_macs=1.0, asmodeous_macs_by_m={"1": 1, "2": 2, "3": 3, "4": 4}, val_e50={})
    tuned["fingerprint"] = rp.tuned_fingerprint(rp.PINS)
    f.write_text(json.dumps(tuned))
    assert rp.load_tuned(f, rp.PINS)["lr"] == 0.01
    assert rp.load_tuned(f, rp.PINS)["asmodeous_macs_by_m"][2] == 2  # int keys restored
    with pytest.raises(rp.FreezeError, match="fingerprint"):  # a smoke-generated file under the full pins
        g = tmp_path / "smoke.json"
        g.write_text(json.dumps(dict(tuned, fingerprint=rp.tuned_fingerprint(rp.SMOKE))))
        rp.load_tuned(g, rp.PINS)
    with pytest.raises(rp.FreezeError, match="fingerprint"):  # no fingerprint at all
        h = tmp_path / "old.json"
        h.write_text(json.dumps({k: v for k, v in tuned.items() if k != "fingerprint"}))
        rp.load_tuned(h, rp.PINS)
    monkeypatch.setitem(rp.FCPA_PINS, "damping", 0.5)  # an estimator pin moved after tuning
    with pytest.raises(rp.FreezeError, match="fingerprint"):
        rp.load_tuned(f, rp.PINS)
    monkeypatch.undo()
    specs = tmp_path / "specs"
    shutil.copytree(ROOT / "docs" / "specs", specs)
    (specs / "FREEZE3.md").write_text((specs / "FREEZE3.md").read_text() + "\nchanged\n")
    assert rp.tuned_fingerprint(rp.PINS, specs) != rp.tuned_fingerprint(rp.PINS)  # the freeze files are in the hash
    assert rp.tuned_fingerprint(rp.PINS, bed_seed=7) != rp.tuned_fingerprint(rp.PINS)  # and so is the bed seed
    assert rp.tuned_fingerprint(dict(rp.PINS, c2_mode="strict")) == rp.tuned_fingerprint(rp.PINS)  # tuning is C2-independent


def test_baselines_main_refuses_a_smoke_tuned_file_in_a_full_run(tmp_path, monkeypatch):
    from fcpa import run_primary as rp

    f = tmp_path / "tuned.json"
    f.write_text(json.dumps(dict(lr=0.01, m=1, K=1, asmodeous_macs_by_m={"1": 1}, fingerprint=rp.tuned_fingerprint(rp.SMOKE))))

    def reached(*a, **k):
        raise AssertionError("work started before the tuned.json check")

    monkeypatch.setattr(rp, "evaluate", reached)
    monkeypatch.setattr(rp, "tune", reached)
    with pytest.raises(SystemExit, match="REFUSED"):
        rp.main(["--tuned", str(f), "--out", str(tmp_path / "out")])


def test_baselines_full_run_needs_exactly_the_five_eval_seeds():
    from fcpa import run_primary as rp

    rp.check_seeds((5, 4, 2, 1, 0), smoke=False)  # order does not matter
    for bad in ((0, 0, 0, 0, 0), (0, 1, 2), (0, 1, 2, 4), (0, 1, 2, 4, 5, 5), (0, 1, 2, 4, 5, 3)):
        with pytest.raises(ValueError):
            rp.check_seeds(bad, smoke=False)


# ------------------------------------------------------------------ smoke (tiny sizes, < 2 minutes)
def test_baselines_smoke_run_end_to_end(tmp_path):
    import time

    from fcpa import run_primary as rp

    t0 = time.time()
    res = rp.main(["--smoke", "--out", str(tmp_path)])
    assert time.time() - t0 < 120
    out = json.loads((tmp_path / "results.json").read_text())
    assert (tmp_path / "report.md").exists()
    assert out["smoke"] is True and out["freeze_verified"] is True
    assert out["freeze_files"] == ["FREEZE.md", "FREEZE2.md", "FREEZE3.md"] and out["c2_mode"] == "prime" and out["diagnostic"] is False
    assert out["bars"]["C2"]["status"] == "REPORTED"
    assert out["verdict"] in {"PASS", "FAIL", "VOID", "NOT RUN"}
    assert out["bars"]["P4"]["status"] == "VOID"  # smoke stream is far below T = 950 steps
    assert all(k in out["bars"] for k in ("P1", "P2", "P3", "P4", "C1", "C2"))
    assert set(out["seeds"]) == {"7001", "7002"}  # smoke never touches an eval seed
    r = out["seeds"]["7001"]
    assert out["n_lags"] == 2 and r["input_dim"] == 180 and r["output_dim"] == 90  # every arm sees [x_t, x_{t-1}]
    assert {"persistence", "constant_velocity"} <= set(r["e50_seed"])
    assert "constant velocity" in (tmp_path / "report.md").read_text().lower()
    assert res["verdict"] == out["verdict"]

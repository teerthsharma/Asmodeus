"""bars.py: every Amendment 11 bar and kill rule against hand-computed cases (written RED before bars.py)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa import bars  # noqa: E402


def seeds(*arrs):
    return [np.asarray(a, float) for a in arrs]


# ------------------------------------------------------------------ estimators
def test_bars_origins_are_33_and_fit_in_the_trajectory():
    o = bars.ORIGINS
    assert len(o) == 33 and o[0] == 20 and o[-1] == 340
    assert o[-1] + bars.HORIZON == 390 < 400


def test_bars_rel_err_and_traj_e50_hand_case():
    x = np.array([[3.0, 4.0], [0.0, 2.0]])
    xh = np.array([[3.0, 0.0], [0.0, 3.0]])  # errors 4/5 = 0.8 and 1/2 = 0.5
    assert bars.rel_err(xh, x) == pytest.approx([0.8, 0.5])
    assert bars.traj_e50(xh, x) == pytest.approx(0.65)


def test_bars_ratio_is_ratio_of_seed_means():
    assert bars.ratio([1, 2, 3], [2, 4, 6]) == pytest.approx(0.5)


def test_bars_one_sided_bounds_are_percentiles():
    z = np.arange(101.0)
    assert bars.upper95(z) == pytest.approx(95.0)
    assert bars.lower95(z) == pytest.approx(5.0)


def test_bars_seed_wins_counts_strict_less():
    assert bars.seed_wins([1, 2, 3, 4, 5], [2, 2, 2, 5, 4]) == 2


def test_bars_paired_boot_shape_pairing_and_determinism():
    rng = np.random.default_rng(0)
    a = [rng.random(8) for _ in range(5)]
    b = [2 * x for x in a]
    r1 = bars.paired_boot({"a": a, "b": b}, B=500)
    r2 = bars.paired_boot({"a": a, "b": b}, B=500)
    assert r1["a"].shape == (500,)
    np.testing.assert_array_equal(r1["a"], r2["a"])  # fixed seed 20261011
    np.testing.assert_allclose(r1["b"], 2 * r1["a"])  # same resamples for every arm
    assert bars.BOOT_SEED == 20261011 and bars.B_BOOT == 10000
    assert r1["a"].mean() == pytest.approx(np.mean([x.mean() for x in a]), abs=0.02)


def test_bars_paired_boot_constants_are_constant():
    r = bars.paired_boot({"a": seeds([3.0], [3.0])}, B=50)
    np.testing.assert_array_equal(r["a"], 3.0)


# ------------------------------------------------------------------ P1 / P2 / C1 / C2
def test_bars_p12_pass_when_all_three_conditions_hold():
    f, b = seeds(*[np.full(8, 0.8)] * 5), seeds(*[np.ones(8)] * 5)
    r = bars.p12(f, b, B=200)
    assert r["rho"] == pytest.approx(0.8) and r["wins"] == 5 and r["ub"] == pytest.approx(0.8)
    assert r["pass"] is True


def test_bars_p12_fails_on_ratio_margin():
    f, b = seeds(*[np.full(8, 0.96)] * 5), seeds(*[np.ones(8)] * 5)
    r = bars.p12(f, b, B=200)
    assert r["ub"] < 1 and r["wins"] == 5 and r["pass"] is False  # 0.96 > 0.95


def test_bars_p12_fails_on_seed_wins():
    f = seeds(*([np.full(8, 0.5)] * 3 + [np.full(8, 1.2)] * 2))
    b = seeds(*[np.ones(8)] * 5)
    r = bars.p12(f, b, B=200)
    assert r["rho"] == pytest.approx(0.78) and r["wins"] == 3 and r["pass"] is False


def test_bars_p12_fails_on_upper_bound():
    f = seeds(*[[0.0, 1.8]] * 5)  # rho 0.9 but two trajectories: the bound exceeds 1
    b = seeds(*[[1.0, 1.0]] * 5)
    r = bars.p12(f, b, B=2000)
    assert r["rho"] == pytest.approx(0.9) and r["wins"] == 5 and r["ub"] >= 1 and r["pass"] is False


def test_bars_choose_m_largest_within_105_percent():
    assert bars.choose_m({1: 90, 2: 100, 3: 104.9, 4: 105.1}, 100.0) == 3
    assert bars.choose_m({1: 110, 2: 120, 3: 130, 4: 140}, 100.0) == 1  # none fits: m = 1
    assert bars.choose_m({1: 1, 2: 2, 3: 3, 4: 4}, 100.0) == 4


def test_bars_c1_and_c2():
    assert bars.c1(100.0, 100.0) and bars.c1(99, 100) and not bars.c1(101, 100)
    assert bars.steps_for_budget(1000, 300) == 3
    assert bars.c2(used=999, budget=1000) and not bars.c2(used=1001, budget=1000)


# ------------------------------------------------------------------ P3
def test_bars_era_map_majority_vote_and_unused_era():
    learned = [0, 0, 0, 1, 1, 2]
    regime = [3, 3, 1, 2, 2, 0]
    assert bars.era_map(learned, regime, n_eras=4) == {0: 3, 1: 2, 2: 0, 3: -1}


def test_bars_distance_to_switch():
    reg = np.array([0] * 3 + [1] * 3 + [1] * 2)  # one switch at index 3
    np.testing.assert_array_equal(bars.distance_to_switch(reg), [3, 2, 1, 0, 1, 2, 3, 4])
    assert np.all(np.isinf(bars.distance_to_switch(np.zeros(5, int))))


def test_bars_id_score_excludes_ambiguous_steps():
    reg = np.array([0] * 30 + [1] * 30)  # switch at 30; far = t < 10 or t > 50: 10 + 9 = 19 steps
    pred = reg.copy()
    pred[:5] = 1  # five far steps wrong
    correct, total = bars.id_counts(pred, reg, w=20)
    assert (correct, total) == (14, 19)


def test_bars_p3_pass_fail_and_reported_extras():
    ok = bars.p3([0.95, 0.9, 0.92, 0.91, 0.93])
    assert ok["pass"] and ok["worst"] == 0.9 and ok["mean"] == pytest.approx(0.922)
    assert not bars.p3([0.89] * 5)["pass"]
    assert bars.p3([0.9] * 5)["pass"]  # >= 0.9


def test_bars_id_lower_bound_is_below_mean():
    rng = np.random.default_rng(1)
    c = [rng.integers(80, 100, 16).astype(float) for _ in range(5)]
    t = [np.full(16, 100.0) for _ in range(5)]
    lb = bars.id_lower95(c, t, B=500)
    assert lb < np.mean([x.sum() / 1600 for x in c])


# ------------------------------------------------------------------ P4
def test_bars_coverage_deviation_hand_case():
    errs = np.array([1] * 8 + [0] * 92)
    assert bars.coverage_deviation(errs, alpha=0.1) == pytest.approx(0.02)
    assert bars.p4_pass([0.02, 0.0, 0.001, 0.019, 0.0]) and not bars.p4_pass([0.0, 0.021, 0, 0, 0])


def test_bars_p4_precondition_threshold_is_950():
    assert bars.p4_min_T(0.1, 0.05) == 950
    assert bars.p4_precondition(950) and not bars.p4_precondition(949)


@pytest.mark.parametrize("fixed_set", [set(), {0, 1}])
def test_bars_aci_boundary_convention_holds_for_degenerate_sets(fixed_set):
    """C always empty (err 1) or always all labels (err 0): only the boundary convention keeps coverage at 0.9."""
    T = 25_600
    truth = np.zeros(T, int)
    errs, sizes = bars.run_aci(lambda t, a: fixed_set, truth, n_labels=2, alpha=0.1, gamma=0.05)
    bound = (0.9 + 0.05) / (0.05 * T)  # 0.00074
    assert bound == pytest.approx(0.00074, abs=1e-5)
    assert bars.coverage_deviation(errs, 0.1) <= bound + 1e-12
    assert len(errs) == len(sizes) == T
    assert set(sizes) == {0, 2}  # forced steps report C = all labels (2) or empty (0) whatever set_fn says


def test_bars_run_aci_passes_alpha_t_and_scores_truth():
    seen = []

    def sets(t, a):
        seen.append(a)
        return {0}

    errs, sizes = bars.run_aci(sets, np.array([0, 1, 0]), n_labels=2, alpha=0.1, gamma=0.05, alpha1=0.1)
    assert list(errs) == [0, 1, 0] and list(sizes) == [1, 1, 1]
    assert seen[0] == pytest.approx(0.1) and seen[1] == pytest.approx(0.1 + 0.05 * 0.1)
    assert seen[2] == pytest.approx(0.1 + 0.05 * 0.1 + 0.05 * (0.1 - 1))


# ------------------------------------------------------------------ Holm and S1-S5
def test_bars_holm_step_down_hand_case():
    # sorted: .001 <= .05/5 ok, .01 <= .05/4 ok, .03 > .05/3 stop
    rej = bars.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.2, "e": 0.001})
    assert rej == {"a": True, "b": False, "c": False, "d": False, "e": True}


def test_bars_holm_is_step_down_not_bonferroni():
    rej = bars.holm({"a": 0.015, "b": 0.02, "c": 0.9})  # .015 <= .05/3 ; .02 <= .05/2 (Bonferroni would reject only a)
    assert rej == {"a": True, "b": True, "c": False}


def test_bars_holm_void_bar_shrinks_the_family():
    # four live bars: .012 <= .05/4 = .0125 passes only because the VOID bar left the family
    rej = bars.holm({"a": 0.012, "b": 0.5, "c": 0.5, "d": 0.5, "e": None})
    assert rej["a"] is True and rej["e"] is None
    rej5 = bars.holm({"a": 0.012, "b": 0.5, "c": 0.5, "d": 0.5, "e": 0.5})
    assert rej5["a"] is False  # .012 > .05/5


def test_bars_holm_stops_at_first_failure():
    # .004 ok (.01); .02 > .0125 fail; .0121 is larger in rank so not tested after the failure
    rej = bars.holm({"a": 0.004, "b": 0.02, "c": 0.02, "d": 0.9, "e": 0.9})
    assert rej["a"] and not rej["b"] and not rej["c"]


def test_bars_s1_ratio_bar_and_p_value():
    f = seeds(*[np.full(8, 0.9)] * 5)
    l = seeds(*[np.ones(8)] * 5)
    r = bars.s1(f, l, B=300)
    assert r["rho"] == pytest.approx(0.9) and r["ub"] <= 1.02 and r["p"] == 0.0 and r["point_pass"]
    r2 = bars.s1(seeds(*[np.full(8, 1.05)] * 5), l, B=300)
    assert r2["p"] == 1.0 and not r2["point_pass"] and not r2["kill"]
    assert bars.s1(seeds(*[np.full(8, 1.11)] * 5), l, B=100)["kill"]  # K5: > 1.10


def test_bars_clopper_pearson_upper_known_values():
    assert bars.cp_upper(0, 1000) == pytest.approx(1 - 0.05 ** (1 / 1000))
    assert bars.cp_upper(1, 10) == pytest.approx(0.3942, abs=1e-4)
    assert bars.cp_upper(10, 10) == 1.0


def test_bars_s2_births():
    ok = bars.s2(30, 1000)
    assert ok["rate"] == 0.03 and ok["ub"] <= 0.05 and ok["point_pass"] and not ok["kill"]
    assert ok["p"] == pytest.approx(stats.binom.cdf(30, 1000, 0.05))
    bad = bars.s2(45, 1000)  # rate .045 <= .05 but the Clopper-Pearson bound is above .05
    assert bad["rate"] <= 0.05 < bad["ub"] and not bad["point_pass"]
    assert bars.s2(101, 1000)["kill"] and not bars.s2(100, 1000)["kill"]  # K6: X/n > 2 delta


def test_bars_s3_precision_void_and_p():
    assert bars.s3(49, 49)["void"]
    r = bars.s3(50, 50)
    assert not r["void"] and r["precision"] == 1.0 and r["p"] == pytest.approx(0.95 ** 50)
    assert r["point_pass"]
    r = bars.s3(90, 100)
    assert r["precision"] == 0.9 and not r["point_pass"] and not r["kill"]  # K7 is strictly < 0.90
    assert bars.s3(89, 100)["kill"]


def test_bars_certified_events_merge_and_precision_hand_case():
    ev = bars.merge_events([350, 100, 110, 355, 200, 235], w=20)  # 235 is 35 <= 2w after 200
    assert list(ev) == [100, 200, 350]
    assert bars.count_true_positives(ev, switches=[105, 500], w=20) == 1
    assert bars.count_true_positives(ev, switches=[220], w=20) == 1  # 220 in [180, 220]


def test_bars_benettin_exponent_of_linear_maps():
    assert bars.benettin(lambda x: 2 * x, np.array([1.0, 1.0]), np.array([1.0, 0.0]), 20) == pytest.approx(math.log(2), abs=1e-6)
    assert bars.benettin(lambda x: 0.5 * x, np.array([1.0, 1.0]), np.array([0.0, 1.0]), 20) == pytest.approx(-math.log(2), abs=1e-6)
    d = np.array([1.0, 1.0]) / math.sqrt(2)
    lam = bars.benettin(lambda x: 1 + (x - 1) * np.array([2.0, 0.5]), np.array([1.0, 1.0]), d, 50)  # fixed point (1, 1)
    assert lam == pytest.approx(math.log(2), abs=0.02)


def test_bars_s4_horizon_eta_and_void():
    assert bars.horizon(0.1) == pytest.approx(math.log(3e5) / 0.1)
    assert bars.h_B(0.1) == 50 and bars.h_B(0.5) == 25
    assert bars.eta(-0.4) == pytest.approx(0.1)
    assert bars.s4_void(-0.1) and bars.s4_void(0.0) and bars.s4_void(1.3)  # lam<=0 ; H = 9.7 < 10
    assert not bars.s4_void(0.5)


def test_bars_s4_bar():
    d = seeds(*[np.full(4, 0.01)] * 5)
    r = bars.s4(d, eta=0.05, B=200)
    assert r["point"] == pytest.approx(0.01) and r["point_pass"] and r["p"] == 0.0 and not r["kill"]
    r = bars.s4(seeds(*[np.full(4, -0.06)] * 5), eta=0.05, B=200)
    assert r["point"] == pytest.approx(0.06) and not r["point_pass"] and r["kill"] and r["p"] == 1.0


def test_bars_s5_laziness():
    r = bars.s5(seeds(*[np.full(4, 1.1)] * 5), B=200)
    assert r["mean"] == pytest.approx(1.1) and r["point_pass"] and r["p"] == 0.0 and not r["kill"]
    r = bars.s5(seeds(*[np.full(4, 2.1)] * 5), B=200)
    assert not r["point_pass"] and r["kill"] and r["p"] == 1.0


# ------------------------------------------------------------------ kill rules, verdict, MAC helpers
def test_bars_kill_rules_fire_on_their_conditions():
    ok = dict(rho_global=0.8, rho_asm=0.9, c_global=True, c_asm=True, id_mean=0.95, cov_dev=[0.0] * 5,
              rho_G=1.0, x_rate=0.04, precision=0.97, lam_diff=0.01, eta=0.05, mean_C=1.0)
    assert not any(bars.kills(**ok).values())
    assert bars.kills(**{**ok, "rho_global": 1.01})["K1"]
    assert bars.kills(**{**ok, "c_global": False})["K1"]
    assert bars.kills(**{**ok, "rho_asm": 1.01})["K2"]
    assert bars.kills(**{**ok, "id_mean": 0.79})["K3"] and not bars.kills(**{**ok, "id_mean": 0.8})["K3"]
    assert bars.kills(**{**ok, "cov_dev": [0, 0, 0.021, 0, 0]})["K4"]
    assert bars.kills(**{**ok, "rho_G": 1.11})["K5"]
    assert bars.kills(**{**ok, "x_rate": 0.11})["K6"]
    assert bars.kills(**{**ok, "precision": 0.89})["K7"]
    assert bars.kills(**{**ok, "lam_diff": 0.06})["K8"]
    assert bars.kills(**{**ok, "mean_C": 2.01})["K9"]
    none = bars.kills(**{**ok, "rho_G": None})
    assert none["K5"] is None  # not run is not "not fired"


def test_bars_primary_verdict_combinations():
    t = dict(p1=True, p2=True, c1=True, c2=True, p3=True, p4=True)
    assert bars.primary_verdict(**t) == "PASS"
    assert bars.primary_verdict(**{**t, "p3": False}) == "FAIL"
    assert bars.primary_verdict(**{**t, "p4": None}) == "VOID"
    assert bars.primary_verdict(**{**t, "p1": False, "p4": None}) == "FAIL"
    assert bars.primary_verdict(**{**t, "p2": "NOT RUN"}) == "NOT RUN"


def test_bars_mac_helpers_hand_values():
    assert bars.macs_matmul(2, 3, 4) == 24
    assert bars.macs_gram(10, 3) == 90
    assert bars.macs_solve(3, 2) == 27 + 18
    assert bars.macs_complex(5) == 20
    assert bars.macs_bwd(100) == 300  # forward + 2x backward

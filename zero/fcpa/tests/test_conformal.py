"""L6' (Mondrian split conformal, trajectory split), L7 (ACI bound), L24 / L24' (births, starvation)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa.conformal import (  # noqa: E402
    ACI,
    SplitConformal,
    BirthTrigger,
    birth_budget,
    first_birth,
    trajectory_split,
)

MU = np.array([0.0, 2.0, 4.0])


def _draw(rng, labels, sigma=1.0):
    h = MU[labels] + sigma * rng.standard_normal(len(labels))
    return np.abs(h[:, None] - MU[None, :])  # nonconformity s(h, k) = |h - mu_k|


# ---- L6' ------------------------------------------------------------------
def test_L6_pooled_split_conformal_exact_finite_sample_coverage():
    """n = 19, alpha = .1: coverage is exactly ceil(20*.9)/20 = .9 on average over calibration draws."""
    rng = np.random.default_rng(0)
    hits, trials = 0, 6000
    for _ in range(trials):
        lab = rng.integers(0, 3, 19)
        sc = _draw(rng, lab)
        sc_cal = sc
        sc_t_lab = rng.integers(0, 3, 1)
        row = _draw(rng, sc_t_lab)[0]
        cp = SplitConformal(0.1, mondrian=False).fit(sc_cal, lab)
        hits += int(sc_t_lab[0] in cp.predict_set(row))
    assert hits / trials >= 0.9 - 3 * math.sqrt(0.09 / trials)


def test_L6_mondrian_per_era_coverage():
    rng = np.random.default_rng(1)
    trials, n = 4000, 19
    hits = np.zeros(3)
    for _ in range(trials):
        lab = np.repeat(np.arange(3), n)
        cp = SplitConformal(0.1).fit(_draw(rng, lab), lab)
        for k in range(3):
            row = _draw(rng, np.array([k]))[0]
            hits[k] += int(k in cp.predict_set(row))
    assert (hits / trials >= 0.9 - 3 * math.sqrt(0.09 / trials)).all()


def test_L6_newborn_era_in_every_set_until_enough_calibration_points():
    alpha = 0.1  # ceil(1/alpha) - 1 = 9
    rng = np.random.default_rng(2)
    for n_new, expect_finite in [(0, False), (5, False), (8, False), (9, True), (30, True)]:
        lab = np.concatenate([np.repeat([0, 1], 50), np.full(n_new, 2)])
        cp = SplitConformal(alpha).fit(_draw(rng, lab), lab)
        far = np.array([50.0, 50.0, 50.0])  # a row no era explains
        C = cp.predict_set(far)
        assert (2 in C) == (not expect_finite)
        assert 0 not in C and 1 not in C
        assert np.isfinite(cp.quantiles_[2]) == expect_finite
    # alpha_t < 0 means "everything", alpha_t > 1 means "nothing" (L7 hypotheses)
    cp = SplitConformal(0.1).fit(_draw(rng, np.repeat([0, 1, 2], 40)), np.repeat([0, 1, 2], 40))
    assert cp.predict_set(np.array([9.0, 9.0, 9.0]), alpha=-0.01) == {0, 1, 2}
    assert cp.predict_set(np.array([0.0, 0.0, 0.0]), alpha=1.01) == set()


def test_L6_trajectory_split_is_disjoint_and_whole():
    rng = np.random.default_rng(3)
    traj = np.repeat(np.arange(20), 7)
    rng.shuffle(traj)
    tr, ca = trajectory_split(traj, 0.3, np.random.default_rng(4))
    assert len(np.intersect1d(tr, ca)) == 0 and len(tr) + len(ca) == len(traj)
    assert not set(traj[tr]) & set(traj[ca])  # whole trajectories, never split across sides
    assert len(set(traj[ca])) == 6
    tr2, ca2 = trajectory_split(traj, 0.3, np.random.default_rng(4))
    assert np.array_equal(ca, ca2)
    with pytest.raises(ValueError):
        trajectory_split(np.zeros(10, int), 0.5, rng)  # one trajectory cannot be split


# ---- L7 -------------------------------------------------------------------
def _adversary(name, rng):
    if name == "ones":
        return lambda t, a: 1
    if name == "zeros":
        return lambda t, a: 0
    if name == "alt":
        return lambda t, a: t % 2
    if name == "coin":
        return lambda t, a: int(rng.random() < 0.5)
    if name == "burst":
        return lambda t, a: int((t // 400) % 2 == 0)
    if name == "anti":  # errs exactly when the controller is relaxed, hides when it is tight
        return lambda t, a: int(a > 0.3)
    raise KeyError(name)


@pytest.mark.parametrize("name", ["ones", "zeros", "alt", "coin", "burst", "anti"])
@pytest.mark.parametrize("alpha,gamma,alpha1", [(0.1, 0.005, 0.1), (0.1, 0.05, 0.0), (0.5, 0.3, 1.0), (0.05, 0.02, 0.5)])
def test_L7_aci_bound_on_adversarial_sequences(name, alpha, gamma, alpha1):
    T = 10_000
    adv = _adversary(name, np.random.default_rng(5))
    aci = ACI(alpha, gamma, alpha1)
    errs = np.empty(T)
    for t in range(T):
        aci.update(adv(t, aci.alpha_t))  # raw adversarial err; update applies the boundary convention
        errs[t] = aci.last_err
    Ts = np.arange(1, T + 1)
    dev = np.abs(np.cumsum(errs) / Ts - alpha)
    bound = np.array([aci.bound(int(k)) for k in (1, 10, 100, 1000, T)])
    assert (dev[[0, 9, 99, 999, T - 1]] <= bound + 1e-12).all()
    assert (dev <= (max(alpha1, 1 - alpha1) + gamma) / (gamma * Ts) + 1e-12).all()  # every prefix
    assert aci.t == T and aci.mean_err == pytest.approx(errs.mean())


def test_L7_boundary_convention_amendment_C():
    """alpha_t < 0 -> C = all labels (err 0); alpha_t > 1 -> C = empty (err 1). Constant err sequences."""
    for const in (0, 1):
        aci = ACI(0.1, 0.05, 0.1)
        errs = []
        for _ in range(10_000):
            aci.update(const)
            errs.append(aci.last_err)
            assert -0.05 - 1e-12 <= aci.alpha_t <= 1.05 + 1e-12  # alpha_t stays in [-gamma, 1+gamma]
        errs = np.array(errs)
        assert abs(errs.mean() - 0.1) <= aci.bound(10_000) + 1e-12
        if const == 1:
            assert errs.mean() < 0.2  # raw err = 1 forever is overridden to 0 once alpha_t < 0
        else:
            assert errs.mean() > 0.0  # raw err = 0 forever is overridden to 1 once alpha_t > 1
    a = ACI(0.1, 0.05, -0.01)  # alpha_t < 0: the set is everything, so err is 0 whatever was reported
    a.update(1)
    assert a.last_err == 0 and a.alpha_t == pytest.approx(-0.01 + 0.05 * 0.1)
    a = ACI(0.1, 0.05, 1.0)
    a.alpha_t = 1.2  # alpha_t > 1: the set is empty, so err is 1
    a.update(0)
    assert a.last_err == 1 and a.alpha_t == pytest.approx(1.2 + 0.05 * (0.1 - 1))


def test_L7_bound_formula_and_validation():
    a = ACI(0.1, 0.01, 0.1)
    assert a.bound(100) == pytest.approx((0.9 + 0.01) / (0.01 * 100))
    assert ACI(0.1, 0.01, 0.8).bound(10) == pytest.approx((0.8 + 0.01) / 0.1)
    with pytest.raises(ValueError):
        a.bound(0)
    with pytest.raises(ValueError):
        a.update(0.5)
    with pytest.raises(ValueError):
        ACI(0.1, 0.0, 0.1)
    with pytest.raises(ValueError):
        ACI(1.5, 0.1, 0.1)
    # one step: alpha_{t+1} = alpha_t + gamma (alpha - err)
    a = ACI(0.1, 0.05, 0.2)
    assert a.update(1) == pytest.approx(0.2 + 0.05 * (0.1 - 1))
    assert a.update(0) == pytest.approx(0.155 + 0.05 * 0.1)


# ---- L24 / L24' -----------------------------------------------------------
def test_L24_birth_budget_formula():
    assert birth_budget(1000, 0.1, 0.05, 0.1) == pytest.approx(100 + (0.9 + 0.05) / 0.05)
    assert birth_budget(1000, 0.1, 0.05, 0.8) == pytest.approx(100 + (0.8 + 0.05) / 0.05)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_L24_births_within_budget_on_no_regime_change_stream(seed):
    rng = np.random.default_rng(seed)
    alpha, gamma, T = 0.1, 0.05, 5000
    lab = np.repeat(np.arange(3), 100)
    cp = SplitConformal(alpha).fit(_draw(rng, lab), lab)
    aci = ACI(alpha, gamma, alpha)
    truth = rng.integers(0, 3, T)
    rows = _draw(rng, truth)
    births = errors = 0
    for t in range(T):
        C = cp.predict_set(rows[t], alpha=aci.alpha_t)
        err = int(truth[t] not in C)
        births += int(len(C) == 0)  # births fire only on C = empty
        errors += err
        aci.update(err)
    budget = birth_budget(T, alpha, gamma, alpha)
    assert births <= errors  # an empty set cannot cover: every birth is an error
    assert errors <= budget  # L7 consequence
    assert births <= budget < T / 2  # and the budget is not vacuous
    assert errors > 0.5 * alpha * T  # the controller really was working at level alpha


# ---- L24* (anytime-valid birth) ---------------------------------------------
def _null_pvalues(seed, runs, T, n_cal=200, eta=0.005):
    """Null runs: fresh calibration per run, stream drawn from the known eras, novelty p-values."""
    rng = np.random.default_rng(seed)
    P = np.empty((runs, T))
    lab = np.repeat(np.arange(3), n_cal)
    for i in range(runs):
        cp = SplitConformal(0.1).fit(_draw(rng, lab), lab)
        P[i] = cp.novelty_pvalues(_draw(rng, rng.integers(0, 3, T)), eta)
    return P


def test_L24_star_false_birth_rate_at_most_delta_on_2000_null_runs_of_1e4_steps():
    delta, runs, T = 0.01, 2000, 10_000
    fired = 0
    for chunk in range(8):
        P = _null_pvalues(chunk, runs // 8, T, eta=delta / 2)  # calibration slack spends delta/2
        fired += int((first_birth(P, delta / 2) >= 0).sum())  # Ville spends delta/2
    assert fired / runs <= delta, f"{fired}/{runs} false births"


def test_L24_star_pvalues_are_conservative_and_novel_points_have_power():
    rng = np.random.default_rng(7)
    lab = np.repeat(np.arange(3), 200)
    cp = SplitConformal(0.1).fit(_draw(rng, lab), lab)
    P0 = cp.novelty_pvalues(_draw(rng, rng.integers(0, 3, 20_000)), 0.005)
    assert P0.min() > 0 and P0.max() <= 1
    for a in (0.1, 0.2, 0.5):
        assert (P0 <= a).mean() <= a  # super-uniform
    far = np.full((50, 3), 30.0)  # an unexplained regime: far from every era
    Pn = cp.novelty_pvalues(far, 0.005)
    assert Pn.max() < 0.2
    assert first_birth(Pn[None, :], 0.005)[0] in range(0, 40)  # fires within ~20 steps


def test_L24_star_calibration_slack_is_necessary_for_frozen_calibration():
    """Control: one era, n_cal = 50, kappa = .9 (near-zero null drift, so calibration luck matters).
    Raw conformal p-values (eta >= K, no slack) falsely birth far above delta; the DKW slack fixes it."""
    rng = np.random.default_rng(9)
    raw = slack = 0
    runs = 300
    for _ in range(runs):
        cp = SplitConformal(0.1).fit(np.abs(rng.standard_normal((50, 1))), np.zeros(50, int))
        S = np.abs(rng.standard_normal((10_000, 1)))
        raw += int(first_birth(cp.novelty_pvalues(S, 1.0)[None], 0.005, 0.9)[0] >= 0)
        slack += int(first_birth(cp.novelty_pvalues(S, 0.005)[None], 0.005, 0.9)[0] >= 0)
    assert raw / runs > 0.05
    assert slack / runs <= 0.01


def test_L24_star_trigger_restarts_with_summable_thresholds():
    t = BirthTrigger(delta=0.01, kappa=0.5)
    p_small = 1e-3  # factor 0.5 * 1e-3^{-0.5} = 15.8 per step
    fires = []
    for i in range(400):
        if t.update(p_small):
            fires.append(i)
    # j-th birth needs M >= j(j+1)/delta; thresholds grow so gaps grow
    assert len(fires) >= 3 and fires[0] == 1  # 15.8^2 = 250 >= 200 = 1/delta_1
    gaps = np.diff([-1] + fires)
    assert gaps[0] <= gaps[1] <= gaps[2] <= gaps[-1]
    assert t.births == len(fires) and t.j == len(fires) + 1
    # quiet stream (p = 1: factor kappa < 1) never fires; p in [0,1] is enforced
    q = BirthTrigger(0.01)
    assert not any(q.update(1.0) for _ in range(10_000))
    with pytest.raises(ValueError):
        q.update(1.5)
    with pytest.raises(ValueError):
        BirthTrigger(0.0)
    with pytest.raises(ValueError):
        BirthTrigger(0.1, kappa=1.0)
    assert BirthTrigger(0.1).update(0.0)  # p = 0: infinite evidence
    # vectorised first_birth agrees with the scalar trigger on the first firing
    rng = np.random.default_rng(8)
    Pm = rng.uniform(0, 1, (20, 300)) ** 3
    fb = first_birth(Pm, 0.1)
    for row, f in zip(Pm, fb):
        tr = BirthTrigger(0.1)
        got = next((i for i, p in enumerate(row) if tr.update(p)), -1)
        assert got == f


def test_L6_quantile_rank_is_ceil_n_plus_1_times_1_minus_alpha():
    """Pins the rank where (n+1)(1-alpha) and n(1-alpha) straddle an integer: n = 10, alpha = .1 -> rank 10."""
    s = np.arange(10, dtype=float)[:, None]
    lab = np.zeros(10, int)
    assert SplitConformal(0.1).fit(s, lab).quantiles_[0] == 9.0  # ceil(11 * .9) = 10th smallest
    s20 = np.arange(20, dtype=float)[:, None]
    assert SplitConformal(0.1).fit(s20, np.zeros(20, int)).quantiles_[0] == 18.0  # ceil(21 * .9) = 19th
    assert SplitConformal(0.1).fit(s20[:19], np.zeros(19, int)).quantiles_[0] == 17.0  # ceil(20 * .9) = 18th

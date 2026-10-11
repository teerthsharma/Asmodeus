"""Every bar and kill rule of Amendment 11 (docs/specs/PREREG_AMENDMENT_11_DRAFT.md), as pure functions.

Per-trajectory inputs are lists over seeds of 1-D arrays (one entry per Test trajectory). Bootstrap = the
amendment's cluster bootstrap: B = 10,000, seed 20261011, whole trajectories resampled within each seed, the
statistic is the mean over seeds; every arm shares the same resamples (paired). A bar returns a dict; a
bar whose precondition fails returns VOID (p = None), never a pass.
"""
import math

import numpy as np
from scipy import stats

from .conformal import ACI

ALPHA, GAMMA, ALPHA1, DELTA = 0.1, 0.05, 0.1, 0.05
W = 20
HORIZON = 50
ORIGINS = tuple(range(20, 341, 10))  # 33 origins
EVAL_SEEDS, TUNE_SEED = (0, 1, 2, 4, 5), 3
B_BOOT, BOOT_SEED = 10_000, 20261011
_EPS = 1e-12  # float slack on <= comparisons against round constants


# ------------------------------------------------------------------ estimators
def rel_err(xhat, x):
    """||xhat - x||_2 / ||x||_2 along the last axis."""
    xhat, x = np.asarray(xhat, float), np.asarray(x, float)
    return np.linalg.norm(xhat - x, axis=-1) / np.linalg.norm(x, axis=-1)


def traj_e50(xhat, x):
    """E50(arm, traj): mean over origins of the relative 50-step error (rows = origins)."""
    return float(np.mean(rel_err(xhat, x)))


def ratio(num, den):
    """rho = mean(num) / mean(den) over the seed scores."""
    return float(np.mean(num) / np.mean(den))


def upper95(x):
    return float(np.percentile(x, 95))


def lower95(x):
    return float(np.percentile(x, 5))


def seed_wins(f, g):
    return int(np.sum(np.asarray(f, float) < np.asarray(g, float)))


def paired_boot(per_traj, B=B_BOOT, seed=BOOT_SEED):
    """{arm: [per-seed arrays (n_traj,)]} -> {arm: (B,) resampled seed-mean}; one index draw per seed, shared by arms."""
    arms = list(per_traj)
    S = len(per_traj[arms[0]])
    rng = np.random.default_rng(seed)
    out = {a: np.zeros(B) for a in arms}
    for s in range(S):
        n = len(per_traj[arms[0]][s])
        idx = rng.integers(0, n, (B, n))
        for a in arms:
            out[a] += np.asarray(per_traj[a][s], float)[idx].mean(1)
    return {a: v / S for a, v in out.items()}


# ------------------------------------------------------------------ P1, P2, C1, C2
def p12(fcpa, base, B=B_BOOT):
    """P1/P2: rho <= 0.95, 95% upper bound of rho < 1, and FCPA wins on >= 4 seeds."""
    boot = paired_boot({"f": fcpa, "b": base}, B)
    rho_b = boot["f"] / boot["b"]
    rho = ratio([np.mean(a) for a in fcpa], [np.mean(a) for a in base])
    wins = seed_wins([np.mean(a) for a in fcpa], [np.mean(a) for a in base])
    ub = upper95(rho_b)
    return dict(rho=rho, ub=ub, wins=wins, n_seeds=len(fcpa), p_boot=float(np.mean(rho_b >= 1)),
                **{"pass": bool(rho <= 0.95 + _EPS and ub < 1 and wins >= 4)})


def choose_m(macs_by_m, fcpa_macs):
    """P2: the largest m in {1,2,3,4} whose inference MACs/step <= 1.05 x FCPA's; m = 1 if none fits."""
    ok = [m for m in (1, 2, 3, 4) if m in macs_by_m and macs_by_m[m] <= 1.05 * fcpa_macs]
    return max(ok) if ok else 1


def c1(fcpa_inf, base_inf):
    return bool(fcpa_inf <= base_inf)


def steps_for_budget(budget, macs_per_step):
    """C2: steps = floor(B_FCPA / MACs per step)."""
    return int(budget // macs_per_step)


def c2(used, budget):
    return bool(used <= budget)


# ------------------------------------------------------------------ P3
def era_map(learned, regime, n_eras):
    """Majority vote of the true regime over each learned era's Cal steps (ties: smallest id); unused era -> -1."""
    learned, regime = np.asarray(learned, int), np.asarray(regime, int)
    out = {}
    for k in range(n_eras):
        r = regime[learned == k]
        out[k] = int(np.bincount(r).argmax()) if len(r) else -1
    return out


def distance_to_switch(regime):
    """Steps to the nearest true switch (a switch at t means regime[t] != regime[t-1]); inf with no switch."""
    regime = np.asarray(regime)
    sw = np.flatnonzero(regime[1:] != regime[:-1]) + 1
    if not len(sw):
        return np.full(len(regime), np.inf)
    return np.abs(np.arange(len(regime))[:, None] - sw[None]).min(1).astype(float)


def id_counts(pred_regime, regime, w=W):
    """(correct, total) over steps more than w from the nearest true switch."""
    far = distance_to_switch(regime) > w
    return int(np.sum(np.asarray(pred_regime)[far] == np.asarray(regime)[far])), int(far.sum())


def id_lower95(correct, total, B=B_BOOT):
    """One-sided 95% lower bound of the seed-mean identification rate (pooled per seed, trajectories resampled)."""
    rng = np.random.default_rng(BOOT_SEED)
    S, acc = len(correct), np.zeros(B)
    for s in range(S):
        n = len(correct[s])
        idx = rng.integers(0, n, (B, n))
        acc += np.asarray(correct[s], float)[idx].sum(1) / np.asarray(total[s], float)[idx].sum(1)
    return lower95(acc / S)


def p3(seed_scores):
    s = np.asarray(seed_scores, float)
    return dict(mean=float(s.mean()), worst=float(s.min()), **{"pass": bool(s.mean() >= 0.9 - _EPS)})


# ------------------------------------------------------------------ P4
def coverage_deviation(errs, alpha=ALPHA):
    """|coverage - (1 - alpha)| = |mean(err) - alpha|."""
    return float(abs(np.mean(errs) - alpha))


def p4_pass(devs):
    return bool(all(d <= 0.02 + _EPS for d in devs))


def p4_min_T(alpha1=ALPHA1, gamma=GAMMA, tol=0.02):
    """T >= (max(alpha1, 1 - alpha1) + gamma) / (tol gamma): 950 steps at the pins."""
    return math.ceil((max(alpha1, 1 - alpha1) + gamma) / (tol * gamma) - 1e-9)


def p4_precondition(T, alpha1=ALPHA1, gamma=GAMMA):
    """False voids P4 (T < 950)."""
    return bool(T >= p4_min_T(alpha1, gamma))


def run_aci(set_fn, truth, n_labels, alpha=ALPHA, gamma=GAMMA, alpha1=None):
    """ACI over a stream with the Gibbs-Candes boundary convention (alpha_t < 0: C = all labels, err 0;
    alpha_t > 1: C = empty, err 1). set_fn(t, alpha_t) -> set of (mapped) labels. Returns (errs, |C_t|)."""
    aci = ACI(alpha, gamma, alpha1)
    errs, sizes = np.zeros(len(truth), int), np.zeros(len(truth), int)
    for t, y in enumerate(truth):
        forced = aci.forced_err()
        if forced is None:
            C = set_fn(t, aci.alpha_t)
            err, size = int(y not in C), len(C)
        else:
            err, size = forced, (n_labels if forced == 0 else 0)
        aci.update(err)
        errs[t], sizes[t] = aci.last_err, size
    return errs, sizes


# ------------------------------------------------------------------ Holm and S1-S5
def holm(pvals, alpha=0.05):
    """Step-down Holm over {name: p}; a None p (VOID bar) leaves the family and maps to None."""
    live = {k: p for k, p in pvals.items() if p is not None}
    m, out, stop = len(live), {k: None for k in pvals}, False
    for i, (k, p) in enumerate(sorted(live.items(), key=lambda kv: kv[1])):
        stop = stop or p > alpha / (m - i)
        out[k] = not stop
    return out


def s1(ns_form, learned, B=B_BOOT):
    boot = paired_boot({"n": ns_form, "l": learned}, B)
    rg = boot["n"] / boot["l"]
    rho = ratio([np.mean(a) for a in ns_form], [np.mean(a) for a in learned])
    ub = upper95(rg)
    return dict(rho=rho, ub=ub, p=float(np.mean(rg > 1.02)), point_pass=bool(ub <= 1.02 + _EPS), kill=bool(rho > 1.10))


def cp_upper(x, n, conf=0.95):
    """Exact one-sided Clopper-Pearson upper bound."""
    return 1.0 if x >= n else float(stats.beta.ppf(conf, x + 1, n - x))


def s2(x, n, delta=DELTA):
    rate, ub = x / n, cp_upper(x, n)
    return dict(rate=rate, ub=ub, p=float(stats.binom.cdf(x, n, delta)),
                point_pass=bool(rate <= delta + _EPS and ub <= delta + _EPS), kill=bool(rate > 2 * delta + _EPS))


def s3(tp, n_events):
    if n_events < 50:
        return dict(void=True, precision=None, p=None, point_pass=None, kill=None)
    pr = tp / n_events
    return dict(void=False, precision=pr, p=float(stats.binom.sf(tp - 1, n_events, 0.95)),
                point_pass=bool(pr >= 0.95 - _EPS), kill=bool(pr < 0.90 - _EPS))


def merge_events(times, w=W):
    """Certified events within 2w of the previous one merge into the first of the cluster."""
    out = []
    for t in sorted(times):
        if not out or t - last > 2 * w:
            out.append(t)
        last = t
    return np.array(out)


def count_true_positives(events, switches, w=W):
    sw = np.asarray(switches)
    return int(sum(bool(np.any((sw >= t - w) & (sw <= t + w))) for t in events))


def benettin(step, x0, direction, h_B, eps0=1e-6):
    """Two-trajectory finite-time Lyapunov exponent per step, separation renormalised to eps0*||x0|| each step."""
    x0 = np.asarray(x0, float)
    d = np.asarray(direction, float)
    scale = eps0 * np.linalg.norm(x0)
    x, xp = x0, x0 + scale * d / np.linalg.norm(d)
    acc = 0.0
    for _ in range(h_B):
        x, xp = step(x), step(xp)
        sep = xp - x
        nrm = np.linalg.norm(sep)
        acc += math.log(nrm / scale)
        xp = x + sep * (scale / nrm)
    return acc / h_B


def horizon(lam_data, eps0=1e-6):
    return math.log(0.3 / eps0) / lam_data


def h_B(lam_data, eps0=1e-6):
    return int(min(50, math.floor(horizon(lam_data, eps0))))


def eta(lam_data):
    return 0.25 * abs(lam_data)


def s4_void(lam_data, eps0=1e-6):
    """lam_data <= 0 or H < 10: the bed is not chaotic enough to test S4."""
    return bool(lam_data <= 0 or horizon(lam_data, eps0) < 10)


def s4(diffs, eta, B=B_BOOT):
    """diffs: per-seed arrays of lam_model - lam_data per trajectory."""
    boot = np.abs(paired_boot({"d": diffs}, B)["d"])
    point = float(abs(np.mean([np.mean(a) for a in diffs])))
    ub = upper95(boot)
    return dict(point=point, ub=ub, p=float(np.mean(boot > eta)), point_pass=bool(ub <= eta + _EPS), kill=bool(point > eta + _EPS))


def s5(csize, B=B_BOOT):
    boot = paired_boot({"c": csize}, B)["c"]
    mean = float(np.mean([np.mean(a) for a in csize]))
    return dict(mean=mean, ub=upper95(boot), p=float(np.mean(boot > 1.2)), point_pass=bool(mean <= 1.2 + _EPS),
                kill=bool(mean > 2.0 + _EPS))


# ------------------------------------------------------------------ kill rules, primary verdict
def _fire(cond, *needs):
    return None if any(n is None for n in needs) else bool(cond())


def kills(rho_global=None, rho_asm=None, c_global=None, c_asm=None, id_mean=None, cov_dev=None, rho_G=None,
          x_rate=None, precision=None, lam_diff=None, eta=None, mean_C=None):
    """K1-K9 on point estimates. None = the bar was not run (the kill is None, not False)."""
    return dict(
        K1=_fire(lambda: rho_global > 1.0 or not c_global, rho_global, c_global),
        K2=_fire(lambda: rho_asm > 1.0 or not c_asm, rho_asm, c_asm),
        K3=_fire(lambda: id_mean < 0.8 - _EPS, id_mean),
        K4=_fire(lambda: any(d > 0.02 + _EPS for d in cov_dev), cov_dev),
        K5=_fire(lambda: rho_G > 1.10 + _EPS, rho_G),
        K6=_fire(lambda: x_rate > 0.10 + _EPS, x_rate),
        K7=_fire(lambda: precision < 0.90 - _EPS, precision),
        K8=_fire(lambda: abs(lam_diff) > eta + _EPS, lam_diff, eta),
        K9=_fire(lambda: mean_C > 2.0 + _EPS, mean_C))


def primary_verdict(p1, p2, c1, c2, p3, p4):
    """PASS iff P1-P4, C1, C2 all True. Any False -> FAIL; else any 'NOT RUN' -> NOT RUN; else a None (VOID) -> VOID."""
    v = [p1, p2, c1, c2, p3, p4]
    if any(x is False for x in v):
        return "FAIL"
    if any(x == "NOT RUN" for x in v if isinstance(x, str)):
        return "NOT RUN"
    return "VOID" if any(x is None for x in v) else "PASS"


# ------------------------------------------------------------------ MAC helpers (real multiply-adds)
def macs_matmul(m, k, n):
    return m * k * n


def macs_gram(n, d):
    """X^T X over n rows of width d."""
    return n * d * d


def macs_solve(d, p):
    """Closed-form solve of a d x d system for p right-hand sides: factor d^3 + back-substitute d^2 p."""
    return d ** 3 + d * d * p


def macs_complex(n):
    """n complex multiply-adds = 4n real ones."""
    return 4 * n


def macs_bwd(fwd):
    """Gradient-trained part: forward + backward at 2x forward."""
    return 3 * fwd

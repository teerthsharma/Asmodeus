"""Conformal layer: L6' (Mondrian split conformal), L7 (ACI), L24 budget, L24* (anytime-valid birth trigger).

Scores are NONconformity scores: smaller means the era explains the history better; k is in C iff s_k <= q_k.
"""
from __future__ import annotations

import math

import numpy as np


def _quantile(sorted_scores, n_min, alpha):
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score.

    +inf (everything in) if too few points (n < n_min, the 'newborn' rule) or alpha < 0; -inf (nothing in)
    if alpha > 1 or the rank is < 1.
    """
    n = len(sorted_scores)
    if n < n_min or alpha < 0:
        return np.inf
    if alpha > 1:
        return -np.inf
    idx = math.ceil((n + 1) * (1.0 - alpha) - 1e-9)
    if idx > n:
        return np.inf
    if idx < 1:
        return -np.inf
    return float(sorted_scores[idx - 1])


class SplitConformal:
    """L6' split conformal era sets.  Calibrate on data disjoint from training (whole trajectories, see
    `trajectory_split`).  `mondrian=True` calibrates each era on its own points, so coverage holds per era;
    an era with fewer than ceil(1/alpha) - 1 calibration points (a newborn) is in every set.
    """

    def __init__(self, alpha=0.1, mondrian=True):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.mondrian = mondrian
        self.n_min_ = max(math.ceil(1.0 / alpha - 1e-9) - 1, 1)

    def fit(self, scores_cal, labels_cal):
        """scores_cal (n, K): s(h_i, k) for every era k; labels_cal (n,): the true era (by timestamp)."""
        S = np.asarray(scores_cal, dtype=float)
        lab = np.asarray(labels_cal, dtype=int)
        n, K = S.shape
        s = S[np.arange(n), lab]
        if self.mondrian:
            self.sorted_ = [np.sort(s[lab == k]) for k in range(K)]
        else:
            pool = np.sort(s)
            self.sorted_ = [pool] * K
        self.n_cal_ = np.array([len(a) for a in self.sorted_])
        self.quantiles_ = np.array([_quantile(a, self.n_min_, self.alpha) for a in self.sorted_])
        return self

    def predict_set(self, score_row, alpha=None):
        """The era set for one history; `alpha` overrides the level (ACI's alpha_t)."""
        row = np.asarray(score_row, dtype=float)
        if alpha is None:
            q = self.quantiles_
        else:
            q = np.array([_quantile(a, self.n_min_, alpha) for a in self.sorted_])
        return {int(k) for k in np.flatnonzero(row <= q)}

    def novelty_pvalues(self, S, eta):
        """L24*: per-era conformal p-values, combined by max over eras (a point of era k* has p >= p_k*).

        p_k = (1 + #{cal_k >= s}) / (n_k + 1) + eps_k with eps_k = sqrt(ln(K/eta) / (2 n_k)) (one-sided DKW):
        with probability >= 1 - eta over the calibration draw, every p_k is conditionally super-uniform, so
        the stream of p-values feeds a martingale test with total false-birth probability <= eta + delta.
        eta >= K switches the slack off (raw p-values; NOT safe with frozen calibration).  Eras without
        calibration points have p = 1.
        """
        if not self.mondrian:
            raise ValueError("novelty p-values need per-era (Mondrian) calibration")
        S = np.atleast_2d(np.asarray(S, dtype=float))
        K_eff = int((self.n_cal_ > 0).sum())
        out = np.zeros(S.shape[0])
        for k, a in enumerate(self.sorted_):
            n = len(a)
            if n == 0:
                out = np.ones(S.shape[0])
                continue
            tail = n - np.searchsorted(a, S[:, k], side="left")
            eps = math.sqrt(max(math.log(K_eff / eta), 0.0) / (2.0 * n))
            out = np.maximum(out, np.minimum(1.0, (1.0 + tail) / (n + 1.0) + eps))
        return out


def trajectory_split(traj_ids, cal_frac, rng=None):
    """(train_idx, cal_idx) with whole trajectories on one side (the exchangeable unit), deterministic in rng."""
    traj_ids = np.asarray(traj_ids)
    u = np.unique(traj_ids)
    if len(u) < 2:
        raise ValueError("need at least 2 trajectories to split")
    n_cal = int(min(max(round(cal_frac * len(u)), 1), len(u) - 1))
    g = np.random.default_rng(rng)
    cal_ids = g.choice(u, size=n_cal, replace=False)
    is_cal = np.isin(traj_ids, cal_ids)
    return np.flatnonzero(~is_cal), np.flatnonzero(is_cal)


class ACI:
    """Adaptive conformal inference (L7): alpha_{t+1} = alpha_t + gamma (alpha - err_t).

    Boundary convention (Amendment C, Gibbs-Candes): alpha_t < 0 means C = all labels (err = 0) and
    alpha_t > 1 means C = empty (err = 1); `update` applies it to whatever err the caller reports, so
    |mean err - alpha| <= bound(T) holds for every sequence.
    """

    def __init__(self, alpha, gamma, alpha1=None):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not gamma > 0:
            raise ValueError("gamma must be > 0")
        alpha1 = alpha if alpha1 is None else alpha1
        if not -gamma <= alpha1 <= 1 + gamma:
            raise ValueError("alpha1 must lie in [-gamma, 1 + gamma]")
        self.alpha, self.gamma, self.alpha1 = alpha, gamma, alpha1
        self.alpha_t = float(alpha1)
        self.t = 0
        self._sum = 0
        self.last_err = None

    def update(self, err):
        """Record one step's miscoverage (0/1); returns alpha_{t+1}."""
        if err not in (0, 1):
            raise ValueError("err must be 0 or 1")
        e = 0 if self.alpha_t < 0 else 1 if self.alpha_t > 1 else int(err)
        self.alpha_t += self.gamma * (self.alpha - e)
        self.t += 1
        self._sum += e
        self.last_err = e
        return self.alpha_t

    def forced_err(self):
        """0 if alpha_t < 0 (C = all labels), 1 if alpha_t > 1 (C = empty), else None (err is free)."""
        return 0 if self.alpha_t < 0 else 1 if self.alpha_t > 1 else None

    @property
    def mean_err(self):
        return self._sum / self.t if self.t else 0.0

    def bound(self, T):
        """|(1/T) sum err_t - alpha| <= (max(alpha1, 1 - alpha1) + gamma) / (gamma T)."""
        if T < 1:
            raise ValueError("T must be >= 1")
        return (max(self.alpha1, 1.0 - self.alpha1) + self.gamma) / (self.gamma * T)


def birth_budget(T, alpha, gamma, alpha1=None):
    """L24: births triggered only on C = empty are errors, so over T no-regime steps they number at most
    alpha T + (max(alpha1, 1 - alpha1) + gamma) / gamma."""
    alpha1 = alpha if alpha1 is None else alpha1
    return alpha * T + (max(alpha1, 1.0 - alpha1) + gamma) / gamma


class BirthTrigger:
    """L24*: power martingale M_t = prod kappa p_s^{kappa-1} over novelty p-values.  The j-th birth fires at
    M >= 1/delta_j, delta_j = delta / (j (j + 1)); M then restarts at 1.  Under a super-uniform p-value stream
    P(any birth) <= sum_j delta_j <= delta (Ville).  Never reads ACI's alpha_t.
    """

    def __init__(self, delta=0.01, kappa=0.5):
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if not 0 < kappa < 1:
            raise ValueError("kappa must be in (0, 1)")
        self.delta, self.kappa = delta, kappa
        self.j, self.births, self.logM = 1, 0, 0.0

    def update(self, p):
        """Feed one p-value; True iff this step fires a birth."""
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        fire = p == 0.0
        if not fire:
            self.logM += np.log(self.kappa) + (self.kappa - 1.0) * np.log(p)
            fire = self.logM >= np.log(self.j * (self.j + 1) / self.delta)
        if fire:
            self.births += 1
            self.j += 1
            self.logM = 0.0
        return bool(fire)


def first_birth(P, delta, kappa=0.5):
    """Vectorised first firing index of `BirthTrigger(delta, kappa)` for each row of P (runs, T); -1 if none."""
    P = np.asarray(P, dtype=float)
    with np.errstate(divide="ignore"):
        lf = np.log(kappa) + (kappa - 1.0) * np.log(P)
    hit = np.cumsum(lf, axis=1) >= np.log(2.0 / delta)
    return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)

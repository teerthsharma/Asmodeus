"""CTC loop (D10, L22, L22*, L22-dagger): the damped fixed point w <- (1-eps) G(w) + (eps/M) 1 on the simplex."""
import math

import numpy as np


def post_lipschitz(temp, delta):
    """L22*: Lipschitz constant of post-selection for FIXED T, on {p_k >= delta, k in T}: 2 + 2/(T*delta)."""
    return 2.0 + 2.0 / (temp * delta)


def contraction(eps, L_G, L_post=1.0):
    """L22*: worst-case c = (1-eps) * L_G * L_post."""
    return (1.0 - eps) * L_G * L_post


def residual_bound(eps):
    """L22*: at the damped fixed point w*, |G(w*) - w*|_1 = eps |u - G(w*)|_1 <= 2 eps (u uniform)."""
    return 2.0 * eps


def bias_bound(eps, c):
    """L22*: |w* - wbar|_1 <= |F(wbar) - wbar|_1 / (1 - c) <= 2 eps / (1 - c) for an undamped fixed point wbar."""
    return 2.0 * eps / (1.0 - c)


def orbit_radius(d0, c):
    """L22*: the orbit of F from w0 stays in the l1 ball of this radius around w0 (d0 = |F(w0) - w0|_1, c < 1)."""
    return d0 / (1.0 - c)


def iteration_bound(d0, K, tol):
    """L22: iterations needed so the last step residual is <= tol: res_n <= d0 K^(n-1)."""
    if d0 <= tol or K <= 0:
        return 1
    return 1 + math.ceil(math.log(tol / d0) / math.log(K))


def km_budget(tol, diam=2.0, alpha=0.5):
    """L22-dagger (CITED, Cominetti-Soto-Vaisman 2014): Krasnosel'skii-Mann with constant alpha,
    |x_n - T x_n| <= diam / sqrt(pi n alpha (1-alpha)); diam = 2 on the simplex in l1. n >= this budget suffices."""
    return math.ceil(diam**2 / (math.pi * alpha * (1 - alpha) * tol**2))


def local_norm(G, w, h=1e-6):
    """Orbit-local l1 Lipschitz estimate of G at w, on the simplex tangent (sum-zero vectors).

    Directional secants D_j = (G(w + h(e_j - w)) - G(w)) / h stay inside the simplex; the extreme points of the
    tangent l1 ball are (e_i - e_j)/2, so |J|_tan = max_{i<j} |D_i - D_j|_1 / 2.
    """
    M, g = len(w), np.asarray(G(w), float)
    if hasattr(G, "batch"):  # all M secant points in one call
        D = (np.asarray(G.batch(w + h * (np.eye(M) - w)), float) - g) / h
    else:
        D = np.stack([(np.asarray(G(w + h * (np.eye(M)[j] - w)), float) - g) / h for j in range(M)])
    return float(np.abs(D[:, None, :] - D[None, :, :]).sum(-1).max() / 2)


class LoopResult(tuple):
    """(w, n_iters, converged) plus .mode in {DAMPED, KM, REFUSE}, .c_loc (orbit-local), .c_worst (declared)."""

    def __new__(cls, w, n, ok, mode, c_loc, c_worst):
        self = super().__new__(cls, (w, n, ok))
        self.mode, self.c_loc, self.c_worst = mode, c_loc, c_worst
        return self


def damped_loop(G, w0, eps, L, tol, max_iter=10_000, certified=True):
    """L22-dagger dispatcher. Returns LoopResult (unpacks as (w, n_iters, converged)).

    L is the declared worst-case l1 Lipschitz constant of G (None = unknown, c_worst = inf); it is only reported.
    The decision uses the orbit-local constant L_n = |J_G(w_n)|_tan along the actual orbit:
      DAMPED  c_loc = (1-eps) max_n L_n < 1: iterate F = (1-eps) G + eps/M. Converged means the last step
              residual <= tol (so |F(w)-w|_1 <= c tol); a-priori n <= iteration_bound(d0, c_loc, tol) (L22).
      KM      G nonexpansive on the orbit (L_n <= 1) but not damped-contractive (eps = 0): w <- (w + G(w))/2;
              converged means |G(w)-w|_1 <= tol, budget km_budget(tol).
      REFUSE  G expansive on the orbit, T not certified (L5), or the orbit later leaves the regime: the
              point is returned unconverged, never claimed.
    """
    w = np.asarray(w0, float)
    if w.ndim != 1 or np.any(w < -1e-12) or abs(w.sum() - 1) > 1e-9:
        raise ValueError("w0 must lie on the probability simplex")
    c_worst = (1.0 - eps) * L if L is not None else np.inf
    M = len(w)
    L0 = local_norm(G, w)
    if not certified:
        return LoopResult(w.copy(), 0, False, "REFUSE", (1.0 - eps) * L0, c_worst)
    if (1.0 - eps) * L0 < 1.0:
        mode, step = "DAMPED", lambda v: (1.0 - eps) * np.asarray(G(v), float) + eps / M
    elif L0 <= 1.0 + 1e-9:
        mode, step = "KM", lambda v: 0.5 * (v + np.asarray(G(v), float))
    else:
        return LoopResult(w.copy(), 0, False, "REFUSE", (1.0 - eps) * L0, c_worst)
    scale = (1.0 - eps) if mode == "DAMPED" else 1.0
    c_loc = scale * L0
    for n in range(1, max_iter + 1):
        w1 = step(w)
        res = np.abs(w1 - w).sum() if mode == "DAMPED" else np.abs(np.asarray(G(w), float) - w).sum()
        w = w1
        if res <= tol:
            return LoopResult(w, n, True, mode, c_loc, c_worst)
        Ln = local_norm(G, w)  # monitor the orbit: leaving the certified regime means REFUSE, not a guess
        c_loc = max(c_loc, scale * Ln)
        if (mode == "DAMPED" and c_loc >= 1.0) or (mode == "KM" and Ln > 1.0 + 1e-9):
            return LoopResult(w, n, False, "REFUSE", c_loc, c_worst)
    return LoopResult(w, max_iter, False, mode, c_loc, c_worst)

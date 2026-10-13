"""E0 arms (PROGRAM.md section 6). Each arm returns per-token log-loss -log p(x_t | x_<t), t = 1..T-1.

Every count model uses the shared Dirichlet prior a (A_PRIOR). numpy only.
"""
import math
from collections import deque

import numpy as np

A_PRIOR = 0.2


def _logml_table(V, n_max, a=A_PRIOR):
    """lgamma tables for the Dirichlet-multinomial log marginal likelihood of integer count rows."""
    n = np.arange(n_max + 1)
    return (np.array([math.lgamma(a + i) for i in n]) - math.lgamma(a),
            math.lgamma(V * a) - np.array([math.lgamma(V * a + i) for i in n]))


def _logml(C, tabs):
    cell, row = tabs
    Ci = C.astype(int)
    assert Ci.min() >= 0, "negative count: era bookkeeping is broken"
    return float(cell[Ci].sum() + row[Ci.sum(1)].sum())


def _pred(row, a=A_PRIOR):
    return (row + a) / (row.sum() + a * row.size)


def oracle(x, r, V):
    """One count table per true regime id."""
    C, out = {}, np.empty(len(x) - 1)
    for t in range(1, len(x)):
        c = C.setdefault(r[t], np.zeros((V, V)))
        out[t - 1] = -np.log(_pred(c[x[t - 1]])[x[t]])
        c[x[t - 1], x[t]] += 1
    return out


def global_o1(x, V):
    C, out = np.zeros((V, V)), np.empty(len(x) - 1)
    for t in range(1, len(x)):
        out[t - 1] = -np.log(_pred(C[x[t - 1]])[x[t]])
        C[x[t - 1], x[t]] += 1
    return out


def global_o2(x, V, beta):
    """Order-2 counts backed off to the order-1 predictive with strength beta (hierarchical Dirichlet)."""
    C1, C2, out = np.zeros((V, V)), np.zeros((V, V, V)), np.empty(len(x) - 1)
    for t in range(1, len(x)):
        p1 = _pred(C1[x[t - 1]])
        if t >= 2:
            c2 = C2[x[t - 2], x[t - 1]]
            p = (c2 + beta * p1) / (c2.sum() + beta)
            C2[x[t - 2], x[t - 1], x[t]] += 1
        else:
            p = p1
        out[t - 1] = -np.log(p[x[t]])
        C1[x[t - 1], x[t]] += 1
    return out


def ema(x, V, lam):
    C, out = np.zeros((V, V)), np.empty(len(x) - 1)
    for t in range(1, len(x)):
        out[t - 1] = -np.log(_pred(C[x[t - 1]])[x[t]])
        C *= lam
        C[x[t - 1], x[t]] += 1
    return out


def bocpd(x, V, hazard, rmax=1500, a=A_PRIOR, return_vec=False):
    """Adams-MacKay BOCPD, Dirichlet-categorical rows; run length capped at rmax (tail merged into rmax)."""
    T = len(x)
    pref = np.zeros((T, V, V), np.int32)  # pref[j] = counts of transitions s = 1..j
    for s in range(1, T):
        pref[s] = pref[s - 1]
        pref[s, x[s - 1], x[s]] += 1
    pi, out = np.array([1.0]), np.empty(T - 1)
    vec = np.empty((T - 1, V)) if return_vec else None
    for t in range(1, T):
        R = len(pi)  # run lengths 0..R-1, R-1 <= min(t-1, rmax)
        rs = np.arange(R)
        prev = x[t - 1]
        cnt = pref[t - 1, prev][None, :] - pref[np.maximum(t - 1 - rs, 0), prev]
        pr = (cnt[:, x[t]] + a) / (cnt.sum(1) + a * V)
        if return_vec:
            vec[t - 1] = pi @ ((cnt + a) / (cnt.sum(1, keepdims=True) + a * V))
        joint = pi * pr
        p = joint.sum()
        out[t - 1] = -np.log(p)
        new = np.empty(min(R + 1, rmax + 1))
        new[0] = hazard * p
        grow = joint * (1 - hazard)
        if R + 1 <= rmax + 1:
            new[1:] = grow
        else:  # merge the oldest run into the capped bin
            new[1:] = grow[:-1]
            new[-1] += grow[-1]
        pi = new / p
    return (out, vec) if return_vec else out


class _Detector:
    """Shin-Ramdas-Rinaldo mixture e-detector on likelihood-ratio e-values; CUSUM argmin as the change-time estimate."""

    def __init__(self, delta):
        self.thr = 1.0 / delta
        self.reset(0)

    def reset(self, t):
        self.S, self.n, self.cum, self.cmin, self.tmin = 0.0, 0, 0.0, 0.0, t

    def update(self, e, t):
        self.n += 1
        self.S = (self.S + 1.0 / (self.n * (self.n + 1.0))) * e
        self.cum += np.log(e)
        if self.cum < self.cmin:
            self.cmin, self.tmin = self.cum, t
        return self.S + 1.0 / (self.n + 1.0) >= self.thr


def fcpa(x, V, delta, w, alphas=(), lam=0.5, n_max=64, a=A_PRIOR, return_info=False, ext=None, merge=False, n_min=0, deciders=(), ext_vec=None):
    """Era bank + e-detector + address/birth.

    HARD: predict with the active era (TT ALONE). SOFT[alpha]: fixed-share forward filter over bank eras plus the
    window model (L14' HMM filter). Both share one bookkeeping (births and era statistics come from HARD).
    E0b: ext = per-step probability of the realised token from an external novelty expert (BOCPD), appended to
    the filter's experts; merge = Dirichlet-multinomial Bayes-factor merge of the active era after every alarm.
    E0c: merge="global" merges the best pair among all eras while its Bayes factor > 1; n_min delays the address
    decision until the segment since the CUSUM change-time estimate holds >= n_min transitions.
    """
    n_tail = 1 + (ext is not None)
    tabs = _logml_table(V, len(x), a) if merge else None
    T = len(x)
    eras = [np.zeros((V, V))]
    k = 0
    W = np.zeros((V, V))
    win = deque()  # last w transitions (prev, cur)
    hist = deque(maxlen=4000)  # (t, prev, cur, era) for re-assignment on alarm
    det = _Detector(delta)
    hard = np.empty(T - 1)
    soft = {al: np.empty(T - 1) for al in alphas}
    vw = {al: np.ones(1 + n_tail) / (1 + n_tail) for al in alphas}  # weights over [eras..., window, ext]
    active = np.empty(T, int)
    active[0] = 0
    births = []
    pending = False
    dec = {d: np.empty(T - 1) for d in deciders}

    def fold(i, j):  # merge era j into era i (i != j); returns the new index of i
        nonlocal eras, hist, k
        eras[i] += eras[j]
        eras.pop(j)
        for al in alphas:
            v = vw[al]
            v[i] += v[j]
            vw[al] = np.delete(v, j)
        for d in deciders:
            d.fold(i, j)
        hist = deque(((tt, p_, c_, (i if kk == j else kk) - ((i if kk == j else kk) > j)) for (tt, p_, c_, kk) in hist),
                     maxlen=4000)
        active[:t][active[:t] == j] = i
        active[:t][active[:t] > j] -= 1
        k = (i if k == j else k) - ((i if k == j else k) > j)
        return i - (i > j)

    for t in range(1, T):
        prev, cur = x[t - 1], x[t]
        A = _pred(eras[k][prev], a)
        Wp = _pred(W[prev], a)
        hard[t - 1] = -np.log(A[cur])
        if alphas:
            P = np.stack([_pred(E[prev], a)[cur] for E in eras] + [Wp[cur]] + ([ext[t - 1]] if ext is not None else []))
            for al in alphas:
                v = vw[al]
                v = (1 - al) * v + al / len(v)
                pm = float(v @ P)
                soft[al][t - 1] = -np.log(pm)
                vw[al] = v * P / pm
        if deciders:
            Y = np.stack([_pred(E[prev], a) for E in eras] + [Wp] + ([ext_vec[t - 1]] if ext_vec is not None else []))
            for d in deciders:
                dec[d][t - 1] = -np.log(d.step(Y, cur))
        e = (lam * A[cur] + (1 - lam) * Wp[cur]) / A[cur]
        eras[k][prev, cur] += 1
        hist.append((t, prev, cur, k))
        win.append((prev, cur))
        W[prev, cur] += 1
        if len(win) > w:
            p0, c0 = win.popleft()
            W[p0, c0] -= 1
        pending = det.update(e, t) or pending
        if pending and t - det.tmin >= n_min:
            pending = False
            seg = [h for h in hist if h[0] > det.tmin]
            if not seg:
                seg = [hist[-1]]
            for (_, p_, c_, kk) in seg:  # take the segment out of the eras it was credited to
                eras[kk][p_, c_] -= 1
            best, best_ll = None, -np.inf
            cands = list(range(len(eras))) + ([None] if len(eras) < n_max else [])
            for j in cands:
                E = np.zeros((V, V)) if j is None else eras[j].copy()
                ll = 0.0
                for (_, p_, c_, _) in seg:  # prequential (marginal) log-likelihood of the segment
                    ll += np.log(_pred(E[p_], a)[c_])
                    E[p_, c_] += 1
                if ll > best_ll:
                    best, best_ll = j, ll
            if best is None:
                eras.append(np.zeros((V, V)))
                best = len(eras) - 1
                births.append(t)
                for d in deciders:
                    d.birth()
                for al in alphas:
                    v = vw[al]
                    vw[al] = np.concatenate([v[:-n_tail] * (1 - al), [al], v[-n_tail:] * (1 - al)])
                    vw[al] /= vw[al].sum()
            for (_, p_, c_, _) in seg:
                eras[best][p_, c_] += 1
            hist = deque(((tt, p_, c_, best if tt > det.tmin else kk) for (tt, p_, c_, kk) in hist), maxlen=4000)
            for d in deciders:
                d.switch(k, best)
            k = best
            while merge and merge != "global":  # E0b: active era vs each other era (Bayes factor > 1)
                lk = _logml(eras[k], tabs)
                j = next((j for j in range(len(eras)) if j != k and
                          _logml(eras[k] + eras[j], tabs) > lk + _logml(eras[j], tabs)), None)
                if j is None:
                    break
                fold(k, j)
            while merge == "global" and len(eras) > 1:  # E0c: best pair among all eras while Bayes factor > 1
                lm = [_logml(E, tabs) for E in eras]
                bf, i, j = max((_logml(eras[i] + eras[j], tabs) - lm[i] - lm[j], i, j)
                               for i in range(len(eras)) for j in range(i + 1, len(eras)))
                if bf <= 0:
                    break
                fold(i, j)
            det.reset(t)
        active[t] = k
    out = {"hard": hard, "soft": soft, "dec": dec}
    if return_info:
        out.update(n_eras=len(eras), births=births, active=active)
    return out

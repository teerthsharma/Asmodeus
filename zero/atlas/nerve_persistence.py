"""Persistence of the observer nerve over the threshold tau (numpy only).

Simplex sigma of charts enters when tau drops to f(sigma) = n_min-th largest over rows of min_{i in sigma} w[x, i].
n_min > 1 demands support from several rows, so isolated outlier rows cannot create simplices.
"""
from itertools import combinations

import numpy as np


def _f(w, sigma, n_min):
    m = w[:, list(sigma)].min(1)
    return float(m.max() if n_min == 1 else np.partition(m, len(m) - n_min)[len(m) - n_min])


def nerve_filtration(w, tau_min=1 / 32, max_dim=4, n_min=1):
    """{sorted tuple of chart ids: f}, keeping f >= tau_min and size <= max_dim + 1. w [N, K] (rows inputs)."""
    if hasattr(w, "detach"):
        w = w.detach().cpu()
    w = np.asarray(w, dtype=np.float64)
    if w.shape[0] < n_min:
        return {}
    filt = {}
    for i in range(w.shape[1]):
        f = _f(w, (i,), n_min)
        if f >= tau_min:
            filt[(i,)] = f
    level = sorted(filt)
    for k in range(2, max_dim + 2):
        keep, nxt = set(level), {}
        for a, b in combinations(level, 2):
            if a[:-1] != b[:-1]:
                continue  # sorted level: pairs sharing k-2 vertices are a prefix match
            s = a + (b[-1],)
            if all(s[:j] + s[j + 1:] in keep for j in range(k - 2)):
                f = _f(w, s, n_min)
                if f >= tau_min:
                    nxt[s] = f
        filt.update(nxt)
        level = sorted(nxt)
        if not level:
            break
    return filt


def persistence(filt, max_hom_dim=3):
    """[(dim, birth, death)] with birth >= death; essential bars die at 0.0; zero-length bars dropped.
    Dims <= max_hom_dim only; filt needs simplices up to dim max_hom_dim + 1 for the top dim to be right."""
    order = sorted(filt, key=lambda s: (-filt[s], len(s), s))
    idx = {s: i for i, s in enumerate(order)}
    low, paired, bars = {}, set(), []
    for j, s in enumerate(order):
        col = sorted(idx[s[:i] + s[i + 1:]] for i in range(len(s))) if len(s) > 1 else []
        while col and col[-1] in low:
            col = sorted(set(col) ^ set(low[col[-1]]))  # xor-merge over Z2
        if col:
            low[col[-1]] = col
            paired.update((col[-1], j))
            b, d = order[col[-1]], s
            if filt[b] > filt[d] and len(b) - 1 <= max_hom_dim:
                bars.append((len(b) - 1, filt[b], filt[d]))
    for i, s in enumerate(order):
        if i not in paired and len(s) - 1 <= max_hom_dim:
            bars.append((len(s) - 1, filt[s], 0.0))
    return bars


def betti_at(bars, tau, max_hom_dim=3):
    return tuple(sum(1 for d, b, e in bars if d == k and b >= tau > e) for k in range(max_hom_dim + 1))


def certified_betti(bars, eps, max_hom_dim=3, tau_min=1 / 32, require_connected=True):
    """Stable-interval read -> (vector | None, interval_length, (lo, hi)).

    Between consecutive bar endpoints (essential deaths = tau_min) betti_at is constant; adjacent intervals with
    equal vectors merge. Returns the vector of the LONGEST merged interval with b0 == 1 (if require_connected=True,
    as b0 >= 1 is not enough: six disjoint octahedral charts hold b0 = 6 over ~0.46, longer than the true (1,0,1) window ~0.25),
    its length and bounds; vector is None if that length <= eps. Support guard (n_min) only covers outlier fractions below n_min / N."""
    bars = [(d, b, tau_min if e == 0.0 else e) for d, b, e in bars]
    pts = sorted({tau_min, *(x for _, b, e in bars for x in (b, e)), max([b for _, b, _ in bars], default=tau_min)})
    pts = [x for x in pts if x >= tau_min]
    segs = []
    for lo, hi in zip(pts, pts[1:]):
        v = betti_at(bars, (lo + hi) / 2, max_hom_dim)
        if segs and segs[-1][0] == v:
            segs[-1][2] = hi
        else:
            segs.append([v, lo, hi])
    if require_connected:
        segs = [t for t in segs if t[0][0] == 1]  # connected nerve only: see docstring
    if not segs:
        return None, 0.0, (tau_min, tau_min)
    v, lo, hi = max(segs, key=lambda t: t[2] - t[1])
    return (v if hi - lo > eps else None), hi - lo, (lo, hi)

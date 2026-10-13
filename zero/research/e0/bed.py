"""E0 beds: switching order-1 Markov sources (PROGRAM.md section 6).

x_0 is uniform; x_t ~ P_{r_t}[x_{t-1}] for t >= 1, so r_t is the law that generated the transition into x_t.
"""
import numpy as np

V, T, CONC, SEG_MIN, SEG_MEAN = 16, 20_000, 0.2, 50, 350


def make_bed(seed, kind="REC", K=4, V=V, T=T, conc=CONC, seg_min=SEG_MIN, seg_mean=SEG_MEAN, p_cycle=0.9):
    """kind in {REC, NOREC, NULL, CYC}. CYC: K recurring laws; the next law is (current + 1) mod K with
    probability p_cycle, else uniform among the others. Returns x (T,), r (T,) regime ids, laws list."""
    rng = np.random.default_rng(seed)
    law = lambda: rng.dirichlet(np.full(V, conc), size=V)
    if kind == "NULL":
        laws, bounds, ids = [law()], [0, T], [0]
    else:
        bounds, ids, laws = [0], [], []
        if kind in ("REC", "CYC"):
            laws = [law() for _ in range(K)]
        while bounds[-1] < T:
            if kind == "REC" or (kind == "CYC" and not ids):
                choices = [k for k in range(K) if not ids or k != ids[-1]]
                ids.append(int(rng.choice(choices)))
            elif kind == "CYC":
                nxt = (ids[-1] + 1) % K
                others = [k for k in range(K) if k not in (ids[-1], nxt)]
                ids.append(nxt if rng.random() < p_cycle else int(rng.choice(others)))
            else:
                laws.append(law())
                ids.append(len(laws) - 1)
            bounds.append(min(T, bounds[-1] + seg_min + int(rng.geometric(1.0 / (seg_mean - seg_min)))))
    r = np.empty(T, int)
    for (a, b), k in zip(zip(bounds[:-1], bounds[1:]), ids):
        r[a:b] = k
    x = np.empty(T, int)
    x[0] = rng.integers(V)
    u = rng.random(T)
    cdfs = [np.cumsum(P, axis=1) for P in laws]
    for t in range(1, T):
        x[t] = min(int(np.searchsorted(cdfs[r[t]][x[t - 1]], u[t], side="right")), V - 1)
    return x, r, laws

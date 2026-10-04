"""T6 (optional, measurement only): order effects and the QQ equality for any arm that exposes observer
weights w [B, T, K] from forward(obs, act). Definitions mirror ref/identities.py §2 exactly.

Question X (X in {A, B}) = present observation segment X to the observer after a shared prefix; the answer
is yes iff the observer's top chart (argmax_k w) at the LAST frame of segment X equals chart_X.
chart_X defaults to the modal top chart at the end of X when X is asked first; it is fixed before the two
orders are compared, so it is the same question in both orders.
Order AB: prefix, A, B.  Order BA: prefix, B, A.  Each row = one context (one prefix/segment triple).

Joint tables P[a, b], a = answer to A, b = answer to B (regardless of which was asked first):
  QQ    = P_AB(a != b) - P_BA(a != b)   = (p(A,~B) + p(~A,B)) - (p(B,~A) + p(~B,A))   [Wang & Busemeyer 2013;
                                                                                    Wang et al. PNAS 2014]
  order = P_AB(1, 1)   - P_BA(1, 1)     = p(A,B) - p(B,A)
  dA    = P(a=1 | A first) - P(a=1 | A second); dB likewise: marginal disturbance (not in identities.py; a
          classical sticky observer can move these with QQ = order = 0, so they are reported beside).
A context-free observer (w_t depends on o_t only) scores exactly 0 on all four.
"""
import numpy as np
import torch


def qq_order(P_ab, P_ba):
    """2x2 joint tables -> (QQ, order effect)."""
    P_ab, P_ba = np.asarray(P_ab, float), np.asarray(P_ba, float)
    return (P_ab[1, 0] + P_ab[0, 1]) - (P_ba[1, 0] + P_ba[0, 1]), P_ab[1, 1] - P_ba[1, 1]


def _point(ab, ba):
    """Binary answers [N, 2] (cols a, b) under each order -> qq, order, dA, dB."""
    diff = lambda x: (x[:, 0] != x[:, 1]).mean()
    both = lambda x: (x[:, 0] & x[:, 1]).mean()
    return np.array([diff(ab) - diff(ba), both(ab) - both(ba),
                     ab[:, 0].mean() - ba[:, 0].mean(), ba[:, 1].mean() - ab[:, 1].mean()])


def stats(ab, ba, n_boot=2000, seed=0, alpha=0.05, paired=True):
    """Point estimates + percentile bootstrap CIs. paired=True resamples contexts jointly (same rows in
    both orders, the arm protocol); paired=False resamples each order independently."""
    ab, ba = np.asarray(ab).astype(bool), np.asarray(ba).astype(bool)
    rng = np.random.default_rng(seed)
    pt = _point(ab, ba)
    boot = np.empty((n_boot, 4))
    for i in range(n_boot):
        ia = rng.integers(0, len(ab), len(ab))
        ib = ia if paired else rng.integers(0, len(ba), len(ba))
        boot[i] = _point(ab[ia], ba[ib])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=0)
    keys = ["qq", "order", "dA", "dB"]
    return {**{k: float(v) for k, v in zip(keys, pt)},
            **{k + "_ci": (float(l), float(h)) for k, l, h in zip(keys, lo, hi)}, "n": len(ab)}


@torch.no_grad()
def _top(arm, obs, act):
    return arm(obs, act)["w"].argmax(-1)  # [N, T]


@torch.no_grad()
def answers(arm, pre, A, B, acts=None, charts=None):
    """pre [N,P,D], A [N,L,D], B [N,L,D] (+ optional acts = (act_pre, act_A, act_B)) -> ab, ba int [N, 2]."""
    was = arm.training
    arm.eval()
    try:
        if acts is None:
            z = lambda x: x.new_zeros(*x.shape[:2], getattr(arm, "A", 1))
            acts = (z(pre), z(A), z(B))
        ap, aA, aB = acts
        P, LA, LB = pre.shape[1], A.shape[1], B.shape[1]
        top_ab = _top(arm, torch.cat([pre, A, B], 1), torch.cat([ap, aA, aB], 1))
        top_ba = _top(arm, torch.cat([pre, B, A], 1), torch.cat([ap, aB, aA], 1))
        endA_ab, endB_ab = P + LA - 1, P + LA + LB - 1
        endB_ba, endA_ba = P + LB - 1, P + LB + LA - 1
        if charts is None:  # modal top chart when the question is asked first
            charts = (int(top_ab[:, endA_ab].mode().values), int(top_ba[:, endB_ba].mode().values))
        cA, cB = charts
        ab = torch.stack([top_ab[:, endA_ab] == cA, top_ab[:, endB_ab] == cB], 1)
        ba = torch.stack([top_ba[:, endA_ba] == cA, top_ba[:, endB_ba] == cB], 1)
        return ab.int().cpu().numpy(), ba.int().cpu().numpy()
    finally:
        arm.train(was)


def bed_segments(bed, n, P=8, L=4, seed=0):
    """Contexts from a Bed dict: prefix + A from one trajectory (A continues the prefix), B from another
    trajectory. Returns tensors (pre, A, B) and acts in the same layout."""
    rng = np.random.default_rng(seed)
    N, T, _ = bed["obs"].shape
    i, j = rng.integers(0, N, n), rng.integers(0, N, n)
    t0, t1 = rng.integers(0, T - P - L + 1, n), rng.integers(0, T - L + 1, n)
    take = lambda x, r, t, k: torch.from_numpy(np.stack([x[a, b:b + k] for a, b in zip(r, t)]))
    o, a = bed["obs"], bed["act"]
    return ((take(o, i, t0, P), take(o, i, t0 + P, L), take(o, j, t1, L)),
            (take(a, i, t0, P), take(a, i, t0 + P, L), take(a, j, t1, L)))

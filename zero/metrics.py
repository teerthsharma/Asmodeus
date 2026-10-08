"""T2 metrics. Pure numpy (torch only inside pred_error, which drives an arm).

Every metric is built so that a constant / collapsed latent cannot win:
  pred_error   -> inf (normalised by the h-step persistence error, not only by latent variance)
  discontinuity_rate, fold_rate -> 1.0 (latent distances are relative to the median pairwise distance)
  linear_probe -> <= 0 (held-out TRAJECTORIES, never held-out frames)
"""
import itertools

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# ---------------- Z2 homology ----------------
def closure(simplices, max_size=None):
    out = set()
    for s in simplices:
        s = tuple(sorted(s))
        top = len(s) if max_size is None else min(len(s), max_size)
        for k in range(1, top + 1):
            out.update(itertools.combinations(s, k))
        if len(out) > 2 * MAX_SIMPLICES:                     # stop before memory, _betti returns None
            break
    return out


def boundary_z2(cx, k, signed=False):
    """d_k : C_k -> C_{k-1}, shape [#(k-1)-simplices, #k-simplices]; GF(2) bool, or oriented ints."""
    lo = sorted(s for s in cx if len(s) == k)
    hi = sorted(s for s in cx if len(s) == k + 1)
    idx = {s: i for i, s in enumerate(lo)}
    M = np.zeros((len(lo), len(hi)), dtype=np.int8 if signed else bool)
    for j, s in enumerate(hi):
        for i, f in enumerate(itertools.combinations(s, k)):    # f omits s[k-i]
            M[idx[f], j] = (-1) ** (k - i) if signed else True
    return M


MAX_SIMPLICES = 300_000    # simplices of size <= D+2 above which _betti returns None ("no answer")


def _rank_cols(cols):
    """GF(2) rank of int-bitmask columns: standard reduction keyed by lowest set bit, O(nnz) memory."""
    pivots = {}
    for col in cols:
        while col and (col.bit_length() - 1) in pivots:
            col ^= pivots[col.bit_length() - 1]
        if col:
            pivots[col.bit_length() - 1] = col
    return len(pivots)


def rank_gf2(M):
    M = np.asarray(M, dtype=bool)
    return _rank_cols(sum(1 << int(i) for i in np.flatnonzero(M[:, j])) for j in range(M.shape[1]))


def _rank_z2_sparse(lo, hi, k):
    idx = {s: i for i, s in enumerate(lo)}
    return _rank_cols(sum(1 << idx[f] for f in itertools.combinations(s, k)) for s in hi)


def _betti(cx, max_dim, field):
    top = max((len(s) for s in cx), default=0) - 1
    D = top if max_dim is None else max_dim
    by = {k: [] for k in range(1, D + 3)}
    for s in cx:
        if len(s) in by:
            by[len(s)].append(s)
    if sum(map(len, by.values())) > MAX_SIMPLICES:
        return None
    for v in by.values():
        v.sort()
    n = [len(by[k + 1]) for k in range(D + 2)]
    if field == "z2":
        rank = lambda k: _rank_z2_sparse(by[k], by[k + 1], k)
    elif max(n[k] * n[k + 1] for k in range(D + 1)) > 5e7:  # dense float boundary would exceed ~400 MB
        return None
    else:   # ponytail: float rank of a +-1 matrix, exact for these sizes; use fraction-free elimination if >1e4
        rank = lambda k: int(np.linalg.matrix_rank(boundary_z2(cx, k, signed=True).astype(np.float64)))
    rk = [0] + [rank(k) if n[k] and n[k - 1] else 0 for k in range(1, D + 2)]
    return tuple(int(n[k] - rk[k] - rk[k + 1]) for k in range(D + 1))


def betti_z2(cx, max_dim=None):
    """Z2 Betti numbers of a face-closed set of sorted tuples. max_dim=None -> up to the top dimension.
    Simplices of dim max_dim+1 must be in cx for b_{max_dim} to be right (nerve_betti_z2 handles that)."""
    return _betti(cx, max_dim, "z2")


def betti_q(cx, max_dim=None):
    """Rational Betti numbers: Z2 cannot tell T2 (1,2,1) from Klein (1,2,1); Q gives Klein (1,1,0)."""
    return _betti(cx, max_dim, "q")


def _flag(cx_vertices_edges, max_size):
    V = sorted({s[0] for s in cx_vertices_edges if len(s) == 1})
    nb = {v: set() for v in V}
    for s in cx_vertices_edges:
        if len(s) == 2:
            nb[s[0]].add(s[1]); nb[s[1]].add(s[0])
    out = {(v,) for v in V}
    frontier = [(v,) for v in V]
    while frontier:
        nxt = []
        for s in frontier:
            if len(s) == max_size:
                continue
            common = set.intersection(*(nb[v] for v in s))
            nxt += [s + (w,) for w in common if w > s[-1]]
        out.update(nxt)
        frontier = nxt if len(out) <= 2 * MAX_SIMPLICES else []
    return out


def nerve_betti_z2(w, tau, max_dim=2, flag=False, field="z2"):
    """Nerve of the observer cover: vertex k if some sample has w_k >= tau, simplex S if one sample has
    w_k >= tau for all k in S. With top-m routing no sample ever holds more than m charts, so the true
    nerve has no simplex above dim m-1; flag=True fills cliques of the co-selection graph instead."""
    W = np.asarray(w).reshape(-1, np.shape(w)[-1]) >= tau
    rows = np.unique(W, axis=0)
    sets = [tuple(np.flatnonzero(r)) for r in rows if r.any()]
    if flag:
        cx = _flag(closure(sets, 2), max_dim + 2)
    else:
        cx = closure(sets, max_dim + 2)
    return _betti(cx, max_dim, field)


def most_persistent(seq):
    """Longest contiguous run in a sweep of Betti vectors (empty complexes, b0 = 0, never win)."""
    best, run, prev, cur = None, 0, None, 0
    for b in seq:
        cur = cur + 1 if b == prev else 1
        prev = b
        if b is not None and b[0] > 0 and cur > run:
            best, run = b, cur
    return best, run


def sweep_tau(w, taus, **kw):
    return most_persistent([nerve_betti_z2(w, t, **kw) for t in taus])


# ---------------- Mapper baseline on a single latent ----------------
OVERLAPS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)


def _mapper_seq(latent, filter_dim, max_dim, n_cover, overlaps, n_samples, seed, field, link_k=5, link_mult=2.0):
    """filter = top-`filter_dim` PCA of the latent; cover = balls of ONE radius (1+overlap) x covering radius
    around farthest-point centres in filter space; clusters = single linkage in the FULL latent at
    max(link_mult x median k-NN distance, half the ball radius); complex = nerve over shared points."""
    X = np.asarray(latent, dtype=np.float64).reshape(-1, np.shape(latent)[-1])
    if n_samples is not None and len(X) > n_samples:
        X = X[np.random.default_rng(seed).choice(len(X), n_samples, replace=False)]
    Xc = X - X.mean(0)
    f = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:filter_dim].T
    C = [0]
    dmin = np.linalg.norm(f - f[0], axis=1)
    for _ in range(n_cover - 1):
        C.append(int(np.argmax(dmin)))
        dmin = np.minimum(dmin, np.linalg.norm(f - f[C[-1]], axis=1))
    dc = np.linalg.norm(f[:, None] - f[C][None], axis=-1)
    tree = cKDTree(X)
    r_knn = link_mult * np.median(tree.query(X, link_k + 1)[0][:, -1])
    seq = []
    for ov in overlaps:
        R = (1 + ov) * dc.min(1).max()                     # one ball radius >= covering radius
        member = dc <= R
        # link scale tied to the ball, never to the longest MST edge (that edge can be the one closing a cycle)
        pairs = tree.query_pairs(max(r_knn, 0.5 * R), output_type="ndarray")
        node_of = [[] for _ in range(len(X))]
        nid = 0
        for j in range(len(C)):
            idx = np.flatnonzero(member[:, j])
            if idx.size == 0:
                continue
            loc = -np.ones(len(X), int); loc[idx] = np.arange(idx.size)
            p = pairs[(loc[pairs[:, 0]] >= 0) & (loc[pairs[:, 1]] >= 0)]
            g = coo_matrix((np.ones(len(p)), (loc[p[:, 0]], loc[p[:, 1]])), shape=(idx.size, idx.size))
            nc, lab = connected_components(g, directed=False)
            for i, l in zip(idx, lab):
                node_of[i].append(nid + l)
            nid += nc
        seq.append(_betti(closure({tuple(sorted(s)) for s in node_of}, max_dim + 2), max_dim, field))
    return seq


def mapper_betti(latent, filter_dim, max_dim, n_cover, overlap, n_samples=None, seed=0, field="z2"):
    """Mapper Betti vector at a FROZEN cover (n_cover, overlap) from tune_mapper on the tuning seed.
    n_samples=None uses every frame, i.e. the same sample count as the nerve readout."""
    return _mapper_seq(latent, filter_dim, max_dim, n_cover, (overlap,), n_samples, seed, field)[0]


def tune_mapper(latent, filter_dim, max_dim, n_covers=None, overlaps=OVERLAPS, n_samples=None, seed=0):
    """Label-free cover choice on the TUNING-seed latent: for each n_cover, the longest run of one Betti
    vector over the overlap sweep; keep the n_cover with the longest run, overlap = middle of that run."""
    n_covers = n_covers or tuple(k * filter_dim for k in (8, 12, 16))
    best = None
    for nc in n_covers:
        seq = _mapper_seq(latent, filter_dim, max_dim, nc, overlaps, n_samples, seed, "z2")
        i = 0
        while i < len(seq):
            j = i
            while j + 1 < len(seq) and seq[j + 1] == seq[i]:
                j += 1
            if seq[i] is not None and seq[i][0] > 0 and (best is None or j - i > best[0]):
                best = (j - i, nc, overlaps[(i + j) // 2])
            i = j + 1
    return None if best is None else dict(n_cover=int(best[1]), overlap=float(best[2]))


# ---------------- geometry metrics ----------------
def _flat(g, state, n_max, seed):
    g = np.asarray(g, dtype=np.float64).reshape(-1, np.shape(g)[-1])
    s = np.asarray(state, dtype=np.float64).reshape(-1, np.shape(state)[-1])
    if len(g) > n_max:
        i = np.random.default_rng(seed).choice(len(g), n_max, replace=False)
        g, s = g[i], s[i]
    return g, s


def _pairs(s, geo, keep, chunk=256):
    # Optimize distance computation by evaluating upper-triangular sub-matrices only:
    # compare s[a:a+chunk] against s[a:] instead of full s, cutting redundant calculations by ~half.
    I, J = [], []
    n = len(s)
    for a in range(0, n, chunk):
        sub_s = s[a:]
        d = geo(s[a:a + chunk, None], sub_s[None])
        i, j = np.nonzero(keep(d))
        m = i < j
        I.append(i[m] + a)
        J.append(j[m] + a)
    if not I:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(I), np.concatenate(J)


def _rel_latent(g, I, J):
    med = np.median(pdist(g))
    if not med > 0:
        return None
    c = max(1, 2 ** 24 // g.shape[1])                       # bounded memory for wide latents / pixels
    return np.concatenate([np.linalg.norm(g[I[k:k + c]] - g[J[k:k + c]], axis=1)
                           for k in range(0, len(I), c)] or [np.zeros(0)]) / med


def discontinuity_rate(g, state, geo, eps, delta, n_max=4000, seed=0):
    """Fraction of state pairs within geodesic eps whose latents differ by > delta * median pairwise
    latent distance. Tears only: a continuous but folded (non-injective) latent scores 0 -> see fold_rate."""
    g, s = _flat(g, state, n_max, seed)
    I, J = _pairs(s, geo, lambda d: d < eps)
    if len(I) == 0:
        raise ValueError("no state pairs within eps")
    r = _rel_latent(g, I, J)
    return 1.0 if r is None else float((r > delta).mean())


def fold_rate(g, state, geo, far, delta, n_max=4000, seed=0):
    """Fraction of POINTS that have a latent neighbour (within delta * median pairwise latent distance)
    lying more than `far` away on the manifold. Catches continuous non-injective latents (folds, collapse).
    Per point, not per pair: a fold is a measure-zero set of pairs but a positive fraction of points."""
    g, s = _flat(g, state, n_max, seed)
    I, J = _pairs(s, geo, lambda d: d > far)
    r = _rel_latent(g, I, J)
    if r is None:
        return 1.0
    hit = np.zeros(len(g), bool)
    hit[I[r < delta]] = True; hit[J[r < delta]] = True
    return float(hit.mean())


def effective_rank(z):
    z = np.asarray(z, dtype=np.float64).reshape(-1, np.shape(z)[-1])
    s = np.linalg.svd(z - z.mean(0), compute_uv=False)
    if not s.sum() > 1e-12 * max(1.0, np.abs(z).max()):
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def linear_probe(g, state, train_frac=0.8, ridge=1e-3):
    """Ridge R^2 of state from g, trained on the first 80% of TRAJECTORIES, scored on the rest."""
    g, state = np.asarray(g, np.float64), np.asarray(state, np.float64)
    n = int(round(train_frac * g.shape[0]))
    Xtr, Xte = g[:n].reshape(-1, g.shape[-1]), g[n:].reshape(-1, g.shape[-1])
    Ytr, Yte = state[:n].reshape(-1, state.shape[-1]), state[n:].reshape(-1, state.shape[-1])
    mx, my = Xtr.mean(0), Ytr.mean(0)
    A = (Xtr - mx).T @ (Xtr - mx)
    lam = ridge * max(np.trace(A) / A.shape[0], 1e-12)
    Wt = np.linalg.solve(A + lam * np.eye(A.shape[0]), (Xtr - mx).T @ (Ytr - my))
    pred = (Xte - mx) @ Wt + my
    sst = ((Yte - Yte.mean(0)) ** 2).sum()
    return float(1 - ((Yte - pred) ** 2).sum() / sst)


def pred_error_from(pred, true, now):
    """nvar = MSE / Var(true)  (spec; fooled by a latent that never moves in time)
    npersist = MSE / MSE(copy-last)  (primary: 1.0 = no better than predicting no change)."""
    pred, true, now = (np.asarray(x, np.float64) for x in (pred, true, now))
    err = ((pred - true) ** 2).sum(-1).mean()
    var = true.reshape(-1, true.shape[-1]).var(0).sum()
    per = ((true - now) ** 2).sum(-1).mean()
    f = lambda a, b: float(a / b) if b > 0 else np.inf
    return dict(mse=float(err), nvar=f(err, var), npersist=f(err, per))


def pred_error(arm, bed, h, stride=8, batch=64, device="cpu"):
    import torch
    obs, act = bed["obs"], bed["act"]
    T = obs.shape[1]
    starts = list(range(0, T - h, stride))
    P, Tr, Nw = [], [], []
    arm.eval()
    with torch.no_grad():
        for b in range(0, obs.shape[0], batch):
            o = torch.as_tensor(obs[b:b + batch], device=device)
            a = torch.as_tensor(act[b:b + batch], device=device)
            g = arm(o, a)["g"]
            for s in starts:
                P.append(arm.rollout(o[:, s], a[:, s:s + h])[:, h - 1].cpu().numpy())
                Tr.append(g[:, s + h].cpu().numpy()); Nw.append(g[:, s].cpu().numpy())
    return pred_error_from(np.concatenate(P), np.concatenate(Tr), np.concatenate(Nw))


def chart_effective_rank(z, zidx=None):
    """Per-chart effective rank from the contract's optional z: [B,T,m,d_c] with zidx [B,T,m] (chart ids
    of the top-m charts), or [B,T,d] for a single latent (reported as chart 0). Charts never selected
    are absent; a chart selected on < 2 frames scores 0."""
    z = np.asarray(z, np.float64)
    if zidx is None:
        return {0: effective_rank(z)}
    zf, kf = z.reshape(-1, z.shape[-1]), np.asarray(zidx).reshape(-1)
    return {int(k): (effective_rank(zf[kf == k]) if (kf == k).sum() > 1 else 0.0) for k in np.unique(kf)}


def chart_transitions(w, z, zidx, min_pairs=20):
    """Orientation sign of every co-observed chart pair. w [..., K] dense weights, z [..., m, dz] per-chart
    coordinates of the top-m charts, zidx [..., m] their chart ids. For pair (j<k) seen together with w>0 for
    both on >= min_pairs rows: least-squares affine fit z_k ~ A z_j + b; value = int(sign(det A))."""
    z = np.asarray(z, np.float64); zidx = np.asarray(zidx)
    w = np.asarray(w).reshape(-1, np.shape(w)[-1])
    m, dz = z.shape[-2:]
    z, zidx = z.reshape(-1, m, dz), zidx.reshape(-1, m)
    live = np.take_along_axis(w, zidx, 1) > 0
    out = {}
    for a, b in itertools.combinations(range(m), 2):
        ok = live[:, a] & live[:, b] & (zidx[:, a] != zidx[:, b])
        lo = np.where(zidx[:, a] < zidx[:, b], a, b); hi = np.where(zidx[:, a] < zidx[:, b], b, a)
        pj, pk = zidx[np.arange(len(zidx)), lo], zidx[np.arange(len(zidx)), hi]
        for j, k in {(int(x), int(y)) for x, y in zip(pj[ok], pk[ok])}:
            s = ok & (pj == j) & (pk == k)
            out.setdefault((j, k), []).append((z[s, lo[s]], z[s, hi[s]]))
    res = {}
    for key, parts in out.items():
        zj = np.concatenate([p[0] for p in parts]); zk = np.concatenate([p[1] for p in parts])
        if len(zj) < min_pairs:
            continue
        # ponytail: overlap assumed connected (an overlap with 2 components of opposite orientation needs per-component fits)
        coef = np.linalg.lstsq(np.c_[zj, np.ones(len(zj))], zk, rcond=None)[0]
        res[key] = int(np.sign(np.linalg.det(coef[:dz].T)))
    return res


def orientation_holonomy(trans, loop_w):
    """+-1 orientation holonomy of a closed loop of weights [L, K]: product of chart-transition signs at every
    change of dominant chart (closing t=L-1 -> 0); None if a needed transition was never fitted."""
    c = np.asarray(loop_w).argmax(-1)
    s = 1
    for a, b in zip(c, np.roll(c, -1)):
        if a != b:
            t = trans.get((int(min(a, b)), int(max(a, b))))
            if t is None:
                return None
            s *= t
    return int(s)

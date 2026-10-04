"""T4b bed: OGBench pointmaze, MoP-JEPA (arXiv 2607.05238v1) protocol. Needs ogbench==1.2.1 (venv ~/.venvs/ogbench;
pip would otherwise lift system mujoco 3.11 -> 3.14).

Two protocols, both from v1:
  graph   planAll / realroute / precision on the maze CELL graph (the paper's 0.851 headline is planAll, BFS over
          the model's proposed graph, NOT closed-loop success). Nodes = free cells; a transition = consecutive
          distinct cells along a dataset episode; 20% of unique directed edges withheld (source support kept);
          edge c->c' proposed when cos(head, latent(c')) > .5 (v3 S10); router gate pi_k > 0.5/K.
  closed  evaluate(): official OGBench goal-conditioned loop, info['success'] at the final step,
          5 tasks x episodes_per_task, actions clipped to [-1, 1], 1000-step limit, terminate at goal.

Pointmaze observations ARE the state (qpos = xy), so obs == state (meta['obs_is_state']). The free space is a
planar domain: one chart suffices (atlas = 1); betti_z2 = (components, interior wall islands).
"""
import hashlib
import os
import urllib.request
from collections import deque
from functools import lru_cache

import numpy as np

DATA_DIR = os.path.expanduser("~/.ogbench/data")
URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"
SHA256 = {  # pinned 2026-10-04
    "pointmaze-medium-stitch-v0": "eb265c0d120935476ff7dce49b34508ead72c544cb59bec33ccd175dc48943ad",
    "pointmaze-medium-stitch-v0-val": "230a50b9cb4f0cb0d64553c731f0e67edec5f131df5960aaa61e910462f40657",
    "pointmaze-large-stitch-v0": "9add335e598e48ebc483447d61415a9755ffb738ddfc406cb9708b2a238992e8",
    "pointmaze-large-stitch-v0-val": "32e2c21a4d71b9cb5f21b0e2d6cf9f7d3b3bb40ca293bfb5209a6f5e7359827a",
    "pointmaze-teleport-navigate-v0": "6d53ce5af9a591392a73d9f303940793769279e5dbfe9ed6ce7e0bccde57a63b",
    "pointmaze-teleport-navigate-v0-val": "43bacdaeb8f5a7126c4a66d4420350dab3d00d5c7d0ed824a5bbf1dc0e7a4d65",
}
NAMES = ["pointmaze-medium-stitch-v0", "pointmaze-large-stitch-v0", "pointmaze-teleport-navigate-v0"]


# ---------- data ----------
def _fetch(fname):
    """ogbench.download_datasets renames a still-open file and dies on Windows (WinError 32); fetch ourselves."""
    path = os.path.join(DATA_DIR, fname + ".npz")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(f"{URL}/{fname}.npz", path + ".tmp")
        os.replace(path + ".tmp", path)
    with open(path, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    if h != SHA256[fname]:
        raise ValueError(f"{fname}: sha256 {h} != pinned {SHA256[fname]}")
    return path


def make_env(name):
    import ogbench
    return ogbench.make_env_and_datasets(name, env_only=True)


@lru_cache(None)
def maze(name):
    """(maze_map [H,W] 1=wall, unit, off_x, off_y, task (init_ij, goal_ij) list, teleport info or None)."""
    u = make_env(name).unwrapped
    tasks = tuple((tuple(t["init_ij"]), tuple(t["goal_ij"])) for t in u.task_infos)
    tp = u._teleport_info
    tp = None if tp is None else (tuple(map(tuple, tp["teleport_in_ijs"])), tuple(map(tuple, tp["teleport_out_ijs"])))
    return np.array(u.maze_map), u._maze_unit, u._offset_x, u._offset_y, tasks, tp


def xy_to_ij(name, xy):
    _, unit, ox, oy, *_ = maze(name)
    xy = np.asarray(xy)
    return np.stack([np.floor((xy[..., 1] + oy + 0.5 * unit) / unit),
                     np.floor((xy[..., 0] + ox + 0.5 * unit) / unit)], -1).astype(int)


def ij_to_xy(name, ij):
    _, unit, ox, oy, *_ = maze(name)
    ij = np.asarray(ij, dtype=float)
    return np.stack([ij[..., 1] * unit - ox, ij[..., 0] * unit - oy], -1)


def betti(name):
    from scipy import ndimage
    m = maze(name)[0]
    b0 = ndimage.label(m == 0)[1]                                   # free cells, 4-connected
    lab, n = ndimage.label(m == 1, structure=np.ones((3, 3)))       # walls, 8-connected
    border = set(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]) - {0}
    return (b0, n - len(border))


def make_bed(name, split="train", n_traj=None, seed=0):
    d = np.load(_fetch(name + ("-val" if split == "val" else "")))
    o, a, term = d["observations"], d["actions"], d["terminals"]
    L = int(np.argmax(term > 0)) + 1
    assert len(o) % L == 0 and (np.flatnonzero(term > 0) % L == L - 1).all(), "non-uniform episodes"
    o, a = o.reshape(-1, L, o.shape[-1]), a.reshape(-1, L, a.shape[-1]).copy()
    a[:, -1] = 0                    # raw files store L+1 states; the last action has no successor (OGBench drops it)
    if n_traj is not None:
        idx = np.sort(np.random.default_rng(seed).choice(len(o), n_traj, replace=False))
        o, a = o[idx], a[idx]
    D = _cell_dist(name)
    unit = maze(name)[1]

    def geo(s1, s2):
        """Maze geodesic at cell resolution (undirected, teleports ignored), in world units. EVAL ONLY."""
        c1, c2 = xy_to_ij(name, s1), xy_to_ij(name, s2)
        return unit * D[c1[..., 0], c1[..., 1], c2[..., 0], c2[..., 1]]

    return dict(obs=o.astype(np.float32), act=a.astype(np.float32), state=o.astype(np.float64), geo=geo,
                meta=dict(name=name, n=2, atlas=1, betti_z2=betti(name), seed=seed, obs="coords", split=split,
                          obs_is_state=True))


# ---------- cell graph ----------
@lru_cache(None)
def true_edges(name):
    """Directed real transitions: 4-neighbour free cells, plus teleport in-cell -> every out-cell."""
    m, *_, tp = maze(name)
    free = {tuple(c) for c in np.argwhere(m == 0)}
    E = {((i, j), (i + di, j + dj)) for i, j in free for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
         if (i + di, j + dj) in free}
    if tp:
        E |= {(a, b) for a in tp[0] for b in tp[1]}
    return frozenset(E)


@lru_cache(None)
def _cell_dist(name):
    m = maze(name)[0]
    H, W = m.shape
    D = np.full((H, W, H, W), np.inf)
    adj = {}
    for a, b in true_edges(name):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
            adj.setdefault(a, []).append(b)
    for s in adj:
        for t, d in _bfs(adj, s).items():
            D[s + t] = d
    return D


def _bfs(adj, s):
    dist, q = {s: 0}, deque([s])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _cells(bed):
    return [tuple(map(tuple, c)) for c in xy_to_ij(bed["meta"]["name"], bed["state"])]


def data_edges(bed):
    """Unique directed cell transitions (consecutive distinct cells) in the bed's episodes. Uses state: EVAL/graph only."""
    E = set()
    for ep in _cells(bed):
        E |= {(a, b) for a, b in zip(ep[:-1], ep[1:]) if a != b}
    return E


def holdout_edges(E, frac=0.2, seed=0):
    """Withhold round(frac*|E|) edges, never a source cell's last remaining edge. -> (train, held)."""
    order = sorted(E)
    rng = np.random.default_rng(seed)
    left = {}
    for a, _ in order:
        left[a] = left.get(a, 0) + 1
    held = set()
    for k in rng.permutation(len(order)):
        if len(held) == round(frac * len(E)):
            break
        a, b = order[k]
        if left[a] > 1:
            held.add((a, b)); left[a] -= 1
    return set(E) - held, held


def cell_transitions(bed, exclude=frozenset(), context="cell"):
    """Training pairs for action-marginalised predictors: (ctx_xy [M,2], next_xy [M,2], src, dst).
    context='cell' -> source-cell centre (branching is real: ~2.1 successors per context);
    context='exit' -> the last raw coordinate before the change (successor nearly determined: no branching).
    Which one v1 used is unpublished; only 'cell' reproduces its dense-predictor collapse story."""
    name = bed["meta"]["name"]
    C = xy_to_ij(name, bed["state"])
    ch = (C[:, 1:] != C[:, :-1]).any(-1)
    n, t = np.nonzero(ch)
    src, dst = C[n, t], C[n, t + 1]
    keep = np.array([(tuple(a), tuple(b)) not in exclude for a, b in zip(src, dst)], bool)
    ctx = ij_to_xy(name, src) if context == "cell" else bed["state"][n, t]
    return ctx[keep], ij_to_xy(name, dst)[keep], src[keep], dst[keep]


def propose_edges(cells, cell_lat, heads, pi=None, thr=0.5):
    """cells [C] (i,j); cell_lat [C,d]; heads [C,K,d] predicted successors per source cell; pi [C,K] or None.
    Edge c->c' iff some (router-active, pi > 0.5/K) head has cosine > thr to latent(c'). Self-loops dropped."""
    lat = cell_lat / np.linalg.norm(cell_lat, axis=-1, keepdims=True)
    h = heads / np.linalg.norm(heads, axis=-1, keepdims=True)
    sim = np.einsum("ckd,ed->cke", h, lat) > thr
    if pi is not None:
        sim &= (np.asarray(pi) > 0.5 / heads.shape[1])[..., None]
    hit = sim.any(1)
    return {(cells[a], cells[b]) for a, b in zip(*np.nonzero(hit)) if a != b}


def decode_edges(arm, cells, xy, train_xy, rule="centred", gate=False, thr=0.5):
    """Edges from any arm or control. rule='centred' (PRIMARY): subtract the arm's mean latent over the training
    points (embed(train_xy), both pair sides) from embeddings and heads, then cos > thr: invariant to a constant
    latent offset. rule='paper' (SECONDARY, v1/v3): raw cos > thr, which offset-heavy latents inflate.
    gate=True keeps heads with pi > 0.5/K."""
    if rule not in ("centred", "paper"):
        raise ValueError(rule)
    lat, (heads, pi) = arm.embed(xy), arm.predict(xy)
    mu = arm.embed(train_xy).mean(0) if rule == "centred" else 0.0
    return propose_edges(cells, lat - mu, heads - mu, pi if gate else None, thr)


def sample_queries(name, n=300, seed=0):
    """Up to n truly reachable (start, goal) cell pairs, sampled without replacement (v3 S10: 'up to 300')."""
    adj = {}
    for a, b in true_edges(name):
        adj.setdefault(a, []).append(b)
    pairs = sorted((s, g) for s in sorted(adj) for g in _bfs(adj, s) if g != s)
    idx = np.random.default_rng(seed).choice(len(pairs), min(n, len(pairs)), replace=False)
    return [pairs[i] for i in idx]


def official_queries(name):
    return list(maze(name)[4])


def graph_metrics(pred, name, queries):
    """planAll: BFS finds a path in the proposal. realroute: path survives after keeping only real edges.
    precision: macro mean over proposing sources of the real fraction. realroute alone is freeloadable
    (propose every edge -> 1.0): always report precision beside it."""
    pred = {(a, b) for a, b in pred if a != b}
    T = true_edges(name)
    real = pred & T

    def frac(E):
        adj = {}
        for a, b in E:
            adj.setdefault(a, []).append(b)
        return float(np.mean([g in _bfs(adj, s) for s, g in queries]))

    src = {}
    for a, b in pred:
        src.setdefault(a, []).append((a, b) in T)
    prec = float(np.mean([np.mean(v) for v in src.values()])) if src else 0.0
    return dict(planAll=frac(pred), realroute=frac(real), precision=prec)


# ---------- closed loop ----------
def clip(a):
    return np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)


def evaluate(policy_fn, name, episodes_per_task=50, seed=0, tasks=None):
    """Official OGBench loop (impls/utils/evaluation.py): reset(options=task_id), goal = info['goal'],
    act until terminated|truncated, score = final-step info['success']. Paper rule: 50 per task (repo default 20).
    The env draws init/goal noise and teleport targets from GLOBAL np.random: seeded here, else irreproducible.
    policy_fn(ob [2], goal [2]) -> action [2]."""
    env = make_env(name)
    np.random.seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    tasks = tasks or range(1, len(maze(name)[4]) + 1)
    per, lengths = {}, []
    for t in tasks:
        s = []
        for _ in range(episodes_per_task):
            ob, info = env.reset(options=dict(task_id=t))
            if hasattr(policy_fn, "reset"):
                policy_fn.reset()
            goal, done, k = info["goal"], False, 0
            while not done:
                ob, _, term, trunc, info = env.step(clip(policy_fn(ob, goal)))
                done, k = term or trunc, k + 1
            s.append(float(info["success"])); lengths.append(k)
        per[t] = float(np.mean(s))
    return dict(success=float(np.mean(list(per.values()))), per_task=per, n=episodes_per_task * len(per),
                lengths=lengths)


class executor:
    """Replanning executor over a proposed cell graph (v1's is unpublished; this is the reconstruction).
    Each step: BFS on (edges - removed) from the current cell to the goal cell, steer at full speed to the next
    hop's centre (non-adjacent hop = teleport: steer to own centre to trigger it). replan=True: a hop not reached
    within `patience` steps in one cell is deleted and the route recomputed. evaluate() calls reset() per episode."""

    def __init__(self, name, edges, replan=True, patience=40):
        self.name, self.E, self.replan, self.patience = name, {(a, b) for a, b in edges if a != b}, replan, patience
        self.reset()

    def reset(self):
        self.removed, self.cell, self.t_in, self.path = set(), None, 0, None

    def _route(self, c, g):
        adj = {}
        for a, b in sorted(self.E - self.removed):
            adj.setdefault(a, []).append(b)
        prev, q = {c: None}, deque([c])
        while q:
            u = q.popleft()
            for v in adj.get(u, ()):
                if v not in prev:
                    prev[v] = u; q.append(v)
        if g not in prev:
            return None
        path = [g]
        while path[-1] != c:
            path.append(prev[path[-1]])
        return path[::-1]

    def __call__(self, ob, goal):
        c, g = tuple(map(int, xy_to_ij(self.name, ob))), tuple(map(int, xy_to_ij(self.name, goal)))
        if c != self.cell:
            self.cell, self.t_in, self.path = c, 0, None
        else:
            self.t_in += 1
        if c == g:
            target = np.asarray(goal, float)
        else:
            if self.replan and self.path and self.t_in > self.patience:
                self.removed.add((c, self.path[1])); self.t_in, self.path = 0, None
            if self.path is None:
                self.path = self._route(c, g)
            if self.path is None:
                return np.zeros(2)
            h = self.path[1]
            target = ij_to_xy(self.name, h if abs(h[0] - c[0]) + abs(h[1] - c[1]) == 1 else c)
        v = target - ob
        return v / max(np.abs(v).max(), 1e-9)


def oracle_policy(name):
    """Executor on the true graph (teleports optimistic). Reference ceiling, not an arm."""
    return executor(name, true_edges(name))

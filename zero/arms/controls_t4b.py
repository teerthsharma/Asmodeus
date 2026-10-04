"""T4b graph-protocol controls (PREREG_DRAFT_t4b §6). Not learned arms: references every learned arm must beat.

Interface (shared with learned arms in the graph protocol):
  fit(ctx [M,2], nxt [M,2]) -> self      training pairs from ogbench_bed.cell_transitions (holdout excluded)
  embed(xy [C,2]) -> [C,d]               latent used for edge decoding (cos > .5)
  predict(ctx [C,2]) -> (heads [C,K,d], pi [C,K])
Controls embed in a fixed one-hot over the distinct points seen in training (so cos > .5 <=> same point);
they get only the fit() pairs, never the maze map or held-out edges.
  Memoriser   heads = exactly the successors seen for that context (pi = empirical frequency)
  ProposeAll  heads = every seen point (pi uniform): realroute freeloader, precision floor
  Codebook    input-agnostic K-means of all targets in COORDINATE space (run_t4b name 'codebook_xy'): a labelled
              extra. v1's codebook control is in latent space: arms/t4b_adapters.CodebookGraph ('codebook_latent').
"""
import numpy as np


class _OneHot:
    def _index(self, xy):
        d = np.linalg.norm(np.asarray(xy, float)[:, None] - self.pts[None], axis=-1)
        return d.argmin(1)

    def _set_points(self, ctx, nxt):
        self.pts = np.unique(np.concatenate([ctx, nxt]).astype(float), axis=0)

    def embed(self, xy):
        return np.eye(len(self.pts))[self._index(xy)]


class Memoriser(_OneHot):
    def fit(self, ctx, nxt):
        self._set_points(ctx, nxt)
        P = len(self.pts)
        self.counts = np.zeros((P, P))
        np.add.at(self.counts, (self._index(ctx), self._index(nxt)), 1)
        self.K = int((self.counts > 0).sum(1).max())
        return self

    def predict(self, ctx):
        C = self.counts[self._index(ctx)]                       # [C, P]
        order = np.argsort(-C, 1, kind="stable")[:, :self.K]
        w = np.take_along_axis(C, order, 1)
        order = np.where(w > 0, order, order[:, :1])            # pad with the top successor, weight 0
        pi = w / np.maximum(w.sum(1, keepdims=True), 1)
        return np.eye(len(self.pts))[order], pi


class ProposeAll(_OneHot):
    def fit(self, ctx, nxt):
        self._set_points(ctx, nxt)
        return self

    def predict(self, ctx):
        P, C = len(self.pts), len(ctx)
        return np.broadcast_to(np.eye(P), (C, P, P)).copy(), np.full((C, P), 1 / P)


class Codebook(_OneHot):
    def __init__(self, K=8, seed=0, iters=50):
        self.K, self.seed, self.iters = K, seed, iters

    def fit(self, ctx, nxt):
        self._set_points(ctx, nxt)
        x = np.asarray(nxt, float)
        rng = np.random.default_rng(self.seed)
        c = x[[rng.integers(len(x))]]
        while len(c) < self.K:                                   # k-means++
            d = np.min(((x[:, None] - c[None]) ** 2).sum(-1), 1)
            c = np.vstack([c, x[rng.choice(len(x), p=d / d.sum())]])
        for _ in range(self.iters):
            lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
            c = np.stack([x[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(self.K)])
        self.codes = self.embed(c)                               # [K, P]
        self.pi = np.bincount(lab, minlength=self.K) / len(x)
        return self

    def predict(self, ctx):
        C = len(ctx)
        return np.broadcast_to(self.codes, (C,) + self.codes.shape).copy(), np.broadcast_to(self.pi, (C, self.K)).copy()

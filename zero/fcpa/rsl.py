"""RSL, recursive self-learning (spec 3.10): grow eras on evidence, never disturb existing routing, train only on
grounded data, retire by Shapley value.

L23/L23'' non-destructive birth along the Ky Fan direction; L24* anytime-valid trigger (mixture-over-change-time e-detector,
H3, never reads ACI's alpha_t); L25' provenance tags (scope: no generated content in era training; NOT an invariance claim, since
births fire on the router's evidence); L26 retire by Shapley with v(empty) = 0 (leave-one-out retires both of two
duplicate eras, Shapley does not).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .edetector import MixtureBirthTrigger
from .observer import birth_direction, split_null


class GroundingError(ValueError):
    """Raised when non-grounded (generated) data is offered to a birth or an era update."""


@dataclass(frozen=True)
class Tagged:
    """Rows X (n, d) with a per-row provenance tag in `src` ('slab' = grounded data, 'tt' = generated)."""
    X: np.ndarray
    t: np.ndarray
    src: np.ndarray
    slab: Optional[tuple] = None

    def __len__(self):
        return len(self.X)


def ground(X, t, t0, t1):
    """The rows of X with t0 <= t < t1, tagged 'slab' (the only data eras may train on)."""
    X, t = np.asarray(X, dtype=float), np.asarray(t)
    m = (t >= t0) & (t < t1)
    return Tagged(X[m], t[m], np.full(int(m.sum()), "slab"), (t0, t1))


def tt_output(Y):
    """Generated rows (TT tokens), tagged 'tt'.  They can be logged but never admitted into an era."""
    Y = np.asarray(Y, dtype=float)
    return Tagged(Y, np.full(len(Y), np.nan), np.full(len(Y), "tt"))


def concat(a, b):
    """Row-wise union; the tags travel with the rows, so one generated row taints the whole batch for admission."""
    return Tagged(np.vstack([a.X, b.X]), np.concatenate([a.t, b.t]), np.concatenate([a.src, b.src]), None)


class Era:
    """Additive sufficient statistics (n, sum, outer sum) of the era's grounded slab data."""

    def __init__(self, index, slab, d):
        self.index, self.slab = index, slab
        self.n, self.s1, self.s2 = 0, np.zeros(d), np.zeros((d, d))

    def add(self, X):
        self.n += len(X)
        self.s1 = self.s1 + X.sum(0)
        self.s2 = self.s2 + X.T @ X

    @property
    def mean(self):
        return self.s1 / self.n if self.n else np.zeros_like(self.s1)  # a fresh array: callers cannot mutate the era

    @property
    def cov(self):
        if not self.n:
            return np.zeros_like(self.s2)
        m = self.s1 / self.n
        return self.s2 / self.n - np.outer(m, m)


def _all_values(value_fn, n):
    return {frozenset(S): value_fn(frozenset(S)) for r in range(n + 1) for S in itertools.combinations(range(n), r)}


def shapley(value_fn, n, n_perm=None, rng=None):
    """Shapley values of n players for value_fn(frozenset) -> float.  Efficiency: sum = v(N) - v(empty).

    Exact (all subsets) for n <= 10 when n_perm is None; otherwise permutation sampling, which keeps
    efficiency exactly (each sampled permutation telescopes).  n > 10 without n_perm raises ValueError.
    """
    if n == 0:
        return np.zeros(0)
    if n_perm is None:
        if n > 10:
            raise ValueError("exact Shapley is limited to n <= 10; pass n_perm for sampling")
        v = _all_values(value_fn, n)
        phi = np.zeros(n)
        for i in range(n):
            for S, vS in v.items():
                if i in S:
                    continue
                k = len(S)
                phi[i] += math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) * (v[S | {i}] - vS)
        return phi
    g = np.random.default_rng(rng)
    phi, v_empty = np.zeros(n), value_fn(frozenset())
    for _ in range(n_perm):
        S, prev = frozenset(), v_empty
        for i in g.permutation(n):
            S = S | {int(i)}
            cur = value_fn(S)
            phi[i] += cur - prev
            prev = cur
    return phi / n_perm


def leave_one_out(value_fn, n):
    """v(N) - v(N \\ {i}).  Cheap, but gives both of two duplicate players value 0."""
    full = frozenset(range(n))
    vf = value_fn(full)
    return np.array([vf - value_fn(full - {i}) for i in range(n)])


class RSL:
    """Holds the effect list (eras..., null last) and grows/retires it.

    `consider_birth(p, data)` is called once per stream step with the step's novelty p-value (conformal
    `novelty_pvalues`); the BirthTrigger decides.  `data` must be grounded (L25') and is checked on every call.
    """

    def __init__(self, effects, delta=0.01, kappa=0.5, n_max=64, n_min=1):
        self.effects = list(effects)
        self.delta, self.j = delta, 1  # j-th birth is tested at delta_j = delta / (j (j + 1))
        self.trigger = MixtureBirthTrigger(delta / 2, (kappa,))
        self.n_max, self.n_min = n_max, n_min
        self.eras = []  # born eras (those with grounded data), each with .index into self.effects
        self.steps = self.births = self.starved = self.refused_cap = self.retired = 0
        self.dataflow = []  # (src, dst) edges
        self.tt_log = []

    # -- grounding ------------------------------------------------------------------------------------------
    def _admit(self, data):
        if not isinstance(data, Tagged) or (data.src != "slab").any():
            raise GroundingError("only grounded slab data may enter era training (L25')")

    def log_tt_output(self, out):
        """The TT's own tokens go to a log that no training path reads."""
        self.tt_log.append(out)
        self.dataflow.append(("tt", "tt_log"))

    def assert_grounded(self):
        for src, dst in self.dataflow:
            if dst.startswith("era") and not src.startswith("slab"):
                raise GroundingError(f"edge {src} -> {dst} feeds an era from a non-slab source")

    def _era(self, k):
        for e in self.eras:
            if e.index == k:
                return e
        raise IndexError(f"no born era with index {k}")

    # -- birth ----------------------------------------------------------------------------------------------
    def consider_birth(self, p, data, Q=None, psis=None, r=1):
        """Returns the new era's index, or None (no evidence, starved of data, or at the cap)."""
        self.steps += 1
        self._admit(data)
        if not self.trigger.update(p):
            return None
        self.j += 1
        self.trigger.reset(self.delta / (self.j * (self.j + 1)))  # fresh mixture for the next birth
        if len(self.eras) >= self.n_max:
            self.refused_cap += 1
            return None
        if len(data) < self.n_min:
            self.starved += 1
            return None
        E_null = self.effects[-1]
        N = E_null.shape[0]
        if Q is None:
            if psis is not None:
                psis = np.atleast_2d(np.asarray(psis, dtype=complex))
                Q = birth_direction(E_null, psis.T @ psis.conj(), r)  # C = sum psi psi^H
            else:
                Q = 0.5 * np.eye(N)
        self.effects = split_null(self.effects, Q)
        era = Era(len(self.effects) - 2, data.slab, data.X.shape[1])
        era.add(data.X)
        self.eras.append(era)
        self.births += 1
        self.dataflow.append((f"slab:{data.slab}", f"era:{era.index}"))
        return era.index

    def update(self, k, data):
        """Add grounded slab rows to born era k."""
        self._admit(data)
        era = self._era(k)
        era.add(data.X)
        self.dataflow.append((f"slab:{data.slab}", f"era:{k}"))

    # -- retire ---------------------------------------------------------------------------------------------
    def retire(self, value_fn, threshold, method="shapley", rng=None):
        """Merge eras whose value (with v(empty) = 0) is below `threshold` into the null effect.

        value_fn(frozenset of current effect indices) -> validation gain.  method 'shapley' (default, the
        decision rule) or 'loo' (the cheap check).  Survivors' effects are untouched; returns retired indices.
        """
        n = len(self.effects) - 1
        v0 = value_fn(frozenset())
        vf = lambda S: value_fn(S) - v0
        if method == "loo":
            phi = leave_one_out(vf, n)
        elif method == "shapley":
            phi = shapley(vf, n) if n <= 10 else shapley(vf, n, n_perm=200, rng=rng)
        else:
            raise ValueError(f"unknown method {method!r}")
        gone = [i for i in range(n) if phi[i] < threshold]
        if gone:
            keep = [i for i in range(n) if i not in gone]
            null = self.effects[-1] + sum(self.effects[i] for i in gone)
            self.effects = [self.effects[i] for i in keep] + [null]
            new_index = {old: new for new, old in enumerate(keep)}
            self.eras = [e for e in self.eras if e.index in new_index]
            for e in self.eras:
                e.index = new_index[e.index]
            self.retired += len(gone)
        return gone

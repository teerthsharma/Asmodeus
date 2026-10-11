"""Differentiator (spec 3.4-3.5): dE = L_k (-) L_j, addressing by law class, unitary crossing, holonomy.

L13 delta (cocycle), L14' law_classes / address / HMMFilter, L15 crossing composition, L16 holonomy +
commutator_estimate, L17 crossing_operator (unitary). numpy / scipy only.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass
class Law:
    """L = (P, nu, f): Markov kernel (or None), viscosity, forcing coordinates (or None)."""
    P: np.ndarray = None
    nu: float = 0.0
    f: np.ndarray = None


@dataclass
class DeltaE:
    """Additive law-difference vector (L13). Supports +, -, negation and scalar multiplication."""
    vec: np.ndarray

    def __post_init__(self):
        self.vec = np.asarray(self.vec)

    def __add__(self, o):
        return DeltaE(self.vec + o.vec)

    def __sub__(self, o):
        return DeltaE(self.vec - o.vec)

    def __neg__(self):
        return DeltaE(-self.vec)

    def __mul__(self, c):
        return DeltaE(self.vec * c)

    __rmul__ = __mul__


def coords(law, chart="log", floor=1e-12):
    """Additive law coordinates [nu, f, chart(P)]; chart 'log' (log-ratio, floored) or 'id' (difference)."""
    parts = [np.array([law.nu], float)]
    if law.f is not None:
        parts.append(np.ravel(law.f).astype(float))
    if law.P is not None:
        P = np.asarray(law.P, float)
        parts.append((np.log(np.maximum(P, floor)) if chart == "log" else P).ravel())
    return np.concatenate(parts)


def delta(law_a, law_b, chart="log"):
    """L13: dE_{a->b} = coords(b) - coords(a). Any chart is a fixed map, so dE_{i->k} = dE_{i->j} + dE_{j->k}
    holds exactly (cocycle), dE_{a->a} = 0 and dE_{b->a} = -dE_{a->b}."""
    return DeltaE(coords(law_b, chart) - coords(law_a, chart))


def fingerprint(law, k, chart="log"):
    """kappa_k = (law coordinates, chronology index k)."""
    return np.concatenate([coords(law, chart), [float(k)]])


def law_classes(fingerprints, tol=1e-9):
    """L14': group fingerprints whose law coordinates agree within tol (regimes recur, so two eras may share
    a class; the chronology index, the last column, is ignored). Returns (class id per row in order of first
    appearance, class centres (C, d))."""
    F = np.asarray(fingerprints, float)[:, :-1]
    centers, cls = [], []
    for row in F:
        for c, ctr in enumerate(centers):
            if np.linalg.norm(row - ctr) <= tol * (1 + np.linalg.norm(ctr)):
                cls.append(c)
                break
        else:
            centers.append(row)
            cls.append(len(centers) - 1)
    return np.array(cls), np.array(centers)


def address(centers, query, belief=None, tol=1e-9):
    """L14': nearest law class. If the class centres are pairwise s-separated and the estimate is within < s/2
    of its true class centre, the true class is returned. Exact ties (within tol) go to the class with the
    largest `belief` (e.g. an HMMFilter posterior; no timestamp exists at inference), else the lowest index."""
    centers = np.asarray(centers, float)
    q = np.asarray(query, float)[: centers.shape[1]]
    d = np.linalg.norm(centers - q, axis=1)
    ties = np.flatnonzero(d <= d.min() + tol * (1 + d.min()))
    if belief is not None and len(ties) > 1:
        return int(ties[np.argmax(np.asarray(belief)[ties])])
    return int(ties[0])


class HMMFilter:
    """L14' forward filter over law classes: alpha_t(j) ∝ e_j(x_t) sum_i alpha_{t-1}(i) A_ij, with Gaussian
    emission e_j(x) = exp(-||x - c_j||^2 / 2 sigma^2) and alpha_1 ∝ e(x_1) * prior. `belief` is alpha."""

    def __init__(self, centers, A, sigma, prior=None):
        self.c = np.asarray(centers, float)
        self.A = np.asarray(A, float)
        self.sigma = sigma
        K = len(self.c)
        self.belief = np.full(K, 1.0 / K) if prior is None else np.asarray(prior, float) / np.sum(prior)
        self._first = True

    def update(self, x):
        pred = self.belief if self._first else self.belief @ self.A
        self._first = False
        d2 = np.sum((self.c - np.asarray(x, float)[: self.c.shape[1]]) ** 2, axis=1)
        a = pred * np.exp(-(d2 - d2.min()) / (2 * self.sigma ** 2))  # common factor cancels in normalisation
        self.belief = a / a.sum()
        return self.belief

    def address(self):
        return int(np.argmax(self.belief))


def make_generators(N, A, rng, commuting=False):
    """A skew-Hermitian generators of size N x N. commuting=True gives i*diag(real) (pairwise commuting)."""
    rng = np.random.default_rng(rng)
    if commuting:
        return [1j * np.diag(rng.standard_normal(N)) for _ in range(A)]
    M = (rng.standard_normal((A, N, N)) + 1j * rng.standard_normal((A, N, N))) / np.sqrt(N)
    return [(m - m.conj().T) / 2 for m in M]


def _sum_gen(dE, generators):
    v = dE.vec if isinstance(dE, DeltaE) else np.asarray(dE)
    if np.iscomplexobj(v):
        if np.any(np.abs(v.imag) > 0):
            raise ValueError("L17 requires real coefficients: complex ones break skew-Hermiticity")
        v = v.real
    if len(v) != len(generators):
        raise ValueError(f"{len(v)} coefficients for {len(generators)} generators")
    for g in generators:
        if not np.allclose(g, -np.conj(g).T, atol=1e-10):
            raise ValueError("generator is not skew-Hermitian")
    return sum(c * np.asarray(g) for c, g in zip(v, generators))


def crossing_operator(dE, generators):
    """L17 / D8: S(dE) = expm(sum_a dE_a G_a) with real dE_a and skew-Hermitian G_a, hence unitary: it
    preserves norms and fidelities, S(-dE) = S(dE)^H. L15: if the G_a commute, S(dE_jk) S(dE_ij) = S(dE_ik)."""
    return expm(_sum_gen(dE, generators))


def holonomy(loop_dEs, generators):
    """L16: U = S(dE_m) ... S(dE_1) around a closed loop (sum of dE = 0 by L13). Commuting generators give
    U = I (L15). In general U != I: the deviation measures the NON-COMMUTATIVITY of the generators G_a (not
    curvature of law space)."""
    tot = sum(d.vec if isinstance(d, DeltaE) else np.asarray(d) for d in loop_dEs)
    scale = max(1.0, max(np.abs(d.vec if isinstance(d, DeltaE) else d).max() for d in loop_dEs))
    if np.abs(tot).max() > 1e-9 * scale:
        raise ValueError("loop is not closed (sum of dE != 0)")
    U = np.eye(np.asarray(generators[0]).shape[0], dtype=complex)
    for d in loop_dEs:
        U = crossing_operator(d, generators) @ U
    return U


def commutator_estimate(loop_dEs, generators):
    """BCH second order: with step generators A_1..A_m applied in order (later ones on the left) and
    sum A_i = 0, log U = (1/2) sum_{i<j} [A_j, A_i] + O(eps^3). For a triangle A, B, -(A+B) this is
    (1/2)[B, A] = -(1/2)[A, B]; the spec's (1/2)[A,B] is the reversed product convention."""
    As = [_sum_gen(d, generators) for d in loop_dEs]
    est = np.zeros_like(As[0])
    for i in range(len(As)):
        for j in range(i + 1, len(As)):
            est = est + 0.5 * (As[j] @ As[i] - As[i] @ As[j])
    return est

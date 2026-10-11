"""Router (D7): certified top-m (L5), error radii, and the decision rule.

L5 follows separatrix `decide.topk_determined`: T is the top-m of every score vector in the box
prod [D_i +- R_i] iff min_{i in T}(D_i - R_i) > max_{j not in T}(D_j + R_j) (strict).  Comparing only the
rank-m and rank-(m+1) boxes is unsound when radii vary; this rule is not.  Otherwise REFUSE, which means blend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np

ALONE, JUMP, BLEND, REFUSE_BLEND, OOD = "ALONE", "JUMP", "BLEND", "REFUSE_BLEND", "OOD"


def gamma(n, dtype=np.float64):
    """Higham gamma_n = n u / (1 - n u), u = eps/2.  Raises ValueError when n u >= 1/2 (bound vacuous)."""
    u = np.finfo(dtype).eps / 2
    if n < 1 or n * u >= 0.5:
        raise ValueError(f"gamma_{n} is vacuous for {np.dtype(dtype).name}")
    return n * u / (1 - n * u)


def enclose_scores(D, gamma_n):
    """Elementwise radii R = gamma_n/(1 - gamma_n) |D| for scores that are sums of nonnegative terms
    (the relative form in terms of the computed value, separatrix enclose.py).  0 <= gamma_n < 1."""
    if not 0.0 <= gamma_n < 1.0:
        raise ValueError(f"gamma_n must be in [0, 1), got {gamma_n}")
    return gamma_n / (1.0 - gamma_n) * np.abs(np.asarray(D, dtype=float))


def state_radius(eps):
    """Per-coordinate score radius for a state error ||psi - psi'|| <= eps (E3).

    ||p(psi) - p(psi')||_1 <= 2 eps and both are probability vectors (the difference sums to 0), so
    |p_k(psi) - p_k(psi')| <= eps for every k.
    """
    if eps < 0:
        raise ValueError("eps must be >= 0")
    return float(eps)


class Frontier(NamedTuple):
    """The blocking pair: the weakest member and the strongest outsider whose boxes touch."""
    inside: int
    outside: int
    inside_lo: float
    outside_hi: float
    gap: float  # D[inside] - D[outside]


def certified_topm(D, R, m):
    """(T, None) when the m largest of D are certified for every score vector in the box, else (None, Frontier).  (L5)"""
    D = np.asarray(D, dtype=float).reshape(-1)
    n = D.size
    if not 1 <= m <= n:
        raise ValueError(f"need 1 <= m <= n; got m={m}, n={n}")
    try:
        R = np.broadcast_to(np.asarray(R, dtype=float), D.shape)
    except ValueError as e:
        raise ValueError(f"R cannot be broadcast against D: {e}") from e
    if (R < 0).any():
        raise ValueError("radii must be >= 0")
    order = np.argsort(-D, kind="stable")
    T, out = order[:m], order[m:]
    if out.size == 0:
        return set(T.tolist()), None
    lo = D - R
    hi = D + R
    i = int(T[np.argmin(lo[T])])
    j = int(out[np.argmax(hi[out])])
    if lo[i] > hi[j]:
        return set(T.tolist()), None
    return None, Frontier(i, j, float(lo[i]), float(hi[j]), float(D[i] - D[j]))


@dataclass(frozen=True)
class Decision:
    kind: str
    eras: tuple
    ood: bool = False  # True: flag as an RSL birth candidate


def decide(conformal_set, certified, requested=None, *, null=None):
    """Router decision rule.

    requested -> JUMP to it.  C empty -> OOD (answer alone, flag).  C == {null} -> ALONE.  |C| == 1 -> JUMP.
    |C| >= 2 -> BLEND over the certified top-m (restricted to C) if L5 certified one, else REFUSE_BLEND over C.
    """
    if requested is not None:
        return Decision(JUMP, (requested,))
    C = set(conformal_set)
    if not C:
        return Decision(OOD, (), True)
    if null is not None and C == {null}:
        return Decision(ALONE, ())
    if len(C) == 1:
        return Decision(JUMP, (next(iter(C)),))
    if certified is not None:
        T = C & set(certified)
        if T:
            return Decision(BLEND, tuple(sorted(T)))
    return Decision(REFUSE_BLEND, tuple(sorted(C)))

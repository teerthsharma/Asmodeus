"""Liquid dynamics between events (D11, L20, L20')."""
import numpy as np


def max_step(tau, M):
    """Largest h for which the box is invariant (L20): h <= tau / (1 + tau*M)."""
    return tau / (1.0 + tau * M)


def invariant_box(A):
    """L20: B = [min(0, A_min), max(0, A_max)]."""
    A = np.asarray(A, float)
    return min(0.0, A.min()), max(0.0, A.max())


def box_distance(x, lo, hi):
    """l_inf distance of x to the box [lo, hi]^n (L20')."""
    x = np.asarray(x, float)
    return float(np.max(np.maximum(np.maximum(lo - x, x - hi), 0.0)))


def liquid_step(x, I, F, tau, A, h, M=1.0, check=True):
    """L20. x' = x + h[-(1/tau + F) x + A*F], F = F(x, I) in [0, M] (callable or array).

    With 0<=F<=M and h <= tau/(1+tau*M) the coefficient c = 1 - h/tau - hF is >= 0 and
    c + hF <= 1, so x' is a convex combination of x, A and 0: the box invariant_box(A) is kept
    (induction), and dist(x', B) <= (1 - h/tau) dist(x, B) (L20').
    Raises ValueError if h exceeds the bound or F leaves [0, M]; check=False skips both (for counterexamples).
    """
    x = np.asarray(x, float)
    Fx = np.asarray(F(x, I) if callable(F) else F, float)
    if check:
        if h > max_step(tau, M) * (1 + 1e-12):
            raise ValueError(f"h={h} > tau/(1+tau*M)={max_step(tau, M)}: invariant box not guaranteed (L20)")
        if np.any(Fx < 0) or np.any(Fx > M):
            raise ValueError("F must lie in [0, M] (L20)")
    return x + h * (-(1.0 / tau + Fx) * x + np.asarray(A, float) * Fx)

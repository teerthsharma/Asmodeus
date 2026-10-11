"""H4: Hilbert projective metric for the CTC loop (undamped contraction certificate, no eps mixing).

d_H(x,y) = osc(log x - log y) is a complete metric on positive vectors modulo scale. Normalisation and a fixed
positive mask are isometries; x -> x^(1/T) scales d_H by exactly 1/T. For the log-linear kernel G(w) ~ exp(beta A w)
on the simplex, d_H(G(w),G(v)) <= (beta D / 4) d_H(w,v) with D = max_{a,b,i,j} A_ai - A_bi - A_aj + A_bj
(sum-zero w-v has |w-v|_1 / 2 <= tanh(d_H/4) <= d_H/4). Top-m: use a soft positive mask, never a hard support.
"""
import numpy as np


def oscillation(v):
    v = np.asarray(v, float)
    return float(v.max() - v.min())


def dH(x, y):
    """Hilbert projective distance between positive vectors."""
    return oscillation(np.log(x) - np.log(y))


def birkhoff_coeff(K):
    """tanh(Delta/4), Delta = max log(K_ai K_bj / (K_aj K_bi)): d_H-contraction of x -> K x for a positive matrix K."""
    L = np.log(np.asarray(K, float))
    delta = (L[:, None, :, None] + L[None, :, None, :] - L[:, None, None, :] - L[None, :, :, None]).max()
    return float(np.tanh(delta / 4))


def loglinear_kappa(A, beta):
    """kappa = beta D / 4: d_H-Lipschitz constant of w -> softmax(beta A w) on the simplex (kernel entries only)."""
    D = A[:, None, :] - A[None, :, :]  # D[a,b,i] = A_ai - A_bi
    return float(beta * (D.max(-1) - D.min(-1)).max() / 4)


def tt_map(A, beta, T=1.0, mu=None):
    """G(w) = normalise(mu * softmax(beta A w)^(1/T)); mu is a fixed positive (soft) mask, None = 1."""
    lm = 0.0 if mu is None else np.log(mu)

    def G(w):
        z = beta * (A @ w) / T + lm
        e = np.exp(z - z.max())
        return e / e.sum()

    return G


def iterate(G, w0, q, tol, max_iter=10_000):
    """Undamped w <- G(w), q = certified d_H contraction (< 1). Stops when q/(1-q) d_H(w_n, w_n-1) <= tol, which
    certifies d_H(w_n, w*) <= tol. Returns (w, n, cert, steps) with steps[k] = d_H(w_k+1, w_k)."""
    if not 0 <= q < 1:
        raise ValueError("needs a certified q in [0, 1)")
    w, steps, cert = np.asarray(w0, float), [], np.inf
    for n in range(1, max_iter + 1):
        w1 = G(w)
        steps.append(dH(w1, w))
        w, cert = w1, q / (1 - q) * steps[-1]
        if cert <= tol:
            return w, n, cert, steps
    return w, max_iter, cert, steps

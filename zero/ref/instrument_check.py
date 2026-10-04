"""Quantum instruments {K_v} on C^N: completeness, max outcome mass, reachable tokens (PLAN.md bound check).

An instrument is any sequence of N x N complex matrices K_v (a list, or a lazy sequence as returned by
diag_q_instrument so |V| = 4096 never has to be materialised).
"""
import numpy as np


def _haar_isometry(rows, cols, rng):
    a = rng.normal(size=(rows, cols)) + 1j * rng.normal(size=(rows, cols))
    q, r = np.linalg.qr(a)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def random_instrument(N, V, rank, rng):
    """Stack of V Kraus blocks of size rank x N cut from a (V*rank) x N Stiefel isometry, padded to N x N."""
    W = _haar_isometry(V * rank, N, rng)
    K = []
    for v in range(V):
        k = np.zeros((N, N), complex)
        k[:rank] = W[v * rank:(v + 1) * rank]
        K.append(k)
    return K


class diag_q_instrument:
    """K_v = diag(sqrt(M[:, v])) Q, M row-stochastic N x V, Q unitary (PLAN.md item 3)."""

    def __init__(self, N, V, rng):
        M = rng.random((N, V)) ** 4
        self.M = M / M.sum(1, keepdims=True)
        self.Q = _haar_isometry(N, N, rng)

    def __len__(self):
        return self.M.shape[1]

    def __getitem__(self, v):
        if not 0 <= v < len(self):
            raise IndexError(v)
        return np.sqrt(self.M[:, v])[:, None] * self.Q


def is_complete(K, tol=1e-9):
    S = sum(k.conj().T @ k for k in K)
    return np.allclose(S, np.eye(S.shape[0]), atol=tol)


def max_masses(K):
    """max over unit psi of |K_v psi|^2 = largest singular value squared, per outcome."""
    return np.array([np.linalg.norm(k, 2) ** 2 for k in K])


def instrument_max_mass(K):
    return float(max_masses(K).sum())


def tokens_reachable(K, p):
    return int((max_masses(K) >= p).sum())


def mutual_information_nats(K, psis):
    """I(X; V) for X uniform over the rows of psis (pure states) and V the outcome of instrument K."""
    P = np.array([[np.linalg.norm(k @ p) ** 2 for k in K] for p in psis])     # p(v | x)
    pv = P.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log(P / pv), 0.0)
    return float(t.sum(1).mean())

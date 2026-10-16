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

    def max_masses(self):
        # Optimization: For K_v = diag(sqrt(M[:, v])) Q with Q unitary,
        # K_v^H K_v = Q^H diag(M[:, v]) Q.
        # Its eigenvalues are M[:, v], so sigma_max(K_v)^2 = max_i M[i, v].
        # Gives ~330x speedup by avoiding V individual SVD calls.
        return self.M.max(axis=0)

    def is_complete(self, tol=1e-9):
        # Optimization: sum_v K_v^H K_v = Q^H diag(sum_v M[:, v]) Q = Q^H I Q = I.
        return np.allclose(self.M.sum(axis=1), 1.0, atol=tol)


def is_complete(K, tol=1e-9):
    if hasattr(K, "is_complete"):
        return K.is_complete(tol=tol)
    if isinstance(K, (list, tuple)) and len(K) > 0 and isinstance(K[0], np.ndarray):
        K_arr = np.asarray(K)
        S = np.einsum("vji,vjk->ik", K_arr.conj(), K_arr)
        return np.allclose(S, np.eye(S.shape[0]), atol=tol)
    S = sum(k.conj().T @ k for k in K)
    return np.allclose(S, np.eye(S.shape[0]), atol=tol)


def max_masses(K):
    """max over unit psi of |K_v psi|^2 = largest singular value squared, per outcome."""
    if hasattr(K, "max_masses"):
        return K.max_masses()
    if isinstance(K, (list, tuple)) and len(K) > 0 and isinstance(K[0], np.ndarray):
        # Batched SVD over V matrices at once via LAPACK (vectorized C implementation)
        K_arr = np.asarray(K)
        s = np.linalg.svd(K_arr, compute_uv=False)
        return s[:, 0] ** 2
    return np.array([np.linalg.norm(k, 2) ** 2 for k in K])


def instrument_max_mass(K):
    return float(max_masses(K).sum())


def tokens_reachable(K, p):
    return int((max_masses(K) >= p).sum())


def mutual_information_nats(K, psis):
    """I(X; V) for X uniform over the rows of psis (pure states) and V the outcome of instrument K."""
    if isinstance(K, diag_q_instrument):
        # Vectorized calculation for diag_q_instrument: ||K_v @ p||^2 = sum_i M[i, v] * |(Q @ p)_i|^2
        QP = psis @ K.Q.T
        P = (np.abs(QP) ** 2) @ K.M
    elif isinstance(K, (list, tuple)) and len(K) > 0 and isinstance(K[0], np.ndarray):
        # Batched matrix-vector products across outcomes and state samples
        K_arr = np.asarray(K)
        KP = np.einsum("vij,xj->xvi", K_arr, psis)
        P = np.sum(np.abs(KP) ** 2, axis=-1)
    else:
        P = np.array([[np.linalg.norm(k @ p) ** 2 for k in K] for p in psis])     # p(v | x)
    pv = P.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log(P / pv), 0.0)
    return float(t.sum(1).mean())

"""Perturbation-theory crossing (H1): S = exp(X) in the eigenbasis of rho_j, with
X_ab = dE_ab / (l_b - l_a) for a in the top-d, b outside (first-order Davis-Kahan rotation), completed skew-Hermitian."""
import numpy as np
from scipy.linalg import expm


def crossing_pt(rho_j, dE, d, n_samples=None):
    w, U = np.linalg.eigh(rho_j)
    w, U = w[::-1], U[:, ::-1]
    E = U.conj().T @ dE @ U
    if n_samples:  # James-Stein shrink: noise var of a complex-Gaussian sample-covariance entry = l_a l_b / n
        v = w[:d, None] * w[None, d:] / n_samples
        E[:d, d:] *= np.maximum(0, 1 - v / np.maximum(abs(E[:d, d:]) ** 2, 1e-300))
    X = np.zeros_like(E)
    gap = w[:d, None] - w[None, d:]  # l_a - l_b > 0
    X[:d, d:] = -E[:d, d:] / gap
    X[d:, :d] = -X[:d, d:].conj().T
    return U @ expm(X) @ U.conj().T


def sin_theta_bound(rho_j, P_k, d):
    """(lhs, rhs): sum_{i<=d} l_i - tr(P_k rho_j)  >=  (l_d - l_{d+1}) * ||sinTheta(P_j, P_k)||_F^2."""
    w, U = np.linalg.eigh(rho_j)
    w, U = w[::-1], U[:, ::-1]
    Pj = U[:, :d] @ U[:, :d].conj().T
    lhs = w[:d].sum() - np.real(np.trace(P_k @ rho_j))
    return lhs, (w[d - 1] - w[d]) * (d - np.real(np.trace(Pj @ P_k)))

"""Observer (D2/D2'): Parseval frame, Born rule, residual, tail certificates, post-selection, null split.

Lemmas: L1 (Born exactness), L2 (exact residual), L3/L3' (tail certificate), L4 (post-selection
temperature), L23/L23''(non-destructive birth, optimal birth direction). numpy only.
Convention: effects E_k = conj(V[k])^T V[k] (N x N), so <psi|E_k|psi> = |V[k] @ psi|^2.
"""
from __future__ import annotations

import numpy as np


def parseval_frame(N, K, rng=None):
    """V in C^{(K+1) x N} with V^H V = I_N; the last row is the null row v_empty.  (L1)

    Needs K + 1 >= N (an isometry C^N -> C^{K+1}); raises ValueError otherwise.
    """
    if K + 1 < N or N < 1:
        raise ValueError(f"need K + 1 >= N >= 1 for V^H V = I_N; got N={N}, K={K}")
    g = np.random.default_rng(rng)
    A = g.standard_normal((K + 1, N)) + 1j * g.standard_normal((K + 1, N))
    return np.linalg.qr(A)[0]


def effects_from_frame(V):
    """List of rank-one effects E_k = v_k^H v_k, summing to I (L1)."""
    return [np.outer(v.conj(), v) for v in V]


def born(V, psi):
    """p_k = |V[k] psi|^2.  psi (N,) -> (K+1,); psi (B, N) -> (B, K+1).  Sums to 1 for unit psi (L1)."""
    psi = np.asarray(psi, dtype=complex)
    if not np.allclose(np.linalg.norm(psi, axis=-1), 1.0, atol=1e-6):
        raise ValueError("born needs unit-norm states (L1 hypothesis ||psi|| = 1)")
    return np.abs(psi @ V.T) ** 2


def _idx(T):
    return np.asarray(T, dtype=int).reshape(-1)


def residual(p, T):
    """r_T = exact mass outside T (L2), summed directly so it is >= 0 by construction."""
    p = np.asarray(p, dtype=float)
    mask = np.ones(p.shape[0], dtype=bool)
    mask[_idx(T)] = False
    return float(p[mask].sum())


def tail_certificate(p, T, temp):
    """Upper bound on sum_{k not in T} p_k^{1/temp}  (L3, L3').

    temp <= 1 (q = 1/temp >= 1): (max_out p)^{q-1} r_T.
    temp  > 1 (q < 1, Amendment B L3'): m_out^{1-q} r_T^q (Holder), tight at a uniform tail; it depends on
    the number of outside outcomes m_out, which the router knows.  temp <= 0 raises ValueError.
    """
    if not temp > 0:
        raise ValueError(f"temp must be > 0, got {temp}")
    p = np.asarray(p, dtype=float)
    mask = np.ones(p.shape[0], dtype=bool)
    mask[_idx(T)] = False
    out = p[mask]
    r = float(out.sum())
    if out.size == 0 or r <= 0.0:
        return 0.0
    q = 1.0 / temp
    if temp <= 1.0:
        return float(out.max() ** (q - 1.0) * r)
    return float(out.size ** (1.0 - q) * r**q)


def post_select(p, T, temp):
    """w_k = (1 - r_T) p_k^{1/temp} / sum_{j in T} p_j^{1/temp} for k in T (order of T), then w_other = r_T.  (L4)

    Requires sum_{k in T} p_k > 0 (Amendment C); raises ValueError otherwise (including empty T).
    """
    if not temp > 0:
        raise ValueError(f"temp must be > 0, got {temp}")
    p = np.asarray(p, dtype=float)
    T = _idx(T)
    if T.size == 0 or p[T].sum() <= 0.0:
        raise ValueError("L4 needs sum_{k in T} p_k > 0")
    r = residual(p, T)
    a = (p[T] / p[T].max()) ** (1.0 / temp)  # scaled by the max so cold temperatures cannot overflow
    return np.append((1.0 - r) * a / a.sum(), r)


def _sqrtm_psd(M):
    w, U = np.linalg.eigh((M + M.conj().T) / 2)
    return (U * np.sqrt(np.clip(w, 0.0, None))) @ U.conj().T


def split_null(effects, Q):
    """L23'/Amendment A: split the null effect (last in the list) by 0 <= Q <= I.

    E_new = E_null^{1/2} Q E_null^{1/2},  E_null' = E_null^{1/2} (I - Q) E_null^{1/2}.  Returns
    effects[:-1] + [E_new, E_null'] (null stays last); the existing effects are untouched, so every old
    p_k(psi) is unchanged and the sum is still I.  Raises ValueError on an illegal Q or effects not summing to I.
    """
    E_null = np.asarray(effects[-1])
    N = E_null.shape[0]
    Q = np.asarray(Q, dtype=complex)
    if Q.shape != (N, N):
        raise ValueError(f"Q must be {(N, N)}, got {Q.shape}")
    if not np.allclose(sum(effects), np.eye(N), atol=1e-9):
        raise ValueError("effects must sum to I")
    if not np.allclose(Q, Q.conj().T, atol=1e-10):
        raise ValueError("Q must be Hermitian")
    w = np.linalg.eigvalsh((Q + Q.conj().T) / 2)
    if w.min() < -1e-10 or w.max() > 1 + 1e-10:
        raise ValueError("need 0 <= Q <= I")
    S = _sqrtm_psd(E_null)
    new = S @ Q @ S
    rest = S @ (np.eye(N) - Q) @ S
    return list(effects[:-1]) + [(new + new.conj().T) / 2, (rest + rest.conj().T) / 2]


def captured_mass(E_null, C, Q):
    """Born mass tr(E_null^{1/2} C E_null^{1/2} Q) the new era captures from the unexplained buffer C."""
    S = _sqrtm_psd(E_null)
    return float(np.trace(S @ C @ S @ Q).real)


def birth_direction(E_null, C, r):
    """L23'': the Q of rank <= r maximising captured mass is the top-r eigenprojector of E^{1/2} C E^{1/2} (Ky Fan).

    C = sum psi psi^H over the unexplained buffer.  r is clipped to [0, N].
    """
    S = _sqrtm_psd(E_null)
    N = S.shape[0]
    M = S @ C @ S
    w, U = np.linalg.eigh((M + M.conj().T) / 2)
    r = int(min(max(r, 0), N))
    if r == 0:
        return np.zeros((N, N), dtype=complex)
    top = U[:, -r:]
    return top @ top.conj().T

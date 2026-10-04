"""POVMs from Naimark dilation: a K-outcome POVM on C^N is E_k = V^† |k><k| V for an isometry V (K x N,
V^†V = I_N). K = N and V = U unitary gives the projective case; K > N gives a genuine POVM."""
import torch


def naimark_povm(V):
    """V [K, N] isometry -> E [K, N, N], E_k = V^† |k><k| V = conj(V[k])^T V[k]."""
    return V.conj()[:, :, None] * V[:, None, :]


def projective_povm(U):
    """U [N, N] unitary -> E [N, N, N], E_k = U^† |k><k| U (rank-1 projectors, sum = I)."""
    return naimark_povm(U)


def povm_probs(E, psi):
    """p_k = <psi|E_k|psi>, real."""
    return torch.einsum("a,kab,b->k", psi.conj(), E, psi).real

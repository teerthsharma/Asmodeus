"""Arm Q: coherent observer. The softmax router of ObserverAtlas is the Born rule on a DECOHERED state
(ref/identities.py §6): p = softmax(s) = |psi|^2 with psi = sqrt(softmax(s)), and it obeys Luce IIA.
Arm Q keeps the chart superposition but lets the observer interfere chart amplitudes before measuring:

    p = (U psi)^2,   U = Cayley(A) = (I - A/2)^{-1} (I + A/2),   A = P Q^T - Q P^T  (rank-2r skew, learned)
    (parameters Cp = P, Cq = Q; self.P is the parent's predictor)

Phases (origin: "small phases all around the sphere but the decider will be a softmax"): psi_k carries a
learned phase theta_k(q) = (W_q c) . theta_key_k. theta_keys start at 0, which is the real sign-only model,
and phases change p only through the coupling (Cp = 0 means softmax for any theta).
U is orthogonal, so p >= 0 and sum p = 1 hold by construction. P = 0 at init gives U = I, which is exactly
the softmax (decoherence limit), and grad_P is nonzero there, so SGD can learn coherence. Woodbury keeps
the apply at O(K r) per query and never forms a KxK matrix. top-m + gate stay the nonlinear "measurement":
route() picks the top-m of p, and the parent's softmax over log p renormalises p on them.

born_gibbs is the exact mixed-state form diag(exp(-beta H))/Z (O(K^3), for reference and tests only). It
reproduces identities §6. The learned arm uses the pure-state amplitude form above. The two are different
coherent models with the same decoherence limit.
Product keys are refused: the coherent mix needs all K amplitudes (O(K r)), so sqrt(K) routing is lost.
"""
import math

import torch
from torch import nn

from atlas.observer_atlas import ObserverAtlas


def born_gibbs(H, beta=1.0):
    w, V = torch.linalg.eigh(H)
    p = (V ** 2 * torch.exp(-beta * w)[..., None, :]).sum(-1)
    return p / p.sum(-1, keepdim=True)


def cayley_apply(x, P, Q):
    """x [N, K] -> x U^T, U = (I - A/2)^{-1}(I + A/2), A = P Q^T - Q P^T = L J L^T, L = [P Q]."""
    r = P.shape[1]
    L = torch.cat([P, Q], 1)                                            # [K, 2r]
    Ir, Zr = torch.eye(r, dtype=x.dtype, device=x.device), torch.zeros(r, r, dtype=x.dtype, device=x.device)
    J = torch.cat([torch.cat([Zr, Ir], 1), torch.cat([-Ir, Zr], 1)], 0)  # A = L J L^T
    M = torch.linalg.inv(-2 * J - L.T @ L)                              # Woodbury core, always invertible
    y = x + (x @ L) @ (J / 2).T @ L.T                                   # (I + A/2) x
    return y + (y @ L) @ M.T @ L.T                                      # (I - A/2)^{-1} y


def coherent_probs(s, P, Q, theta=None):
    """|U psi|^2 with psi_k = sqrt(softmax(s))_k e^{i theta_k}; U real, so it acts on Re and Im separately."""
    amp = torch.exp((s - s.logsumexp(-1, keepdim=True)) / 2)
    if theta is None:
        p = cayley_apply(amp, P, Q) ** 2
    else:
        p = cayley_apply(amp * torch.cos(theta), P, Q) ** 2 + cayley_apply(amp * torch.sin(theta), P, Q) ** 2
    return p / p.sum(-1, keepdim=True)                                  # exact up to rounding


class CoherentAtlas(ObserverAtlas):
    def __init__(self, D, A, rank=4, **kw):
        if kw.get("product_keys"):
            raise ValueError("coherent observer needs all K amplitudes; product keys not supported")
        super().__init__(D, A, **kw)
        self.rank = rank
        self.Cp = nn.Parameter(torch.zeros(self.K, rank))
        self.Cq = nn.Parameter(torch.randn(self.K, rank) / math.sqrt(self.K))
        self.theta_keys = nn.Parameter(torch.zeros(self.K, self.d_key))  # phase_k(q) = W_q c . theta_key_k

    def full_logits(self, c):
        theta = self.W_q(c) @ self.theta_keys.T
        return torch.log(coherent_probs(super().full_logits(c), self.Cp, self.Cq, theta).clamp_min(1e-30))

    def route(self, c):
        return self.full_logits(c).topk(self.m, -1)

    def active_flops(self):
        """+ phase scores K d_key + 2 x (4 K x 2r projections + 2r x 2r core matvecs) (Re, Im) + 2K squares.
        (W_q c is recomputed in full_logits; a cached query would save d_h d_key, not counted.)"""
        r = self.rank
        return super().active_flops() + self.K * (self.d_key + 16 * r + 2) + 16 * r * r

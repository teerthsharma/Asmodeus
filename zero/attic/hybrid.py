"""WITHDRAWN (Seal, 2026-10-04): replaced by ASMODEOUS slot-by-slot design; kept for the record, not scored, not on the test path.

HYBRID: OBSERVER-ATLAS real charts + Arm Q Born/phase decider + time-traveller CTC loop (no binding;
CLAUDE.md rule 2). Composition is plain multiple inheritance, MRO HybridAtlas -> TimeTravelAtlas ->
CoherentAtlas -> ObserverAtlas: the CTC loop's super().__init__ lands in CoherentAtlas, so the coherent
full_logits/route (Born rule over interfering, phased chart amplitudes) drive both forward() routing and
the CTC START preselection. The charts, gluing maps and gate stay the real-valued ObserverAtlas ones.

Three limits, each tested against the parent:
  gamma = 0 (or M = m)     -> CoherentAtlas   (no travel: HALT never re-scores)
  Cp = 0 and theta_keys = 0 -> TimeTravelAtlas (decohered decider is the plain softmax)
  both                      -> ObserverAtlas

Readouts (amplitudes, observer_states, chern_*, sign_holonomy_readout) are EVAL ONLY: they run under
no_grad on a trained arm and never feed back into any training loss (L-GIVEN: structure is read, not given).
"""
import numpy as np
import torch

from atlas.coherent_observer import CoherentAtlas, cayley_apply
from atlas.observer_atlas import ObserverAtlas
from atlas.time_traveller import TimeTravelAtlas
from beds import lift
from ref.berry_check import chern_number, icosphere


class HybridAtlas(TimeTravelAtlas, CoherentAtlas):
    def __init__(self, D, A, rank=4, M=4, gamma=1.0, max_loops=1, **kw):
        TimeTravelAtlas.__init__(self, D, A, M=M, gamma=gamma, max_loops=max_loops, rank=rank, **kw)

    def rollout_flops(self, H):
        """CTC rollout cost + the coherent decider, paid once per rollout (at START)."""
        return TimeTravelAtlas.rollout_flops(self, H) + (CoherentAtlas.active_flops(self) - ObserverAtlas.active_flops(self))

    def amplitudes(self, c):
        """Complex chart amplitudes psi [N, K] after the Cayley coupling; |psi|^2 = exp(full_logits(c))."""
        s = ObserverAtlas.full_logits(self, c)
        amp = torch.exp((s - s.logsumexp(-1, keepdim=True)) / 2)
        theta = self.W_q(c) @ self.theta_keys.T
        psi = torch.complex(cayley_apply(amp * torch.cos(theta), self.Cp, self.Cq),
                            cayley_apply(amp * torch.sin(theta), self.Cp, self.Cq))
        return psi / psi.norm(dim=-1, keepdim=True)


@torch.no_grad()
def observer_states(arm, obs):
    """obs [N, D] (numpy or torch) -> complex numpy [N, K]: amplitudes from a single-frame context (eval)."""
    was = arm.training
    arm.eval()
    try:
        obs = torch.as_tensor(obs, dtype=torch.float32)
        return arm.amplitudes(arm.backbone(obs[:, None])[0][:, 0]).numpy()
    finally:
        arm.train(was)


def chern_from_states(V, F, psi):
    """Lattice Chern number of the state field psi [len(V), K] over the triangulation (V, F)."""
    row = {tuple(np.round(v, 12)): psi[i] for i, v in enumerate(V)}
    return chern_number(V, F, lambda v: row[tuple(np.round(v, 12))])


def chern_s2(arm, level=2):
    """Chern number of the observer's amplitude field on the S2 bed (arm.D must be 256 for the lift)."""
    V, F = icosphere(level)
    return chern_from_states(V, F, observer_states(arm, lift("s2", np.array(V)).astype(np.float32)))


def sign_holonomy_readout(*a, **k):
    from atlas.phase_sphere import sign_holonomy  # ponytail: lazy, atlas/phase_sphere.py not landed yet
    return sign_holonomy(*a, **k)

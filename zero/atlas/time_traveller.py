"""TIME-TRAVELLER (PROMPT.md §4.7) as a CTC start/halt loop (ref/origin/godel_ctc_figure.jpg).

Each imagined life-line (one CEM action sequence) picks its own self-consistent chart set.
  START : the static keys preselect M candidate charts at the present frame (the point on the r = const hypersurface).
  LIFE-LINE : every candidate chart predicts the whole glued future F [N, H, M, D_g] under the same actions.
  HALT : the observer re-selects the m charts whose futures sit closest to the consensus future of its current set
         (score = static key logit - gamma * mean-over-H squared deviation); it halts when the set maps to itself
         (Novikov fixed point: the future it reads picks the charts that produced that future) or after max_loops.
gamma and max_loops are planning hyperparameters (tuned on the tune seed like CEM settings), not parameters.
gamma = 0 or M = m recovers ObserverAtlas.rollout exactly.
"""
import torch

from atlas.observer_atlas import ObserverAtlas


def _flops(arm, H, M, loops):
    """Multiply-adds of one rollout per sample: observer + M charts encoded and rolled H steps + loops x re-scoring."""
    D, A, h, dc, Dg, m = arm.D, arm.A, arm.hidden, arm.d_c, arm.D_g, arm.m
    keys = 2 * arm.side * (arm.d_key // 2) if arm.pk else arm.K * arm.d_key
    router = 3 * arm.d_h * (D + arm.d_h) + arm.d_h * arm.d_key + arm.d_h + keys
    return (router + M * (D * h + h * dc) + H * M * ((dc + A) * h + h * dc + dc * h + h * Dg)
            + loops * (H * m * Dg + 2 * H * M * Dg) + H * m * Dg)


def rollout_flops(arm, H):
    """Charging rule (PREREG_DRAFT_t5_t6 'equal planning compute'): cost of one imagined rollout."""
    if hasattr(arm, "rollout_flops"):
        return arm.rollout_flops(H)
    if isinstance(arm, ObserverAtlas):
        return _flops(arm, H, arm.m, 0)
    return H * arm.active_flops()


def budget_samples(arm, B, H, iters):
    """CEM samples per iteration that fit in a total multiply-add budget B."""
    return max(1, B // (iters * rollout_flops(arm, H)))


class TimeTravelAtlas(ObserverAtlas):
    def __init__(self, D, A, M=4, gamma=1.0, max_loops=1, **kw):
        super().__init__(D, A, **kw)
        if not self.m <= M <= self.K:
            raise ValueError("need m <= M <= K")
        self.M, self.gamma, self.max_loops = M, gamma, max_loops

    def rollout(self, obs0, act, return_info=False):
        M, m = self.M, self.m
        c = self.backbone(obs0[:, None])[0][:, 0]
        s0, cand = self.full_logits(c).topk(M, -1)  # START
        gate = torch.sigmoid(self.gate(c))
        z, fut = self.E(obs0[:, None].expand(-1, M, -1), cand), []
        for h in range(act.shape[1]):
            z = z + self.P(torch.cat([z, act[:, h, None].expand(-1, M, -1)], -1), cand)
            fut.append(self.phi(z, cand))
        F = torch.stack(fut, 1)  # [N, H, M, Dg]: every candidate's predicted future

        def mix(scores, S):
            w = scores.gather(-1, S).softmax(-1)[:, None, :, None]
            return (w * F.gather(2, S[:, None, :, None].expand(-1, F.shape[1], -1, F.shape[-1]))).sum(2)

        scores, S, loops = s0, s0.topk(m).indices, 0
        for i in range(self.max_loops):
            dev = ((F - mix(scores, S)[:, :, None]) ** 2).sum(-1).mean(1)  # [N, M]
            scores = s0 - self.gamma * dev
            S_new, loops = scores.topk(m).indices, i + 1
            fixed = bool((S_new.sort(-1).values == S.sort(-1).values).all())
            S = S_new
            if fixed:  # HALT: self-consistent chart set
                break
        out = gate[:, None] * mix(scores, S)
        return (out, dict(start=cand, halt=cand.gather(-1, S), loops=loops)) if return_info else out

    def rollout_flops(self, H):
        return _flops(self, H, self.M, self.max_loops)

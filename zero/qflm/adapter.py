"""lm/cores adapter around QFLM (imported, not edited): forward -> nll/steps/flops [B,T], step, n_params.
Emission is Kutner's QFLM._emit (one loop for forward/observe/generate); the adapter never copies it."""
import math

import torch

from .controls import DephasedQFLM, RealQFLM
from .model import QFLM

KINDS = {"quantum": QFLM, "dephased": DephasedQFLM, "real": RealQFLM}


class QF(torch.nn.Module):
    def __init__(self, vocab, d, N=None, arm="qflm", s_max=4, seed=0, kind="quantum", fixed_k=None, think_rank=None):
        super().__init__()
        N = N or d
        assert N & (N - 1) == 0, "N must be a power of 2"
        self.fixed_k = fixed_k                                    # None = adaptive; read at forward time
        self.m = KINDS[kind](N, vocab, s_max=s_max, arm=arm, seed=seed, think_rank=think_rank)

    def _flops(self, steps):
        return 2 * self.m.active_macs(rounds=steps + 1)          # FLOPs = 2 MACs; MAC model is Kutner's

    def _token(self, st, x):
        """-> (state', nll (B,), think steps (B,) chosen before x is seen, flops (B,))."""
        m = self.m
        if self.fixed_k is not None:                              # think k times, then one forced emission
            for _ in range(self.fixed_k):
                st = m._think(st)
            sm, m.s_max = m.s_max, 0                              # _emit at s_max=0 == forced emission, same post
            try:
                logp, nxt, _ = m._emit(st)[2](x)
            finally:
                m.s_max = sm
            k = torch.full(x.shape, float(self.fixed_k))
            # ponytail: Kutner's MAC model charges a measurement per round; fixed-k measures once, so this
            # over-counts k measurements (upper bound). Split active_macs into parts if (b)'s FLOP match is tight.
            return nxt, -logp, k, self._flops(k)
        _, rounds, fin = m._emit(st)
        logp, nxt, _ = fin(x)
        steps = rounds.to(logp.dtype)
        return nxt, -logp, steps, self._flops(steps)

    def _init(self, B):
        return self.m.psi0().expand(B, self.m.N)

    def forward(self, tokens):
        st, out = self._init(tokens.shape[0]), []
        for t in range(tokens.shape[1]):
            st, *r = self._token(st, tokens[:, t])
            out.append(r)
        return {k: torch.stack([o[i] for o in out], 1) for i, k in enumerate(("nll", "steps", "flops"))}

    def step(self, state, token):
        st, nll, _, _ = self._token(self._init(token.shape[0]) if state is None else state, token)
        return st, -nll.reshape(())

    def n_params(self):
        return self.m.n_params()

    def active_flops_per_token(self, seq=256):
        if self.fixed_k is not None:
            return 2 * self.m.active_macs(rounds=self.fixed_k + 1)
        return 2 * self.m.active_macs(worst_case=True)

"""Deciding controls for QF-LM: dephased HMM (state = distribution) and real-amplitude psi."""
import torch

from . import EPS, _w
from .model import QFLM, TAU
from .stability import null_record_logp


def classical_butterfly(b, theta):
    """Real distribution b [B,N] through the layerwise doubly-stochastic mixers [[c^2,s^2],[s^2,c^2]].
    Same pairing as qflm.butterfly; equals b @ |Q|^2.T (unique path per input/output pair)."""
    B, N = b.shape
    for k in range(theta.shape[-2]):
        s = 1 << k
        t = theta[..., k, :].reshape(*theta.shape[:-2], N // (2 * s), s)
        c2, s2 = torch.cos(t) ** 2, torch.sin(t) ** 2
        x = b.reshape(B, N // (2 * s), 2, s)
        a, d = x[:, :, 0], x[:, :, 1]
        b = torch.stack([c2 * a + s2 * d, s2 * a + c2 * d], 2).reshape(B, N)
    return b


class DephasedQFLM(QFLM):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        assert self.arm in ("qflm", "nothink", "nonull") and self.think_rank is None

    def psi0(self):
        """State is a distribution: the dephased start |psi0|^2 (QFLM._run/reset expand this)."""
        return _w(super().psi0())

    # Hooks use QFLM's names but act on a distribution b; QFLM's quantum _p/_post are never called here.
    def _dp(self, b, M):
        return classical_butterfly(b, self.theta) @ M

    def _dpost(self, b, M, v):
        out = M[:, v].T * classical_butterfly(b, self.theta)
        return out / out.sum(1, keepdim=True).clamp_min(1e-30)

    def _think(self, b, zt=None):
        if self.arm == "nothink":
            return b
        th0 = self.theta0
        return classical_butterfly(b, th0 + (b @ self.A_theta.T).reshape(-1, *th0.shape))

    def generate(self, *a, **kw):
        raise NotImplementedError("dephased control is scored by likelihood only; QFLM.generate assumes amplitudes")

    def _emit(self, b, ctx=None):
        """Classical twin of QFLM._emit (same TAU halting, forced emission, null-record sum). Differences:
        null update is the classical Bayes step and the post-state is the exact mixture sum w_s q_s / sum w_s.
        Overriding _emit routes forward/observe/dist/generate through it: no path reaches the quantum loop."""
        # ponytail: mirrors QFLM._emit line by line; tests/test_qflm_adapter.py dense reference catches drift.
        B, V, sm = b.shape[0], self.V, self.s_max
        M = self.M()
        nulls = torch.full((B,), V, dtype=torch.long)
        logP = torch.zeros(B, dtype=self.logits.dtype)
        alive = torch.ones(B, dtype=torch.bool)
        rounds = torch.zeros(B, dtype=torch.long)
        sts, lpc, lpn, lPs, als = [], [], [], [], []
        for s in range(sm + 1):
            if s:
                b = self._think(b)
            p = self._dp(b, M)
            forced = logP.exp() < TAU if s < sm else torch.ones_like(alive)
            den = torch.where(forced, (1 - p[:, V]).clamp_min(EPS), torch.ones_like(logP))
            sts.append(b); lPs.append(logP); als.append(alive)
            lpc.append((p[:, :V] / den[:, None]).clamp_min(1e-30).log())
            lpn.append(p[:, V].clamp_min(1e-30).log())
            rounds = torch.where(alive & forced, torch.full_like(rounds, s), rounds)
            alive = alive & ~forced
            logP = logP + lpn[-1]
            if s < sm:
                b = self._dpost(b, M, nulls)
        al = torch.stack(als, 1)
        lP, lc = torch.stack(lPs, 1), torch.stack(lpc, 1)
        marg = ((lP[:, :, None] + lc).exp() * al[:, :, None]).sum(1)
        lpn_t = torch.stack(lpn[:sm], -1) if sm else logP.new_zeros(B, 0)

        def fin(x):
            lx = lc.gather(2, x[:, None, None].expand(B, sm + 1, 1))[:, :, 0].masked_fill(~al, -1e30)
            logp = null_record_logp(lpn_t, lx)
            w = (lP + lx).exp() * al
            mix = sum(w[:, s, None] * self._dpost(sts[s], M, x) for s in range(sm + 1))
            return logp, mix / mix.sum(1, keepdim=True).clamp_min(1e-30), sts

        return marg, rounds, fin


class RealQFLM(QFLM):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        for n in ("phi", "phi0", "A_phi", "psi0_im"):
            z = torch.zeros_like(getattr(self, n))
            delattr(self, n)
            self.register_buffer(n, z)

"""Single JEPA (owned baseline): encoder D->d, residual predictor (z, a)->z', EMA target, SIGReg. g = z.

d is the T2 knob ({n, 2n, 2n+2}). Loss = (1-lam)*||P(z_t,a_t) - sg(E_tgt(o_{t+1}))||^2 + lam*SIGReg(z),
the LeJEPA mixing; lam = sigreg.LAM (N-free units, one table for every arm).
"""
import torch
import torch.nn.functional as F

from arms import LAM, Arm, frozen_copy, macs, mlp, sigreg


class SingleJEPA(Arm):
    def __init__(self, D: int, A: int, d: int, hidden: int = 128, depth: int = 2, lam: float = LAM,
                 tau: float = 0.996):
        super().__init__()
        self.lam, self.tau = lam, tau
        self.enc = self._mlp(D, hidden, d, depth)
        self.pred = self._mlp(d + A, hidden, d, depth)
        self.tgt_enc = frozen_copy(self.enc)

    def _mlp(self, i, h, o, depth):  # PEER-JEPA overrides
        return mlp(i, h, o, depth)

    def step(self, z, a):
        return z + self.pred(torch.cat([z, a], -1))

    def observe(self, obs, act):
        z = self.enc(obs)
        return z, self.step(z, act)[:, :-1], None  # predictor runs on every step; last one has no target

    def forward(self, obs, act):
        if self.training:
            self.ema_()
        g, ghat, w = self.observe(obs, act)
        with torch.no_grad():
            tz = self.tgt_enc(obs[:, 1:])
        pred = F.mse_loss(ghat, tz)
        sig = sigreg(g)
        return dict(loss=(1 - self.lam) * pred + self.lam * sig, g=g, ghat=ghat, w=w, w_soft=None, pred=pred, sig=sig)

    def rollout(self, obs0, act):
        z, out = self.enc(obs0), []
        for h in range(act.shape[1]):
            z = self.step(z, act[:, h])
            out.append(z)
        return torch.stack(out, 1)

    def active_flops(self) -> int:
        return macs(self.enc) + macs(self.pred)

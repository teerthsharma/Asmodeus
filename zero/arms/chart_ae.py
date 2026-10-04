"""Chart-Autoencoder-style multi-chart JEPA (Schonsheck, Chen, Lai, arXiv 1912.10094, Fig. 3 / Eq. 3), dense.

CAE: initial encoder E: x -> z in R^l; chart encoders E_k: z -> (-1,1)^d_c (tanh; CAE uses (0,1) - sigmoid charts
left seed 1 at erank(g) 1.75 on s2: decoders got too little signal from coordinates centred at 0.5); chart decoders D_k: chart -> R^l;
chart predictor P -> p = softmax over ALL charts; loss = min_k e_k - sum_k softmax(-e)_k log p_k.
JEPA adaptation (no pixel decoder; every target is a latent):
  e_k = ||D_k(z_k) - sg(E_tgt(o))||^2 + ||z_k + P_k(z_k, a) - sg(E_tgt_k(E_tgt(o')))||^2  (chart fidelity + dynamics)
  g = sum_k p_k D_k(z_k) in R^{D_g = l};  ghat = sum_k p_k D_k(zhat_k);  glue loss ||ghat - sg(g_{t+1})||^2
  loss = (1-lam)(min_k e_k + CE(softmax(-e), p) + glue) + lam (SIGReg(z) + SIGReg(g))
Every chart computes on every step (not lazy). P reads z (the paper allows x, z or z_k).
E ends in a parameter-free LayerNorm: without it SIGReg (shared LAM) inflates z to an isotropic 6-D cloud that
2-D charts cannot reconstruct, and g stalls at effective rank 1.6-2.0 < n (Chase, test_sigreg).
Skipped: CAE's Lipschitz penalty on chart encoders; add if charts blow up on T2.
"""
import torch
import torch.nn.functional as F
from torch import nn

from arms import LAM, LAM_BAL, Arm, Heads, frozen_copy, macs, mlp, sigreg


class ChartAEJEPA(Arm):
    chart_act = staticmethod(torch.tanh)
    ln_z = True  # CAE's E is a minimal-embedding step: fixed scale, so 2-D charts can reconstruct it
    def __init__(self, D: int, A: int, K: int = 4, d_c: int = 2, D_g: int = 8, hidden: int = 128,
                 chart_hidden: int = 32, depth: int = 2, lam: float = LAM, tau: float = 0.996, lam_bal: float = LAM_BAL):
        super().__init__()
        self.K, self.lam, self.tau, self.lam_bal = K, lam, tau, lam_bal
        self.E = mlp(D, hidden, D_g, depth)
        if self.ln_z:
            self.E.append(nn.LayerNorm(D_g, elementwise_affine=False))
        self.Ec = Heads(K, D_g, chart_hidden, d_c)
        self.Pc = Heads(K, d_c + A, chart_hidden, d_c)
        self.Dc = Heads(K, d_c, chart_hidden, D_g)  # gluing maps phi_k
        self.member = mlp(D_g, chart_hidden, K, depth=1)
        self.tgt_E, self.tgt_Ec = frozen_copy(self.E), frozen_copy(self.Ec)

    def _charts(self, z):  # z [N, Dg] -> chart coords [N, K, d_c]
        return self.chart_act(self.Ec(z))

    def _advance(self, zc, a):  # zc [N, K, d_c], a [N, A]
        return zc + self.Pc(torch.cat([zc, a[:, None].expand(-1, self.K, -1)], -1))

    def _parts(self, obs, act):
        B, T, _ = obs.shape
        z = self.E(obs).reshape(B * T, -1)
        zc = self._charts(z)
        p = self.member(z).softmax(-1)
        rec = self.Dc(zc)  # [N, K, Dg]
        zh = self._advance(zc, act.reshape(B * T, -1))
        g = (p[..., None] * rec).sum(1)
        gh = (p[..., None] * self.Dc(zh)).sum(1)
        v = lambda x: x.view(B, T, *x.shape[1:])
        return v(z), v(zc), v(p), v(rec), v(zh), v(g), v(gh)

    def observe(self, obs, act):
        _, _, p, _, _, g, gh = self._parts(obs, act)
        return g, gh[:, :-1], p

    def forward(self, obs, act):
        if self.training:
            self.ema_()
        z, zc, p, rec, zh, g, gh = self._parts(obs, act)
        with torch.no_grad():  # EMA targets: a detached online z is a moving target and diverges
            B, T, _ = obs.shape
            tz = self.tgt_E(obs)
            tzc = self.chart_act(self.tgt_Ec(tz[:, 1:].reshape(B * (T - 1), -1))).view(B, T - 1, self.K, -1)
        e = ((rec[:, :-1] - tz[:, :-1, None]) ** 2).sum(-1) + ((zh[:, :-1] - tzc) ** 2).sum(-1)  # [B,T-1,K]
        fid = e.min(-1).values.mean()
        chart = -(F.softmax(-e.detach(), -1) * p[:, :-1].clamp_min(1e-12).log()).sum(-1).mean()
        glue = F.mse_loss(gh[:, :-1], g[:, 1:].detach())
        sig = sigreg(z) + sigreg(g)
        pf = p.reshape(-1, self.K)
        bal = self.K * (F.one_hot(pf.argmax(-1), self.K).to(pf.dtype).mean(0) * pf.mean(0)).sum()  # Switch, as lazy
        loss = (1 - self.lam) * (fid + chart + glue + self.lam_bal * bal) + self.lam * sig
        return dict(loss=loss, g=g, ghat=gh[:, :-1], w=p, w_soft=p, fid=fid, chart=chart, glue=glue, sig=sig)

    def rollout(self, obs0, act):
        z = self.E(obs0)
        zc, p, out = self._charts(z), self.member(z).softmax(-1), []
        for h in range(act.shape[1]):
            zc = self._advance(zc, act[:, h])
            g = (p[..., None] * self.Dc(zc)).sum(1)
            p = self.member(g).softmax(-1)  # g lives in z's space (fidelity term), so P reads it
            out.append(g)
        return torch.stack(out, 1)

    def active_flops(self) -> int:
        per_chart = self.Ec.head_macs() + self.Pc.head_macs() + 2 * self.Dc.head_macs()
        return macs(self.E) + macs(self.member) + self.K * per_chart

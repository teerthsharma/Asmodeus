"""Lazy chart-AE JEPA: the cheapest ablation of the OBSERVER-ATLAS observer.

ChartAEJEPA (arms/chart_ae.py) with the atlas's lazy rule and nothing else: the chart-membership softmax
w_soft = softmax(P(E(o_t))) over all K is the router (O(K) work, counted); only the top-m charts run
E_k, P_k, D_k; w = w_soft restricted to the top-m and renormalised; g = sum_{top-m} w_k D_k(z_k).
No observer backbone, no gate, no learned keys. If this ties the atlas, the observer adds nothing.
Loss per selected chart as in ChartAEJEPA (min_k e_k, CE(softmax(-e), w), glue, SIGReg on z and g) plus
a Switch-style load balance K * sum_k f_k * mean(w_soft_k) so unselected charts keep a router gradient.
The t+1 target for chart k is read in the same chart k (chart chosen at t).
dense=True is the reference: every chart computes and the mixture uses w scattered to all K (zeros
outside the top-m); outputs equal the lazy path, and unselected charts are touched (0 * x).
"""
import torch
import torch.nn.functional as F

from arms import LAM, LAM_BAL, macs, sigreg
from arms.chart_ae import ChartAEJEPA


class LazyChartAEJEPA(ChartAEJEPA):
    dense = False

    def __init__(self, D: int, A: int, K: int = 8, m: int = 2, d_c: int = 2, D_g: int = 8, hidden: int = 128,
                 chart_hidden: int = 32, depth: int = 2, lam: float = LAM, lam_bal: float = LAM_BAL,
                 tau: float = 0.996):
        super().__init__(D, A, K=K, d_c=d_c, D_g=D_g, hidden=hidden, chart_hidden=chart_hidden, depth=depth,
                         lam=lam, tau=tau)
        self.m, self.lam_bal = m, lam_bal

    def _route(self, z):  # z [N, Dg] -> w_soft [N, K], idx [N, m], w [N, m]
        ws = self.member(z).softmax(-1)
        v, idx = ws.topk(self.m, -1)
        return ws, idx, v / v.sum(-1, keepdim=True)

    def _one(self, H, x, idx):  # x [N, m, i] -> selected head per slot [N, m, o]; only those heads compute
        return H.one(x.flatten(0, 1), idx.flatten()).view(*idx.shape, -1)

    def _lazy_coords(self, H, z, idx):
        return self.chart_act(self._one(H, z[:, None].expand(-1, self.m, -1), idx))

    def _lazy_step(self, zc, a, idx):
        return zc + self._one(self.Pc, torch.cat([zc, a[:, None].expand(-1, self.m, -1)], -1), idx)

    def _parts(self, obs, act):
        B, T, _ = obs.shape
        z, a = self.E(obs).reshape(B * T, -1), act.reshape(B * T, -1)
        ws, idx, w = self._route(z)
        if self.dense:
            sel = lambda x: x.gather(1, idx[..., None].expand(-1, -1, x.shape[-1]))
            zc_all = self._charts(z)
            zh_all = self._advance(zc_all, a)
            w_full = torch.zeros_like(ws).scatter(1, idx, w)
            rec_all, dh_all = self.Dc(zc_all), self.Dc(zh_all)
            zc, zh, rec = sel(zc_all), sel(zh_all), sel(rec_all)
            g, gh = (w_full[..., None] * rec_all).sum(1), (w_full[..., None] * dh_all).sum(1)
        else:
            zc = self._lazy_coords(self.Ec, z, idx)
            zh = self._lazy_step(zc, a, idx)
            rec = self._one(self.Dc, zc, idx)
            g, gh = (w[..., None] * rec).sum(1), (w[..., None] * self._one(self.Dc, zh, idx)).sum(1)
        v = lambda x: x.view(B, T, *x.shape[1:])
        return v(z), v(ws), v(idx), v(w), v(rec), v(zh), v(g), v(gh)

    def _w_full(self, idx, w):
        return torch.zeros(*idx.shape[:-1], self.K, dtype=w.dtype, device=w.device).scatter(-1, idx, w)

    def observe(self, obs, act):
        _, _, idx, w, _, _, g, gh = self._parts(obs, act)
        return g, gh[:, :-1], self._w_full(idx, w)

    def forward(self, obs, act):
        if self.training:
            self.ema_()
        z, ws, idx, w, rec, zh, g, gh = self._parts(obs, act)
        B, T, _ = obs.shape
        it = idx[:, :-1].reshape(-1, self.m)
        with torch.no_grad():
            tz = self.tgt_E(obs)
            tz1 = tz[:, 1:].reshape(-1, tz.shape[-1])
            if self.dense:
                tzc = self.chart_act(self.tgt_Ec(tz1)).gather(1, it[..., None].expand(-1, -1, zh.shape[-1]))
            else:
                tzc = self._lazy_coords(self.tgt_Ec, tz1, it)
        e = ((rec[:, :-1].reshape(-1, self.m, rec.shape[-1]) - tz[:, :-1].reshape(-1, 1, tz.shape[-1])) ** 2).sum(-1) \
            + ((zh[:, :-1].reshape(-1, self.m, zh.shape[-1]) - tzc) ** 2).sum(-1)  # [N', m]
        fid = e.min(-1).values.mean()
        chart = -(F.softmax(-e.detach(), -1) * w[:, :-1].reshape(-1, self.m).clamp_min(1e-12).log()).sum(-1).mean()
        glue = F.mse_loss(gh[:, :-1], g[:, 1:].detach())
        f = F.one_hot(idx.reshape(-1, self.m), self.K).to(ws.dtype).sum(1).mean(0) / self.m
        bal = self.K * (f * ws.reshape(-1, self.K).mean(0)).sum()
        sig = sigreg(z) + sigreg(g)
        loss = (1 - self.lam) * (fid + chart + glue + self.lam_bal * bal) + self.lam * sig
        return dict(loss=loss, g=g, ghat=gh[:, :-1], w=self._w_full(idx, w), w_soft=ws, fid=fid, chart=chart,
                    glue=glue, bal=bal, sig=sig)

    def rollout(self, obs0, act):
        z = self.E(obs0)
        _, idx, w = self._route(z)
        zc, out = self._lazy_coords(self.Ec, z, idx), []
        for h in range(act.shape[1]):
            zh = self._lazy_step(zc, act[:, h], idx)
            g = (w[..., None] * self._one(self.Dc, zh, idx)).sum(1)
            out.append(g)
            _, idx, w = self._route(g)  # g lives in z's space: re-route and re-enter the new charts
            zc = self._lazy_coords(self.Ec, g, idx)
        return torch.stack(out, 1)

    def active_flops(self) -> int:
        per_chart = self.Ec.head_macs() + self.Pc.head_macs() + 2 * self.Dc.head_macs()
        return macs(self.E) + macs(self.member) + self.m * per_chart

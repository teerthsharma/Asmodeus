"""MoP-JEPA (Song et al., arXiv 2607.05238v1, 2026-07-06; v3 was retitled Branch-JEPA). No public code.

Faithful to v1 §method:
  z = f(o)/||f(o)||, EMA target z+ = sg(norm(f_tgt(o'))), tau = 0.996
  K heads u_k = norm(g_k([z, a])), Linear-ReLU-Linear
  context-only router pi = softmax(r([z, a])), Linear-ReLU-Linear; never sees z+
  d_k = 1 - u_k . z+ ;  k* = argmin_k d_k ;  gamma = onehot(k*)  (hard winner-take-all, plain hard EM)
  L = mean d_{k*} + lam_route * CE(pi, k*) + lam_bal * sum_k gbar_k log(K gbar_k)
Deviations: (1) SIGReg on the pre-normalisation encoder output is added (PROMPT §3: anti-collapse
mandatory; paper has none; lam=0 restores the paper). (2) lam_route, lam_bal, widths unpublished: defaults
1.0 / 0.0. The balance term uses hard gamma, so it carries no gradient in the paper either; it is reported.
Inference: the paper emits all K (u_k, pi_k); the contract needs one ghat, so ghat = u_{argmax pi}
(the paper's router-top-1 readout). lazy=True runs only that head at inference (same outputs, 1/K head cost);
training always runs all K, because the argmin needs every head.
g = shared normalised z; w = w_soft = pi (no top-m truncation).
Published setting (v3, per zero/OWNERS.md): K=8, latent 512, 30k updates; encoder size/batch/lr unpublished
(chosen here, tune on the tuning seed). Headline 0.85 is planAll (BFS over the predicted transition graph),
not closed-loop success: match that protocol in T4b.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from arms import LAM, Arm, Heads, frozen_copy, macs, mlp, sigreg


class MoPJEPA(Arm):
    def __init__(self, D: int, A: int, d: int, K: int = 8, hidden: int = 128, head_hidden: int = 128,
                 router_hidden: int = 64, depth: int = 2, lam: float = LAM, lam_route: float = 1.0,
                 lam_bal: float = 0.0, tau: float = 0.996, lazy: bool = False):
        super().__init__()
        self.K, self.lam, self.lam_route, self.lam_bal, self.tau, self.lazy = K, lam, lam_route, lam_bal, tau, lazy
        self.enc = mlp(D, hidden, d, depth)
        self.heads = Heads(K, d + A, head_hidden, d)
        self.router = nn.Sequential(nn.Linear(d + A, router_hidden), nn.ReLU(), nn.Linear(router_hidden, K))
        self.tgt_enc = frozen_copy(self.enc)

    def _route_predict(self, z, a, all_heads):
        """z [N,d], a [N,A] -> pi [N,K], u_all [N,K,d] or None, u_top [N,d]."""
        c = torch.cat([z, a], -1)
        pi = self.router(c).softmax(-1)
        k = pi.argmax(-1)
        if all_heads:
            u = F.normalize(self.heads(c), dim=-1)
            return pi, u, u[torch.arange(len(k)), k]
        return pi, None, F.normalize(self.heads.one(c, k), dim=-1)

    def observe(self, obs, act):
        B, T, _ = obs.shape
        r = self.enc(obs)
        z = F.normalize(r, dim=-1)
        pi, _, u = self._route_predict(z.reshape(B * T, -1), act.reshape(B * T, -1), all_heads=not self.lazy)
        return z, u.view(B, T, -1)[:, :-1], pi.view(B, T, -1)

    def forward(self, obs, act):
        if self.training:
            self.ema_()
        B, T, _ = obs.shape
        r = self.enc(obs)
        z = F.normalize(r, dim=-1)
        c = torch.cat([z, act], -1)
        logits_all = self.router(c)  # [B, T, K]
        c, logits = c[:, :-1].reshape(B * (T - 1), -1), logits_all[:, :-1].reshape(B * (T - 1), -1)
        u = F.normalize(self.heads(c), dim=-1)  # [N, K, d]: all K needed for the assignment
        g, pi = z, logits_all.softmax(-1)
        ghat = u[torch.arange(len(u)), logits.argmax(-1)].view(B, T - 1, -1)
        with torch.no_grad():
            zp = F.normalize(self.tgt_enc(obs[:, 1:]), dim=-1).reshape(-1, 1, u.shape[-1])
        dist = 1 - (u * zp).sum(-1)  # [N, K] cosine distance
        kstar = dist.detach().argmin(-1)
        pred = dist.gather(1, kstar[:, None]).mean()  # only the winner's distance -> only the winner learns
        route = F.cross_entropy(logits, kstar)
        gbar = F.one_hot(kstar, self.K).float().mean(0)
        bal = (gbar * (self.K * gbar).clamp_min(1e-12).log()).sum()
        sig = sigreg(r)
        loss = (1 - self.lam) * (pred + self.lam_route * route + self.lam_bal * bal) + self.lam * sig
        return dict(loss=loss, g=g, ghat=ghat, w=pi, w_soft=pi, pred=pred, route=route, bal=bal, sig=sig,
                    kstar=kstar.view(obs.shape[0], -1))

    def rollout(self, obs0, act):
        return self._roll(F.normalize(self.enc(obs0), dim=-1), act)

    def _roll(self, z, act):
        out = []
        for h in range(act.shape[1]):
            _, _, z = self._route_predict(z, act[:, h], all_heads=not self.lazy)
            out.append(z)
        return torch.stack(out, 1) if out else z.new_zeros(len(z), 0, z.shape[-1])

    def rollout_modes(self, obs0, act):
        """Enumerate the K successor modes at the first step (the paper's one-pass mode set), then follow the
        router-argmax head. obs0 [B,D], act [B,H,A] -> (g [B,K,H,d], pi [B,K] first-step mode masses).
        Branch argmax(pi) equals rollout(). ponytail: branches only at step 1 (K paths, not K^H)."""
        z = F.normalize(self.enc(obs0), dim=-1)
        c = torch.cat([z, act[:, 0]], -1)
        pi, u = self.router(c).softmax(-1), F.normalize(self.heads(c), dim=-1)  # [B,K], [B,K,d]
        B, K, d = u.shape
        rest = self._roll(u.reshape(B * K, d), act[:, 1:].repeat_interleave(K, 0)).view(B, K, -1, d)
        return torch.cat([u[:, :, None], rest], 2), pi

    def active_flops(self) -> int:
        return macs(self.enc) + macs(self.router) + (1 if self.lazy else self.K) * self.heads.head_macs()


class MoPGraph:
    """MoP-JEPA in the T4b graph protocol (beds/ogbench_bed.py; interface of arms/controls_t4b.py):
    fit(ctx [M,2], nxt [M,2]) on action-marginalised cell transitions (A = 0, router reads z only);
    embed(xy) = unit-norm EMA-target latent (what the heads regress onto); predict(ctx) = (all K unit heads,
    pi). Edges are decoded by ogbench_bed.propose_edges (cos > .5, pi > .5/K), planAll by graph_metrics.
    Inputs are standardised with the fit-set mean/std (preprocessing, no maze structure)."""

    def __init__(self, K=8, d=16, steps=3000, batch=256, lr=1e-3, seed=0, **kw):
        self.K, self.d, self.steps, self.batch, self.lr, self.seed, self.kw = K, d, steps, batch, lr, seed, kw

    def _x(self, xy):
        return torch.as_tensor((np.asarray(xy, np.float64) - self.mu) / self.sd, dtype=torch.float32)

    def fit(self, ctx, nxt):
        pts = np.concatenate([ctx, nxt]).astype(np.float64)
        self.mu, self.sd = pts.mean(0), pts.std(0) + 1e-8
        torch.manual_seed(self.seed)
        self.model = MoPJEPA(D=2, A=0, d=self.d, K=self.K, **self.kw).train()
        obs = torch.stack([self._x(ctx), self._x(nxt)], 1)  # [M, 2, 2]
        act = obs.new_zeros(len(obs), 2, 0)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.steps):
            i = torch.randint(0, len(obs), (min(self.batch, len(obs)),), generator=g)
            opt.zero_grad()
            self.model(obs[i], act[i])["loss"].backward()
            opt.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def embed(self, xy):
        return F.normalize(self.model.tgt_enc(self._x(xy)), dim=-1).numpy()

    @torch.no_grad()
    def predict(self, ctx):
        z = F.normalize(self.model.enc(self._x(ctx)), dim=-1)
        return F.normalize(self.model.heads(z), dim=-1).numpy(), self.model.router(z).softmax(-1).numpy()

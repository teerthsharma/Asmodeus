"""EventDriven: AETHER 'observed -> exists' in time. Wraps any contract arm (common.py); a frame is computed
only if Delta_t = ||o_t - o_tau|| / sqrt(D) >= eps (tau = last trigger). Unobserved frames do not exist for the
arm: eval runs it once on the compacted triggers and maps back by sample-and-hold."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventDriven(nn.Module):
    def __init__(self, arm, eps_init=0.1, fixed_eps=None, temp=0.05, lam_hold=1.0, lam_cost=0.01):
        super().__init__()
        self.arm, self.temp, self.lam_hold, self.lam_cost = arm, temp, lam_hold, lam_cost
        if fixed_eps is None and eps_init in (0.0, float("inf")):
            fixed_eps = eps_init                                  # softplus cannot be exactly 0 / inf
        self.fixed_eps = fixed_eps
        e = 1.0 if fixed_eps is not None else eps_init
        self.rho = nn.Parameter(torch.tensor(math.log(math.expm1(e))))  # eps = softplus(rho)
        self._frac = 1.0

    def eps(self, like):
        if self.fixed_eps is not None:
            return torch.tensor(self.fixed_eps, dtype=like.dtype, device=like.device)  # not float32: ties are exact
        return F.softplus(self.rho).to(like.dtype)

    def triggers(self, obs, eps):
        """-> (trig [B,T] bool, delta [B,T]); delta is measured against the last HARD trigger."""
        ref, trig, delta = obs[:, 0], [torch.ones_like(obs[:, 0, 0], dtype=torch.bool)], [torch.zeros_like(obs[:, 0, 0])]
        for t in range(1, obs.shape[1]):  # ponytail: sequential scan over T (T <= 64); parallel scan if T grows
            d = (obs[:, t] - ref).norm(dim=-1) / math.sqrt(obs.shape[-1])
            tr = d >= eps
            ref = torch.where(tr[:, None], obs[:, t], ref)
            trig.append(tr)
            delta.append(d)
        return torch.stack(trig, 1), torch.stack(delta, 1)

    def forward(self, obs, act):
        eps = self.eps(obs)
        trig, delta = self.triggers(obs, eps)
        k = trig.long().cumsum(1) - 1                             # index of the last trigger <= t
        self._frac = float(trig.float().mean())
        if self.training and self.fixed_eps is None:
            out = self._soft(obs, act, trig, delta, eps, k)
        else:
            out = self._exact(obs, act, trig, k)
        return {**out, "trig": trig, "eps": eps.detach(), "exec_frac": trig.float().mean()}

    def _exact(self, obs, act, trig, k):
        B, T, Dm = obs.shape
        ar = torch.arange(T, device=obs.device).expand(B, T)
        n = trig.sum(1)
        Tc = max(int(n.max()), 2)                                 # ponytail: arms need T' >= 2 (ghat, SIGReg)
        pos = torch.where(trig, ar, T).sort(1).values[:, :Tc]
        pos = torch.where(pos == T, pos.gather(1, (n - 1)[:, None]).expand_as(pos), pos)  # pad: repeat last trigger
        gat = lambda x: x.gather(1, pos[..., None].expand(-1, -1, x.shape[-1]))
        # ponytail: actions between triggers are dropped; upgrade = integrate them into the action at the trigger
        r = self.arm(gat(obs), gat(act))
        hold = lambda x: x.gather(1, k.reshape(B, T, *[1] * (x.dim() - 2)).expand(-1, -1, *x.shape[2:]))
        out = {n_: hold(v) if torch.is_tensor(v) and v.shape[:2] == (B, Tc) else v for n_, v in r.items()}
        g = out["g"]
        gh = r["ghat"].gather(1, k[:, :-1].clamp(max=Tc - 2)[..., None].expand(-1, -1, g.shape[-1]))
        out["ghat"] = torch.where(trig[:, 1:, None], gh, g[:, :-1])  # held frame predicts no change
        return out

    def _soft(self, obs, act, trig, delta, eps, k):
        """Training relaxation: arm runs on the full sequence, straight-through gate mixes full and held outputs."""
        B, T, _ = obs.shape
        r = self.arm(obs, act)
        soft = torch.sigmoid((delta - eps) / self.temp)
        soft = torch.cat([torch.ones_like(soft[:, :1]), soft[:, 1:]], 1)
        s = (trig.to(soft.dtype) + soft - soft.detach())[..., None]
        g_full = r["g"]
        held = g_full.gather(1, torch.where(trig, torch.arange(T, device=obs.device), 0).cummax(1).values[..., None]
                             .expand_as(g_full))                  # g_full at the last hard trigger
        g = s * g_full + (1 - s) * held
        gh = s[:, 1:] * r["ghat"] + (1 - s[:, 1:]) * g[:, :-1]
        loss = (r["loss"] + self.lam_hold * F.mse_loss(g, g_full.detach()) + self.lam_cost * soft.mean())
        return {**r, "g": g, "ghat": gh, "loss": loss}

    def active_flops(self):
        return round(self.arm.active_flops() * self._frac)

    def executed_flops(self, trig):
        return int(trig.sum()) * self.arm.active_flops()

    def rollout(self, obs0, act):
        return self.arm.rollout(obs0, act)                        # a fresh frame always triggers

    def n_params(self):
        return self.arm.n_params() + 1

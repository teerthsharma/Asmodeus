"""OBSERVER-ATLAS (PROMPT.md §4): K tiny JEPAs with private charts, a gated softmax observer that
reads only the top-m of them, and SGD-learned gluing maps into R^{D_g}. Every parameter is learned (L-GIVEN).

Design decisions (conservative; PREREG must pin the open ones):
- Router sees the CURRENT observation. c_t = GRU(o_1..o_t): the GRU output at t has consumed o_t, so the
  observer can pick the chart that covers the current point. Strictly causal: nothing from t+1 enters c_t.
- ghat_{t+1} = gate_t * sum_{k in S_t} w_{t,k} phi_k(P_k(z_{t,k}, a_t)) uses the weights, chart set and gate of
  time t. Weights at t+1 depend on o_{t+1}, so using them would leak the target. The target g_{t+1} may
  use a different chart set S_{t+1}; crossing charts is exactly what the gluing maps must learn.
- Per-chart JEPA loss is UNWEIGHTED by w (weighting would let the router pick trivially predictable,
  i.e. collapsed, charts). Overlap consistency uses stop-grad weights for the same reason.
- Overlap term = sum_k w_k ||a_k - abar||^2 (weighted variance), which equals 1/2 sum_{j,k} w_j w_k ||a_j - a_k||^2:
  O(m) instead of O(m^2), and exactly 0 for m=1 (no pair, no 0/0).
- Loss in the arms' LeJEPA mixing form with the shared table (zero/sigreg.py: LAM, LAM_BAL):
  (1 - lam) * (jepa + task + lam_overlap * overlap + lam_bal * lb) + lam * (SIGReg per chart + SIGReg(g)).
  SIGReg per chart = shared sigreg grouped by chart id (one slice draw, O(N), charts with < 2 rows skipped).
  It does not standardise, so g = 0 (gate closed) gives a finite penalty, no 0/0. lam=0 is the T4 ablation.
- Gate: scalar sigmoid(W c_t + b) on g and ghat, so the observer can observe nothing (identities.out §3).
- Target encoder E_tgt = EMA of E, updated at the start of every forward with grad enabled in train mode
  (so any trainer gets it). It is a frozen copy: excluded from n_params().
- Lazy rule: charts run on gathered stacked weights [K,...][idx]; unselected rows are never read.
  'dense' mode computes all K charts and mixes with weights zeroed outside top-m (reference for parity
  tests); 'soft' mode mixes all K with the full softmax (T3 reference).
"""
import math
from itertools import combinations

import torch
import torch.nn.functional as F
from torch import nn

import common  # noqa: F401  pins TF32 off for every arm
from sigreg import LAM, LAM_BAL, sigreg  # shared with every arm: N-free Epps-Pulley + one lambda table



class StackedMLP(nn.Module):
    """K independent 2-layer MLPs i -> h -> o, weights stacked on a leading K axis."""

    def __init__(self, K, i, h, o):
        super().__init__()
        u = lambda *s, fan: nn.Parameter(torch.empty(*s).uniform_(-fan ** -0.5, fan ** -0.5))
        self.W1, self.b1, self.W2, self.b2 = u(K, i, h, fan=i), u(K, h, fan=i), u(K, h, o, fan=h), u(K, o, fan=h)

    def forward(self, x, idx):
        """x [N, m, i], idx [N, m] -> [N, m, o]; reads only rows idx of the stacked weights."""
        n, m, i = x.shape
        step = max(1, 2 ** 25 // (m * i * self.W1.shape[-1]))  # ponytail: chunk caps gathered-weight memory
        if step >= n:
            hid = F.silu(torch.einsum("nmi,nmih->nmh", x, self.W1[idx]) + self.b1[idx])
            return torch.einsum("nmh,nmho->nmo", hid, self.W2[idx]) + self.b2[idx]
        out = []
        for s in range(0, n, step):
            j, xs = idx[s:s + step], x[s:s + step]
            hid = F.silu(torch.einsum("nmi,nmih->nmh", xs, self.W1[j]) + self.b1[j])
            out.append(torch.einsum("nmh,nmho->nmo", hid, self.W2[j]) + self.b2[j])
        return torch.cat(out) if len(out) > 1 else out[0]

    def dense(self, x):
        """x [N, i] or [N, K, i] -> [N, K, o]: all K charts (reference path, FLOPs prop. to K)."""
        hid = torch.einsum("ni,kih->nkh" if x.dim() == 2 else "nki,kih->nkh", x, self.W1)
        return torch.einsum("nkh,kho->nko", F.silu(hid + self.b1), self.W2) + self.b2


def erank(x):
    """Effective rank exp(H(sigma / sum sigma)) of centred rows; 0 for an all-constant matrix."""
    s = torch.linalg.svdvals((x - x.mean(0)).double())
    if s.sum() <= 0:
        return 0.0
    p = s[s > 0] / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def nerve(w, tau, max_dim=4):
    """Simplices (frozensets of chart ids, closed under faces, dimension <= max_dim) of charts that all carry
    weight >= tau on a common input. w [..., K].
    Default input is forward()['w_soft'] (pre-top-m softmax over all K): top-m 'w' caps simplices at dim m-1,
    so m=2 gives a graph and b2 = b3 = 0 by construction (keep it for ablation only). Betti_k needs simplices
    up to dim k+1, hence max_dim=4 for b0..b3 (SO(3)); it also bounds the 2^v face blow-up at small tau.
    Multiply by forward()['gate'] if closed-gate inputs must not add simplices."""
    if tau <= 0:
        raise ValueError("tau must be > 0 (unselected charts carry weight 0)")
    out = set()
    for row in torch.unique(w.reshape(-1, w.shape[-1]) >= tau, dim=0):
        v = row.nonzero().flatten().tolist()
        for r in range(1, min(len(v), max_dim + 1) + 1):
            out.update(frozenset(s) for s in combinations(v, r))
    return out


def chart_diversity(w_soft, eps=1e-6):
    """Mean pairwise KL(rho_i || rho_j) over i < j, rho_k = chart k's occupancy over the N rows of w_soft [N,K].
    Paper's L_Div with diagonal density matrices; one [K,N]@[N,K] matmul, O(N K^2). K == 1 -> 0."""
    K = w_soft.shape[1]
    if K == 1:
        return w_soft.sum() * 0
    rho = (w_soft + eps) / (w_soft + eps).sum(0, keepdim=True)
    logr = rho.log()
    kl = (rho * logr).sum(0)[:, None] - rho.T @ logr  # kl[i, j] = KL(rho_i || rho_j)
    return kl.triu(1).sum() / (K * (K - 1) / 2)


class ObserverAtlas(nn.Module):
    # D_g = 2n+2 (PREREG ruling); runners pass it per bed
    def __init__(self, D, A, K=8, m=2, d_c=2, D_g=6, d_h=64, d_key=32, hidden=32, product_keys=False,
                 ema=0.996, lam=LAM, lam_bal=LAM_BAL, lam_overlap=0.1, sig_slices=256, time_travel=False, lam_div=0.0, lam_z=0.0):
        super().__init__()
        if time_travel:
            raise NotImplementedError("time-traveller re-scoring (§4.7) is T5 work, not built yet")
        self.side = math.isqrt(K)
        if product_keys and (self.side ** 2 != K or d_key % 2 or m > self.side):
            raise ValueError("product keys need K square, d_key even, m <= sqrt(K)")
        if not 1 <= m <= K:
            raise ValueError("need 1 <= m <= K")
        self.D, self.A, self.K, self.m, self.d_c, self.D_g = D, A, K, m, d_c, D_g
        self.d_h, self.d_key, self.hidden, self.pk = d_h, d_key, hidden, product_keys
        self.ema, self.sig_slices = ema, sig_slices
        self.lam, self.lam_bal, self.lam_overlap, self.lam_div = lam, lam_bal, lam_overlap, lam_div
        self.lam_z = lam_z  # router z-loss (ST-MoE, arXiv 2202.08906); 0 = off, PREREG decides
        self.backbone = nn.GRU(D, d_h, batch_first=True)  # whole sequences (training, observe)
        self.cell = nn.GRUCell(D, d_h)  # streaming path (step, rollout): same Parameter objects, not a copy
        for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            setattr(self.cell, n, getattr(self.backbone, n + "_l0"))
        self.W_q = nn.Linear(d_h, d_key, bias=False)
        if product_keys:
            self.sub_keys = nn.Parameter(torch.randn(2, self.side, d_key // 2) / math.sqrt(d_key // 2))
        else:
            self.keys = nn.Parameter(torch.randn(K, d_key) / math.sqrt(d_key))
        self.gate = nn.Linear(d_h, 1)
        self.E = StackedMLP(K, D, hidden, d_c)
        self.P = StackedMLP(K, d_c + A, hidden, d_c)
        self.phi = StackedMLP(K, d_c, hidden, D_g)
        self.E_tgt = StackedMLP(K, D, hidden, d_c).requires_grad_(False)
        self.E_tgt.load_state_dict(self.E.state_dict())

    # ---- observer ----------------------------------------------------------------------------
    def _scores(self, c):
        q = self.W_q(c)
        if not self.pk:
            return q @ self.keys.T, None
        q1, q2 = q.chunk(2, -1)
        return q1 @ self.sub_keys[0].T, q2 @ self.sub_keys[1].T

    def full_logits(self, c):
        s1, s2 = self._scores(c)
        full = s1 if s2 is None else (s1[..., :, None] + s2[..., None, :]).flatten(-2)
        return full / math.sqrt(self.d_key)

    def route(self, c):
        """-> (top-m logits, top-m chart ids). Product keys: exact top-m of s1_i + s2_j from m x m candidates."""
        s1, s2 = self._scores(c)
        if s2 is None:
            v, idx = s1.topk(self.m, -1)
        else:
            v1, i1 = s1.topk(self.m, -1)
            v2, i2 = s2.topk(self.m, -1)
            v, j = (v1[..., :, None] + v2[..., None, :]).flatten(-2).topk(self.m, -1)
            idx = i1.gather(-1, j // self.m) * self.side + i2.gather(-1, j % self.m)
        return v / math.sqrt(self.d_key), idx

    # ---- inference path (what active_flops counts) ------------------------------------------
    def _charts(self, o, a, idx):
        """o [N,D], a [N,A], idx [N,m] -> z, zhat [N,m,d_c], phi(z), phi(zhat) [N,m,D_g] (lazy)."""
        m = idx.shape[-1]
        z = self.E(o[:, None].expand(-1, m, -1), idx)
        zh = z + self.P(torch.cat([z, a[:, None].expand(-1, m, -1)], -1), idx)
        return z, zh, self.phi(z, idx), self.phi(zh, idx)

    def observe(self, obs, act, mode="lazy"):
        Bs, Ts, _ = obs.shape
        c = self.backbone(obs)[0].reshape(Bs * Ts, -1)
        return dict(self._read(c, obs.reshape(Bs * Ts, -1), act.reshape(Bs * Ts, -1), mode), B=Bs, T=Ts)

    def step(self, o, a, h=None):
        """Streaming one step: o [B,D], a [B,A], GRUCell state h [B,d_h] or None -> (g_t, ghat_{t+1}, top-m w), h.
        Equals observe() on the whole sequence (tested); nn.GRU at T=1 is ~3.4x slower than GRUCell (Taub)."""
        h = self.cell(o, h)
        r = self._read(h, o, a, "lazy")
        return dict(g=r["g"], ghat=r["gh"], w=r["w"], idx=r["idx"], gate=r["gate"]), h

    def _read(self, c, o, a, mode):
        """Observer + charts for N tokens with context c [N,d_h] (everything after the backbone)."""
        if mode == "soft":
            logits, idx = self.full_logits(c), torch.arange(self.K, device=c.device).expand(len(c), -1)
        else:
            logits, idx = self.route(c)
        w = logits.softmax(-1)
        gate = torch.sigmoid(self.gate(c))
        if mode == "lazy":
            z, zh, phz, phzh = self._charts(o, a, idx)
            # Fast weighted sum avoids einsum overhead
            g, gh = (w.unsqueeze(-1) * phz).sum(1), (w.unsqueeze(-1) * phzh).sum(1)
        elif mode in ("dense", "soft"):
            zall = self.E.dense(o)
            zhall = zall + self.P.dense(torch.cat([zall, a[:, None].expand(-1, self.K, -1)], -1))
            pall, phall = self.phi.dense(zall), self.phi.dense(zhall)
            wf = torch.zeros(len(c), self.K, dtype=w.dtype, device=w.device).scatter(-1, idx, w)
            g, gh = torch.einsum("nk,nkd->nd", wf, pall), torch.einsum("nk,nkd->nd", wf, phall)
            take = lambda x: x.gather(1, idx[..., None].expand(-1, -1, x.shape[-1]))
            z, zh, phz, phzh = take(zall), take(zhall), take(pall), take(phall)
        else:
            raise ValueError(mode)
        return dict(c=c, idx=idx, w=w, gate=gate, z=z, zh=zh, phz=phz, g=gate * g, gh=gate * gh)

    # ---- contract ----------------------------------------------------------------------------
    def forward(self, obs, act, mode="lazy"):
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                for pt, p in zip(self.E_tgt.parameters(), self.E.parameters()):
                    pt.lerp_(p, 1 - self.ema)
        r = self.observe(obs, act, mode)
        Bs, Ts, m, N = r["B"], r["T"], r["idx"].shape[-1], len(r["c"])
        idx = r["idx"].reshape(Bs, Ts, m)
        with torch.no_grad():  # stop-grad target Ebar_k(o_{t+1}) for the charts chosen at t
            o1, i0 = obs[:, 1:].reshape(-1, self.D), idx[:, :-1].reshape(-1, m)
            if mode == "lazy":
                tgt = self.E_tgt(o1[:, None].expand(-1, m, -1), i0)
            else:
                tgt = self.E_tgt.dense(o1).gather(1, i0[..., None].expand(-1, -1, self.d_c))
        zh = r["zh"].reshape(Bs, Ts, m, -1)[:, :-1].reshape(-1, m, self.d_c)
        g = r["g"].reshape(Bs, Ts, -1)
        gh = r["gh"].reshape(Bs, Ts, -1)[:, :-1]
        wd = r["w"].detach()[..., None]
        abar = (wd * r["phz"]).sum(1, keepdim=True)
        full = self.full_logits(r["c"])
        w_soft = full.softmax(-1)  # O(K) routing work, already counted; charts stay lazy
        p_full = w_soft.mean(0)
        f = torch.bincount(r["idx"].flatten(), minlength=self.K).to(p_full.dtype) / r["idx"].numel()
        terms = dict(
            jepa=((zh - tgt) ** 2).sum(-1).mean(),
            overlap=(wd[..., 0] * ((r["phz"] - abar) ** 2).sum(-1)).sum(-1).mean(),
            sig_chart=sigreg(r["z"].reshape(-1, self.d_c), r["idx"].flatten(), self.K, self.sig_slices),
            sig_g=sigreg(r["g"], None, 1, self.sig_slices),
            lb=self.K * (f * p_full).sum(),
            task=((gh - g[:, 1:].detach()) ** 2).sum(-1).mean(),
        )
        if self.lam_div != 0:  # O(N K^2): never computed when off
            terms["div"] = chart_diversity(w_soft)
        if self.lam_z != 0:
            terms["z"] = (torch.logsumexp(full, -1) ** 2).mean()
        t = terms
        loss = (1 - self.lam) * (t["jepa"] + t["task"] + self.lam_overlap * t["overlap"] + self.lam_bal * t["lb"]
                                 # div is REWARDED (minus): the paper's +gamma*L_Div, minimised, would pull
                                 # modules together; subtracting pushes duplicate charts apart. eps bounds
                                 # KL by about log(1/eps), so the reward cannot diverge.
                                 - (self.lam_div * t["div"] if self.lam_div != 0 else 0)
                                 + (self.lam_z * t["z"] if self.lam_z != 0 else 0)) + self.lam * (t["sig_chart"] + t["sig_g"])
        w = torch.zeros(N, self.K, dtype=r["w"].dtype, device=r["w"].device).scatter(-1, r["idx"], r["w"])
        return dict(loss=loss, g=g, ghat=gh, w=w.reshape(Bs, Ts, self.K), w_soft=w_soft.reshape(Bs, Ts, self.K),
                    gate=r["gate"].reshape(Bs, Ts, 1),
                    terms={k: v.detach() for k, v in terms.items()})

    def rollout(self, obs0, act):
        """h-step prediction from the first frame. Chart set, weights and gate are frozen at t=0: the
        observer has no observation to re-route on (re-routing on predictions is the §4.7 time-traveller)."""
        c = self.cell(obs0)
        logits, idx = self.route(c)
        w, gate = logits.softmax(-1), torch.sigmoid(self.gate(c))
        z = self.E(obs0[:, None].expand(-1, self.m, -1), idx)
        out = []
        for h in range(act.shape[1]):
            z = z + self.P(torch.cat([z, act[:, h, None].expand(-1, self.m, -1)], -1), idx)
            # Fast weighted sum avoids einsum overhead
            out.append(gate * (w.unsqueeze(-1) * self.phi(z, idx)).sum(1))
        return torch.stack(out, 1)

    def active_flops(self):
        """Multiply-adds per time step of observe() in lazy mode (training-only target/loss work excluded):
        GRU 3 d_h (D + d_h) + query d_h d_key + gate d_h + keys (K d_key | 2 sqrt(K) d_key/2)
        + m [E: D h + h d_c, P: (d_c + A) h + h d_c, 2 x phi: d_c h + h D_g] + mix 2 m D_g."""
        D, A, h, dc, Dg, m = self.D, self.A, self.hidden, self.d_c, self.D_g, self.m
        keys = 2 * self.side * (self.d_key // 2) if self.pk else self.K * self.d_key
        router = self.d_h * self.d_key + self.d_h + keys
        chart = D * h + h * dc + (dc + A) * h + h * dc + 2 * (dc * h + h * Dg)
        return 3 * self.d_h * (D + self.d_h) + router + m * chart + 2 * m * Dg

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # EMA target excluded

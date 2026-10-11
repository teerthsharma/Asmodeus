"""PHASE-SPHERE (ORIGIN.md §3): amplitudes psi in C^N, Born-rule decider, local unitary (Cayley) steps per chart,
and a closed-loop observer solved as a fixed point (start -> iterate -> halt). Every parameter is learned (L-GIVEN).

Conventions shared with atlas/observer_atlas.py: EMA target (`tgt_` prefix, excluded from n_params), rollout freezes
the chart set / weights / gate at t=0, ghat_{t+1} uses the weights, chart set and gate of time t, active_flops counts
multiply-adds per time step of the inference path (Linear in*out).

Design decisions:
- Loop candidates: the loop runs over the top loop_m = min(K, max(4, m)) charts of the Born start weights only
  (lazy: cost independent of K, K enters only through the K x N Born scores). The candidates' predicted futures
  y_c = V_c U_c(a) V_c^H psi are w-independent, so each loop iteration is mix -> normalise -> Born over candidates.
- Gauge: z_k = V_k^H psi e^{-i arg alpha_k}, so a global phase of psi cancels in z, in y (the phase is restored by
  e^{+i arg alpha_k}) and in every Born weight; g is invariant to it.
- The loop is a plain (undamped) iteration; non-convergence within loop_iters is reported in ctc_residual.
- Loss weights default to the shared mixing table of sigreg.py: (1 - LAM) * prediction terms + LAM * SIGReg(g).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

import common  # noqa: F401  pins TF32 off for every arm
from sigreg import LAM, LAM_BAL, sigreg


# ---------------- primitives (complex64 / complex128, batched over leading dims) ----------------
def born(alpha, null=None):
    """p = |alpha|^2 / sum |alpha|^2 over the last dim; `null` (shape alpha.shape[:-1]) is the LAST outcome."""
    if null is not None:
        alpha = torch.cat([alpha, null[..., None]], -1)
    e = alpha.real ** 2 + alpha.imag ** 2
    return e / e.sum(-1, keepdim=True).clamp_min(torch.finfo(e.dtype).tiny)


def fixed_point(f, x0, tol=1e-8, max_iter=500):
    """x <- f(x), batched over dim 0, rows freeze once max|f(x)-x| < tol. -> (x, iters, residual[B])
    with residual = max|f(x)-x| per row at return."""
    x = x0
    for i in range(max_iter):
        fx = f(x)
        res = (fx - x).abs().flatten(1).amax(1)
        done = res < tol
        if bool(done.all()):
            return x, i, res
        x = torch.where(done.view(-1, *[1] * (x.dim() - 1)), x, fx)
    return x, max_iter, (f(x) - x).abs().flatten(1).amax(1)


def skew_hermitian(X):
    return X - X.conj().transpose(-1, -2)


def cayley(G):
    """(I - G/2)^{-1}(I + G/2); unitary when G is skew-Hermitian."""
    eye = torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)
    return torch.linalg.solve(eye - G / 2, eye + G / 2)


def holonomy(psi_loop):
    """psi [..., L, N], closed loop (step L-1 -> 0 included): arg prod_t <psi_t|psi_{t+1}| in (-pi, pi]."""
    S = torch.angle((psi_loop.conj() * psi_loop.roll(-1, -2)).sum(-1)).sum(-1)
    h = torch.atan2(S.sin(), S.cos())
    return float(h) if h.dim() == 0 else h


def sign_holonomy(psi_loop):
    h = holonomy(psi_loop)
    return (1 if math.cos(h) >= 0 else -1) if isinstance(h, float) else torch.where(h.cos() >= 0, 1, -1)


def chern_fhs(psi, tris):
    """Fukui-Hatsugai-Suzuki: (1/2pi) sum_tri arg(<a|b><b|c><c|a>), triangles taken as oriented. psi [V,N], tris [F,3]."""
    a, b, c = psi[tris[:, 0]], psi[tris[:, 1]], psi[tris[:, 2]]
    ip = lambda x, y: (x.conj() * y).sum(-1)
    return float(torch.angle(ip(a, b) * ip(b, c) * ip(c, a)).sum() / (2 * math.pi))


def _unit(a, eps=1e-12):
    """a / |a| with a finite gradient at 0: e^{i arg a}."""
    return a / (a.real ** 2 + a.imag ** 2 + eps).sqrt()


def _rv(z):
    """complex [..., d] -> real view [Re, Im] [..., 2d]."""
    return torch.cat([z.real, z.imag], -1)


class PhaseSphere(nn.Module):
    def __init__(self, D, A, K=8, m=2, N=16, d_c=2, D_g=8, hidden=64, loop_iters=8, loop_tol=1e-4, ema=0.996,
                 lam=None):
        super().__init__()
        if not 1 <= m <= K:
            raise ValueError("need 1 <= m <= K")
        self.D, self.A, self.K, self.m, self.N, self.d_c, self.D_g = D, A, K, m, N, d_c, D_g
        self.hidden, self.loop_iters, self.loop_tol, self.ema = hidden, loop_iters, loop_tol, ema
        self.loop_m = min(K, max(4, m))
        lw = dict(jepa=1 - LAM, overlap=0.1 * (1 - LAM), task=1 - LAM, bal=LAM_BAL * (1 - LAM), kl=1 - LAM, sig=LAM)
        self.lam = {**lw, **(lam or {})}
        mlp = lambda: nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                    nn.Linear(hidden, 2 * N))
        self.enc = mlp()
        self.tgt_enc = mlp().requires_grad_(False)
        self.tgt_enc.load_state_dict(self.enc.state_dict())
        P = lambda *s, scale=1.0: nn.Parameter(torch.randn(*s) * scale)
        self.u_re, self.u_im, self.nu_re, self.nu_im = P(K, N), P(K, N), P(N), P(N)  # chart gauges + null gauge
        self.V_re, self.V_im = P(K, N, d_c, scale=N ** -0.5), P(K, N, d_c, scale=N ** -0.5)  # local frames
        self.Wx, self.bx = P(K, A, 2 * d_c * d_c, scale=0.1), nn.Parameter(torch.zeros(K, 2 * d_c * d_c))
        self.Wphi = nn.Parameter(torch.empty(K, 2 * d_c, D_g).uniform_(-1, 1) * (2 * d_c) ** -0.5)
        self.bphi = nn.Parameter(torch.zeros(K, D_g))
        self.head_h = 2 * N
        self.head = nn.Sequential(nn.Linear(2 * N, self.head_h), nn.GELU(), nn.Linear(self.head_h, K))

    # ---- parameters as complex objects ------------------------------------------------------
    def _u(self):
        u = torch.complex(self.u_re, self.u_im)
        return u / u.abs().pow(2).sum(-1, keepdim=True).sqrt()

    def _un(self):
        u = torch.complex(self.nu_re, self.nu_im)
        return u / u.abs().pow(2).sum().sqrt()

    def _V(self):
        return torch.complex(self.V_re, self.V_im)

    def _encode(self, enc, o):
        re, im = enc(o).chunk(2, -1)
        psi = torch.complex(re, im)
        return psi / (psi.real ** 2 + psi.imag ** 2).sum(-1, keepdim=True).add(1e-12).sqrt()

    def _U(self, a, idx):
        """a [n,A], idx [n,c] -> unitary steps [n,c,d_c,d_c]: cayley(skew_hermitian(X_k(a)))."""
        d = self.d_c
        # Fast batched matmul (a.unsqueeze(1).unsqueeze(-2) @ self.Wx[idx]).squeeze(-2) avoids einsum overhead
        X = (a.unsqueeze(1).unsqueeze(-2) @ self.Wx[idx]).squeeze(-2) + self.bx[idx]
        X = torch.complex(X[..., :d * d], X[..., d * d:]).reshape(*X.shape[:-1], d, d)
        return cayley(skew_hermitian(X))

    def _phi(self, z, idx):
        """z complex [n,c,d_c], idx [n,c] -> [n,c,D_g] per-chart linear on the real view."""
        # Fast batched matmul (_rv(z).unsqueeze(-2) @ self.Wphi[idx]).squeeze(-2) avoids einsum overhead
        return (_rv(z).unsqueeze(-2) @ self.Wphi[idx]).squeeze(-2) + self.bphi[idx]

    # ---- inference path (what active_flops counts) ------------------------------------------
    def observe(self, obs, act, mode="loop", phase=None):
        return self._observe(obs, act, mode, phase, with_head=False)

    def _observe(self, obs, act, mode, phase, with_head):
        if mode not in ("loop", "amortised"):
            raise ValueError(mode)
        B, T, _ = obs.shape
        psi = self._encode(self.enc, obs)
        if phase is not None:
            psi = psi * phase[..., None].to(psi.dtype)
        x, a, K, m = psi.reshape(B * T, -1), act.reshape(B * T, -1), self.K, self.m
        u = self._u()
        alpha = x @ u.conj().T                                         # <u_k|psi>   [BT,K]
        p = born(alpha, x @ self._un().conj())                         # K charts + "observe nothing" outcome
        gate = 1 - p[:, -1]
        w_soft = p[:, :K] / p[:, :K].sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
        logits = self.head(torch.cat([x.real, x.imag], -1)) if (mode == "amortised" or with_head) else None
        if mode == "amortised":
            vals, cidx = logits.topk(m, -1)
            w_soft = logits.softmax(-1)
        else:
            _, cidx = w_soft.detach().topk(self.loop_m, -1)
            vals = None
        # candidate charts (loop: loop_m, amortised: the m selected): gauge-fixed coords and local unitary step
        Vc = self._V()[cidx]                                           # [BT,c,N,d_c]
        ac = alpha.gather(1, cidx)
        ph = _unit(ac).conj()                                          # e^{-i arg alpha_k}
        # Fast batched matmuls avoid slow 4D einsums
        zc = (x.unsqueeze(1).unsqueeze(-2) @ Vc.conj()).squeeze(-2) * ph[..., None]
        zhc = (self._U(a, cidx) @ zc.unsqueeze(-1)).squeeze(-1)
        if mode == "amortised":
            w1 = vals.softmax(-1)
            wstar, iters, res = w1.detach(), 0, torch.zeros(B * T, device=x.device)
        else:
            y = (Vc @ zhc.unsqueeze(-1)).squeeze(-1) * ph.conj()[..., None]   # V U V^H psi (phase restored)
            uc = u[cidx]

            def step(w, y=y, uc=uc):                                   # CTC map: w -> Born(<u|normalise(sum w y)>)
                s = (w[..., None] * y).sum(1)
                s = s / (s.real ** 2 + s.imag ** 2).sum(-1, keepdim=True).add(1e-12).sqrt()
                return born(torch.einsum("bcn,bn->bc", uc.conj(), s))

            w0 = torch.gather(w_soft, 1, cidx).detach()
            w0 = w0 / w0.sum(-1, keepdim=True).clamp_min(torch.finfo(w0.dtype).tiny)
            with torch.no_grad():
                wstar, iters, _ = fixed_point(lambda w: step(w, y.detach(), uc.detach()), w0, self.loop_tol,
                                              self.loop_iters)
            w1 = step(wstar)                                           # ONE differentiable step from w*
            res = (w1.detach() - wstar).abs().amax(-1)
        ic = wstar.topk(m, -1)[1]                                      # final top-m from the converged weights
        zidx = cidx.gather(1, ic)
        w_top = w1.gather(1, ic)
        w_top = w_top / w_top.sum(-1, keepdim=True).clamp_min(torch.finfo(w_top.dtype).tiny)
        gz = lambda t: t.gather(1, ic[..., None].expand(-1, -1, t.shape[-1]))
        z, zh, atop = gz(zc), gz(zhc), ac.gather(1, ic)
        phz, phzh = self._phi(z, zidx), self._phi(zh, zidx)
        g = gate[:, None] * (w_top[..., None] * phz).sum(1)
        gh = gate[:, None] * (w_top[..., None] * phzh).sum(1)
        chi = torch.angle(atop[..., :, None] * atop[..., None, :].conj())
        r3 = lambda t: t.reshape(B, T, *t.shape[1:])
        dense = torch.zeros(B * T, K, dtype=w_top.dtype, device=x.device).scatter(1, zidx, w_top)
        return dict(g=r3(g), ghat=r3(gh)[:, :-1], w=r3(dense), w_soft=r3(w_soft), gate=r3(gate[:, None]), psi=psi,
                    z=r3(_rv(z)), zidx=r3(zidx), chi=r3(chi), ctc_iters=int(iters), ctc_residual=r3(res),
                    zh=r3(_rv(zh)), phz=r3(phz), w_top=r3(w_top), wstar=wstar, cidx=cidx, logits=logits,
                    zc=r3(z), x=x)

    # ---- contract ----------------------------------------------------------------------------
    def forward(self, obs, act, mode="loop"):
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                for pt, p in zip(self.tgt_enc.parameters(), self.enc.parameters()):
                    pt.lerp_(p, 1 - self.ema)
        r = self._observe(obs, act, mode, None, with_head=True)
        B, T, K, m = *obs.shape[:2], self.K, self.m
        zidx, w_top, g = r["zidx"], r["w_top"], r["g"]
        with torch.no_grad():  # stop-grad target: same chart k, gauge-fixed by the target's own alpha phase
            pt = self._encode(self.tgt_enc, obs[:, 1:])
            zi = zidx[:, :-1]
            at = torch.einsum("btmn,btn->btm", self._u()[zi].conj(), pt)
            tgt = _rv(torch.einsum("btmnd,btn->btmd", self._V()[zi].conj(), pt) * _unit(at).conj()[..., None])
        wd = w_top.detach()
        abar = (wd[..., None] * r["phz"]).sum(2, keepdim=True)
        f = torch.bincount(zidx.flatten(), minlength=K).to(g.dtype) / zidx.numel()
        if mode == "loop":
            lp = r["logits"].log_softmax(-1).gather(1, r["cidx"])
            kl = (r["wstar"] * (r["wstar"].clamp_min(1e-12).log() - lp)).sum(-1).mean()
        else:
            kl = g.sum() * 0
        terms = dict(
            jepa=((r["zh"][:, :-1] - tgt) ** 2).sum(-1).mean(),
            overlap=(wd * ((r["phz"] - abar) ** 2).sum(-1)).sum(-1).mean(),
            sig=sigreg(g),
            bal=K * (f * r["w_soft"].reshape(B * T, K).mean(0)).sum(),
            task=((r["ghat"] - g[:, 1:].detach()) ** 2).sum(-1).mean(),
            kl=kl,
        )
        loss = sum(self.lam[k] * v for k, v in terms.items())
        out = {k: r[k] for k in ("g", "ghat", "w", "w_soft", "gate", "psi", "z", "zidx", "chi", "ctc_iters",
                                 "ctc_residual")}
        return dict(loss=loss, **out, terms={k: v.detach() for k, v in terms.items()})

    def rollout(self, obs0, act):
        """h-step prediction from the first frame; chart set, weights and gate frozen at t=0 (as observer_atlas)."""
        r = self._observe(obs0[:, None], act[:, :1], "loop", None, with_head=False)
        idx, w, gate = r["zidx"][:, 0], r["w_top"][:, 0], r["gate"][:, 0]
        z = r["zc"][:, 0]
        out = []
        for h in range(act.shape[1]):
            z = torch.einsum("bcij,bcj->bci", self._U(act[:, h], idx), z)
            out.append(gate * (w[..., None] * self._phi(z, idx)).sum(1))
        return torch.stack(out, 1)

    def active_flops(self, mode="loop"):
        """Multiply-adds per time step of the inference path (training-only target/loss work excluded).
        The amortised head is counted only for mode='amortised' (loop mode does not execute it at inference)."""
        D, A, h, N, dc, Dg, m, K = self.D, self.A, self.hidden, self.N, self.d_c, self.D_g, self.m, self.K
        Lc = self.loop_m if mode == "loop" else m
        enc = D * h + h * h + h * 2 * N
        born_ = 4 * N * K + 4 * N                                      # K chart scores + null score
        chart = 4 * N * dc + A * 2 * dc * dc + 4 * dc ** 3 + 4 * dc * dc + 4 * N * dc   # V^H psi, X(a), cayley, U z, V zhat
        loop = (self.loop_iters + 1) * (6 * Lc * N + 2 * N) if mode == "loop" else 0   # mix + Born scores + norm
        head = 2 * N * self.head_h + self.head_h * K if mode == "amortised" else 0
        return int(enc + born_ + Lc * chart + loop + head + m * 4 * dc * Dg + 2 * m * Dg)

    def n_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("tgt_"))

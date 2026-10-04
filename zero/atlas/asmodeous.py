"""ASMODEOUS primitives; see ASMODEOUS.md"""
import torch
from torch import nn

from atlas.phase_sphere import _unit, cayley, fixed_point, skew_hermitian
from sigreg import LAM, LAM_BAL, sigreg


def _abs2(z):
    return z.real ** 2 + z.imag ** 2 if z.is_complex() else z ** 2


def stiefel(W):
    """W [K, N] (complex or real), K >= N -> isometry V with V^H V = I_N.

    Reduced QR, then each column of Q is multiplied by the phase (sign if real) of the matching diag(R) entry, so
    V^H W has a positive real diagonal. That fixes the QR gauge, making the map continuous and differentiable."""
    Q, R = torch.linalg.qr(W)
    d = torch.diagonal(R, dim1=-2, dim2=-1)
    if d.is_complex():
        ph = d / d.abs().clamp_min(torch.finfo(d.real.dtype).tiny)
    else:
        ph = torch.where(d >= 0, 1.0, -1.0).to(d.dtype)
    return Q * ph[..., None, :]


def povm_probs(V, psi):
    """V [K, N] isometry, psi [..., N] unit -> p [..., K] = |V psi|^2. Sums to 1 by V^H V = I; no renormalisation."""
    return _abs2(torch.einsum("kn,...n->...k", V, psi))


def povm_probs_subset(V, psi, idx):
    """p[..., j] = |V[idx[..., j]] psi|^2 for idx [..., m] (or [m]); m*N work, equal to gather(povm_probs, idx)."""
    return _abs2((V[idx] * psi.unsqueeze(-2)).sum(-1))


def cptp_fixed_point(F, w0, eps=0.1, tol=1e-10, max_iter=500):
    """Solve w = G(w), G(w) = (1 - eps) F(w) + eps * uniform, with phase_sphere.fixed_point. -> (w*, iters, residual).

    If F maps the simplex into the simplex and is 1-Lipschitz in L1, G is a (1-eps)-contraction, so the fixed point
    is unique (Banach) and iteration converges geometrically. eps = 0 recovers the undamped loop."""
    K = w0.shape[-1]
    return fixed_point(lambda w: (1 - eps) * F(w) + eps / K, w0, tol, max_iter)


def link_cayley(zj, zk, W):
    """zj, zk complex [..., d] -> unitary G [..., d, d] = cayley(skew_hermitian(X)).

    X = (real view [Re zj, Im zj, Re zk, Im zk] @ W) reshaped to a complex d x d; W is real [4d, 2*d*d]."""
    d = zj.shape[-1]
    v = torch.cat([zj.real, zj.imag, zk.real, zk.imag], -1)
    r = v @ W.to(v.dtype)
    X = torch.complex(r[..., :d * d], r[..., d * d:]).reshape(*r.shape[:-1], d, d)
    return cayley(skew_hermitian(X))


def cocycle_loss(G, zj, zk):
    """mean over batch of ||zk - G zj||^2."""
    return _abs2(zk - torch.einsum("...ij,...j->...i", G, zj)).sum(-1).mean()


def fidelity(a, b):
    """1 - |<a, b>|^2 for unit a, b [..., N]; invariant to a global phase on either."""
    return 1 - _abs2((a.conj() * b).sum(-1))


def haar_cp_stat(psi, n_dirs=64, generator=None):
    """Sliced Haar test: unit psi [..., N] in C^N uniform on CP^{N-1}.

    For random unit u, x = |<u, psi>|^2 ~ Beta(1, N-1) under Haar, CDF 1 - (1 - x)^(N-1). Returns the mean over
    directions and over a fixed 32-point grid on [0, 1] of the squared distance between the empirical CDF (sigmoid
    indicator, temperature 0.02, differentiable) and that CDF."""
    psi = psi.reshape(-1, psi.shape[-1])
    N = psi.shape[-1]
    u = torch.randn(n_dirs, N, 2, generator=generator, dtype=psi.real.dtype, device=psi.device)
    u = torch.complex(u[..., 0], u[..., 1])
    u = u / _abs2(u).sum(-1, keepdim=True).sqrt()
    x = _abs2(psi @ u.conj().T)                                       # [n, n_dirs]
    t = torch.linspace(0, 1, 32, dtype=x.dtype, device=x.device)[:, None, None]
    emp = torch.sigmoid((t - x) / 0.02).mean(1)                       # [32, n_dirs]
    ref = 1 - (1 - t[:, :, 0]) ** (N - 1)                             # [32, 1]
    return ((emp - ref) ** 2).mean()


# ---------------- the model ----------------
MODES = ("loop", "one_pass", "amortised")
_TINY = 1e-12


def _cx(t):
    """real or complex tensor -> complex (real-psi ablation runs the chart maths on a zero imaginary part)."""
    return t if t.is_complex() else torch.complex(t, torch.zeros_like(t))


def _rv(z):
    """complex [..., d] -> real view [Re, Im] [..., 2d]."""
    return torch.cat([z.real, z.imag], -1)


def _take(t, i):
    """t [n, c, ...], i [n, j] -> t[b, i[b, j], ...] [n, j, ...]."""
    return t.gather(1, i.reshape(*i.shape, *[1] * (t.dim() - 2)).expand(*i.shape, *t.shape[2:]))


def _norm(s):
    return s / _abs2(s).sum(-1, keepdim=True).add(_TINY).sqrt()


class Asmodeous(nn.Module):
    """ASMODEOUS: Parseval-frame POVM decider, lazy gauge-fixed charts, unitary local dynamics, gauge links and a
    damped CPTP observer loop (see ASMODEOUS.md slot table).

    Conventions worth knowing:
    - V (K+1 effects on C^N) is an isometry, so it needs K + 1 >= N (ValueError otherwise).
    - real=True makes psi, V and the encoder real (the Chern ablation); the chart maths still runs on complex
      tensors with a zero imaginary part, so psi_hat in the fidelity loss is complex-valued.
    - one_pass / amortised read the top-m of p (amortised: of the head logits). The loop runs over the M candidates
      (top-M of p), reads the top-m of w* among them, and mixes with w1 renormalised on the read set. Reported w is
      the exact unnormalised POVM probability on whatever is read.
    """

    def __init__(self, D, A, K=8, m=2, N=16, d_c=2, D_g=8, hidden=64, M=4, loop_iters=8, loop_eps=0.1,
                 loop_tol=1e-6, ema=0.996, lam=None, real=False, dense=False):
        super().__init__()
        if not 1 <= m <= K:
            raise ValueError("need 1 <= m <= K")
        if K + 1 < N:
            raise ValueError("the K+1 effects must form an isometry on C^N: need K + 1 >= N")
        M = min(M, K)
        if M < m:
            raise ValueError("need m <= M (loop candidates)")
        self.D, self.A, self.K, self.m, self.M, self.N, self.d_c, self.D_g = D, A, K, m, M, N, d_c, D_g
        self.hidden, self.loop_iters, self.loop_eps, self.loop_tol, self.ema = hidden, loop_iters, loop_eps, loop_tol, ema
        self.real, self.dense = real, dense
        lw = dict(fid=1 - LAM, chart=1 - LAM, cocycle=0.1 * (1 - LAM), task=1 - LAM, sig=LAM, haar=0.1 * (1 - LAM),
                  bal=LAM_BAL * (1 - LAM), distill=1 - LAM)
        self.lam = {**lw, **(lam or {})}
        out = N if real else 2 * N
        mlp = lambda: nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                    nn.Linear(hidden, out))
        self.enc = mlp()
        self.tgt_enc = mlp().requires_grad_(False)
        self.tgt_enc.load_state_dict(self.enc.state_dict())
        cs = () if real else (2,)                                      # complex parameters are stored as (re, im)
        self.frame_W = nn.Parameter(torch.randn(K + 1, N, *cs))
        self.chart_V = nn.Parameter(torch.randn(K, N, d_c, *cs) * N ** -0.5)
        self.Wx = nn.Parameter(torch.randn(K, A, 2 * d_c * d_c) * 0.1)
        self.bx = nn.Parameter(torch.zeros(K, 2 * d_c * d_c))
        self.Wphi = nn.Parameter(torch.empty(K, 2 * d_c, D_g).uniform_(-1, 1) * (2 * d_c) ** -0.5)
        self.bphi = nn.Parameter(torch.zeros(K, D_g))
        self.link_W = nn.Parameter(torch.randn(4 * d_c, 2 * d_c * d_c) * 0.1)
        self.amort_head = nn.Sequential(nn.Linear(out, hidden), nn.GELU(), nn.Linear(hidden, K))

    # ---- parameters as complex objects ------------------------------------------------------
    def V(self):
        """[K+1, N] isometry (V^H V = I); row K is the "observe nothing" effect."""
        return stiefel(self.frame_W if self.real else torch.view_as_complex(self.frame_W))

    def _Vc(self):
        return self.chart_V.to(torch.complex64) if self.real else torch.view_as_complex(self.chart_V)

    def _encode(self, enc, o):
        r = enc(o)
        psi = r if self.real else torch.complex(*r.chunk(2, -1))
        return psi / _abs2(psi).sum(-1, keepdim=True).add(_TINY).sqrt()

    def encode(self, obs):
        return self._encode(self.enc, obs)

    def link(self, zj, zk):
        return link_cayley(zj, zk, self.link_W)

    def _U(self, a, idx):
        """a [n,A], idx [n,c] -> unitary steps [n,c,d_c,d_c] = cayley(skew_hermitian(X_k(a)))."""
        d = self.d_c
        X = torch.einsum("na,ncaj->ncj", a, self.Wx[idx]) + self.bx[idx]
        X = torch.complex(X[..., :d * d], X[..., d * d:]).reshape(*X.shape[:-1], d, d)
        return cayley(skew_hermitian(X))

    def _phi(self, z, idx):
        """z complex [n,c,d_c], idx [n,c] -> [n,c,D_g], linear per chart on the real view."""
        return torch.einsum("nci,ncio->nco", _rv(z), self.Wphi[idx]) + self.bphi[idx]

    def _charts(self, x, a, alpha, idx):
        """Only the charts in idx [n,c]: gauge-fixed z, stepped zh, and the predicted ambient state y [n,c,N]."""
        Vc = self._Vc()[idx]                                           # [n,c,N,d_c]
        ph = _unit(alpha).conj()                                       # e^{-i arg alpha_k}
        z = torch.einsum("bcnd,bn->bcd", Vc.conj(), x) * ph[..., None]
        zh = torch.einsum("bcij,bcj->bci", self._U(a, idx), z)
        y = torch.einsum("bcnd,bcd->bcn", Vc, zh) * ph.conj()[..., None]   # phase restored
        return z, zh, y

    # ---- inference path (what active_flops counts) ------------------------------------------
    def _observe(self, obs, act, mode):
        if mode not in MODES:
            raise ValueError(mode)
        B, T, _ = obs.shape
        n, K, m, M = B * T, self.K, self.m, self.M
        psi = self.encode(obs)
        x, a = _cx(psi).reshape(n, -1), act.reshape(n, -1)
        Vq = _cx(self.V())
        alpha_all = x @ Vq.T                                           # V psi          [n, K+1]
        p = _abs2(alpha_all)                                           # Born probs, sum to 1 exactly
        pK, gate = p[:, :K], 1 - p[:, K]
        if mode == "amortised":
            hin = psi if self.real else _rv(psi)
            vals, cand = self.amort_head(hin.reshape(n, -1)).topk(m, -1)
        else:
            cand = pK.detach().topk(M if mode == "loop" else m, -1).indices
        if self.dense:                                                 # all K charts, then pick: same outputs
            z, zh, y = (_take(t, cand) for t in self._charts(x, a, alpha_all[:, :K], torch.arange(K).expand(n, K)))
        else:
            z, zh, y = self._charts(x, a, alpha_all.gather(1, cand), cand)
        if mode == "loop":
            # Column-stochastic channel P[d, c] = Born prob of candidate d on the normalised predicted state of
            # candidate c; F(w) = P w maps the simplex to itself and is 1-Lipschitz in L1, so the damped map is a
            # (1 - eps)-contraction by theorem (a coherent mix of the y_c is not nonexpansive: see the report).
            Pm = _abs2(torch.einsum("bdn,bcn->bdc", Vq[cand], _norm(y)))
            Pm = Pm / Pm.sum(1, keepdim=True).clamp_min(_TINY)
            F = lambda w, P: torch.einsum("bdc,bc->bd", P, w)

            w0 = pK.gather(1, cand).detach()
            w0 = w0 / w0.sum(-1, keepdim=True).clamp_min(_TINY)
            Pd = Pm.detach()
            with torch.no_grad():
                wstar, iters, _ = cptp_fixed_point(lambda w: F(w, Pd), w0, self.loop_eps, self.loop_tol,
                                                   self.loop_iters)
            w1 = (1 - self.loop_eps) * F(wstar, Pm) + self.loop_eps / M  # ONE differentiable step from w*
            res = (w1.detach() - wstar).abs().amax(-1)
            ic = wstar.topk(m, -1).indices                             # loop re-selects: top-m of w* in cand
            wmix = w1.gather(1, ic)
            wmix = wmix / wmix.sum(-1, keepdim=True).clamp_min(_TINY)
            iters = max(int(iters), 1)                                 # the convergence check is itself one F call
        else:
            ic = torch.arange(m).expand(n, m)
            wmix = pK.gather(1, cand) if mode == "one_pass" else vals.softmax(-1)
            wstar, iters, res = None, 0, torch.zeros(n)
        zidx = cand.gather(1, ic)
        w = pK.gather(1, zidx)                                         # exact POVM prob, NOT renormalised
        zr, zhr = _take(z, ic), _take(zh, ic)
        glue = lambda zz: gate[:, None] * (wmix[..., None] * self._phi(zz, zidx)).sum(1)
        r3 = lambda t: t.reshape(B, T, *t.shape[1:])
        return dict(g=r3(glue(zr)), ghat=r3(glue(zhr)), w=r3(torch.zeros(n, K).scatter(1, zidx, w)),
                    gate=r3(gate[:, None]), psi=psi, zidx=r3(zidx), z=r3(zr), zh=r3(zhr), wmix=r3(wmix),
                    psi_hat=r3(_norm((wmix[..., None] * _take(y, ic)).sum(1))),
                    alpha=r3(alpha_all.gather(1, zidx)), pK=pK, cand=cand, wstar=wstar, cand3=r3(cand), wstar3=None if wstar is None else r3(wstar), ctc_iters=iters,
                    ctc_residual=r3(res), n=n)

    # ---- contract ----------------------------------------------------------------------------
    def forward(self, obs, act, mode="loop"):
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                for pt, p in zip(self.tgt_enc.parameters(), self.enc.parameters()):
                    pt.lerp_(p, 1 - self.ema)
        r = self._observe(obs, act, mode)
        B, T, K, m = *obs.shape[:2], self.K, self.m
        g, zr, zh, zidx, alpha = r["g"], r["z"], r["zh"], r["zidx"], r["alpha"]
        ghat = r["ghat"][:, :-1]
        zero = g.sum() * 0
        # gauge links on co-read pairs, G_kk = I
        eye = torch.eye(m, dtype=torch.bool)
        zj, zk = (t.expand(B, T, m, m, self.d_c) for t in (zr[:, :, :, None], zr[:, :, None]))
        G = torch.where(eye[..., None, None], torch.eye(self.d_c, dtype=zr.dtype), self.link(zj, zk))
        chi = torch.angle(alpha[..., :, None] * alpha[..., None, :].conj())
        # losses
        fid = chart = task = zero
        if T > 1:
            with torch.no_grad():                                      # stop-grad EMA target at t+1
                pt = _cx(self._encode(self.tgt_enc, obs[:, 1:]))
                zi = zidx[:, :-1]
                at = torch.einsum("btmn,btn->btm", _cx(self.V())[zi], pt)
                tz = torch.einsum("btmnd,btn->btmd", self._Vc()[zi].conj(), pt) * _unit(at).conj()[..., None]
            fid = fidelity(r["psi_hat"][:, :-1], pt).mean()
            chart = _abs2(zh[:, :-1] - tz).sum(-1).mean()
            task = ((ghat - g[:, 1:].detach()) ** 2).sum(-1).mean()
        cocycle = cocycle_loss(G[:, :, ~eye], zj[:, :, ~eye], zk[:, :, ~eye]) if m > 1 else zero
        pK, f = r["pK"], torch.bincount(zidx.flatten(), minlength=K).to(g.dtype) / zidx.numel()
        soft = pK / pK.sum(-1, keepdim=True).clamp_min(_TINY)
        distill = zero
        if mode == "loop" and self.training:                           # halt before start: head learns w*
            hin = r["psi"].detach() if self.real else _rv(r["psi"].detach())
            lc = self.amort_head(hin.reshape(r["n"], -1)).gather(1, r["cand"]).log_softmax(-1)
            distill = (r["wstar"] * (r["wstar"].clamp_min(_TINY).log() - lc)).sum(-1).mean()
        terms = dict(fid=fid, chart=chart, cocycle=cocycle, task=task, sig=sigreg(g),
                     haar=zero if self.real else haar_cp_stat(r["psi"]),
                     bal=K * (f * soft.mean(0)).sum(), distill=distill)
        loss = sum(self.lam[k] * v for k, v in terms.items())
        return dict(loss=loss, g=g, ghat=ghat, w=r["w"], gate=r["gate"], psi=r["psi"], ctc_iters=r["ctc_iters"],
                    ctc_residual=r["ctc_residual"], phase=torch.angle(alpha).detach(), chi=chi.detach(), zidx=zidx,
                    link=G.detach(), cand=r["cand3"], wstar=r["wstar3"], terms={k: v.detach() for k, v in terms.items()})

    def rollout(self, obs0, act):
        """h-step prediction from the first frame; chart set, mixing weights and gate frozen at t=0."""
        r = self._observe(obs0[:, None], act[:, :1], "loop")
        idx, w, gate, z = r["zidx"][:, 0], r["wmix"][:, 0], r["gate"][:, 0], r["z"][:, 0]
        out = []
        for h in range(act.shape[1]):
            z = torch.einsum("bcij,bcj->bci", self._U(act[:, h], idx), z)
            out.append(gate * (w[..., None] * self._phi(z, idx)).sum(1))
        return torch.stack(out, 1)

    def active_flops(self, mode="loop"):
        """Multiply-adds per time step of the inference path (complex MAC = 4 real MACs).
        enc + decider + Lc * chart + (loop_iters + 1) * loop + readout [+ amortised head], where
          enc     = D*h + h*h + h*E                 (E = 2N, or N if real)
          decider = (K + 1) * 4N                    <- the ONLY K-dependent term (affine) in loop / one_pass
          chart   = 8*N*dc + A*2*dc^2 + 4*dc^3 + 4*dc^2   (V^H psi, X(a), cayley, U z, V zhat)
          Lc      = M (loop), m (one_pass, amortised), K (dense)
          loop    = (iters + 1) * M*M + M*(2*N + 4*N*M)   (channel P w each step; P built once); linear in iters
          readout = m * 4*dc*D_g + 2*m*D_g          (phi on z and zhat, then mix)
          head    = E*h + h*K  in mode 'amortised' only.  Training-only target/loss work is excluded."""
        D, A, h, N, dc, Dg, m, K, M = self.D, self.A, self.hidden, self.N, self.d_c, self.D_g, self.m, self.K, self.M
        E = N if self.real else 2 * N
        Lc = K if self.dense else (M if mode == "loop" else m)
        chart = 8 * N * dc + A * 2 * dc * dc + 4 * dc ** 3 + 4 * dc * dc
        loop = (self.loop_iters + 1) * M * M + M * (2 * N + 4 * N * M) if mode == "loop" else 0
        head = E * h + h * K if mode == "amortised" else 0
        return int(D * h + h * h + h * E + (K + 1) * 4 * N + Lc * chart + loop + head + m * 4 * dc * Dg + 2 * m * Dg)

    def n_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("tgt_"))

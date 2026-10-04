"""QFLM module: learned instrument M=softmax(logits), butterfly Q, think params, start state."""
import math
import torch
from . import EPS, _w
from .stability import null_record_logp, renorm_psi

TAU = 1e-3  # a row stops running null rounds once its remaining null mass P(null^s) < TAU
EPS_CTC = 1e-4  # CTC halting threshold on ||p_{s+1} - p_s||_1 (TO2 residual number)
RHO_MAX = 0.9  # PREREG Amendment 10: ctc contraction bound is <= RHO_MAX by construction
GRAD_CLIP = 1.0  # Luders gradient scales ~1/sqrt(p)


def _ang(theta, phi, N):
    """Per-layer butterfly factors (c, conj(e)*sin, e*sin) from theta, phi: (L,N/2) or (B,L,N/2)."""
    out = []
    for k in range(theta.shape[-2]):
        s = 1 << k
        t = theta[..., k, :].reshape(*theta.shape[:-2], N // (2 * s), s)
        e = torch.polar(torch.ones_like(t), phi[..., k, :].reshape(t.shape))
        sn = torch.sin(t)
        out.append((torch.cos(t), e.conj() * sn, e * sn))
    return out


def butterfly(psi, theta, phi, ang=None):
    """Apply Q in O(N log N); ang = precomputed _ang(theta, phi, N) (same maths, no recompute)."""
    B, N = psi.shape
    for k, (c, ecs, es) in enumerate(ang if ang is not None else _ang(theta, phi, N)):
        x = psi.reshape(B, N // (2 << k), 2, 1 << k)
        a, b = x[:, :, 0], x[:, :, 1]
        psi = torch.stack([c * a - ecs * b, es * a + c * b], 2).reshape(B, N)
    return psi


def _lud(q, sq):  # Luders post from Q psi and sqrt(M[:, v]) rows, renormalised
    out = sq * q
    return out / _w(out).sum(1, keepdim=True).clamp_min(EPS ** 2).sqrt()


def clip(model):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)


class QFLM(torch.nn.Module):
    def __init__(self, N, V, s_max=4, arm="qflm", seed=0, think_rank=None):
        super().__init__()
        assert arm in ("qflm", "nothink", "nonull", "embed", "softmax", "ctc", "ctc_hinge")
        self.is_ctc = arm in ("ctc", "ctc_hinge")  # ctc_hinge = ctc without the rescale (reported only)
        if self.is_ctc and think_rank is None:
            think_rank = 16  # ctc needs low-rank think factors A_*_U (L*N/2, r); default r = 16
        self.N, self.V, self.arm, self.think_rank = N, V, arm, think_rank
        self.s_max = 0 if arm == "nonull" else s_max
        L, g = int(math.log2(N)), torch.Generator().manual_seed(seed)
        U = lambda *s: (torch.rand(*s, generator=g) * 2 - 1) * math.pi  # Uniform(-pi, pi)
        P = torch.nn.Parameter
        self.logits = P(torch.randn(N, V + 1, generator=g) * 0.5)
        self.theta, self.phi = P(U(L, N // 2)), P(U(L, N // 2))
        self.theta0, self.phi0 = P(U(L, N // 2)), P(U(L, N // 2))
        r = think_rank
        for k in ("A_theta", "A_phi"):
            if r is None:
                setattr(self, k, P(torch.randn(L * N // 2, N, generator=g) * 0.01))
            else:  # A = U @ Vt (ctc: U only, angles read W p instead of |psi|^2)
                setattr(self, k + "_U", P(torch.randn(L * N // 2, r, generator=g) * 0.1))
                if not self.is_ctc:
                    setattr(self, k + "_Vt", P(torch.randn(r, N, generator=g) * 0.1))
        self.psi0_re = P(torch.randn(N, generator=g))
        self.psi0_im = P(torch.randn(N, generator=g) * 0.5)
        if arm == "embed":
            self.write_phase = P(torch.zeros(V, N))
        if self.is_ctc:
            self.ctc_W = P(torch.randn(r, V + 1, generator=g) * 0.1 / math.sqrt(V + 1))
        if arm == "softmax":
            self.R = P(torch.randn(N, V + 1, generator=g) * 0.01)
            self.b = P(torch.zeros(V + 1))
        if arm == "nothink":  # fixed identity think: angles 0, not learned
            self.register_buffer("_z", torch.zeros(L, N // 2))
            self.register_buffer("_zA", torch.zeros(L * N // 2, N))
        self.psi = None
        self.last_norm_dev = 0.0
        self._info = None

    def M(self):
        return torch.softmax(self.logits, 1)

    def psi0(self):
        z = torch.complex(self.psi0_re, self.psi0_im)
        return z / _w(z).sum().sqrt()

    def _tp(self):
        if self.arm == "nothink":
            return (self._z, self._z, self._zA, self._zA)
        return (self.theta0, self.phi0, self.A_theta, self.A_phi)

    def _ctx(self):
        """Per-call cache (built fresh each call, never stored): M, sqrt(M)^T, Q factors, nothink factors."""
        M = self.M()
        zt = _ang(self._z, self._z, self.N) if self.arm == "nothink" else None
        return M, M.sqrt().T, _ang(self.theta, self.phi, self.N), zt

    def _think(self, psi, zt=None):
        if self.arm == "nothink":
            return butterfly(psi, self._z, self._z, zt)
        w, sh = _w(psi), self.theta0.shape  # low rank: (w Vt^T) U^T, never forming U Vt
        if self.think_rank is None:
            lr = lambda At, _: w @ At.T
            th, ph = lr(self.A_theta, 0), lr(self.A_phi, 0)
        else:
            lr = lambda U, Vt: (w @ Vt.T) @ U.T
            th, ph = lr(self.A_theta_U, self.A_theta_Vt), lr(self.A_phi_U, self.A_phi_Vt)
        return butterfly(psi, self.theta0 + th.reshape(-1, *sh), self.phi0 + ph.reshape(-1, *sh))

    def _pq(self, q, M):  # (B, V+1) outcome distribution from q = Q psi
        if self.arm == "softmax":
            return torch.softmax(torch.log(_w(q) + 1e-9) @ self.R + self.b, 1)
        return _w(q) @ M

    def _postq(self, q, sqT, v):  # Luders update with K_v, then the +embed write phase
        out = _lud(q, sqT[v])
        if self.arm == "embed":
            wp = self.write_phase[v.clamp(max=self.V - 1)]
            out = out * torch.polar(torch.ones_like(wp), wp)
        return out

    def _p(self, psi):  # standalone (uncached) API for external steppers
        M, _, aq, _ = self._ctx()
        return self._pq(butterfly(psi, self.theta, self.phi, aq), M)

    def _post(self, psi, v):
        _, sqT, aq, _ = self._ctx()
        return self._postq(butterfly(psi, self.theta, self.phi, aq), sqT, v)

    # --- ctc arm: CTC fixed-point halting (PREREG Amendment 8, D1 + D6) ---
    def _ctc_AW(self):  # (L, N/2, V+1) RAW angle sensitivities (Ath W, Aph W)
        sh = (*self.theta0.shape, self.V + 1)
        return (self.A_theta_U @ self.ctc_W).reshape(sh), (self.A_phi_U @ self.ctc_W).reshape(sh)

    def rho_raw_t(self):
        """D6 bound (tensor, differentiable) on the RAW params: 2 * sum_l max_j (max_k|(Ath W)[l,j,k]| + max_k|(Aph W)[l,j,k]|).
        Proof: ||F(p)-F(p')||_1 <= || |a|^2 - |a'|^2 ||_1 (M row-stochastic) <= 2||a-a'||_2 (unit vectors)
        <= 2 sum_l ||R_l - R'_l||_2 (product of unitaries); each 2x2 block [[c,-conj(e)s],[e s,c]] moves by
        <= |dtheta|+|dphi| (partials have operator norm <= 1), and |dtheta_(l,j)| <= max_k|(Ath W)_(l,j),k| ||p-p'||_1.
        Certified iff < 1. Angles depend on p only (not psi), so the bound holds for every token."""
        at, ap = self._ctc_AW()
        return 2 * (at.abs().amax(2) + ap.abs().amax(2)).amax(1).sum()

    def rho_raw(self):
        return self.rho_raw_t().item()

    def _ctc_scale(self):
        """c = min(1, RHO_MAX / rho_raw) for arm ctc (differentiable, 0-guarded); 1 for ctc_hinge. The map used
        is F with angle sensitivities c*(Ath W, Aph W); rho_raw is homogeneous of degree 1 in them."""
        r = self.rho_raw_t()
        if self.arm != "ctc":
            return torch.ones_like(r)
        return torch.where(r > RHO_MAX, RHO_MAX / r.clamp_min(1e-30), torch.ones_like(r))

    def rho_global(self):
        """EFFECTIVE D6 bound of the map actually used: c * rho_raw = min(rho_raw, RHO_MAX) for arm ctc
        (rho_raw for ctc_hinge). This is what the certificate covers."""
        return (self._ctc_scale() * self.rho_raw_t()).item()

    def ctc_penalty(self, lam=1.0):
        """ctc_hinge training term lam * relu(rho_raw - RHO_MAX); added to the loss by the caller (forward stays NLL)."""
        return lam * torch.relu(self.rho_raw_t() - RHO_MAX)

    def certified(self):
        return self.is_ctc and self.rho_global() < 1

    def _ctc_F(self, p, psi, ctx, c=None):
        """F(p) = |Q U(p) psi|^2 @ M with U(p) the think butterfly at theta0 + Ath W p, phi0 + Aph W p.
        Returns (F(p), Q U(p) psi); the post-state of an emission from F(p) uses U(p), the unitary that produced it."""
        M, _, aq, _ = ctx
        a = (self._ctc_scale() if c is None else c) * (p @ self.ctc_W.T)
        sh = self.theta0.shape
        th = self.theta0 + (a @ self.A_theta_U.T).reshape(-1, *sh)
        ph = self.phi0 + (a @ self.A_phi_U.T).reshape(-1, *sh)
        q = butterfly(butterfly(psi, th, ph), self.theta, self.phi, aq)
        return _w(q) @ M, q

    def ctc_F(self, p, psi=None):
        psi = self.psi0().expand(p.shape[0], self.N) if psi is None else psi
        return self._ctc_F(p, psi, self._ctx())[0]

    def point_jacobian_l1(self, p, psi=None):
        """Diagnostic only (not a certificate): max column abs sum of dF/dp at p (V+1,); small N only."""
        psi = (self.psi0() if psi is None else psi).reshape(1, -1)
        J = torch.autograd.functional.jacobian(lambda x: self.ctc_F(x[None], psi)[0], p.detach())
        return J.abs().sum(0).max().item()

    def _emit_ctc(self, psi, ctx=None):
        """p0 = Born(psi); iterate p_{s+1} = F(p_s) (angles from p_s only); a row halts at the first s with
        ||p_{s+1}-p_s||_1 < EPS_CTC, only if the global certificate holds (else it runs s_max, flagged). Rounds =
        F applications (<= s_max); a capped row emits from the last (s_max-th) iterate, halted=False.
        p(x) = p[x]/(1-p[null]); post-state = renorm(K_x U(p_s) psi), U(p_s) the unitary that made the emitted p."""
        ctx = ctx or self._ctx()
        M, sqT, aq, _ = ctx
        B, V, sm = psi.shape[0], self.V, self.s_max
        q0 = butterfly(psi, self.theta, self.phi, aq)
        p = p0 = self._pq(q0, M)
        cert = self.certified()
        c = self._ctc_scale()
        pe, qe = p, q0
        alive = torch.ones(B, dtype=torch.bool, device=psi.device)
        rounds = torch.zeros(B, dtype=torch.long, device=psi.device)
        halted = torch.zeros_like(alive)
        d0 = resid = torch.zeros(B, device=psi.device, dtype=p.dtype)
        for s in range(sm):
            pn, q = self._ctc_F(p, psi, ctx, c)
            res = (pn - p).abs().sum(1)
            if s == 0:
                d0 = res.detach()
            stop = alive & ((res < EPS_CTC) if cert else torch.zeros_like(alive))
            last = alive & (stop | (s == sm - 1))
            pe, qe = torch.where(last[:, None], pn, pe), torch.where(last[:, None], q, qe)
            resid = torch.where(last, res.detach(), resid)
            rounds = torch.where(last, torch.full_like(rounds, s + 1), rounds)
            halted = halted | stop
            alive = alive & ~last
            p = torch.where(alive[:, None], pn, p)
        den = (1 - pe[:, V]).clamp_min(EPS)
        marg = pe[:, :V] / den[:, None]
        self._info = dict(rounds=rounds, halted=halted, resid=resid, d0=d0, p0=p0.detach(),
                          p_emit=pe.detach(), psi=psi.detach())

        def fin(x):
            lp = (pe.gather(1, x[:, None])[:, 0] / den).clamp_min(1e-30).log()
            return lp, renorm_psi(self._postq(qe, sqT, x)), [psi]

        return marg, rounds, fin

    def _emit(self, psi, ctx=None):
        """THE emission loop (forward, observe, dist, generate all use it). Returns (marginal (B,V),
        rounds (B,), fin) with fin(x) -> (log p(x) (B,), post-state f(psi,x) (B,N)).
        Round s: think (s>0), measure; row halts at s_max or once P(null^s) < TAU (forced emission)."""
        if self.is_ctc:
            return self._emit_ctc(psi, ctx)
        M, sqT, aq, zt = ctx or self._ctx()
        B, V, sm = psi.shape[0], self.V, self.s_max
        dev = psi.device
        nulls = torch.full((B,), V, dtype=torch.long, device=dev)
        logP = torch.zeros(B, device=dev, dtype=self.logits.dtype)
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        rounds = torch.zeros(B, dtype=torch.long, device=dev)
        sts, qs, lpc, lpn, lPs, als = [], [], [], [], [], []
        for s in range(sm + 1):
            if s:
                psi = self._think(psi, zt)
            q = butterfly(psi, self.theta, self.phi, aq)  # Q psi_s, once per round
            p = self._pq(q, M)
            forced = logP.exp() < TAU if s < sm else torch.ones_like(alive)
            den = torch.where(forced, (1 - p[:, V]).clamp_min(EPS), torch.ones_like(logP))
            sts.append(psi); qs.append(q); lPs.append(logP); als.append(alive)
            lpc.append((p[:, :V] / den[:, None]).clamp_min(1e-30).log())
            lpn.append(p[:, V].clamp_min(1e-30).log())
            rounds = torch.where(alive & forced, torch.full_like(rounds, s), rounds)
            alive = alive & ~forced
            logP = logP + lpn[-1]
            if s < sm:
                psi = _lud(q, sqT[nulls])
        al = torch.stack(als, 1)                                        # (B, S)
        lP, lc = torch.stack(lPs, 1), torch.stack(lpc, 1)                # (B,S) log P(null^s); (B,S,V) log p_s(x)
        marg = ((lP[:, :, None] + lc).exp() * al[:, :, None]).sum(1)
        lpn_t = torch.stack(lpn[:sm], -1) if sm else logP.new_zeros(B, 0)

        def fin(x):
            lx = lc.gather(2, x[:, None, None].expand(B, sm + 1, 1))[:, :, 0].masked_fill(~al, -1e30)
            logp = null_record_logp(lpn_t, lx)
            w = (lP + lx).exp().clamp_min(1e-30).sqrt() * al
            mix = sum(w[:, s, None] * self._postq(qs[s], sqT, x) for s in range(sm + 1))
            return logp, renorm_psi(mix), sts

        return marg, rounds, fin

    def _run(self, tokens):
        B, T = tokens.shape
        psi, ctx = self.psi0().expand(B, self.N), self._ctx()
        nl, st, ss = [], [], []
        for t in range(T):
            _, r, fin = self._emit(psi, ctx)
            lp, psi, _ = fin(tokens[:, t])
            nl.append(-lp); st.append(r); ss.append(psi)
        return torch.stack(nl, 1), torch.stack(st, 1), torch.stack(ss, 1)

    def forward(self, tokens, return_states=False):
        nl, _, ss = self._run(tokens)
        return (nl.mean(), ss) if return_states else nl.mean()

    def nll_per_token(self, tokens):
        return self._run(tokens)[0]

    @torch.no_grad()
    def steps_per_token(self, tokens):
        return self._run(tokens)[1]

    def _ctc_run(self, tokens, key):
        B, T = tokens.shape
        psi, ctx, out = self.psi0().expand(B, self.N), self._ctx(), []
        for t in range(T):
            _, _, fin = self._emit(psi, ctx)
            out.append(self._info[key])
            psi = fin(tokens[:, t])[1].detach()
        return torch.stack(out, 1)

    @torch.no_grad()
    def halted_per_token(self, tokens):  # (B,T) bool: halted by the residual test (ctc arm)
        return self._ctc_run(tokens, "halted")

    @torch.no_grad()
    def residual_per_token(self, tokens):  # (B,T) ||p_{s+1}-p_s||_1 at emission (ctc arm)
        return self._ctc_run(tokens, "resid")

    def n_params(self):
        return sum(p.numel() * (2 if p.is_complex() else 1) for n, p in self.named_parameters()
                   if not (self._think_name(n) and self.arm in ("nothink", "nonull")))

    @staticmethod
    def _think_name(n):
        return n in ("theta0", "phi0") or n.startswith(("A_theta", "A_phi"))

    def think_params_count(self):
        return sum(p.numel() for n, p in self.named_parameters() if self._think_name(n))

    def active_macs(self, rounds=None, worst_case=False):
        """Real multiply-accumulates per emitted token (complex multiply = 4 real MACs).
        worst_case: s_max+1 rounds; else `rounds` = mean rounds run (steps_per_token(...).mean()+1)."""
        if worst_case:
            rounds = self.s_max + 1
        elif rounds is None:
            raise ValueError("pass rounds= (mean measured rounds) or worst_case=True")
        N, V, L, r = self.N, self.V, int(math.log2(self.N)), self.think_rank
        bf = 8 * N * L
        if self.is_ctc:  # `rounds` = F applications (steps_per_token mean, no +1); worst case s_max
            if worst_case:
                rounds = self.s_max
            meas = 2 * N + N * (V + 1)  # |.|^2 + matvec with M
            F = r * (V + 1) + 2 * (L * N // 2) * r + 2 * bf + meas  # W p, Ath/Aph, think + Q butterflies
            return bf + meas + rounds * F + N + 2 * N  # round 0, F rounds, Luders post (N + renorm 2N)
        amap = 2 * (L * N // 2) * N if r is None else 2 * r * (L * N // 2 + N)
        meas, lud, thk = bf + N * (V + 1) + 2 * N, bf + N, bf + amap
        return rounds * (meas + lud) + (rounds - 1) * (lud + thk)

    # --- stateful API ---
    def reset(self, B=1):
        self.psi = self.psi0().detach().expand(B, self.N).clone()

    def dist(self):
        marg, rounds, _ = self._emit(self.psi)
        return marg, rounds

    def observe(self, v):
        self.psi = self._emit(self.psi)[2](v)[1]

    step = observe

    def think(self):
        self.psi = self._think(self.psi)

    @torch.no_grad()
    def generate(self, n_tokens, B=1, temperature=1.0, generator=None, return_steps=False, trace=None):
        self.reset(B)
        ctx = self._ctx()
        out, steps, dev = [], [], []
        for _ in range(n_tokens):
            marg, r, fin = self._emit(self.psi, ctx)
            x = torch.multinomial(torch.softmax(marg.clamp_min(1e-30).log() / temperature, 1), 1, generator=generator)[:, 0]
            _, self.psi, sts = fin(x)
            if trace is not None:
                trace.append(dict(post=self.psi.clone(), loop=torch.stack(sts), steps=r))
            dev.append((_w(self.psi).sum(1) - 1).abs().max())
            out.append(x); steps.append(r)
        self.last_norm_dev = torch.stack(dev).max().item() if dev else 0.0
        x, st = torch.stack(out, 1), torch.stack(steps, 1)
        return (x, st) if return_steps else x

"""QFLM module: learned instrument M=softmax(logits), butterfly Q, think params, start state."""
import math
import torch
from qflm import EPS, _w, butterfly, luders_update, probs, think_step
from qflm.stability import null_record_logp, renorm_psi

TAU = 1e-3  # a row stops running null rounds once its remaining null mass P(null^s) < TAU
GRAD_CLIP = 1.0  # Luders gradient scales ~1/sqrt(p)


def clip(model):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)


class QFLM(torch.nn.Module):
    def __init__(self, N, V, s_max=4, arm="qflm", seed=0, think_rank=None):
        super().__init__()
        assert arm in ("qflm", "nothink", "nonull", "embed", "softmax")
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
            else:  # A = U @ Vt
                setattr(self, k + "_U", P(torch.randn(L * N // 2, r, generator=g) * 0.1))
                setattr(self, k + "_Vt", P(torch.randn(r, N, generator=g) * 0.1))
        self.psi0_re = P(torch.randn(N, generator=g))
        self.psi0_im = P(torch.randn(N, generator=g) * 0.5)
        if arm == "embed":
            self.write_phase = P(torch.zeros(V, N))
        if arm == "softmax":
            self.R = P(torch.randn(N, V + 1, generator=g) * 0.01)
            self.b = P(torch.zeros(V + 1))
        if arm == "nothink":  # fixed identity think: angles 0, not learned
            self.register_buffer("_z", torch.zeros(L, N // 2))
            self.register_buffer("_zA", torch.zeros(L * N // 2, N))
        self.psi = None
        self.last_norm_dev = 0.0

    def M(self):
        return torch.softmax(self.logits, 1)

    def psi0(self):
        z = torch.complex(self.psi0_re, self.psi0_im)
        return z / _w(z).sum().sqrt()

    def _tp(self):
        if self.arm == "nothink":
            return (self._z, self._z, self._zA, self._zA)
        return (self.theta0, self.phi0, self.A_theta, self.A_phi)

    def _think(self, psi):
        if self.think_rank is None or self.arm == "nothink":
            return think_step(psi, self._tp())
        w, sh = _w(psi), self.theta0.shape  # low rank: (w Vt^T) U^T, never forming U Vt
        lr = lambda U, Vt: (w @ Vt.T) @ U.T
        return butterfly(psi, self.theta0 + lr(self.A_theta_U, self.A_theta_Vt).reshape(-1, *sh),
                         self.phi0 + lr(self.A_phi_U, self.A_phi_Vt).reshape(-1, *sh))

    def _p(self, psi):  # (B, V+1) outcome distribution
        if self.arm == "softmax":
            h = torch.log(_w(butterfly(psi, self.theta, self.phi)) + 1e-9)
            return torch.softmax(h @ self.R + self.b, 1)
        return probs(psi, self.M(), self.theta, self.phi)

    def _post(self, psi, v):  # Luders update with K_v, then the +embed write phase
        out = luders_update(psi, v, self.M(), self.theta, self.phi)
        if self.arm == "embed":
            wp = self.write_phase[v.clamp(max=self.V - 1)]
            out = out * torch.polar(torch.ones_like(wp), wp)
        return out

    def _emit(self, psi):
        """THE emission loop (forward, observe, dist, generate all use it). Returns (marginal (B,V),
        rounds (B,), fin) with fin(x) -> (log p(x) (B,), post-state f(psi,x) (B,N)).
        Round s: think (s>0), measure; row halts at s_max or once P(null^s) < TAU (forced emission)."""
        B, V, sm = psi.shape[0], self.V, self.s_max
        dev = psi.device
        nulls = torch.full((B,), V, dtype=torch.long, device=dev)
        logP = torch.zeros(B, device=dev, dtype=self.logits.dtype)
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        rounds = torch.zeros(B, dtype=torch.long, device=dev)
        sts, lpc, lpn, lPs, als = [], [], [], [], []
        for s in range(sm + 1):
            if s:
                psi = self._think(psi)
            p = self._p(psi)
            forced = logP.exp() < TAU if s < sm else torch.ones_like(alive)
            den = torch.where(forced, (1 - p[:, V]).clamp_min(EPS), torch.ones_like(logP))
            sts.append(psi); lPs.append(logP); als.append(alive)
            lpc.append((p[:, :V] / den[:, None]).clamp_min(1e-30).log())
            lpn.append(p[:, V].clamp_min(1e-30).log())
            rounds = torch.where(alive & forced, torch.full_like(rounds, s), rounds)
            alive = alive & ~forced
            logP = logP + lpn[-1]
            if s < sm:
                psi = luders_update(psi, nulls, self.M(), self.theta, self.phi)
        al = torch.stack(als, 1)                                        # (B, S)
        lP, lc = torch.stack(lPs, 1), torch.stack(lpc, 1)                # (B,S) log P(null^s); (B,S,V) log p_s(x)
        marg = ((lP[:, :, None] + lc).exp() * al[:, :, None]).sum(1)
        lpn_t = torch.stack(lpn[:sm], -1) if sm else logP.new_zeros(B, 0)

        def fin(x):
            lx = lc.gather(2, x[:, None, None].expand(B, sm + 1, 1))[:, :, 0].masked_fill(~al, -1e30)
            logp = null_record_logp(lpn_t, lx)
            w = (lP + lx).exp().clamp_min(1e-30).sqrt() * al
            mix = sum(w[:, s, None] * self._post(sts[s], x) for s in range(sm + 1))
            return logp, renorm_psi(mix), sts

        return marg, rounds, fin

    def _run(self, tokens):
        B, T = tokens.shape
        psi = self.psi0().expand(B, self.N)
        nl, st, ss = [], [], []
        for t in range(T):
            _, r, fin = self._emit(psi)
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
        out, steps, dev = [], [], []
        for _ in range(n_tokens):
            marg, r, fin = self._emit(self.psi)
            x = torch.multinomial(torch.softmax(marg.clamp_min(1e-30).log() / temperature, 1), 1, generator=generator)[:, 0]
            _, self.psi, sts = fin(x)
            if trace is not None:
                trace.append(dict(post=self.psi.clone(), loop=torch.stack(sts), steps=r))
            dev.append((_w(self.psi).sum(1) - 1).abs().max())
            out.append(x); steps.append(r)
        self.last_norm_dev = torch.stack(dev).max().item() if dev else 0.0
        x, st = torch.stack(out, 1), torch.stack(steps, 1)
        return (x, st) if return_steps else x

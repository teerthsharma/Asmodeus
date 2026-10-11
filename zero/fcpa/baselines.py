"""Baselines of Amendment 11: Global (one model, matched parameters) and the ASMODEOUS top-m adapter.

Both expose the same closed-loop surface, `step(X) -> X'` on the raw (unnormalised) encoded state, so every arm is
scored on identical targets. MACs follow bars.py (real multiply-adds; backward counted at 2x forward).
"""
import numpy as np

from . import bars
from .eras import EraModel


# ------------------------------------------------------------------ Global: one era-model family member, no router
def param_count(d, p, width):
    """Fitted parameters of GlobalModel: readout W (d + width, p) plus bias (p). Fixed random features are not counted."""
    return (d + width + 1) * p


def solve_width(target, d, p, tol=0.02):
    """Integer search (bisection on the monotone count) for the width whose count is within `tol` of `target`."""
    lo, hi = 0, 1
    while param_count(d, p, hi) < target:
        hi *= 2
    while lo < hi:  # smallest width with count >= target
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if param_count(d, p, mid) >= target else (mid + 1, hi)
    best = min((w for w in (lo - 1, lo) if w >= 0), key=lambda w: abs(param_count(d, p, w) - target))
    if abs(param_count(d, p, best) - target) > tol * target:
        raise ValueError(f"no width within {tol:.0%} of {target} parameters (nearest {param_count(d, p, best)})")
    return best


def global_macs(n, d, p, width, with_solve=True):
    """Fit cost: features n d w, Gram n F^2, cross-moment n F p, plus one closed-form solve; F = d + width."""
    F = d + width
    return n * (d * width + F * F + F * p) + (bars.macs_solve(F + 1, p) if with_solve else 0)


def global_n_afford(budget, d, p, width, n_max):
    """Largest n <= n_max whose fit stays within the budget (C2); 0 if even the solve does not fit."""
    fixed, per = global_macs(0, d, p, width), global_macs(1, d, p, width, with_solve=False)
    return int(np.clip((budget - fixed) // per, 0, n_max))


class GlobalModel:
    """The JEPA-in-slab family member (ridge readout with isotropy shrinkage and collapse guard: eras.EraModel) on
    [x, tanh(x W + b)] with `width` random features, trained on every era and with no router and no birth."""

    def __init__(self, d, p, width, seed=0, lam=1e-2):
        g = np.random.default_rng(seed)
        self.d, self.p, self.width = d, p, width
        self.Wf = g.standard_normal((d, width)) / np.sqrt(d)
        self.bf = g.standard_normal(width)
        self.era = EraModel(lam=lam)
        self.scale = 1.0

    def _feat(self, X):
        Xs = np.asarray(X, float) / self.scale
        return np.c_[Xs, np.tanh(Xs @ self.Wf + self.bf)]

    def fit(self, X, Y):
        self.scale = float(np.sqrt(np.mean(np.asarray(X, float) ** 2))) or 1.0
        self.era.fit(self._feat(X), np.asarray(Y, float) / self.scale)
        return self

    def predict(self, X):
        return self.era.predict(self._feat(X)) * self.scale

    step = predict

    @property
    def n_params(self):
        return param_count(self.d, self.p, self.width)

    def inference_macs(self):
        return self.d * self.width + (self.d + self.width) * self.p

    def fit_macs(self, n):
        """global_macs, with the guard's extra solves (each lam/10 retry) counted at their cost."""
        tries = max(1, int(round(np.log10(self.era.lam / self.era.lam_eff_))) + 1)
        return global_macs(n, self.d, self.p, self.width) + (tries - 1) * bars.macs_solve(self.d + self.width + 1, self.p)


# ------------------------------------------------------------------ ASMODEOUS top-m, end to end, no era labels
def _asmodeous(D, K, m, hidden, device="cpu"):
    from atlas.asmodeous import Asmodeous

    m = min(m, K)
    return Asmodeous(D=D, A=1, K=K, m=m, N=min(16, K + 1), M=max(min(4, K), m), hidden=hidden).to(device)


def asmodeous_macs_by_m(D, K, hidden=64, D_out=None):
    """Mean inference MACs per step of the arm (net + linear readout probe) for m = 1..4 (P2's compute rule)."""
    out = {}
    for m in (1, 2, 3, 4):
        net = _asmodeous(D, K, m, hidden)
        out[m] = int(net.active_flops("loop")) + (2 * net.N + net.D_g) * (D if D_out is None else D_out)
    return out


class AsmodeousArm:
    """atlas/asmodeous.py trained on windows of the raw encoded stream (D observed dims, one dummy action), with
    the router and top-m selection learned end to end (no era labels). To be scored in the common space, a ridge
    probe (EraModel, closed form) maps the net's predicted next latent state [psi_hat, g_hat] to the next x."""

    def __init__(self, D, K, m, lr, seed=0, device="cpu", batch=64, window=8, hidden=64, D_out=None):
        import torch

        torch.manual_seed(seed)
        self.seed = seed
        self.torch, self.D, self.K, self.m, self.lr = torch, D, K, min(m, K), lr
        self.D_out = D if D_out is None else D_out  # observed rows may be lag stacks; the probe targets the first D_out dims
        self.device, self.batch, self.window = device, batch, window
        self.net = _asmodeous(D, K, m, hidden, device)
        self.feat_dim = 2 * self.net.N + self.net.D_g
        self.rng = np.random.default_rng(seed)
        self.dec = EraModel(lam=1e-3)
        self.scale, self.steps_done = 1.0, 0

    def _t(self, a):
        return self.torch.as_tensor(np.ascontiguousarray(a), dtype=self.torch.float32, device=self.device)

    def fit(self, trajs, steps, probe_pairs=20_000):
        """`steps` Adam steps (batch x window) on trajs (n, L, D), then the closed-form probe on up to probe_pairs pairs."""
        self.torch.manual_seed(self.seed)  # the net draws torch randoms (Haar/SIGReg directions) while training
        trajs = np.asarray(trajs, float)
        self.scale = float(np.sqrt(np.mean(trajs ** 2))) or 1.0
        Xs = trajs / self.scale
        n, L, _ = Xs.shape
        opt = self.torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for _ in range(steps):
            i = self.rng.integers(0, n, self.batch)
            s = self.rng.integers(0, L - self.window + 1, self.batch)
            obs = self._t(np.stack([Xs[a, b:b + self.window] for a, b in zip(i, s)]))
            out = self.net(obs, self.torch.zeros(self.batch, self.window, 1, device=self.device))
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            self.steps_done += 1
        cur, nxt = Xs[:, :-1].reshape(-1, self.D), Xs[:, 1:, :self.D_out].reshape(-1, self.D_out)
        keep = self.rng.permutation(len(cur))[:probe_pairs]
        self.probe_n = len(keep)
        self.dec.fit(self._feats(cur[keep]), nxt[keep])
        return self

    def _feats(self, Xs):
        self.net.eval()
        with self.torch.no_grad():
            r = self.net._observe(self._t(Xs)[:, None], self.torch.zeros(len(Xs), 1, 1, device=self.device), "loop")
            ph, gh = r["psi_hat"][:, 0], r["ghat"][:, 0]
            return self.torch.cat([ph.real, ph.imag, gh], -1).cpu().numpy().astype(float)

    def step(self, X):
        return self.dec.predict(self._feats(np.asarray(X, float) / self.scale)) * self.scale

    @property
    def n_params(self):
        return self.net.n_params() + self.dec.W_.size + self.dec.b_.size

    def inference_macs(self):
        return int(self.net.active_flops("loop")) + self.feat_dim * self.D_out

    def train_step_macs(self):
        return bars.macs_bwd(int(self.net.active_flops("loop"))) * self.batch * self.window

    def probe_pair_macs(self):
        """Per probe pair: net forward, Gram, cross-moment."""
        F = self.feat_dim
        return int(self.net.active_flops("loop")) + F * F + F * self.D_out

    def probe_macs(self, n=None):
        """Closed-form probe: n pairs plus one solve (guard retries ignored)."""
        n = self.probe_n if n is None else n
        return n * self.probe_pair_macs() + bars.macs_solve(self.feat_dim + 1, self.D_out)

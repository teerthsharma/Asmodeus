"""HOPF-CHART: one global chart on CP^{N-1}, no observer, no router (ASMODEOUS.md, "Hopf chart").

An encoder maps one frame to psi in C^N (a point of S^{2N-1}); an action step is the unitary
U = cayley(skew_hermitian(X(a, psi))); the observable g = bloch(psi) is the real traceless part of psi psi^H, which is
invariant to the global phase of psi. For N = 2 it is exactly the Hopf map S^3 -> S^2 (the Bloch vector).

bloch coordinates: g_a = tr(rho L_a) with tr(L_a L_b) = 2 delta_ab (generalised Gell-Mann scale), rho = psi psi^H:
  [2 Re rho_ji, 2 Im rho_ji for i < j (row-major; rho_ji = conj(psi_i) psi_j), sqrt(2) * H p]   with p = |psi|^2 and H the (N-1) x N Helmert
matrix (orthonormal rows, orthogonal to the ones vector). For a pure state ||g||^2 = 2 (1 - 1/N): 1 for N = 2.
D_g = N^2 - 1. bloch is 2-Lipschitz in psi.

Conventions shared with atlas/phase_sphere.py: EMA target named with the `tgt_` prefix and excluded from n_params;
the loss mixes (1 - LAM) * (fid + task) + LAM * sigreg(g); structure is learned from (obs, act) alone (L-GIVEN).
Losses: fid = mean(1 - |<psi_hat_{t+1}, sg tgt_psi_{t+1}>|^2) (phase-invariant), task = MSE(ghat, sg g_{t+1}).

active_flops (multiply-adds per time step of the inference path):
  enc    D*h + h*h + h*2N
  X map  (A + 2N) * 2N^2
  cayley 8 N^3   (nominal: complex solve of an N x N system with N right-hand sides; the exact LAPACK count differs)
  U psi  4 N^2
  bloch  4 N^2
"""
import torch
from torch import nn

import common  # noqa: F401  pins TF32 off for every arm
from atlas.phase_sphere import cayley, skew_hermitian
from sigreg import LAM, sigreg


def _helmert(N):
    H = torch.zeros(N - 1, N, dtype=torch.float64)
    for k in range(1, N):
        H[k - 1, :k], H[k - 1, k] = 1.0, -float(k)
        H[k - 1] /= (k * (k + 1)) ** 0.5
    return H


def bloch(psi):
    """psi complex [..., N] -> real [..., N^2 - 1] (see module docstring)."""
    N = psi.shape[-1]
    p = psi.real ** 2 + psi.imag ** 2
    i, j = torch.triu_indices(N, N, 1)
    w = psi[..., i].conj() * psi[..., j]
    H = _helmert(N).to(device=p.device, dtype=p.dtype)
    return torch.cat([2 * w.real, 2 * w.imag, 2 ** 0.5 * p @ H.T], -1)


def fidelity_loss(a, b):
    """1 - |<a, b>|^2 over the last dim; invariant to a global phase on either argument."""
    ip = (a.conj() * b).sum(-1)
    return 1 - (ip.real ** 2 + ip.imag ** 2)


def _normalise(re, im):
    psi = torch.complex(re, im)
    return psi / (psi.real ** 2 + psi.imag ** 2).sum(-1, keepdim=True).add(1e-12).sqrt()


class HopfChart(nn.Module):
    def __init__(self, D, A, N=2, hidden=64, ema=0.996, lam=None):
        super().__init__()
        self.D, self.A, self.N, self.hidden, self.ema = D, A, N, hidden, ema
        self.D_g = N * N - 1
        self.lam = LAM if lam is None else lam
        mlp = lambda: nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                    nn.Linear(hidden, 2 * N))
        self.enc = mlp()
        self.tgt_enc = mlp().requires_grad_(False)
        self.tgt_enc.load_state_dict(self.enc.state_dict())
        self.X = nn.Linear(A + 2 * N, 2 * N * N)

    def _encode(self, enc, obs):
        return _normalise(*enc(obs).chunk(2, -1))

    def step(self, a, psi):
        """a [..., A], psi [..., N] -> U(a, psi) psi, unitary and norm-preserving."""
        N = self.N
        X = self.X(torch.cat([a, psi.real, psi.imag], -1))
        X = torch.complex(X[..., :N * N], X[..., N * N:]).reshape(*X.shape[:-1], N, N)
        return (cayley(skew_hermitian(X)) @ psi[..., None])[..., 0]

    def forward(self, obs, act):
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                for pt, p in zip(self.tgt_enc.parameters(), self.enc.parameters()):
                    pt.lerp_(p, 1 - self.ema)
        psi = self._encode(self.enc, obs)
        g = bloch(psi)
        psi_hat = self.step(act[:, :-1], psi[:, :-1])
        ghat = bloch(psi_hat)
        with torch.no_grad():
            tgt = self._encode(self.tgt_enc, obs[:, 1:])
        n = max(psi_hat.shape[0] * psi_hat.shape[1], 1)  # T = 1 has no transitions: fid and task are 0, not NaN
        terms = dict(fid=fidelity_loss(psi_hat, tgt).sum() / n, sig=sigreg(g),
                     task=((ghat - g[:, 1:].detach()) ** 2).sum() / n)
        loss = (1 - self.lam) * (terms["fid"] + terms["task"]) + self.lam * terms["sig"]
        return dict(loss=loss, g=g, ghat=ghat, w=None, psi=psi, terms={k: v.detach() for k, v in terms.items()})

    def rollout(self, obs0, act):
        """H unitary steps from the first frame; returns bloch of each step, [B, H, D_g]."""
        psi, out = self._encode(self.enc, obs0), []
        for h in range(act.shape[1]):
            psi = self.step(act[:, h], psi)
            out.append(bloch(psi))
        return torch.stack(out, 1)

    def active_flops(self):
        D, A, h, N = self.D, self.A, self.hidden, self.N
        return int(D * h + h * h + h * 2 * N + (A + 2 * N) * 2 * N * N + 8 * N ** 3 + 4 * N * N + 4 * N * N)

    def n_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("tgt_"))

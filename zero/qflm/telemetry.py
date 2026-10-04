"""Dynamics telemetry for the QF-LM. Eval readouts only; never a training signal."""
import torch

from . import _w, butterfly


def participation_ratio(amp):
    """1/sum(w^2), w=|amp|^2 normalised: effective number of basis directions occupied (1..N)."""
    w = _w(amp) if amp.is_complex() else amp ** 2
    w = w / w.sum(-1, keepdim=True)
    return 1.0 / (w ** 2).sum(-1)


def traj_erank(Psi):
    """exp(entropy of normalised squared singular values) of a (T, N) trajectory (Roy & Vetterli)."""
    p = torch.linalg.svdvals(Psi) ** 2
    p = p / p.sum()
    return torch.exp(-(p * p.clamp_min(1e-30).log()).sum())


def cap_hit_rate(steps, s_max):
    return 0.0 if s_max == 0 else float((steps == s_max).float().mean())


def period2_flip_rate(loop, tol=0.99):
    """loop (K,B,N): flag b if every fid(k,k+2)>tol and every fid(k,k+1)<tol (fixed point is not a flip)."""
    u = loop / loop.abs().pow(2).sum(-1, keepdim=True).sqrt()
    fid = lambda j: (u[:-j].conj() * u[j:]).sum(-1).abs()  # (K-j, B)
    return float(((fid(2) > tol).all(0) & (fid(1) < tol).all(0)).float().mean())


@torch.no_grad()
def dynamics(model, n_tokens=256, B=4, seed=0):
    tr = []
    model.generate(n_tokens, B=B, generator=torch.Generator().manual_seed(seed), return_steps=True, trace=tr)
    post = torch.stack([t["post"] for t in tr])  # (T,B,N)
    pr = participation_ratio(butterfly(post.reshape(-1, model.N), model.theta, model.phi))
    er = torch.stack([traj_erank(post[:, b]) for b in range(B)]).mean()
    steps = torch.stack([t["steps"] for t in tr])
    full = [t["loop"][:, b] for t in tr for b in range(B) if int(t["steps"][b]) == model.s_max]
    flip = period2_flip_rate(torch.stack(full, 1)) if full and model.s_max >= 2 else 0.0
    return dict(pr_mean=float(pr.mean()), pr_min=float(pr.min()), traj_erank=float(er),
                cap_hit_rate=cap_hit_rate(steps, model.s_max), period2_flip_rate=flip,
                n_tokens=n_tokens, N=model.N)

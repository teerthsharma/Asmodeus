"""SIGReg (LeJEPA, arXiv 2511.08544): sliced Epps-Pulley test of embeddings against N(0, I).

Port of rbalestr-lab/lejepa `univariate/epps_pulley.py` (trapezoid on [0, t_max], 17 points, symmetric
weights times phi(t)=exp(-t^2/2)) and `multivariate/slicing.py` (unit-norm Gaussian slices, fresh each call,
mean over slices). Shared by every arm and the atlas.
Deviation: the official statistic is multiplied by N; here it is not, so lambda is in N-free units and does
not depend on batch size. In these units LeJEPA's lam=0.05 is far too weak: on the S2 bed (200 Adam steps)
every arm collapsed by shrinkage (per-dim std 0.05-0.25) while its total loss still fell.
Grouped mode (per-chart SIGReg): ONE slice draw shared by all groups (relabelling-invariant) and a scatter
over groups, so cost is O(N), independent of the group count.
Deviation: a second-moment scale anchor on the same slices is added to the EP term. EP alone has almost no
restoring force when embeddings expand (x = s*N(0,I): d sigreg/ds = -0.658 at s=0.3 but +0.0076 at s=15 and ~0
at s=30), so arms drifted to std 10-50 unpulled. The anchor is ANCHOR * mean over groups and slices of
log(E[(x.a)^2] + eps)^2, which is scale-symmetric and also catches mean drift. This is not in the LeJEPA reference.
"""
import torch

# One table for every arm (fairness). Arms mix loss = (1 - LAM) * prediction terms + LAM * sum of SIGReg terms.
# LAM: grid {0.5, 0.75, 0.9, 0.95}, S2, D_g=6, 3 seeds, with the scale anchor: 0.5 fails (atlas 2/3 diverge), 0.75 is
# the smallest all-pass, 0.9 taken = one grid step of margin (atlas g std 0.94-0.98 vs 0.93-1.42 at 0.75).
LAM = 0.9
# Switch-style load balance K * sum_k f_k P_k, inside the (1 - LAM) prediction bracket (top-m routers only).
LAM_BAL = 0.3
# Scale-anchor weight: minimum at E[(x.a)^2] = 1 (N(0, I) unchanged); log makes the force scale-symmetric;
# the 1e-6 eps keeps an all-zero input (closed gate) finite.
ANCHOR = 0.1


def sigreg(x: torch.Tensor, grp: torch.Tensor | None = None, n_groups: int | None = None,
           num_slices: int = 256, t_max: float = 3.0, n_points: int = 17) -> torch.Tensor:
    """x [..., d] -> scalar. With grp (int labels [N]) the statistic is the mean over groups with >= 2 rows."""
    x = x.reshape(-1, x.shape[-1])
    if grp is None:
        grp, n_groups = torch.zeros(len(x), dtype=torch.long, device=x.device), 1
    grp = grp.reshape(-1)
    n_groups = n_groups or int(grp.max()) + 1
    cnt = torch.bincount(grp, minlength=n_groups)
    ok = cnt >= 2
    if not ok.any():
        return x.sum() * 0  # no signal, no NaN, keeps the graph
    A = torch.randn(x.shape[-1], num_slices, device=x.device, dtype=x.dtype)
    A = A / A.norm(dim=0, keepdim=True)
    t = torch.linspace(0, t_max, n_points, device=x.device, dtype=x.dtype)
    dt = t_max / (n_points - 1)
    w = torch.full_like(t, 2 * dt)
    w[[0, -1]] = dt
    phi = (-0.5 * t * t).exp()
    proj = x @ A  # [N, S]
    xt = proj.unsqueeze(-1) * t  # [N, S, P]
    z = x.new_zeros(n_groups, *xt.shape[1:])
    c = cnt[ok].to(x.dtype)[:, None, None]
    # ponytail: index_add uses atomics on CUDA (bitwise nondeterministic ~1e-7); sorted segment sums if that matters
    re = z.index_add(0, grp, xt.cos())[ok] / c
    im = z.index_add(0, grp, xt.sin())[ok] / c
    err = (re - phi).square() + im.square()  # [G, S, P]
    ep = (err @ (w * phi)).mean()
    m2 = x.new_zeros(n_groups, proj.shape[1]).index_add(0, grp, proj.square())[ok] / c[:, :, 0]  # raw 2nd moment
    return ep + ANCHOR * (m2 + 1e-6).log().square().mean()

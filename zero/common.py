"""Shared contract between beds (Foreman), OBSERVER-ATLAS (Chase) and owned arms (Cameron).

Bed: dict with
  obs   float32 [N, T, D]   what every arm sees (64x64 render flattened, or random nonlinear lift to D=256)
  act   float32 [N, T, A]   actions (zeros if passive)
  state float64 [N, T, S]   ground-truth manifold state. EVAL ONLY: never passed to any model (L-GIVEN)
  geo   callable(s1, s2) -> geodesic distance on the manifold, broadcasting over leading dims. EVAL ONLY
  meta  dict: name, n (manifold dim), atlas (min chart count), betti_z2 (tuple), seed

Arm: torch.nn.Module with
  forward(obs [B,T,D], act [B,T,A]) -> dict(
      loss   scalar, all training terms summed (anti-collapse term mandatory, §3)
      g      [B,T,Dg]  glued / global latent used by every metric
      ghat   [B,T-1,Dg] one-step prediction of g[:,1:]
      w      [B,T,K] or None   observer weights after top-m renormalisation (atlas, MoP routers)
      z      OPTIONAL [B,T,m,d_c] per-chart latents of the top-m charts, with zidx [B,T,m] (long) chart ids;
             or [B,T,d] for a single-chart arm. Used only for per-chart effective rank (T2, T4a).
      w_soft OPTIONAL [B,T,K] pre-top-m softmax weights (soft-nerve readout, tau <= 1/4 for b3)
      psi    OPTIONAL [B,T,N] complex unit amplitude (Chern / Berry-holonomy readouts, readouts.py)
      )
  rollout(obs0 [B,D], act [B,H,A]) -> [B,H,Dg]   h-step latent prediction from the first frame
  active_flops() -> int   multiply-adds per time step actually executed (lazy: only top-m charts count)
  n_params() -> int
"""
import torch

# Numerics pin for every arm (PREREG fact): TF32 off, so GRU vs GRUCell and lazy vs dense parities hold at fp32
# (Taub: equal within 7e-6 only with cuDNN TF32 off). Imported by arms/__init__.py and atlas/observer_atlas.py.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

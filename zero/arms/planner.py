"""V-JEPA-2-AC-style CEM planner (arXiv 2506.09985 §planning; code github.com/facebookresearch/vjepa2).

Energy = ||rollout(o_k, a_{1:H})[:, -1] - g_goal||_1; Gaussian per step initialised N(0, 1), refit to the
top-k elites each iteration; receding horizon: caller executes the first action and replans.
Paper: 800 samples, 10 iterations, actions clipped to an L1 ball; elite count and momentum unpublished.
Here: box clip [-a_max, a_max] (ponytail: L1-ball projection when a bed needs it). Works with any arm.
modes=True plans over all K MoP-JEPA modes: energy = sum_k pi_k ||g_k[H] - g_goal||_1 over the K branches of
arm.rollout_modes (risk-neutral expectation over the first-step successor set).
"""
import numpy as np
import torch


@torch.no_grad()
def cem_plan(arm, obs0, goal, horizon: int, act_dim: int, n_samples: int = 800, n_elites: int = 32,
             iters: int = 10, a_max: float = 1.0, momentum: float = 0.0, modes: bool = False):
    """obs0 [B, D], goal [B, Dg] -> (first action [B, A], mean plan [B, H, A])."""
    B = obs0.shape[0]
    mu = obs0.new_zeros(B, horizon, act_dim)
    std = torch.ones_like(mu)
    for _ in range(iters):
        a = (mu[:, None] + std[:, None] * torch.randn(B, n_samples, horizon, act_dim, device=obs0.device))
        a = a.clamp(-a_max, a_max)
        o, gl = obs0.repeat_interleave(n_samples, 0), goal.repeat_interleave(n_samples, 0)
        if modes:
            g, pi = arm.rollout_modes(o, a.flatten(0, 1))  # [BS, K, H, Dg], [BS, K]
            cost = (pi * (g[:, :, -1] - gl[:, None]).abs().sum(-1)).sum(-1).view(B, n_samples)
        else:
            cost = (arm.rollout(o, a.flatten(0, 1))[:, -1] - gl).abs().sum(-1).view(B, n_samples)
        elite = a.gather(1, cost.topk(n_elites, -1, largest=False).indices[..., None, None]
                         .expand(-1, -1, horizon, act_dim))
        mu = momentum * mu + (1 - momentum) * elite.mean(1)
        std = momentum * std + (1 - momentum) * elite.std(1)
    return mu[:, 0], mu


def mpc_policy(arm, goal_obs, horizon: int, act_dim: int, **cem):
    """Receding-horizon policy for bed evaluators: plan_fn(obs [n, D] numpy) -> first actions [n, A] numpy.
    Goal latent z_g = the arm's own g of the goal observation (V-JEPA 2-AC: z_g = E(x_g)), via observe()."""
    x = torch.as_tensor(np.asarray(goal_obs), dtype=torch.float32)[None, None]
    with torch.no_grad():
        goal = arm.observe(x, x.new_zeros(1, 1, act_dim))[0][:, 0]  # [1, Dg]

    def plan(obs):
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        return cem_plan(arm, o, goal.expand(len(o), -1), horizon, act_dim, **cem)[0].numpy()
    return plan

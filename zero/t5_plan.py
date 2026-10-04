"""T5 shared planner: CEM over the RUNNING latent energy, identical for every arm.

    E(a) = mean_{h=1..H} || rollout(o_0, a)[:, h-1] - g_goal ||_1

Why not arms.planner.cem_plan's terminal L1: calibration with a PERFECT model solved swing-up 3/10 with the
terminal energy but 10/10 with the running height cost (PREREG_DRAFT_t5_t6.md). Sampling, elites and refit are
cem_plan's, so horizon=1 reproduces it exactly. LatentMPC fixes equal planning compute via budget_samples.
"""
import numpy as np
import torch

from atlas.time_traveller import budget_samples


@torch.no_grad()
def cem_running(arm, obs0, goal, horizon: int, act_dim: int, n_samples: int = 800, n_elites: int = 32,
                iters: int = 10, a_max: float = 1.0, momentum: float = 0.0, init_mean=None):
    """obs0 [B, D], goal [B, Dg] -> (first action [B, A], mean plan [B, H, A])."""
    B = obs0.shape[0]
    mu = obs0.new_zeros(B, horizon, act_dim) if init_mean is None else init_mean.clone()
    std = torch.ones_like(mu)
    k = min(n_elites, n_samples)
    for _ in range(iters):
        a = (mu[:, None] + std[:, None] * torch.randn(B, n_samples, horizon, act_dim, device=obs0.device))
        a = a.clamp(-a_max, a_max)
        o, gl = obs0.repeat_interleave(n_samples, 0), goal.repeat_interleave(n_samples, 0)
        cost = (arm.rollout(o, a.flatten(0, 1)) - gl[:, None]).abs().sum(-1).mean(-1).view(B, n_samples)
        elite = a.gather(1, cost.topk(k, -1, largest=False).indices[..., None, None]
                         .expand(-1, -1, horizon, act_dim))
        mu = momentum * mu + (1 - momentum) * elite.mean(1)
        std = momentum * std + (1 - momentum) * elite.std(1)
    return mu[:, 0], mu


class LatentMPC:
    """Receding-horizon policy for beds. goal_obs: fixed obs [D] (swingup) or None (goal passed per call)."""

    def __init__(self, arm, goal_obs, horizon: int, act_dim: int, budget, iters: int = 10, n_elites: int = 32,
                 a_max: float = 1.0, warm: bool = False, device: str = "cpu"):
        self.arm, self.goal_obs, self.H, self.A = arm, goal_obs, horizon, act_dim
        self.iters, self.n_elites, self.a_max, self.warm, self.device = iters, n_elites, a_max, warm, device
        self.n_samples = budget_samples(arm, budget, horizon, iters)
        self.mu = None

    def reset(self):
        self.mu = None

    @torch.no_grad()
    def __call__(self, obs, goal=None):
        single = np.ndim(obs) == 1
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        o = o[None] if single else o
        gl = self.goal_obs if goal is None else goal
        x = torch.as_tensor(np.asarray(gl), dtype=torch.float32, device=self.device)[None, None]
        was = self.arm.training
        self.arm.eval()
        try:
            g = self.arm(x, x.new_zeros(1, 1, self.A))["g"][:, 0]
            init = None
            if self.warm and self.mu is not None and self.mu.shape[0] == len(o):
                init = torch.cat([self.mu[:, 1:], torch.zeros_like(self.mu[:, :1])], 1)
            a, mu = cem_running(self.arm, o, g.expand(len(o), -1), self.H, self.A, self.n_samples,
                                self.n_elites, self.iters, self.a_max, init_mean=init)
        finally:
            self.arm.train(was)
        self.mu = mu if self.warm else None
        return a[0].numpy() if single else a.numpy()

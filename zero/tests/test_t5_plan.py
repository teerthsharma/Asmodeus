import numpy as np
import torch
from torch import nn

from arms.planner import cem_plan
from atlas.time_traveller import budget_samples
from beds import swingup as sw
from t5_plan import LatentMPC, cem_running

torch.set_num_threads(4)


class Toy(nn.Module):
    """Point mass: g = first 2 obs dims, o_h = o_0 + 0.1 * cumsum(a)."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, obs, act):
        return dict(g=obs[..., :2], ghat=obs[..., :2], loss=torch.zeros(()), w=None)

    def rollout(self, obs0, act):
        return obs0[:, None, :2] + 0.1 * act.cumsum(1)

    def active_flops(self):
        return 10


def _goal(v=1.0, n=1):
    return torch.full((n, 2), v)


def test_horizon1_matches_cem_plan():
    arm, o = Toy(), torch.zeros(3, 2)
    torch.manual_seed(0)
    a, _ = cem_plan(arm, o, _goal(1.0, 3), 1, 2, n_samples=64, n_elites=8, iters=3)
    torch.manual_seed(0)
    b, _ = cem_running(arm, o, _goal(1.0, 3), 1, 2, n_samples=64, n_elites=8, iters=3)
    assert torch.allclose(a, b, atol=1e-6)


def test_point_mass_first_action_and_closed_loop():
    arm = Toy()
    torch.manual_seed(0)
    a, mu = cem_running(arm, torch.zeros(1, 2), _goal(), 10, 2, n_samples=256, iters=5)
    assert mu.shape == (1, 10, 2) and (a[0] > 0.5).all()
    H, iters = 5, 5
    mpc = LatentMPC(arm, None, H, 2, budget=256 * iters * H * 10, iters=iters)
    assert mpc.n_samples == 256
    x, g = np.zeros(2, np.float32), np.ones(2, np.float32)
    for _ in range(30):
        x = x + 0.1 * mpc(x, g)
    assert np.linalg.norm(x - g) < 0.15


def test_running_differs_from_terminal():
    arm, r, t = Toy(), [], []
    for s in range(5):
        torch.manual_seed(s)
        r.append(cem_running(arm, torch.zeros(1, 2), _goal(), 10, 2, n_samples=256, iters=5)[0].abs().sum())
        torch.manual_seed(s)
        t.append(cem_plan(arm, torch.zeros(1, 2), _goal(), 10, 2, n_samples=256, iters=5)[0].abs().sum())
    assert np.mean([float(x) for x in r]) > np.mean([float(x) for x in t])


def test_budget_scaling():
    arm, H, it = Toy(), 10, 10
    n1 = LatentMPC(arm, None, H, 2, budget=100_000).n_samples
    n2 = LatentMPC(arm, None, H, 2, budget=200_000).n_samples
    assert n1 == budget_samples(arm, 100_000, H, it) == 100
    assert abs(n2 - 2 * n1) <= 1


def test_swingup_adapter_plumbing():
    mpc = LatentMPC(Toy(), sw.goal_obs(), 3, 2, budget=3 * 10 * 5 * 40, iters=5)
    out = sw.evaluate(mpc, n_episodes=2, seed=0, horizon=3)
    assert out["tip"].shape == (2, 3)


def test_training_flag_restored():
    for flag in (True, False):
        arm = Toy().train(flag)
        LatentMPC(arm, None, 2, 2, budget=2000, iters=2)(np.zeros(2, np.float32), np.ones(2, np.float32))
        assert arm.training is flag


def test_warm_memory_and_reset():
    mpc = LatentMPC(Toy(), None, 4, 2, budget=4000, iters=2, warm=True)
    assert mpc.mu is None
    mpc(np.zeros(2, np.float32), np.ones(2, np.float32))
    assert mpc.mu is not None and mpc.mu.shape == (1, 4, 2)
    mpc.reset()
    assert mpc.mu is None

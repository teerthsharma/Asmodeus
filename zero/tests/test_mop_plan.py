"""MoP-JEPA planning: planAll graph protocol (T4b, beds/ogbench_bed.py) and CEM over all K modes (T5, beds/swingup.py)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from arms.mop_jepa import MoPGraph, MoPJEPA
from arms.planner import cem_plan, mpc_policy
from beds import ogbench_bed as ob  # pure numpy at import; ogbench is needed only by maze()


# ---- T4b: graph-protocol adapter (interface of arms/controls_t4b.py) ----------------------------
def branching_toy(n_rep=100, seed=0):
    """4 contexts on a ring, each with 2 equally likely successors among 8 outer points (real branching)."""
    rng = np.random.default_rng(seed)
    ang = np.arange(4) * np.pi / 2
    ctx_pts = np.stack([np.cos(ang), np.sin(ang)], -1) * 2
    out = np.stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)], -1) * 6
    succ = {c: (2 * c, 2 * c + 1) for c in range(4)}
    c = rng.integers(0, 4, 4 * n_rep)
    s = np.array([succ[i][rng.integers(2)] for i in c])
    pts = [tuple(p) for p in np.round(np.concatenate([ctx_pts, out]), 6)]
    true = {(pts[a], pts[4 + b]) for a in range(4) for b in succ[a]}
    return ctx_pts[c], out[s], ctx_pts, pts, true


def test_mop_graph_interface_and_recovers_both_successors():
    ctx, nxt, ctx_pts, pts, true = branching_toy()
    torch.manual_seed(0)
    m = MoPGraph(K=4, d=16, hidden=64, head_hidden=64, steps=1000, batch=128, lr=3e-3, seed=0).fit(ctx, nxt)
    xy = np.array(pts)
    h, pi = m.predict(xy)
    assert h.shape == (12, 4, 16) and pi.shape == (12, 4) and np.allclose(pi.sum(1), 1, atol=1e-5)
    assert m.embed(xy).shape == (12, 16)
    E = {e for e in ob.propose_edges(pts, m.embed(xy), h, pi) if e[0] in pts[:4]}  # outer points are never contexts
    assert len(E & true) >= 7 and len(E & true) / max(len(E), 1) >= 0.75, (len(E & true), len(E))


def test_mop_graph_never_sees_actions_or_state():
    m = MoPGraph(K=2, d=4, steps=1)
    m.fit(np.zeros((8, 2)), np.ones((8, 2)))
    assert m.model.heads.W1.shape[1] == 4  # router/heads read z only (A = 0: action-marginalised protocol)


# ---- T5: CEM over all K modes ------------------------------------------------------------------
def mop(K=3):
    torch.manual_seed(0)
    return MoPJEPA(16, 2, d=4, K=K, hidden=16, head_hidden=8, router_hidden=8).eval()


def test_rollout_modes_branches_on_every_head_then_follows_router():
    m = mop()
    obs0, act = torch.randn(5, 16), torch.randn(5, 4, 2)
    with torch.no_grad():
        g, pi = m.rollout_modes(obs0, act)
        z = F.normalize(m.enc(obs0), dim=-1)
        c = torch.cat([z, act[:, 0]], -1)
        u0 = F.normalize(m.heads(c), dim=-1)
        pi0 = m.router(c).softmax(-1)
    assert g.shape == (5, 3, 4, 4) and pi.shape == (5, 3)
    assert torch.allclose(g[:, :, 0], u0, atol=1e-6) and torch.allclose(pi, pi0, atol=1e-6)
    k = pi0.argmax(-1)
    with torch.no_grad():
        greedy = m.rollout(obs0, act)
    assert torch.allclose(g[torch.arange(5), k], greedy, atol=1e-6)  # router-argmax branch = greedy rollout


class TwoMode(torch.nn.Module):
    """branch 0: g + sum(a) with prob 0.9; branch 1: g - sum(a) with prob 0.1."""

    def rollout_modes(self, obs0, act):
        c = act.cumsum(1)
        g = torch.stack([obs0[:, None] + c, obs0[:, None] - c], 1)
        return g, torch.tensor([[0.9, 0.1]]).expand(len(obs0), -1)


def test_cem_over_modes_minimises_expected_energy():
    torch.manual_seed(0)
    goal = torch.tensor([[1.0, 0.0]])
    _, plan = cem_plan(TwoMode(), torch.zeros(1, 2), goal, horizon=2, act_dim=2, n_samples=512, n_elites=32,
                       iters=10, modes=True)
    assert (plan.sum(1) - goal).abs().max() < 0.1  # expected-L1 optimum follows the 0.9 branch


def test_swingup_mpc_policy_smoke():
    sw = pytest.importorskip("beds.swingup")
    data = sw.make_dataset(32, 16, seed=0)
    obs, act = torch.as_tensor(data["obs"]), torch.as_tensor(data["act"])
    torch.manual_seed(0)
    m = MoPJEPA(obs.shape[-1], 2, d=8, K=4, hidden=32, head_hidden=16, router_hidden=16)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(30):
        opt.zero_grad()
        m(obs, act)["loss"].backward()
        opt.step()
    plan = mpc_policy(m.eval(), sw.goal_obs(), horizon=4, act_dim=2, modes=True, n_samples=32, n_elites=4, iters=2)
    r = sw.evaluate(plan, n_episodes=2, seed=0, horizon=3)
    assert 0.0 <= r["success_rate"] <= 1.0 and r["tip"].shape == (2, 3)

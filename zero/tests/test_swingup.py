"""T5 swing-up bed: contract, success predicate, no state leak to model planners, determinism."""
import numpy as np
import pytest

from beds import lift
from beds import swingup as sw


def test_dataset_contract():
    b = sw.make_dataset(n_traj=8, T=32, seed=0)
    N, T, D = b["obs"].shape
    assert (N, T, D) == (8, 32, 256) and b["obs"].dtype == np.float32
    assert b["act"].shape == (8, 32, 2) and np.abs(b["act"]).max() <= 1
    assert b["state"].shape == (8, 32, 6) and b["state"].dtype == np.float64
    assert np.isfinite(b["obs"]).all()
    m = b["meta"]
    assert (m["name"], m["n"], m["atlas"], m["betti_z2"]) == ("swingup", 4, 3, (1, 2, 1))
    s = b["state"]
    assert np.allclose(b["geo"](s, s), 0, atol=1e-6)
    # obs = Foreman's lift keyed by bed name (+ noise)
    r = b["obs"] - lift("swingup", sw.lift_state(s))
    assert r.std() < 0.05 * b["obs"].std()


def test_dataset_covers_upright():
    b = sw.make_dataset(n_traj=64, T=64, seed=0)
    assert (sw.tip_height(b["state"]) > sw.H_SUCC).mean() > 0.005  # coverage of the goal region


def test_held_predicate():
    h = np.array([[0, 1, 1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0, 0, 0]], float)
    assert sw.longest_hold(h > 0.5).tolist() == [4, 3]


def test_zero_policy_fails_and_is_deterministic():
    zero = lambda o: np.zeros((len(o), 2))
    r1 = sw.evaluate(zero, n_episodes=4, seed=3, horizon=40)
    r2 = sw.evaluate(zero, n_episodes=4, seed=3, horizon=40)
    assert r1["success_rate"] == 0.0
    assert np.array_equal(r1["tip"], r2["tip"]) and r1["tip"].shape == (4, 40)


def test_model_planner_never_sees_state():
    seen = []

    def plan(*args, **kw):
        seen.append((len(args), tuple(kw)))
        return np.zeros((len(args[0]), 2))
    sw.evaluate(plan, n_episodes=2, seed=0, horizon=3)
    assert set(seen) == {(1, ())}


def test_oracle_gets_state_and_swings_up_somewhat():
    pf = sw.oracle_mpc(horizon=10, n_samples=64, n_elites=8, iters=2, seed=0)
    r = sw.evaluate(pf, n_episodes=2, seed=0, horizon=30)
    assert r["tip"].max() > sw.evaluate(lambda o: np.zeros((len(o), 2)), 2, 0, horizon=30)["tip"].max()


def test_simulate_chunking_is_exact():
    rng = np.random.default_rng(0)
    q, v, a = rng.normal(size=(20, 2)), rng.normal(size=(20, 2)), rng.uniform(-1, 1, (20, 6, 2))
    p1, v1 = sw.simulate(q, v, a)
    p2, v2 = sw.simulate(q, v, a, chunk=7)
    assert np.array_equal(p1, p2) and np.array_equal(v1, v2)

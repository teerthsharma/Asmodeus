"""T4b OGBench bed (beds/ogbench_bed.py). Runs only where ogbench is installed (venv ~/.venvs/ogbench).

Hidden-variable checks: cell-level successor structure matches the paper's "2.1-2.2 successor modes per
cell"; every data transition is a true maze edge; the 20% edge holdout keeps source support; realroute
alone is freeloadable (propose-everything scores 1.0), so precision must be reported beside it; the
closed-loop evaluator can register success (oracle) and is seed-deterministic; action clipping blocks
wall tunnelling.
"""
import numpy as np
import pytest

pytest.importorskip("ogbench")
from beds import ogbench_bed as ob  # noqa: E402

MED, LARGE, TELE = "pointmaze-medium-stitch-v0", "pointmaze-large-stitch-v0", "pointmaze-teleport-navigate-v0"


@pytest.fixture(scope="module")
def med():
    return ob.make_bed(MED)


def test_contract(med):
    N, T, D = med["obs"].shape
    assert (N, T, D) == (5000, 201, 2) and med["obs"].dtype == np.float32   # 200 transitions + final state
    assert not med["act"][:, -1].any()          # last raw action has no successor (OGBench drops it)
    assert med["act"].shape == (N, T, 2) and med["act"].dtype == np.float32
    assert med["state"].dtype == np.float64 and med["state"].shape == (N, T, 2)
    m = med["meta"]
    assert m["name"] == MED and m["n"] == 2 and m["atlas"] == 1 and m["obs_is_state"]
    assert m["betti_z2"] == (1, 1)              # free space of medium maze: one interior wall island
    s = med["state"][:3]
    assert np.allclose(med["geo"](s, s), 0) and (med["geo"](s[:, :1], s[:, -1:]) >= 0).all()


def test_val_split_disjoint_episodes():
    v = ob.make_bed(MED, split="val")
    assert v["obs"].shape == (500, 201, 2)


@pytest.mark.parametrize("name,lo,hi", [(MED, 2.0, 2.3), (LARGE, 2.0, 2.3), (TELE, 2.1, 2.3)])
def test_cell_successors_match_paper(name, lo, hi):
    """MoP-JEPA v1: 'measured average 2.1-2.2 successor modes per cell'. Our cell graph must reproduce it."""
    bed = ob.make_bed(name)
    E = ob.data_edges(bed)
    succ = {}
    for a, b in E:
        succ.setdefault(a, set()).add(b)
    assert lo <= np.mean([len(v) for v in succ.values()]) <= hi
    assert E <= ob.true_edges(name)             # no transition through a wall


def test_holdout_keeps_source_support(med):
    E = ob.data_edges(med)
    tr, ho = ob.holdout_edges(E, frac=0.2, seed=0)
    assert tr | ho == E and not tr & ho and len(ho) == round(0.2 * len(E))
    assert {a for a, _ in E} == {a for a, _ in tr}
    assert ob.holdout_edges(E, 0.2, 0) == (tr, ho) and ob.holdout_edges(E, 0.2, 1)[1] != ho


def test_graph_metrics_and_freeloading():
    T = ob.true_edges(MED)
    cells = sorted({a for a, _ in T})
    Q = ob.sample_queries(MED, n=300, seed=0)
    assert len(Q) == 300 and all(s != g for s, g in Q)
    r = ob.graph_metrics(T, MED, Q)
    assert r == dict(planAll=1.0, realroute=1.0, precision=1.0)
    assert ob.graph_metrics(set(), MED, Q)["planAll"] == 0.0
    everything = {(a, b) for a in cells for b in cells if a != b}
    r = ob.graph_metrics(everything, MED, Q)
    assert r["planAll"] == 1.0 and r["realroute"] == 1.0 and r["precision"] < 0.15   # realroute is freeloadable
    assert len(ob.official_queries(MED)) == 5


def test_propose_edges_threshold():
    lat = np.eye(3)                              # 3 cells, orthonormal latents
    heads = np.stack([np.eye(3)[[1, 2]], np.eye(3)[[0, 0]], np.eye(3)[[0, 1]]])   # [C=3,K=2,d]
    cells = [(0, 0), (0, 1), (0, 2)]
    E = ob.propose_edges(cells, lat, heads)
    assert E == {((0, 0), (0, 1)), ((0, 0), (0, 2)), ((0, 1), (0, 0)), ((0, 2), (0, 0)), ((0, 2), (0, 1))}
    pi = np.array([[0.9, 0.1], [0.5, 0.5], [0.5, 0.5]])
    assert ((0, 0), (0, 2)) not in ob.propose_edges(cells, lat, heads, pi=pi)   # router gate pi > 0.5/K


def test_evaluate_random_near_zero_and_deterministic():
    rng = np.random.default_rng(0)
    pol = lambda o, g: rng.uniform(-1, 1, 2)                            # noqa: E731
    r1 = ob.evaluate(pol, MED, episodes_per_task=2, seed=3)
    rng = np.random.default_rng(0)
    r2 = ob.evaluate(pol, MED, episodes_per_task=2, seed=3)
    assert r1 == r2 and r1["success"] <= 0.2 and r1["n"] == 10


def test_evaluate_oracle_succeeds():
    """A broken evaluator reads 0 for everyone; the BFS oracle on the true cell graph must succeed."""
    r = ob.evaluate(ob.oracle_policy(MED), MED, episodes_per_task=2, seed=0)
    assert r["success"] >= 0.9
    assert ob.evaluate(ob.oracle_policy(MED), MED, episodes_per_task=2, seed=0)["lengths"] == r["lengths"]
    assert ob.evaluate(ob.oracle_policy(MED), MED, episodes_per_task=2, seed=1)["lengths"] != r["lengths"]


def test_actions_clipped_no_tunnelling():
    env = ob.make_env(MED)
    np.random.seed(0)
    o, info = env.reset(options=dict(task_id=1))
    walls = np.array(env.unwrapped.maze_map)
    for _ in range(200):
        o, *_ = env.step(ob.clip(np.array([1e3, 1e3])))
        i, j = env.unwrapped.xy_to_ij(o)
        assert walls[i, j] == 0


def _fake_edges(name, frac, seed):
    T = sorted(ob.true_edges(name))
    cells = sorted({a for a, _ in T})
    rng = np.random.default_rng(seed)
    fake = set()
    while len(fake) < round(frac * len(T)):
        a, b = (cells[i] for i in rng.choice(len(cells), 2, replace=False))
        if (a, b) not in ob.true_edges(name):
            fake.add((a, b))
    return set(T) | fake


def test_executor_true_graph_and_replanning_repair():
    """v1: executor 15/15 on true paths; 10% false edges: replanning lifts 0.40 -> 1.00."""
    r = ob.evaluate(ob.executor(MED, ob.true_edges(MED)), MED, episodes_per_task=3, seed=0)
    assert r["success"] == 1.0 and r["n"] == 15
    bad = _fake_edges(MED, 0.10, 0)
    rep = ob.evaluate(ob.executor(MED, bad), MED, episodes_per_task=3, seed=0)["success"]
    norep = ob.evaluate(ob.executor(MED, bad, replan=False), MED, episodes_per_task=3, seed=0)["success"]
    assert rep >= 0.9 and norep < rep


def test_executor_resets_between_episodes():
    bad = _fake_edges(MED, 0.10, 0)
    ex = ob.executor(MED, bad)
    first = ob.evaluate(ex, MED, episodes_per_task=1, seed=0)
    assert ex.removed                            # this graph forces deletions
    again = ob.evaluate(ex, MED, episodes_per_task=1, seed=0)        # reused executor: no state may leak
    assert again["lengths"] == first["lengths"] == ob.evaluate(ob.executor(MED, bad), MED, 1, seed=0)["lengths"]

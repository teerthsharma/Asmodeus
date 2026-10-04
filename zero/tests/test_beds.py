"""Bed contract (common.py) plus the hidden-flaw checks: lift shared across seeds, coverage,
seam crossings, first-order (Markov) dynamics, geodesics consistent with the quotient metric."""
import numpy as np
import pytest

from beds import make_bed, lift, torn_chart, sample_uniform, klein_embed, BEDS

NAMES = ["s2", "t2", "klein", "so3", "r2"]
META = {"s2": (2, 2, (1, 0, 1)), "t2": (2, 3, (1, 2, 1)), "klein": (2, 3, (1, 2, 1)),
        "so3": (3, 4, (1, 1, 1, 1)), "r2": (2, 1, (1, 0, 0))}


@pytest.fixture(scope="module", params=NAMES)
def bed(request):
    return make_bed(request.param, n_traj=64, T=64, obs="lift", seed=0)


def test_contract(bed):
    m = bed["meta"]
    assert (m["n"], m["atlas"], m["betti_z2"]) == META[m["name"]]
    N, T, D = bed["obs"].shape
    assert (N, T, D) == (64, 64, 256) and bed["obs"].dtype == np.float32
    assert bed["act"].dtype == np.float32 and bed["act"].shape[:2] == (N, T)
    assert bed["state"].dtype == np.float64 and bed["state"].shape[:2] == (N, T)
    assert np.isfinite(bed["obs"]).all()
    s = bed["state"]
    assert np.allclose(bed["geo"](s, s), 0, atol=1e-6)
    a, b = s[:, :5], s[:, 5:10]
    assert np.allclose(bed["geo"](a, b), bed["geo"](b, a))


def test_lift_is_shared_across_seeds():
    for name in NAMES:
        for seed in (0, 1):
            b = make_bed(name, n_traj=8, T=16, obs="lift", seed=seed)
            r = b["obs"] - lift(name, b["state"])
            assert r.std() < 0.05 * b["obs"].std(), name


def test_lift_is_well_conditioned():
    for name in NAMES:
        x = sample_uniform(name, 4000, np.random.default_rng(0)).reshape(2000, 2, -1)
        de = np.linalg.norm(x[:, 0] - x[:, 1], axis=-1)
        do = np.linalg.norm(lift(name, x[:, 0]) - lift(name, x[:, 1]), axis=-1)
        ratio = do / de
        assert ratio.min() > 0.2 * np.median(ratio), name           # no near-collisions in obs


def test_coverage_and_seam_crossings(bed):
    name = bed["meta"]["name"]
    s = bed["state"].reshape(-1, bed["state"].shape[-1])
    ref = sample_uniform(name, 400, np.random.default_rng(1))
    sub = s[np.random.default_rng(2).choice(len(s), 3000, replace=False)]
    d = np.stack([bed["geo"](r[None], sub).min() for r in ref])
    diam = {"s2": np.pi, "t2": np.pi * 2 ** .5, "klein": np.pi * 2 ** .5, "so3": np.pi, "r2": 2 * 2 ** .5}[name]
    assert d.max() < 0.25 * diam, (name, d.max())
    if name != "r2":
        c = torn_chart(name, bed["state"])
        jump = np.linalg.norm(np.diff(c, axis=1), axis=-1)
        step = bed["geo"](bed["state"][:, 1:], bed["state"][:, :-1])
        crossings = ((jump > 1.0) & (step < 0.3)).sum()
        assert crossings >= 10, (name, crossings)                 # dynamics must cross every seam


def test_first_order_dynamics():
    """Pendulum/rigid-body physics is 2nd order: a 1-frame JEPA cannot predict it. Beds are overdamped,
    so displacement follows the CURRENT action, not the previous one."""
    for name in ["s2", "t2", "so3", "r2"]:
        b = make_bed(name, n_traj=64, T=64, obs="lift", seed=0)
        dx = BEDS[name]["tangent_step"](b["state"][:, :-1], b["state"][:, 1:])   # [N,T-1,A]
        a = b["act"]
        cur = np.corrcoef(dx[:, 1:].ravel(), a[:, 1:-1].ravel())[0, 1]
        prev = np.corrcoef(dx[:, 1:].ravel(), a[:, :-2].ravel())[0, 1]
        assert cur > 0.5 and abs(prev) < 0.12, (name, cur, prev)


def test_klein_has_no_global_action_frame():
    b = make_bed("klein", n_traj=4, T=8, obs="lift", seed=0)
    assert not b["act"].any()


def test_klein_geo_is_quotient_metric():
    u = np.array([2 * np.pi - 0.01, 1.0]); v = np.array([0.5, 0.3])
    p = klein_embed(u, v)
    q = klein_embed(np.array([0.02, 1.0]), np.array([-0.5 + 0.0, 0.3 + 0.04]))
    g = BEDS["klein"]["geo"](p, q)
    assert abs(g[0] - 0.03) < 1e-9 and abs(g[1] - 0.04) < 1e-9


def test_render_t2():
    b = make_bed("t2", n_traj=2, T=4, obs="render", seed=0)
    assert b["obs"].shape == (2, 4, 64 * 64) and b["obs"].std() > 0

"""bench_event contracts: exact trigger fraction on known sequences; event streaming step vs always-compute."""
import math

import pytest
import torch

from atlas.event_driven import EventDriven
from atlas.observer_atlas import ObserverAtlas

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_trigger_rate_known_sequences():
    from bench_event import trigger_rate
    ones = torch.ones(4)
    rep = torch.stack([torch.tensor(v) * ones for v in (0., 0., 1., 1., 1., 2.)])   # step of 1 -> delta 1
    drift = torch.stack([torch.tensor(v) * ones for v in (0., .3, .6, .9)])         # delta vs LAST trigger
    assert trigger_rate(rep[None].repeat(2, 1, 1), 0.5) == pytest.approx(3 / 6)                    # t = 0, 2, 5
    assert trigger_rate(drift[None], 0.5) == pytest.approx(2 / 4)                                  # t = 0, 2 (not 1-q of prev frame)
    assert trigger_rate(rep[None], 0.0) == 1.0 and trigger_rate(rep[None], math.inf) == pytest.approx(1 / 6)


def _setup(B=4, T=5):
    from bench_t7 import CFG, CellBackbone
    torch.manual_seed(0)
    at = ObserverAtlas(K=16, m=2, **CFG).cuda().eval()
    return at, CellBackbone(at.backbone), torch.randn(B, T, CFG["D"], device="cuda"), torch.randn(B, T, CFG["A"], device="cuda")


@cuda
@torch.no_grad()
def test_event_step_eps0_equals_always_compute_and_inf_holds():
    from bench_event import EventStream
    from bench_t7 import fused_observe
    at, cell, obs, act = _setup()
    s0, sinf = EventStream(at, cell, obs, act, 4, 0.0), EventStream(at, cell, obs, act, 4, math.inf)
    first = fused_observe(at, obs[:, :1], act[:, :1], cell)["g"]
    for t in range(5):
        ref = fused_observe(at, obs[:, t:t + 1], act[:, t:t + 1], cell)["g"]
        s0.step(), sinf.step()
        assert torch.allclose(s0.g, ref, atol=1e-5)
        assert torch.allclose(sinf.g, first, atol=1e-5)
    assert s0.n_trig == 4 * 5 and sinf.n_trig == 4


@cuda
@torch.no_grad()
def test_stream_rate_equals_event_driven_triggers():
    from bench_event import EventStream, prev_deltas
    at, cell, obs, act = _setup(B=4, T=8)
    eps = float(prev_deltas(obs).median())
    s = EventStream(at, cell, obs, act, 4, eps)
    for _ in range(8):
        s.step()
    trig, _ = EventDriven(torch.nn.Identity()).triggers(obs, eps)
    assert 0 < s.n_trig < 32 and s.n_trig == int(trig.sum())


def _walk(at_kw, B=4, T=8):
    from bench_t7 import CFG, CellBackbone
    torch.manual_seed(0)
    at = ObserverAtlas(**at_kw, m=2, **CFG).cuda().eval()
    obs = (0.3 * torch.randn(B, T, CFG["D"], device="cuda")).cumsum(1)
    return at, CellBackbone(at.backbone), obs, torch.randn(B, T, CFG["A"], device="cuda")


def _graph_run(at, cell, obs, act, eps, mode, graph, n):
    from bench_event import EventStreamGraph
    from bench_t7 import graphed
    s = EventStreamGraph(at, cell, obs, act, obs.shape[0], eps, mode)
    step = graphed(s.step) if graph else s.step
    s.reset()
    gs = []
    for _ in range(n):
        step()
        gs.append((s.g.clone(), s.gh.clone()))
    return gs, int(s.n_trig)


@cuda
@torch.no_grad()
@pytest.mark.parametrize("at_kw", [dict(K=16), dict(K=256, product_keys=True)])
@pytest.mark.parametrize("q", [0.5, 0.9])
@pytest.mark.parametrize("mode", ["masked", "inkernel"])
@pytest.mark.parametrize("graph", [False, True])
def test_graph_event_matches_eventstream(at_kw, q, mode, graph):
    from bench_event import EventStream, prev_deltas
    at, cell, obs, act = _walk(at_kw)
    T = obs.shape[1]
    eps = float(torch.quantile(prev_deltas(obs).flatten(), q))
    ref = EventStream(at, cell, obs, act, 4, eps)
    want = []
    for _ in range(2 * T):
        ref.step()
        want.append((ref.g.clone(), ref.gh.clone()))
    got, n = _graph_run(at, cell, obs, act, eps, mode, graph, 2 * T)
    assert n == ref.n_trig and 4 < n < 4 * 2 * T
    for (g, gh), (wg, wgh) in zip(got, want):
        assert torch.allclose(g, wg, atol=1e-5) and torch.allclose(gh, wgh, atol=1e-5)


@cuda
@torch.no_grad()
@pytest.mark.parametrize("mode", ["masked", "inkernel"])
def test_graph_event_eps0_always_and_inf_holds(mode):
    from bench_t7 import fused_observe
    at, cell, obs, act = _walk(dict(K=16))
    T = obs.shape[1]
    first = fused_observe(at, obs[:, :1], act[:, :1], cell)
    got0, n0 = _graph_run(at, cell, obs, act, 0.0, mode, True, T)
    goti, ni = _graph_run(at, cell, obs, act, math.inf, mode, True, T)
    for t in range(T):
        w = fused_observe(at, obs[:, t:t + 1], act[:, t:t + 1], cell)
        assert torch.allclose(got0[t][0], w["g"], atol=1e-5) and torch.allclose(got0[t][1], w["gh"], atol=1e-5)
        assert torch.allclose(goti[t][0], first["g"], atol=1e-5) and torch.allclose(goti[t][1], first["gh"], atol=1e-5)
    assert n0 == 4 * T and ni == 4

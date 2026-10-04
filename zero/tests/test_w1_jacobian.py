"""Contracts for the owned w1 (orientability) baseline read from Jacobian signs of learned chart transitions (arXiv 2602.22873)."""
import pathlib
import random
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from arms.chart_ae import ChartAEJEPA  # noqa: E402
from arms.w1_jacobian import edge_signs, jac_sign, loop_sign, w1_class  # noqa: E402


@pytest.mark.parametrize("d", [2, 3])
def test_jac_sign_linear(d):
    g = torch.Generator().manual_seed(d)
    for want in (1, -1):
        while True:
            A = torch.randn(d, d, generator=g, dtype=torch.float64)
            if A.det().sign() == want:
                break
        u = torch.randn(20, d, generator=g, dtype=torch.float64)
        assert (jac_sign(lambda x: x @ A.T, u) == want).all()


def test_w1_class_cases():
    assert w1_class({(0, 1): 1, (1, 2): 1, (0, 2): 1}) == 0
    assert w1_class({(0, 1): 1, (1, 2): 1, (0, 2): -1}) == 1
    assert w1_class({(0, 1): -1, (1, 2): 1, (2, 3): -1, (0, 3): 1}) == 0
    # disconnected: a consistent triangle plus a twisted triangle -> 1; consistent + a tree -> 0
    tri, bad = {(0, 1): 1, (1, 2): 1, (0, 2): 1}, {(3, 4): 1, (4, 5): 1, (3, 5): -1}
    assert w1_class({**tri, **bad}) == 1
    assert w1_class({**tri, (3, 4): -1, (4, 5): -1}) == 0


def test_w1_class_gauge_invariant():
    for seed in range(20):
        rng = random.Random(seed)
        n = rng.randint(3, 8)
        edges = {(j, k): rng.choice([1, -1]) for j in range(n) for k in range(j + 1, n) if rng.random() < 0.6}
        v = rng.randrange(n)
        flipped = {(j, k): -s if v in (j, k) else s for (j, k), s in edges.items()}
        assert w1_class(edges) == w1_class(flipped)


def test_loop_sign():
    e = {(0, 1): 1, (1, 2): 1, (0, 2): -1}
    assert loop_sign(e, [0, 1, 2]) == -1
    assert loop_sign(e, [0, 1, 0]) == 1
    assert loop_sign(e, [0, 0, 1, 1, 2]) == loop_sign(e, [0, 1, 2])
    assert loop_sign({(0, 1): 1}, [0, 1, 2]) == 0


def _model():
    torch.manual_seed(0)
    m = ChartAEJEPA(16, 1, K=3, d_c=2, D_g=6, hidden=16, chart_hidden=8).double().eval()
    return m, torch.randn(4, 6, 16, dtype=torch.float64)


def test_edge_signs_model():
    m, obs = _model()
    es = edge_signs(m, obs, tau=0.0, min_points=1)
    assert set(es) == {(0, 1), (0, 2), (1, 2)}
    for s, a in es.values():
        assert s in (1, -1) and 0.5 <= a <= 1


def test_chart_orientation_gauge():
    m, obs = _model()
    before = edge_signs(m, obs, tau=0.0, min_points=1)
    with torch.no_grad():
        m.Ec.W2[1] = m.Ec.W2[1][:, [1, 0]].clone()
        m.Ec.b2[1] = m.Ec.b2[1][[1, 0]].clone()
        m.Dc.W1[1] = m.Dc.W1[1][[1, 0]].clone()
    after = edge_signs(m, obs, tau=0.0, min_points=1)
    for (j, k), (s, a) in before.items():
        s2, a2 = after[(j, k)]
        assert s2 == (-s if 1 in (j, k) else s) and a2 == pytest.approx(a)
    assert w1_class({e: s for e, (s, _) in before.items()}) == w1_class({e: s for e, (s, _) in after.items()})

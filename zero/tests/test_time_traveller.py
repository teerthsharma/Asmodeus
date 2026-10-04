import pytest
import torch

from arms.planner import cem_plan
from atlas.observer_atlas import ObserverAtlas
from atlas.time_traveller import TimeTravelAtlas, budget_samples, rollout_flops

D, A, K = 16, 2, 8


def arm(cls=TimeTravelAtlas, seed=0, **kw):
    torch.manual_seed(seed)
    return cls(D=D, A=A, K=K, **kw).eval()


def data(n=5, h=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, D, generator=g), torch.randn(n, h, A, generator=g)


@pytest.mark.parametrize("m,M", [(2, 2), (2, 6)])
def test_gamma_zero_is_base(m, M):
    tt, base = arm(m=m, M=M, gamma=0.0, max_loops=3), arm(ObserverAtlas, seed=3, m=m)
    tt.load_state_dict(base.state_dict())
    o, a = data()
    with torch.no_grad():
        assert torch.allclose(tt.rollout(o, a), base.rollout(o, a), atol=1e-5)


def test_single_chart_is_base():
    tt, base = arm(m=1, M=1, gamma=5.0, max_loops=3), arm(ObserverAtlas, seed=3, m=1)
    tt.load_state_dict(base.state_dict())
    o, a = data()
    with torch.no_grad():
        assert torch.allclose(tt.rollout(o, a), base.rollout(o, a), atol=1e-5)


def test_reselection_drops_deviant_chart():
    tt, base = arm(m=2, M=K, gamma=10.0, max_loops=3), arm(ObserverAtlas, seed=3, m=2)
    tt.load_state_dict(base.state_dict())
    o, a = data()
    with torch.no_grad():
        _, idx = base.route(base.cell(o))
        k = int(idx[0, 1])  # the lower-weight chart of row 0 (weight < 1/2), so it is the outlier of the pair
        tt.phi.b2.data[k] += 100
        _, info = tt.rollout(o, a, return_info=True)
    assert k in idx[0].tolist() and k not in info["halt"][0].tolist()


def test_halt_is_fixed_point():
    tt = arm(m=2, M=K, gamma=3.0, max_loops=10)
    o, a = data()
    with torch.no_grad():
        _, info = tt.rollout(o, a, return_info=True)
        assert info["loops"] <= 10
        if info["loops"] < 10:
            tt.max_loops = info["loops"] + 1
            _, more = tt.rollout(o, a, return_info=True)
            assert torch.equal(more["halt"].sort(-1).values, info["halt"].sort(-1).values)
            assert more["loops"] == info["loops"]


def test_flops_and_budget():
    base = arm(ObserverAtlas, m=2)
    assert rollout_flops(arm(m=2, M=2, max_loops=0), 10) == rollout_flops(base, 10)
    for kw in ("M", "max_loops"):
        vals = [rollout_flops(arm(m=2, **{"M": 2, "max_loops": 1, kw: v}), 10) for v in ((2, 4, 8) if kw == "M" else (0, 1, 2))]
        assert vals[0] < vals[1] < vals[2]
    B = 10 * 800 * rollout_flops(base, 10)
    assert budget_samples(arm(m=2, M=8, max_loops=1), B, 10, 10) < budget_samples(base, B, 10, 10) == 800


def test_planner_compatible():
    tt = arm(m=2, M=4, gamma=1.0, max_loops=2)
    a0, plan = cem_plan(tt, torch.randn(2, D), torch.randn(2, tt.D_g), horizon=3, act_dim=A,
                        n_samples=16, n_elites=4, iters=2)
    assert a0.shape == (2, A) and plan.shape == (2, 3, A)


@pytest.mark.parametrize("m,M", [(3, 2), (2, K + 1)])
def test_bad_M(m, M):
    with pytest.raises(ValueError):
        arm(m=m, M=M)

"""EventDriven (AETHER 'observed -> exists' in time): wraps any contract arm; recompute only when
Delta_t = ||o_t - o_tau|| / sqrt(D) >= eps. CPU, float64 tiny models."""
import math
import pathlib
import sys

import pytest
import torch

torch.set_num_threads(4)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from atlas.event_driven import EventDriven  # noqa: E402
from atlas.asmodeous import Asmodeous  # noqa: E402
from atlas.observer_atlas import ObserverAtlas  # noqa: E402
from atlas.phase_sphere import PhaseSphere  # noqa: E402

D, A, B, T = 16, 2, 3, 6
ARMS = {
    "observer": lambda: ObserverAtlas(D=D, A=A, K=4, m=2, d_c=2, D_g=6, d_h=16, d_key=8, hidden=8),
    "phase": lambda: PhaseSphere(D=D, A=A, K=4, m=2, N=4, d_c=2, D_g=6, hidden=8),
    "asmodeous": lambda: Asmodeous(D=D, A=A, K=8, m=2, N=8, d_c=2, D_g=6, hidden=16),
}
kinds = pytest.mark.parametrize("kind", list(ARMS))


def wrap(kind, seed=0, **kw):
    torch.manual_seed(seed)
    return EventDriven(ARMS[kind](), **kw).double().eval()


def data(seed=1, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(B, T, D, generator=g, dtype=torch.float64) * scale
    return obs, torch.randn(B, T, A, generator=g, dtype=torch.float64)


def run(m, obs, act, seed=7):
    torch.manual_seed(seed)  # SIGReg draws random projections
    return m(obs, act)


@kinds
def test_eps_zero_is_the_arm(kind):
    m = wrap(kind, fixed_eps=0.0)
    obs, act = data()
    ref, out = run(m.arm, obs, act), run(m, obs, act)
    assert out["trig"].all() and out["trig"].shape == (B, T) and float(out["exec_frac"]) == 1.0
    assert torch.allclose(out["g"], ref["g"], atol=1e-12) and torch.allclose(out["ghat"], ref["ghat"], atol=1e-12)
    assert torch.allclose(out["loss"], ref["loss"], atol=1e-12)


@kinds
def test_eps_inf_only_first_frame_exists(kind):
    m = wrap(kind, fixed_eps=float("inf"))
    obs, act = data()
    out = run(m, obs, act)
    assert out["trig"][:, 0].all() and not out["trig"][:, 1:].any()
    assert out["g"].shape[:2] == (B, T) and out["ghat"].shape[:2] == (B, T - 1)
    g0 = out["g"][:, :1].expand_as(out["g"])
    assert torch.equal(out["g"], g0) and torch.equal(out["ghat"], g0[:, :-1])
    assert torch.isfinite(out["loss"])


@kinds
def test_compaction_exact(kind):
    m = wrap(kind, fixed_eps=1e-9)
    obs, act = data()
    obs[0, 1], obs[0, 4] = obs[0, 0], obs[0, 3]                       # row 0 triggers at {0,2,3,5}
    obs[1, 2], obs[1, 3] = obs[1, 1], obs[1, 1]                       # row 1 triggers at {0,1,4,5}
    pos = torch.tensor([[0, 2, 3, 5], [0, 1, 4, 5], [0, 1, 2, 3]])    # row 2 triggers at {0,1,2,3}
    obs[2, 4], obs[2, 5] = obs[2, 3], obs[2, 3]
    out = run(m, obs, act)
    want = torch.zeros(B, T, dtype=torch.bool)
    want.scatter_(1, pos, True)
    assert torch.equal(out["trig"], want)
    gat = lambda x: x.gather(1, pos[..., None].expand(-1, -1, x.shape[-1]))
    ref = run(m.arm, gat(obs), gat(act))
    for b in range(B):
        for k, t in enumerate(pos[b].tolist()):
            assert torch.allclose(out["g"][b, t], ref["g"][b, k], atol=1e-12)
        for t in range(T):                                            # sample-and-hold between triggers
            k = int(want[b, : t + 1].sum()) - 1
            assert torch.allclose(out["g"][b, t], ref["g"][b, k], atol=1e-12)
            if t < T - 1:                                             # ghat: arm's at a trigger, hold otherwise
                e = ref["ghat"][b, k] if want[b, t + 1] else ref["g"][b, k]
                assert torch.allclose(out["ghat"][b, t], e, atol=1e-12)


@kinds
def test_flops_proportional_to_triggers(kind):
    obs, act = data()
    prev, fl = None, []
    for eps in (0.0, 0.5, 1.0, 1.5, 2.0, float("inf")):
        m = wrap(kind, fixed_eps=eps)
        out = run(m, obs, act)
        ef = m.executed_flops(out["trig"])
        assert ef == int(out["trig"].sum()) * m.arm.active_flops()
        assert m.active_flops() == round(m.arm.active_flops() * float(out["exec_frac"]))
        fl.append(ef)
    assert fl == sorted(fl, reverse=True) and fl[0] == B * T * m.arm.active_flops() and fl[-1] == B * m.arm.active_flops()


@kinds
def test_training_grad_reaches_rho(kind):
    m = wrap(kind, eps_init=0.1).train()
    obs, act = data(scale=0.1)                                        # Delta ~ 0.14 ~ eps: gate is live
    torch.manual_seed(7)
    out = m(obs, act)
    assert 0 < float(out["exec_frac"]) <= 1 and torch.isfinite(out["loss"])
    out["loss"].backward()
    gr = m.rho.grad
    assert gr is not None and torch.isfinite(gr).all() and gr.abs().sum() > 0


def test_tie_triggers():
    m = wrap("observer", fixed_eps=1.0)
    obs, act = data()
    obs[:, 0] = 0
    obs[:, 1] = 1.0                                                   # ||1_16|| / sqrt(16) == 1.0 exactly
    obs[:, 2:] = 1.0                                                  # no change since the last trigger
    assert run(m, obs, act)["trig"].tolist() == [[True, True] + [False] * (T - 2)] * B
    m2 = wrap("observer", fixed_eps=math.nextafter(1.0, 2.0))
    assert run(m2, obs, act)["trig"][:, 1:].sum() == 0


def test_params_and_rollout():
    m = wrap("observer", eps_init=0.1)
    assert m.n_params() == m.arm.n_params() + 1
    obs, act = data()
    r = m.rollout(obs[:, 0], act[:, :3])
    assert torch.equal(r, m.arm.rollout(obs[:, 0], act[:, :3]))

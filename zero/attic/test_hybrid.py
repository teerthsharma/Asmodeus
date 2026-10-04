import numpy as np
import pytest
import torch

import t6_contextuality as t6
from arms.planner import cem_plan
from atlas.coherent_observer import CoherentAtlas
from atlas.hybrid import HybridAtlas, chern_from_states, chern_s2, observer_states
from atlas.observer_atlas import ObserverAtlas
from atlas.time_traveller import TimeTravelAtlas, budget_samples, rollout_flops
from ref.berry_check import icosphere, spin_state

D, A, K, m = 16, 2, 8, 2


def mk(cls, seed=0, **kw):
    torch.manual_seed(seed)
    return cls(D=kw.pop("D", D), A=A, K=K, m=m, **kw).eval()


def randomise(arm, seed=7):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in (arm.Cp, arm.Cq, arm.theta_keys):
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)


def data(n=3, T=5, h=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, T, D, generator=g), torch.randn(n, T, A, generator=g), torch.randn(n, h, A, generator=g)


def test_mro():
    assert HybridAtlas.__mro__[1:4] == (TimeTravelAtlas, CoherentAtlas, ObserverAtlas)


@torch.no_grad()
def test_full_decoherence_no_travel_is_observer_atlas():
    hy, base = mk(HybridAtlas, M=4, gamma=0.0), mk(ObserverAtlas)
    hy.load_state_dict(base.state_dict(), strict=False)
    assert hy.Cp.abs().sum() == 0 and hy.theta_keys.abs().sum() == 0
    obs, act, ra = data()
    a, b = hy(obs, act), base(obs, act)
    assert torch.allclose(a["w"], b["w"], atol=1e-5) and torch.allclose(a["g"], b["g"], atol=1e-5)
    assert torch.allclose(hy.rollout(obs[:, 0], ra), base.rollout(obs[:, 0], ra), atol=1e-5)


@torch.no_grad()
def test_gamma_zero_is_coherent_atlas():
    hy, co = mk(HybridAtlas, M=4, gamma=0.0), mk(CoherentAtlas, seed=3)
    hy.load_state_dict(co.state_dict(), strict=False)
    randomise(hy)
    randomise(co)
    obs, _, ra = data()
    assert torch.allclose(hy.rollout(obs[:, 0], ra), co.rollout(obs[:, 0], ra), atol=1e-5)


@torch.no_grad()
def test_zero_coherence_is_time_traveller():
    kw = dict(M=6, gamma=3.0, max_loops=3)
    hy, tt = mk(HybridAtlas, **kw), mk(TimeTravelAtlas, seed=3, **kw)
    hy.load_state_dict(tt.state_dict(), strict=False)
    obs, _, ra = data()
    (o1, i1), (o2, i2) = hy.rollout(obs[:, 0], ra, return_info=True), tt.rollout(obs[:, 0], ra, return_info=True)
    assert torch.allclose(o1, o2, atol=1e-5) and torch.equal(i1["halt"], i2["halt"])


@torch.no_grad()
def test_amplitudes_are_born_consistent():
    hy = mk(HybridAtlas)
    randomise(hy)
    c = torch.randn(10, hy.d_h)
    psi = hy.amplitudes(c)
    assert psi.is_complex() and psi.shape == (10, K)
    assert torch.allclose(psi.abs() ** 2, hy.full_logits(c).exp(), atol=1e-5)
    assert torch.allclose(psi.norm(dim=-1), torch.ones(10), atol=1e-5)


def test_flops_charge_the_coherent_decider():
    kw = dict(M=4, gamma=1.0, max_loops=2)
    hy, tt = mk(HybridAtlas, **kw), mk(TimeTravelAtlas, **kw)
    extra = CoherentAtlas.active_flops(hy) - ObserverAtlas.active_flops(hy)
    assert extra > 0 and rollout_flops(hy, 10) == rollout_flops(tt, 10) + extra
    B = 10 * 800 * rollout_flops(tt, 10)
    assert budget_samples(hy, B, 10, 10) <= budget_samples(tt, B, 10, 10)


def test_chern_plumbing():
    V, F = icosphere(2)
    psi = np.stack([spin_state(v, 0.5) for v in V])
    assert abs(round(chern_from_states(V, F, psi))) == 1
    assert round(chern_from_states(V, F, np.tile(psi[:1], (len(V), 1)))) == 0


def test_decoherence_limit_is_topologically_trivial():
    hy = mk(HybridAtlas, D=256)
    val = chern_s2(hy, level=1)
    assert round(val) == 0 and abs(val) < 1e-6


def test_planner_and_t6_compatible():
    hy = mk(HybridAtlas, M=4)
    first, plan = cem_plan(hy, torch.randn(2, D), torch.randn(2, hy.D_g), horizon=3, act_dim=2,
                           n_samples=16, n_elites=4, iters=2)
    assert first.shape == (2, 2) and plan.shape == (2, 3, 2)
    ab, ba = t6.answers(hy, torch.randn(16, 3, D), torch.randn(16, 3, D), torch.randn(16, 3, D))
    assert ab.shape == (16, 2) and ba.shape == (16, 2)


def test_contract_and_param_count():
    hy, base = mk(HybridAtlas, M=4), mk(ObserverAtlas)
    hy.train()
    obs, act, _ = data()
    r = hy(obs, act)
    assert {"loss", "g", "ghat", "w"} <= set(r)
    r["loss"].backward()
    assert hy.Cp.grad is not None and hy.Cp.grad.abs().sum() > 0
    assert hy.n_params() - base.n_params() == 2 * K * hy.rank + K * hy.d_key


def test_observer_states_shape():
    hy = mk(HybridAtlas, D=256)
    assert observer_states(hy, np.random.randn(5, 256).astype(np.float32)).shape == (5, K)


def test_observer_states_restores_training_mode():
    hy = mk(HybridAtlas, D=256).train()
    observer_states(hy, np.random.randn(3, 256).astype(np.float32))
    assert hy.training is True


def test_sign_holonomy_readout_is_callable():
    pytest.importorskip("atlas.phase_sphere")
    from atlas.hybrid import sign_holonomy_readout
    assert callable(sign_holonomy_readout)


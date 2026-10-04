import math

import pytest
import torch

import beds
import candidates
from candidates import CANDIDATES, build, shared_knobs

DG = 6  # D_g = 2n + 2 with n = 2


def _make(name):
    torch.manual_seed(0)
    return build(name, 256, 3, 2, tiny=True)


def _batch(poison=False):
    bed = beds.make_bed("s2", 4, 8)
    if poison:
        bed["state"][:] = float("nan")
    return torch.from_numpy(bed["obs"]), torch.from_numpy(bed["act"])


def _lam_value(arm):
    lam = arm.lam
    return lam["sig"] if isinstance(lam, dict) else lam


def test_candidates_are_the_three():
    assert CANDIDATES == ("atlas", "phase_sphere")   # hybrid -> attic; ASMODEOUS joins when atlas/asmodeous.py lands


@pytest.mark.parametrize("name", CANDIDATES)
def test_arm_contract(name):
    arm = _make(name)
    obs, act = _batch()
    out = arm(obs, act)
    assert {"loss", "g", "ghat", "w"} <= set(out)
    assert out["g"].shape == (4, 8, DG) and out["ghat"].shape == (4, 7, DG)
    assert out["loss"].dim() == 0 and math.isfinite(out["loss"].item())
    out["loss"].backward()
    if "z" in out:
        assert out["z"].shape[:3] == (4, 8, 2) and out["zidx"].shape == (4, 8, 2)
    with torch.no_grad():
        assert arm.rollout(obs[:, 0], act[:, :3]).shape == (4, 3, DG)
    for v in (arm.active_flops(), arm.n_params()):
        assert isinstance(v, int) and v > 0


@pytest.mark.parametrize("name", CANDIDATES)
def test_deterministic(name):
    obs, act = _batch()
    g = [_make(name)(obs, act)["g"] for _ in range(2)]
    assert torch.equal(g[0], g[1])


@pytest.mark.parametrize("name", CANDIDATES)
def test_lgiven_state_poison_is_invisible(name):
    # build/forward take only obs/act; the bed (with its true state) never reaches the arm
    obs, act = _batch()
    obs2, act2 = _batch(poison=True)
    assert torch.equal(obs, obs2) and torch.equal(act, act2)
    assert torch.equal(_make(name)(obs, act)["g"], _make(name)(obs2, act2)["g"])


@pytest.mark.parametrize("name", CANDIDATES)
def test_shared_knobs(name):
    knobs = shared_knobs(name)
    arm = build(name, 256, 3, 2, tiny=True, lam=0.0, ema=0.0)
    assert _lam_value(arm) == 0.0
    assert getattr(arm, knobs["momentum"]) == 0.0
    assert knobs["lam"] == "lam"

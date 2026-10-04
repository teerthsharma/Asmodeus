"""Router causality / batch-independence contracts for every arm with a router (or BatchNorm)."""
import copy
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from atlas.observer_atlas import ObserverAtlas  # noqa: E402
from test_arms import CFG, D, A  # noqa: E402

torch.set_num_threads(4)
B, T = 4, 6
ARMS = ["atlas"] + list(CFG)
KEYS = ("g", "w", "w_soft")
# PEER's query net has BatchNorm: train-mode batch statistics couple samples (expected, max diff 2.8e-01 on g).
TRAIN_XFAIL = {"peer"}


def make(name):
    torch.manual_seed(0)
    if name == "atlas":
        return ObserverAtlas(D, A, K=8, m=2, d_c=2, D_g=6, d_h=16, d_key=8, hidden=8)
    return CFG[name][0]()


def data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(B, T, D, generator=g), torch.randn(B, T, A, generator=g)


def fwd(m, obs, act):
    torch.manual_seed(7)  # SIGReg-style random projections
    return m(obs, act)


def check(ref, out, sl, keys, tag):
    """allclose(1e-6) must hold; bitwise mismatches are printed, not failed."""
    for k in keys:
        if ref.get(k) is None:
            continue
        a, b = ref[k][sl], out[k][sl]
        d = (a - b).abs().max().item()
        if not torch.equal(a, b):
            print(f"NON-BITWISE {tag} {k}: max abs diff {d:.3e}")
        assert torch.allclose(a, b, atol=1e-6), f"{tag} {k} max diff {d:.3e}"


@pytest.mark.parametrize("name", ARMS)
def test_router_weights_causal_in_obs_and_act(name):
    m = make(name).eval()
    obs, act = data()
    with torch.no_grad():
        ref = fwd(m, obs, act)
        o2 = obs.clone()
        o2[:, 3:] += 5.0
        out = fwd(m, o2, act)
        check(ref, out, (slice(None), slice(0, 3)), KEYS + ("ghat",), "obs")
        a2 = act.clone()
        a2[:, 3:] += 5.0
        out = fwd(m, obs, a2)
        check(ref, out, (slice(None), slice(0, 3)), KEYS + ("ghat",), "act")


@pytest.mark.parametrize("name", ARMS)
def test_batch_independence_eval(name):
    m = make(name).eval()
    obs, act = data()
    with torch.no_grad():
        ref = fwd(m, obs, act)
        o2 = obs.clone()
        o2[1:] += 5.0
        out = fwd(m, o2, act)
        check(ref, out, slice(0, 1), KEYS + ("ghat",), "batch-eval")


@pytest.mark.parametrize(
    "name", [pytest.param(n, marks=pytest.mark.xfail(strict=False, reason="train-mode batch coupling"))
             if n in TRAIN_XFAIL else n for n in ARMS])
def test_batch_independence_train_mode_report(name):
    base = make(name).train()
    obs, act = data()
    with torch.no_grad():
        ref = fwd(copy.deepcopy(base), obs, act)
        o2 = obs.clone()
        o2[1:] += 5.0
        out = fwd(copy.deepcopy(base), o2, act)
        check(ref, out, slice(0, 1), KEYS + ("ghat",), "batch-train")

"""Inspector L-GIVEN audit (PROMPT.md §1, §10): no bed-supplied structure reaches a model.

Static: every model's forward/rollout takes only (obs|obs0, act) plus defaulted config; no module under
atlas/ or arms/ imports beds or reads state/geo/meta.
Dynamic: on a real bed, outputs are bit-identical when state/geo/meta are destroyed; the router's autograd
leaves are only (parameters, obs, act); routing at t never reads obs/act after t (no target leak).
"""
import inspect
import tokenize
from pathlib import Path

import numpy as np
import pytest
import torch

from arms.chart_ae import ChartAEJEPA
from arms.mop_jepa import MoPJEPA
from arms.peer_jepa import PEERJEPA
from arms.single_jepa import SingleJEPA
from atlas.observer_atlas import ObserverAtlas
from beds import make_bed
from metrics import pred_error

ZERO = Path(__file__).resolve().parents[1]
MODEL_DIRS = ("atlas", "arms")
FORBIDDEN = {"state", "geo", "meta"}

ARMS = {
    "atlas": lambda D, A: ObserverAtlas(D, A, K=4, m=2, d_c=2, D_g=6, d_h=16, d_key=8, hidden=8),
    "single": lambda D, A: SingleJEPA(D, A, d=4, hidden=16),
    "mop": lambda D, A: MoPJEPA(D, A, d=4, K=3, hidden=16, head_hidden=8, router_hidden=8),
    "chart_ae": lambda D, A: ChartAEJEPA(D, A, K=3, d_c=2, D_g=6, hidden=16, chart_hidden=8),
    "peer": lambda D, A: PEERJEPA(D, A, d=4, hidden=16, n_experts=16, heads=2, topk=2, d_key=8),
}
ROUTED = ("atlas", "mop", "chart_ae")                    # arms whose output dict carries observer weights w


def sources():
    for d in MODEL_DIRS:
        yield from sorted((ZERO / d).rglob("*.py"))


# ---- static ------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ARMS)
def test_signature_obs_act_only(name):
    arm = ARMS[name](8, 2)
    for meth, first in (("forward", "obs"), ("rollout", "obs0")):
        ps = list(inspect.signature(getattr(arm, meth)).parameters.values())
        assert [p.name for p in ps[:2]] == [first, "act"], f"{name}.{meth} inputs {[p.name for p in ps]}"
        extra = ps[2:]
        assert all(p.default is not inspect.Parameter.empty for p in extra), f"{name}.{meth} extra required arg"
        assert not {p.name for p in extra} & FORBIDDEN, f"{name}.{meth} takes a bed-only field"


def test_no_bed_access_in_models():
    hits = []
    for f in sources():
        toks = [t for t in tokenize.generate_tokens(f.open(encoding="utf-8").readline)
                if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE)]
        for i, t in enumerate(toks):
            prev = toks[i - 1].string if i else ""
            if t.type == tokenize.NAME and t.string in FORBIDDEN | {"beds", "make_bed"}:
                hits.append(f"{f.relative_to(ZERO)}:{t.start[0]} name {t.string}")
            if (t.type == tokenize.STRING and t.string.strip("'\"") in FORBIDDEN and
                    (prev == "[" or (prev == "(" and toks[i - 2].string in ("get", "getattr", "pop")))):
                hits.append(f"{f.relative_to(ZERO)}:{t.start[0]} key {t.string}")
    assert not hits, hits


# ---- dynamic -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bed():
    return make_bed("s2", 3, 7, seed=0)


def run(arm, b, seed=0):
    torch.manual_seed(seed)
    out = arm(torch.from_numpy(b["obs"]), torch.from_numpy(b["act"]))
    torch.manual_seed(seed)
    ro = arm.rollout(torch.from_numpy(b["obs"][:, 0]), torch.from_numpy(b["act"][:, :3]))
    return {k: v for k, v in out.items() if torch.is_tensor(v)} | {"rollout": ro}


@pytest.mark.parametrize("name", ARMS)
def test_state_never_reaches_model(name, bed):
    D, A = bed["obs"].shape[-1], bed["act"].shape[-1]
    torch.manual_seed(0)
    arm = ARMS[name](D, A).eval()
    poisoned = dict(bed, state=np.full_like(bed["state"], np.nan), geo=None, meta=None)
    a, b = run(arm, bed), run(arm, poisoned)
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), f"{name}: output {k} changed when state was destroyed"
        assert torch.isfinite(a[k]).all(), f"{name}: output {k} not finite"
    # control: the same harness does see obs
    moved = dict(bed, obs=bed["obs"] + 1.0)
    assert not torch.equal(run(arm, moved)["g"], a["g"]), f"{name}: g ignores obs (harness not live)"
    # Foreman's bed-taking scorer: the one path where a bed dict meets an arm
    torch.manual_seed(0)
    pa = pred_error(arm, bed, h=1)
    torch.manual_seed(0)
    assert pred_error(arm, poisoned, h=1) == pa, f"{name}: metrics.pred_error changes with state"


@pytest.mark.parametrize("name", ROUTED)
def test_router_reads_only_obs_act(name, bed):
    D, A = bed["obs"].shape[-1], bed["act"].shape[-1]
    torch.manual_seed(0)
    arm = ARMS[name](D, A).eval()
    obs = torch.from_numpy(bed["obs"]).requires_grad_(True)
    act = torch.from_numpy(bed["act"]).requires_grad_(True)
    w = arm(obs, act)["w"]
    assert w is not None
    leaves, seen, stack = set(), set(), [w.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        if hasattr(fn, "variable"):
            leaves.add(id(fn.variable))
        stack += [nf for nf, _ in fn.next_functions]
    allowed = {id(p) for p in arm.parameters()} | {id(obs), id(act)}
    assert leaves <= allowed, f"{name}: router reads a tensor that is neither a parameter nor obs/act"
    assert id(obs) in leaves, f"{name}: router does not read obs"


@pytest.mark.parametrize("name", ROUTED)
def test_router_causal(name, bed):
    """w_t and g_t must not change when obs/act after t are perturbed (no next-frame leak into routing)."""
    D, A = bed["obs"].shape[-1], bed["act"].shape[-1]
    torch.manual_seed(0)
    arm = ARMS[name](D, A).eval()
    obs, act = torch.from_numpy(bed["obs"]), torch.from_numpy(bed["act"])
    t = obs.shape[1] // 2
    o2, a2 = obs.clone(), act.clone()
    o2[:, t + 1:] += torch.randn_like(o2[:, t + 1:])
    a2[:, t + 1:] += 1.0
    with torch.no_grad():
        torch.manual_seed(1)
        r1 = arm(obs, act)
        torch.manual_seed(1)
        r2 = arm(o2, a2)
    for k in ("w", "g"):
        assert torch.allclose(r1[k][:, :t + 1], r2[k][:, :t + 1], atol=1e-6), f"{name}: {k}[:t] reads the future"

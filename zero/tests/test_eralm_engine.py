"""Era engine parity: the torch bank + decider filter reproduce the numpy reference (research/e0) token for token."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from eralm import filter as F
from eralm.engine import EraBank, bocpd, logml

E0 = Path(__file__).resolve().parents[1] / "research" / "e0"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"e0_{name}", E0 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"e0_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


bed = _load("bed")
arms = _load("arms")
D = _load("deciders")
V, T = bed.V, 1500
BEDS = [("REC", 7, {}), ("CYC", 8, {"seg_min": 20, "seg_mean": 100}), ("NOREC", 9, {}), ("NULL", 10, {})]
SPECS = [("MOM", 0.003, None), ("SOFT", 0.003, None), ("HARD", 0.003, 0.9), ("TEMP", 0.01, 0.75), ("GBAYES", 0.003, 0.8),
         ("MOM+GB", 0.003, 0.8)]
MAKE = {"SOFT": lambda a, p: D.Soft(2, a), "MOM": lambda a, p: D.Momentum(2, a), "HARD": lambda a, p: D.HardJump(2, a, p),
        "TEMP": lambda a, p: D.Tempered(2, a, p), "GBAYES": lambda a, p: D.GeneralBayes(2, a, p),
        "MOM+GB": lambda a, p: D.MomentumGB(2, a, p)}


@pytest.fixture(scope="module")
def runs():
    xs, refs = [], []
    for kind, seed, kw in BEDS:
        x, _, _ = bed.make_bed(seed, kind=kind, T=T, **kw)
        bl, bvec = arms.bocpd(x, V, 1 / 400, return_vec=True)
        ds = [MAKE[n](a, p) for n, a, p in SPECS]
        res = arms.fcpa(x, V, 0.05, 32, ext=np.exp(-bl), merge="global", n_min=64, deciders=ds, ext_vec=bvec,
                        return_info=True)
        xs.append(x)
        refs.append((bl, res, ds))
    x = torch.tensor(np.stack(xs))
    pe = bocpd(x, V, 1 / 400)
    out = EraBank(V, merge_rule="bf1", K=4).run(x, pe, block=128)     # K=4 forces slot growth on NOREC
    return x, pe, out, refs


def test_bocpd_parity(runs):
    _, pe, _, refs = runs
    for b, (bl, _, _) in enumerate(refs):
        assert np.abs(-np.log(pe[b].numpy()) - bl).max() < 1e-10


def test_bank_parity(runs):
    _, _, out, refs = runs
    for b, (_, res, _) in enumerate(refs):
        assert out.n_eras[b] == res["n_eras"]
        assert out.births[b] == res["births"]
        assert np.abs(out.hard[b].numpy() - res["hard"]).max() < 1e-10
    assert out.K > 4


@pytest.mark.parametrize("i", range(len(SPECS)))
def test_decider_parity(runs, i):
    _, _, out, refs = runs
    n, a, p = SPECS[i]
    d = F.Decider(n, a, p)
    ls, ss = F.sequential(out, d)
    lc, sc = F.scan(out, d, chunk=64)
    for b, (_, res, ds) in enumerate(refs):
        assert np.abs(ls[b].numpy() - res["dec"][ds[i]]).max() < 1e-10
        assert abs(float(ss[b]) - ds[i].single / (T - 1)) < 1e-12
    assert (lc - ls).abs().max() < 1e-10
    assert (sc - ss).abs().max() < 1e-12


def test_anytime_merge_is_stricter(runs):
    x, pe, out, _ = runs
    strict = EraBank(V, merge_rule="anytime", merge_delta=0.05).run(x, pe)
    folds = lambda o: [sum(e[1] == "fold" for e in ev) for ev in o.events]
    assert sum(folds(strict)) <= sum(folds(out))
    assert strict.n_eras[3] == 1                                    # NULL: still a single era


def test_logml_matches_reference():
    g = torch.Generator().manual_seed(0)
    C = torch.randint(0, 9, (V, V), generator=g).double()
    tabs = arms._logml_table(V, 200)
    assert abs(float(logml(C)) - arms._logml(C.numpy(), tabs)) < 1e-9

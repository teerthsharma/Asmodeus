"""Phase 2 bench checks: frozen verdict rule, LR schedule, a CUDA smoke cell, and prereg-before-results order.
Run from zero/: python -m pytest -q p2/test_p2.py"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import bench  # noqa: E402
from frontier import SEEDS, verdict  # noqa: E402

B = 100_000_000


def rows(arm_loss, arm_flops, arm_params, sd=0.0):
    lad = {"gpt-s": (1e15, 1e6, 2.0), "gpt-m": (4e15, 3e6, 1.8), "gpt-l": (1.6e16, 1e7, 1.6)}
    out = []
    for i, s in enumerate(SEEDS):
        e = (i - 1) * sd
        for n, (f, p, l) in lad.items():
            out.append(dict(name=n, seed=s, tokens_budget=B, train_flops=f, n_params=p, test_loss=l + e))
        out.append(dict(name="era", seed=s, tokens_budget=B, train_flops=arm_flops, n_params=arm_params,
                        test_loss=arm_loss + e))
    return out


def test_verdict_rules():
    v = verdict(rows(1.70, 2e15, 3e6), "era", B)        # frontier at 2e15 FLOPs = 1.9; at 3e6 params = 1.8
    assert v["B1_equal_flops"]["result"] == "win" and v["B1_equal_flops"]["frontier_loss"] == pytest.approx(1.9)
    assert v["B2_equal_params"]["result"] == "win" and v["B2_equal_params"]["delta"] == pytest.approx(-0.1)
    assert verdict(rows(1.81, 4e15, 3e6), "era", B)["B2_equal_params"]["result"] == "tie"
    assert verdict(rows(1.90, 4e15, 3e6), "era", B)["B1_equal_flops"]["result"] == "lose"
    assert verdict(rows(1.5, 1e17, 1e8), "era", B)["B1_equal_flops"]["result"] == "out_of_range"
    wide = verdict(rows(1.77, 4e15, 3e6, sd=0.03), "era", B)  # seed sd 0.03 widens the margin to 0.06
    assert wide["margin"] == pytest.approx(0.06) and wide["B1_equal_flops"]["result"] == "tie"
    assert not verdict(rows(1.7, 2e15, 3e6)[:-1], "era", B)["complete"]


def test_lr_schedule():
    assert bench.lr_at(0, 1000, 1.0) == pytest.approx(0.1)            # warmup 10 steps
    assert bench.lr_at(10, 1000, 1.0) == pytest.approx(1.0)
    assert bench.lr_at(999, 1000, 1.0) == pytest.approx(0.1, abs=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bench cells run on CUDA")
def test_cuda_smoke_cell(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for s in ("train", "val", "test"):                                  # a learnable stream: tokens repeat
        rng.integers(0, 64, 300_000).astype(np.uint16).repeat(2).tofile(tmp_path / f"{s}.bin")
    monkeypatch.setattr(bench, "VAL_WINDOWS", 64)
    kw = dict(d_model=64, n_layer=2, n_head=2)
    r = bench.cell("gpt-s", 0, 64 * 256 * 40, 3e-3, tmp_path, kw=kw, evals=4)
    c = r["curve"]
    assert len(c) == 4 and [x["tokens"] for x in c] == sorted(x["tokens"] for x in c)
    assert c[-1]["val_loss"] < c[0]["val_loss"] < math.log(4096)
    assert math.isfinite(r["test_loss"]) and r["peak_mem_mb"] > 0 and r["train_tok_per_s"] > 0
    assert r["fwd_flops_per_token"] == max(r["fwd_flops_counted"], r["fwd_flops_declared"])


def test_prereg_before_results():
    git = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.split()
    res = git("log", "--format=%H", "--", "results")
    if not res:
        return
    pre = git("log", "--format=%H", "--diff-filter=A", "--", "PREREG.md")
    assert pre, "results committed but p2/PREREG.md never was"
    for c in res:
        assert c != pre[-1], f"{c[:9]} adds PREREG.md and results together"
        assert subprocess.run(["git", "merge-base", "--is-ancestor", pre[-1], c], cwd=HERE).returncode == 0

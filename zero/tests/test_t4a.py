"""T4a collapse-timing runner (run_t4a.py): lam scaling, arms, latents, collapse stats, verdict, plan, run_job, CLI."""
import json
import math

import numpy as np
import pytest
import torch

import beds
import run_t4a as R

KINDS = ("single", "atlas", "mop", "lazy_chart_ae")
TINY = dict(steps=4, log_every=2, n_traj=8, T=40, device="cpu", tiny=True)


def job(tmp_path, arm="single", scale=0.1, seed=0):
    key = f"{arm}__s2__lam{scale:g}__s{seed}"
    return dict(arm=arm, bed="s2", scale=scale, seed=seed, key=key, path=tmp_path / f"{key}.json", done=False)


def test_lam_for():
    assert R.lam_for(1.0) == pytest.approx(0.9)
    assert R.lam_for(0.1) == pytest.approx(0.9 / 1.9)
    assert R.lam_for(0.0) == 0.0
    assert R.lam_for(0.1, "naive") == pytest.approx(0.09)


@pytest.mark.parametrize("kind", KINDS)
def test_build_arm_lam(kind):
    meta = beds.make_bed("s2", 4, 8)["meta"]
    arm = R.build_arm(kind, meta, 256, 3, 0.3, tiny=True)
    assert isinstance(arm, torch.nn.Module) and arm.lam == 0.3


def _pairs(arm):
    """(target, online) module pairs: atlas E_tgt/E, else every child tgt_X / X."""
    if hasattr(arm, "E_tgt"):
        return [(arm.E_tgt, arm.E)]
    return [(c, getattr(arm, n[4:])) for n, c in arm.named_children() if n.startswith("tgt_")]


@pytest.mark.parametrize("kind", KINDS)
def test_build_arm_target(kind):
    assert R.TARGETS == ("ema", "online")
    b = beds.make_bed("s2", 4, 8)
    mom = lambda a: a.ema if kind == "atlas" else a.tau
    assert mom(R.build_arm(kind, b["meta"], 256, 3, 0.3, tiny=True)) == 0.996
    obs, act = torch.tensor(b["obs"]), torch.tensor(b["act"])
    same = {}
    for tgt in R.TARGETS:
        torch.manual_seed(0)
        arm = R.build_arm(kind, b["meta"], 256, 3, 0.3, tiny=True, target=tgt).train()
        assert mom(arm) == (0 if tgt == "online" else 0.996)
        opt = torch.optim.AdamW(arm.parameters(), lr=1e-2)
        arm(obs, act)["loss"].backward()
        opt.step()
        arm(obs, act)  # the lerp runs at the start of a train-mode forward
        pairs = _pairs(arm)
        assert pairs
        same[tgt] = all(torch.equal(pt, po) for t, o in pairs for pt, po in zip(t.parameters(), o.parameters()))
    assert same == {"ema": False, "online": True}


@pytest.mark.parametrize("kind", KINDS)
def test_latents_shapes(kind):
    b = beds.make_bed("s2", n_traj=4, T=8)
    arm = R.build_arm(kind, b["meta"], 256, 3, 0.3, tiny=True)
    out = R.latents(arm, torch.tensor(b["obs"]), torch.tensor(b["act"]))
    g, z = out["g"], out["z"]
    assert g.shape[0] == 32 and np.isfinite(g).all() and z
    if kind in ("atlas", "lazy_chart_ae"):
        assert set(z) <= set(range(arm.K)) and all(isinstance(k, int) for k in z)
        assert all(v.shape[-1] == b["meta"]["n"] for v in z.values())
    else:
        assert list(z) == [0] and z[0].shape[-1] == g.shape[-1]


def test_steps_to_collapse():
    assert R.steps_to_collapse([0, 50, 100], [3, 1.6, 1.4]) == (100, False)
    assert R.steps_to_collapse([0, 50, 100], [3, 2, 2], budget=100) == (100, True)
    assert R.steps_to_collapse([0, 10, 20], [1.0, 3, 1.0]) == (0, False)


def test_ratio_ci():
    gm, lo, hi = R.ratio_ci([200, 220, 240], [100, 100, 100])
    assert gm == pytest.approx(2.2, rel=0.02) and lo > 1 and hi >= gm >= lo
    assert R.ratio_ci([200, 220, 240], [100, 100, 100]) == (gm, lo, hi)


def test_verdict():
    F = [False] * 3
    assert R.verdict([200, 220, 240], F, [100] * 3, F) == "PASS"
    assert R.verdict([100] * 3, F, [100] * 3, F) == "KILL"
    assert R.verdict([50, 60, 55], F, [100] * 3, F) == "KILL"
    assert R.verdict([200, 220, 240], F, [100] * 3, [True] * 3) == "VOID"


def test_plan_grid_and_resume(tmp_path):
    jobs = R.plan(["single", "mop"], ["s2"], [1.0, 0.1], [0, 1], out=tmp_path)
    assert [j["key"] for j in jobs] == [f"{a}__s2__lam{s:g}__s{k}" for a in ("single", "mop") for s in (1.0, 0.1) for k in (0, 1)]
    assert not any(j["done"] for j in jobs)
    (tmp_path / "mop__s2__lam0.1__s1.json").write_text("{}")
    assert [j["done"] for j in R.plan(["single", "mop"], ["s2"], [1.0, 0.1], [0, 1], out=tmp_path)].count(True) == 1


def test_plan_targets(tmp_path):
    jobs = R.plan(["single"], ["s2"], [1.0, 0.0], [0], out=tmp_path, targets=["ema", "online"])
    assert [(j["target"], j["key"]) for j in jobs] == [
        (t, f"single__s2__lam{s:g}__s0{x}") for t, x in (("ema", ""), ("online", "__online")) for s in (1.0, 0.0)]
    assert {j["target"] for j in R.plan(["single"], ["s2"], [1.0], [0], out=tmp_path)} == {"ema"}
    with pytest.raises(KeyError):
        R.plan(["single"], ["s2"], [1.0], [0], out=tmp_path, targets=["nope"])


def test_unknown_arm_rejected(tmp_path):
    with pytest.raises(KeyError):
        R.plan(["nope"], ["s2"], [1.0], [0], out=tmp_path)


def test_run_job_record(tmp_path):
    j = job(tmp_path)
    rec = R.run_job(j, "abc", **TINY)
    assert json.loads(j["path"].read_text()) == rec and rec["sha"] == "abc"
    assert [c["step"] for c in rec["curve"]] == [0, 2, 4]
    for c in rec["curve"]:
        assert isinstance(c["g_erank"], float) and isinstance(c["chart_erank"], dict) and "loss" in c
    assert rec["lam"] == pytest.approx(R.lam_for(0.1))
    assert isinstance(rec["censored_g"], bool) and isinstance(rec["censored_chart"], bool)
    assert isinstance(rec["t_collapse_g"], int) and isinstance(rec["t_collapse_chart"], int)
    assert rec["censored_gate"] is True and all(c["n_charts_used"] <= c["n_charts_seen"] for c in rec["curve"])


def test_run_job_target(tmp_path):
    (tmp_path / "a").mkdir()
    assert R.run_job(job(tmp_path / "a"), "abc", **TINY)["target"] == "ema"
    j = {**job(tmp_path), "target": "online", "key": "single__s2__lam0.1__s0__online"}
    j["path"] = tmp_path / f"{j['key']}.json"
    assert R.run_job(j, "abc", **TINY)["target"] == "online"


def test_atlas_gate_logged(tmp_path):
    rec = R.run_job(job(tmp_path, arm="atlas"), "abc", **TINY)
    assert all(0 <= c["gate_mean"] <= 1 for c in rec["curve"]) and isinstance(rec["t_gate_closed"], int)


def test_run_job_scale_zero(tmp_path):
    assert R.run_job(job(tmp_path, scale=0.0), "abc", **TINY)["lam"] == 0.0


def test_lgiven_poison(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    clean = R.run_job(job(tmp_path / "a"), "abc", **TINY)["curve"]
    real = beds.make_bed

    def poisoned(*a, **k):
        b = real(*a, **k)
        return {**b, "state": np.full_like(b["state"], np.nan), "geo": None}

    monkeypatch.setattr(R, "make_bed", poisoned)
    (tmp_path / "b").mkdir()
    curve = R.run_job(job(tmp_path / "b"), "abc", **TINY)["curve"]
    assert curve == clean and not any(math.isnan(c["g_erank"]) or math.isnan(c["loss"]) for c in curve)


def test_procs_capped(tmp_path):
    with pytest.raises(SystemExit):
        R.main(["--procs", "4", "--out", str(tmp_path)])


def test_dry_run_writes_nothing(tmp_path, capsys):
    R.main(["--arms", "single", "--beds", "s2", "--scales", "1", "--seeds", "0", "--dry-run", "--out", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()
    out = capsys.readouterr().out
    assert "single__s2__lam1__s0" in out and "sha" in out


def test_dry_run_targets_online(tmp_path, capsys):
    R.main(["--arms", "single", "--beds", "s2", "--scales", "0", "--seeds", "0", "--targets", "online", "--dry-run",
            "--out", str(tmp_path / "o")])
    assert "single__s2__lam0__s0__online" in capsys.readouterr().out


def test_tuning_seed_never_evaluated(tmp_path):
    """Seed 3 calibrated the T4a budget (PREREG_DRAFT_t4a.md): it may never enter an eval grid."""
    assert 3 in R.TUNING_SEEDS
    with pytest.raises(ValueError):
        R.plan(["single"], ["s2"], [0.0], [0, 3], out=tmp_path)
    assert not set(R.EVAL_SEEDS) & set(R.TUNING_SEEDS) and len(R.EVAL_SEEDS) == 5

"""T3/T7 runner plan: job list, tuning-seed isolation, resumability, L-REPRO refusal, GPU cap, SoftAtlas parity."""
import json
import re
import subprocess

import pytest
import torch

import run_t3 as R
from atlas.observer_atlas import ObserverAtlas


KS_FAST = (16, 256)    # one flat-key and one product-key K: PEER matching costs ~19 s per K, so the plan tests skip 64/1024


@pytest.fixture(scope="module")
def jobs_r2():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(R, "KS", KS_FAST)
        return R.jobs(beds=("r2",))


def test_spec_constants():
    assert R.KS == (16, 64, 256, 1024) and R.M_ACTIVE == 2 and R.PK_FROM == 256 and R.BEDS_T3 == ("so3", "r2")
    assert R.WIDTHS == (64, 128, 256, 512, 1024, 2048, 4096)
    assert R.SOFT_KS == (16, 64, 256) and R.SOFT_CHUNK == 16384


def test_job_list(jobs_r2):
    tags = [j["tag"] for j in jobs_r2]
    assert len(tags) == len(set(tags))
    atlas = {j["K"]: j for j in jobs_r2 if j["arm"] == "atlas"}
    assert set(atlas) == set(KS_FAST)
    assert all(j["kw"]["product_keys"] == (j["K"] >= 256) and j["kw"]["m"] == 2 for j in atlas.values())
    assert {j["K"] for j in jobs_r2 if j["arm"] == "atlas_soft"} == set(KS_FAST) & set(R.SOFT_KS)
    assert 1024 not in R.SOFT_KS                            # soft K=1024: ~9 GiB, ~56 s/step on the 8 GB 4060
    dense = [j for j in jobs_r2 if j["arm"] == "single"]
    assert {j["K"] for j in dense} == set(KS_FAST)
    for j in dense:                                         # equal active FLOPs to that K's lazy atlas
        assert j["match"] == "flop_matched"
        assert abs(j["budget"]["flops"] / atlas[j["K"]]["budget"]["flops"] - 1) <= 0.05, j["tag"]
    peer = [j for j in jobs_r2 if j["arm"] == "peer"]
    assert {j["K"] for j in peer} == set(KS_FAST)
    for j in peer:                                          # matched on the axis it claims, within 5%
        ax = {"both": ("flops", "params"), "flop_matched": ("flops",), "param_matched": ("params",)}[j["match"]]
        assert all(abs(j["budget"][a] / j["ref"][a] - 1) <= 0.05 for a in ax), j["tag"]
    w = [j["kw"]["hidden"] for j in jobs_r2 if j["arm"] == "densew" and j["phase"] == "eval" and j["seed"] == 0]
    assert w == list(R.WIDTHS)


def test_tuning_seed_isolated(jobs_r2):
    tune = [j for j in jobs_r2 if j["phase"] == "tune"]
    ev = [j for j in jobs_r2 if j["phase"] == "eval"]
    assert {j["seed"] for j in tune} == {R.TUNE_SEED} and {j["lr"] for j in tune} == set(R.LRS)
    assert {j["seed"] for j in ev} == set(R.EVAL_SEEDS) and R.TUNE_SEED not in R.EVAL_SEEDS
    assert all(j["lr"] is None for j in ev)
    assert len(tune) == len(R.LRS) * len(ev) // len(R.EVAL_SEEDS)


def test_resumable(jobs_r2, tmp_path):
    done = jobs_r2[0]["tag"]
    (tmp_path / f"{done}.json").write_text(json.dumps({"tag": done}))
    left = R.pending(jobs_r2, tmp_path)
    assert done not in {j["tag"] for j in left} and len(left) == len(jobs_r2) - 1


def test_refuses_dirty_code(monkeypatch, jobs_r2, tmp_path):
    monkeypatch.setattr(R, "dirty_code", lambda: ["x.py"])
    with pytest.raises(SystemExit, match="L-REPRO"):
        R.run_one(jobs_r2[0], tmp_path)
    with pytest.raises(SystemExit, match="L-REPRO"):
        R.schedule(jobs_r2, tmp_path)


def test_gpu_cap(monkeypatch):
    assert 1 <= R.MAX_GPU <= 3
    monkeypatch.setattr(R.subprocess, "check_output", lambda *a, **k: "11\n22\n33\n")
    assert R.gpu_others({22}) == 2

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(R.subprocess, "check_output", boom)
    assert R.gpu_others({22}) == 0
    monkeypatch.setattr(R.subprocess, "check_output",
                        lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "nvidia-smi")))
    assert R.gpu_others(set()) == 0


def test_can_launch_gate():
    assert R.can_launch(0, 0, 8.0) and R.can_launch(2, 0, 3.5)
    assert not R.can_launch(0, 0, 3.0) and not R.can_launch(0, 0, 1.0)      # low RAM blocks
    assert not R.can_launch(2, 1, 8.0) and not R.can_launch(3, 0, 8.0)      # GPU cap
    assert R.can_launch(0, 0, float("inf"))


def test_free_ram_gb_is_positive_number():
    assert R.free_ram_gb() > 0


def test_schedule_gate_uses_free_ram(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "dirty_code", lambda: [])
    monkeypatch.setattr(R, "pending", lambda js, out: js)
    monkeypatch.setattr(R, "gpu_others", lambda pids: 0)
    monkeypatch.setattr(R, "free_ram_gb", lambda: 1.0)
    monkeypatch.setattr(R.time, "sleep", lambda s: (_ for _ in ()).throw(RuntimeError("blocked")))
    monkeypatch.setattr(R.subprocess, "Popen", lambda *a, **k: pytest.fail("launched with low RAM"))
    with pytest.raises(RuntimeError, match="blocked"):
        R.schedule([dict(tag="t", phase="tune")], tmp_path)


def _tiny(K=4, m=4):
    torch.manual_seed(0)
    return ObserverAtlas(D=12, A=2, K=K, m=m, d_c=2, D_g=6, d_h=16, d_key=8, hidden=8).double().eval()


def test_soft_atlas_parity():
    at = _tiny()
    soft = R.SoftAtlas(at).eval()
    g = torch.Generator().manual_seed(1)
    obs, act = torch.randn(3, 5, 12, generator=g).double(), torch.randn(3, 5, 2, generator=g).double()
    assert (soft.rollout(obs[:, 0], act[:, :4]) - at.rollout(obs[:, 0], act[:, :4])).abs().max() < 1e-10
    torch.manual_seed(7); a = soft(obs, act)
    torch.manual_seed(7); b = at(obs, act, mode="soft")
    assert torch.equal(a["g"], b["g"]) and torch.equal(a["loss"], b["loss"])
    assert soft.n_params() == at.n_params()
    assert soft.active_flops() == at.active_flops()            # m = K: nothing to add
    # m=2 of K=4, soft runs all 4 charts: same cost as the m=K atlas
    assert R.SoftAtlas(_tiny(4, 2)).active_flops() == at.active_flops() > _tiny(4, 2).active_flops()


@pytest.fixture
def tiny_run(monkeypatch):
    for k, v in dict(dirty_code=lambda: [], STEPS=2, N_TRAIN=4, N_EVAL=8, T=16, WINDOW=8, BATCH=2).items():
        monkeypatch.setattr(R, k, v)


def test_run_one_end_to_end(tiny_run, jobs_r2, tmp_path):
    jobs = jobs_r2
    ev = next(j for j in jobs if j["arm"] == "atlas" and j["K"] == 16 and j["phase"] == "eval")
    soft = next(j for j in jobs if j["arm"] == "atlas_soft" and j["K"] == 16 and j["phase"] == "tune")
    dw = next(j for j in jobs if j["arm"] == "densew" and j["phase"] == "tune")
    for j, key in ((dict(ev, lr=1e-3), "probe_r2"), (soft, "pred_h1"), (dw, "pred_h1")):
        rec = R.run_one(j, tmp_path, device="cpu")
        saved = json.loads((tmp_path / f"{j['tag']}.json").read_text())
        assert key in saved and "npersist" in saved["pred_h1"]
        assert re.fullmatch(r"[0-9a-f]{40}", saved["git_sha"]) and saved["steps"] == 2
        assert rec["tag"] == j["tag"]
    assert {"pred_h10", "flops", "params"} <= set(json.loads((tmp_path / f"{ev['tag']}.json").read_text()))


def test_chunked_sigreg_equals_reference_value_and_grad():
    """Soft mode feeds [B*T*K, d_c] rows to per-chart SIGReg (34 GiB at K=1024). The chunked, checkpointed
    version must give the same value and gradient as atlas.observer_atlas.sigreg."""
    import atlas.observer_atlas as oa
    import sigreg as S
    from run_t3 import sigreg_chunked
    assert oa.sigreg is S.sigreg                               # SoftAtlas patches the name the atlas actually calls
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1000, 3, generator=g, dtype=torch.float64, requires_grad=True)
    grp = torch.randint(0, 7, (1000,), generator=g)
    grp[grp == 5] = 4                                          # an empty group and uneven counts
    vals = []
    for f in (lambda: S.sigreg(x, grp, 7, num_slices=16), lambda: sigreg_chunked(x, grp, 7, num_slices=16, chunk=128)):
        torch.manual_seed(3)
        v = f()
        vals.append((v.detach(), torch.autograd.grad(v, x)[0]))
    assert torch.allclose(vals[0][0], vals[1][0], atol=1e-12, rtol=0)
    assert torch.allclose(vals[0][1], vals[1][1], atol=1e-12, rtol=0)


def test_soft_atlas_uses_soft_chunk(monkeypatch):
    import atlas.observer_atlas as oa
    seen, real = [], R.sigreg_chunked
    monkeypatch.setattr(R, "sigreg_chunked", lambda *a, **k: (seen.append(k.get("chunk")), real(*a, **k))[1])
    ref = oa.sigreg
    g = torch.Generator().manual_seed(1)
    R.SoftAtlas(_tiny()).eval()(torch.randn(3, 5, 12, generator=g).double(), torch.randn(3, 5, 2, generator=g).double())
    assert seen and set(seen) == {R.SOFT_CHUNK}
    assert oa.sigreg is ref                                    # swap restored


def test_workers_pin_threads(monkeypatch):
    calls = []
    monkeypatch.setattr(R.torch, "set_num_threads", calls.append)
    monkeypatch.setattr(R, "run_one", lambda job: None)
    import sys
    monkeypatch.setattr(sys, "argv", ["run_t3.py", "--one", str(__file__)])
    monkeypatch.setattr(R.json, "loads", lambda s: {})
    R.main()
    assert calls == [4]

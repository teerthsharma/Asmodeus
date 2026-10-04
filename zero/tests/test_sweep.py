"""T2 sweep plan: job list shape, tuning-seed isolation, resumability, L-REPRO refusal, GPU cap."""
import json
import re

import pytest

import sweep_t2 as S


@pytest.fixture(scope="module")
def jobs_s2():
    return S.jobs(beds=("s2",))


def test_job_list(jobs_s2):
    tags = [j["tag"] for j in jobs_s2]
    assert len(tags) == len(set(tags))
    for cand in S.CANDIDATES:                                  # atlas, phase_sphere, hybrid: same decisive grid
        cells = {(j["kw"]["K"], j["kw"]["m"]) for j in jobs_s2 if j["arm"] == cand and j["phase"] == "eval"}
        assert cells == set(S.cells("s2")), cand
    assert set(S.cells("s2")) == {(1, 1), (2, 1), (2, 2), (4, 2), (4, 3), (16, 2), (64, 2)}
    assert all(j["kw"]["D_g"] <= 2 * 2 + 2 and j["kw"]["d_c"] == 2 for j in jobs_s2 if j["arm"] in S.CANDIDATES)
    base = [j for j in jobs_s2 if j["arm"] != "atlas"]
    base = [j for j in base if j["arm"] not in S.CANDIDATES]
    assert {j["arm"] for j in base} == {"single", "mop", "mop_lazy", "chart_ae", "lazy_chart_ae", "peer"}
    assert {j["kw"]["d"] for j in base if j["arm"] == "single"} == {2, 4, 6}
    assert all(j["match"] in ("both", "flop_matched", "param_matched") for j in base)
    for j in base:                                          # matched on the axis it claims, within 5%
        ax = {"both": ("flops", "params"), "flop_matched": ("flops",), "param_matched": ("params",)}[j["match"]]
        assert all(abs(j["budget"][a] / j["ref"][a] - 1) <= 0.05 for a in ax), j["tag"]


def test_tuning_seed_isolated(jobs_s2):
    tune = [j for j in jobs_s2 if j["phase"] == "tune"]
    ev = [j for j in jobs_s2 if j["phase"] == "eval"]
    assert {j["seed"] for j in tune} == {S.TUNE_SEED} and {j["lr"] for j in tune} == set(S.LRS)
    assert {j["seed"] for j in ev} == set(S.EVAL_SEEDS) and S.TUNE_SEED not in S.EVAL_SEEDS
    assert all(j["lr"] is None for j in ev)                 # lr is read from the tuning records
    # lr tuned ONCE per arm family, on s2 only, at the family's reference config; reused on every bed
    assert {j["bed"] for j in tune} == {"s2"}
    assert sorted(j["family"] for j in tune) == sorted(f for f in S.FAMILIES for _ in S.LRS)
    assert all(S.tune_tag(j) in {t["tag"] for t in tune} for j in ev)


def test_cells_per_bed():
    # atlas(M), general-chart count 2, good-cover ct(M) (with m = n+1 for a true nerve), 16, 64, K=1 control;
    # m = 1 only at K = atlas(M) and K = 1 (tear test)
    assert set(S.cells("t2")) == {(1, 1), (2, 2), (3, 1), (3, 2), (7, 2), (7, 3), (16, 2), (64, 2)}
    assert set(S.cells("klein")) == {(1, 1), (2, 2), (3, 1), (3, 2), (8, 2), (8, 3), (16, 2), (64, 2)}
    assert set(S.cells("so3")) == {(1, 1), (2, 2), (4, 1), (4, 2), (11, 2), (11, 4), (16, 2), (64, 2)}
    assert set(S.cells("r2")) == {(1, 1), (2, 2), (4, 2)}


def test_single_gpu_tiers_and_order():
    """Seal: ONE GPU job box-wide. Decisive cells (tier 1) run first, seed-major, so the T2 verdict lands
    earliest; tier 1 alone must fit 24 h at 1 process (timing table measured on the 4060)."""
    assert S.MAX_GPU == 1
    js = S.jobs()
    tiers = [j["tier"] for j in js]
    assert tiers == sorted(tiers) and set(tiers) == {1, 2}
    t1 = [j for j in js if j["tier"] == 1]
    assert [j["phase"] for j in t1] == sorted((j["phase"] for j in t1), key=lambda p: p != "tune")
    seeds = [j["seed"] for j in t1 if j["phase"] == "eval"]
    assert seeds == sorted(seeds)                                # seed 0 of every decisive cell first
    assert S.wall_hours(t1) <= 24.0, S.wall_hours(t1)
    for bed in ("s2", "t2", "klein", "so3"):
        n, atl = S.BEDS[bed]["n"], S.BEDS[bed]["atlas"]
        cand = {(j["arm"], j["kw"]["K"], j["kw"]["m"]) for j in t1 if j["bed"] == bed and j["arm"] in S.CANDIDATES}
        assert cand == {(c, K, m) for c in S.CANDIDATES for K, m in ((atl, 2), (S.CT[bed], n + 1))}
        base = {(j["arm"], j["match"]) for j in t1 if j["bed"] == bed and j["arm"] not in S.CANDIDATES}
        assert {a for a, _ in base} == {"single", "mop", "chart_ae", "peer"}       # kill-clause + best-owned
        assert all(mt in ("both", "flop_matched") for _, mt in base)              # bar axis = equal active FLOPs
        assert all(j["kw"]["d"] == 2 * n + 2 for j in t1 if j["bed"] == bed and j["arm"] == "single")


def test_resumable(jobs_s2, tmp_path):
    done = jobs_s2[0]["tag"]
    (tmp_path / f"{done}.json").write_text(json.dumps({"tag": done}))
    left = S.pending(jobs_s2, tmp_path)
    assert done not in {j["tag"] for j in left} and len(left) == len(jobs_s2) - 1


def test_refuses_dirty_code(monkeypatch, jobs_s2):
    monkeypatch.setattr(S, "dirty_code", lambda: ["zero/metrics.py"])
    with pytest.raises(SystemExit, match="L-REPRO"):
        S.run_one(jobs_s2[0], S.OUT)


def test_dirty_code_detection(tmp_path):
    import subprocess
    g = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1"); (tmp_path / "PREREG.md").write_text("p")
    g("add", "."); g("commit", "-qm", "c")
    assert S.dirty_code(tmp_path) == []
    (tmp_path / "runs").mkdir(); (tmp_path / "runs" / "r.json").write_text("{}")
    (tmp_path / "notes.md").write_text("n")
    assert S.dirty_code(tmp_path) == []                          # results and prose are not code
    (tmp_path / "a.py").write_text("x = 2")
    (tmp_path / "b.py").write_text("y = 1")
    assert sorted(S.dirty_code(tmp_path)) == ["a.py", "b.py"]    # modified tracked + untracked code


def test_record_provenance_and_gpu_cap():
    assert 1 <= S.MAX_GPU <= 3
    assert re.fullmatch(r"[0-9a-f]{40}", S.git_head())

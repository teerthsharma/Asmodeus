"""T4b job runner (run_t4b.py): plan/resume, git-sha stamp, dirty-tree refusal, <=3 procs, dry run writes nothing."""
import json
import subprocess

import pytest

import run_t4b as R


def git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")
    return tmp_path


def test_plan_grid_and_resume(tmp_path):
    jobs = R.plan(["memoriser", "mop"], ["pointmaze-medium-stitch-v0"], [0, 1], out=tmp_path)
    assert [j["key"] for j in jobs] == ["memoriser__pointmaze-medium-stitch-v0__s0", "memoriser__pointmaze-medium-stitch-v0__s1",
                                        "mop__pointmaze-medium-stitch-v0__s0", "mop__pointmaze-medium-stitch-v0__s1"]
    assert not any(j["done"] for j in jobs)
    (tmp_path / "mop__pointmaze-medium-stitch-v0__s1.json").write_text("{}")
    assert [j["done"] for j in R.plan(["memoriser", "mop"], ["pointmaze-medium-stitch-v0"], [0, 1], out=tmp_path)] == [False, False, False, True]


def test_unknown_arm_rejected(tmp_path):
    with pytest.raises(KeyError):
        R.plan(["nope"], ["pointmaze-medium-stitch-v0"], [0], out=tmp_path)


def test_clean_sha(repo):
    sha = R.clean_sha(repo)
    assert sha == git(repo, "rev-parse", "HEAD") and len(sha) == 40
    (repo / "f.txt").write_text("dirty")
    with pytest.raises(SystemExit):
        R.clean_sha(repo)


def test_procs_capped(tmp_path):
    with pytest.raises(SystemExit):
        R.main(["--procs", "4", "--dry-run", "--out", str(tmp_path)])


def test_dry_run_writes_nothing(tmp_path, capsys):
    R.main(["--arms", "memoriser", "--mazes", "pointmaze-medium-stitch-v0", "--seeds", "0", "--dry-run",
            "--out", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()
    out = capsys.readouterr().out
    assert "memoriser__pointmaze-medium-stitch-v0__s0" in out and "sha" in out


def test_run_job_record(tmp_path):
    pytest.importorskip("ogbench")
    rec = R.run_job(dict(arm="memoriser", maze="pointmaze-medium-stitch-v0", seed=0,
                         key="k", path=tmp_path / "k.json"), sha="abc", device="cpu", exec_eps=0)
    saved = json.loads((tmp_path / "k.json").read_text())
    assert saved == rec and rec["sha"] == "abc" and rec["holdout_seed"] == 0
    for r in ("all", "gated", "paper_all", "paper_gated"):
        assert set(rec[r]["sampled"]) == {"planAll", "realroute", "precision"} and set(rec[r]["official"]) == {"planAll", "realroute", "precision"}
    assert rec["n_queries"] == 300 and "execution" not in rec

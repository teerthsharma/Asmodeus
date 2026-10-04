"""Chained Kaggle sessions: plan generation, resume contract, sha refusal, never pushes."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ZERO = Path(__file__).resolve().parents[1]
ROOT = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ZERO, text=True).strip())
REL = ZERO.relative_to(ROOT).as_posix()
sys.path.insert(0, str(ZERO / "kaggle"))
import chain  # noqa: E402
import chain_kernel  # noqa: E402

FAKE = '''import argparse, json, os, sys
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt"); ap.add_argument("--time-budget-s", type=float); ap.add_argument("--amp"); ap.add_argument("--out")
ap.add_argument("--target", type=int, default=12)
a = ap.parse_args()
c = Path(a.ckpt)
step = json.loads(c.read_text())["step"] if c.exists() else 0
step += 5
c.parent.mkdir(parents=True, exist_ok=True)
tmp = c.with_suffix(".tmp"); tmp.write_text(json.dumps({"step": step})); os.replace(tmp, c)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
if step >= a.target:
    (out / "DONE").write_text(str(step))
'''
STAGES = [{"name": "ts", "runner_args": "--target 12", "data_slug": "U/ts"},
          {"name": "inst", "runner_args": "--target 24", "data_slug": "U/inst"}]


def git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, encoding="utf-8", errors="replace", check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "zero").mkdir(parents=True)
    (r / "zero" / "fake_runner.py").write_text(FAKE)
    git(r, "init", "-q")
    git(r, "config", "user.email", "t@t")
    git(r, "config", "user.name", "t")
    git(r, "add", "zero/fake_runner.py")
    git(r, "commit", "-qm", "x")
    return r


def plan(repo, out, sha=None, budget=None, per=2, stages=STAGES):
    argv = ["--repo", str(repo), "--sha", sha or git(repo, "rev-parse", "HEAD"), "--runner", "fake_runner.py",
            "--stages", json.dumps(stages), "--sessions-per-stage", str(per), "--out", str(out),
            "--config", str(out / "nope.json")]
    if budget:
        argv += ["--budget-s", str(budget)]
    return chain.main(argv)


def test_plan_kernels_and_chaining(repo, tmp_path, capsys):
    out = tmp_path / "o"
    plan(repo, out)
    ids = [f"USER/zero-repro-s{s}-k{i}" for s in (0, 1) for i in (0, 1)]
    mds = [json.loads((out / f"s{s}_k{i}" / "kernel-metadata.json").read_text()) for s in (0, 1) for i in (0, 1)]
    assert [m["id"] for m in mds] == ids
    for n, m in enumerate(mds):
        assert m["is_private"] is True and m["enable_gpu"] is True and m["enable_internet"] is False
        assert m["code_file"] == "chain_kernel.py"
        assert m["dataset_sources"] == ["USER/zero-bundle", STAGES[n // 2]["data_slug"]]
        assert m.get("kernel_sources") == ([ids[n - 1]] if n else None)
    assert "kernel_sources" not in mds[0]
    assert len(json.loads((out / "chain_plan.json").read_text())) == 4
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("Seal only:")]
    assert len(lines) == 5 and "datasets create" in lines[0] and "kernels push" in lines[1]


def test_refusals(repo, tmp_path):
    with pytest.raises(SystemExit):
        plan(repo, tmp_path / "a", budget=12 * 3600)
    with pytest.raises(SystemExit):
        plan(repo, tmp_path / "b", sha="0" * 40)


def simulate(repo, tmp_path, only=None, tamper=None):
    """Run every session locally, feeding each the previous session's working dir as input."""
    out = tmp_path / "o"
    plan(repo, out, per=3)
    sessions = json.loads((out / "chain_plan.json").read_text())
    works, prev = [], None
    for n, s in enumerate(sessions):
        inp = tmp_path / f"in{n}"
        inp.mkdir()
        import shutil
        shutil.copytree(out / "dataset", inp / "zero-bundle")
        if prev:
            shutil.copytree(prev, inp / "prevkernel")
            if tamper and n == tamper:
                p = inp / "prevkernel" / "CHAIN.json"
                d = json.loads(p.read_text()); d["sha"] = "f" * 40; p.write_text(json.dumps(d))
        work = tmp_path / f"w{n}"
        rc = chain_kernel.main(cfg=s["config"], input_root=inp, work=work)
        works.append((rc, work))
        prev = work
    return works


def chain_json(w):
    return json.loads((w / "CHAIN.json").read_text())


def test_chain_progress_and_stage_switch(repo, tmp_path):
    w = simulate(repo, tmp_path)
    assert all(rc == 0 for rc, _ in w)
    steps = [json.loads((p / "ckpt" / "ckpt.pt").read_text())["step"] for _, p in w]
    cj = [chain_json(p) for _, p in w]
    assert steps == [5, 10, 15, 20, 25, 25]
    assert [c["done"] for c in cj] == [False, False, True, False, True, True]
    assert [c["stage"] for c in cj] == ["ts"] * 3 + ["inst"] * 3
    assert cj[0]["sha"] == git(repo, "rev-parse", "HEAD")


def test_session_after_done_same_stage_exits_immediately(repo, tmp_path):
    out = tmp_path / "o"
    plan(repo, out, per=4, stages=STAGES[:1])
    sessions = json.loads((out / "chain_plan.json").read_text())
    import shutil
    prev = None
    ran = []
    for n, s in enumerate(sessions):
        inp = tmp_path / f"in{n}"; inp.mkdir()
        shutil.copytree(out / "dataset", inp / "zero-bundle")
        if prev:
            shutil.copytree(prev, inp / "prev")
        work = tmp_path / f"w{n}"
        assert chain_kernel.main(cfg=s["config"], input_root=inp, work=work) == 0
        ran.append(json.loads((work / "ckpt" / "ckpt.pt").read_text())["step"])
        prev = work
    assert ran == [5, 10, 15, 15]  # 4th session did not call the runner
    assert chain_json(prev)["done"] is True and chain_json(prev)["returncode"] == 0


def test_mismatched_sha_refuses(repo, tmp_path):
    with pytest.raises(Exception):
        simulate(repo, tmp_path, tamper=1)


def test_never_pushes_and_ignored():
    for f in ("chain.py", "chain_kernel.py"):
        src = (ZERO / "kaggle" / f).read_text()
        for call in re.findall(r"subprocess\.\w+\(.*", src):
            assert not re.search(r"""["'](push|create)["']""", call), call
        assert not re.search(r"""\[\s*["']kaggle["']""", src)
    for p in ("kaggle/out/chain/x", "kaggle/private.json"):
        assert subprocess.run(["git", "check-ignore", "-q", "--no-index", f"{REL}/{p}"], cwd=ROOT).returncode == 0

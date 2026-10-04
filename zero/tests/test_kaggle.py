"""Kaggle reproduction bundle: build refuses uncommitted work, never pushes, private slugs gitignored."""
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
import build  # noqa: E402
import kernel  # noqa: E402


def git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, encoding="utf-8", errors="replace", check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "zero").mkdir(parents=True)
    (r / "zero" / "run_x.py").write_text(
        "import os, sys, pathlib\n"
        "p = pathlib.Path(os.environ['ZERO_OUT']); p.mkdir(parents=True, exist_ok=True)\n"
        "(p / 'x.txt').write_text(' '.join(sys.argv[1:]))\n")
    (r / "zero" / "untracked_note.txt").write_text("n")
    git(r, "init", "-q")
    git(r, "config", "user.email", "t@t")
    git(r, "config", "user.name", "t")
    git(r, "add", "zero/run_x.py")
    git(r, "commit", "-qm", "x")
    return r


def run_build(repo, out, sha, runner="run_x.py", extra=()):
    return build.main(["--repo", str(repo), "--sha", sha, "--runner", runner, "--runner-args", "--seed 1",
                       "--out", str(out), "--config", str(out / "nope.json"), *extra])


def test_refuses_missing_sha(repo, tmp_path):
    with pytest.raises(SystemExit):
        run_build(repo, tmp_path / "o", "0" * 40)


def test_refuses_untracked_runner(repo, tmp_path):
    with pytest.raises(SystemExit):
        run_build(repo, tmp_path / "o", git(repo, "rev-parse", "HEAD"), runner="untracked_note.txt")


def test_build_writes_bundle_and_private_metadata(repo, tmp_path):
    sha, out = git(repo, "rev-parse", "HEAD"), tmp_path / "o"
    run_build(repo, out, sha)
    assert (out / "dataset" / "zero.bundle").stat().st_size > 0
    man = json.loads((out / "dataset" / "MANIFEST.json").read_text())
    assert man["sha"] == sha and man["runner"] == "run_x.py" and man["runner_args"] == "--seed 1" and man["bundle_sha256"]
    assert json.loads((out / "dataset" / "dataset-metadata.json").read_text())["id"] == "USER/zero-bundle"
    km = json.loads((out / "kernel" / "kernel-metadata.json").read_text())
    assert km["is_private"] is True and km["enable_gpu"] is True and km["dataset_sources"] == ["USER/zero-bundle"]
    assert (out / "kernel" / "kernel.py").exists()
    assert git(repo, "branch", "--list") .count("\n") == 0  # temp ref cleaned up (only the default branch remains)


def test_kernel_reproduces_from_bundle(repo, tmp_path):
    sha, out, work = git(repo, "rev-parse", "HEAD"), tmp_path / "o", tmp_path / "work"
    run_build(repo, out, sha)
    assert kernel.main(bundle_dir=out / "dataset", work=work) == 0
    rep = json.loads((work / "REPRO.json").read_text())
    assert rep["sha"] == sha and rep["returncode"] == 0 and rep["runner"] == "run_x.py"
    assert (work / "out" / "x.txt").read_text() == "--seed 1"
    assert git(work / "repo", "rev-parse", "HEAD") == sha
    assert (work / "repro.log").exists()


def ignored(path):
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", f"{REL}/{path}"], cwd=ROOT)
    return r.returncode == 0


@pytest.mark.parametrize("path", ["kaggle/private.json", "kaggle/out/x"])
def test_private_paths_gitignored(path):
    assert ignored(path)


def test_build_never_pushes():
    src = (ZERO / "kaggle" / "build.py").read_text()
    for call in re.findall(r"subprocess\.\w+\(.*", src):
        assert not re.search(r"""["'](push|create)["']""", call), call
    assert not re.search(r"""\[\s*["']kaggle["']""", src)

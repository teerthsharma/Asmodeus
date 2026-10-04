"""zero/queue/pre-commit refuses AI attribution and PREREG edits after results exist (Wilson)."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "queue" / "pre-commit"
SH = shutil.which("sh") or shutil.which("bash")


def _git(repo, *a):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True)


def _hook(repo):
    env = dict(os.environ, ZERO_HOOK_NO_TESTS="1")
    return subprocess.run([SH, str(HOOK)], cwd=repo, capture_output=True, text=True, env=env).returncode


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "zero").mkdir()
    (tmp_path / "zero" / "PREREG.md").write_text("bars\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


@pytest.mark.skipif(SH is None, reason="no sh")
def test_clean_change_passes(repo):
    (repo / "zero" / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    assert _hook(repo) == 0


@pytest.mark.skipif(SH is None, reason="no sh")
def test_attribution_refused(repo):
    (repo / "zero" / "a.md").write_text("Co-Authored-" + "By: Cl" + "aude <noreply@" + "anthropic.com>\n")
    _git(repo, "add", "-A")
    assert _hook(repo) == 1


@pytest.mark.skipif(SH is None, reason="no sh")
def test_prereg_edit_refused_after_results(repo):
    (repo / "zero" / "runs").mkdir()
    (repo / "zero" / "runs" / "r.json").write_text("{}")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "results")
    (repo / "zero" / "PREREG.md").write_text("moved bars\n")
    _git(repo, "add", "-A")
    assert _hook(repo) == 1


@pytest.mark.skipif(SH is None, reason="no sh")
def test_prereg_edit_allowed_before_results(repo):
    (repo / "zero" / "PREREG.md").write_text("amended bars\n")
    _git(repo, "add", "-A")
    assert _hook(repo) == 0

"""Runs on Kaggle: one link of a chained pretraining run (contract in chain.py). Stdlib only; no side effects on import."""
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _git(*a, cwd=None):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, encoding="utf-8", errors="replace", check=True).stdout.strip()


def _gpu():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def main(cfg, input_root="/kaggle/input", work="/kaggle/working"):
    """cfg: {sha, runner, runner_args, stage, session, budget_s}. Returns the process exit code."""
    t0 = time.time()
    input_root, work = Path(input_root), Path(work)
    found = sorted(input_root.glob("zero.bundle")) + sorted(input_root.glob("*/zero.bundle"))
    assert found, f"no zero.bundle under {input_root}"
    bundle = found[0]
    man = json.loads((bundle.parent / "MANIFEST.json").read_text())
    sha = cfg["sha"]
    assert man["sha"] == sha, "bundle sha != session sha"
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == man["bundle_sha256"], "bundle sha256 mismatch"

    prev = next(((p.parent, json.loads(p.read_text())) for p in sorted(input_root.glob("*/CHAIN.json"))
                 if (p.parent / "ckpt").is_dir()), None)
    if prev and prev[1]["sha"] != sha:
        raise RuntimeError(f"refusing to resume: previous CHAIN.json sha {prev[1]['sha']} != {sha}")

    work.mkdir(parents=True, exist_ok=True)
    ckpt_dir = work / "ckpt"
    if prev:
        shutil.copytree(prev[0] / "ckpt", ckpt_dir, dirs_exist_ok=True)  # carry forward so later sessions find it
    base = {"sha": sha, "stage": cfg["stage"], "session": cfg["session"]}

    def record(rc, done):
        (work / "CHAIN.json").write_text(json.dumps(
            {**base, "returncode": rc, "done": done, "wall_s": round(time.time() - t0, 1), "gpu": _gpu()}, indent=2))
        return rc

    if prev and prev[1]["done"] and prev[1]["stage"] == cfg["stage"]:
        return record(0, True)

    repo, out = work / "repo", work / "out"
    shutil.rmtree(repo, ignore_errors=True)
    _git("clone", "-q", str(bundle), str(repo))
    _git("checkout", "-q", sha, cwd=repo)
    assert _git("rev-parse", "HEAD", cwd=repo) == sha, "checked-out HEAD != sha"
    ckpt_dir.mkdir(exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    budget = max(1.0, cfg["budget_s"] - (time.time() - t0))
    with open(work / "chain.log", "a") as log:
        rc = subprocess.run([sys.executable, cfg["runner"], *shlex.split(cfg["runner_args"]), "--ckpt",
                             str(ckpt_dir / "ckpt.pt"), "--time-budget-s", str(int(budget)), "--amp", "fp16",
                             "--out", str(out)], cwd=repo / "zero", stdout=log, stderr=subprocess.STDOUT).returncode
    return record(rc, (out / "DONE").exists())

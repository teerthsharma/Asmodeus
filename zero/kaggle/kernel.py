"""Runs on Kaggle: clone the bundled commit, verify the sha, run the runner, record REPRO.json."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

BUNDLE_DIR = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def _git(*a, cwd=None):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, encoding="utf-8", errors="replace", check=True).stdout.strip()


def _gpu():
    try:
        import torch
        return (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), torch.__version__
    except Exception:
        return None, None


def main(bundle_dir=BUNDLE_DIR, work=WORK):
    bundle_dir, work = Path(bundle_dir), Path(work)
    found = sorted(bundle_dir.glob("zero.bundle")) + sorted(bundle_dir.glob("*/zero.bundle"))
    assert found, f"no zero.bundle under {bundle_dir}"
    bundle = found[0]
    man = json.loads((bundle.parent / "MANIFEST.json").read_text())
    sha, runner, args = man["sha"], man["runner"], man["runner_args"]
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == man["bundle_sha256"], "bundle sha256 mismatch"

    work.mkdir(parents=True, exist_ok=True)
    repo, out = work / "repo", work / "out"
    _git("clone", "-q", str(bundle), str(repo))
    _git("checkout", "-q", sha, cwd=repo)
    assert _git("rev-parse", "HEAD", cwd=repo) == sha, "checked-out HEAD != manifest sha"
    if runner == "run_t4b.py":
        subprocess.run([sys.executable, "-m", "pip", "install", "ogbench==1.2.1"], check=True)

    import shlex
    out.mkdir(exist_ok=True)
    with open(work / "repro.log", "w") as log:
        rc = subprocess.run([sys.executable, runner, *shlex.split(args)], cwd=repo / "zero", stdout=log,
                            stderr=subprocess.STDOUT, env={**os.environ, "ZERO_OUT": str(out)}).returncode
    gpu, torch_v = _gpu()
    (work / "REPRO.json").write_text(json.dumps(
        {"sha": sha, "runner": runner, "args": args, "returncode": rc, "gpu": gpu,
         "python": sys.version.split()[0], "torch": torch_v}, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())

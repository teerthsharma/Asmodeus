"""Build a Kaggle reproduction bundle for a COMMITTED sha. Writes files and prints push commands; never pushes."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZERO = HERE.parent
DEFAULT_REPO = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ZERO, text=True).strip())
PLACEHOLDERS = {"user": "USER", "dataset_slug": "zero-bundle", "kernel_slug": "zero-repro"}


def git(repo, *a, check=True):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, encoding="utf-8", errors="replace", check=check)


def die(msg):
    sys.exit(f"refused: {msg}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", required=True)
    ap.add_argument("--runner")
    ap.add_argument("--runner-args", default="")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--config", default=str(HERE / "private.json"))
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    a = ap.parse_args(argv)
    repo, out = Path(a.repo), Path(a.out)

    if git(repo, "cat-file", "-e", f"{a.sha}^{{commit}}", check=False).returncode:
        die(f"{a.sha} is not a commit in {repo}")
    sha = git(repo, "rev-parse", a.sha).stdout.strip()
    committed = (git(repo, "merge-base", "--is-ancestor", sha, "HEAD", check=False).returncode == 0
                 or git(repo, "branch", "-r", "--contains", sha).stdout.strip())
    if not committed:
        die(f"{sha} is not reachable from HEAD or a remote branch")
    if not a.runner or git(repo, "cat-file", "-e", f"{sha}:zero/{a.runner}", check=False).returncode:
        die(f"runner zero/{a.runner} is not tracked at {sha}")

    cfg = dict(PLACEHOLDERS)
    cp = Path(a.config)
    if cp.exists():
        cfg.update(json.loads(cp.read_text()))
    else:
        print(f"no {cp}: using placeholders {PLACEHOLDERS}")
    user, ds, kn = cfg["user"], cfg["dataset_slug"], cfg["kernel_slug"]

    shutil.rmtree(out, ignore_errors=True)
    (out / "dataset").mkdir(parents=True)
    (out / "kernel").mkdir()
    bundle = out / "dataset" / "zero.bundle"
    ref = "refs/heads/zero-bundle-tmp"  # bundle needs a ref; clone needs a branch. Removed right after.
    git(repo, "update-ref", ref, sha)
    try:
        git(repo, "bundle", "create", str(bundle), ref)
    finally:
        git(repo, "update-ref", "-d", ref)

    (out / "dataset" / "dataset-metadata.json").write_text(json.dumps(
        {"title": ds, "id": f"{user}/{ds}", "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    (out / "dataset" / "MANIFEST.json").write_text(json.dumps(
        {"sha": sha, "runner": a.runner, "runner_args": a.runner_args,
         "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}, indent=2))
    (out / "kernel" / "kernel-metadata.json").write_text(json.dumps(
        {"id": f"{user}/{kn}", "title": kn, "code_file": "kernel.py", "language": "python", "kernel_type": "script",
         "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [f"{user}/{ds}"]}, indent=2))
    shutil.copy(HERE / "kernel.py", out / "kernel" / "kernel.py")

    print(f"Seal only: kaggle datasets create -p {out / 'dataset'}")
    print(f"Seal only: kaggle kernels push -p {out / 'kernel'}")
    return 0


if __name__ == "__main__":
    main()

"""Plan chained private Kaggle sessions that pretrain ONE model across several <=12 h sessions. Never pushes.

Runner contract (runner-agnostic):
    python <runner> <runner_args> --ckpt <path> --time-budget-s <S> --amp fp16 --out <dir>
  The runner resumes from --ckpt if present, saves it atomically before the budget ends, writes <out>/DONE when
  the stage's step target is reached, and exits 0.
Each kernel s{stage}_k{i} takes the previous kernel as kernel_sources, finds <input>/<prev>/ckpt + CHAIN.json,
refuses to resume if CHAIN.json's sha differs, exits at once if that stage is already DONE, else runs the runner.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from build import DEFAULT_REPO, HERE, PLACEHOLDERS, die, git

MAX_BUDGET_S = 12 * 3600 - 600


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", required=True)
    ap.add_argument("--runner")
    ap.add_argument("--stages", required=True)
    ap.add_argument("--sessions-per-stage", type=int, default=1)
    ap.add_argument("--budget-s", type=int, default=11 * 3600 - 1800)
    ap.add_argument("--config", default=str(HERE / "private.json"))
    ap.add_argument("--out", default=str(HERE / "out" / "chain"))
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    a = ap.parse_args(argv)
    repo, out = Path(a.repo), Path(a.out)
    if a.budget_s > MAX_BUDGET_S:
        die(f"budget {a.budget_s}s exceeds {MAX_BUDGET_S}s")
    stages = json.loads(a.stages)

    # same checks as build.py (inline there, so repeated here)
    if git(repo, "cat-file", "-e", f"{a.sha}^{{commit}}", check=False).returncode:
        die(f"{a.sha} is not a commit in {repo}")
    sha = git(repo, "rev-parse", a.sha).stdout.strip()
    if not (git(repo, "merge-base", "--is-ancestor", sha, "HEAD", check=False).returncode == 0
            or git(repo, "branch", "-r", "--contains", sha).stdout.strip()):
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
    bundle = out / "dataset" / "zero.bundle"
    ref = "refs/heads/zero-bundle-tmp"
    git(repo, "update-ref", ref, sha)
    try:
        git(repo, "bundle", "create", str(bundle), ref)
    finally:
        git(repo, "update-ref", "-d", ref)
    (out / "dataset" / "dataset-metadata.json").write_text(json.dumps(
        {"title": ds, "id": f"{user}/{ds}", "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    (out / "dataset" / "MANIFEST.json").write_text(json.dumps(
        {"sha": sha, "runner": a.runner, "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}, indent=2))

    plan, prev_id = [], None
    src = (HERE / "chain_kernel.py").read_text()
    for s, st in enumerate(stages):
        for i in range(a.sessions_per_stage):
            kid, d = f"{user}/{kn}-s{s}-k{i}", out / f"s{s}_k{i}"
            d.mkdir()
            conf = {"sha": sha, "runner": a.runner, "runner_args": st["runner_args"], "stage": st["name"],
                    "session": len(plan), "budget_s": a.budget_s}
            md = {"id": kid, "title": f"{kn}-s{s}-k{i}", "code_file": "chain_kernel.py", "language": "python",
                  "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False,
                  "dataset_sources": [f"{user}/{ds}", st["data_slug"]]}
            if prev_id:
                md["kernel_sources"] = [prev_id]
            (d / "kernel-metadata.json").write_text(json.dumps(md, indent=2))
            (d / "chain_kernel.py").write_text(f"{src}\n\nsys.exit(main({conf!r}))\n")  # config baked in: script kernels run one file
            plan.append({"kernel": kid, "dir": str(d), "config": conf})
            prev_id = kid
    (out / "chain_plan.json").write_text(json.dumps(plan, indent=2))

    print(f"Seal only: kaggle datasets create -p {out / 'dataset'}")
    for p in plan:  # push in order; each needs the previous to COMPLETE first
        print(f"Seal only: kaggle kernels push -p {p['dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

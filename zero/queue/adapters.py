"""Emit queue job JSONL from the existing runners, without editing them.

  python queue/adapters.py sweep_t2|run_t3 [--beds ...] [--specs DIR]      > jobs.jsonl
  python queue/adapters.py run_t4a [--arms --beds --scales --targets --seeds --steps --device --out]
  python queue/adapters.py run_t4b [--arms --mazes --seeds --steps --exec-eps --out]
  python queue/adapters.py --probe MODULE SPEC.json                        (used as a job's probe_cmd)

sweep_t2 / run_t3 run one job via `--one <job.json>`: a spec file per job is written to --specs (default queue/specs).
run_t4a / run_t4b have no spec-file mode, but single-valued --arms/--beds/--seeds flags select exactly one job.
est_s for t2/t3 is a relative ordering weight (the job's active FLOPs), not seconds; the type is the job's config id,
so a probe covers all tune/eval seeds of one config. eval jobs wait (`after`) for the tune jobs whose records they read.
"""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZERO = HERE.parent
VENV_PY = str(Path.home() / ".venvs" / "ogbench" / "Scripts" / "python.exe")
sys.path.insert(0, str(ZERO))


def dump(o):
    return o.item() if hasattr(o, "item") else list(o)


def t2_like(name, a):
    mod = importlib.import_module(name)
    specs = Path(a.specs)
    specs.mkdir(parents=True, exist_ok=True)
    have_probe = hasattr(importlib.import_module("run"), "probe")
    if not have_probe:
        print("adapters: run.probe missing -> no probe_cmd emitted (no VRAM admission)", file=sys.stderr)
    js = mod.jobs(tuple(a.beds or (mod.BEDS_T2 if name == "sweep_t2" else mod.BEDS_T3)))
    ids = {j["tag"] for j in js}

    def tune_deps(j):  # eval reads the tune records at every lr (tuned_lr); mirror the module's own tag rule
        tag = getattr(mod, "tune_tag", None) or (lambda j, lr: f"{j['cfg']}__tune_s{mod.TUNE_SEED}_lr{lr:g}")
        return [f"{name}:{tag(j, lr)}" for lr in mod.LRS if tag(j, lr) in ids]

    for j in js:
        spec = specs / f"{name}__{j['tag']}.json"
        spec.write_text(json.dumps(j, default=dump))
        row = dict(id=f"{name}:{j['tag']}", cmd=[sys.executable, str(ZERO / f"{name}.py"), "--one", str(spec)],
                   out=str(mod.OUT / f"{j['tag']}.json"), gpu=True, est_s=float(j["budget"]["flops"]), cwd=str(ZERO),
                   type=f"{name}:{j['cfg']}", steps=mod.STEPS)
        if j["phase"] == "eval":
            row["after"] = tune_deps(j)
        if have_probe:
            row["probe_cmd"] = [sys.executable, str(HERE / "adapters.py"), "--probe", name, str(spec)]
        yield row


def t4a(a):
    import run_t4a as m
    out = Path(a.out or m.OUT).resolve()
    arms, beds, scales = a.arms or list(m.ARMS), a.beds or ["s2", "so3", "r2"], a.scales or [1.0, 0.1, 0.0]
    targets, seeds = a.targets or list(m.TARGETS), a.seeds or list(m.EVAL_SEEDS)
    for j in m.plan(arms, beds, scales, seeds, out=out, targets=targets):
        cmd = [sys.executable, str(ZERO / "run_t4a.py"), "--arms", j["arm"], "--beds", j["bed"], "--scales",
               f"{j['scale']:g}", "--targets", j["target"], "--seeds", str(j["seed"]), "--out", str(out),
               "--device", a.device] + (["--steps", str(a.steps)] if a.steps else [])
        yield dict(id=f"t4a:{j['key']}", cmd=cmd, out=str(j["path"]), gpu=a.device == "cuda", est_s=float(a.steps or 5000),
                   cwd=str(ZERO))


def t4b(a):
    import run_t4b as m
    from beds import ogbench_bed as ob
    out = Path(a.out or m.OUT).resolve()
    for j in m.plan(a.arms or list(m.ARMS), a.mazes or list(ob.NAMES), a.seeds or [0, 1, 2, 3, 4], out=out):
        cmd = [VENV_PY, str(ZERO / "run_t4b.py"), "--arms", j["arm"], "--mazes", j["maze"], "--seeds", str(j["seed"]),
               "--out", str(out)] + (["--steps", str(a.steps)] if a.steps else []) + (
                  ["--exec-eps", str(a.exec_eps)] if a.exec_eps else [])
        yield dict(id=f"t4b:{j['key']}", cmd=cmd, out=str(j["path"]), gpu=False, est_s=60.0, cwd=str(ZERO))


def probe(name, spec):
    import torch
    import run
    mod = importlib.import_module(name)
    job = json.loads(Path(spec).read_text())
    bed = mod.make_bed(job["bed"], mod.N_TRAIN, mod.T, "lift", job["seed"])
    r = run.probe(mod.build(job), bed, steps=20, device="cuda" if torch.cuda.is_available() else "cpu",
                  batch=mod.BATCH, window=mod.WINDOW)
    Path(os.environ["QUEUE_PROBE_OUT"]).write_text(json.dumps(r))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runner", nargs="?", choices=("sweep_t2", "run_t3", "run_t4a", "run_t4b"))
    ap.add_argument("--probe", nargs=2, metavar=("MODULE", "SPEC"))
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--beds", nargs="+")
    ap.add_argument("--mazes", nargs="+")
    ap.add_argument("--scales", nargs="+", type=float)
    ap.add_argument("--targets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--exec-eps", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    ap.add_argument("--specs", default=str(HERE / "specs"))
    a = ap.parse_args(argv)
    if a.probe:
        return probe(*a.probe)
    rows = {"sweep_t2": lambda: t2_like("sweep_t2", a), "run_t3": lambda: t2_like("run_t3", a),
            "run_t4a": lambda: t4a(a), "run_t4b": lambda: t4b(a)}[a.runner]()
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    sys.exit(main())

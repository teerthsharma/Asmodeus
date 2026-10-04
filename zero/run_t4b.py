"""T4b job runner (PREREG_DRAFT_t4b.md): arms x mazes x seeds on the graph protocol, optional closed-loop execution.

One job = holdout(seed) -> fit arm on train cell transitions -> decode edges (centred PRIMARY, paper SECONDARY; all heads and pi-gated) ->
planAll / realroute / precision on 300 sampled + 5 official queries [-> executor, exec_eps per task].
Writes runs/t4b/<arm>__<maze>__s<seed>.json atomically; existing files are skipped (resumable).
Refuses: a dirty zero/ tree (house-events.jsonl and the output dir excepted), an untracked PREREG.md, --procs > 3.
CPU-only (graph arms are tiny); --procs worker processes split the cores.
Run in the ogbench venv:  ~/.venvs/ogbench/Scripts/python.exe run_t4b.py --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from beds import ogbench_bed as ob

ZERO = Path(__file__).resolve().parent
OUT = ZERO / "runs" / "t4b"
CONTROLS = ("memoriser", "propose_all", "codebook_xy")
LEARNED = ("single", "mop", "atlas", "lazy_chart_ae", "codebook_latent")
ARMS = CONTROLS + LEARNED
MAX_PROCS = 3


def make_arm(kind, seed, steps=None):
    from arms import controls_t4b as C
    if kind in CONTROLS:
        return dict(memoriser=C.Memoriser, propose_all=C.ProposeAll,
                    codebook_xy=lambda: C.Codebook(K=8, seed=seed))[kind]()
    from arms import t4b_adapters as A
    return A.make(kind, seed=seed, **({} if steps is None else dict(steps=steps)))


def plan(arms, mazes, seeds, out=OUT):
    out = Path(out)
    for a in arms:
        if a not in ARMS:
            raise KeyError(a)
    return [dict(arm=a, maze=m, seed=s, key=f"{a}__{m}__s{s}", path=out / f"{a}__{m}__s{s}.json",
                 done=(out / f"{a}__{m}__s{s}.json").exists()) for a in arms for m in mazes for s in seeds]


def dirty(cwd=ZERO, ignore=()):
    spec = ["--", "."] + [f":(exclude){p}" for p in ignore]
    return subprocess.run(["git", "status", "--porcelain", *spec], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def clean_sha(cwd=ZERO, ignore=()):
    d = dirty(cwd, ignore)
    if d:
        raise SystemExit(f"refusing: dirty tree under {cwd}:\n{d}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def run_job(job, sha, device="cpu", exec_eps=0, steps=None):
    name, seed = job["maze"], job["seed"]
    bed = ob.make_bed(name)
    E = ob.data_edges(bed)
    tr, ho = ob.holdout_edges(E, 0.2, seed)
    ctx, nxt, _, _ = ob.cell_transitions(bed, exclude=frozenset(ho))
    arm = make_arm(job["arm"], seed, steps).fit(ctx, nxt)
    cells = sorted({c for e in ob.true_edges(name) for c in e})
    xy = ob.ij_to_xy(name, np.array(cells))
    heads = arm.predict(xy)[0]
    train_xy = np.concatenate([ctx, nxt])
    Q, off = ob.sample_queries(name, 300, seed), ob.official_queries(name)
    rec = dict(arm=job["arm"], maze=name, seed=seed, sha=sha, device=device, holdout_seed=seed, steps=steps,
               n_queries=len(Q), n_edges_data=len(E), n_edges_train=len(tr), n_edges_held=len(ho), K=int(heads.shape[1]))
    # PRIMARY: centred rule ('all', 'gated'); SECONDARY: v1's raw rule ('paper_all', 'paper_gated')
    for r, rule, gate in (("all", "centred", False), ("gated", "centred", True),
                          ("paper_all", "paper", False), ("paper_gated", "paper", True)):
        P = ob.decode_edges(arm, cells, xy, train_xy, rule=rule, gate=gate)
        rec[r] = dict(sampled=ob.graph_metrics(P, name, Q), official=ob.graph_metrics(P, name, off), n_edges=len(P))
        if r == "all":
            P_all = P
    if exec_eps:
        x = ob.evaluate(ob.executor(name, P_all), name, exec_eps, seed=seed)
        rec["execution"] = dict(success=x["success"], per_task={str(k): v for k, v in x["per_task"].items()}, n=x["n"])
    rec = json.loads(json.dumps(rec))
    path = Path(job["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, path)
    return rec


def _worker(args):
    import torch
    job, sha, exec_eps, steps, threads = args
    torch.set_num_threads(threads)
    return job["key"], run_job(job, sha, exec_eps=exec_eps, steps=steps)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--mazes", nargs="+", default=list(ob.NAMES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--exec-eps", type=int, default=0, help="episodes per task for closed-loop execution (PREREG: 50)")
    p.add_argument("--steps", type=int, default=None, help="override training steps (default: FULL config)")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.procs <= MAX_PROCS:
        raise SystemExit(f"--procs must be in [1, {MAX_PROCS}]")
    out = Path(a.out).resolve()
    jobs = plan(a.arms, a.mazes, a.seeds, out)
    ign = ["house-events.jsonl"] + ([os.path.relpath(out, ZERO)] if out.is_relative_to(ZERO) else [])
    todo = [j for j in jobs if not j["done"]]
    if a.dry_run:
        d = dirty(ZERO, ign)
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ZERO, capture_output=True, text=True).stdout.strip()
        print(f"sha {sha} {'DIRTY (a real run would refuse)' if d else 'clean'}; {len(todo)}/{len(jobs)} pending -> {out}")
        for j in jobs:
            print(("done " if j["done"] else "todo ") + j["key"])
        return
    sha = clean_sha(ZERO, ign)
    if subprocess.run(["git", "ls-files", "--error-unmatch", "PREREG.md"], cwd=ZERO, capture_output=True).returncode:
        raise SystemExit("refusing: PREREG.md is not committed (L-PREREG)")
    threads = max(1, (os.cpu_count() or 1) // a.procs)
    args = [(j, sha, a.exec_eps, a.steps, threads) for j in todo]
    if a.procs == 1:
        results = map(_worker, args)
    else:
        results = ProcessPoolExecutor(a.procs).map(_worker, args)
    for key, rec in results:
        print(key, json.dumps(rec["all"]["sampled"]), flush=True)


if __name__ == "__main__":
    sys.exit(main())

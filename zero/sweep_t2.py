"""T2 sweep: beds x candidates (atlas, phase_sphere, hybrid; same cells) + owned baselines (flop-/param-matched
via arms/match.py) x seeds. ONE GPU job box-wide (MAX_GPU = 1): tier 1 = decisive cells + tuning, then tier 2.

  python sweep_t2.py --dry-run [--beds s2 t2] [--no-smoke]   tier/job counts + wall hours (+ timed 1-job smoke)
  python sweep_t2.py [--beds ...]                            run (refused unless every zero/*.py is committed: L-REPRO)

Phase 1 'tune': tuning seed 3, lr in LRS, ONCE per family on s2 at the family's reference config, scored by
pred_h1 npersist on eval bed 1003. Phase 2 'eval': seeds 0,1,2 at the family's tuned lr; full evaluate() with
readouts frozen on bed seed 3.
Records: zero/runs/t2/<tag>.json (written atomically; an existing record means done -> resumable).
Do not launch before PREREG.md is committed.

Step budget (preregistered rule): each family's eval step budget = the tuning-seed plateau step (EMA loss improves
< 1% over the last 2000 steps), rounded up to 1000, capped at 16000. STEPS = 8000 is the planning value used by
wall_hours.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ZERO = Path(__file__).resolve().parent
sys.path.insert(0, str(ZERO))
from beds import make_bed, BEDS  # noqa: E402
import metrics as M  # noqa: E402
from run import train, tune_readout, evaluate  # noqa: E402

OUT = ZERO / "runs" / "t2"
BEDS_T2 = ("s2", "t2", "klein", "so3", "r2")
EVAL_SEEDS, TUNE_SEED, LRS = (0, 1, 2), 3, (3e-4, 1e-3, 3e-3)
STEPS, BATCH, WINDOW, N_TRAIN, N_EVAL, T = 8_000, 64, 32, 512, 128, 64
MAX_GPU = 1
CANDIDATES = ("atlas", "phase_sphere", "hybrid")
BASELINES = ("single", "mop", "mop_lazy", "chart_ae", "lazy_chart_ae", "peer")
FAMILIES = CANDIDATES + BASELINES
TIER1_BASELINES = ("single", "mop", "chart_ae", "peer")
TIER1_FAMILIES = CANDIDATES + TIER1_BASELINES
CT = {"s2": 4, "t2": 7, "klein": 8, "so3": 11, "r2": 1}      # good-cover chart counts (Wilson facts)
G = 2                                                        # general-chart count
# measured on the RTX 4060, batch 64x32, so3 dims; launch-bound, ~K-independent (ms per train step)
MS_PER_STEP = dict(atlas=46, hybrid=64, phase_sphere=56, single=16, mop=21, mop_lazy=22, peer=23, chart_ae=33,
                   lazy_chart_ae=41)
EVAL_S = {("tune", True): 15, ("tune", False): 15, ("eval", True): 120, ("eval", False): 75}  # (phase, candidate?)


# ---------------- provenance ----------------
def git(*a, root=ZERO):
    return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, check=True).stdout


def git_head(root=ZERO):
    return git("rev-parse", "HEAD", root=root).strip()


def dirty_code(root=ZERO):
    """Modified, staged or untracked .py files under root (results in runs/ excluded). Non-empty = refuse."""
    out = []
    for line in git("status", "--porcelain=v1", "--untracked-files=all", "--", ".", root=root).splitlines():
        p = line[3:].strip().strip('"').split(" -> ")[-1]
        if p.endswith(".py") and "runs/" not in p:
            out.append(p)
    return out


# ---------------- job list ----------------
def _cls(arm):
    from atlas.observer_atlas import ObserverAtlas
    from atlas.coherent_observer import CoherentAtlas
    from atlas.phase_sphere import PhaseSphere
    return dict(atlas=ObserverAtlas, hybrid=CoherentAtlas, phase_sphere=PhaseSphere)[arm]


def build(job):
    if job["arm"] in CANDIDATES:
        return _cls(job["arm"])(**job["kw"])
    from arms.match import SPECS
    cls, fixed, _ = SPECS[job["arm"]]
    return cls(**{**fixed, **job["kw"]})


@lru_cache(maxsize=None)
def cand_budget(arm, kw):
    with torch.device("meta"):
        a = _cls(arm)(**dict(kw))
    return dict(flops=int(a.active_flops()), params=int(a.n_params()))


def atlas_budget(D, A, K, m, d_c, D_g):
    return cand_budget("atlas", tuple(sorted(dict(D=D, A=A, K=K, m=m, d_c=d_c, D_g=D_g).items())))


@lru_cache(maxsize=None)
def ps_hidden(D, A, K, m, n, Dg):
    """PhaseSphere hidden width whose active_flops is within 5% of the atlas at the same (K, m)."""
    from arms.match import _bisect
    target = atlas_budget(D, A, K, m, n, Dg)["flops"]
    kw = lambda h: tuple(sorted(dict(D=D, A=A, K=K, m=m, N=Dg, d_c=n, D_g=Dg, hidden=h).items()))
    h = _bisect(lambda v: cand_budget("phase_sphere", kw(v))["flops"], target)
    assert abs(cand_budget("phase_sphere", kw(h))["flops"] / target - 1) <= 0.05, (K, m, h)
    return h


@lru_cache(maxsize=None)
def _match(arm, flops, params, base):
    from arms.match import match
    return match(arm, flops, params, **dict(base))


def cells(bed):
    """(K, m) grid: K=1 control; G general charts; atlas(M) at m=1,2; good cover CT at m=2 and n+1; 16; 64."""
    if bed == "r2":
        return [(1, 1), (2, 2), (4, 2)]
    n, a = BEDS[bed]["n"], BEDS[bed]["atlas"]
    raw = [(1, 1), (G, 2), (a, 1), (a, 2), (CT[bed], 2), (CT[bed], n + 1), (16, 2), (64, 2)]
    return [c for c in dict.fromkeys(raw) if c[1] <= c[0]]


@lru_cache(maxsize=None)
def configs(bed):
    n, atlas = BEDS[bed]["n"], BEDS[bed]["atlas"]
    D, A = 256, make_bed(bed, 1, 2, "lift", 0)["act"].shape[-1]
    Dg = 2 * n + 2                                      # PREREG pin: D_g <= 2n+2, single JEPA at d = D_g swept
    rK = atlas                                          # reference atlas: K = atlas(M), m = min(2, K)
    ref = atlas_budget(D, A, rK, min(2, rK), n, Dg)
    out = []
    for K, m in cells(bed):
        kw = dict(D=D, A=A, K=K, m=m, d_c=n, D_g=Dg)
        at = atlas_budget(**kw)
        out.append(dict(arm="atlas", kw=kw, match=None, budget=at))
        out.append(dict(arm="hybrid", kw=kw, match=None, budget=cand_budget("hybrid", tuple(sorted(kw.items())))))
        pk = dict(D=D, A=A, K=K, m=m, N=Dg, d_c=n, D_g=Dg, hidden=ps_hidden(D, A, K, m, n, Dg))
        out.append(dict(arm="phase_sphere", kw=pk, match=None, ref=at,
                        budget=cand_budget("phase_sphere", tuple(sorted(pk.items())))))
    bases = [("single", dict(d=d)) for d in (n, 2 * n, Dg)] + [
        ("mop", dict(d=Dg, K=rK)), ("mop_lazy", dict(d=Dg, K=rK)), ("chart_ae", dict(K=rK, d_c=n, D_g=Dg)),
        ("lazy_chart_ae", dict(K=rK, m=min(2, rK), d_c=n, D_g=Dg)), ("peer", dict(d=Dg))]
    for arm, b in bases:
        r = _match(arm, ref["flops"], ref["params"], tuple(sorted(dict(D=D, A=A, **b).items())))
        rows = [("both", r["both"])] if r["both"] else [("flop_matched", r["flop_matched"]),
                                                      ("param_matched", r["param_matched"])]
        for kind, x in rows:
            out.append(dict(arm=arm, kw=x["kw"], match=kind, budget=dict(flops=x["flops"], params=x["params"]), ref=ref))
    return out


def _cfg_id(bed, c):
    s = "-".join(f"{k}{v}" for k, v in sorted(c["kw"].items()) if k not in ("D", "A"))
    return f"{bed}__{c['arm']}{'-' + c['match'] if c['match'] else ''}__{s}"


def tune_tag(job, lr=1e-3):
    """Tag of the tuning record for job's family at lr (default: the 1e-3 record)."""
    return f"tune__{job['family']}__s{TUNE_SEED}_lr{lr:g}"


def _decisive(bed, c):
    """Tier-1 eval cell: candidates at (atlas(M), 2) and (CT, n+1); baselines on the bar axis (equal FLOPs)."""
    n = BEDS[bed]["n"]
    if c["arm"] in CANDIDATES:
        K, m = c["kw"]["K"], c["kw"]["m"]
        if bed == "r2":
            return (K, m) == (1, 1)
        return (K, m) in ((BEDS[bed]["atlas"], 2), (CT[bed], n + 1))
    if c["match"] not in ("both", "flop_matched"):
        return False
    if bed == "r2":
        return c["arm"] == "single" and c["kw"]["d"] == 2 * n + 2
    return c["arm"] in TIER1_BASELINES and (c["arm"] != "single" or c["kw"]["d"] == 2 * n + 2)


def tune_cfgs():
    """family -> its s2 reference config: candidates at K=atlas(s2), m=2; baselines' both/flop_matched row."""
    n, a = BEDS["s2"]["n"], BEDS["s2"]["atlas"]
    ok = lambda f, c: ((c["kw"]["K"], c["kw"]["m"]) == (a, 2) if f in CANDIDATES else
                       c["match"] in ("both", "flop_matched") and (f != "single" or c["kw"]["d"] == 2 * n + 2))
    return {f: next(c for c in configs("s2") if c["arm"] == f and ok(f, c)) for f in FAMILIES}


def jobs(beds=BEDS_T2):
    tune, ev = [], []
    for f, c in tune_cfgs().items():
        base = dict(bed="s2", cfg=_cfg_id("s2", c), family=f, **c)
        tune += [dict(base, tag=tune_tag(base, lr), phase="tune", seed=TUNE_SEED, lr=lr,
                      tier=1 if f in TIER1_FAMILIES else 2) for lr in LRS]
    for bed in beds:
        for c in configs(bed):
            cid = _cfg_id(bed, c)
            base = dict(bed=bed, cfg=cid, family=c["arm"], tier=1 if _decisive(bed, c) else 2, **c)
            ev += [dict(base, tag=f"{cid}__eval_s{s}", phase="eval", seed=s, lr=None) for s in EVAL_SEEDS]
    out = []
    for t in (1, 2):                                    # tune first, then evals seed-major (stable: bed, then config)
        out += [j for j in tune if j["tier"] == t]
        out += sorted((j for j in ev if j["tier"] == t), key=lambda j: j["seed"])
    return out


def wall_hours(js):
    """Single process: sum of STEPS x ms/step + eval seconds, over jobs."""
    return sum(STEPS * MS_PER_STEP[j["arm"]] / 1000 + EVAL_S[j["phase"], j["arm"] in CANDIDATES] for j in js) / 3600


def pending(js, out=OUT):
    return [j for j in js if not (Path(out) / f"{j['tag']}.json").exists()]


# ponytail: tuned_lr should also return the plateau step once tune records log loss curves (see module docstring)
def tuned_lr(job, out=OUT):
    scores = {}
    for lr in LRS:
        p = Path(out) / f"{tune_tag(job, lr)}.json"
        if not p.exists():
            raise SystemExit(f"tuning record missing: {p.name}")
        scores[lr] = json.loads(p.read_text())["pred_h1"]["npersist"]
    return min(scores, key=lambda k: scores[k])


# ---------------- execution ----------------
def _dump(path, rec):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(rec, f, default=lambda x: x.item() if hasattr(x, "item") else list(x))
    os.replace(tmp, path)


def run_one(job, out=OUT, device=None):
    bad = dirty_code()
    if bad:
        raise SystemExit(f"L-REPRO: refusing to score with uncommitted code: {bad[:5]}")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lr = job["lr"] or tuned_lr(job, out)
    t0 = time.time()
    torch.manual_seed(job["seed"])
    arm = build(job)
    train(arm, make_bed(job["bed"], N_TRAIN, T, "lift", job["seed"]), STEPS, job["seed"], BATCH, WINDOW, lr=lr,
          log=out / f"{job['tag']}.log.jsonl", device=device)
    ev = make_bed(job["bed"], N_EVAL, T, "lift", 1000 + job["seed"])
    if job["phase"] == "tune":
        rec = dict(pred_h1=M.pred_error(arm, ev, 1, device=device))
    else:
        ro = tune_readout(arm, make_bed(job["bed"], N_EVAL, T, "lift", TUNE_SEED), device=device)
        rec = evaluate(arm, ev, ro, device=device)
        rec.update(readout=ro, eval_bed_seed=1000 + job["seed"])
    rec.update(job=job, tag=job["tag"], seeds=[job["seed"]], lr=lr, steps=STEPS, git_sha=git_head(),
               wall_sec=round(time.time() - t0, 1), torch=torch.__version__, numpy=np.__version__)
    _dump(out / f"{job['tag']}.json", rec)
    return rec


def schedule(js, out=OUT):
    bad = dirty_code()
    if bad:
        raise SystemExit(f"L-REPRO: refusing to run with uncommitted code: {bad[:5]}")
    q = Path(out) / "queue"; q.mkdir(parents=True, exist_ok=True)
    todo, live = pending(js, out), []                   # job order is the schedule (tier 1 first, tune before eval)
    while todo or live:
        live = [p for p in live if p.poll() is None]
        while todo and len(live) < MAX_GPU:
            j = todo.pop(0)
            (q / f"{j['tag']}.json").write_text(json.dumps(j))
            live.append(subprocess.Popen([sys.executable, __file__, "--one", str(q / f"{j['tag']}.json")]))
        time.sleep(5)


def smoke(bed="so3", steps=50):
    """Time one worst-case job (largest atlas, K=64 m=2) for `steps` steps + full evaluate, in a temp dir."""
    c = [c for c in configs(bed) if c["arm"] == "atlas" and c["kw"]["K"] == 64 and c["kw"]["m"] == 2][0]
    job = dict(c, bed=bed, seed=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    arm = build(job)
    tb = make_bed(bed, N_TRAIN, T, "lift", 0)
    train(arm, tb, 5, 0, BATCH, WINDOW, device=device)                      # warm-up (cuda init)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    train(arm, tb, steps, 0, BATCH, WINDOW, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    sps = (time.time() - t0) / steps
    t0 = time.time()
    ro = tune_readout(arm, make_bed(bed, N_EVAL, T, "lift", TUNE_SEED), device=device)
    evaluate(arm, make_bed(bed, N_EVAL, T, "lift", 1000), ro, device=device)
    return sps, time.time() - t0, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--beds", nargs="+", default=list(BEDS_T2))
    ap.add_argument("--one")
    a = ap.parse_args()
    if a.one:
        run_one(json.loads(Path(a.one).read_text()))
        return
    js = jobs(tuple(a.beds))
    if not a.dry_run:
        return schedule(js)
    for j in js:
        print(f"T{j['tier']} {j['tag']:80s} F={j['budget']['flops']:>9} P={j['budget']['params']:>9}")
    print()
    for name, sel in (("tier 1", [j for j in js if j["tier"] == 1]), ("tier 2", [j for j in js if j["tier"] == 2]),
                      ("total", js)):
        print(f"{name}: {len(sel)} jobs ({sum(j['phase'] == 'tune' for j in sel)} tune), "
              f"{len(pending(sel))} pending, {wall_hours(sel):.1f} h at 1 process")
    print(f"code dirty (run would be refused): {bool(dirty_code())}")
    if not a.no_smoke:
        sps, ev, dev = smoke()
        print(f"smoke on {dev}: {sps * 1e3:.1f} ms/step (atlas K=64 m=2, so3, batch {BATCH}x{WINDOW}), eval {ev:.0f} s")


if __name__ == "__main__":
    main()

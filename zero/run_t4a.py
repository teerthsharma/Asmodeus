"""T4a representation-collapse runner (PREREG_DRAFT_t4a.md): single JEPA vs OBSERVER-ATLAS (MoP-JEPA, lazy chart-AE extras)
at SIGReg scale in {1, 0.1, 0} of the shared LAM, beds s2, so3 (+ r2 neutral).
Logs effective rank of g and of each chart's z (plus scale-sensitive std, effective rank is scale-invariant) every log_every
steps on a fixed held-out eval bed; collapse time = first logged step with eff-rank < THR=1.5 (censored at the budget).
Writes runs/t4a/<arm>__<bed>__lam<scale>__s<seed>.json atomically; existing files are skipped (resumable).
Refuses: a dirty zero/ tree (house-events.jsonl and the output dir excepted), an untracked PREREG.md, --procs > 3.
Target stressor (--targets, default ema online): "online" = EMA momentum 0, i.e. the target is the current online encoder
under stop-grad (classic collapse setting). PRIMARY decisive cell = SIGReg scale 0 with target online; secondary = scale 0
with EMA. Key of an online job is the ema key + "__online". Applied identically to every arm.
Run with SYSTEM python (the T2 beds need system mujoco):  python run_t4a.py --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sigreg
from arms.lazy_chart_ae import LazyChartAEJEPA
from arms.mop_jepa import MoPJEPA
from arms.single_jepa import SingleJEPA
from atlas.observer_atlas import ObserverAtlas
from beds import BEDS, make_bed  # make_bed by module-level name: tests monkeypatch it
from metrics import effective_rank
from run_t4b import MAX_PROCS, clean_sha, dirty

ZERO = Path(__file__).resolve().parent
OUT = ZERO / "runs" / "t4a"
ARMS = ("single", "atlas", "mop", "lazy_chart_ae")
TUNING_SEEDS = (3,)            # calibration seed (single, s2, scale 0): never an eval seed
EVAL_SEEDS = (0, 1, 2, 4, 5)
TARGETS = ("ema", "online")
THR, SHRINK, GATE = 1.5, 0.01, 0.05  # erank collapse; 100x shrink vs step 0; atlas gate closed


def lam_for(scale, mode="ratio"):
    """Arms mix loss = (1-lam)*pred + lam*SIGReg, so naive 0.1x (lam*0.1) weakens SIGReg relative to prediction ~91x
    (0.9/0.1 = 9 -> 0.09/0.91), ratio mode scales the SIGReg:prediction weight ratio, exactly 10x at scale 0.1."""
    base = sigreg.LAM
    if mode == "naive":
        return scale * base
    r = scale * base / (1 - base)
    return r / (1 + r)


def build_arm(kind, meta, D, A, lam, tiny=False, target="ema"):
    # widths are PREREG choices; L-BR matching via arms/match.py is a PREREG decision
    n = meta["n"]
    Dg = 2 * n + 2
    t = tiny
    mom = dict(tau=0.0) if target == "online" else {}  # momentum 0: target = online under stop-grad
    if kind == "single":
        return SingleJEPA(D, A, d=Dg, lam=lam, **mom, **(dict(hidden=8) if t else {}))
    if kind == "atlas":
        return ObserverAtlas(D, A, K=4 if t else 8, m=2, d_c=n, D_g=Dg, lam=lam, **({"ema": 0.0} if mom else {}),
                             **(dict(d_h=8, d_key=8, hidden=8) if t else {}))
    if kind == "mop":
        return MoPJEPA(D, A, d=Dg, K=3 if t else 8, lam=lam, **mom,
                       **(dict(hidden=8, head_hidden=8, router_hidden=8) if t else {}))
    if kind == "lazy_chart_ae":
        return LazyChartAEJEPA(D, A, K=4 if t else 8, m=2, d_c=n, D_g=Dg, lam=lam, **mom,
                               **(dict(hidden=8, chart_hidden=8) if t else {}))
    raise KeyError(kind)


def _group(z, idx=None):
    """z [N,m,d] or [N,d], idx matching leading dims (None = one chart, id 0) -> {chart id: rows [n_k, d]} (float64)."""
    z = z.double().cpu().numpy().reshape(-1, z.shape[-1])
    k = np.zeros(len(z), int) if idx is None else idx.cpu().numpy().reshape(-1)
    return {int(c): z[k == c] for c in np.unique(k)}


@torch.no_grad()
def latents(arm, obs, act):
    """-> dict(g [N*T, Dg], z {chart id: [n_k, d]}, gate float|None); eval mode, restored afterwards."""
    was = arm.training
    arm.eval()
    try:
        out = arm(obs, act)
        g = out["g"]
        zk = out.get("z")
        if zk is not None:
            zi = out.get("zidx")
            z = _group(zk, zi)
        elif isinstance(arm, ObserverAtlas):
            r = arm.observe(obs, act)
            z = _group(r["z"], r["idx"])
        elif isinstance(arm, LazyChartAEJEPA):
            zz, _, idx = arm._parts(obs, act)[:3]
            z = _group(arm._lazy_coords(arm.Ec, zz.flatten(0, 1), idx.flatten(0, 1)), idx)
        elif isinstance(arm, MoPJEPA):
            z = _group(F.normalize(arm.enc(obs), dim=-1))
        else:  # SingleJEPA: g = z
            z = _group(g)
        gate = out.get("gate")
        return dict(g=g.double().cpu().numpy().reshape(-1, g.shape[-1]), z=z,
                    gate=None if gate is None else float(gate.mean()))
    finally:
        arm.train(was)


def _std(x):
    return float(np.sqrt(x.var(0).mean()))


def summary(lat):
    """Per-log scalars. Charts with fewer than max(2d, 1% of rows) rows (unused/rare) never count as collapsed."""
    g, z = lat["g"], lat["z"]
    ge = effective_rank(g)
    ok = {k: v for k, v in z.items() if len(v) >= max(2 * v.shape[-1], 0.01 * len(g))}
    ce = {str(k): effective_rank(v) for k, v in ok.items()}
    return dict(g_erank=float(ge), g_std=_std(g), chart_erank=ce, n_charts_used=len(ok), n_charts_seen=len(z),
                chart_erank_min=float(min(ce.values())) if ce else float(ge),
                chart_std_min=min(_std(v) for v in ok.values()) if ok else _std(g), gate_mean=lat["gate"])


def steps_to_collapse(steps, values, thr=THR, budget=None):
    for s, v in zip(steps, values):
        if v < thr:
            return int(s), False
    return int(budget if budget is not None else steps[-1]), True


def ratio_ci(t_atlas, t_single, n_boot=10000, seed=0):
    """Paired geometric-mean ratio atlas/single over seeds with a 95% bootstrap CI over seeds."""
    # ponytail: times clipped to >= 1 so a step-0 collapse does not give log(0); fine while log_every >= 1
    logs = np.log(np.maximum(np.asarray(t_atlas, float), 1) / np.maximum(np.asarray(t_single, float), 1))
    i = np.random.default_rng(seed).integers(0, len(logs), (n_boot, len(logs)))
    m = logs[i].mean(1)
    return float(np.exp(logs.mean())), float(np.exp(np.quantile(m, 0.025))), float(np.exp(np.quantile(m, 0.975)))


def verdict(t_atlas, cens_atlas, t_single, cens_single):
    """Censored times enter as the budget: conservative (underestimates the slower arm's time)."""
    if all(cens_single):
        return "VOID"
    return "PASS" if ratio_ci(t_atlas, t_single)[1] > 1 else "KILL"


def plan(arms, beds, scales, seeds, out=OUT, targets=("ema",)):
    out = Path(out)
    for a in arms:
        if a not in ARMS:
            raise KeyError(a)
    for b in beds:
        if b not in BEDS:
            raise KeyError(b)
    for t in targets:
        if t not in TARGETS:
            raise KeyError(t)
    if set(seeds) & set(TUNING_SEEDS):
        raise ValueError(f"tuning seeds {TUNING_SEEDS} are never evaluated")
    jobs = []
    for a in arms:
        for b in beds:
            for t in targets:
                for s in scales:
                    for k in seeds:
                        key = f"{a}__{b}__lam{s:g}__s{k}" + ("__online" if t == "online" else "")
                        jobs.append(dict(arm=a, bed=b, scale=s, target=t, seed=k, key=key, path=out / f"{key}.json",
                                         done=(out / f"{key}.json").exists()))
    return jobs


def run_job(job, sha, steps, log_every, device="cpu", tiny=False, n_traj=256, T=64, lam_mode="ratio"):
    name, seed, scale, target = job["bed"], job["seed"], job["scale"], job.get("target", "ema")
    tr = make_bed(name, n_traj, T, obs="lift", seed=seed)
    ev = make_bed(name, 4 if tiny else 64, T, obs="lift", seed=10_000 + seed)
    meta = tr["meta"]  # L-GIVEN: only obs, act, meta["n"] (arm width) are read from a bed
    obs, act = torch.as_tensor(tr["obs"]), torch.as_tensor(tr["act"])
    eo, ea = torch.as_tensor(ev["obs"]).to(device), torch.as_tensor(ev["act"]).to(device)
    lam = lam_for(scale, lam_mode)
    torch.manual_seed(seed)
    arm = build_arm(job["arm"], meta, obs.shape[-1], act.shape[-1], lam, tiny, target).to(device).train()
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(arm.parameters(), lr=1e-3, weight_decay=1e-2)
    N, Tt, batch, window = *obs.shape[:2], 32, 32
    curve = []

    def log(step, loss):
        curve.append(dict(step=step, loss=float(loss), **summary(latents(arm, eo, ea))))

    with torch.no_grad():
        was = arm.training
        arm.eval()
        l0 = arm(eo[:batch, :window], ea[:batch, :window])["loss"].item()
        arm.train(was)
    log(0, l0)
    for step in range(1, steps + 1):  # same conventions as run.train
        i = rng.integers(0, N, batch)
        s = rng.integers(0, Tt - window + 1, batch)
        idx = s[:, None] + np.arange(window)
        loss = arm(obs[i[:, None], idx].to(device), act[i[:, None], idx].to(device))["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps:
            log(step, loss.item())
    st = [c["step"] for c in curve]
    gs = np.array([c["g_std"] for c in curve])
    tg, cg = steps_to_collapse(st, [c["g_erank"] for c in curve], budget=steps)
    tc, cc = steps_to_collapse(st, [c["chart_erank_min"] for c in curve], budget=steps)
    ts, cs = steps_to_collapse(st, gs / max(gs[0], 1e-12), SHRINK, budget=steps)
    gm = [c["gate_mean"] for c in curve]
    tgate, cgate = steps_to_collapse(st, gm, GATE, budget=steps) if gm[0] is not None else (steps, True)
    rec = dict(arm=job["arm"], bed=name, scale=scale, target=target, lam=lam, lam_mode=lam_mode, seed=seed, sha=sha, steps=steps,
               log_every=log_every, device=device, n_traj=n_traj, T=T, n_params=int(arm.n_params()), curve=curve,
               t_collapse_g=tg, censored_g=cg, t_collapse_chart=tc, censored_chart=cc,
               t_shrink_g=ts, censored_shrink_g=cs, t_gate_closed=tgate, censored_gate=cgate)
    rec = json.loads(json.dumps(rec))
    path = Path(job["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, path)
    return rec


def _worker(args):
    job, sha, kw, threads = args
    torch.set_num_threads(threads)
    return job["key"], run_job(job, sha, **kw)


def summarise(out=OUT):
    recs = {}
    for f in Path(out).glob("*.json"):
        r = json.loads(f.read_text())
        recs[(r["arm"], r["bed"], r["scale"], r.get("target", "ema"), r["seed"])] = r
    rows = []
    for bed, scale, tgt in sorted({(b, s, t) for _, b, s, t, _ in recs}):
        for arm in ("atlas", "mop", "lazy_chart_ae"):
            seeds = sorted(k for a, b, s, t, k in recs if (a, b, s, t) == (arm, bed, scale, tgt) and ("single", bed, scale, tgt, k) in recs)
            if not seeds:
                continue
            row = [bed, scale, tgt, arm]
            for t, c in (("t_collapse_g", "censored_g"), ("t_collapse_chart", "censored_chart")):
                ta, ca = [recs[(arm, bed, scale, tgt, k)][t] for k in seeds], [recs[(arm, bed, scale, tgt, k)][c] for k in seeds]
                ts, cs = [recs[("single", bed, scale, tgt, k)][t] for k in seeds], [recs[("single", bed, scale, tgt, k)][c] for k in seeds]
                row += [*ratio_ci(ta, ts), verdict(ta, ca, ts, cs)]
            rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--beds", nargs="+", default=["s2", "so3", "r2"])
    p.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.1, 0.0])
    p.add_argument("--targets", nargs="+", default=list(TARGETS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(EVAL_SEEDS))
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--lam-mode", choices=("ratio", "naive"), default="ratio")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summarise", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.procs <= MAX_PROCS:
        raise SystemExit(f"--procs must be in [1, {MAX_PROCS}]")
    out = Path(a.out).resolve()
    if a.summarise:
        print("bed scale target arm | g: ratio [lo, hi] verdict | chart: ratio [lo, hi] verdict")
        for b, s, tgt, arm, *x in summarise(out):
            print(f"{b} {s:g} {tgt} {arm} | g {x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}] {x[3]} | chart {x[4]:.2f} [{x[5]:.2f}, {x[6]:.2f}] {x[7]}")
        return
    jobs = plan(a.arms, a.beds, a.scales, a.seeds, out, a.targets)
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
    kw = dict(steps=a.steps, log_every=a.log_every, device=a.device, lam_mode=a.lam_mode)
    threads = max(1, (os.cpu_count() or 1) // a.procs)
    args = [(j, sha, kw, threads) for j in todo]
    results = map(_worker, args) if a.procs == 1 else ProcessPoolExecutor(a.procs).map(_worker, args)
    for key, rec in results:
        c = rec["curve"][-1]
        print(key, f"t_g={rec['t_collapse_g']}{'*' if rec['censored_g'] else ''} "
                   f"t_chart={rec['t_collapse_chart']}{'*' if rec['censored_chart'] else ''} g_erank={c['g_erank']:.2f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

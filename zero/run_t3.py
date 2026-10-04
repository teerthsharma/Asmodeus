"""T3 lazy-scaling runner + the T7 dense-width sweep (PREREG_DRAFT_t3_t7.md §2-3). Reuses sweep_t2's plumbing.

  python run_t3.py --dry-run [--beds so3 r2]   job list, budgets, wall-time estimate from a timed smoke (no files)
  python run_t3.py [--beds ...]                run (refused unless every zero/*.py is committed: L-REPRO)

Arms per bed: atlas (lazy top-2 of K) and atlas_soft (full softmax over all K) for K in KS; single (equal-FLOP
dense JEPA) and peer (PEER-JEPA, matched on the axes it can reach) per K; densew once per bed (dense JEPA
width sweep, T7). atlas_soft runs only for K in SOFT_KS: soft K=1024 needs ~9 GiB peak and ~56 s/step on the
8 GB 4060 (measured 2026-10-04), so the lazy-vs-soft 1% check runs at K <= 256 and K=1024 is "not run (resource)".
Records: zero/runs/t3/<tag>.json; an existing record means done -> resumable.
"""
import argparse
import functools
import json
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

ZERO = Path(__file__).resolve().parent
sys.path.insert(0, str(ZERO))
from beds import make_bed, BEDS  # noqa: E402
import metrics as M  # noqa: E402
from run import train, encode  # noqa: E402
from sweep_t2 import (dirty_code, git_head, _dump, pending, tuned_lr, _match, STEPS, BATCH,  # noqa: E402,F401
                      WINDOW, N_TRAIN, N_EVAL, T, EVAL_SEEDS, TUNE_SEED, LRS)


def atlas_cls():  # removed from sweep_t2; defined locally
    from atlas.observer_atlas import ObserverAtlas
    return ObserverAtlas

OUT = ZERO / "runs" / "t3"
BEDS_T3 = ("so3", "r2")
KS, M_ACTIVE, PK_FROM = (16, 64, 256, 1024), 2, 256
WIDTHS = (64, 128, 256, 512, 1024, 2048, 4096)
MAX_GPU = 3
SOFT_KS, SOFT_CHUNK = (16, 64, 256), 16384


# ---------------- soft reference ----------------
def sigreg_chunked(x, grp=None, n_groups=None, num_slices=256, t_max=3.0, n_points=17, chunk=65536):
    """sigreg.sigreg with the [N, S, P] cos/sin scatter-sums built per row-chunk under checkpoint (memory O(chunk)).
    Same RNG draw order and statistic; differs from the reference only in float summation order."""
    from torch.utils.checkpoint import checkpoint
    from sigreg import ANCHOR
    x = x.reshape(-1, x.shape[-1])
    if grp is None:
        grp, n_groups = torch.zeros(len(x), dtype=torch.long, device=x.device), 1
    grp = grp.reshape(-1)
    n_groups = n_groups or int(grp.max()) + 1
    cnt = torch.bincount(grp, minlength=n_groups)
    ok = cnt >= 2
    if not ok.any():
        return x.sum() * 0
    A = torch.randn(x.shape[-1], num_slices, device=x.device, dtype=x.dtype)
    A = A / A.norm(dim=0, keepdim=True)
    t = torch.linspace(0, t_max, n_points, device=x.device, dtype=x.dtype)
    dt = t_max / (n_points - 1)
    w = torch.full_like(t, 2 * dt)
    w[[0, -1]] = dt
    phi = (-0.5 * t * t).exp()

    def part(xc, gc):
        pc = xc @ A
        xt = pc[..., None] * t
        z = xc.new_zeros(n_groups, *xt.shape[1:])
        return (z.index_add(0, gc, xt.cos()), z.index_add(0, gc, xt.sin()),
                xc.new_zeros(n_groups, pc.shape[1]).index_add(0, gc, pc.square()))

    re = im = q = 0
    for s in range(0, len(x), chunk):
        a, b, d = checkpoint(part, x[s:s + chunk], grp[s:s + chunk], use_reentrant=False)
        re, im, q = re + a, im + b, q + d
    c = cnt[ok].to(x.dtype)[:, None, None]
    err = (re[ok] / c - phi).square() + (im[ok] / c).square()
    m2 = q[ok] / c[:, :, 0]
    return (err @ (w * phi)).mean()


class SoftAtlas(nn.Module):
    """ObserverAtlas trained and read with the full softmax over all K charts (the T3 soft reference).
    Soft mode feeds [B*T*K, d_c] rows to the per-chart SIGReg; the reference sigreg materialises
    [rows, 64, 17] (OOM from K=256 at batch 64x32), so forward swaps in sigreg_chunked (identical value/grad)."""

    def __init__(self, atlas):
        super().__init__()
        self.atlas = atlas

    def forward(self, obs, act):
        import atlas.observer_atlas as oa
        ref, oa.sigreg = oa.sigreg, functools.partial(sigreg_chunked, chunk=SOFT_CHUNK)  # ponytail: module-global swap; drop once Chase chunks sigreg
        try:
            return self.atlas(obs, act, mode="soft")
        finally:
            oa.sigreg = ref

    def rollout(self, obs0, act):
        at, out = self.atlas, []
        c = at.backbone(obs0[:, None])[0][:, 0]
        w = at.full_logits(c).softmax(-1); gate = torch.sigmoid(at.gate(c))
        z = at.E.dense(obs0)                                   # [N, K, d_c]
        for h in range(act.shape[1]):
            z = z + at.P.dense(torch.cat([z, act[:, h, None].expand(-1, at.K, -1)], -1))
            out.append(gate * torch.einsum("nk,nkd->nd", w, at.phi.dense(z)))
        return torch.stack(out, 1)

    def n_params(self):
        return self.atlas.n_params()

    def active_flops(self):
        """Lazy cost plus the K - m charts the lazy path skips: soft runs every chart."""
        at = self.atlas
        D, A, h, dc, Dg = at.D, at.A, at.hidden, at.d_c, at.D_g
        per_chart = D * h + h * dc + (dc + A) * h + h * dc + 2 * (dc * h + h * Dg) + 2 * Dg
        return at.active_flops() + (at.K - at.m) * per_chart


# ---------------- job list ----------------
def build(job):
    kw = job["kw"]
    if job["arm"] == "atlas":
        return atlas_cls()(**kw)
    if job["arm"] == "atlas_soft":
        return SoftAtlas(atlas_cls()(**kw))
    from arms.match import SPECS
    cls, fixed, _ = SPECS[job["arm"]] if job["arm"] != "densew" else SPECS["single"]
    return cls(**{**fixed, **kw})


@lru_cache(maxsize=None)
def _budget(arm, kw):
    with torch.device("meta"):
        m = build(dict(arm=arm, kw=dict(kw)))
    return dict(flops=int(m.active_flops()), params=int(m.n_params()))


def configs(bed):
    n = BEDS[bed]["n"]
    D, A, Dg = 256, make_bed(bed, 1, 2, "lift", 0)["act"].shape[-1], 2 * n + 2
    key = lambda d: tuple(sorted(d.items()))
    out = []
    for K in KS:
        kw = dict(D=D, A=A, K=K, m=M_ACTIVE, d_c=n, D_g=Dg, product_keys=K >= PK_FROM)
        ref = _budget("atlas", key(kw))
        for arm in ("atlas", "atlas_soft") if K in SOFT_KS else ("atlas",):
            out.append(dict(arm=arm, K=K, kw=kw, match=None, budget=_budget(arm, key(kw)), ref=ref))
        base = key(dict(D=D, A=A, d=Dg))
        r = _match("single", ref["flops"], ref["params"], base)
        out.append(dict(arm="single", K=K, kw=r["flop_matched"]["kw"], match="flop_matched", ref=ref,
                        budget=dict(flops=r["flop_matched"]["flops"], params=r["flop_matched"]["params"])))
        r = _match("peer", ref["flops"], ref["params"], base)
        rows = [("both", r["both"])] if r["both"] else [("flop_matched", r["flop_matched"]),
                                                      ("param_matched", r["param_matched"])]
        for kind, x in rows:
            out.append(dict(arm="peer", K=K, kw=x["kw"], match=kind, ref=ref,
                            budget=dict(flops=x["flops"], params=x["params"])))
    for H in WIDTHS:
        kw = dict(D=D, A=A, d=Dg, hidden=H)
        out.append(dict(arm="densew", K=None, kw=kw, match=None, ref=None, budget=_budget("densew", key(kw))))
    return out


def _cfg_id(bed, c):
    s = "-".join(f"{k}{v}" for k, v in sorted(c["kw"].items()) if k not in ("D", "A"))
    return f"{bed}__{c['arm']}{'-' + c['match'] if c['match'] else ''}__{'K%s__' % c['K'] if c['K'] else ''}{s}"


def jobs(beds=BEDS_T3):
    out = []
    for bed in beds:
        for c in configs(bed):
            cid = _cfg_id(bed, c)
            base = dict(bed=bed, cfg=cid, **c)
            out += [dict(base, tag=f"{cid}__tune_s{TUNE_SEED}_lr{lr:g}", phase="tune", seed=TUNE_SEED, lr=lr)
                    for lr in LRS]
            out += [dict(base, tag=f"{cid}__eval_s{s}", phase="eval", seed=s, lr=None) for s in EVAL_SEEDS]
    return out


# ---------------- execution ----------------
def evaluate_t3(arm, ev, device):
    """h=1/h=10 prediction error, state probe on g, and the arm's budget (no topology readouts: T3 does not use them)."""
    return dict(pred_h1=M.pred_error(arm, ev, 1, device=device), pred_h10=M.pred_error(arm, ev, 10, device=device),
                probe_r2=M.linear_probe(encode(arm, ev, device=device)["g"], ev["state"]),
                flops=int(arm.active_flops()), params=int(arm.n_params()))


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
    rec = dict(pred_h1=M.pred_error(arm, ev, 1, device=device)) if job["phase"] == "tune" \
        else evaluate_t3(arm, ev, device)
    rec.update(job=job, tag=job["tag"], seeds=[job["seed"]], lr=lr, steps=STEPS, git_sha=git_head(),
               wall_sec=round(time.time() - t0, 1), torch=torch.__version__, numpy=np.__version__)
    _dump(out / f"{job['tag']}.json", rec)
    return rec


def gpu_others(own_pids):
    """Compute processes on the GPU that are not ours (other lanes); 0 if nvidia-smi is missing or fails."""
    try:
        txt = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    except (OSError, subprocess.SubprocessError):
        return 0
    return len({int(p) for p in txt.split() if p.isdigit()} - set(own_pids))


def free_ram_gb():
    """Free physical memory in GiB (Windows CIM, else psutil, else /proc/meminfo); inf if all fail."""
    try:
        if sys.platform == "win32":
            txt = subprocess.check_output(["powershell", "-NoProfile", "-c",
                                           "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"], text=True)
            return float(txt.strip()) / 2**20
        try:
            import psutil
            return psutil.virtual_memory().available / 2**30
        except ImportError:
            return next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) / 2**20
    except (OSError, subprocess.SubprocessError, ValueError, StopIteration):
        return float("inf")


def can_launch(n_live, n_others, free_gb):
    return n_live + n_others < MAX_GPU and free_gb > 3


def schedule(js, out=OUT):
    bad = dirty_code()
    if bad:
        raise SystemExit(f"L-REPRO: refusing to run with uncommitted code: {bad[:5]}")
    q = Path(out) / "queue"; q.mkdir(parents=True, exist_ok=True)
    for phase in ("tune", "eval"):
        todo, live = [j for j in pending(js, out) if j["phase"] == phase], []
        while todo or live:
            live = [p for p in live if p.poll() is None]
            # ponytail: nvidia-smi is polled per launch decision, a lane that starts between polls can overshoot by one
            while todo and can_launch(len(live), gpu_others({p.pid for p in live}), free_ram_gb()):
                j = todo.pop(0)
                (q / f"{j['tag']}.json").write_text(json.dumps(j))
                live.append(subprocess.Popen([sys.executable, __file__, "--one", str(q / f"{j['tag']}.json")]))
            time.sleep(5)


def smoke(steps=20, bed="so3"):
    """-> (lazy s/step, soft s/step, eval s, device): so3, lazy at K=1024, soft at K=SOFT_KS[-1], 3 warmup steps, nothing written."""
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sync = lambda: torch.cuda.synchronize() if device == "cuda" else None
    cs = {c["K"]: c for c in configs(bed) if c["arm"] == "atlas"}
    c = cs[KS[-1]]
    tb, sps = make_bed(bed, N_TRAIN, T, "lift", 0), {}
    for arm in ("atlas", "atlas_soft"):
        torch.manual_seed(0)
        m = build(dict(cs[SOFT_KS[-1]] if arm == "atlas_soft" else c, arm=arm))   # soft K=1024 is not run
        try:
            train(m, tb, 3, 0, BATCH, WINDOW, device=device); sync()
            t0 = time.time()
            train(m, tb, steps, 0, BATCH, WINDOW, device=device); sync()
            sps[arm] = (time.time() - t0) / steps
        except torch.OutOfMemoryError:                          # soft mode holds [batch*window*K, d_c] in SIGReg
            sps[arm] = float("nan")
    m = build(dict(c, arm="atlas")).to(device)
    t0 = time.time()
    evaluate_t3(m, make_bed(bed, 16, T, "lift", 1000), device)
    return sps["atlas"], sps["atlas_soft"], time.time() - t0, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--beds", nargs="+", default=list(BEDS_T3))
    ap.add_argument("--one")
    ap.add_argument("--smoke-steps", type=int, default=20)
    a = ap.parse_args()
    if a.one:
        torch.set_num_threads(4)
        run_one(json.loads(Path(a.one).read_text()))
        return
    js = jobs(tuple(a.beds))
    if not a.dry_run:
        return schedule(js)
    for j in js:
        print(f"{j['tag']:90s} F={j['budget']['flops']:>10} P={j['budget']['params']:>10}")
    print()
    for arm in dict.fromkeys(j["arm"] for j in js):
        s = [j for j in js if j["arm"] == arm]
        print(f"{arm:12s} {len(s):4d} jobs  sum F={sum(j['budget']['flops'] for j in s):>13}  "
              f"sum P={sum(j['budget']['params'] for j in s):>13}")
    lazy, soft, ev, dev = smoke(a.smoke_steps)
    if soft != soft:
        print(f"SOFT K={SOFT_KS[-1]} OOM at batch {BATCH}x{WINDOW}: atlas_soft jobs cannot train as configured; excluded below")
    sec = sum(STEPS * (soft if j["arm"] == "atlas_soft" else lazy) + ev for j in js
              if not (soft != soft and j["arm"] == "atlas_soft"))  # lazy s/step = upper bound
    print(f"\n{len(js)} jobs, {len(pending(js))} pending; smoke on {dev}: lazy {lazy * 1e3:.1f} ms/step, "
          f"soft {soft * 1e3:.1f} ms/step (K={SOFT_KS[-1]}, so3, batch {BATCH}x{WINDOW}), eval {ev:.0f} s")
    print(f"upper-bound wall time at {MAX_GPU} concurrent: {sec / MAX_GPU / 3600:.1f} h "
          f"(concurrency assumed to scale linearly)")
    print(f"code dirty (run would be refused): {bool(dirty_code())}")


if __name__ == "__main__":
    main()

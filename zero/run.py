"""T2 harness: train(arm, bed, ...) with AdamW, evaluate(arm, bed) -> every T2 metric, budget_ratio(a, b).
Run outputs go to zero/runs/<tag>.jsonl ONLY after PREREG.md is committed."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import metrics as M
from common import n_params

RUNS = Path(__file__).resolve().parent / "runs"

# DRAFT calibration (PREREG_DRAFT_beds.md, calibrate.py seed 1000): eps = 0.08 diam, far = 0.30 diam.
CAL = {
    "s2": dict(eps=0.251, delta=0.5, far=0.942, fold_delta=0.05),
    "t2": dict(eps=0.355, delta=0.5, far=1.333, fold_delta=0.05),
    "klein": dict(eps=0.355, delta=0.5, far=1.333, fold_delta=0.05),
    "so3": dict(eps=0.251, delta=0.5, far=0.942, fold_delta=0.05),
    "r2": dict(eps=0.226, delta=0.5, far=0.849, fold_delta=0.05),
    # calibrate.py berry berry0 (seed 1000, diam pi): disc delta=0.5 oracle=pca=0; torn berry 0.067 @ eps=0.08 diam,
    # berry0 0.027 @ eps=0.10 diam (0.021 @ 0.08); fold far=0.30 diam, fdelta=0.05: oracle=pca=0, fold-latent 0.947 / 0.413
    "berry": dict(eps=0.251, delta=0.5, far=0.942, fold_delta=0.05),
    "berry0": dict(eps=0.314, delta=0.5, far=0.942, fold_delta=0.05),
}
TAUS = tuple(np.round(np.arange(0.05, 0.51, 0.05), 2))


def save_state(path, arm, opt, step, rng):
    """Atomic checkpoint: params + EMA buffers, AdamW state, step, torch/cuda/numpy RNG (the data cursor)."""
    tmp = str(path) + ".tmp"
    torch.save(dict(arm=arm.state_dict(), opt=opt.state_dict(), step=step, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    np_rng=rng.bit_generator.state), tmp)
    os.replace(tmp, path)


def load_state(path, arm, opt, rng):
    """Restore everything save_state wrote -> saved step."""
    dev = next(arm.parameters()).device
    st = torch.load(path, map_location=dev, weights_only=False)
    arm.load_state_dict(st["arm"])
    opt.load_state_dict(st["opt"])
    torch.set_rng_state(st["torch_rng"].cpu())
    if st["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([r.cpu() for r in st["cuda_rng"]])
    rng.bit_generator.state = st["np_rng"]
    return st["step"]


def train(arm, bed, steps, seed, batch=32, window=32, lr=1e-3, wd=1e-2, tag=None, log=None, log_every=50,
          device="cpu", ckpt=None, ckpt_every_s=600, progress=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    arm.to(device).train()
    opt = torch.optim.AdamW(arm.parameters(), lr=lr, weight_decay=wd)
    start = load_state(ckpt, arm, opt, rng) if ckpt and Path(ckpt).exists() else 0   # after seeding: restored RNG wins
    obs, act = torch.as_tensor(bed["obs"]), torch.as_tensor(bed["act"])
    N, T = obs.shape[:2]
    log = Path(log) if log else (RUNS / f"{tag}.jsonl" if tag else None)
    fhs = [open(f, "a") for f in (log, progress) if f]
    t0 = last = time.time()
    try:
        for step in range(start + 1, steps + 1):
            i = rng.integers(0, N, batch)
            s = rng.integers(0, T - window + 1, batch)
            idx = s[:, None] + np.arange(window)
            o = obs[i[:, None], idx].to(device)
            a = act[i[:, None], idx].to(device)
            loss = arm(o, a)["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if fhs and step % log_every == 0:
                row = json.dumps(dict(step=step, loss=loss.item(), sec=round(time.time() - t0, 3),
                                      flops=int(arm.active_flops()))) + "\n"
                for fh in fhs:
                    fh.write(row)
            if ckpt and (step == steps or time.time() - last >= ckpt_every_s):
                save_state(ckpt, arm, opt, step, rng)
                last = time.time()
    finally:
        for fh in fhs:
            fh.close()
    return arm


def probe(arm, bed, steps=20, device="cpu", batch=32, window=32):
    """Short real training run -> dict(vram_peak_bytes, step_s)."""
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    train(arm, bed, steps=steps, seed=0, batch=batch, window=min(window, bed["obs"].shape[1]), device=device)
    if cuda:
        torch.cuda.synchronize(device)
    return dict(vram_peak_bytes=int(torch.cuda.max_memory_allocated(device)) if cuda else 0,
                step_s=(time.time() - t0) / steps)


def encode(arm, bed, batch=64, device="cpu"):
    """-> dict(g, w, w_soft, z, zidx) as numpy over the whole bed (optional keys None if absent)."""
    arm.to(device).eval()
    acc = {k: [] for k in ("g", "w", "w_soft", "z", "zidx")}
    with torch.no_grad():
        for b in range(0, bed["obs"].shape[0], batch):
            out = arm(torch.as_tensor(bed["obs"][b:b + batch], device=device),
                      torch.as_tensor(bed["act"][b:b + batch], device=device))
            for k in acc:
                if out.get(k) is not None:
                    acc[k].append(out[k].cpu().numpy())
    return {k: (np.concatenate(v) if v else None) for k, v in acc.items()}


NERVES = (("tau", dict(flag=False), "w"), ("tau_flag", dict(flag=True), "w"), ("tau_soft", dict(flag=False), "w_soft"))


def _mid_of_longest_run(grid, seq):
    best, i = None, 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1] == seq[i]:
            j += 1
        if seq[i][0] > 0 and (best is None or j - i > best[0]):
            best = (j - i, grid[(i + j) // 2])
        i = j + 1
    return None if best is None else float(best[1])


def tune_readout(arm, tune_bed, device="cpu"):
    """Freeze every topology-readout knob on the TUNING seed (label-free: middle of the most persistent
    run). evaluate() then reads the eval seeds at these frozen values only."""
    e = encode(arm, tune_bed, device=device)
    n = tune_bed["meta"]["n"]
    fd = min(e["g"].shape[-1], 2 * n + 2)
    out = dict(tune_seed=tune_bed["meta"]["seed"], mapper=M.tune_mapper(e["g"], filter_dim=fd, max_dim=n))
    for key, kw, src in NERVES:
        if e[src] is not None:
            out[key] = _mid_of_longest_run(TAUS, [M.nerve_betti_z2(e[src], t, max_dim=n, **kw) for t in TAUS])
    return out


def evaluate(arm, bed, readout, cal=None, device="cpu"):
    meta, st, geo = bed["meta"], bed["state"], bed["geo"]
    assert readout["tune_seed"] != meta["seed"], "readout tuned on this eval seed"
    c = cal or CAL[meta["name"]]
    e = encode(arm, bed, device=device)
    g = e["g"]
    n, ns = meta["n"], int(np.prod(g.shape[:-1]))
    fd = min(g.shape[-1], 2 * n + 2)
    cov = readout["mapper"]
    out = dict(
        seeds=[meta["seed"]],
        pred_h1=M.pred_error(arm, bed, 1, device=device),
        pred_h10=M.pred_error(arm, bed, 10, device=device),
        probe_r2=M.linear_probe(g, st),
        disc=M.discontinuity_rate(g, st, geo, c["eps"], c["delta"]),
        fold=M.fold_rate(g, st, geo, c["far"], c["fold_delta"]),
        erank_g=M.effective_rank(g),
        erank_chart=None if e["z"] is None else {str(k): v for k, v in M.chart_effective_rank(e["z"], e["zidx"]).items()},
        nerve_n_samples=ns,
        mapper=dict(n_samples=ns, cover=cov, tune_seed=readout["tune_seed"], filter_dim=fd,
                    betti_z2=cov and M.mapper_betti(g, fd, n, **cov),
                    betti_q=cov and M.mapper_betti(g, fd, n, field="q", **cov)),
        flops=int(arm.active_flops()), params=int(n_params(arm)),
    )
    for key, kw, src in NERVES:
        if e[src] is not None:
            t = readout.get(key)
            out["nerve_betti" + key[3:]] = None if t is None else dict(
                tau=t, betti_z2=M.nerve_betti_z2(e[src], t, max_dim=n, **kw),
                betti_q=M.nerve_betti_z2(e[src], t, max_dim=n, field="q", **kw))
    return out


def budget_ratio(a, b):
    return dict(flops=a.active_flops() / b.active_flops(), params=n_params(a) / n_params(b))

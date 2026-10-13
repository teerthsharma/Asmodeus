"""Phase 2 bench: train one LM cell on pre-tokenized TinyStories and log loss against tokens, FLOPs and
wall-clock. The same recipe trains every arm (owned baselines and the era LM); the protocol is frozen in
p2/PREREG.md.

Arm contract (same as arms/lm_baselines.py): Class(vocab=V, **kw); forward(tokens [B, T+1] long) -> dict
with "loss" (the training objective) and, if "loss" carries more than next-token nll, "nll" [B, T] (per-token
-log p(tokens[:, t+1] | tokens[:, :t+1])), which evaluation then uses; n_params(). Optional active_flops(T) (forward MACs
per token, the arm's own count).

  python p2/bench.py --bins ../../Asmodeus/zero/data/p2 --arm gpt-m --seed 100 --out p2/results
"""
import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402,F401  (TF32 off; bf16 autocast is the frozen training precision)

V, SEQ, BATCH = 4096, 256, 64
VAL_WINDOWS = 4096                      # fixed eval sets: the first 4096 non-overlapping 257-token windows
                                        # of val.bin (curves, LR tuning) and of test.bin (the frozen bars)
ARMS = {                                # name -> (class, kwargs); sizes are frozen in PREREG.md
    "gpt-s": ("arms.lm_baselines:GPTLM", dict(d_model=128, n_layer=4, n_head=4, seq_len=SEQ)),
    "gpt-m": ("arms.lm_baselines:GPTLM", dict(d_model=192, n_layer=6, n_head=6, seq_len=SEQ)),
    "gpt-l": ("arms.lm_baselines:GPTLM", dict(d_model=320, n_layer=8, n_head=8, seq_len=SEQ)),
    "gru-m": ("arms.lm_baselines:GRULM", dict(d_emb=256, hidden=512, n_layer=2)),
}


def build(spec, kw=None):
    path, base = ARMS[spec] if spec in ARMS else (spec, {})
    mod, cls = path.split(":")
    return getattr(importlib.import_module(mod), cls)(vocab=V, **{**base, **(kw or {})})


def windows(arr, starts):
    return torch.from_numpy(np.stack([arr[s:s + SEQ + 1] for s in starts]).astype(np.int64))


def lr_at(step, steps, lr):
    warm = max(1, steps // 100)
    if step < warm:
        return lr * (step + 1) / warm
    p = (step - warm) / max(1, steps - warm)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))         # cosine from lr to 0.1 lr


@torch.no_grad()
def evaluate(model, val, dev, dtype, bs=128):
    n = VAL_WINDOWS
    assert len(val) >= n * (SEQ + 1), "eval split shorter than the fixed eval set"
    was = model.training
    model.eval()
    tot = 0.0
    for i in range(0, n, bs):
        x = windows(val, range(i * (SEQ + 1), min(n, i + bs) * (SEQ + 1), SEQ + 1)).to(dev)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            o = model(x)
            tot += (o["nll"].float().mean() if "nll" in o else o["loss"].float()).item() * x.shape[0]
    model.train(was)
    return tot / n


def fwd_flops_per_token(model, dev):
    """Forward FLOPs per target token from FlopCounterMode (matmul, conv, attention; elementwise not counted),
    and the arm's own count (2 * active_flops MACs) when it declares one. The bench uses the larger."""
    x = torch.randint(0, V, (2, SEQ + 1), device=dev)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    counted = fc.get_total_flops() / (2 * SEQ)
    declared = 2 * model.active_flops(SEQ) if hasattr(model, "active_flops") else None
    return counted, declared


def cell(arm, seed, tokens, lr, bins, kw=None, dtype=torch.bfloat16, evals=20, dev="cuda"):
    bins = Path(bins)
    train = np.memmap(bins / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(bins / "val.bin", dtype=np.uint16, mode="r")
    torch.manual_seed(seed)
    model = build(arm, kw).to(dev)
    counted, declared = fwd_flops_per_token(model, dev)
    fwd = max(counted, declared or 0)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=rest, weight_decay=0.0)],
                            lr=lr, betas=(0.9, 0.95), fused=True)
    rng = np.random.Generator(np.random.PCG64(seed))
    steps = tokens // (BATCH * SEQ)
    marks = {round(steps * (k + 1) / evals) for k in range(evals)}
    curve, wall = [], 0.0
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        x = windows(train, rng.integers(0, len(train) - SEQ - 1, BATCH)).pin_memory().to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            loss = model(x)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step + 1 in marks:
            torch.cuda.synchronize(dev)
            wall += time.perf_counter() - t0
            seen = (step + 1) * BATCH * SEQ
            vl = evaluate(model, val, dev, dtype)
            curve.append(dict(step=step + 1, tokens=seen, train_flops=3 * fwd * seen, wall_s=wall,
                              train_loss=loss.item(), val_loss=vl))
            print(json.dumps(dict(arm=arm, seed=seed, **curve[-1])), flush=True)
            if not math.isfinite(vl):
                break
            t0 = time.perf_counter()
    peak = torch.cuda.max_memory_allocated(dev)
    vl = curve[-1]["val_loss"]
    test = evaluate(model, np.memmap(bins / "test.bin", dtype=np.uint16, mode="r"), dev, dtype)
    return dict(arm=arm, seed=seed, lr=lr, tokens=curve[-1]["tokens"], steps=curve[-1]["step"],
                n_params=model.n_params(), fwd_flops_per_token=fwd, fwd_flops_counted=counted,
                fwd_flops_declared=declared, train_flops=curve[-1]["train_flops"], dtype=str(dtype),
                val_loss=vl, test_loss=test, test_ppl=math.exp(test) if test < 50 else None,
                train_tok_per_s=curve[-1]["tokens"] / wall, wall_s=wall, peak_mem_mb=peak / 2**20,
                device=torch.cuda.get_device_name(dev), torch=torch.__version__, curve=curve)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", required=True)
    ap.add_argument("--arm", required=True, help="a name in ARMS or pkg.module:Class")
    ap.add_argument("--kw", default="{}", help="JSON kwargs merged over the arm's defaults")
    ap.add_argument("--name", help="row name (default: --arm)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=200_000_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    name = a.name or a.arm
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = out / "results.jsonl"
    key = (name, a.seed, a.tokens, a.lr)
    rows = [json.loads(l) for l in res.read_text().splitlines()] if res.exists() else []
    if any((r["name"], r["seed"], r["tokens_budget"], r["lr"]) == key for r in rows):
        print(f"done already: {key}")
        return
    r = cell(a.arm, a.seed, a.tokens, a.lr, a.bins, json.loads(a.kw),
             torch.float32 if a.fp32 else torch.bfloat16)
    r.update(name=name, tokens_budget=a.tokens, kw=json.loads(a.kw))
    with open(res, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "curve"}))


if __name__ == "__main__":
    main()

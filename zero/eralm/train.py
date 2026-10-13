"""End-to-end TinyStories run (ROADMAP stage 7), resumable.

  A. train the TT network alone on random windows of train.bin (AdamW, warmup + cosine, bf16, optional compile);
  B. freeze it and grow the era bank online over a train stream (B persistent streams);
  C. evaluate on val and test streams: TT alone, EraLM with a frozen bank, EraLM prequential (the bank keeps learning
     from the stream it predicts, every token predicted before it is used), and the owned Jelinek-Mercer n-gram.

  python -m eralm.train --bins zero/data/p2 --out runs/era0 --tokens 2e8
"""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import eval_loss, train_step
from .config import EraLMConfig
from .data import RandomWindows, Streams, load_bins
from .model import EraLM
from .ngram import JMNGram


def log(out, rec):
    rec = {"time": time.time(), **rec}
    print(json.dumps(rec), flush=True)
    with open(out / "log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def lr_at(step, steps, lr, warmup, floor=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    q = (step - warmup) / max(1, steps - warmup)
    return lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(q, 1.0))))


def phase_a(a, cfg, arrs, out, dev):
    ck = out / "tt.pt"
    model = EraLM(cfg).to(dev)
    model.cfg.bank = False                                     # TT alone while training
    opt = model.configure_optimizer(lr=a.lr, weight_decay=a.wd)
    data = RandomWindows(arrs["train"], a.batch, a.seq, a.seed, dev)
    steps = int(a.tokens // (a.batch * a.seq))
    step = 0
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        data.load(st["data"])
        step = st["step"]
        torch.set_rng_state(st["rng"])
        if st.get("done"):
            return model
    val = [b for _, b in zip(range(a.val_batches), RandomWindows(arrs["val"], a.batch, a.seq, 12345, dev))]
    last = time.time()
    t0, tok0 = time.time(), step * a.batch * a.seq
    while step < steps:
        s = train_step(model, opt, next(data), clip=1.0, lr=lr_at(step, steps, a.lr, a.warmup))
        step += 1
        if step % a.log_every == 0 or step == steps:
            tps = (step * a.batch * a.seq - tok0) / (time.time() - t0)
            rec = {"phase": "A", "step": step, "steps": steps, "loss": s["loss"], "lr": s["lr"], "grad_norm": s["grad_norm"],
                   "tok_per_s": tps}
            if step % (a.log_every * 10) == 0 or step == steps:
                with torch.no_grad():
                    model.eval()
                    rec["val_nll"] = float(np.mean([model.step(b[:, :-1], b[:, 1:])["tt_nll"].mean().item() for b in val]))
                    model.train()
            log(out, rec)
        if time.time() - last > a.ckpt_minutes * 60 or step == steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "data": data.state(), "step": step,
                        "rng": torch.get_rng_state(), "done": step == steps, "cfg": cfg.to_dict()}, str(ck) + ".tmp")
            os.replace(str(ck) + ".tmp", ck)
            last = time.time()
    return model


def phase_b(a, model, arrs, out, dev):
    path = out / "era.pt"
    if path.exists():
        m, _ = EraLM.load(path, dev)
        return m
    model.cfg.bank = True
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.online, model.stream_online = True, True
    s = Streams(arrs["train"], a.streams, a.seq, limit=int(a.bank_tokens), device=dev)
    model.reset_streams(a.streams)
    t0 = time.time()
    with torch.no_grad():
        for i, b in enumerate(s):
            o = model.step(b[:, :-1], b[:, 1:])
            if (i + 1) % 50 == 0:
                log(out, {"phase": "B", "window": i + 1, "of": len(s), "nll": o["nll"].mean().item(),
                          "tt_nll": o["tt_nll"].mean().item(), **{k: model.monitor[k] for k in
                                                                  ("n_eras", "era_mass", "cache_mass", "collapse")},
                          "births": int(model.bank.counters[2]), "folds": int(model.bank.counters[3]),
                          "tok_per_s": (i + 1) * a.streams * a.seq / (time.time() - t0)})
    model.reset_streams(None)
    model.save(path, extra={"bank_tokens": a.bank_tokens})
    return model


def phase_c(a, model, arrs, out, dev):
    res = {}
    for split in ("val", "test"):
        lim = int(a.eval_tokens)
        st = lambda: Streams(arrs[split], a.streams, a.seq, limit=lim, device=dev)
        frozen = copy.deepcopy(model)
        r_frozen = eval_loss(frozen, st(), online=False)
        preq = copy.deepcopy(model)
        r_preq = eval_loss(preq, st(), online=True)
        nocache = copy.deepcopy(model)
        nocache.cfg.cache = 0
        r_nocache = eval_loss(nocache, st(), online=False)
        res[split] = {"tt_alone": {"nll": r_frozen["tt_nll"], "ppl": r_frozen["tt_ppl"]}, "era_frozen": r_frozen,
                      "era_prequential": r_preq, "era_frozen_no_cache": r_nocache}
        log(out, {"phase": "C", "split": split, **{k: v["nll"] for k, v in res[split].items()}})
    # owned n-gram baseline on exactly the evaluated tokens (each stream shard, all positions after its first)
    train = np.asarray(arrs["train"][: int(a.ngram_tokens)], dtype=np.int64)
    ng = {}
    for order in (3, 4, 5):
        lam0 = {3: (0.1, 0.3, 0.6), 4: (0.1, 0.2, 0.3, 0.4), 5: (0.05, 0.1, 0.2, 0.3, 0.35)}[order]
        m = JMNGram(model.cfg.vocab, order, lam0).fit(train)

        def stream_nll(split, lam):
            s = Streams(arrs[split], a.streams, a.seq, limit=int(a.eval_tokens))
            tot = n = 0
            for st in s.starts:
                x = np.asarray(arrs[split][st: st + len(s) * a.seq + 1], dtype=np.int64)
                lp = m.logprob(x, lam)
                tot += -lp.sum()
                n += lp.size
            return tot / n

        grid = [lam0] + [tuple(np.asarray(g) / sum(g)) for g in _lambda_grid(order)]
        val = {g: stream_nll("val", g) for g in grid}
        best = min(val, key=val.get)
        ng[order] = {"project_lambdas": list(lam0), "val_project": val[lam0], "test_project": stream_nll("test", lam0),
                     "tuned_lambdas": [float(v) for v in best], "val_tuned": val[best], "test_tuned": stream_nll("test", best)}
        log(out, {"phase": "C", "ngram_order": order, **ng[order]})
    res["ngram"] = ng
    return res


def _lambda_grid(order):
    g = []
    for top in (0.3, 0.5, 0.7):
        rest = 1 - top
        w = np.linspace(1, 2, order - 1)
        w = w / w.sum() * rest
        g.append((*w, top))
        g.append((0.01, *(np.full(order - 2, rest / (order - 2)) if order > 2 else ()), top + 0.0))
    return [x for x in g if len(x) == order]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=float, default=2e8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--n-max", type=int, default=16)
    ap.add_argument("--cache", type=int, default=512)
    ap.add_argument("--decider", default="MOM")
    ap.add_argument("--alpha", type=float, default=0.003)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--streams", type=int, default=32)
    ap.add_argument("--bank-tokens", type=float, default=2e7)
    ap.add_argument("--eval-tokens", type=float, default=4e6)
    ap.add_argument("--ngram-tokens", type=float, default=1e8)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-minutes", type=float, default=10)
    ap.add_argument("--phases", default="ABC")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta, arrs = load_bins(a.bins)
    cfg = EraLMConfig(vocab=4096, d=a.d, layers=a.layers, n_max=a.n_max, cache=a.cache, decider=a.decider,
                      alpha=a.alpha, amp=not a.no_amp, compile=a.compile)
    (out / "args.json").write_text(json.dumps({"args": vars(a), "cfg": cfg.to_dict(), "meta": meta}, indent=1))
    model = phase_a(a, cfg, arrs, out, dev)
    log(out, {"phase": "A", "done": True, "n_params": model.n_params()})
    if "B" in a.phases:
        model = phase_b(a, model, arrs, out, dev)
    if "C" in a.phases:
        res = phase_c(a, model, arrs, out, dev)
        res.update(n_params=model.n_params(), n_stats=model.n_stats(), flops_per_token=model.active_flops_per_token(a.seq),
                   n_eras=model.bank.n_alive, births=int(model.bank.counters[2]), folds=int(model.bank.counters[3]))
        (out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

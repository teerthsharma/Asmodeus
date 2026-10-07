"""M0 milestone module (END_GOAL.md §5): LM harness functions for test_m0.py."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

import run
from beds.lang import TokenStream, pin
from lm.cores import build


def token_nll(model, batch):
    out = model(batch)
    if "nll" in out:
        return out["nll"][:, 1:]
    lg = out["logits"][:, :-1]
    nll = torch.nn.functional.cross_entropy(
        lg.reshape(-1, lg.shape[-1]), batch[:, 1:].reshape(-1), reduction="none"
    ).view(lg.shape[:2])
    return nll


def train(model, stream, steps, lr=3e-3, warmup=3, ckpt=None, stop_at=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    step = 0
    if ckpt and Path(ckpt).exists():
        step = run.load_state(ckpt, model, opt, stream.rng)
        sc_path = Path(str(ckpt) + ".stream")
        if sc_path.exists():
            sc = torch.load(sc_path, weights_only=False)
            assert sc["step"] == step
            stream.load(sc["stream"])
            stream.rng.bit_generator.state = sc["stream"]["rng"]

    losses = []
    model.train()

    def save():
        if ckpt:
            run.save_state(ckpt, model, opt, step, stream.rng)
            sc_path = Path(str(ckpt) + ".stream")
            tmp = str(sc_path) + ".tmp"
            torch.save(dict(step=step, stream=stream.state()), tmp)
            os.replace(tmp, sc_path)

    while step < steps:
        if stop_at is not None and step >= stop_at:
            save()
            return losses

        # Warmup LR adjustment
        if step < warmup:
            for param_group in opt.param_groups:
                param_group["lr"] = lr * (step + 1) / warmup
        else:
            for param_group in opt.param_groups:
                param_group["lr"] = lr

        batch = next(stream)
        loss = token_nll(model, batch).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step += 1
        losses.append(loss.item())

    save()
    return losses


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-cores", default="gpt")
    ap.add_argument("--params", type=float, default=2e5)
    ap.add_argument("--vocab", type=int, default=300)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--core", default="gpt")
    ap.add_argument("--data", default="")
    a = ap.parse_args(argv)

    if a.probe:
        cores = a.probe_cores.split(",")
        out = {}
        batch = torch.randint(0, a.vocab, (a.batch, a.seq))
        for core in cores:
            m = build(core, a.vocab, 32, **({"layers": 2, "max_seq": a.seq + 1} if core == "gpt" else {}))
            t0 = time.time()
            for _ in range(a.steps):
                _ = m(batch)
            dt = time.time() - t0
            tok_per_s = (a.steps * a.batch * a.seq) / max(dt, 1e-6)
            out[core] = {"tokens_per_s": tok_per_s, "dt": dt}
        print(json.dumps(out))
        return

    if a.data and not Path(a.data).exists():
        sys.stderr.write(f"data not found: {a.data}\n")
        sys.exit(f"data not found: {a.data}")

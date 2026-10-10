"""PLAN.md item 5 harness: equal-FLOPs LM bake-off (qf, qf_embed, qf_fixed, gpt) with resumable cells and criteria a/b/c."""
import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tokenizers import Tokenizer

import run
from beds.lang import TokenStream, pin
from lm.cores import build, match_params
from qflm import telemetry


def spearman(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = float(spearmanr(x, y)[0])
    return None if math.isnan(r) else r                                   # constant input: no correlation, valid JSON


def steps_for_budget(flops_budget, flops_per_token, batch, seq):
    return int(flops_budget // (3 * flops_per_token * batch * seq))      # fwd + bwd = 3x fwd


def token_nll(model, batch):
    """-> (nll [B,T-1], steps [B,T-1] or None, flops [B,T-1] or None), all aligned to target tokens batch[:,1:]."""
    out = model(batch)
    if "nll" in out:
        nll = out["nll"][:, 1:]
    else:
        lg = out["logits"][:, :-1]
        nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), batch[:, 1:].reshape(-1), reduction="none").view(lg.shape[:2])
    return nll, *(out[k][:, 1:] if k in out else None for k in ("steps", "flops"))


def _nbytes(s):
    if torch.is_tensor(s):
        return s.numel() * s.element_size()
    if isinstance(s, dict):
        s = s.values()
    return sum(_nbytes(x) for x in s) if isinstance(s, (tuple, list, type({}.values()))) else 0


@torch.no_grad()
def state_flat(model, vocab, lengths=(64, 256)):
    if not hasattr(model, "step"):
        return False
    g = torch.Generator().manual_seed(0)
    toks = torch.randint(0, vocab, (max(lengths), 1), generator=g)
    state, sizes = None, {}
    for t in range(max(lengths)):
        state, _ = model.step(state, toks[t])
        if t + 1 in lengths:
            sizes[t + 1] = _nbytes(state)
    return bool(max(sizes.values()) <= 1.05 * min(sizes.values()))


def cell(core, kw, d, data, tok, seed, steps, seq, batch, lr, val_batches, ckpt, stop_at=None, ckpt_every_s=600):
    ckpt = Path(ckpt)
    side = Path(str(ckpt) + ".stream")
    tk = Tokenizer.from_file(str(tok))
    V = tk.get_vocab_size()
    torch.manual_seed(seed)
    model = build(core, V, d, **kw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    stream = TokenStream(data, tk, "train", seq, batch, seed)
    step = 0
    if ckpt.exists():
        sc = torch.load(side, weights_only=False)
        step = run.load_state(ckpt, model, opt, stream.rng)
        assert sc["step"] == step, f"stream sidecar step {sc['step']} != checkpoint step {step}"
        stream.load(sc["stream"])
        stream.rng.bit_generator.state = sc["stream"]["rng"]

    def save():
        tmp = str(side) + ".tmp"
        torch.save(dict(step=step, stream=stream.state()), tmp)
        os.replace(tmp, side)
        run.save_state(ckpt, model, opt, step, stream.rng)

    model.train()
    last = time.time()
    while step < steps:
        if time.time() - last >= ckpt_every_s:                            # Kaggle session limits
            save()
            last = time.time()
        if stop_at is not None and step >= stop_at:
            save()
            return None
        loss = token_nll(model, next(stream))[0].mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
    save()

    model.eval()
    vs = TokenStream(data, tk, "val", seq, batch, 0)
    nl, st, fl = [], [], []
    with torch.no_grad():
        for _ in range(val_batches):
            n, s, f = token_nll(model, next(vs))
            nl.append(n.flatten()); st.append(None if s is None else s.flatten()); fl.append(None if f is None else f.flatten())
    nll = torch.cat(nl)
    nominal = model.active_flops_per_token(seq)
    ppl_s0 = None
    if hasattr(model, "fixed_k"):                                         # forcing s=0 must raise val loss (Adams)
        k0, model.fixed_k = model.fixed_k, 0
        vs0, l0 = TokenStream(data, tk, "val", seq, batch, 0), []
        with torch.no_grad():
            for _ in range(val_batches):
                l0.append(token_nll(model, next(vs0))[0].flatten())
        model.fixed_k = k0
        ppl_s0 = math.exp(torch.cat(l0).mean().item())
    out = dict(val_ppl_s0=ppl_s0, steps=step, val_ppl=math.exp(nll.mean().item()), flops_per_token=nominal,
                val_flops_per_token=float(torch.cat(fl).mean()) if fl[0] is not None else nominal,
                n_params=model.n_params(),
                spearman_steps_surprisal=spearman(torch.cat(st).numpy(), nll.numpy()) if st[0] is not None else None,
                state_flat=state_flat(model, V))
    if hasattr(model, "s_max") and hasattr(model, "generate"):
        out["dynamics"] = telemetry.dynamics(model, n_tokens=256, B=4, seed=seed)
    return out


def criteria(rows):
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r

    def need(*arms):
        miss = [a for a in arms if a not in by]
        return f"missing arm(s): {', '.join(miss)}" if miss else None

    def res(ok, why, **extra):
        return dict({"pass": bool(ok), "why": why}, **extra)

    def shared(a, b):
        return sorted(set(by[a]) & set(by[b]))

    def quota(n):
        return math.ceil(2 * n / 3)

    out = {}
    if m := need("qf", "qf_embed"):
        out["a"] = res(False, m)
    else:
        ss = shared("qf", "qf_embed")
        wins = [s for s in ss if by["qf"][s]["val_ppl"] / by["qf_embed"][s]["val_ppl"] - 1 > 0.02]
        out["a"] = res(ss and len(wins) < quota(len(ss)), f"embed beats qf by >2% on {len(wins)}/{len(ss)} seeds", seeds=wins)
    if m := need("qf", "qf_fixed"):
        out["b"] = res(False, m)
    else:
        q, ss = by["qf"], shared("qf", "qf_fixed")
        rho = [s for s in q if (q[s]["spearman_steps_surprisal"] or -1) >= 0.3]
        beat = [s for s in ss if q[s]["val_ppl"] < by["qf_fixed"][s]["val_ppl"]]
        ratio = np.mean([q[s]["val_flops_per_token"] for s in q]) / np.mean([r["val_flops_per_token"] for r in by["qf_fixed"].values()])
        s0 = [s for s in q if q[s].get("val_ppl_s0") is not None]
        s0ok = [s for s in s0 if q[s]["val_ppl_s0"] > q[s]["val_ppl"]]
        s0pass = not s0 or len(s0ok) >= quota(len(s0))
        s0why = f"forcing s=0 raises ppl on {len(s0ok)}/{len(s0)}" if s0 else "s=0 check absent (skipped)"
        ok = len(rho) >= quota(len(q)) and ss and len(beat) >= quota(len(ss)) and abs(ratio - 1) <= 0.05 and s0pass
        out["b"] = res(ok, f"spearman>=0.3 on {len(rho)}/{len(q)}; beats fixed on {len(beat)}/{len(ss)}; flops ratio {ratio:.3f}; {s0why}", flops_ratio=float(ratio))
    if m := need("qf"):
        out["c"] = res(False, m)
    else:
        flat = [s for s, r in by["qf"].items() if r["state_flat"] is True]
        out["c"] = res(len(flat) == len(by["qf"]), f"state flat on {len(flat)}/{len(by['qf'])} seeds")
    if m := need("qf", "gpt"):
        out["talks_ppl"] = res(False, m)
    else:
        ratio = float(np.mean([r["val_ppl"] for r in by["qf"].values()]) / np.mean([r["val_ppl"] for r in by["gpt"].values()]))
        out["talks_ppl"] = res(ratio <= 1.05, f"qf/gpt val_ppl ratio {ratio:.4f}", ratio=ratio)
    if m := need("qf", "qf_dephased"):
        out["quantum"] = res(False, m)
    else:
        ss = shared("qf", "qf_dephased")
        wins = [s for s in ss if by["qf_dephased"][s]["val_ppl"] / by["qf"][s]["val_ppl"] - 1 > 0.02]
        out["quantum"] = res(ss and len(wins) >= quota(len(ss)), f"qf beats dephased by >2% on {len(wins)}/{len(ss)} seeds", seeds=wins)
    if "qf_real" in by and "qf" in by:                                    # reported only, does not gate
        ss = shared("qf", "qf_real")
        out["quantum"]["real_seeds_qf_beats_by_2pct"] = [s for s in ss if by["qf_real"][s]["val_ppl"] / by["qf"][s]["val_ppl"] - 1 > 0.02]
        out["quantum"]["real_ppl_ratio"] = float(np.mean([by["qf_real"][s]["val_ppl"] / by["qf"][s]["val_ppl"] for s in ss])) if ss else None
    out["new_complete"] = "qf_dephased" in by
    out["new"] = bool(out["a"]["pass"] and out["b"]["pass"] and (out["quantum"]["pass"] if out["new_complete"] else True))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--arm-kw", action="append", default=[])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--params", type=float, default=1e7)
    ap.add_argument("--macs-budget", type=float, default=5e11)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-batches", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not Path(a.data).exists():
        sys.exit(f"data not found: {a.data}")
    try:
        pin(a.data, a.sha256)
    except ValueError as e:
        sys.exit(str(e))
    arms = dict(x.split("=", 1) for x in a.arms.split(","))
    kws = {n: json.loads(j) for n, j in (x.split("=", 1) for x in a.arm_kw)}
    V = Tokenizer.from_file(a.tok).get_vocab_size()
    plan = {}
    for name, core in arms.items():
        kw = kws.get(name, {})
        d = kw["N"] if "N" in kw else match_params(core, a.params, V, **kw)[0]
        torch.manual_seed(0)
        model = build(core, V, d, **kw)
        worst = model.active_flops_per_token(a.seq) / 2
        with torch.no_grad():
            o = model(next(TokenStream(a.data, Tokenizer.from_file(a.tok), "train", a.seq, a.batch, 0)))
        # ponytail: budget fixed from the init-time expected think steps; they drift during training,
        # re-measure per epoch if the drift exceeds 5%.
        macs = float(o["flops"][:, 1:].mean()) / 2 if "flops" in o else worst
        plan[name] = (core, kw, d, steps_for_budget(a.macs_budget, macs, a.batch, a.seq), macs, worst)
    out = Path(a.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    res = out / "results.jsonl"
    rows = [json.loads(l) for l in res.read_text().splitlines()] if res.exists() else []
    done = {(r["arm"], r["seed"]) for r in rows}
    for seed in map(int, a.seeds.split(",")):
        for name, (core, kw, d, steps, macs, worst) in plan.items():
            if (name, seed) in done:
                continue
            r = dict(arm=name, seed=seed, core=core, d=d, macs_per_token=macs, macs_per_token_worst=worst,
                     **cell(core, kw, d, a.data, a.tok, seed, steps, a.seq, a.batch, a.lr, a.val_batches,
                            out / "ckpt" / f"{name}_s{seed}.pt"))
            r["val_macs_per_token"] = r["val_flops_per_token"] / 2
            rows.append(r)
            with open(res, "a") as f:
                f.write(json.dumps(r) + "\n")
    crit = criteria(rows)
    (out / "criteria.json").write_text(json.dumps(crit, indent=1))
    print(json.dumps(crit, indent=1))


if __name__ == "__main__":
    main()

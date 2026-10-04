"""Regular-language capacity test (PREREG Amendment 5): langs x arms x seeds, excess entropy in bits/symbol.

Refuses to run any arm (eval or --tune) unless PREREG.md has the exact heading line "## Amendment 5 (pre-result)"
(--dry-run always works).
Per-arm lr tuning, on tuning seed TUNE_SEED=3 only: for each (arm, N, lang), one model per lr in LR_GRID is trained
on seed-3 init, and the lr with the lowest excess on the VAL split (never test) wins; ties -> smaller lr. Results are
cached in TUNE_DIR/<arm>_N<N>_<lang>.json. Eval seeds are EVAL_SEEDS=(0,1,2); seed 3 is rejected as an eval seed.
run_job reads the tuned lr from the cache and raises if missing, so eval cannot start before tuning.
Writes runs/reglang/<lang>__<arm>__s<seed>.json atomically; existing files are skipped (resumable).
"""
import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from beds import reglang as rl
from run_t4b import MAX_PROCS, clean_sha, dirty

ZERO = Path(__file__).resolve().parent
OUT = ZERO / "runs" / "reglang"
PREREG = ZERO / "PREREG.md"
AMENDMENT = "## Amendment 5 (pre-result)"
TUNE_DIR = OUT / "tune"
LR_GRID = (1e-3, 3e-3, 1e-2)
TUNE_SEED = 3
EVAL_SEEDS = (0, 1, 2)
SPLITS = dict(train=(20000, 0), val=(2000, 1), test=(2000, 2))
LANGS = rl.LANGS
NS = (8, 16, 32)
# (arm, N) job variants; N=0 for the non-QF-LM arms; dephased has one N (ponytail: N=16, spec lists a single job)
ARMS = [(a, n) for a in ("qflm", "embed", "nothink", "nonull") for n in NS] + [("gru", 0), ("gpt", 0), ("dephased", 16)]
LN2 = math.log(2)


def build(arm, N, seed):
    torch.manual_seed(seed)
    if arm in ("qflm", "embed", "nothink", "nonull"):
        from qflm.model import QFLM
        return QFLM(N, 3, arm=arm, seed=seed)
    if arm == "gru":
        from arms.lm_baselines import GRULM
        return GRULM(4, 16, 16, 1)
    if arm == "gpt":
        from arms.lm_baselines import GPTLM
        return GPTLM(4, 32, 2, 2, 65)
    if arm == "dephased":
        try:
            from qflm.controls import DephasedQFLM
        except ImportError:
            raise NotImplementedError("PENDING: qflm/controls.py")
        if "forward" not in vars(DephasedQFLM):  # controls.py in flux: inherited forward would be the quantum path
            raise NotImplementedError("PENDING: qflm/controls.py (DephasedQFLM has no forward)")
        return DephasedQFLM(N, 3, seed=seed)
    raise KeyError(arm)


def kind_of(model):
    from qflm.model import QFLM
    return "qflm" if isinstance(model, QFLM) else "lm"


def _by_len(seqs):
    g = defaultdict(list)
    for q in seqs:
        g[len(q)].append(q)
    return g


def qflm_nll_sum(model, seqs):
    """Summed NLL (nats) over all symbols incl. EOS. QF-LM forward has no padding mask, so score equal-length
    groups (forward is mean over B*T, so mean * B*T = group sum); sequences carry no BOS (psi0 plays BOS)."""
    tot = 0.0
    with torch.no_grad():
        for L, g in _by_len(seqs).items():
            for i in range(0, len(g), 256):
                x = torch.tensor(g[i:i + 256])
                tot += model(x).item() * x.numel()
    return tot


def lm_nll_sum(model, seqs, bs=256):
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            x, m = rl.batch(seqs[i:i + bs], bos=True)
            tot += (model.nll(x) * m).sum().item()
    return tot


def excess(model, kind, lang, seqs, max_len=64):
    """(sum model NLL - sum true NLL) / n_symbols, bits/symbol, on the real symbols (incl. EOS)."""
    n = sum(map(len, seqs))
    true = -sum(rl.true_logprob(lang, q, max_len).sum() for q in seqs)
    mod = qflm_nll_sum(model, seqs) if kind == "qflm" else lm_nll_sum(model, seqs)
    return (mod - true) / n / LN2


def train(model, lang, train_seqs, steps, lr, batch, seed):
    g = np.random.default_rng(seed)
    kind = kind_of(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    groups = _by_len(train_seqs)
    for _ in range(steps):
        if kind == "qflm":  # no padding: draw a length, then a batch of that length
            q = train_seqs[g.integers(len(train_seqs))]
            gr = groups[len(q)]
            loss = model(torch.tensor([gr[i] for i in g.integers(len(gr), size=batch)]))
        else:
            x, m = rl.batch([train_seqs[i] for i in g.integers(len(train_seqs), size=batch)], bos=True)
            loss = (model.nll(x) * m).sum() / m.sum()
        opt.zero_grad()
        loss.backward()
        if kind == "qflm":
            from qflm.model import clip
            clip(model)
        opt.step()
    return model


def load_split(lang, which):
    n, sd = SPLITS[which]
    return rl.sample(lang, n, sd)


def plan(langs, arms, seeds, out=OUT):
    if TUNE_SEED in seeds:
        raise ValueError(f"seed {TUNE_SEED} is the tuning seed, not an eval seed")
    out = Path(out)
    jobs = []
    for l in langs:
        for a, n in arms:
            for s in seeds:
                k = f"{l}__{a}{n or ''}__s{s}"
                jobs.append(dict(lang=l, arm=a, N=n, seed=s, key=k, path=out / f"{k}.json", done=(out / f"{k}.json").exists()))
    return jobs


def tune_path(arm, N, lang):
    return Path(TUNE_DIR) / f"{arm}_N{N}_{lang}.json"


def tune_plan(langs, arms):
    return [dict(lang=l, arm=a, N=n, key=tune_path(a, n, l).stem, done=tune_path(a, n, l).exists())
            for l in langs for a, n in arms]


def _atomic_json(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, path)


def tune(arm, N, lang, steps, batch):
    """Pick the lr from LR_GRID with the lowest VAL excess on TUNE_SEED; never loads the test split."""
    tr, va = load_split(lang, "train"), load_split(lang, "val")
    val = {}
    for lr in LR_GRID:
        m = train(build(arm, N, TUNE_SEED), lang, tr, steps, lr, batch, TUNE_SEED)
        val[lr] = float(excess(m, kind_of(m), lang, va))
    best = min(LR_GRID, key=lambda lr: (val[lr], lr))  # ties -> smaller lr
    _atomic_json(tune_path(arm, N, lang), dict(arm=arm, N=N, lang=lang, seed=TUNE_SEED, steps=steps, batch=batch,
                                              lr=best, val_excess={str(k): v for k, v in val.items()}))
    return best


def tuned_lr(arm, N, lang):
    p = tune_path(arm, N, lang)
    if not p.exists():
        raise FileNotFoundError(f"no tune cache {p}: run --tune first")
    return json.loads(p.read_text())["lr"]


def require_amendment():
    if AMENDMENT not in Path(PREREG).read_text(encoding="utf-8").splitlines():
        raise SystemExit(f'refusing: PREREG.md has no "{AMENDMENT}" heading (L-PREREG)')


def run_job(job, sha, steps=5000, batch=64):
    lang, seed = job["lang"], job["seed"]
    lr = tuned_lr(job["arm"], job["N"], lang)  # raises before any training if untuned
    tr, va, te = (load_split(lang, w) for w in ("train", "val", "test"))
    m = train(build(job["arm"], job["N"], seed), lang, tr, steps, lr, batch, seed)
    k = kind_of(m)
    rec = dict(job={x: job[x] for x in ("lang", "arm", "N", "seed")}, sha=sha, steps=steps, lr=lr, batch=batch,
               excess_test=excess(m, k, lang, te), excess_val=excess(m, k, lang, va),
               hash=dict(train=rl.data_hash(tr), val=rl.data_hash(va), test=rl.data_hash(te)))
    _atomic_json(job["path"], rec)
    return rec


def _tune_worker(args):
    job, steps, batch, threads = args
    torch.set_num_threads(threads)
    return job["key"], tune(job["arm"], job["N"], job["lang"], steps, batch)


def _worker(args):
    job, sha, steps, threads = args
    torch.set_num_threads(threads)
    return job["key"], run_job(job, sha, steps)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--langs", nargs="+", default=list(LANGS))
    p.add_argument("--arms", nargs="+", default=None, help="arm names (all N variants); default all")
    p.add_argument("--seeds", nargs="+", type=int, default=list(EVAL_SEEDS))
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--tune", action="store_true", help="run only the lr-tuning jobs (seed 3, val split)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.procs <= MAX_PROCS:
        raise SystemExit(f"--procs must be in [1, {MAX_PROCS}]")
    out = Path(a.out).resolve()
    arms = [x for x in ARMS if a.arms is None or x[0] in a.arms]
    jobs = plan(a.langs, arms, a.seeds, out)
    todo = [j for j in jobs if not j["done"]]
    tjobs = tune_plan(a.langs, arms)
    ttodo = [j for j in tjobs if not j["done"]]
    if a.dry_run:
        print(f"{len(ttodo)}/{len(tjobs)} tune pending -> {TUNE_DIR}")
        for j in tjobs:
            print(("tune-done " if j["done"] else "tune-todo ") + j["key"])
        print(f"{len(todo)}/{len(jobs)} pending -> {out}")
        for j in jobs:
            print(("done " if j["done"] else "todo ") + j["key"])
        return
    require_amendment()
    threads = max(1, (os.cpu_count() or 1) // a.procs)
    if a.tune:
        args = [(j, a.steps, 64, threads) for j in ttodo]
        results = map(_tune_worker, args) if a.procs == 1 else ProcessPoolExecutor(a.procs).map(_tune_worker, args)
        for key, lr in results:
            print(key, f"lr={lr}", flush=True)
        return
    ign = ["house-events.jsonl"] + ([os.path.relpath(out, ZERO)] if out.is_relative_to(ZERO) else [])
    sha = clean_sha(ZERO, ign)
    args = [(j, sha, a.steps, threads) for j in todo]
    results = map(_worker, args) if a.procs == 1 else ProcessPoolExecutor(a.procs).map(_worker, args)
    for key, rec in results:
        print(key, f"excess_test={rec['excess_test']:.4f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

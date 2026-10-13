"""Phase 2 verdict: an arm's mean test loss against the GPT ladder's frontier at equal training FLOPs (B1) and
equal parameters (B2). Rules frozen in p2/PREREG.md; reads p2/results/results.jsonl.

  python p2/frontier.py p2/results/results.jsonl era
"""
import json
import sys
from statistics import mean, stdev

import numpy as np

LADDER, SEEDS, MARGIN = ("gpt-s", "gpt-m", "gpt-l"), (100, 101, 102), 0.02


def rows_by(rows, budget):
    by = {}
    for r in rows:
        if r["seed"] in SEEDS and r["tokens_budget"] == budget:
            by.setdefault(r["name"], {})[r["seed"]] = r
    return by


def interp(xs, ys, x):
    """Piecewise-linear in log10(x) through the ladder points; None outside their range (no extrapolation)."""
    lx, order = np.log10(xs), np.argsort(xs)
    lx, ys = lx[order], np.asarray(ys)[order]
    t = np.log10(x)
    return float(np.interp(t, lx, ys)) if lx[0] <= t <= lx[-1] else None


def verdict(rows, arm, budget):
    by = rows_by(rows, budget)
    miss = [n for n in (*LADDER, arm) if set(by.get(n, {})) != set(SEEDS)]
    if miss:
        return dict(complete=False, missing=miss)
    lad = {n: by[n] for n in LADDER}
    out = dict(complete=True, arm=arm, test_loss=mean(r["test_loss"] for r in by[arm].values()))
    sd = max(stdev([r["test_loss"] for r in s.values()]) for s in lad.values())
    delta_bar = max(MARGIN, 2 * sd)
    out["margin"] = delta_bar
    for bar, key in (("B1_equal_flops", "train_flops"), ("B2_equal_params", "n_params")):
        xs = [mean(r[key] for r in lad[n].values()) for n in LADDER]
        f = lambda x: interp(xs, [mean(r["test_loss"] for r in lad[n].values()) for n in LADDER], x)
        x = mean(r[key] for r in by[arm].values())
        ref = f(x)
        if ref is None:
            out[bar] = dict(result="out_of_range", x=x, ladder=xs)
            continue
        d = out["test_loss"] - ref
        below = sum(by[arm][s]["test_loss"] < ref for s in SEEDS)
        res = ("win" if d <= -delta_bar and below >= 2 else "lose" if d >= delta_bar else "tie")
        out[bar] = dict(result=res, x=x, frontier_loss=ref, delta=d, seeds_below=below)
    return out


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1])]
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000_000
    print(json.dumps(verdict(rows, sys.argv[2], budget), indent=1))

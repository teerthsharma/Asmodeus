"""E0 runner: tune on seeds 100-104 (REC), then evaluate on seeds 0-9 (REC, NOREC, NULL). Writes results.json.

Usage: python run.py [--quick]
"""
import itertools
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

import arms
from bed import V, make_bed

TUNE, EVAL = [100, 101, 102, 103, 104], list(range(10))
GRID = {
    "EMA": [0.98, 0.99, 0.995, 0.998, 0.999],
    "BOCPD": [1 / 100, 1 / 200, 1 / 400, 1 / 800, 1 / 1600],
    "GLOBAL-o2": [0.5, 2.0, 8.0, 32.0],
    "FCPA": list(itertools.product([0.001, 0.01, 0.05], [32, 64, 128])),
}
ALPHAS = (1e-3, 3e-3, 1e-2, 3e-2)


def run_arm(job):
    seed, kind, arm, hp = job
    x, r, _ = make_bed(seed, kind=kind)
    t0 = time.time()
    info = {}
    if arm == "ORACLE":
        out = {"ORACLE": arms.oracle(x, r, V)}
    elif arm == "GLOBAL-o1":
        out = {arm: arms.global_o1(x, V)}
    elif arm == "GLOBAL-o2":
        out = {arm: arms.global_o2(x, V, hp)}
    elif arm == "EMA":
        out = {arm: arms.ema(x, V, hp)}
    elif arm == "BOCPD":
        out = {arm: arms.bocpd(x, V, hp)}
    elif arm == "FCPA":
        delta, w = hp[0], hp[1]
        alphas = tuple(hp[2:]) or ALPHAS
        res = arms.fcpa(x, V, delta, w, alphas=alphas, return_info=True)
        out = {"FCPA-HARD": res["hard"]}
        out.update({f"FCPA-SOFT@{al}": res["soft"][al] for al in alphas})
        info = {"n_eras": res["n_eras"], "births": res["births"], "delay": _delays(r, res["active"])}
    return {"seed": seed, "kind": kind, "arm": arm, "hp": hp, "secs": time.time() - t0,
            "loss": {k: v.tolist() for k, v in out.items()}, "info": info}


def _delays(r, active):
    """Per true switch: steps until the active era is the era that holds most of the new regime's tokens."""
    owner = {}
    for reg in np.unique(r):
        vals, cnt = np.unique(active[r == reg], return_counts=True)
        owner[reg] = vals[cnt.argmax()]
    sw = np.flatnonzero(np.diff(r)) + 1
    ends = list(sw[1:]) + [len(r)]
    out = []
    for s, e in zip(sw, ends):
        hit = np.flatnonzero(active[s:e] == owner[r[s]])
        out.append(int(hit[0]) if len(hit) else int(e - s))
    return out


def reentry_mask(r, n=100):
    """Token mask (aligned with losses, index t-1) for the first n tokens of segments whose regime was seen before."""
    m = np.zeros(len(r) - 1, bool)
    seen, starts = {r[0]}, [0] + list(np.flatnonzero(np.diff(r)) + 1)
    for s in starts[1:]:
        if r[s] in seen:
            m[max(s - 1, 0):min(s - 1 + n, len(r) - 1)] = True
        seen.add(r[s])
    return m


def main(quick=False):
    tune, ev = (TUNE[:2], EVAL[:3]) if quick else (TUNE, EVAL)
    with Pool() as pool:
        jobs = [(s, "REC", a, hp) for s in tune for a, hps in GRID.items() for hp in hps]
        tres = pool.map(run_arm, jobs)
        picks = {}
        for a, hps in GRID.items():
            scores = {}
            for res in (j for j in tres if j["arm"] == a):
                for name, l in res["loss"].items():
                    scores.setdefault((name, tuple(np.atleast_1d(res["hp"]))), []).append(np.mean(l))
            for (name, hp), v in scores.items():
                key = name.split("@")[0]
                hp_full = hp + ((float(name.split("@")[1]),) if "@" in name else ())
                if key not in picks or np.mean(v) < picks[key][1]:
                    picks[key] = (hp_full, float(np.mean(v)))
        # the better FCPA arm is chosen on tuning seeds only
        fcpa_pick = min(("FCPA-HARD", "FCPA-SOFT"), key=lambda k: picks[k][1])
        ejobs = []
        for kind in ("REC", "NOREC", "NULL"):
            for s in ev:
                ejobs += [(s, kind, "ORACLE", None), (s, kind, "GLOBAL-o1", None)]
                ejobs += [(s, kind, a, picks[a][0][0]) for a in ("GLOBAL-o2", "EMA", "BOCPD")]
                ejobs += [(s, kind, "FCPA", picks["FCPA-HARD"][0]), (s, kind, "FCPA", picks["FCPA-SOFT"][0])]
        eres = pool.map(run_arm, ejobs)
    rows = {}
    for res in eres:
        x, r, _ = make_bed(res["seed"], kind=res["kind"])
        mask = reentry_mask(r)
        for name, l in res["loss"].items():
            if name.startswith("FCPA-SOFT@") and float(name.split("@")[1]) != picks["FCPA-SOFT"][0][2]:
                continue
            if name.startswith("FCPA-SOFT") and tuple(res["hp"][:2]) != tuple(picks["FCPA-SOFT"][0][:2]):
                continue
            if name == "FCPA-HARD" and tuple(res["hp"][:2]) != tuple(picks["FCPA-HARD"][0][:2]):
                continue
            key = name.split("@")[0]
            l = np.asarray(l)
            rows[(res["kind"], res["seed"], key)] = {
                "loss": float(l.mean()), "reentry": float(l[mask].mean()) if mask.any() else None,
                "secs": res["secs"], **({k: v for k, v in res["info"].items()} if key.startswith("FCPA") else {})}
    out = {"picks": {k: list(v[0]) + [v[1]] for k, v in picks.items()}, "fcpa_pick": fcpa_pick,
           "rows": [{"kind": k, "seed": s, "arm": a, **v} for (k, s, a), v in sorted(rows.items())]}
    with open("results_quick.json" if quick else "results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["picks"]), fcpa_pick)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)

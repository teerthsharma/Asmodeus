"""E0c runner (E0c_PREREG.md): tune alpha x n_min for SOFT+BMX on seeds 100-104 (REC), evaluate on fresh seeds 20-29."""
import itertools
import json
from multiprocessing import Pool

import numpy as np

import arms
from bed import V, make_bed
from run import reentry_mask

E0 = json.load(open("results.json"))["picks"]
DELTA, W, HAZ = E0["FCPA-SOFT"][0], int(E0["FCPA-SOFT"][1]), E0["BOCPD"][0]
ALPHAS, NMIN = (1e-3, 3e-3, 1e-2, 3e-2), (0, 32, 64)
TUNE, EVAL, K = [100, 101, 102, 103, 104], list(range(20, 30)), 4


def purity(active, r):
    maj = {e: np.bincount(r[active == e]).argmax() for e in np.unique(active)}
    return float(np.mean([maj[a] == rr for a, rr in zip(active, r)]))


def tune_job(args):
    seed, n_min = args
    x, r, _ = make_bed(seed, kind="REC")
    b = np.exp(-arms.bocpd(x, V, HAZ))
    out = {}
    res = arms.fcpa(x, V, DELTA, W, alphas=ALPHAS, return_info=True, ext=b, merge="global", n_min=n_min)
    for al in ALPHAS:
        out[("BMX", n_min, al)] = (res["soft"][al].mean(), res["n_eras"])
    if n_min == 0:
        ref = arms.fcpa(x, V, DELTA, W, alphas=ALPHAS, return_info=True, ext=b, merge=True)
        for al in ALPHAS:
            out[("BM", 0, al)] = (ref["soft"][al].mean(), ref["n_eras"])
    return out


def eval_job(args):
    seed, kind, al, n_min = args
    x, r, _ = make_bed(seed, kind=kind)
    bl = arms.bocpd(x, V, HAZ)
    res = arms.fcpa(x, V, DELTA, W, alphas=(al,), return_info=True, ext=np.exp(-bl), merge="global", n_min=n_min)
    return seed, kind, {"BMX": res["soft"][al], "eras": res["n_eras"], "purity": purity(res["active"], r),
                        "BOCPD": bl, "ORACLE": arms.oracle(x, r, V), "GLOBAL-o1": arms.global_o1(x, V),
                        "GLOBAL-o2": arms.global_o2(x, V, E0["GLOBAL-o2"][0]), "EMA": arms.ema(x, V, E0["EMA"][0]),
                        "mask": reentry_mask(r)}


def main():
    with Pool() as pool:
        tres = pool.map(tune_job, [(s, n) for s in TUNE for n in NMIN])
        agg = {}
        for o in tres:
            for key, (l, ne) in o.items():
                agg.setdefault(key, []).append((l, ne))
        ref_loss = min(np.mean([l for l, _ in v]) for key, v in agg.items() if key[0] == "BM")
        ok = {key: (np.mean([abs(ne - K) for _, ne in v]), np.mean([l for l, _ in v]))
              for key, v in agg.items() if key[0] == "BMX" and np.mean([l for l, _ in v]) <= 1.02 * ref_loss}
        pick = min(ok, key=lambda key: ok[key])
        _, n_min, al = pick
        eres = pool.map(eval_job, [(s, k, al, n_min) for k in ("REC", "NOREC", "NULL") for s in EVAL])
    owned = ["GLOBAL-o1", "GLOBAL-o2", "EMA", "BOCPD"]
    L = ["# E0c results (fresh seeds 20-29)", "",
         f"Tuning: SOFT+BM reference loss {ref_loss:.4f}; admissible (|eras-4|, loss): "
         + ", ".join(f"n_min={kk[1]} a={kk[2]}: ({v[0]:.2f}, {v[1]:.4f})" for kk, v in sorted(ok.items())),
         f"Pick: n_min = {n_min}, alpha = {al}", ""]
    reg, re = {}, {}
    for kind in ("REC", "NOREC", "NULL"):
        rows = [o for s, k, o in sorted(eres, key=lambda z: z[0]) if k == kind]
        L += [f"## {kind}", "", "| arm | regret vs oracle (mean ± sd) | re-entry excess |", "|---|---|---|"]
        for a in owned + ["BMX"]:
            reg[kind, a] = np.array([o[a].mean() - o["ORACLE"].mean() for o in rows])
            re[kind, a] = np.mean([o[a][o["mask"]].mean() - o["ORACLE"][o["mask"]].mean() for o in rows]) if kind == "REC" else np.nan
            L.append(f"| {'SOFT+BMX' if a == 'BMX' else a} | {reg[kind, a].mean():.4f} ± {reg[kind, a].std():.4f} | {re[kind, a]:.4f} |")
        L += ["", f"SOFT+BMX eras per seed {[o['eras'] for o in rows]}; purity {np.round([o['purity'] for o in rows], 3).tolist()}", ""]
        if kind == "REC":
            rec_eras = [o["eras"] for o in rows]
        if kind == "NULL":
            null_births = sum(o["eras"] - 1 for o in rows)
    best = np.min(np.stack([reg["REC", a] for a in owned]), 0)
    n1 = int((reg["REC", "BMX"] <= 0.5 * best).sum())
    r2 = reg["NOREC", "BMX"].mean() / reg["NOREC", "BOCPD"].mean()
    n3 = sum(e <= 6 for e in rec_eras)
    b4 = re["REC", "BMX"] <= 0.5 * re["REC", "BOCPD"]
    v = lambda b: "**PASS**" if b else "**FAIL**"
    L += ["## Bars (frozen in E0c_PREREG.md)", "",
          f"- B3'': NULL births {null_births} (need <= 2); REC seeds with <= 6 eras {n3}/10 (need >= 8): {v(null_births <= 2 and n3 >= 8)}. Exactly 4 eras on {sum(e == 4 for e in rec_eras)}/10.",
          f"- B1': {n1}/10 (need >= 8): {v(n1 >= 8)}. Mean regret {reg['REC', 'BMX'].mean():.4f} vs best owned {best.mean():.4f}.",
          f"- B2': NOREC ratio to BOCPD {r2:.2f} (need <= 1.25): {v(r2 <= 1.25)}.",
          f"- B4': re-entry {re['REC', 'BMX']:.4f} vs 0.5 x BOCPD {0.5 * re['REC', 'BOCPD']:.4f}: {v(b4)}.", ""]
    open("RESULTS_E0c.md", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

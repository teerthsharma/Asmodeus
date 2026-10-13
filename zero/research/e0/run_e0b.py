"""E0b runner (E0b_PREREG.md): tune alpha per arm on seeds 100-104 (REC), evaluate on fresh seeds 10-19. Writes results_e0b.json."""
import json
import sys
from multiprocessing import Pool

import numpy as np

import arms
from bed import V, make_bed
from run import reentry_mask

E0 = json.load(open("results.json"))["picks"]
DELTA, W, HAZ = E0["FCPA-SOFT"][0], int(E0["FCPA-SOFT"][1]), E0["BOCPD"][0]
ALPHAS = (1e-3, 3e-3, 1e-2, 3e-2)
VARIANTS = {"SOFT": (False, False), "SOFT+M": (False, True), "SOFT+B": (True, False), "SOFT+BM": (True, True)}
TUNE, EVAL = [100, 101, 102, 103, 104], list(range(10, 20))


def job(args):
    seed, kind, alphas = args
    x, r, _ = make_bed(seed, kind=kind)
    b = arms.bocpd(x, V, HAZ)
    out = {"BOCPD": b.mean()}
    for name, (use_b, merge) in VARIANTS.items():
        res = arms.fcpa(x, V, DELTA, W, alphas=alphas[name], return_info=True, ext=np.exp(-b) if use_b else None, merge=merge)
        for al in alphas[name]:
            out[f"{name}@{al}"] = res["soft"][al]
        out[f"{name}#eras"] = res["n_eras"]
    if kind != "TUNE":
        mask = reentry_mask(r)
        out["mask"] = mask
        out["BOCPD_l"] = b
        out["ORACLE_l"] = arms.oracle(x, r, V)
        out["GLOBAL-o1_l"] = arms.global_o1(x, V)
        out["GLOBAL-o2_l"] = arms.global_o2(x, V, E0["GLOBAL-o2"][0])
        out["EMA_l"] = arms.ema(x, V, E0["EMA"][0])
    return seed, kind, out


def main():
    with Pool() as pool:
        tres = pool.map(job, [(s, "REC", {n: ALPHAS for n in VARIANTS}) for s in TUNE])
        pick = {n: min(ALPHAS, key=lambda al: np.mean([o[f"{n}@{al}"].mean() for _, _, o in tres])) for n in VARIANTS}
        eres = pool.map(job, [(s, k, {n: (pick[n],) for n in VARIANTS}) for k in ("REC", "NOREC", "NULL") for s in EVAL])
    lines = ["# E0b results (fresh seeds 10-19)", "", f"alpha picks (tuning seeds 100-104): {pick}; delta {DELTA}, w {W}, BOCPD hazard {HAZ}", ""]
    owned = ["GLOBAL-o1", "GLOBAL-o2", "EMA", "BOCPD"]
    reg, reent, eras = {}, {}, {}
    for kind in ("REC", "NOREC", "NULL"):
        rows = sorted([(s, o) for s, k, o in eres if k == kind], key=lambda z: z[0])
        lines += [f"## {kind}", "", "| arm | regret vs oracle (mean ± sd) | re-entry excess | eras per seed |", "|---|---|---|---|"]
        for a in owned + list(VARIANTS):
            l = [o[f"{a}_l"] if a in owned else o[f"{a}@{pick[a]}"] for _, o in rows]
            orc = [o["ORACLE_l"] for _, o in rows]
            reg[kind, a] = np.array([li.mean() - oi.mean() for li, oi in zip(l, orc)])
            reent[kind, a] = np.mean([li[o["mask"]].mean() - oi[o["mask"]].mean() for li, oi, (_, o) in zip(l, orc, rows)]) if kind == "REC" else np.nan
            eras[kind, a] = [o[f"{a}#eras"] for _, o in rows] if a in VARIANTS else []
            lines.append(f"| {a} | {reg[kind, a].mean():.4f} ± {reg[kind, a].std():.4f} | {reent[kind, a]:.4f} | {eras[kind, a]} |")
        lines.append("")
    P = "SOFT+BM"
    best = np.min(np.stack([reg["REC", a] for a in owned]), 0)
    n1 = int((reg["REC", P] <= 0.5 * best).sum())
    r2 = reg["NOREC", P].mean() / reg["NOREC", "BOCPD"].mean()
    nb = int(sum(np.array(eras["NULL", P]) - 1)); n3 = int(sum(np.array(eras["REC", P]) <= 6))
    v = lambda b: "**PASS**" if b else "**FAIL**"
    lines += ["## Bars (primary arm SOFT+BM, frozen in E0b_PREREG.md)", "",
              f"- B1': {n1}/10 seeds with regret <= 0.5 x best owned (need >= 8): {v(n1 >= 8)}. Mean regret {reg['REC', P].mean():.4f} vs best owned {best.mean():.4f}.",
              f"- B2': NOREC regret ratio to BOCPD {r2:.2f} (need <= 1.25): {v(r2 <= 1.25)}.",
              f"- B3': NULL spurious births {nb} (need <= 2); REC seeds with <= 6 eras {n3}/10 (need >= 8): {v(nb <= 2 and n3 >= 8)}.",
              f"- B4': re-entry excess {reent['REC', P]:.4f} vs 0.5 x BOCPD {0.5 * reent['REC', 'BOCPD']:.4f}: {v(reent['REC', P] <= 0.5 * reent['REC', 'BOCPD'])}.", ""]
    open("RESULTS_E0b.md", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

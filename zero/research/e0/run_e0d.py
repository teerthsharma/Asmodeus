"""E0d/E0e tournament runner (E0d_PREREG.md, E0e_PREREG.md). `python run_e0d.py e0e` runs round 2."""
import sys
import itertools
import json
from multiprocessing import Pool

import numpy as np

import arms
import deciders as D
from bed import V, make_bed
from run import reentry_mask

E0 = json.load(open("results.json"))["picks"]
DELTA, W, HAZ, NMIN = 0.05, 32, 1 / 400, 64
BEDS = {"REC": ("REC", {}), "NOREC": ("NOREC", {}), "NULL": ("NULL", {}), "CYC": ("CYC", {}),
        "CYC-S": ("CYC", {"seg_min": 20, "seg_mean": 100})}
ALPHAS = (1e-3, 3e-3, 1e-2, 3e-2)
GRID = ([("SOFT", a, None) for a in ALPHAS] + [("MOM", a, None) for a in ALPHAS]
        + [("CTC", a, q) for a, q in itertools.product(ALPHAS, (0.3, 0.8))]
        + [("TEMP", a, T) for a, T in itertools.product(ALPHAS, (0.5, 0.75, 1.5))]
        + [("HARD", a, th) for a, th in itertools.product(ALPHAS, (0.9, 0.99))]
        + [("GBAYES", a, e) for a, e in itertools.product(ALPHAS, (0.5, 0.8))])
ROUND2 = len(sys.argv) > 1 and sys.argv[1] == "e0e"
if ROUND2:
    GRID += [("MOM+GB", a, e) for a, e in itertools.product(ALPHAS, (0.5, 0.8))]
TUNE = [("REC", s) for s in range(100, 105)] + [("CYC-S", s) for s in range(200, 205)]
TUNE_BEDS = ("REC", "CYC-S")
EVAL = list(range(30, 40))
if ROUND2:
    TUNE += [("NOREC", s) for s in range(300, 305)]
    TUNE_BEDS = ("REC", "CYC-S", "NOREC")
    EVAL = list(range(40, 50))
TAG = "E0e" if ROUND2 else "E0d"


def make(spec):
    name, a, p = spec
    return {"SOFT": lambda: D.Soft(2, a), "MOM": lambda: D.Momentum(2, a), "CTC": lambda: D.CTC(2, a, p),
            "TEMP": lambda: D.Tempered(2, a, p), "HARD": lambda: D.HardJump(2, a, p),
            "GBAYES": lambda: D.GeneralBayes(2, a, p), "MOM+GB": lambda: D.MomentumGB(2, a, p)}[name]()


def job(args):
    bed, seed, specs, owned = args
    kind, kw = BEDS[bed]
    x, r, _ = make_bed(seed, kind=kind, **kw)
    bl, bvec = arms.bocpd(x, V, HAZ, return_vec=True)
    ds = [make(s) for s in specs]
    res = arms.fcpa(x, V, DELTA, W, ext=np.exp(-bl), merge="global", n_min=NMIN, deciders=ds, ext_vec=bvec,
                    return_info=True)
    out = {"loss": {s: float(res["dec"][d].mean()) for s, d in zip(specs, ds)},
           "single": {s: d.single / (len(x) - 1) for s, d in zip(specs, ds)}, "eras": res["n_eras"]}
    if owned:
        orc = arms.oracle(x, r, V)
        out["oracle"] = float(orc.mean())
        out["owned"] = {"BOCPD": float(bl.mean()), "GLOBAL-o1": float(arms.global_o1(x, V).mean()),
                        "GLOBAL-o2": float(arms.global_o2(x, V, E0["GLOBAL-o2"][0]).mean()),
                        "EMA": float(arms.ema(x, V, E0["EMA"][0]).mean())}
    return bed, seed, out


def main():
    with Pool() as pool:
        tres = pool.map(job, [(b, s, GRID, False) for b, s in TUNE])
        score = {spec: np.mean([np.mean([o["loss"][spec] for b, s, o in tres if b == bed]) for bed in TUNE_BEDS])
                 for spec in GRID}
        picks = {name: min((sp for sp in GRID if sp[0] == name), key=score.get)
                 for name in dict.fromkeys(sp[0] for sp in GRID)}
        specs = list(picks.values())
        eres = pool.map(job, [(b, s, specs, True) for b in BEDS for s in EVAL])
    R = {(b, s): o for b, s, o in eres}
    reg = {(b, sp): np.array([R[b, s]["loss"][sp] - R[b, s]["oracle"] for s in EVAL]) for b in BEDS for sp in specs}
    oreg = {(b, a): np.array([R[b, s]["owned"][a] - R[b, s]["oracle"] for s in EVAL]) for b in BEDS
            for a in ("BOCPD", "GLOBAL-o1", "GLOBAL-o2", "EMA")}
    best_owned = {b: np.min(np.stack([oreg[b, a] for a in ("BOCPD", "GLOBAL-o1", "GLOBAL-o2", "EMA")]), 0) for b in BEDS}
    null_births = sum(R["NULL", s]["eras"] - 1 for s in EVAL)
    SB = ("REC", "CYC", "CYC-S", "NOREC")
    L = [f"# {TAG} tournament results (fresh seeds {EVAL[0]}-{EVAL[-1]})", "", f"Picks (tuned on {TUNE_BEDS}): "
         + ", ".join(f"{sp[0]} alpha={sp[1]}" + (f" p={sp[2]}" if sp[2] is not None else "") for sp in specs), "",
         "| decider | " + " | ".join(BEDS) + " | score | G1 | G2 | G3 |", "|---|---|---|---|---|---|---|---|---|---|"]
    gated, scores, avg = [], {}, {}
    for sp in specs:
        g1 = int((reg["REC", sp] <= 0.5 * best_owned["REC"]).sum()) >= 8
        g2 = reg["NOREC", sp].mean() <= 1.25 * oreg["NOREC", "BOCPD"].mean()
        g3 = null_births <= 2
        avg[sp] = np.mean([reg[b, sp] for b in SB], 0)
        scores[sp] = float(avg[sp].mean())
        if g1 and g2 and g3:
            gated.append(sp)
        L.append(f"| {sp[0]} | " + " | ".join(f"{reg[b, sp].mean():.4f}" for b in BEDS)
                 + f" | **{scores[sp]:.4f}** | {'✓' if g1 else '✗'} | {'✓' if g2 else '✗'} | {'✓' if g3 else '✗'} |")
    L.append("| best owned | " + " | ".join(f"{best_owned[b].mean():.4f}" for b in BEDS) + " | | | | |")
    soft = picks["SOFT"]
    chal = min(gated, key=scores.get) if gated else None
    wins = int((avg[chal] < avg[soft]).sum()) if chal else 0
    winner = chal if chal is not None and (chal == soft or wins >= 8) else soft
    hard = picks["HARD"]
    L += ["", f"NULL births (shared bookkeeping): {null_births}. HARD answers with one expert on "
          f"{np.mean([R[b, s]['single'][hard] for b in SB for s in EVAL]):.1%} of steps.",
          f"Challenger: {chal[0] if chal else 'none'}; beats SOFT on {wins}/10 seeds (bed-averaged, paired).",
          f"**Winner: {winner[0]}** (alpha={winner[1]}" + (f", p={winner[2]}" if winner[2] is not None else "") + ").", ""]
    open(f"RESULTS_{TAG}.md", "w").write("\n".join(L))
    json.dump({"picks": {k: list(v) for k, v in picks.items()}, "winner": list(winner),
               "rows": [{"bed": b, "seed": s, **{k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v)
                                                for k, v in o.items()}} for (b, s), o in R.items()]}, open(f"results_{TAG.lower()}.json", "w"),
              default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()

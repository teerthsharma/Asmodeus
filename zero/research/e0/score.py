"""Score E0 results.json against the frozen bars B1-B4 (PROGRAM.md section 6). Writes RESULTS.md."""
import json

import numpy as np

R = json.load(open("results.json"))
OWNED = ["GLOBAL-o1", "GLOBAL-o2", "EMA", "BOCPD"]
F = R["fcpa_pick"]
ARMS = ["ORACLE"] + OWNED + ["FCPA-HARD", "FCPA-SOFT"]


def get(kind, arm, field="loss"):
    rows = sorted((r for r in R["rows"] if r["kind"] == kind and r["arm"] == arm), key=lambda r: r["seed"])
    return np.array([r[field] for r in rows], dtype=object if field in ("births", "delay") else float)


lines = ["# E0 results", "", f"FCPA arm chosen on tuning seeds: **{F}**. Picks (hp..., tuning mean loss): `{R['picks']}`", ""]
reg = {}
for kind in ("REC", "NOREC", "NULL"):
    orc = get(kind, "ORACLE")
    lines += [f"## {kind}", "", "| arm | mean loss | regret vs oracle (mean ± sd) | re-entry excess |", "|---|---|---|---|"]
    for a in ARMS:
        l = get(kind, a)
        reg[kind, a] = l - orc
        re = ""
        if kind == "REC":
            re = f"{np.mean(get(kind, a, 'reentry') - get(kind, 'ORACLE', 'reentry')):.4f}"
        lines.append(f"| {a} | {l.mean():.4f} | {reg[kind, a].mean():.4f} ± {reg[kind, a].std():.4f} | {re} |")
    for a in ("FCPA-HARD", "FCPA-SOFT"):
        ne = get(kind, a, "n_eras")
        d = np.concatenate([np.array(x, float) for x in get(kind, a, "delay")]) if kind != "NULL" else np.array([])
        lines.append(f"\n{a}: eras per seed {ne.astype(int).tolist()}" + (f"; switch delay median {np.median(d):.0f}, p90 {np.percentile(d, 90):.0f}" if d.size else ""))
    lines.append("")

best_owned = np.min(np.stack([reg["REC", a] for a in OWNED]), axis=0)
b1_seeds = reg["REC", F] <= 0.5 * best_owned
B1 = b1_seeds.sum() >= 8
ratio_rec = np.mean(reg["REC", "BOCPD"]) / np.mean(reg["REC", F])
ratio_norec = np.mean(reg["NOREC", "BOCPD"]) / np.mean(reg["NOREC", F])
B2 = ratio_rec > ratio_norec and np.mean(reg["NOREC", F]) <= 1.25 * np.mean(reg["NOREC", "BOCPD"])
null_births = int(sum(get("NULL", F, "n_eras") - 1))
rec_ok = int(sum(get("REC", F, "n_eras") <= 6))
B3 = null_births <= 2 and rec_ok >= 8
re_f = np.mean(get("REC", F, "reentry") - get("REC", "ORACLE", "reentry"))
re_b = np.mean(get("REC", "BOCPD", "reentry") - get("REC", "ORACLE", "reentry"))
B4 = re_f <= 0.5 * re_b
v = lambda b: "**PASS**" if b else "**FAIL**"
lines += ["## Bars (frozen)", "",
          f"- B1: {F} regret ≤ 0.5 × best owned on {int(b1_seeds.sum())}/10 seeds (need ≥ 8): {v(B1)}. "
          f"Mean regret {np.mean(reg['REC', F]):.4f} vs best owned {best_owned.mean():.4f} (ratio {np.mean(reg['REC', F]) / best_owned.mean():.2f}).",
          f"- B2: BOCPD/FCPA regret ratio REC {ratio_rec:.2f} vs NOREC {ratio_norec:.2f}; NOREC FCPA/BOCPD {np.mean(reg['NOREC', F]) / np.mean(reg['NOREC', 'BOCPD']):.2f} (need ≤ 1.25): {v(B2)}.",
          f"- B3: NULL spurious births {null_births} (need ≤ 2); REC seeds with ≤ 6 eras {rec_ok}/10 (need ≥ 8): {v(B3)}.",
          f"- B4: re-entry excess FCPA {re_f:.4f} vs 0.5 × BOCPD {0.5 * re_b:.4f}: {v(B4)}.", ""]
open("RESULTS.md", "w").write("\n".join(lines))
print("\n".join(lines))

# TT decider tournament: verdict (E0d, E0e; post-result)

Written 2026-10-13. Bars: `E0d_PREREG.md`, `E0e_PREREG.md` (sha256 in `e0/FREEZE.txt`). Numbers: `e0/RESULTS_E0d.md`, `e0/RESULTS_E0e.md`. Code: `e0/deciders.py`, `e0/run_e0d.py`.

## Bottom line

- **Winner: MOM, the momentum decider.** The TT's weight over eras (the chronal fluid density w_t) moves along a learned era-to-era kernel instead of spreading uniformly. This is Seal's "momentum … in a dimension of liquid".
- It beat the incumbent SOFT on **10/10 fresh seeds** (40–49), bed-averaged and paired, and passes every gate. Score 0.0486 vs 0.0515 (−5.6%).
- Gains are largest where the eras follow a path: CYC −7%, CYC-S −6%, NOREC −8%, REC −2%.

## Round results

| Decider (lemma) | Round 1, seeds 30–39 | Round 2, seeds 40–49 | Read |
|---|---|---|---|
| SOFT (L14′, fixed share) | 0.0541, fails G2 | 0.0515, all gates | the incumbent |
| **MOM (D3 Markov law over eras)** | **0.0508** (best), fails G2 at 1.28× | **0.0486, all gates, 10/10 vs SOFT** | **winner** |
| MOM+GB | n/a | 0.0489, all gates | ties MOM; the extra knob buys nothing |
| GBAYES (tempered likelihood) | 0.0539, all gates, 7/10 vs SOFT | 0.0523 | no gain |
| TEMP (L4 temperature) | 0.0533, fails G2 | 0.0516, fails G2 | no gain |
| CTC (L32 Hilbert fixed point) | 0.0536, fails G2 | 0.0520, fails G2 | **adds nothing as an answer-time reweighting; L32 predicted it** |
| HARD (L5, top expert alone) | same as SOFT; alone on 1.4% of steps | 0.0516 ≈ SOFT; **alone on 45.8% of steps** | **a compute result:** about half the steps need one era, at no loss |

Scores are the mean regret vs oracle over REC, CYC, CYC-S and NOREC, in nats/token. Owned baselines on the same seeds: REC 0.303, CYC 0.328, CYC-S 0.425, NOREC 0.016 (BOCPD).

## Why round 2 exists (disclosed)

Round 1's tuning objective left out NOREC. It pushed every decider to α = 0.01, which broke gate G2 for five of six deciders, the incumbent included. Round 1's frozen rule still produced a verdict: SOFT stays, because GBAYES won only 7/10.

Round 2 changed only the tuning objective (adding NOREC) and added one combined candidate. It was preregistered before its result, on fresh seeds 40–49. MOM was already the best score in round 1, so round 2 confirms a lead rather than discovering one.

## What it means

- **Seal's momentum is measurable.** The density w_t is not just where the TT is. Where it came from predicts where it goes, and using that pays on every bed with structure. It costs nothing where there is none: REC switches uniformly, and MOM still ties or wins there.
- **The CTC loop is not the TT's decider.** As a contraction it cannot sharpen weights (L32(b)), and measured it never beat the plain filter. It keeps its role as the provable return trip, not the answer.
- **Hard jumps are a compute tool.** With a 0.99 threshold, nearly half the steps can be answered by one era with no measurable loss. This is the "lazy compute" claim of ASMODEOUS, seen here on eras.

## Limits

- All beds are synthetic order-1 Markov switching streams built to favour eras. Language and in-context learners (E1) are untested.
- MOM's kernel Q is learned from switch counts with a Dirichlet(1) prior and a fixed 50/50 split between Q and uniform. The split was not tuned.
- "Proven" in Seal's sense (lemma + Lean) is still open for L38–L40 and for MOM itself. The candidate is L41: MOM is an HMM forward filter with a learned transition matrix, so its regret is bounded by the fixed-share bound with the uniform switch prior replaced by Q.

## What goes into main

`zero/research/`: the program, all preregistrations and freezes, verdicts, the numpy code, tests and results. MOM is the TT decider of record, on top of the E0c era bank (BOCPD novelty expert, global Bayes-factor merge, minimum evidence).

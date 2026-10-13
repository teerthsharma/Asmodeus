# E0d preregistration: the TT decider tournament (pre-result)

Filed 2026-10-13. Seal: "proceed in R&D way, try all theorems, winner gets into main." Every candidate is a way for the TT to move its weight w_t over the era bank (the chronal fluid density, Addendum 2). Each one rests on a lemma already in the record. The era bank, births, merges and BOCPD expert are shared and fixed at the E0c winner (δ = 0.05, w = 32, λ = 0.5, global merge, n_min = 64, hazard 1/400). Only the decider differs.

## Candidates (`e0/deciders.py`)

| ID | Decider | Lemma it rests on | Tuned grid |
|---|---|---|---|
| D0 SOFT | fixed share + Bayes step (incumbent, E0c) | L14′ forward filter; "after multiplication and normalisation" | α ∈ {1e-3, 3e-3, 1e-2, 3e-2} |
| D1 MOM | half the shared mass moves along a learned era-to-era kernel Q (counts of observed switches) | D3 Markov law over observer outcomes; note: "momentum … in a dimension of liquid" | α, same grid |
| D2 CTC | answer with the fixed point u* = G(u*), G(u)_i ∝ v_i·exp(−c·|Y_i − Σu_kY_k|²), c = 2q/Δ² | L32(e) Hilbert contraction, L22H Lean | α × q ∈ {0.3, 0.8} |
| D3 TEMP | answer with w^{1/T} | L4 post-selection temperature | α × T ∈ {0.5, 0.75, 1.5} |
| D4 HARD | answer with the top expert alone when its weight ≥ θ, else the blend | L5 spirit (certified top-1 for compute) | α × θ ∈ {0.9, 0.99} |
| D6 GBAYES | tempered likelihood w ← w·p^η | generalised Bayes | α × η ∈ {0.5, 0.8} |

## Beds and seeds

- Beds:
  - **REC**, **NOREC** and **NULL**, as in E0.
  - **CYC**: 4 laws, segments 50 + Geometric(mean 350). The next law is the cyclic successor with probability 0.9, otherwise uniform among the rest.
  - **CYC-S**: CYC with segments 20 + Geometric(mean 100).
- Tuning: each decider's grid point is chosen by the mean of (mean loss on REC seeds 100–104, mean loss on CYC-S seeds 200–204).
- Evaluation: **fresh seeds 30–39** on all five beds. The owned arms (GLOBAL-o1, GLOBAL-o2, EMA, BOCPD) and ORACLE are rerun with their E0 picks.

## Gates (each decider must pass all three)

- **G1:** REC regret ≤ 0.5 × best owned on ≥ 8/10 seeds.
- **G2:** NOREC regret ≤ 1.25 × BOCPD (mean over seeds).
- **G3:** NULL births ≤ 2 in total. The bookkeeping is shared, so G3 is common to all deciders.

## Winner rule

- **Score** = the mean, over REC, CYC, CYC-S and NOREC, of the decider's mean regret vs ORACLE.
- Among the gated deciders, the lowest score is the challenger.
- The challenger replaces SOFT only if its bed-averaged regret is lower than SOFT's on ≥ 8 of 10 seeds (paired). Otherwise SOFT stays the winner.
- Reported, no bar: per-bed regrets; the share of steps HARD answers with one expert (a compute proxy).

## What "into main" means

The winner's code (bed, arms, deciders, runners, tests) and these documents land in `zero/research/` through a PR to `main`. FCPA files and PREREG.md are not touched.

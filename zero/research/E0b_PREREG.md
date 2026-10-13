# E0b preregistration (post-E0, pre-E0b-result)

Filed 2026-10-12 after E0 scored (`e0/RESULTS.md`): B1 and B4 passed; B2 failed (NOREC: FCPA-SOFT regret 3.71× BOCPD) and B3 failed (REC: 7–11 eras for 4 laws; NULL: 0 spurious births). E0's frozen bars and verdict are not edited.

## Two diagnoses (from E0 numbers, eval seeds 0–9)

1. **B2: hard commitment costs when nothing recurs.** BOCPD integrates over every change time; FCPA's novelty expert is a 32-token window, and its era only re-anchors after the alarm (median 21 tokens). Prediction: adding BOCPD as the novelty expert in the forward filter makes NOREC ≈ BOCPD while REC keeps the bank's recurrence gain.
2. **B3: births duplicate laws.** The address step judges a short, possibly mixed segment, and an era born from it is never merged back. Prediction: a Dirichlet–multinomial Bayes-factor merge (merge eras i, j when log ML(C_i + C_j) > log ML(C_i) + log ML(C_j)), run after every alarm, brings the era count to ≈ K. This is L26's merge step with a Bayes-factor value instead of Shapley.

## Arms

All use E0's tuned δ = 0.05, w = 32, λ = 0.5; α re-tuned per arm on tuning seeds 100–104 (REC) over the E0 grid {1e-3, 3e-3, 1e-2, 3e-2}.
- SOFT (E0 arm, reference).
- SOFT+M: merge only.
- SOFT+B: BOCPD (E0's tuned hazard 1/400) is an expert in the fixed-share filter, beside the window model.
- **SOFT+BM: both (primary).**

## Seeds

E0 eval seeds 0–9 have been seen. E0b evaluates on **fresh seeds 10–19** for REC, NOREC, NULL. Tuning stays on 100–104.

## Bars (primary arm SOFT+BM)

- B1′: REC regret ≤ 0.5 × best owned (E0 owned set and picks, rerun on seeds 10–19) on ≥ 8/10 seeds.
- B2′: NOREC regret ≤ 1.25 × BOCPD regret (mean over seeds).
- B3′: NULL spurious births summed over 10 seeds ≤ 2; REC final era count ≤ 6 on ≥ 8/10 seeds.
- B4′: REC re-entry excess ≤ 0.5 × BOCPD's.

Verdict: all four pass → SOFT+BM is the E0 architecture going into E1. Any failure is reported as is; no further repair is run on seeds 10–19.

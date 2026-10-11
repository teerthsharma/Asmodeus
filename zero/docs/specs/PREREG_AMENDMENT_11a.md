## Amendment 11a (pre-result)
Filed 2026-10-11, before any FCPA evaluation result exists. Appended to Amendment 11 (`PREREG_AMENDMENT_11_DRAFT.md`, frozen in `FREEZE.md`); the draft is not edited. Where this text and the draft differ on C2, this text wins. The argument below uses MAC counts only; no error number enters it.

**What changes.** Only the compute rule C2 of the primary claim and the clauses of the draft that cite it (the P1/P2 preconditions, Kills K1 and K2, the revert rule). P1 to P4, C1, the beds, the seeds, the bootstrap and every constant are unchanged.

**C2 (draft, withdrawn).** Each baseline is trained, per seed, to a budget equal to FCPA's total training MACs (steps = ⌊B_FCPA / MACs per step⌋).

**C2′ (replaces C2).** Each baseline trains for its pinned budget.
- **Global:** one closed-form fit on the full Train split (all 127 × 128 consecutive pairs of the 128 Train trajectories, 51,072 pairs). The fit has no epoch count.
- **ASMODEOUS top-m:** the pinned 5,000 Adam steps at batch 64 (the draft's evaluation line), after lr and m are chosen on tuning seed 3 as in the draft. Its closed-form readout probe is fitted on up to 20,000 Train pairs.
- Training MACs of all three arms are **reported** per seed and are **not equalised**. C2′ is a reporting rule: it has no pass or fail value and no entry in the verdict. Wherever the draft says "C1/C2 fail" (preconditions of P1 and P2, K1, K2), read "C1 fails".
- **C1 (inference MACs) is unchanged:** FCPA's mean inference MACs/step must be at most each baseline's.
- The original rule is kept as a diagnostic. The runner flag `--c2 strict` trains the baselines to B_FCPA; its output is reported beside the primary run, labelled DIAGNOSTIC, and does not enter the verdict.

**Why (arithmetic only).** FCPA trains in closed form from additive statistics, so its training cost is linear in the number of pairs with a small constant. The baselines' costs per pair are larger by one to two orders. This is a property of the MAC counts at the pinned sizes (D = 90 encoded dimensions, 51,072 Train pairs, 8 eras, N = 16, d_c = 8, window 20), independent of any error:

| quantity | value | derivation |
|---|---|---|
| FCPA training, per pair | 54,905 MACs | statistics 32,204 + window test, Gaussian inverse and era solve 11,405 + per-fit finalisation 11,297 (sums to 54,905 after rounding) |
| B_FCPA | 2.804e9 MACs | 51,072 pairs × 54,905 |
| Global width for 83,232 parameters | 834 random features (F = 924) | (90 + w + 1) × 90 ≈ 83,232; 83,250 parameters |
| Global, per pair | 1,011,996 MACs | features 90 × 834, Gram F², cross-moment F × 90 |
| Global, one solve | 8.68e8 MACs | F³ + F² × 90 |
| Global on the full Train split | 5.26e10 MACs = 18.7 × B_FCPA | 51,072 × 1,011,996 + 8.68e8 |
| Pairs Global can afford under C2 | 1,912 of 51,072 (3.7%) | (B_FCPA − 8.68e8) / 1,011,996 |
| ASMODEOUS step (K = 8, m = 2, hidden 64, batch 64, window 8) | 2.01e7 MACs | forward 13,084 × 3 (backward at 2× forward) × 64 × 8 |
| ASMODEOUS, 5,000 steps | 1.00e11 MACs = 35.8 × B_FCPA | 5,000 × 2.01e7 |
| Steps ASMODEOUS can afford under C2 | 139 of 5,000 (2.8%) | ⌊2.804e9 / 2.01e7⌋ |

Under C2 the comparison stops measuring architectures and measures how much data and how many updates each method is charged for. Global would train on 3.7% of the Train split and ASMODEOUS for 2.8% of its pinned steps. FCPA's advantage in training cost (18.7× against Global, 35.8× against ASMODEOUS) is a MAC fact that the report states beside P1 and P2. It is not a reason to cap the baselines' training. P1 and P2 test accuracy at the baselines' pinned budgets. The FCPA row uses the analytic model `run_primary.fcpa_train_macs`; the baseline rows use `baselines.global_macs` and `Asmodeous.active_flops`. All are recomputed per seed in the run, and the table is the sizing at the pins.

## Limits
- The FCPA cost is an analytic MAC model of the estimator internals, not a measured count. The starvation conclusion holds for Global while FCPA's training cost per pair stays below 1.03e6 MACs and for ASMODEOUS while it stays below 1.97e6 MACs; the model gives 5.49e4.
- The table is sized at 8 eras. A different final era count changes B_FCPA and the Global width; per-seed values are in the results.
- C2′ removes a constraint on the baselines, so it can only make P1 and P2 harder for FCPA than the draft's C2 did. The diagnostic run reports the C2 outcome for comparison.

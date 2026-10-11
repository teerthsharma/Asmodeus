# AUDIT 2 (Health Inspector, FCPA build, pass 2)

Audited 2026-10-11 against worktree `new-session-d650db` (HEAD be1b269, FCPA tree untracked), Windows, Python 3.11.9.
Scope: `zero/house-events.jsonl` lines 1-817 for the finding-binding check; lines 591-817 for test claims; Nurse N10 ledger flags C1, C2, C3, C7, C9, C11, C25, C27, C30, C31, C32. The Inspector rules on binding only, never on correctness. No code was edited.

## Summary

| Check | Audited | Clean | Struck |
|---|---|---|---|
| Finding events have a prior named RED (whole log) | 200 | 28 | 172 |
| Test-count and test-status claims, lines 591-817 | 29 | 24 | 5 |
| N10 ledger flags | 12 | 4 | 8 |

## 1. Test reruns

`python -m pytest zero/fcpa/tests -q -p no:cacheprovider`, run twice from the repo root, one process at a time:
- Run 1: **373 passed, 4 skipped, 2 xfailed**, exit 0, 147.5 s.
- Run 2 (`-rA`): **373 passed, 4 skipped, 2 xfailed**, exit 0, 135.8 s. No estimator failure, so no extra estimator rerun was needed for these two runs.
- After log lines 800-814 landed, a third run of `test_estimator.py` plus `test_launch_guards.py` gave **84 passed, 3 failed, 1 skipped**. The 3 failures are the intended REDs `test_c_full_run_requires_exactly_the_five_eval_seeds[duplicates|subset3|subset4]` (Chase: red 3, green 14). `test_estimator.py` went from 68 to 70 passing with the two energy tests.
- Skips: 2 sklearn array-api (`SCIPY_ARRAY_API` unset), 2 `test_estimator_ns3.py` tests gated on `FCPA_NS3_CACHE`. Xfails: `test_H1_crossing_pt[0.05]`, `test_H2_nll_margin_vs_lda_and_bigram[2]`.
- `zero/lean/check_no_sorry.py` (nurse, haiku): `check_no_sorry: 0 sorry, 0 cited-sorry`, exit 0. 0 `sorry` hits in 26 `FCPA/*.lean` plus `FCPA.lean`. `CITED.txt` absent (none needed).

Per file, run 2: bars 39, baselines 36, bounds 12, conformal 39, crossing_pt 2 (+1 xfail), ctc 15, differentiator 9, edetector 4, eras 10, estimator 68 (+1 skip), estimator_ns3 4 (+2 skip), hilbert 7, liquid 4, ns2d 13, observer 19, router 8, rsl 16, tt_attention 68 (+1 xfail, +1 skip).

| Claim | Log line | Rerun | Verdict |
|---|---|---|---|
| 75 passed bars/baselines (correction) | L762 | `-k "bars or baselines"`: 75/398 collected, 39 + 36 = 75 PASSED | clean |
| 76 passed | L761 | 75; corrected in-log at L762. `test_baselines_freeze3_tampered_hash_makes_the_runner_refuse` passes but has no prior named RED (L716 is FREEZE2, L748 is the lag stack) | **struck** |
| 145 passed, 1 skipped (rsl+observer+router+conformal+edetector+estimator) | L699-701 | 154 passed + 1 skip; the 9 estimator tests logged after L701 account for the difference | clean |
| check_estimator FCPARegressor 50 passed / 1 skipped | L589, L746 | 50 PASSED + 1 SKIPPED of 51 collected | clean |
| check_estimator AttentionTT 55 passed / 1 skipped (also ERRATA 16) | L702 | 55 collected = **54** PASSED + 1 SKIPPED; the skip was counted as a pass | **struck** |
| test_tt_attention 68 passed, 1 xfailed | L702 | 68 PASSED + 1 XFAIL | clean |
| edetector H3, 4 passed | L687, L690 | 4 PASSED; RED L685 precedes | clean |
| hilbert, 7 passed | L695, L698 | 7 PASSED; file-level RED L691 precedes | clean |
| bars 38; bars+baselines 56 / 65 / 73 | L598, L711, L721, L757 | superseded by L762; REDs L591, L627, L716, L748 precede | clean |
| H1 numbers 0.781 / 0.944 / 0.984, JS 0.821, sinTheta 60/60 | L688 | suite stdout reproduces each value; sinTheta 20/20 at each of 3 eps | clean |
| ctc L22* residual / bias / orbit (follow-up to AUDIT_1 #3) | L616-618 | 15 PASSED in test_ctc; REDs L599-601, mutation REDs L640-645 | clean |
| AUDIT_1 #1b unbound lemma tests | L593 | all 16 now carry a mutation RED before re-green (L602-L639); AUDIT_1 said 17 but listed 16 | clean (see C30) |
| 17 more core lemma tests, mutation-bound | L646-L683 | each has a mutation RED then green; all PASS | clean |
| estimator rewrite, eras and estimator tests | L712-L785 | REDs L712-715, L722-723, L751-753, L758-759 precede; all PASS | clean |
| energy tests | L813-L814 | REDs L809-810 precede; PASS | clean |
| `test_H2_lag_stack_replacement_probe` green | L705 | no such test in `zero/fcpa/tests`; no RED. Labelled exploratory, but logged as a green test | **struck** |
| ns2d rfft2 equivalence green | L710 | no named test in `test_ns2d.py`; no RED; cannot be rerun | **struck** |
| P3 stays RED (0.268) | L736, L746 | superseded by L803; serves as the RED that binds L803 | clean |
| `test_ns3_P3_identification_at_least_0_9...` green (P3 0.928) | L803 | skipped by default; gated rerun: PASSED (FCPA_NS3_CACHE set, 6 passed in 1619 s) | clean |
| `test_ns3_lag_stack_one_step_error_is_at_most_half_of_persistence` green | L804 | RED L753 precedes; skipped by default; gated rerun: PASSED (FCPA_NS3_CACHE set, 6 passed in 1619 s) | clean |
| ns3 one-step, E50, encoder, dev-bed tests | L727-L734, L780-L783 | 4 PASSED; mutation REDs precede | clean |
| check_no_sorry, 0 sorry, 26 files | L749 | exit 0, 0 sorry; the prior red is AUDIT_1 L596 (4 sorry). AUDIT_1 #4 is resolved | clean |
| L22H2 `hilbert_ctc_fixed_point` | L755 | theorem at `FCPA/L22H2_HilbertFixedPoint.lean:254`, 0 sorry; open state recorded at L750. `lake build` not rerun | clean |
| `test_launch_guards.py` red 3 / green 14 | L800, L812 | 14 PASSED, 3 FAILED, as claimed | clean |
| "354 passed" | none | no log line makes this claim | clean (nothing to audit) |
| Contradiction: 76 vs 75 | L761 / L762 | corrected in-log; the rerun confirms 75 | clean |
| Contradiction: L750 "Banach step open" vs L754 "gap closed" | L750 / L754 | a supersession, not a contradiction | clean |
| Contradiction: AttentionTT 55 vs 54 | L702, ERRATA 16 | see the check_estimator row above | **struck** |

## 2. Finding binding (whole log, 200 finding events)

The rule is mechanical: a finding is bound only when its `bound_by` (or `test` / `name`) field names a test, and a `test` (or `red`) event with status red names that test, or names its file at file level, earlier in the log. The column "lane RED" in the raw output records whether the same agent had any earlier RED. Earlier Inspector passes (L82, L87) accepted that lane-level RED as binding, but it does not meet this rule.

- **Clean, 28:** L125-131, 145, 243, 249, 256-258, 339, 387, 410, 456, 471, 484, 487, 489, 498, 509, 511, 525, 537, 540, 547. Most are bound by the Wilson `tests/test_facts.py` RED at L101. Four are bound by file-level REDs: L55 for `test_run.py`, L30 for `test_laws.py`, L447 for `test_zeno_probe.py` and L480 for `test_qflm_adapter.py`.
- **Struck, cites a test with no prior RED (4):** L425 (`test_qflm_stability.py`), L501 (`test_qflm_telemetry.py`), L530 (`test_renorm_psi_cuda_graph_capture_reports_zero_row`): chase-nurse logged only greens, with the mutation REDs inside the green text. L811, Chase CONFIRMED risk c (`test_c_full_run_requires_exactly_the_five_eval_seeds`): the RED reproduces on rerun (3 failed), but no red test event precedes the finding.
- **Struck, `bound_by` is not a test (52):** literature, coordinator rulings, code line references and PREREG merges: L132-140, 155, 197, 216, 221, 233, 234, 238-241, 248, 260, 268, 270, 274, 275, 277, 288, 306, 330, 331, 333, 342, 345, 385, 386, 388, 392, 411, 423, 439, 442, 448, 453, 463-465, 467, 475, 479, 483, 488, 497.
- **Struck, no `bound_by` at all (116):** L19-45, 52, 53, 57, 62-71, 90-95, 108-124, 142-156, 177-226, 231-250, 264-307, 319-377, 404-462, 492, 506, 507, 510, 527, 533, 534, 543. The FCPA range adds **L750** (Lean Tier-A closed), **L760** (Prereg 11b frozen), **L807** (N7 CTC is a no-op in the live path) and **L808** (N6 Lean weaker than spec). Many of these are process notices or literature notes that no test could bind, and they are struck under the rule as given.

## 3. Nurse N10 ledger flags (binding only)

| Flag | Claim | Source | Ruling |
|---|---|---|---|
| C1 | e3 product delay 2191 vs 1848, same sample | e3_edetectors.md:23-24 | **struck.** Nurse scratchpad only; no log test; no estimator named. Neither number is bound. |
| C2 | H3 median delay 23 vs product 2392 | L687, ERRATA 14 | **clean** for the test's bars. `test_H3_delay_after_shift_beats_product_martingale_at_long_null_prefix` (RED L685, PASS on rerun) fixes the bed: uniform p, post-change p**3, delta 0.05, kappa 0.5, rng 1, n 500. It asserts median ≤ 40, old ≥ 1000 and old > 10× new. The exact 23 and 2392 are not asserted, so as values they are unbound. e3 uses a different shift (mu = 1 on the \|x\| score), so C2 compares two beds. |
| C3 | martingale property at the wired kappa 0.5 | L699, rsl.py:133 | **struck.** The Monte-Carlo martingale test runs only kappa in (0.8, 0.9). Kappa 0.5 gets the M_0 = 1 check and the Ville false-birth test, not the martingale check. No test covers small-shift (mu = 0.5) detection. |
| C7 | ERRATA 13 values 0.9997, 0.9984, 0.885 (4000 psi), 0.944 (8000 psi) | ERRATA.md:17 | **struck.** No log line, test or e-file holds them; the grep hits in e3/results.json and n1 are unrelated quantities. Only 0.9999 has a source (L688). |
| C9 | pilot Global within the matched budget | PILOT_DISCLOSURE.md:12-13 | **struck** for the pilot: 1.805e9 against 1.402e9 MACs (×1.29), and afforded 527 < used 925. On the current code `test_baselines_global_budget_is_respected` passes (file RED L627), so the current-code claim is bound. |
| C11 / C31 | closed-loop E50 not worse than persistence | L731-732, L803-805 | **struck.** `test_ns3_closed_loop_E50_is_not_worse_than_persistence` **runs and passes** on the current code, but it uses `_fit` (no lag stack, no NS2DLaw) on the 32-trajectory dev bed, not the L803 configuration. No test checks E50 for the lag-stack + NS2DLaw estimator. L803-805 log E50 1.329 vs 0.6585 inside green events that no test backs. |
| C11 | one-step lag stack beats persistence on the dev bed | L805 | **clean.** The test body is one-step only (mutation REDs L780-782, PASS). The L805 log text quotes full-bed numbers that this dev-bed test does not compute (mis-attributed text). |
| C25 | L711 smoke seed | L711 | **struck.** No seed stated. PILOT:21 says the first-built smoke used eval seeds 0 and 1; L721 and L757 use 7001/7002. |
| C25 / C27 | seed-1 smoke disclosure | PILOT_DISCLOSURE.md:21-28 | **clean.** Disclosed before evaluation. `test_baselines_seed_guard_keeps_development_off_eval_seeds` passes. Seed 1 is outside the seed-0 allowance but is disclosed. |
| C27 | seeds [0], [1], [2] at L692-694 | L692 | **clean.** These are rng seeds for a synthetic stream (`test_rsl.py:280`), not the FCPA bed. REDs L692-694 precede the greens at L699-701. |
| C30 | AUDIT_1 unbound count 17 vs its list of 16 | L593 | **struck.** The list sums to 16 (9 + 3 + 4); confirmed independently. |
| C32 | Lean N6 (5 weaker, 11 missing) vs CHANGES:43 (9 not in v1) | L808 | **struck.** L808 has no `bound_by` and no RED. The counts and contents disagree with CHANGES.md:43, and no check binds either list. |

## Limits

The finding rule is applied literally. That strikes 168 literature, ruling and process notices that no test could ever bind, and several of them were ruled clean under the lane-level rule in Inspector pass 1 (L82, L119-122). The log is live: it grew from 789 to 817 lines during this audit, and lines after 817 are not covered. The binding matcher is a substring match on test names, plus file-level REDs on the named test file. Mutation REDs that appear only inside green-event text (chase-nurse) are not counted. `lake build` was not rerun. The `FCPA_NS3_CACHE`-gated tests needed a full 256 × 400 bed, generated in the session scratchpad.

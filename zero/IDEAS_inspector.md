# IDEAS_inspector: threats to the verdict's validity, and the gate for each

Lens: audit and evaluation. Each item gives the risk to a YES/NO/VOID line in `VERDICT.md`, the source, and the rule (for PREREG) or test (for `test_laws.py`) that closes it.
Sources were fetched by research nurses on 2026-10-04 unless marked [not fetched]. Lines marked [derived] are arithmetic, checked by the Inspector.

## 1. The best of three candidates inflates YES
- **Risk:** the scoring rule says "the verdict line per claim names whichever candidate passes."
  - 14 claims (C1–C5, with C4 split into T4a and T4b, TO1–TO7, T-EV) × 3 candidates = 42 claim–candidate cells.
  - Most cells are also a "≥ k of M beds" vote.
  - Taking the max over candidates is a selection step. With 3 candidates, each with a false-pass rate α, the chance that some candidate passes a null claim is about 3α.
- **Source:** Wikipedia, "Multiple comparisons problem" (https://en.wikipedia.org/wiki/Multiple_comparisons_problem). Wikipedia, "Holm–Bonferroni method" (https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method).
- **PREREG rule:**
  - Declare the confirmatory family now as {claim × candidate}, m = 42, minus N/A cells. Write it as an explicit list.
  - Any p-valued cell is judged by Holm at family-wise α = 0.05: reject H(k) if P(k) ≤ α/(m+1−k).
  - Cells with a fixed threshold and no p-value must state their per-cell false-pass rate under the null, or be labelled DESCRIPTIVE.
- **Test:** `test_family_declared`. PREREG.md must contain a machine-readable family table: one row per cell, with columns `claim`, `candidate`, `scored|NA`, `test`, `alpha`. Its row count must equal claims × candidates.

## 2. Arms and beds added later shrink or grow the family silently
- **Risk:** the candidate table already has a PENDING row (ASMODEOUS), and TO10–TO18 are "PENDING coordinator adoption."
  - A cell added after the data, or a dropped arm, changes m.
  - Not counting dropped attempts understates the false-pass rate.
- **Source:** Wikipedia, "Forking paths problem" (https://en.wikipedia.org/wiki/Forking_paths_problem). Gelman & Loken 2013 [not fetched; cited via Wikipedia].
- **PREREG rule:**
  - The family is frozen at the PREREG commit.
  - A cell added later is EXPLORATORY by construction and cannot reach the verdict line.
  - A dropped cell is still listed, with the reason.
- **Test:** `test_prereg_read_only`, already live. Any edit to the frozen text fails, except an appended post-hoc section. To add: `test_verdict_cells_in_family`. Every cell named on a VERDICT.md line must appear in the committed family table.

## 3. Three seeds cannot support a significance claim
- **Risk:** T2, TO1–TO7 and T-EV run 3 seeds; T4a and T4b run 5.
  - For a paired exact sign-flip or Wilcoxon test the minimum two-sided p is 2/2ⁿ [derived]: 0.25 at n = 3, 0.0625 at n = 5. A 3-seed cell can never reject at α = 0.05.
  - Bootstrap CIs over 3 seeds have only C(5,3) = 10 distinct multiset resamples [derived]. They collapse to min..max.
  - Colas et al.: bootstrap-CI tests at n = 5 give about 10% false positives. They should not be used at n < 10.
- **Source:** Colas, Sigaud & Oudeyer, arXiv 1806.08295. Agarwal et al., arXiv 2108.13264 (3–10 runs are typical; percentile CIs are "good for as few as N=10"). Henderson et al., arXiv 1709.06560 (two groups of 5 seeds with identical settings differ significantly, e.g. TRPO t = −9.09, p = 0.0016).
- **PREREG rule:**
  - 3-seed bars are threshold rules, not tests. The word "significant" is banned at n < 6.
  - Every 3-seed cell reports all per-seed values and the sign count (3/3 or 2/3), plus the exact sign-flip p as a floor.
  - A 3-seed YES needs unanimous sign AND a minimum per-seed margin larger than the tuning-seed seed-to-seed spread.
  - Where a p-value enters the Holm family, the cell needs n ≥ 6 paired seeds; else it is DESCRIPTIVE.
- **Test:** `test_run_records_per_seed`. Every run JSON carries `per_seed: {seed: value}` and `n_seeds`. A record that reports only a mean fails.

## 4. A 3-seed mean against a threshold hides the worst seed
- **Risk:** "the 3-seed mean discontinuity ≤ 0.01" (K_first) and "3-seed mean ΔQ" (T-EV) pass when one seed fails badly and two are good.
- **Source:** Agarwal et al., 2108.13264 (IQM and probability of improvement exist because means over few runs are fragile). NeurIPS checklist Q7 (https://neurips.cc/public/guides/PaperChecklist): error bars must state what varies.
- **PREREG rule:**
  - Threshold bars over seeds use the worst seed, or "k of n seeds", stated per cell.
  - Means are reported beside them, not scored.
  - T-EV already uses the worst seed for its FLOP ratio. Apply the same rule to its ΔQ, and to the discontinuity in K_first.
- **Test:** none needed. The rule is checked at the verdict audit by recomputing from `per_seed`.

## 5. Null and neutral claims need equivalence tests, not "no difference found"
- **Risk:** several cells pass or fail on absence of an effect. One is "neutral error within +2% of single JEPA" (L-WALD neutral bed). Another is the C3 bar "lazy top-m within 1% rel of soft mixture over all K".
  - With 3 seeds, failing to find a difference is almost certain whatever the truth, so these cells pass by low power.
- **Source:** Lakens, "Equivalence Tests: A Practical Primer" (https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906). TOST against preset bounds (SESOI) has four outcomes: equivalent; different; both; undetermined.
- **PREREG rule:**
  - Every "within x%" bar is an equivalence claim. It passes only if the whole interval (per-seed range at n ≤ 5) lies inside ±x%.
  - Otherwise it scores UNDETERMINED, which is distinct from PASS, FAIL and VOID.
  - The bound x is the preregistered SESOI.
- **Test:** `test_verdict_vocabulary`. VERDICT.md lines use only YES / NO / VOID / UNDETERMINED / N/A.

## 6. VOID must be decided before the outcome is seen, from conditions blind to the candidate
- **Risk:** VOID rules exist: T4a when single is censored on ≥ 3/5 seeds; T5 when the true-graph executor ceiling is < 0.8; and bed-level VOIDs.
  - A VOID condition computed from data that includes the candidate's result is an escape hatch: a run that would have failed can be voided.
- **Source:** Center for Open Science, Registered Reports (https://www.cos.io/initiatives/registered-reports): acceptance is fixed at Stage 1 and the result is published regardless of outcome. Wikipedia, "Registered report".
- **PREREG rule:**
  - Every VOID condition must be a function of baseline arms, oracles or the bed only, never of a candidate's score.
  - Each must be evaluated and logged in the event log before the candidate's numbers for that cell are read.
- **Test:** `test_void_precedes_score`. For each VOID verdict, `house-events.jsonl` holds a `void_check` event for that cell, logged earlier than any `score` event for the same cell.

## 7. Selection on the eval seed or eval data leaks into the score
- **Risk:** τ, the Mapper cover, lr, ε/δ, the SIGReg λ (0.9) and λ_bal (0.3) are all tuned.
  - Tuning seed 3 and `test_tuning_seed_never_evaluated` exist.
  - The ε/δ calibration ran on "eval bed seed 1000". That is the eval bed, not the tuning bed (Appendix A §A.3).
  - Chase's λ and λ_bal sweeps were on smoke beds of unstated seed.
- **Source:** Kapoor & Narayanan, arXiv 2207.07048, leakage class L1.2 (pre-processing fit on train + test) and L1.3 (feature selection on train + test). Recht et al., arXiv 1902.10811: fresh test sets cost 3–15% on CIFAR-10. Most of the drop is distribution shift, not adaptivity, but the defence is still a held-out set.
- **PREREG rule:**
  - Every tuned quantity names its tuning bed seed and the number of configs tried.
  - ε/δ calibration on eval seed 1000 must be either redone on the tuning bed (2000 + 3), or declared label-free and arm-free. A geometry-only calibration that sees no arm's latent is not leakage, and the declaration must say so.
- **Tests:**
  - `test_mapper_baseline_fair`, already live, checks the cover's tune seed.
  - To add: `test_tuned_params_logged`. Every run record carries `tuned: {name: {value, tune_seed, n_configs}}`, and no `tune_seed` is in `seeds`.

## 8. Test suites that skip silently count as green
- **Risk:** `test_ogbench_bed.py` and `test_controls_t4b.py` call `pytest.importorskip("ogbench")`.
  - In the system Python they report "1 skipped" with exit code 5. Inside a combined run that is exit 0 ("17 passed, 2 skipped").
  - A gate run in the wrong environment would pass T4b's controls without executing them. This was observed in audit pass 2.
- **Source:** observed in house audit pass 2, 2026-10-04.
- **PREREG rule:** the full gate runs each suite in its declared environment. T4b uses `~/.venvs/ogbench`.
- **Test:** `test_gate_no_skips`. The sign-off gate script fails on any skip in a scored suite.

## 9. The compute budget leaks through retries and failed runs
- **Risk:**
  - L-BR matches active FLOPs and params per step.
  - A candidate rerun after a crash, a divergence or a "smoke" gets more total training compute and more chances (a fork in the path).
  - Taub logged CPU contention from other lanes during T7 timing. That makes µs/step depend on who else was running.
- **Source:** NeurIPS checklist Q8: total compute must include preliminary and failed runs. Bouthillier et al., arXiv 2103.03098: randomize every variance source; a single-point comparison gives about 10% false positives and about 75% false negatives.
- **PREREG rule:**
  - Every scored run is the first attempt for its (cell, seed). A rerun is allowed only for an infrastructure fault logged with its traceback, and both attempts are reported.
  - T7 timings run with no other GPU or CPU lane active. Machine load is recorded in the run record.
- **Test:** `test_one_attempt_per_cell`. `runs/` holds at most one scored record per (cell, candidate, seed). Extra attempts live under `runs/retries/`, each with a `fault` field.

## 10. Report every VOID, null and N/A in full, beside the positives
- **Risk:**
  - VERDICT.md is one line per test.
  - RESULTS.md could carry only the passing cells, and the reader sees YES lines with no denominator.
- **Source:** Rosenthal 1979, the file-drawer problem [not fetched; via https://septentrio.uit.no/index.php/SCS/article/view/8315]. COS Registered Reports: nulls are published with the same detail.
- **PREREG rule:**
  - RESULTS.md has one row per family cell (item 1), including NO, VOID, UNDETERMINED and N/A, with the same columns as passing rows.
  - The verdict line gives the denominator, e.g. "C1: YES for OBSERVER-ATLAS (1/3 candidates; 3/4 manifolds)".
- **Test:** `test_results_cover_family`. Every family row appears in RESULTS.md once the file exists.

## 11. Error bars must say what varies
- **Risk:** "bootstrap CIs" (T6, T4a's geometric-mean ratio) and "± over seeds" are ambiguous without stating the following:
  - what is resampled: episodes, seeds or points;
  - n;
  - whether the bar is SD or SEM.
- An episode-level bootstrap over 150 paired episodes (T5) ignores seed variance. It is much narrower than a seed-level interval.
- **Source:** NeurIPS checklist Q7. Pineau et al., arXiv 2003.12206 (checklist item: "clearly defined error bars").
- **PREREG rule:**
  - Every interval states its resampling unit and n.
  - For T5 and T4b: a stratified bootstrap resampling seeds, then episodes within seeds, as in Agarwal et al.'s stratified bootstrap. An episode-only interval is reported as secondary.
- **Test:** `test_run_records_per_seed` (item 3) is extended: any `ci` field carries `unit` and `n`.

## 12. Leakage audit sentence per bed (L1/L2/L3)
- **Risk:** the beds are synthetic, so the classic leakage classes look moot, but they map onto the beds:
  - L2, illegitimate features: does `obs` encode `state` trivially? The lift is a fixed MLP of state, which is fine by design. A render that prints the angle is not.
  - L3.2, non-independence: train and eval trajectories from the same seed stream.
  - T4b: the 20% held-out transitions must not share start cells with training. Thirteen's memoriser (planAll 0.321 / 0.237 / 0.419) shows held-out edges are partly predictable from train edges.
- **Source:** Kapoor & Narayanan, 2207.07048. Model info sheets (https://reproducible.cs.princeton.edu/).
- **PREREG rule:** Appendix A (beds) gets one sentence per leakage class per bed, stating why it is absent. T4b states the memoriser score as the leakage floor, and the bar is measured above it.
- **Test:** `test_lgiven.py`, already live, covers state, geo, meta, causality and the router. To add: `test_eval_split_disjoint`. Train and eval bed draws share no trajectory (the seed streams differ: s vs 1000 + s), checked by hashing the initial states.

## Limits
- Pineau et al.'s code-availability statistics came back garbled from the fetch and are not used.
- Gelman & Loken and Rosenthal were not fetched. They are cited through Wikipedia or secondary pages.
- The small-n p-value floors and resample counts are [derived], not fetched.
- None of the proposed tests is written yet. Items 1, 3, 6, 8 and 9 change PREREG text and need coordinator adoption before the PREREG commit.

# AUDIT 1 (Health Inspector, FCPA build)

Audited 2026-10-11 against worktree `new-session-d650db` (HEAD be1b269, FCPA tree untracked), Windows, Python 3.11.9.
Scope: `zero/house-events.jsonl` lines 554-590 (Builder-core, Builder-physics, Builder-model). No code was edited.

| # | Check | Verdict |
|---|-------|---------|
| 1a | Every claimed green test passes | clean |
| 1b | Every lemma test has a named red before its green | STRUCK (17 unbound) |
| 2 | FREEZE hashes | clean |
| 3 | Spot-check of 6 lemma tests | STRUCK (1 of 6 is weaker than its lemma) |
| 4 | Lean tree and `sorry` count | STRUCK (4 sorry, gate exits 1) |
| 5 | No evaluation results before the freeze | clean |

## 1. Test re-run and red/green binding

`python -m pytest zero/fcpa/tests -q -rA` from the repo root, run by a nurse (haiku):
exit 0, **195 passed, 1 skipped** in 31.4 s. The skip is sklearn's array-api check (`SCIPY_ARRAY_API` unset).

Per file: bounds 12, conformal 39, ctc 11, differentiator 9, eras 8, estimator 59 (+1 skip), liquid 4, ns2d 13, observer 19, router 8, rsl 13.

**1a clean.** Every test named in a Builder `green` event shows as PASSED. The aggregate counts also reconcile:
- Builder-physics claimed 34 (13 + 12 + 9). The re-run has 34.
- Builder-core claimed 75. The 3 post-mutation tests from line 590 add 4 parametrised cases, giving 79. The re-run has 79 (observer 19 + router 8 + conformal 39 + rsl 13).
- Builder-model claimed 23 (eras, liquid, ctc) plus check_estimator 50 passed / 1 skipped. The re-run has 23 + 59 + 1 skip.

**1b struck.** Builder-physics is bound. Its file-level red events (lines 557, 560, 575) list every lemma, and the red counts equal the green counts (13/13, 12/12, 9/9).

Builder-core red events (lines 564-574) name tests individually. The following lemma tests are green with no prior red that names them:

- `test_observer.py::test_L3_nonpositive_temperature_raises`
- `test_router.py::test_E3_radius_from_state_error_is_sound_and_certifies`
- `test_conformal.py::test_L6_newborn_era_in_every_set_until_enough_calibration_points`
- `test_conformal.py::test_L6_trajectory_split_is_disjoint_and_whole`
- `test_rsl.py::test_L23_birth_fires_only_on_evidence_and_leaves_old_probabilities`
- `test_conformal.py::test_L24_star_calibration_slack_is_necessary_for_frozen_calibration`
- `test_rsl.py::test_L25_prime_no_generated_content_in_era_training`
- `test_rsl.py::test_L26_shapley_efficiency_exact_and_sampled`
- `test_rsl.py::test_L26_retire_merges_low_value_eras_into_null`
- line 590: `test_L26_retire_reindexes_survivors_contiguously_and_update_still_works`, `test_L26_retire_hands_the_decision_a_value_function_with_v_empty_zero`, `test_L6_quantile_rank_is_ceil_n_plus_1_times_1_minus_alpha`. These are mutation-bound (red on mutants S16, S14 and C1, recorded in the same event), not red-first.

Builder-model's red at line 556 names L19, L18*, L27, L20, L20′, L22 and L22*, but not L22†. Four tests are therefore unbound by name:

- `test_ctc.py::test_L22dagger_refuses_expansive_G`
- `test_ctc.py::test_L22dagger_km_for_nonexpansive_undamped`
- `test_ctc.py::test_L22dagger_local_beats_worst_case`
- `test_ctc.py::test_L22dagger_uncertified_T_refuses_and_budget_honest`

That makes 17 unbound tests. Every Builder-core red is a collection `ModuleNotFoundError`, so it applies to every test in the file that existed at that time. The log does not show whether the 17 tests above existed then.

A further 16 Builder-core lemma tests appear in no event by name and are covered only by the aggregate "75 passed" (for example `test_L5_separatrix_counterexample_is_refused`, `test_L24_star_trigger_restarts_with_summable_thresholds` and `test_L26_shapley_known_games_and_loo`).

## 2. FREEZE

The nurse computed both hashes with `sha256sum` and with Python `hashlib`. Both tools gave the same values, and both match `FREEZE.md` (frozen 2026-10-11T15:57:27+05:30):

- `PREREG_AMENDMENT_11_DRAFT.md`: `4b8596e2a5a296338b2d5560d4fe708fb400bbccd9c4bf5d657d3c30364e200b`
- `2026-10-11-fcpa-design.md`: `4bf2ddd2f9fe297d351d95faada071f9a4a91a453dbfea25a3c2e71ed3786c47`

**Clean.** `zero/docs/specs/` is untracked in git, so FREEZE.md is the only anchor. Nothing prevents a spec and its hash from being edited together until the directory is committed.

## 3. Spot-check of 6 lemma tests

Sample: Python `random.seed(12); random.sample(L, 6)` over the 85 tests whose names start with `test_L<n>`, `test_E<n>` or `test_Lgiven`, listed in sorted file order.

| Test | Lemma (design spec) | Verdict |
|------|---------------------|---------|
| `test_observer.py::test_L2_exact_residual` | L2: r_T = 1 − Σ_T p_k ≥ 0 equals the mass outside T | **clean.** 200 random (p, T) draws check r ≥ 0, r equal to the mass outside T to 1e-15 and to 1 − Σ_T to 1e-12, plus the empty-set and full-set edges. |
| `test_ctc.py::test_L22star_contraction_constant` | L22*: on {p_k ≥ δ}, L_post ≤ 2 + 2/(Tδ). If c = (1−ε)L_G L_post < 1, then T is constant on the orbit, the residual is ≤ 2ε and the bias is ≤ 2ε/(1−c) | **STRUCK, weaker.** It asserts that `contraction(0.2, 0.5, 1.0) == 0.4` and that `post_lipschitz` returns 2 + 2/(Tδ). That only re-states the implementation's formulas (ctc.py:7-14) and tests no inequality. The sibling test samples the L_post bound empirically, but no test in `test_ctc.py` covers the residual ≤ 2ε, the bias ≤ 2ε/(1−c), or orbit constancy of T. |
| `test_observer.py::test_L23_double_prime_top_eigenprojector_is_optimal` | L23″: tr(E^{1/2} C E^{1/2} Q) ≤ Σ_{i≤r} λ_i for rank(Q) ≤ r, with equality at the top-r projector | **clean.** It checks equality at `birth_direction` to rel 1e-10 and the bound against 10,000 random admissible Q (0 ⪯ Q ⪯ I, rank ≤ r) at 3 values of (N, r). It also checks that the optimum is a legal split and the r = 0 and r ≥ N clips. |
| `test_eras.py::test_L19_collapse_guard_on_embedding_rank` | L19 (C, cited): SIGReg plus a rank-of-embeddings collapse test as the guard | **clean.** The theorem is cited, so the testable claim is the guard. Unguarded ridge at λ = 1e3 falls below 0.8 of the target effective rank, and the guard restores it with λ_eff < 1e3. |
| `test_conformal.py::test_L24_births_within_budget_on_no_regime_change_stream` | L24: births ≤ αT + (max(α₁, 1−α₁) + γ)/γ with no new regime | **clean.** Over 3 seeds × 5,000 steps it checks births ≤ errors ≤ budget < T/2 and that the controller works at level α (errors > 0.5αT). The budget formula is pinned separately. The stream is i.i.d. only; the adversarial case is covered by the L7 tests. |
| `test_estimator.py::test_L27prime_chunked_partial_fit_equals_fit_with_buffer` | L27′: with the buffer state, partial_fit equals fit | **clean.** It uses 6 chunk sizes, none a multiple of the window. It checks that the era counts match (≥ 2) and that law counts are exactly equal, with predict agreeing to 1e-8 and tt_proba to 1e-10. The negative half ("without the buffer it does not") is the sibling test `test_L27prime_exactness_fails_without_the_buffer`. |

## 4. Lean

`zero/lean` exists: 26 project `.lean` files plus a vendored `.lake`. It was not built.

**4 `sorry`** in project files:
- `FCPA/L21_CommonNorm.lean:53`
- `FCPA/L24s_Ville.lean:15`
- `FCPA/L27_PartialFit.lean:77`
- `FCPA/L27_PartialFit.lean:81`

The other 150 grep hits are in `.lake` (Mathlib, std, aesop and other dependencies).

**Struck.** `check_no_sorry.py` exits 1 (4 sorry, 0 cited-sorry) and `CITED.txt` is absent. The spec allows Ville to be CITED, but it is not registered as cited, so the project's own gate is red.

## 5. No evaluation before the freeze

**Clean.** `zero/runs/` does not exist, and no path matching `*/runs*` exists within depth 3 of `zero/`. `zero/fcpa/` contains no `.json`, `.jsonl`, `.csv`, `.npz` or `.pt` files. `fcpa/beds/` holds only `__init__.py` and `ns2d.py`.

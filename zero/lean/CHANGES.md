# FCPA Lean: statement changes against the spec

Toolchain: `leanprover/lean4:v4.7.0`, Mathlib `a45ae63` (v4.7.0 era), local path dependency copied from the Epsilon-Hollow
checkout (nothing in Epsilon-Hollow was modified). Spec numbers refer to `docs/specs/2026-10-11-fcpa-design.md` including
Amendments A, B and C (Chase's audit, `scratchpad/plan/chase_spec.md`) and `docs/specs/ERRATA.md`.

Spec pin note: spec §0.2 names Lean v4.34.1 / Mathlib `d13f23b`. This project builds on v4.7.0 (no network). Names are
v4.7.0-era (`∑ x in s`, `Real.rpow_*`, `set_integral_*`, `Matrix.exp_*`). Porting to the later pin is a rename pass and needs a
Mathlib download that has not been approved.

Status of this file: `python check_no_sorry.py` reports **0 sorry, 0 cited-sorry**. Nothing in the Lean tree is `sorry`,
`axiom`, `admit` or `native_decide`; spot-checked theorems depend only on `propext`, `Classical.choice`, `Quot.sound`.
"CITED" below means a spec lemma that is **not** in Lean and rests on a literature citation plus a numerical test (there is no
`CITED.txt` because no Lean file carries a cited `sorry`). OPEN means a spec lemma with no Lean proof yet, with the reason.

## Proved (statement = spec unless a change is listed)

| Lemma | File (`FCPA/…`) | Change vs spec / what is proved |
|---|---|---|
| L1 | `L01_BornExactness` | sum over all `K+1` rows; general `parseval` isometry also proved (feeds E3) |
| L2 | `L02_ExactResidual` | none |
| L3 | `L03_TailCertificate` | `max` replaced by any upper bound `M` on the tail; `q = 1/T ≥ 1`; counterexample for `q < 1` proved (`no_Kfree_bound`) |
| L3′ (Amend. B) | same | `tail_power_mean`: `∑ p^q ≤ m^(1-q) r^q` for `0<q<1`; K-dependent |
| L4 | `L04_PostSelection` | extra hypothesis `∃ j ∈ S, 0 < p j`; `w` on `Option S`; any `T > 0` |
| L5 | `L05_CertifiedTopM` | pairwise form; REFUSE is policy, converse not claimed |
| L6 / L6′ | `L06_SplitConformal` | split-conformal coverage `P(α < p) ≥ 1-α` under exchangeability, ties allowed. Ties-safe counting lemma (`conformal_count`), event-level exchangeability from permutation-invariant law (`exch_of_perm_invariant`), p-value form (`pval_gt_iff`), newborn rule `⌈(n+1)(1-α)⌉ ≤ n ↔ n ≥ 1/α - 1` (`rank_le_iff`). Mondrian = apply inside each class. Prediction set stated in rank/p-value form (equivalent to "score ≤ ⌈(n+1)(1-α)⌉-th smallest calibration score") |
| L7 | `L07_ACICoverage` | **spec statement is FALSE as written.** Added boundary convention `α_t<0 ⇒ err_t=0`, `α_t>1 ⇒ err_t=1`, `err_t∈{0,1}`; initial condition relaxed to `α_0 ∈ [-γ,1+γ]` |
| L8, E2 | `L08_NSIdentifiability` | real inner-product space, `f` known; E2 as an "if" (`ns_identifiable_time`) |
| L9 | `L09_NSNoiseBound` | error in the residual only |
| E2′ (Amend. B) | same | `ns_noise_slab` (arbitrary slab vectors) and, since N6 flagged the gap, `ns_noise_demeaned`: demeaning step in the statement (`R_t = ν Δ_t + f + e_t`, time-constant `f`), bound `√Σ‖ẽ‖² / √Σ‖Δ̃‖²`; `forcing_error`: `f̂ - f = mean e - (ν̂-ν) mean Δ`. REFUSE when `Σ‖Δ̃‖² = 0` is hypothesis `hD` |
| L11 | `L11_CertifiedChange` | measure-theoretic union bound; H0 (equal true law) built in; `2δ` is per test |
| L13 | `L13_Cocycle` | any `AddCommGroup`, plus the log-ratio chart |
| L14 | `L14_Address` | metric-space form; no tie-break claim |
| L14′ | `L14b_HMMFilter` | filter update is a probability vector (`hmm_filter_prob`), predict step normalised, and `hmm_filter_is_posterior` (the update equals the Bayes posterior of the joint model `b_i P_ij ℓ_j`). Not formalised: end-to-end "address by law class" beyond the metric lemma `address` applied to class fingerprints |
| L15 | `L15_CrossingComposition` | abelian (commuting generators) case; loops return to `1` |
| L16 (ERRATA 1 sign) | `L16a_ExpRemainder`, `L16_Holonomy` | `holonomy_bch2`: in any complete normed algebra `‖exp(-(A+B)) exp B exp A - 1 + ½[A,B]‖ ≤ 432 ε³` for `‖A‖,‖B‖ ≤ ε ≤ ½`, derived from the exponential series (no BCH in Mathlib); `crossing_holonomy` is the instance for `S(-(a+b)) S(b) S(a)` on matrices (`ℓ^∞` operator norm). It measures non-commutativity of the generators, not curvature of law space (Chase). The `det ∏G` wiring is Python-side |
| L17 | `L17_ElasticCrossing` | **real coefficients** required (complex coefficient is a counterexample); norm preservation |
| L18* floor | `L18s_ZetaFloor` | `P(ζ ≤ ζ₀) ≤ (1-s)/(1-ζ₀)` under `E[ζ] ≥ s` (Chase's hypothesis: holds when `V_k` is the top-`d_c` eigenspace). `Lip(R_k) ≤ 2L_J/√ζ_min` not verified (OPEN) |
| L19 | `L19_CollapseGuard` | collapse guard: a flat direction (`∃ v ≠ 0, ∀ i ⟨v,z_i⟩ = 0`) contradicts the isotropic covariance `C = I`, so `C = I` forces the embeddings to span (rank test). SIGReg's scope statement itself is CITED (LeJEPA), not in Lean |
| L20, L20′ | `L20_LiquidInvariant` | scalar/componentwise (diagonal `F`); box invariance, one-step distance contraction, `d_n ≤ Σ c^{n-1-j}|J_j|` and `d_n ≤ δτ/h` |
| L21 | `L21_CommonNorm` | constants form `c₁‖x‖² ≤ xᴴHx ≤ c₂‖x‖²`; `H`-contraction as a pointwise hypothesis plus `hcontr_of_psd`; demoted by Amendment A to the linear part |
| L22* | `L22_DampedLoop` | abstract complete convex `C` of diameter ≤ 2; contraction hypothesis `c = (1-ε)L < 1` explicit; unique fixed point, rate, residual `≤ 2ε`, bias `≤ 2ε/(1-c)`. Hypothesis `hG` (global Lipschitz on `C`) is **not** discharged for post-selection: the `L_post ≤ 2 + 2/(Tδ)` bound is not proved (superseded in practice by L22H and L22†) |
| L22H (ERRATA 15) | `L22H_HilbertLipschitz`, `L22H2_HilbertFixedPoint` | `softmax_hilbert_lipschitz` (`(βD/(4T))`-Lipschitz in the Hilbert/oscillation distance) and `hilbert_ctc_fixed_point` (unique fixed point + geometric rate for `T > βD/4`, via an isometric embedding of log-coordinates mod constants into a finite-dimensional space; no completeness of `d_H` needed) |
| L22† (Amend. B) | `L22d_OrbitLocal` | **contract branch only**: `orbit_local_converges` (orbit-local contraction `c<1` ⇒ convergence to a fixed point, geometric rate, residual `≤ d₀cⁿ`) and `iteration_budget` (a-priori iteration count `⌈log(ε/d₀)/log c⌉`). The Krasnosel'skii–Mann branch and the dispatcher are OPEN (below) |
| L23′ | `L23_NonDestructiveBirth` | Hermitian `S` with `S*S = E∅` is an input (no matrix square root in v4.7.0 Mathlib); `birth_split`, `birth_preserves_sum`, `birth_born`, and (flagged by N6) `birth_old_unchanged` (`p_k` of every old outcome is the same expression, by `rfl`) and `birth_total_born` (Born masses still sum to 1 for unit `ψ`). Rank-one vacuity not formalised |
| L24′ | `L24_NoRunawayGrowth` | `births_le` (linear bound, corollary of L7) and `birth_starvation`. Superseded by L24* (kept as the ACI-based fallback) |
| L24* (Amend. B, birth-specific) | `L24s_Ville`, `L24b_BirthEProcess`, `L24c_Calibrator` | PROVED, not cited. `ville`: nonneg supermartingale, `c·P(∃n, M_n ≥ c) ≤ E[M_0]`. `mixM_supermartingale`: the mixture e-process `S_t = (S_{t-1}+w_t)e_t`, `w_k = 1/(k(k+1))`, `M_t = S_t + 1/(t+1)` is a nonnegative supermartingale with `M_0 = 1`, given conditional e-values (`∫⁻_s e_t ≤ μ s` for `s ∈ F_{t-1}`). `calibrator_integral_le`/`calibrator_cond`: `e = κ p^{κ-1}` is a conditional e-value when `p` is conditionally super-uniform (layer-cake). `birth_false_alarm` / `birth_false_alarm_pvalues`: `P(∃ era j, ∃ t, M^{(j)}_t ≥ (j+1)(j+2)/δ) ≤ δ`: no false birth, ever. This is the ERRATA 14 (H3) mixture-over-change-time detector with weights `w_k = 1/(k(k+1))` (the Monte-Carlo martingale check there is now a proof, given conditional e-values) |
| L25 | `L25_Grounding` | well-founded dataflow: a node with no TT ancestor is invariant to the TT's rule. Modelling statement; the invariance-to-TT-depth claim is withdrawn by Amendment C |
| L25′ | same | `admit` / `admit_grounded` / `admit_append_generated`: only grounded data is admitted, so TT-generated data never reaches era training. Router-selection bias (births fire on the TT's `C = ∅`) is not covered (Chase) |
| L26 | `L26_Shapley` | Shapley efficiency `Σφ = v(univ) - v(∅)` over all `n!` orderings (telescoping per ordering); `loo_not_efficient` (leave-one-out retires both of two duplicates) |
| L27′ | `L27_PartialFit` | boundary transition fixed (the spec's L27 is FALSE for transition counts). Full state now as in `estimator.py::_State`: `PFState` = (closed-segment statistics, open-segment statistics, w-sample buffer); `partialFit_exact` (`partial_fit(fit xs, ys) = fit (xs ++ ys)`), `partialFit_chunks` (any chunking), `no_buffer_inexact` (dropping the buffer breaks it). `step` is a generic window consumer, so the statement holds for any deterministic segmenter. Gram `gram_append_matrix` (matrices, `Σ x xᵀ`) and `xty_append` (`Σ y x`) are additive. Exact in exact arithmetic only; JEPA SGD is not additive |
| L-GIVEN | `LGiven_LibraryTerms` | PRESENT/SMALL soundness, `η = ‖e‖/σ`, and (flagged by N6) `structure_learned`: only NS terms PRESENT and all others SMALL ⇒ NS terms truly nonzero and others `≤ 2η` |
| E3 | `E3_BornLipschitz` | `Σ_k |p_k(ψ) - p_k(ψ')| ≤ 2 √(Σ|ψ_i - ψ'_i|²)` for unit vectors |
| L38 (era bank) | `L38_GrowingBankRegret` | `bank_regret`: pathwise regret of the TT filter (move `(1-α)w + ρ/|s| + wK`, Bayes) over a bank that grows by births and shrinks by folds, vs any comparator path; stay cost `-log(1-α+K_cc) + b·log(1/(1-α))`, switch cost `≤ log(N/ρ)` (`cost_switch_le`) or `-log K_ji(1-α)^b` (`cost_kernel_le`). `bank_regret_simple`, `bank_regret_fixed` (vs a never-folded tail expert: `T log(1/(1-α+κ)) + B log(1/(1-α))`), `window_regret` (re-entry from any time), `butterfly`. Regret is vs pool paths, not vs the oracle (L10/L12 CITED) |
| L39 (merge, exact) | `L39_MergeRule` | Dirichlet–multinomial `ML` via rising factorials; `chain_rule`, `preq_eq` (address loop's prequential product = `ML(D|C)`), `merge_iff_address` (merge ⇔ era beats a fresh era on the other era's tokens), `address_symm`, `false_merge_le`/`false_split_le` (BF is an e-value), `seq_mass_one`, `threshold_one_vacuous` (the code's `BF > 1` is `δ = 1`). "BF > 1 never merges different laws past n_min": FALSE for every n_min |
| L39 (merge, consistency) | `L39c_MergeConsistency` | `log_bf_sandwich`: `|log BF - (-Div + Gain)| ≤ 3|R|(Vκ(a)+κ(Va))` (Stirling, `theta_bounds`); `div_row_nonneg` (Gibbs); `refuse_merge`, `accept_merge`, `same_law_merges`, `proportional_div`/`proportional_gain`. Deterministic; the sampling bridge (strong law, Wilks) is CITED. Constant is loose (≈3.5·10³ nats at V=16, a=0.2, |R|=16): rates, not thresholds |
| L40 (no forgetting) | `L40_NoForgetting` | Ledger model with all code ops (`credit`, `birth`, `fold`, `readdress τ b`). `count_birth_old_unchanged`, `settled_kept`, `no_forgetting` (settled counts survive, relabelled only by the merge map), `untouched_era_unchanged`, `ema_forgets` (contrast). Unconditional "counts change only through tokens assigned to it": FALSE (readdress removes unsettled tokens; folds add) |
| L41 (MOM) | `L41_Momentum` | `mom_is_move` (MOM = L38 with `ρ = α/2`, `K = (α/2)Q̃`), `mom_transition_stochastic`, `mom_eq_hmm_predict` (HMM forward filter, `A = (1-α)I + (α/2)J/n + (α/2)Q̃`), `mom_switch_cost` (`≤ -log((α/2)Q̃_ji) + b log(1/(1-α))`, any switch `≤ log(2N/α)`), `mom_regret_tail` (`log(1/(1-α/2))` per token vs a tail expert), `momQ_pos`, `momQ_row_sum`, `soft_is_move`, `mom_identity_is_soft`, `mom_uniform_is_soft`. Learned `Q` covered pathwise; its convergence rate is OPEN |
| L39 (threshold) | `L39d_MergeThreshold` | `bf_running_product` (BF = running product of one-token ratios `pred_{C+D}/pred_D`), `pred_ratio_evalue` (each ratio has mean 1 under the separate-era predictive), `prodE_supermartingale`, `prodE_ville`, `merge_false_alarm`: merge iff `BF_j ≥ (j+1)(j+2)/δ` gives P(any false merge, any pair, any time) `≤ δ` under the separate-era Dirichlet model. `no_uniform_merge_guarantee`: merge probability is continuous in the laws, so no rule that merges same-law eras can bound false merges uniformly over all different laws |
| L41b (learned kernel) | `L41b_LearnedKernel` | `rise_le`, `multinom_le`, `multinom_types` (method of types), `laplace_regret`, `learned_kernel_regret`: the add-one kernel's log loss on any switch sequence is within `(k-1) log(n+1)` of the best fixed kernel in hindsight. Pathwise, no generating-process assumption |
| L42 (shipped filter) | `L42_ShippedFilter` | `lse_shift`, `lse_shift_max` (max-shifted log-sum-exp is exact; terms in `(0,1]`, sum in `[1,n]`), `lse_log`, `logBayes_exact`, `logLoss_exact` (log-space Bayes and loss = L38's), `hard_cost` (`log θ ≤ log mix - log P_top ≤ log(θ + (1-θ)/ε)`), `hard_regret` (HARD = mixture + that term per HARD step) |

## OPEN (no Lean proof; never a `sorry`)

| Lemma | Reason |
|---|---|
| L10 Markov plug-in concentration | CITED (Paulin 2015; Chung et al.). Needs Hoeffding/Bernstein for dependent chains; Mathlib v4.7.0 has no Hoeffding or Azuma. Numerical test only |
| L12 window ≥ mixing time | CITED (Hsu–Kontorovich–Szepesvári; Levin–Peres–Wilmer). Needs spectral theory of reversible chains and total-variation mixing; not in Mathlib v4.7.0 |
| L18 Ky Fan zoom ceiling | CITED (Ky Fan 1949). The Ky Fan maximum principle is not in Mathlib v4.7.0 (a proof via the spectral theorem and `Matrix.IsHermitian.eigenvalues` is possible but was not done). The floor L18* is proved above under `E[ζ] ≥ Σλ_i`; `Lip(R_k) ≤ 2L_J/√ζ_min` is unverified (Chase) |
| L23″ optimal birth direction | needs Ky Fan (same reason). Recorded as out; L23′ is proved |
| L22† Krasnosel'skii–Mann branch and dispatcher | KM convergence for nonexpansive maps (averaged operators) is not in Mathlib v4.7.0; only the contract branch is proved. The dispatcher is a decision rule, not a theorem |
| L22* worst-case `L_post ≤ 2 + 2/(Tδ)` | superseded by L22H (T above `κ_H`) and L22† (orbit-local); not machine-checked |
| TT-0 / TT-1 | TT-1 (softmax is 1-Lipschitz in ℓ², actually ½) is **CITED** (Gao–Pavel 2017), numerically tested, not in Lean. TT-0 is a definition (state update `m_t = Φ(m_{t-1}, x_t)`); E1 (encoder) and D2′ are definitions, not lemmas |
| L25′ router-selection bias | modelling gap (Chase); the Lean theorem covers provenance only |
| L39 sampling bridge | CITED: `Div/N → JS > 0` a.s. (strong law) and `2Div → χ²` (Wilks). The deterministic sandwich (`L39c`) and the anytime threshold (`L39d`) are proved. A uniform frequentist guarantee over all different laws is impossible (`L39d.no_uniform_merge_guarantee`); one with a separation margin needs method-of-types concentration |
| L41 learned-`Q` parameter rate | CITED (Hoeffding not in Mathlib v4.7.0). The rate the regret needs is PROVED in regret form (`L41b.learned_kernel_regret`: `(k-1) log(n+1)` per row) |

Earlier wording "nothing is cited" is retracted: L10, L12, L18 (Ky Fan) and TT-1 are CITED in the spec and not in Lean; they are
listed above and nothing in the Lean tree pretends to prove them.

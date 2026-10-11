# FCPA — Fluid Chrononal Predictive Architecture: design spec (lemma-first)

Status: DRAFT for review. Scope: Asmodeus `zero/` (a reshape, not a new repo). Origin: `ref/origin/AT_I_notes.pdf`, the notes "That if we combine" (2026-10-11), and the Gödel CTC figure.
Folder rule: not JEPA, not self-attention. JEPA, MoE and attention are vocabulary and baselines. L-PREREG: bars come before results. L-GIVEN: structure is learned, not given.

## 0. Requirements (from the user)
1. Every architectural piece is written as mathematics first: a definition plus a lemma.
2. Every lemma is either proved in Lean (`zero/lean/`, Mathlib pinned to the `math` repo's pin `d13f23b`, Lean v4.34.1, committed with `sorry` first, CI fails on `sorry`), or explicitly marked CITED, with the source and a numerical test.
3. A reference implementation as a scikit-learn estimator (numpy/scipy/sklearn only) that passes `check_estimator`. A torch fast path may follow, and must match the reference.
4. Deep application: (C) a Navier–Stokes and Markov physics bed first, then language.
5. RSL (recursive self-learning, defined in §3.10) is a first-class mechanism.

## 0.1 Folder-rule compliance
- Rule 2: FCPA enters as a third first-class candidate beside OBSERVER-ATLAS and PHASE-SPHERE, plus hybrids. It reuses their parts (POVM observer, Cayley, charts) and replaces nothing by fiat.
- Rule 4 (L-GIVEN): the Navier–Stokes and Markov law families are the *bed's* ground truth and the differentiator's hypothesis class. To keep structure learned rather than given, a mandatory arm replaces the NS form with a learned operator (sparse library regression in the PDE-FIND/SINDy style, with no fixed terms). The NS-form arm must beat or tie it, or the claim is reported as "given structure".

## 1. The picture (the diagram with its dimensions shifted)
| Gödel CTC diagram | FCPA object |
|---|---|
| t (vertical) | chronology of the data. Eras are slabs of it |
| red square (a spacelike slice) | **era** E_k: a JEPA world model of one slab |
| blue life-line | **TT** (time traveller): a softmax LLM streaming forward that stores only *results* of history |
| light cones tilting with r | r = how far the TT leaves softmax mode. Near the axis it talks; far out it can loop back |
| CTC | **asking an era**: the TT returns to slab k, the JEPA expands, the TT returns changed |
| Φ | the frame or phase inside a slab (the scattering rotation) |

## 2. Objects (definitions)
- **D1 Data fluid.** An ensemble ("multiverse") of trajectories x^{(u)}_t, u ∈ U, encoded to states ψ_t ∈ S ⊂ C^N with ‖ψ‖ = 1. ρ_t is the empirical density of {ψ^{(u)}_t}.
- **D2 Observer.** A Parseval frame V ∈ C^{(K+1)×N} with V†V = I. The rows v_0..v_{K−1} are the era effects and v_∅ is "observe nothing". The effects are E_k = v_k v_k†, with Σ_k E_k = I.
- **D3 Laws.** The Markov law P ∈ R^{(K+1)×(K+1)} is the row-stochastic transition matrix of observer outcomes over a window. The flow law θ = (ν, f) is the viscosity and forcing of a 2-D periodic vorticity equation ∂_t ω + u·∇ω = νΔω + f fitted to ρ's velocity field. L = (P, θ).
- **D4 Bounds.** The Markov bound fires at t when a certified test separates P on [t−w, t) from P on [t, t+w). The NS bound fires when a certified test separates θ on those two windows.
- **D5 Era.** E_k = (I_k = [t_k, t_{k+1}), L_k, J_k, κ_k): the maximal interval on which neither bound fires, its law, its JEPA J_k, and its fingerprint κ_k = (L_k, chronology index k).
- **D6 Differentiator.** ΔE_{j→k} = L_k ⊖ L_j: the componentwise difference of θ, and the log-ratio or difference of P in a chosen chart.
- **D7 TT.** A softmax predictor π(y | h) over the next output given the history h, plus the router of §3.2.
- **D8 Crossing operator.** S(ΔE) = exp(Σ_a ΔE_a G_a), with skew-Hermitian generators G_a. It is unitary ("elastic").
- **D9 Zoom.** The era chart z_k = V_k† S ψ (d_c coordinates). J_k predicts z_k forward, and the answer returns through S†.
- **D10 CTC loop.** The damped fixed point w ← (1−ε) G(w) + (ε/M)·1 on the simplex Δ_M, where G is the TT's self-consistency map.
- **D11 Liquid step.** Between events: x_{n+1} = x_n + h[−(1/τ + F(x_n, I_n)) x_n + A ⊙ F(x_n, I_n)], with 0 ≤ F ≤ M.

## 3. Mechanisms and their lemmas. Tier A = Lean v1; Tier B = Lean v2; C = CITED + test

### 3.1 Observer and laziness
- **L1 Born exactness (A).** V†V = I and ‖ψ‖ = 1 ⇒ Σ_k |⟨v_k,ψ⟩|² = 1. *Mathlib:* the inner-product algebra plus `OrthonormalBasis.sum_sq_norm_inner_right` in spirit; proved directly from V†V = I.
- **L2 Exact residual (A).** For any T ⊆ outcomes, the residual r_T = 1 − Σ_{k∈T} p_k ≥ 0 equals the exact mass outside T.
- **L3 Tail certificate, T ≤ 1 only (A).** Σ_{k∉T} p_k^{1/T} ≤ (max_{k∉T} p_k)^{1/T−1} · r_T. For T > 1 no K-free bound exists, and a counterexample is part of the proof file.
- **L4 Post-selection temperature (A).** w_k = (1−r_T) p_k^{1/T} / Σ_{j∈T} p_j^{1/T} for k ∈ T, with w_other = r_T. Then w is a probability vector, and its order on T is the order of p.

### 3.2 Router decision (the TT talks unless it asks for an era)
- **L5 Certified top-m (A)** (separatrix `decide.topk_determined`). If the true scores s ∈ ∏[D_i ± R_i] and min_{i∈T}(D_i − R_i) > max_{j∉T}(D_j + R_j), then T is the top-m of every s in the box. Otherwise the router returns REFUSE, which means blend.
- **L6 Split-conformal era set (B).** Exchangeable calibration pairs (h_i, k_i), where k_i is the era by timestamp (ground truth exists), and score s(h, k) give P(k_{n+1} ∈ C_α(h_{n+1})) ≥ 1 − α. *Lean core:* the rank of an exchangeable score is uniform, a Finset counting lemma.
- **L7 ACI under drift (A).** α_{t+1} = α_t + γ(α − err_t) ⇒ |(1/T) Σ err_t − α| ≤ (max(α_1, 1−α_1) + γ)/(γT) for every sequence (Gibbs–Candès Prop. 4.1). The proof is telescoping plus α_t ∈ [−γ, 1+γ].
- **Decision rule.** C = ∅ means answer alone and flag OOD (a candidate RSL birth). C = {k} means jump to k. 2 ≤ |C| ≤ n+1 means a blend over C. Explicit request means the named era.

### 3.3 Bounds (Markov and NS) and certified change
- **L8 NS identifiability (A).** If ω solves the vorticity equation with known f and ⟨Δω, Δω⟩ > 0, then ν = ⟨∂_tω + u·∇ω − f, Δω⟩ / ‖Δω‖². This is an orthogonal projection; no existence theory is needed.
- **L9 NS noise bound (A).** If the residual carries an error e, then |ν̂ − ν| ≤ ‖e‖ / ‖Δω‖ (Cauchy–Schwarz).
- **L10 Markov plug-in concentration (C).** Paulin 2015 (McDiarmid type with mixing time τ_min, and Bernstein type with spectral gap γ) and Chung et al. (Chernoff–Hoeffding with L1 mixing time). Test: empirical kernel error against the bound on chains with known gap.
- **L11 Certified change soundness (A).** If each window's interval contains its true law with probability ≥ 1 − δ, and a change is certified only when the intervals are disjoint, then P(false certified change) ≤ 2δ (union bound). Overlap means REFUSE, never "no change" (tangle `certify` pattern).
- **L12 Window ≥ mixing time (C).** Use Hsu–Kontorovich–Szepesvári data-dependent intervals for t_relax, with (t_relax − 1) ln 2 ≤ t_mix ≤ t_relax ln(4/π⋆). This sets w.

### 3.4 Differentiator and addressing ("the change becomes what data is where")
- **L13 Cocycle (A).** ΔE_{i→k} = ΔE_{i→j} + ΔE_{j→k} for additive law coordinates.
- **L14 Address (A).** If the fingerprints κ_k are pairwise s-separated and the estimates are within < s/2, nearest-fingerprint routing returns the true era. When two eras share a law, the chronology index in κ breaks the tie.
- **L15 Crossing composition (A).** If the G_a pairwise commute, then S(ΔE_{jk}) S(ΔE_{ij}) = S(ΔE_{ik}) (`Matrix.exp_add_of_commute`), so every loop of eras returns to I.
- **L16 CTC holonomy (C, then B).** With non-commuting G_a, a loop i→j→k→i returns to exp(½[A,B] + O(‖·‖³)) (BCH), where A and B are the generator steps: **a closed loop through eras measures the curvature of law-space**. It is tested numerically, wired to the existing `det ∏G` readout, and goes to Lean only if BCH-2 lands in Mathlib.

### 3.5 Crossing and zoom
- **L17 Elastic crossing (A).** S(ΔE) is unitary, so it preserves norms and fidelities. Information is conserved, and "expansion" means zoom gain, not created information.
- **L18 Zoom ceiling (C).** By the Ky Fan maximum principle, ζ_k = ‖V_k† Sψ‖² averaged over the era ≤ Σ_{i≤d_c} λ_i(ρ_k). Tested against eigh.

### 3.6 Era model (JEPA inside a slab)
- **L19 Anti-collapse objective (C).** The LeJEPA SIGReg theorem holds for nonlinear probes under an isotropic covariance constraint, so its scope is narrower than its billing. The era loss is the prediction loss in the chart plus SIGReg, with a collapse test (rank of embeddings) as the guard.

### 3.7 Liquid dynamics between events
- **L20 Discrete liquid invariant interval (A).** With 0 ≤ F ≤ M and step h ≤ τ/(1+τM), if x_0 ∈ [m, M'] where m = min(0, A_min) and M' = max(0, A_max), then x_n stays in that box for all n. This is the discrete analogue of LTC Lemma 5, proved by induction.

### 3.8 Butterfly (sensitivity), linear part proved and nonlinear part measured
- **L21 Common-norm product bound (A, moderate).** If H ≻ 0 and A_t* H A_t ⪯ H for all t, then ‖A_n ⋯ A_1‖₂ ≤ √κ(H). Every non-unitary step is parameterised as A_t = H^{-1/2} C_t H^{1/2} with ‖C_t‖ ≤ 1.
- **Measured:** the finite-time Lyapunov exponent of the full nonlinear rollout, against the unconstrained arm. Kill rule: if it is not lower, the "butterfly bounded" claim leaves.

### 3.9 Observer loop (CTC)
- **L22 `damped_loop_depth` (A).** If G maps Δ_M to itself, is L-Lipschitz in ℓ¹, and (1−ε)L < 1, then F has a unique fixed point and dist(Fⁿx, w*) ≤ dist(x, Fx)·Kⁿ/(1−K), with K = (1−ε)L. This is the direct instance of Mathlib `ContractingWith.apriori_dist_iterate_fixedPoint_le`. With post-selection temperature, L ≤ L₀/T_min, so ε is tied to T_min. The README:55-57 claim ("ε > 0 alone") is corrected by this hypothesis.

### 3.10 RSL: recursive self-learning (hot-reload eras)
**Definition.** A system is RSL if it (i) grows its own model set when its own calibrated uncertainty says the current set does not cover the data, (ii) never perturbs the routing of existing models when it grows, and (iii) trains each new model only on grounded data, never on its own outputs.
- **L23 Non-destructive birth (A).** Split the null effect, E_∅ = E_new + E_∅′, with 0 ⪯ E_new ⪯ E_∅. All other effects stay unchanged, so p_k(ψ) is unchanged for every existing era k and every ψ, and Σ = I still holds.
- **L24 No runaway growth (A).** If births fire only on C = ∅ under ACI at level α, then in any stretch with no new regime, the number of births is ≤ αT + (max(α_1, 1−α_1) + γ)/γ. This is a corollary of L7.
- **L25 Grounding / anti-collapse (A).** If each era's training set is a function only of its data slab (the dataflow graph has no edge from TT outputs to era training data), then era estimates are invariant to the TT's recursion depth. That rules out the self-consuming loop behind model collapse (Shumailov et al. 2024). The proof is a DAG argument.
- **L26 Retire / merge by Shapley (B).** The Shapley values of eras with respect to the validation gain sum to the total gain (efficiency). Eras with value below a threshold merge or retire, and the leave-one-out arm is the cheap check.

### 3.11 scikit-learn contract
- **L27 Exact `partial_fit` (A).** Era statistics are additive sufficient statistics (transition counts, normal-equation Gram matrices), so `partial_fit` over chunks equals `fit` on their concatenation.
- **API.** `FCPARegressor(BaseEstimator, RegressorMixin)` with `fit(X, y=None, t=...)`, `predict`, `partial_fit` (the RSL hot-reload entry point), `score`, and `random_state`. It must pass `sklearn.utils.estimator_checks.check_estimator`.

### 3.12 Computability ceiling (no beyond-Turing claim)
- **C.** P_CTC = PSPACE (Aaronson–Watrous): the CTC loop is a computable fixed-point search. Forced Navier–Stokes is Turing-universal (corpus 376), so NS-bed bars are bounded-horizon.

## 4. Proof order (Lean v1, Tier A)
L1 → L2 → L4 → L3 → L5 → L7 → L24 → L11 → L13 → L14 → L15 → L17 → L8 → L9 → L20 → L22 → L23 → L25 → L27 → L21.
Tier B: L6, L16 (if BCH), L26. Tier C: tested in pytest with numerical oracles.

## 5. Beds and the one primary claim
- **NS-2D multiverse bed:** periodic 2-D vorticity, pseudo-spectral; an ensemble of trajectories; piecewise-constant (ν, f) switching at random times (NS bounds); and a Markov-switching forcing regime (Markov bounds). The eras are ground truth.
- **Language bed (phase 2):** a timestamped corpus, with eras = periods.
- **Primary claim (preregistered, Amendment 11):** on the NS-2D bed, FCPA has lower 50-step error than (i) a single global model with equal parameters and (ii) ASMODEOUS top-m, at ≤ equal compute, with era identification ≥ 0.9 and ACI long-run coverage within 0.02 of target. Secondary bars are Holm-corrected. If the primary claim fails, the reshape reverts behind a feature flag.

## 6. Reuse from past work
separatrix (`enclose.py` γ_n, `decide.topk_determined`) gives L5. tangle (`certify` verdicts, interval REFUSE) gives L11. planimeter (DSU, EMST window invariance) and cleave (`h0_barcode`, `division_signature`) give era split and merge. lambda-topo and hamliton (topological fingerprints) give κ. faraday (`power_iterate` with an honest `converged` flag) gives the L22 tests. Epsilon-Hollow `AetherVerified` (`geometric_bound`) is the style reference for L22. In-repo: `atlas/povm.py` and `asmodeous.py` (stiefel, cayley), `qflm` butterfly, and `event_driven.py`.

## 7. Out of scope
Beyond-Turing claims. 3-D Navier–Stokes (regularity is open). Corpus results dropped by the differential: 325 as the butterfly bound, 374, 300, 261, 148, 143, 328.

## Amendment A (review by Foreman, 2026-10-11) — these supersede the lemmas they name
Full derivations: `scratchpad/plan/foreman_spec.md`. "To be machine-checked" means a Lean or pytest pass decides it.
- **L22\* replaces L22, and it fixes L4/L5 coupling.** Post-selection is not globally Lipschitz: it jumps when the top-m set T flips, and p^{1/T} has infinite slope at 0. So: fix T and work on Δ_{|T|+1}; on {p_k ≥ δ for k ∈ T}, L_post ≤ 2 + 2/(Tδ) (to be machine-checked). If c = (1−ε)·L_G·L_post < 1 and T is L5-certified on the ℓ¹ ball of radius d₀/(1−c), then T is constant along the orbit, the iterates converge, the residual is ≤ 2ε, and the bias to an undamped fixed point is ≤ 2ε/(1−c).
- **TT-0 / TT-1 (new).** The TT state update is m_t = Φ(m_{t−1}, x_t). Softmax is 1-Lipschitz in ℓ² (CITED: Gao–Pavel 2017). An objective ledger lists every trainable part and its loss.
- **D2′ + L23′ replace D2 + L23.** E_∅ may have any rank. For any 0 ⪯ Q ⪯ I: E_new = E_∅^{1/2} Q E_∅^{1/2} and E_∅′ = E_∅^{1/2}(I−Q)E_∅^{1/2}. Old p_k are unchanged and Σ = I. When rank(E_∅) is exhausted, extend with an ancilla (Naimark).
- **L24′ replaces L24.** A birth needing n_min points requires γ ≤ α_start/((n_min−1)(1−α)), about 0.0022 for n_min = 50 and α = 0.1. Otherwise ACI refills C before the newborn has data ("birth starvation").
- **L6′ replaces L6.** Train on B and calibrate on a disjoint Cal, with whole trajectories as the exchangeable units. Use Mondrian (per-era) calibration. A newborn era is included in every C until n_cal ≥ ⌈1/α⌉ − 1. Timestamp labels and the C = ∅ trigger must use one era definition.
- **E1/E2/E3 (new) fix D1 and L8.** E1: the encoder is a truncated rfft plus an explicit amplitude channel (‖ψ‖ = 1 loses the amplitude, which confounds ν under NS scaling). E2: (ν, f) is identifiable iff f is time-constant in a subspace that Δω(t) leaves for some t. Otherwise (ν′, f + (ν−ν′)Δω) gives the same ω. E3: ‖p(ψ) − p(ψ′)‖₁ ≤ 2‖ψ − ψ′‖, which gives L5's radius R.
- **L25′ replaces L25.** Provenance tags on every datum. Births and updates admit only grounded data, so the TT's own tokens can never enter era training.
- **L20′ replaces L20 across events.** dist(x_n, B) ≤ Σ_j (1−h/τ)^{n−n_j} δ_j ≤ δ_max·τ/h. **L21 is demoted** to the linear/unitary part. The butterfly kill rule becomes |λ_model − λ_data| ≤ η, with the bar horizon inside the predictability horizon H.
- **L18\* replaces L18.** Floor: P(ζ ≤ ζ₀) ≤ (1−Σλ_i)/(1−ζ₀). Lip(R_k) ≤ 2L_J/√ζ_min. The G_a objective is max tr(V_k V_k† S ρ_j S†), using ρ_j (the source era).
- **L14′ replaces L14.** Address by law class, because regimes recur. The tie-break is an HMM forward filter over era classes, since queries carry no timestamp at inference.
- **L27′ replaces L27.** The partial_fit state is (sufficient statistics, w-sample buffer, open-segment statistics). With that state, partial_fit equals fit; without the buffer it does not.
- **Bar sizing.** ACI coverage within 0.02 needs T ≥ (max(α₁, 1−α₁) + γ)/(0.02γ): 4,550 steps at γ = 0.01, about 20,500 at the L24′ γ. The era-identification bar needs λ_sw·w ≤ 0.1.

## Amendment B (review by Cameron, 2026-10-11) — adopted
Full text: `scratchpad/plan/cameron_spec.md`.
- **L23″ (optimal birth direction, Tier A).** For rank(Q) ≤ r, the captured Born mass tr(E_∅^{1/2} C E_∅^{1/2} Q) ≤ Σ_{i≤r} λ_i(E_∅^{1/2} C E_∅^{1/2}), where C = Σψψ† over the unexplained buffer. Equality holds at the top-r eigenprojector (Ky Fan), so Q is that projector. Births are capped at N/τ_min.
- **L24\* replaces L24′ (anytime-valid birth, Tier B; Ville's inequality CITED if not in Mathlib).** Per-era conformal p-values feed a power martingale M_t = Π κ p^{κ−1}. A birth fires at M_t ≥ 1/δ_j with δ_j = δ/(j(j+1)). With probability ≥ 1−δ there is no false birth, ever, and births ≤ #distinct laws. The trigger never reads ACI's α_t, so γ = 0.05 is allowed and birth starvation is gone. The ACI bar then needs ≈ 950 steps.
- **E2′ (explicit within-slab estimator, Tier A).** Demean over time: ν̂ = Σ⟨R̃, Δ̃⟩ / Σ‖Δ̃‖², with |ν̂ − ν| ≤ ‖e‖_F/‖Δ̃‖_F, and f̂ = R̄ − ν̂·mean(Δω). If Δ̃ = 0, return REFUSE.
- **L3′ (T > 1 tail, Tier A).** Σ_{k∉T} p_k^q ≤ m_out^{1−q} r^q for q = 1/T < 1 (Hölder/power mean), tight at the uniform tail. This is K-dependent, which is fine because the router knows K. It replaces the "T > 1 banned" clause.
- **L-GIVEN lemma (Tier A).** η = ‖e‖/σ_min(A). A library term is PRESENT if |x⁰_j| > η, SMALL if |x⁰_j| ≤ η (then |x*_j| ≤ 2η), and REFUSE in between. If only the NS terms are PRESENT and all others SMALL, "structure learned" is certified at resolution 2η.
- **L22† (orbit-local certification and dispatcher).** The worst-case L22* constant is vacuous at practical settings (L_post ≤ 42). Certify with the Jacobian norm along the actual orbit plus an a-priori iteration budget. The dispatcher chooses damped contraction, Krasnosel'skii–Mann averaging, or REFUSE.
- **Beds to preregister:** revisit recall (switching Kolmogorov flow, the PDO climate index, finance VaR, robotics against JEPA-TTT), and LLM-by-era on TempLAMA (per-era LoRA, F1 ≥ 28.2 with the timestamp hidden).

## Amendment C (truth audit by Chase, 2026-10-11) — corrections
Full per-lemma verdicts, Lean-ready statements and Mathlib names: `scratchpad/plan/chase_spec.md`. Result: 11 lemmas true as written, 8 needing a hypothesis, 2 false (L7, L27), L23 vacuous at rank 1, L16 provisional.
- **L7 is FALSE as written.** It needs the Gibbs–Candès boundary convention: α_t < 0 gives C = all labels, so err_t = 0; α_t > 1 gives C = ∅, so err_t = 1. Only then is α_t ∈ [−γ, 1+γ] and the bound true. The bar sizing in Amendment A holds only under this fix (its arithmetic is verified: 4,550 at γ = 0.01, about 20,500 at γ = 0.0022).
- **L18\* floor** needs E[ζ] ≥ Σλ_i, which holds only when V_k is the top-d_c eigenspace of ρ. That hypothesis is added.
- **L24′'s linear birth growth** (about αT births) is resolved by Amendment B's L24\*, the anytime-valid trigger, which supersedes it.
- **L25′ is scoped down.** It guarantees "no generated content in era training". It does NOT make era estimates invariant to the TT, because births fire on the TT router's C = ∅ (selection bias). The invariance claim is withdrawn.
- **E2: "iff" becomes "if".** With f time-constant, ν is identifiable iff Δω(t) is not time-constant.
- **L14′** needs its own lemma: the HMM forward filter recursion α_t(j) ∝ e_j(x_t) Σ_i α_{t−1}(i) A_ij is the posterior over era classes (Bayes, Tier A).
- **Hypotheses added:** L4 needs Σ_{k∈T} p_k > 0. L6 needs exchangeability plus a ties-safe counting lemma. L11 needs equal true laws under H0 and joint-box coverage. L14 needs the query to carry its index (or L14′). L17 needs real coefficients on skew-Hermitian generators. L26 needs v(∅) = 0, and leave-one-out retires both of two duplicate eras (Shapley does not).
- **L16 is RENAMED.** It measures non-commutativity of the generators, not "curvature of law space" (the law space is flat). The earlier "CTC loop measures curvature" wording is withdrawn.
- **L27** loses the chunk-boundary transition ([a],[b]). L27′'s buffer state fixes it, and a test covers [a],[b].
- **L22** "L ≤ L₀/T_min" is not provable. L22\*/L22† supersede it.

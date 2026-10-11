# Amendment D (theory, post-measurement, pre-result)

Filed 2026-10-11. Appended to the frozen spec `2026-10-11-fcpa-design.md` (hash in `FREEZE.md`); that file is not edited and nothing here supersedes it silently. Where this text touches an earlier statement, the touch is named in the lemma's Status or in "Corrections". Lemma IDs follow the theory memos; **L34 is unused (reserved).**

Sources (compiled, not re-derived): `F` = `scratchpad/plan/theory/foreman.md` (L28, L28b, L29); `C` = `scratchpad/plan/theory/cameron.md` (L30, L31); `H` = `scratchpad/plan/theory/chase.md` (L32, L33, L35, L36). Evidence scripts: `fellows/foreman/{l28_check,l28b_velocity,l28b_gamma,l29_check}.py`, `fellows/cameron/toy_L30_L31.py`, `theory/chase_sanity{,2,3}.py`.

"Post-measurement" means: every number below comes from the tuning seed 3 (16 trajectories, 12 train / 4 test, unless stated), from toys, or from earlier builder/nurse reports as quoted in the memos. No eval-seed (0, 1, 2, 4, 5) result enters. "Pre-result" means no FCPA evaluation result exists for the lag-stack input (FREEZE3). Status key: PROVED (proof below), CITED (standard theorem, not reproved), CONJECTURE (gap named), REFUTED (a brief claim shown false). Mathlib means the Lean v4.7.0-era pin in `zero/lean`.

Notation. ‖·‖ is L2 over the sampling measure unless a matrix norm is evident. E_k = k-step closed-loop error, persistence E_pers(k) = ‖x_k − x_0‖. "Bed" = NS-2D seed 3, x ∈ R^90, one step = 10 RK4 steps of dt 0.01.

---

## L28 — closed-loop error identity and bound; the momentum mode

**Statement.** Truth x_{t+1} = F(x_t); lifted state z_t = (x_t, x_{t−1}). Model ẑ_{t+1} = A ẑ_t + c + N(ẑ_t), A the block companion [[I+W₁, W₂],[I, 0]], N a nonlinear remainder (N = 0 for the ridge). Rollout starts at truth. One-step residual along the true trajectory r_j = F̂(z_j) − z_{j+1} = (ρ_j, 0); δ_j = ‖r_j‖_{L2}; under an invariant sampling measure δ_j = δ. s_k = Σ_{j<k} ‖A^j‖₂. ε_k := ẑ_k − z_k.
- (a) Identity (N = 0): ε_k = Σ_{j<k} A^{k−1−j} r_j.
- (b) Bound: ‖ε_k‖_{L2} ≤ Σ_j ‖A^{k−1−j} Σ_j^{1/2}‖_F ≤ Σ_j ‖A^{k−1−j}‖ δ_j ≤ δ s_k, Σ_j = E r_j r_jᵀ.
- (c) Nonlinear remainder: if N is ℓ-Lipschitz on the region both trajectories visit and a := ℓ s_K < 1, then max_{k≤K} ‖ε_k‖ ≤ δ s_K (1 − a^K)/(1 − a) ≤ δ s_K/(1 − a); for a ≥ 1 the factor is (a^K − 1)/(a − 1).
- (d) Beat-persistence criterion. With p_k = ‖x_k − x_0‖ (for an invariant measure p_k² = 2(tr Σ − tr C_k), C_k = Cov(x_k, x_0), and p_k ≤ k·dt·sup‖G‖): E_k(F̂) < E_k(persistence) if δ s_k/(1 − a) < p_k. With ‖A^j‖ ≤ K_A(r) r^j, K_A(r) := sup_j ‖A^j‖ r^{−j} (finite for r > ρ(A), Gelfand), a sufficient condition is δ < (1 − a) p_k (r − 1)/(K_A(r)(r^k − 1)) for r ≠ 1, and δ < (1 − a) p_k/(K_A k) for r = 1.
- (e) Non-normality; clipping does not control it.
  - (e1) ‖A^j‖ ≥ ρ^j always, and no inequality ‖A^j‖ ≤ f(ρ) holds: for A = [[ρ, M],[0, ρ]], A^j = [[ρ^j, jρ^{j−1}M],[0, ρ^j]], so ‖A^j‖ ≥ jρ^{j−1}M, unbounded in M at fixed ρ < 1. Eigenvalue clipping edits Λ only; the off-diagonal and eigenvector geometry (M, κ(V)) is untouched.
  - (e2) Momentum mode: per mode A_a = [[1+a, −a],[1, 0]] has eigenvalues {1, a}; the first row of A_a^m is (1 + g_m(a), −g_m(a)), g_m(a) = a(1 − a^m)/(1 − a). As a → 1 the eigenvectors (1,1) and (a,1) coalesce, κ(V) = O(1/(1 − a)), ‖A^m‖ ~ m, so ρ = 1 but s_k ~ k²/2.
  - (e3) ‖A‖₂ ≤ 1 on the 2-lag companion forces A₁ = 0 (A₁ = I + W₁, the top-left block). Feasible only for n_lags = 1, or in a weighted norm ‖A‖_P ≤ 1 (P a Lyapunov matrix), which costs √κ(P) in K_A.

**Hypotheses (exact).** Rollout started at truth; N = 0 for (a), (b); (c) needs N ℓ-Lipschitz on the common region and a = ℓ s_K < 1 for the first form; (d) needs the (c) hypotheses and r > ρ(A) for K_A finite; (b) in the L2 form needs the sampling measure's second moments; "δ_j = δ" needs invariance.

**Proof.**
(a) ε_{k+1} = Aẑ_k + c − z_{k+1} = Aε_k + (Az_k + c − z_{k+1}) = Aε_k + r_k, ε_0 = 0. Induct.
(b) Triangle inequality (Minkowski in L2) on (a); ‖M r‖²_{L2} = tr(M Σ_j Mᵀ), whence the Frobenius form; then ‖M Σ^{1/2}‖_F ≤ ‖M‖‖r‖.
(c) Put φ_k := ‖ε_k‖. From ε_{k+1} = Aε_k + [N(ẑ_k) − N(z_k)] + r_k, φ_k ≤ Σ_{j<k} ‖A^{k−1−j}‖(δ + ℓφ_j) ≤ δ s_k + ℓ s_k M_{k−1}, with M_k := max_{i≤k} φ_i. Since s_k ≤ s_K, M_k ≤ δ s_K + a M_{k−1}, M_0 = 0; iterate.
(d) E_k(persistence) = p_k by definition; combine with (b)/(c). The K_A form follows from s_k ≤ K_A Σ_{j<k} r^j (geometric sum). Kreiss matrix theorem, sup_j ‖A^j‖ ≤ e·n·K(A) with K(A) = sup_{|z|>1}(|z| − 1)‖(zI − A)^{−1}‖ (n the dimension), is CITED and not needed.
(e1) Induction on j for the displayed power; the operator norm dominates each block. (e2) Induction on m (m = 1, 2 checked by hand). (e3) ‖A(x, 0)‖² = ‖A₁x‖² + ‖x‖² ≤ ‖x‖² for all x forces A₁ = 0.

**Status.** (a)–(e) PROVED. Kreiss CITED. CONJECTURE (gap): the quantitative Schur bound ‖A^j‖ ≤ Σ_{m<n} C(j,m) ρ^{j−m} ‖N_schur‖^m (strictly upper N, product of n of them vanishes) is standard but was not run against the data. Bound (d) is sufficient, not necessary; L28 ranks maps by s_k and explains why growth is possible, it does not predict E50.

**Evidence** (`F`, `l28_check.py`; rms-absolute on 4 test trajectories × 5 origins, 20 samples; not the mean-of-ratios metric of `bars.rel_err`; ordering agrees with Cameron's stage-A table). p_50 = 0.922, δ_rms = 0.0254.

| map | ρ | s_50 | normal bound Σρ^j | max_j‖A^j‖ | ‖A‖₂ | e50 | bound (b) | gain rms‖A^m r‖/rms‖r‖, m = 1/10/49 |
|---|---|---|---|---|---|---|---|---|
| LS ridge 2-lag | 1.0069 | 4195 | 59.4 | 140.9 | 3.04 | 1.624 | 8.07 | 1.88/5.23/7.00 |
| eigclip 1.0 | 1.0000 | 4095 | 50.0 | 132.5 | 3.06 | 1.590 | 7.96 | 1.88/5.22/6.93 |
| eigclip 0.999 | 0.9990 | 4079 | 48.8 | 131.5 | 3.06 | 1.577 | 7.93 | 1.88/5.22/6.92 |
| eigclip 0.99 | 0.9900 | 3789 | 39.5 | 110.2 | 3.03 | 1.399 | 7.42 | 1.88/5.20/6.14 |

The 2-lag ridge has W₁ + W₂ ≈ 0 (‖W₁ + W₂‖/‖W₁‖ = 0.036), i.e. x_{t+1} − x_t ≈ W₁(x_t − x_{t−1}); 29 of 180 companion eigenvalues lie within 0.02 of 1. One step: constant velocity 0.0060 vs persistence 0.0331 (ridge LS 0.0214). Horizon 50: constant velocity E50 1.703 vs persistence 0.655. Clipping ρ 1.0069 → 0.999 moves s_50 by 2.8% (s_50 is 84× the normal-bound value 48.8) and leaves ratio50 ≈ 1.31 (Cameron: E50 1.338). One-step residual amplified ≈ 7× at lag 49 irrespective of clipping. Limits: bound (d) needs δ s_50 < 0.92, actual δ s_50 = 107; bound (b) 8.07 vs measured 1.62 (5× slack from sign cancellation). RED/GREEN: GREEN asserts identity and bound on 4 maps (max rel diff 1.1e-13, 1.3e-13/1.2e-13/9.3e-14); RED `S50(eigclip 0.999) ≤ Σ0.999^j` is 4079.4 ≤ 48.8, False, and "‖A‖₂ ≤ 1 feasible" fails (LS ‖A‖₂ = 3.04, ‖A₁‖_F = 17.7), both as intended.

**Lean plan** (names from `F`, recalled, NOT grepped against the pin). (a), (b)-pointwise, (c): `Finset.sum_range_succ`, induction, `ContinuousLinearMap.le_opNorm`, `norm_sum_le`, `norm_pow_le'` on `E →L[ℝ] E`, `geom_sum_eq`. L2 form of (b): `MeasureTheory.Memℒp.snorm_add_le` iterated (`snorm` predates `eLpNorm`). (e1): `Matrix` power of `!![ρ, M; 0, ρ]` by induction, `Matrix.linfty_opNorm`. Gelfand: `spectrum.pow_nnnorm_pow_one_div_tendsto_nhds_spectralRadius`. (e3): `le_opNorm` on `WithLp 2 (E × E)`. Missing in Mathlib at the pin: Kreiss, Schur triangularization.

---

## L28b — shrunk increment model: one step versus horizon

**Statement.** Stationary increments Δ_t = x_{t+1} − x_t; undamped increment model m(z_t), damped model γm. P = ‖Δ‖ (persistence one-step error), μ = ‖m‖/P, c = ⟨m,Δ⟩/(‖m‖P) (alignment cosine), η = m − Δ, δ_m = ‖η‖.
- **Prop 1 (one step, exact).** e₁(γ)² = P²(1 − 2γμc + γ²μ²); δ_m² = P²((μ − c)² + 1 − c²); the minimum over γ is at γ₁* = c/μ = ⟨m,Δ⟩/‖m‖², with e₁² = P²(1 − c²). The bar e₁ ≤ θP (θ = ½) is attainable iff c² ≥ 1 − θ² = ¾ (c ≥ 0.866, angle ≤ 30°), and then γμ ∈ [c − √(c² − ¾), c + √(c² − ¾)]. Shrinkage repairs MAGNITUDE (μ) never DIRECTION (c). In δ_m terms: δ_m ≤ P/2 is sufficient for attainability (δ_m² ≥ P²(1 − c²)); no condition on δ_m alone is necessary.
- **Prop 2 (horizon, exact, any linear closed loop).** z = lag stack (centred, intercept included), M_{γ,k} = first block row of A_γ^k, A_γ = J + γ(A − J), J = [[I,0],[I,0]] (persistence, J² = J), M*_k the L2-optimal linear k-step map, Σ = Cov(z). Then E_k(γ)² = D_k² + ‖(M_{γ,k} − M*_k)Σ^{1/2}‖_F² and E_pers(k)² = D_k² + Δ_k, Δ_k = ‖([I 0] − M*_k)Σ^{1/2}‖_F², D_k = best-linear error. So E_k(γ) < E_pers(k) iff M_{γ,k} is Σ-nearer M*_k than persistence is, and the most any linear predictor gains is Δ_k = E_pers² − D_k².
- **Prop 3 (momentum channel, closed form).** Class x̂_{t+k} = x_t + g v_t, v_t = x_t − x_{t−1}; damped momentum map x_{t+1} = x_t + a v_t (a = γw) has closed-loop g = g_k(a) = a(1 − a^k)/(1 − a) = Σ_{i=1..k} a^i, strictly increasing in a > 0. With α_l = ⟨v_{t+l}, v_t⟩/‖v‖², D_k = x_{t+k} − x_t, g*_k = ⟨D_k, v_t⟩/‖v‖² (= Σ_{i≤k} α_i):
  - (i) e₁(a)² = P²(1 − 2aα₁ + a²); the one-step bar holds iff a ∈ [a₋, a₊], a_∓ = α₁ ∓ √(α₁² − ¾) (needs α₁ ≥ √3/2).
  - (ii) E_k(a)² = p_k² − g*_k²‖v‖² + (g_k(a) − g*_k)²‖v‖². Beats persistence iff g_k(a) lies strictly between 0 and 2g*_k. Optimal a*(k): g_k(a*) = g*_k (unique). a*(k) < α₁ iff Σ_{i≤k} α₁^i > Σ_{i≤k} α_i (true whenever α_i ≤ α₁^i). Gain ceiling (p_k² − E_k²)/p_k² ≤ g*_k²‖v‖²/p_k².
  - (iii) EXACT FEASIBILITY (both bars, one γ): α₁ ≥ √3/2 AND g*_k > 0 AND Σ_{i=1..k} a₋^i < 2g*_k, a₋ = α₁ − √(α₁² − ¾). Window a ∈ [a₋, a_k^hi], g_k(a_k^hi) = 2g*_k.

**Hypotheses (exact).** Stationarity of increments; L2 quantities as defined; Prop 3 uses one pooled velocity channel (per-mode a_i = γw_i replaces the sums by Σ-weighted ones, Prop 2 is the general form); Prop 2 needs the normal equations E[(Y − M*z)zᵀ] = 0 (centred, intercept included).

**Proof.**
Prop 1: r = γm − Δ; expand ‖r‖²; minimise the quadratic in γ; e₁² ≤ θ²P² iff 1 − 2γμc + γ²μ² ≤ θ², a quadratic in γμ with discriminant ≥ 0 iff c² ≥ 1 − θ². The δ_m formula: expand ‖m − Δ‖² with ‖m‖ = μP.
Prop 2: normal equations orthogonalise the residual of M* against z; Pythagoras on M_{γ,k}z − Y = (M_{γ,k} − M*_k)z + (M*_k z − Y), and on [I 0]z − Y likewise.
Prop 3 (the memo states Prop 3 without separate proof text; the steps below are the direct derivation from its statement): the closed form of g_k is (e2) of L28 with a = γw. (ii) expands E_k(a)² = ‖D_k − g v‖² = p_k² − 2g⟨D_k, v⟩ + g²‖v‖² and completes the square at g*_k; "beats persistence" is |g − g*| < |g*|. Monotonicity of g_k in a gives uniqueness of a*(k) (intermediate value theorem). (iii) combines (i) and (ii): the smallest feasible a for the one-step bar is a₋; the horizon bar then requires g_k(a₋) < 2g*_k, and g_k increasing gives the window.

**Verdict on the brief's claim "no single γ meets one-step ≤ 0.5 × persistence AND E50 < persistence unless δ_m ≤ …": REFUTED** (see Corrections).

**Status.** Prop 1–3 PROVED; claim REFUTED. The feasibility sign of (iii) is not statistically resolved on 12/4 trajectories (g*_50 differs in sign between splits).

**Evidence** (`F`, `l28b_gamma.py`, `l28b_velocity.py`).
- Witness in builder data (quoted from the brief, not re-run): damping 0.7 with energy: 0.0114/0.0284 = 0.40 ≤ 0.5 and E50 0.622 < 0.6335; a second row 0.0154 / 0.595. The energy-projected model lies outside the momentum class of Prop 3.
- Toy, momentum, test split: α₁ = 0.977, a₋ = 0.524, g*_50 = 1.23, a_k^hi = 0.712; brute-force window [0.524, 0.711] equals the prediction (assert GREEN); at a = 0.524 one-step = 0.499 P, E50 = 0.998 pers. Margin: ceiling gain g*²‖v‖²/p² ≈ 0.4% in E²; best a*(50) = 0.553 gives E50 = 0.9975 pers.
- 12 train trajectories: g*_50 = −0.10, so (iii) is INFEASIBLE there; α₁ = 0.958, α₁₀ = −0.04, α₂₅ = −0.12, Σ_{i≤50} α_i = 0.27, against 22.9 for a geometric memory at α₁. The velocity forgets in ~10 steps; AR(1) with α₁ = 0.96 would remember ~25.
- One-step is a direction (c) condition that any α₁ ≥ 0.866 passes (bed α₁ = 0.96–0.98: constant velocity passes at 0.18× with E50 = 2.6× persistence). A 1.8%–6% E50 win (builder 0.622, 0.595) must come from the nonlinear part.
- RED (`l28b_gamma.py`): "no a meets both bars" evaluates False, as it should.

**Lean plan.** Prop 1: real-quadratic algebra (`discrim`, `quadratic_le_zero_iff` / `nlinarith`), `inner_sub_sub_self`, `norm_sub_sq_real`. Prop 2: `Submodule.orthogonalProjection`, `norm_sq_eq_add_norm_sq_projection`; finite-sample form in `EuclideanSpace ℝ (Fin n)`. Prop 3: `geom_sum_eq`, `Finset.sum_lt_sum`, `pow_lt_pow_left`, `intermediate_value_Icc` for a*(k). All Lean-able; names unverified against the pin.

---

## L29 — linear insufficiency (corrected)

**Statement.** Aff = closed span of {1, x_i} in L2(μ); P_aff its orthogonal projection; D_lin(τ) := min over affine (M, m) of ‖Φ_τ − Mx − m‖_{L2(μ)}; Σ = Cov(x), C_τ = Cov(Φ_τ x, x), g := ‖DG·G‖_{L2(μ)} (rms acceleration), q := min_{M,m} E‖B(x,x) − Mx − m‖².
- (a) Exact gap. E_pers(τ)² = D_lin(τ)² + tr[(C_τ − Σ)Σ^{−1}(C_τ − Σ)ᵀ] (mean terms cancel by invariance). Hence D_lin ≤ E_pers always, with equality iff C_τ = Σ.
- (b) Nonlinearity floor. D_lin(τ) ≥ τ√q − (τ²/2) g. So D_lin > 0 on (0, 2√q/g), and the bound increases until τ = √q/g (peak q/(2g)).
- (c) Composition gap (exact). For any one-step map A, E_comp(k)² = D_lin(k)² + ‖(A^k_{x·} − M*_k)Σ_z^{1/2}‖_F², so the composed map beats persistence iff A^k is Σ-nearer M*_k than [I 0] is. The one-step objective does not control that distance (L28).
- (d) Climatology. E_pers(τ) < E_clim = √(tr Σ) iff tr C_τ > ½ tr Σ.
- The literal claim "persistence beats every linear k-step map" is REFUTED: persistence is the linear map M = [I 0]. What holds: persistence beats the COMPOSED one-step maps A^k, and the room above persistence for any linear map is Δ_k.

**Hypotheses (exact).** H1: G(x) = −νKx + B(x,x) + f, K symmetric positive definite, B symmetric bilinear (energy conservation ⟨B(x,x),x⟩ = 0 only gives an absorbing ball and moments, not used in the proofs). H2: μ an invariant probability for the flow Φ_t with finite 6th moment (then Ġ := DG·G ∈ L2). H3: q > 0.

**Proof.**
(a) The affine LS residual is orthogonal to Aff; Pythagoras on Φ_τ − x = (Φ_τ − M*x) + (M* − I)x with M* = C_τΣ^{−1}.
(b) Φ_τ x = x + τG(x) + R_τ, R_τ = ∫₀^τ (τ − s) Ġ(Φ_s x) ds. The map x + τ(−νKx + f) is affine, so (I − P_aff)Φ_τ = τ(I − P_aff)B(x,x) + (I − P_aff)R_τ, and (I − P_aff) is a contraction, so ‖(I − P_aff)Φ_τ‖ ≥ τ√q − ‖R_τ‖. Cauchy–Schwarz in s with weight (τ − s) (∫w = τ²/2), Fubini, and invariance E‖Ġ(Φ_s x)‖² = g² give ‖R_τ‖_{L2} ≤ (τ²/2) g. No Lipschitz or sup constants are needed (the worst-case L_G Γ form was computed and is vacuous).
(c) Same Pythagoras as Prop 2 of L28b.
(d) E_pers² = 2(tr Σ − tr C_τ) < tr Σ iff tr C_τ > ½ tr Σ.

**Status.** (a)–(d) PROVED. CONJECTURE: for large τ, if correlations decay (mixing of the NS attractor, not proven), D_lin → √(tr Σ) (D_lin ≤ √(tr Σ) always, by the constant map). Bound (b) is certified only for τ < √q/g (vacuous past 2√q/g).

**Evidence** (`F`, `l29_check.py`; Lorenz-96 N = 8 for identities; the bed for E50).
- Lorenz-96: identity (a) residual ≤ 2.5e-16 at 7 horizons; bound (b) holds at 7/7 horizons; √q = 46.8, g = 527.8, peak τ = 0.089, vanishes at τ = 0.177 (3.5 steps of 0.05); energy conservation ⟨B(x,x),x⟩ = 0 to 1e-9. Correlation ≈ 0 by τ = 0.4: E_pers 14.335 vs E_comp 10.2 vs E_lin 8.667 at j = 8 (RED of the literal claim evaluates False).
- Bed: direct 50-step ridge head on [x_t, x_{t−1}], 2988 rows, 180 features, λ by 3-fold over train trajectories: TEST E50 = 1.566 (λ 1e-2), 1.257 (1), 1.036 (100), 0.786 (1e3), 0.663 (1e4), 0.658 (1e5, CV choice), 0.671 (1e6), vs persistence 0.6552 and composed LS 1.3686. Train-origin E50 at λ = 1e5 is 0.652 vs persistence 0.669 (2.6% below). On the bed E_pers(50) = 0.655 gives trace-correlation ≈ 0.785 > ½, so any predictor relaxing to the mean (relative error ≈ 1) loses to persistence; an unstable composed map overshoots further (1.37).
- Supported: no linear map, direct or composed, beats persistence by more than Δ_50, and Δ_50 ≈ 0 on this bed, so a winner at k = 50 must be nonlinear (B(x,x) off the span of x). Not supported: "persistence beats the best linear map" (impossible).

**Lean plan.** (a), (c): `Submodule.orthogonalProjection`, `norm_sq_eq_add_norm_sq_projection`, `Matrix.trace_mul_comm`, `Matrix.mul_nonsing_inv`, `Matrix.PosDef`. (b): flow as a hypothesis `Phi` with `HasDerivAt` (Picard–Lindelöf availability at the pin unverified); Taylor remainder via `intervalIntegral.integral_eq_sub_of_hasDerivAt`; Cauchy–Schwarz `inner_mul_le_norm_mul_norm` / `integral_mul_le_Lp_mul_Lq`; Fubini `MeasureTheory.integral_integral_swap`; invariance `MeasureTheory.MeasurePreserving.integral_comp`. Minkowski's integral inequality is NOT in Mathlib; the Cauchy–Schwarz route avoids it. (d): one-line algebra.

---

## L30 — quadratic-library era model: exact and identifiable

**Setting.** `FCPARegressor`, `n_lags = 2`, `law = NS2DLaw()`, bed NS-2D. x_t ∈ R^D, D = 90: raw truncated rfft2 vorticity (9 kx × 5 ky = 45 complex modes; real parts then imaginary parts), `x = a·z`. Rows s_t = [x_t, x_{t−1}] (180); target y_t = x_{t+1} − x_t (h = 0.1). Model per era: Δx = A₁x_t + A₂x_{t−1} + W q(x_t) + c, q(x) = (x_j x_k)_{j≤k} over a fixed coordinate set J of r real coordinates, feature row z_t = [s_t, q(x_t)] ∈ R^d, d = 2D + r(r+1)/2. Fit: the existing centred ridge of `EraModel` (Chan-merged `n_, mu_, mu1_, M_, C_, T_`), unchanged.

**F0 (redundancy).** The 90 coordinates have rank 80: the 4 conjugate pairs (±kx, ky = 0) give 8 dependent reals and the mean mode (0,0) gives 2 zero reals (CHECKED: rank of 400 random Hermitian band-limited fields in R^90 is 80). Every Gram is singular by construction; "identifiable" is read modulo this 10-dimensional null space, D_eff = 80.

**Statement.**
- **(a) Exactness (extends L27′).** For any fixed data-independent ψ: R^{2D} → R^d applied row by row, chunked `partial_fit` on feature rows ψ(s_t) with targets y_t equals batch `fit` on the concatenation, in exact arithmetic, for every ridge level and the rank guard. Hypotheses: (H1) ψ depends on one row only; (H2) J and any scaling are fixed before the first chunk (same status as `y_edges`); (H3) chunks are in time order and carry whole stacked rows, as `_train` in `test_estimator_ns3.py` does.
- **(b) Identifiability.** With centred feature Gram G_c = Σ(z_t − z̄)(z_t − z̄)ᵀ: (1) for λ > 0 the ridge solution (A₁, A₂, W) is unique; (2) for λ → 0, W is uniquely determined iff R_q = q − proj(q | 1, s) has full column rank on the sample, i.e. the Schur complement G_qq − G_qs G_ss^+ G_sq is nonsingular on the identifiable quotient; equivalently no nonzero quadratic form wᵀq(x_t) is an affine function of (x_t, x_{t−1}) on every sample row; (3) n ≥ 1 + rank(s block) + rank(q block). With F0 the effective ranks are 160 and 3240 for the full library, so n ≥ 3401 rows; for a reduced library n ≥ 1 + 160 + r(r+1)/2, and the repo rule "3 rows per feature" gives n ≥ 3(160 + r(r+1)/2).
- **(c) Cost** (D = 90, float64, per era): per-row update d² multiply-adds; state bytes 8(d² + dp + p²), p = 90; solve d³/3 flops per `_finalize`.

| library | r | q | d = 180 + q | Gram + cross + T | per-row update | solve |
|---|---|---|---|---|---|---|
| linear (today) | 0 | 0 | 180 | 0.4 MB | 3.2e4 | 1.9e6 |
| low-k / POD | 12 | 78 | 258 | 0.78 MB | 6.7e4 | 5.7e6 |
| low-k / POD | 20 | 210 | 390 | 1.56 MB | 1.5e5 | 2.0e7 |
| low-k / POD | 40 | 820 | 1000 | 8.8 MB | 1.0e6 | 3.3e8 |
| full ambient | 90 | 4095 | 4275 | 149.4 MB | 1.8e7 | 2.6e10 |

**Proof.**
(a) For chunks a, b with n_a, n_b rows, means μ_a, μ_b and centred scatters M_a, M_b, the scatter of the union is M_a + M_b + (n_a n_b/(n_a + n_b))(μ_a − μ_b)(μ_a − μ_b)ᵀ: expand Σ over the union of (z − μ)(z − μ)ᵀ with μ = (n_aμ_a + n_bμ_b)/n, and use Σ_a(z − μ_a) = 0; the cross-scatter C and target scatter T follow with the same algebra (exactly `EraModel.partial_fit`). So the state is a monoid homomorphism of the chunk list and state(chunks) = state(concatenation). `_solve` reads only (n_, mu_, mu1_, M_, C_, T_) and constants (`lam`, `beta`, `min_rank_frac`); `scale = tr(G)/(dn)`, the ridge and the rank-guard loop (λ /= 10) are functions of that state. Hence W_, b_ coincide.
(b) (1) G_c is positive semidefinite (sum of rank-one terms) and λ·scale > 0 when tr G_c > 0, so G_c + λ·scale·I is positive definite and the normal equations have a unique solution. (2) With z = [s, q], the LS minimiser is unique iff G_c is nonsingular on span(coordinates); block elimination of the s block shows the q-block is determined iff the Schur complement is nonsingular, and a kernel vector is exactly a w with wᵀ(q_t − proj_t) = 0 for all t. (3) rank(G_c) ≤ min(n − 1, d_eff).
(c) Counting; the measured memory is below.

**Pitfalls.** (i) A data-dependent J (top-r POD of x) breaks (H2). Fix: store the FULL ambient q (D(D+1)/2) and form any chart's quad block post hoc: for a linear map P, q(Px) = Sym²(P)q(x), so the reduced Gram is L G_qq Lᵀ (exact, like the existing chart); cost in (c). (ii) `_shrink` (gam, gam_tt) uses held-out windows that are functions of the same buffer and are unchanged.

**Status.** (a) PROVED, CHECKED (`EraModel(guard=False)` on [x, q(x)] features of a 6-dimensional toy, 7 uneven chunks vs batch: max |W diff| 2.6e-15, max |b diff| 2.2e-16). (b) PROVED. (c) arithmetic. CONJECTURE C1: the 4 NS regimes' data satisfy (b)(2) (attractor not on a quadric hypersurface); falsified by a rank deficiency of G_c at fit time, so detected not assumed. Era size: eras are ≥ 60 rows and share ≈ n/E of the data, so only pooled or large eras can carry r ≈ 40; small eras need r ≤ 20 or the linear fallback (finite per-era library size r_k(n_k) with 3n_k ≥ 160 + r_k(r_k+1)/2).

**Evidence** (`C`, stage C, one process). Gram + cross measured 0.7 / 2.2 / 8.7 / 149.3 MB for 258 / 480 / 1000 / 4275 features (the 480 row is lowk m = 12, 24 real coordinates with the 180 linear features). The full library costs 149 MB per era in float64 (75 MB float32); n_eras_max = 8 is 1.2 GB, over the 1 GB per-process budget, and solves a 4275² system per era per `_finalize` (2.6e10 flops, per `partial_fit` call). Top-r with r ≤ 40 fits in under 9 MB per era. A frozen coordinate set J (data-independent low-|k| modes, as stage C chose) is the cheap option; full-ambient storage is the expensive one.

**Lean plan** (`C`; names checked by grep in `zero/lean/.lake/packages/mathlib`). (a): same style as `FCPA/L27_PartialFit.lean` (`Finset.sum` over append, `field_simp; ring`), easy. (b)(1): `Matrix.PosDef` of G + λI by hand (`PosDef.add_posSemidef` not found), `Matrix.posSemidef_conjTranspose_mul_self`, `Matrix.PosDef.det_pos`, `Matrix.isUnit_iff_isUnit_det`, `Matrix.dotProduct_mulVec`; easy–medium. (b)(2): block elimination on `Matrix.fromBlocks`, kernel argument; medium.

---

## L31 — energy-conserving constraint, with Parseval M-weights

**Facts (supporting).**
- **F1 (metric of the invariant).** M = diag(m), m = 1 on ky = 0 and m = 2 on ky > 0 (Parseval multiplicity of the half plane). xᵀMx is the enstrophy of the truncated field and xᵀMK^{−1}x its energy (K = diag k²). CHECKED on the solver's dealiased `spectral_terms`, 20 random band-limited fields: |⟨x, MB⟩|/(|x||B|) = 1.4e-16 (enstrophy), 2.6e-17 (energy), but 4.6e-2 for the unweighted Euclidean product. The Euclidean statement ⟨x, B(x,x)⟩ = 0 is false in these coordinates; the true statement carries M.
- **Lemma B0 (Galerkin bracket conserves both).** For real periodic w with Fourier support in a set closed under k → −k, P the Fourier projector onto it, B(w) = P(u·∇w), u = ∇^⊥ψ, −Δψ = w: ∫ψB(w) = 0 and ∫wB(w) = 0. Proof: P is self-adjoint with Pw = w, Pψ = ψ, so ∫wP(u·∇w) = ∫w(u·∇w) = ½∫u·∇(w²) = −½∫(div u)w² = 0 (periodic, div u = 0). Likewise ∫ψ(u·∇w) = −∫w(u·∇ψ) = 0 since u·∇ψ = 0. Parseval turns the integrals into the M-weighted and MK^{−1}-weighted coordinate sums. PROVED, CHECKED (F1).
- **F2 (truncation is not closed).** The retained 45 modes feel triads with unretained modes (the solver runs to |k| ≤ 10). The true resolved enstrophy flux ⟨x, MPB(w_full)⟩ is not zero; it equals minus the flux into the unresolved band. The Galerkin bracket of the 90 retained coordinates is a hypothesis class, not the data-generating law.

**Statement.** Notation: W has entries w_{i;jk} (output i, feature x_jx_k, j ≤ k); Q(x)_i = Σ_{j≤k} w_{i;jk}x_jx_k; M = diag(m), m_i > 0.
- **(i) Subspace.** ⟨x, MQ(x)⟩ = 0 for all x ∈ R^r iff for every multiset {a,b,c}: a,b,c distinct: m_a w_{a;bc} + m_b w_{b;ac} + m_c w_{c;ab} = 0; {a,a,b}, a ≠ b: m_a w_{a;ab} + m_b w_{b;aa} = 0; {a,a,a}: m_a w_{a;aa} = 0. The solution set Wcons_M is a linear subspace of dimension r(r² − 1)/3, the same for every positive M. Equivalent skew form (M = I): Q(x) = S(x)x with S(x)_{ij} = Σ_k s_{ijk}x_k, s skew in (i,j); for general M replace S by M^{−1}Σ(x), Σ skew.
- **(ii) Constrained ridge, closed form.** Let G_λ = G_c + λ·scale·I, w_u the unconstrained ridge solution (vectorised over outputs), R the constraint matrix of (i) on the q-block coordinates (zero on the linear block and intercept; one row per multiset). Then w_c = w_u − (G_λ^{−1} ⊗ I_p)Rᵀ S^{−1} R w_u, S = R(G_λ^{−1} ⊗ I_p)Rᵀ; S is positive definite and w_c is the unique minimiser of ‖Y − ΦW‖² + λ·scale‖W‖² over Rw = 0. It uses only G_λ and C, the additive statistics of L30(a).
- **(iii) Energy law.** ⟨u,v⟩_M = uᵀMv; sym_M(A) = (A + A*)/2 with A* = M^{−1}AᵀM; μ_M(A) = max eigenvalue of sym_M(A).
  - (E1) Continuous time: if x′ = Ax + Q(x) + c with Q ∈ Wcons_M then d/dt‖x‖²_M = 2⟨x, Ax + c⟩_M exactly (the brief's "≤" is an equality), hence d/dt‖x‖_M ≤ μ_M(A)‖x‖_M + ‖c‖_M, and if μ_M(A) < 0 then limsup‖x‖_M ≤ ‖c‖_M/|μ_M(A)|.
  - (E2) Discrete, exact: for x_{t+1} = x_t + Δ_t, Δ_t = Ax_t + c + Q(x_t) (position law, A₂ = 0): ‖x_{t+1}‖²_M = ‖x_t‖²_M + 2⟨x_t, Ax_t + c⟩_M + ‖Δ_t‖²_M. The explicit step adds δ² = ‖Δ‖²/‖x‖² per step (forward-Euler energy growth); the growth factor per step is 1 + 2μ_M(A) + δ², below 1 iff |μ_M(A)| ≥ δ²/2.
  - (E3) Energy-shell rescale at inference (no effect on fitting or L27′): after the step, rescale x_{t+1} by √(‖x_t‖² + 2⟨x_t, Ax_t + c⟩_M)/‖x_{t+1}‖_M (when the radicand is positive). Then ‖x_{t+1}‖²_M = ‖x_t‖²_M + 2⟨x_t, Ax_t + c⟩_M exactly, so rollout energy obeys only the linear/forcing law of (E1). The correction is a relative O(δ²) ≈ 4e-4 move, below the model's one-step error (7.5e-3 on the full bed).
  - (E4) Lag terms are NOT covered. For A₂ ≠ 0, write x_{t−1} = x_t − v_t, v_t = Δ_{t−1}: ⟨x_t, A₂x_{t−1}⟩_M = ⟨x_t, A₂x_t⟩_M − ⟨x_t, A₂v_t⟩_M, with |⟨x_t, A₂v_t⟩_M| ≤ ‖A₂‖_M‖x_t‖δ_{t−1}‖x_t‖, a first-order-in-δ term (2‖A₂‖δ ≈ 0.05‖A₂‖ against damping ≈ 1e-3). For A₂ = −αI the term is −(α/2)(‖x_t‖² − ‖x_{t−1}‖² + ‖v_t‖²), an AR(1) filter on the energy increment, stable iff |α| < 1 in every mode. The Frobenius-nearest A with μ_M(A) ≤ 0 is A′ = skew_M(A) + P_NSD(sym_M(A)) (the two parts are Frobenius-orthogonal, so the nearest-point problem separates).
- **(iv) Learned, not given (L-GIVEN).** Assumed of the hypothesis class: a positive diagonal metric M for which the nonlinear part has zero flux. With M = the half-plane Parseval multiplicities this is a property of how the encoder indexes the half spectrum plus the conservative-transport structure; no ν, no forcing pattern, no k² weights, no triad selection rule p + q = k, no number of regimes. Everything else is learned: A₁, A₂, c, every tensor entry of W inside the subspace (the triad selection rule is not imposed and can only appear as learned near-sparsity), the eras and their laws. Learned-M variant (data only): M enters linearly (Sym(MW) = 0); for a fitted W_u the best conserved metric is the smallest singular vector of M ↦ Sym(MW_u) on the PSD cone with tr M = 1.
- **Contract compatibility.** Chart (`era_chart`, `set_chart`, `d_c = N` for `n_lags > 1`): the chart is an orthogonal gauge of the linear block, so linear coefficients are unchanged; the quad block is defined on ambient x (recovered exactly from φ by `_rows_from_phi`, injectivity E1) and is chart-free; with a frozen coordinate set it is also exact (L30(a)). Crossing operator S_k (D8, L15/L17): S is unitary on ψ; for the FULL library q(Sx) = Sym²(S)q(x) is linear so statistics transform exactly; conservation is preserved iff S preserves the invariant metric; S is Euclidean-unitary, so rescale x by M^{1/2} (fixed, data-independent) before the encoder, after which Wcons is invariant under orthogonal changes of basis (W ∈ Wcons ⇒ S W Sym²(S^{−1}) ∈ Wcons, since ⟨Sx, SQ(S^{−1}Sx)⟩ = ⟨x, Q(x)⟩). Without the rescale the crossing can map a conservative era into a non-conservative one. Shrink γ (toward persistence) scales the whole increment, Wcons is a cone: kept. Blend (damped CTC loop): ⟨x, Σ_k w_k(x)Q_k(x)⟩ = Σ_k w_k(x)⟨x, Q_k(x)⟩ = 0 pointwise, so any-weight blends of conservative eras are conservative (linear parts blend normally): kept. Router (Mahalanobis or law-residual scores) depends on ρ and law statistics, not the predictor: unchanged. The TT (softmax over bins) answers ALONE/OOD rows and is not a quadratic model: **the energy guarantee covers era-routed rows only.** L27′ kept (L30(a)); the solve at d = 4275 is paid at every `partial_fit` (cache the solve per era version).

**Hypotheses (exact).** (i)–(ii): m_i > 0; (ii) K = G_λ ⊗ I_p positive definite (λ > 0), R of full row rank; (iii) Q ∈ Wcons_M; (E2) position law A₂ = 0; (E3) positive radicand.

**Proof.**
(i) ⟨x, MQ⟩ = Σ_i m_i x_i Σ_{j≤k} w_{i;jk}x_jx_k is a cubic form. The monomial x_ax_bx_c (as a multiset) arises exactly from the triples (i; jk) with {i,j,k} = {a,b,c}: three when distinct (i = a, jk = bc; i = b, jk = ac; i = c, jk = ab), two for {a,a,b}, one for {a,a,a}. A real cubic form is identically zero iff all monomial coefficients vanish (extract by finite differences at ±e_a ± e_b ± e_c). Each w_{i;jk} occurs in exactly one constraint, so rows have disjoint column supports and are linearly independent (all m > 0). Dimension = r·r(r+1)/2 − C(r+2,3) = r(r² − 1)/3. Skew form: x^T S x = 0 puts it in Wcons; the map s ↦ Q has kernel the totally antisymmetric tensors (dimension C(r,3)), so the image has dimension r²(r−1)/2 − C(r,3) = r(r² − 1)/3 = dim Wcons, hence equal.
(ii) Objective ½wᵀKw − gᵀw, K = G_λ ⊗ I_p, g = vec(C). Lagrangian stationarity Kw + Rᵀμ = g, Rw = 0 gives w = K^{−1}(g − Rᵀμ); Rw = 0 gives Sμ = RK^{−1}g = Rw_u. S = RK^{−1}Rᵀ is positive definite because K^{−1} is and R has full row rank (disjoint supports). Substitute μ. Strict convexity gives uniqueness. Each row of R has at most three nonzeros, so S has entries that are sums of at most nine entries of G_λ^{−1}; assembling S costs O(#multisets²), #multisets = C(r+2,3) (r = 12: 364 rows; r = 20: 1540 rows, S is 19 MB; r = 40: 11,480 rows, 1.1 GB dense, so use matrix-free CG for r ≥ 40). A post-solve step; one extra O(#multisets³/3 + d³) solve per `_finalize`. Identifiability: unique for λ > 0 as in L30(b)(1); for λ → 0 the Schur condition of L30(b)(2) restricted to Wcons (fewer unknowns, strictly weaker than the unconstrained condition).
(E1) Differentiate; the Q term is ⟨x, MQ⟩ = 0; Gronwall on ‖x‖_M. (E2) Expand ‖x + Δ‖²_M; the Q cross term vanishes. (E3) Algebra from (E2): the radicand is the prescribed ‖x_{t+1}‖²_M. (E4) Expansion above; the nearest-point statement: sym_M and skew_M parts are Frobenius-orthogonal, so minimise each separately (skew part is free; the symmetric part is clipped to NSD).
(iv) Learned-M variant: the source states only that M enters the constraint linearly (Sym(MW) = 0), i.e. (i) with m as the unknown.

**Status.**
- PROVED: B0, (i) including the dimension and skew form, (ii), (E1)–(E3), the E4 nearest-point separation (the two parts are Frobenius-orthogonal).
- CHECKED: nullity of the constraint matrix is 20 (r = 4) and 70 (r = 6) with random positive M, formula 20 and 70; r = 90 gives dimension 242,970 of 368,550 coefficients. Constrained ridge on a 6-dimensional conservative toy plus noise: max |⟨x, MQ(x)⟩| = 5.3e-15 over 500 random x constrained vs 1.7e1 unconstrained; one-step RMSE 2.995e-2 vs 2.993e-2 (the constraint costs no accuracy when the truth is conservative). (E2) relative residual 4.9e-16 over a 300-step rollout constrained vs 1.9e-2 unconstrained.
- CONJECTURE: (C2) the enstrophy and energy subspaces intersect in the generic dimension (the energy constraint is the same system with m_i/k_i²; rank of the stacked constraint = rank(R_enst) + rank(R_energy) generically; exact count unproved; falsified by a rank computation). (C3) reduced libraries are closed under the crossing S unless J is a union of S-invariant coordinate blocks (commuting generators of `diff.make_generators`; falsified by one commuting-generator example). (C4) the learned M recovers the Parseval weights (uniqueness and positivity of the singular vector). (C5) lag-term stability needs a Lyapunov LMI CᵀPC ≤ P with no closed form known; use of A′ on the 2-lag companion is CONJECTURE because ‖C‖₂ ≥ √(1 + ‖A₁‖²) > 1 for the block companion (Euclidean contraction impossible by structure). (C6) the retained-band enstrophy drain (F2) is absorbed by the linear part without hurting E50.
- The L-GIVEN definition in the PREREG was not read by the author of `C`; "structure learned, not given" is what `zero/CLAUDE.md` states. A one-line check against the amendment text is required before any claim of compliance; the learned-M variant is the fallback if L-GIVEN forbids any equation-derived invariant.
- Guarantee as it stands: the nonlinear part cannot inject energy (E1, E2) up to the δ² term (removed by E3); the linear and lag parts must be made dissipative separately. **No model meeting the bar has been shown.**

**Evidence** (`C`, seed 3, 12 train / 4 test, persistence one-step 0.02685, E50 0.6552; `toy_L30_L31.py` for identities).
- Undamped 2-lag LS: one 0.02144, E50 1.3686, ratio50 1.329, ρ 1.0069. Damping 0.5: one 0.01914, E50 0.8166. Estimator (eras, shrink): damping 1.0 one 0.04162 / E50 1.1325; damping 0.5 one 0.02999 / E50 0.8589.
- Unconstrained quadratic ridge (stage C): at λ ≤ 0.01 the rollout diverges (E50 = inf for r = 12, 24, 40 and full, except one 5.7e51), as (E2) predicts; at λ = 1 it is stable (E50 0.88–0.93) but one-step 0.031–0.038 is worse than persistence. No linear or quadratic variant tested reaches E50 < 0.655.
- Stage A: clipping the companion spectrum to ρ ≤ 1 left E50 = 1.346 (struct-eigclip r = 1.0) and 1.338 (r = 0.999); nuclear-norm 1-lag E50 = 0.945.
- Discrete margin: persistence one-step error 0.02685 gives δ² = 7.2e-4 per step (50 steps compound to at most 3.6% in energy); the weakest retained mode (|k|² = 1, ν = 0.010, h = 0.1) damps at 1.0e-3 per step, a factor 2.8 of margin, a property of the true flow and not of the fitted A.
- The bar "one-step ≤ 0.5 × persistence" cannot be met by any linear 2-lag model on 12 training trajectories (0.80× at best); `test_estimator_ns3.py` already says 24 trajectories are too few (docstring of `test_ns3_lag_stack_beats_persistence_on_the_dev_bed`). The 0.5× bar belongs to the gated 128-trajectory test; on the 16-trajectory sample only E50 is a fair RED/GREEN.
- Proposed tests, NOT run: **T1** (premise, RED if false): f = ‖W_u − ΠW_u‖_F/‖W_u‖_F (Π the block-local projector of (i)) against f_perm with the output index permuted; survives if f ≤ 0.5 f_perm (random tensor: f² ≈ 125,580/368,550, f ≈ 0.58). **T2** (the fix): constrained ridge + (E3) + dissipative linear part vs unconstrained and persistence on the same 16 trajectories; RED is the current model E50 = 1.1325 > 0.655; bar E50 < 0.655 AND one-step ≤ 0.5 × persistence. **T3** (learned M): gap between smallest and second-smallest singular value of M ↦ Sym(MW_u) on the PSD cone, and the cosine between recovered M and the Parseval weights.

**Lean plan** (`C`; names grepped in `zero/lean/.lake/packages/mathlib`).

| item | plan | Mathlib pieces found | effort |
|---|---|---|---|
| (i) subspace and dimension | cubic-form coefficient extraction by finite differences at ±e_a ± e_b ± e_c; disjoint-support rank via rank–nullity | `LinearMap.finrank_range_add_finrank_ker` | medium–hard |
| (ii) KKT closed form | complete the square with K^{−1}; S positive definite | `Matrix.PosDef`, `Matrix.nonsing_inv_mul` | medium |
| (E1) Gronwall | scalar Gronwall on ‖x‖_M | `norm_le_gronwallBound_of_norm_deriv_right_le` (Analysis/ODE/Gronwall) | medium |
| (E2), (E3) | expand ‖x + Δ‖², cross term zero by hypothesis | `Matrix.dotProduct_mulVec`, `Matrix.mulVec_transpose` | easy |
| NSD projection (E4) | sym/skew decomposition, Frobenius orthogonality, spectral clip | `Matrix.IsHermitian.spectral_theorem`, `skewAdjointMatricesSubmodule` | medium–hard |
| B0 (PDE identity) | do NOT formalise the torus integral; formalise the finite triad-sum identity (antisymmetry of the triad coefficient) over `Finset` | `Finset.sum_comm` | medium; not needed for L31 itself |

---

## L32 — the CTC loop: contracting AND informative

**Setting.** Per query row, E eras with predictions Y₁..Y_E ∈ R^p. D_ik = |Y_i − Y_k|², Δ² = max_ik D_ik. The live blend (`estimator.py:631-649`) is G(w)_i = exp(−c d_i(w))/Z, d_i(w) = |Y_i − Σ_k w_kY_k|², c = 1/(s2·T), s2 = Y.var(), T = 1. osc(v) = max v − min v; hdist(w,w′) = osc(log w − log w′) (Hilbert projective distance).

**Statement.**
- **Lemma 0 (the mask is not a mask).** d_i(w) = (Dw)_i − V(w), V(w) = Σ_k w_k|Y_k − Ȳ_w|² independent of i. Hence G(w) = softmax(−cDw), and the live map is exactly the Lean `Gmap A beta T` with A := −D, β/T := c, no mask.
- **Lemma 1 (contraction constant).** For A = −D: A_ik − A_jk − A_il + A_jl ≤ 2Δ², so the Lean hypothesis `hD` holds with D_Lean = 2Δ² and K = c·2Δ²/4 = cΔ²/2 =: q. For Euclidean D this is attained (k = i, l = j gives 2|Y_i − Y_j|²), so q is the least constant of this form. In the live parametrisation (A = YYᵀ, β = 2/s2) the same q is κ_H/T = Δ²/(2·s2·T).
- **Theorem 32.** Let q = cΔ²/2 < 1, u the uniform vector, ω := osc_i(Du)_i/Δ² ∈ [0,1].
  - (a) [Lean `hilbert_ctc_fixed_point`] a unique fixed point w* in the open simplex; hdist(Gⁿw₀, w*) ≤ qⁿ hdist(w₀, w*); hdist(w₀, w*) ≤ hdist(w₀, Gw₀)/(1 − q).
  - (b) Range cap, every w in the simplex: osc(log G(w)) ≤ cΔ² = 2q. Hence w*_max/w*_min ≤ e^{2q} < e², and with r = e^{2q}: w*_max ≤ r/(r + E − 1), w*_min ≥ 1/(1 + (E−1)r).
  - (c) Two-sided: 2qω/(1+q) ≤ osc(log w*) ≤ min{2q, 2qω/(1−q)}. In particular ω = 0 ⇒ w* = u; for E = 2, (Du)₁ = (Du)₂ = Δ²/2, so ω = 0 and **w* = u for every c.**
  - (d) Separation: g_ij = (Du)_i − (Du)_j, ρ₊ = 2qω/(1−q): |ln(w*_j/w*_i) − c g_ij| ≤ 2q(e^{ρ₊} − 1). Normalised: ln(w*_j/w*_i) = 2q[γ_ij ± (e^{ρ₊} − 1)], γ_ij = g_ij/Δ². Sign-correct separation is certified when γ_ij > e^{ρ₊} − 1; for small ω that is q < 1/3.
  - (e) Scale-invariant normalisation (exact). Replace c = 1/(s2·T) by **c = 2q/Δ²**, Δ² = max_{i,j}|Y_i − Y_j|² (if Δ² = 0 return u): G_q(w)_i = exp(−(2q/Δ²)|Y_i − Σ_k w_kY_k|²)/Z. Then κ_H/T = q exactly, for every E, p, scale Y → λY and translation Y → Y + t. Live form: set s2·T := Δ²/(2q), with the blend temperature decoupled from the TT `temp`.

**Hypotheses and hidden hypotheses.**
- H32-1 The contraction is per call: κ_H varies 66 to 112 across rows (n7), so no single fixed T certifies; only a per-call normalisation can.
- H32-2 The live path never calls a Hilbert certificate: it uses `ctc.damped_loop` with an ℓ¹ local norm (L₀ up to 111.7), not `hilbert.iterate`. After the normalisation call `hilbert.iterate(G_q, u, q, tol)`, not `damped_loop`.
- H32-3 The blend temperature is coupled to the TT `temp` (n7: `estimator.py:137, 495, 559, 564`). Decouple it, or the fix moves the TT.
- H32-4 An informative contracting loop needs ω > 0, i.e. E ≥ 3 with unequal mean disagreements. All 598 n7 calls had E = 2, so at those calls the loop is provably uninformative AND the REFUSE fallback to uniform returned the same vector as the converged loop would have. **The 598/598 refusals cost nothing in accuracy.**
- H32-5 Weights affect only the blend value; conformal coverage (P4) does not depend on them.

**Proof.**
Lemma 0: Σ_k w_k|Y_i − Y_k|² = |Y_i − Ȳ|² + Σ_k w_k|Y_k − Ȳ|² (expand about Ȳ; the cross term 2⟨Y_i − Ȳ, Σ_k w_k(Ȳ − Y_k)⟩ = 0); softmax is invariant to the i-independent V(w).
Lemma 1: −D_ik + D_jk + D_il − D_jl ≤ 2Δ² because D_jk, D_il ≤ Δ² and −D_ik, −D_jl ≤ 0.
(a) Lemma 0 + Lemma 1 + the Lean theorem; nothing new.
(b) ln G(w)_i − ln G(w)_j = −cΣ_k w_k(D_ik − D_jk), each |D_ik − D_jk| ≤ Δ² (both in [0, Δ²]), weights sum to 1. Max/min: with w_max ≤ r·w_min, w_max + (E−1)w_max/r ≤ 1 gives the max bound; w_min(1 + (E−1)r) ≥ 1 the min bound.
(c) u is constant in log, so hdist(u, Gu) = osc(log Gu) = c·osc(Du) = 2qω, and hdist(u, w*) = osc(log w*). Upper: the a-priori bound of (a) with w₀ = u. Lower: hdist(u, Gu) ≤ hdist(u, w*) + hdist(w*, Gu) and hdist(w*, Gu) = hdist(Gw*, Gu) ≤ q·hdist(w*, u), so hdist(u, w*) ≥ hdist(u, Gu)/(1+q). The cap 2q is (b).
(d) The fixed-point equation gives ln(w*_j/w*_i) = c[(Dw*)_i − (Dw*)_j]. Write w* = u + (w* − u). With x_k = D_ik − D_jk ∈ [−Δ², Δ²] and Σ(w* − u) = 0: |Σ_k(w* − u)_k x_k| ≤ ‖w* − u‖₁·osc(x)/2 ≤ ‖w* − u‖₁Δ². From w*_k/w*_l ∈ [e^{−ρ}, e^{ρ}]: w*_k ≤ e^ρ/E and ≥ e^{−ρ}/E, so |w*_k − 1/E| ≤ (e^ρ − 1)/E and ‖w* − u‖₁ ≤ e^ρ − 1, with ρ ≤ ρ₊ by (c). Multiply by c, use cΔ² = 2q.
(e) κ_H/T = cΔ²/2 = q; invariance because only Δ² and distances enter.

**Status.** PROVED (a)–(e). CONJECTURE C32-1: sign(w*_j − w*_i) = sign(g_ij) for all q < 1 (certified only for q < 1/3 above). C32-2: behaviour for q ≥ 1 (no uniqueness or convergence claim made). Consistent with ERRATA 15 (rule T > κ_H); (e) replaces the fixed-T rule by a per-call normalisation with the same contraction conclusion.

**Evidence** (`H`, `chase_sanity.py`, peak 289.6 MB, numpy seed 20261011, random Y with a common mode, p = 90).
- Lemma 1: brute-force D/Δ² = 1.000000000000 at E = 2, 3, 5, 8; κ_H = Δ²/(2·s2) = 85.5, 84.8, 85.1, 69.6, matching n7's 66 to 112. The size is a dimension effect (κ_H ~ p; n7: 0.94p).
- Empirical Hilbert Lipschitz ratio over 3,000 Dirichlet pairs never exceeds q; at E = 2 it equals q (0.5000, 0.8000, 0.9499), so the bound is tight there.
- E = 3, ω = 0.117: osc(log w*) = 0.179 (q = .5, bracket [0.078, 0.235]), 0.399 (q = .8, [0.104, 0.940]), 0.568 (q = .95, [0.115, 1.900]); max/min of w* = 1.20, 1.49, 1.77 versus cap e^{2q} = 2.72, 4.95, 6.69. All sandwiches, pairwise bound (d) and scale-plus-translation invariance (7.3Y + noise gives the same w* to 1e-9) hold in all 12 runs. E = 2: w* = uniform to 1e-12.
- Iterations n ≥ ln(2q/((1−q)tol))/ln(1/q) at tol = 1e-9: q = .5 → 31, .8 → 102, .9 → 224, .95 → 476; tilt cap e^{2q} = 2.7, 5.0, 6.0, 6.7. Pin recommendation in the source: **q = 0.8** (102 iterations, cap 4.95×), `max_iter = 500`. If sharper weights than e^{2q} are wanted the loop is the wrong tool: use the one-step weights G_c(u) for any c (tilt exp(c·osc(Du)), unbounded, no fixed-point claim).

**Lean plan.** (a) done (`L22H2_HilbertFixedPoint.lean`). Lemma 0 + Lemma 1: ≈ 40 lines (`Finset.sum` identity, `linarith` for `hD`). (b): ≈ 30 lines (reuse `Eemb_log_softmax`, `norm_Eemb_le`). (c): triangle inequality for hdist (`Eemb_sub`, `norm_add_le`) plus the n = 1 instance of the proved rate; ≈ 60 lines. (d): ≈ 80 lines. (e): arithmetic only.

---

## L33 — energy projection

**Setting.** x_t with r = ‖x_t‖; raw one-step prediction x̂ ≠ 0; router era k chosen on the arm's own state; ĝ > 0 fitted on Train only; projection Π(x̂) = ĝ·r·x̂/‖x̂‖. True next state x, g = ‖x‖/r, raw error e = ‖x̂ − x‖. (In the encoded space ‖x‖ = a, the amplitude; max difference 9.1e-13, n1.)

**Statement.**
- (i) Identity. With ŷ = x̂/‖x̂‖, n = ‖x̂‖, s = ⟨ŷ, x⟩, ρ = ĝr: ‖Π(x̂) − x‖² = e² + (ρ − s)² − (n − s)². The projection does not hurt iff |ĝr − s| ≤ |n − s|.
- (ii) One-step bound. If ε = e/‖x‖ < 1: ‖Π(x̂) − x‖ ≤ e + |ĝ − g|r + e²/‖x‖, i.e. rel ≤ ε(1+ε) + |ĝ/g − 1|. The projection adds at most the radial error plus a second-order direction term.
- (iii) K-step energy, closed loop. ‖x̃_{t+K}‖ = ‖x_t‖ Π_{j=1..K} ĝ_{k_j} exactly (k_j the router's era on the arm's own state at step j, no label). With h_K = ‖x_{t+K}‖/‖x_t‖: |‖x̃_{t+K}‖ − ‖x_{t+K}‖| = ‖x_t‖·|Π ĝ − h_K| is a LOWER bound on ‖x̃_{t+K} − x_{t+K}‖, and the relative error is ≤ 1 + Πĝ/h_K (no blow-up; the pilot's unprojected E50 reached 696). For a constant era: |ĝ^K − h_K| ≤ |ĝ^K − g_k^K| + |g_k^K − h_K|, with |ĝ^K − g_k^K| ≤ K·max(ĝ, g_k)^{K−1}·|ĝ − g_k|; at K = 50 and values ≤ 1.005 the factor is ≤ 63.9.

**Hypotheses (exact).** (ii) needs ε < 1. (iii) uses the closed loop on the arm's own states; the constant-era form also needs a fixed era. **Hidden hypothesis H33-1:** the coordinator's 50-step form |ĝ⁵⁰ − g_k⁵⁰| measures drift against the era's typical power, not against the truth; it equals drift against truth only if the realised ratio is a function of the era alone (constant growth). The data refute that (table below).

**Proof.**
(i) Decompose x = s·ŷ + x_⊥, x̂ = n·ŷ, Π = ρ·ŷ. Then e² = (n − s)² + ‖x_⊥‖² and ‖Π − x‖² = (ρ − s)² + ‖x_⊥‖². Subtract.
(ii) e < ‖x‖ ⇒ ⟨x̂, x⟩ > ‖x̂‖²/2 > 0, so cos θ = s/‖x‖ > 0. Then 0 ≤ ‖x‖ − s = ‖x‖(1 − cos θ) = ‖x‖sin²θ/(1 + cos θ) ≤ ‖x_⊥‖²/‖x‖ ≤ e²/‖x‖, since ‖x_⊥‖ is the distance from x to the line ℝŷ and x̂ lies on that line. So ρ − s = (ĝ − g)r + (‖x‖ − s) has |ρ − s| ≤ |ĝ − g|r + e²/‖x‖. From (i), ‖Π − x‖² ≤ e² + (ρ − s)², and √(a² + b²) ≤ a + b.
(iii) The projection fixes the norm at each step to ĝ_{k_j} times the previous predicted norm, whatever the direction. Reverse triangle inequality for the lower bound, triangle inequality for the upper bound. Power bound: a^K − b^K = (a − b)Σ_i a^i b^{K−1−i}.
Auxiliary lemma (used in the check): ‖a/‖a‖ − b/‖b‖‖ ≤ 2‖a − b‖/(‖a‖ + ‖b‖). Proof: with s_ = ‖a‖ + ‖b‖, P = ‖a‖‖b‖, the gap between 4‖a − b‖² and (2 − 2cos θ)s_² is 2(‖a‖ − ‖b‖)²(1 + cos θ) ≥ 0.

**Status.** (i)–(iii) PROVED. The 50-step floor (below) is data-dependent (estimates, not measurements). CONJECTURE C33-1: the mean-reverting projection (replacement route) lowers the floor relative to constant ĝ; needs a seed-3 test before use.

**Evidence** (`H`, `chase_sanity.py`; n1 data for the table; numbers on seed 3).
- Identity: max relative error 4.2e-14 over 20,000 random cases in R^90, ε ∈ [0.01, 0.9]; one-step bound and unit-vector lemma hold in all cases; MVT power bound holds in 5,000 of 5,000 random (g, ĝ).

| era | g_k | g_k⁵⁰ | measured E[h₅₀], no-switch | gap | sd(h₅₀) |
|---|---|---|---|---|---|
| 0 | 0.999695 | 0.9849 | 0.9991 | +0.0143 | 0.061 |
| 1 | 0.998371 | 0.9217 | 0.9497 | +0.0280 | 0.053 |
| 2 | 1.002318 | 1.1227 | 1.0929 | −0.0298 | 0.260 |
| 3 | 1.004635 | 1.2601 | 0.9955 | −0.2646 | 0.110 |

- Energy is mean-reverting on the attractor; eras 2 and 3 have g_k > 1, so ĝ^K grows without bound as K grows while true energy stays bounded. Era 3 rests on 349 transitions from 3 trajectories.
- Floor: by (iii) any energy-pinned rollout has E50 ≥ mean |Πĝ/h₅₀ − 1|, of order sd(h₅₀) = 5 to 26 percent (estimate, not measured). Irrelevant while E50 ≈ 0.66 (persistence, seed 3); binding if E50 drops below ≈ 0.1.
- Estimation part is small: SE of ĝ_k ≈ σ/√n = 0.0020/√1452 = 5.3e-5 (if iid) × 63.9 = 0.34 percent (era 0); era 3: 0.0142/√349 = 7.6e-4 × 63.9 = 4.9 percent.
- One step: at ε ≈ 0.03 (persistence one-step 0.030169, seed 3) the second-order term is 9e-4; the radial term |ĝ/g − 1| is about the per-step std: 0.0020, 0.0025, 0.0073, 0.0142 for eras 0 to 3, so the projection costs at most 7 percent (era 0) to 47 percent (era 3) of the base one-step error in the worst case, and wins whenever |ρ − s| ≤ |n − s|.

**Replacement route.** Mean-reverting projection: log r_{t+1} = (1 − a_k) log r_t + a_k log r_k*, (a_k, r_k*) fitted per era on Train (an AR(1) in log-energy); Πĝ then stays bounded for all K and the floor becomes the AR(1) residual spread (C33-1).

**Lean plan.** (i) inner-product algebra (`norm_sub_sq_real`, `inner_smul_left`, `real_inner_self_eq_norm_sq`), ≈ 40 lines. (ii) ≈ 40 lines (work with squares to avoid `sqrt`). (iii) `Finset.prod`, `norm_smul`, geometric-sum factorisation (`Commute.geom_sum₂_mul`), ≈ 50 lines.

---

## L35 — the e-detector: validity for all κ, hidden hypothesis, delay, best κ

**Setting.** e_t = κp_t^{κ−1} (infinite at p = 0), S₀ = 0, S_t = (S_{t−1} + w_t)e_t, w_t = 1/(t(t+1)), M_t = S_t + 1/(t+1), M₀ = 1 (`edetector.py`). Expansion S_t = Σ_{i≤t} w_i Π_{s=i..t} e_s (all terms ≥ 0).

**Statement.**
- **Theorem 35a (validity, every κ ∈ (0,1)).** Hypothesis H-cond: P(p_t ≤ u | F_{t−1}) ≤ u for all u, a.s. Then E[e_t | F_{t−1}] ≤ 1 (equality for uniform p), M is a nonnegative supermartingale with M₀ = 1, and P(∃t: M_t ≥ 1/δ) ≤ δ.
- **Theorem 35b (detection delay).** Change after step ν; L = ln(1/δ_j) + ln((ν+1)(ν+2)); X_i = ln e_{ν+i}. (H1) ln κ ≤ X_i ≤ B (automatic for split conformal with p ≥ 1/(m+1): B = ln κ + (1−κ)ln(m+1); m = 500, κ = .5 gives B = 2.41). (H2) E[X_i | F_{ν+i−1}] ≥ λ > 0. Then E[(τ − ν)⁺] ≤ (L + B)/λ, τ the first alarm time. It is **2 ln ν, not ln ν**: −ln w_{ν+1} = ln((ν+1)(ν+2)).
- **Proposition 35c (best κ).** m₁ = E_Q[−ln p] (null value 1). λ(κ) = ln κ + (1−κ)m₁ is strictly concave on (0,1). (i) If m₁ ≤ 1, λ(κ) < 0 for every κ ∈ (0,1): no fixed κ detects at first order. (ii) If m₁ > 1, κ* = 1/m₁ and λ* = m₁ − 1 − ln m₁ = KL(Exp(mean m₁) ‖ Exp(1)). Grid: the average of test martingales is a test martingale and S ≥ S^κ/|G|, so a grid G gives (35b) with L + ln|G| and λ = max_{κ∈G} λ_κ.
- **Hidden hypothesis H35-1 (KILL for the FCPA instantiation): conditional super-uniformity.** 35a needs the conditional statement, not marginal uniformity. Two failures: (1) Fixed calibration set: given the calibration scores the test p-value has a random CDF. (2) Temporal dependence: windows of an NS-2D trajectory are strongly autocorrelated, so the "no false birth, ever" theorem (`L24b`, Amendment B L24*) is about a hypothesis the instantiation does not meet.

**Proof.**
35a. Write f(u) = κu^{κ−1}, decreasing, f ≥ κ on (0,1]. Layer cake: E[f(p) | F] = ∫₀^∞ P(f(p) > s | F) ds. For s < κ the probability is 1. For s ≥ κ, f(p) > s iff p < a(s) = (s/κ)^{−1/(1−κ)} (or p = 0), and P(p < a | F) ≤ a. So E[e | F] ≤ κ + ∫_κ^∞ (s/κ)^{−b} ds with b = 1/(1−κ) > 1, = κ + κ/(b−1) = κ + (1−κ) = 1. Supermartingale: S_{t−1} + w_t = M_{t−1} − 1/(t+1) is F_{t−1}-measurable and ≥ 0, so E[M_t | F_{t−1}] ≤ (S_{t−1} + w_t) + 1/(t+1) = M_{t−1}. Ville (`L24s_Ville.ville`) gives the tail bound. Infinite variance is irrelevant: Var(e) = κ²/(2κ−1) − 1 for uniform p and κ > ½, infinite for κ ≤ ½; Ville needs only E M_t ≤ 1. Discrete p (split conformal, p = (1+rank)/(m+1)) is super-uniform, and E[e] = κn^{−κ}Σ_j j^{κ−1} ≤ 1 − (1−κ)n^{−κ} (n = m+1).
35b. M_t ≥ S_t ≥ w_{ν+1}Π_{s=ν+1..t}e_s, so M_t ≥ 1/δ_j once R_n = Σ_{i≤n}X_i ≥ L (n = t − ν). Let N be the first such n; then (τ − ν)⁺ ≤ N. By (H2), R_n − λn is a submartingale; for the bounded stopping time N∧n, optional stopping gives λE[N∧n] ≤ E[R_{N∧n}] ≤ L + B (before N, R < L; at N, R_N ≤ L + B by H1). Let n grow; monotone convergence. No finiteness of E N is assumed. For unbounded p (analytic p-values in the toy) replace B by Lorden's E[(X⁺)²]/λ (cited, not proved).
35c. λ′ = 1/κ − m₁, λ″ = −1/κ² < 0. At κ = 1, λ = 0 and λ′ = 1 − m₁; concavity gives λ(κ) ≤ (1−m₁)(κ−1), negative for κ < 1 when m₁ ≤ 1. The stationary point is 1/m₁ and λ(1/m₁) = −ln m₁ + m₁ − 1.
Tightness: for any e-value with E_P e ≤ 1, E_Q ln e ≤ KL(Q‖P) (Gibbs), so λ_κ ≤ KL; the first-order delay ln(1/δ)/λ_κ meets Lai's lower bound only when e_κ is the exact likelihood ratio.

**Status.** 35a, 35b (bounded increments), 35c PROVED; H35-1 is a KILL of the FCPA instantiation's guarantee. CONJECTURES: C35-1 thinning by G restores conditional validity up to ε(G) geometric in G, so a correction e/(1+ε) restores validity; C35-2 Lai optimality of the first-order delay (cited, not proved; the unknown change time costs the additive 2 ln ν); C35-3 PAC rescale e → e/(1 + 1.645·sd) for a fixed calibration set (or fresh calibration per birth test); C35-4 for λ ≤ 0 but E_Q e_κ > 1, delay is governed by the Kesten exponent θ solving E_Q[e_κ^θ] = 1 (polynomial, not logarithmic, in 1/δ), consistent with the ledger's detection of μ = .5 by the Vovk κ-mixture after ≈ 900 steps. Replacement routes: R1 feed the detector every G-th window, G ≥ the decorrelation lag of the novelty score under the null (estimated on the Train stream); R2 standardise the score by its AR(1) conditional law (exact only if the model is true). **Test that must be RED then GREEN:** null alarm frequency on within-era seed-3 windows ≤ δ (also on the AR(1) surrogate at the pinned G). Note: `L24c_Calibrator.lean` (see Corrections) now formalises 35a(1) under the conditional super-uniformity hypothesis, which is the same hypothesis H35-1 shows is violated.

**Evidence** (`H`, `chase_sanity.py`, `chase_sanity2.py`; null N(0,1), score |x|, two-sided p, δ = .05, ν = 200).
- Fixed calibration set, m = 500, κ = .5, 20,000 Dirichlet-spacing draws: E[e | cal] has mean 0.9680, sd 0.0387, max 1.194, exceeds 1 for 19.3 percent of calibration sets; closed form sd = √(Var_unif(φ)/(m+2)), φ_j = κ(j/n)^{κ−1} gives 0.0389. Discrete-p E[e] at m = 500: 0.9877/0.9679/0.9989 for κ = .1/.5/.9 against bounds 0.5167/0.9777/0.9996, all satisfied.
- AR(1) stream, marginal p exactly uniform, null, κ = .5, δ = .05, 4,000 streams × 400 steps: alarm frequency 0.025 (ρ = 0), 0.120 (ρ = .5), 0.742 (ρ = .9), 0.694 (ρ = .99). Thinning ρ = .9 by 20: 0.025 (empirical, not a theorem). Ville null check, κ = .5, uniform p, 20,000 × 400: 0.0243 against δ = .05 (within a factor 2 of tight).
- Infinite-m closed form:

| shift μ | m₁ | λ(.5) | λ(.8) | κ* | λ* |
|---|---|---|---|---|---|
| 0.5 | 1.1733 | −0.1065 | +0.0115 | 0.852 | 0.0135 |
| 1.0 | 1.6781 | +0.1459 | +0.1125 | 0.596 | 0.1604 |
| 2.0 | 3.5415 | +1.0776 | +0.4852 | 0.282 | 1.2770 |

- Sign check against the ledger: ledger I15 reports E₁ log L = −0.068 for μ = .5, κ = .5; closed form −0.1065; finite m = 500 conformal p-value −0.0827 (MC, 400,000 draws); the remaining 0.015 to the ledger is unreconciled (setup of e3 not visible). Null drift ln .5 + .5 = −0.1931 matches ledger I1.
- Delay (300 streams each, analytic p, bound = (L + Lorden)/λ):

| μ | κ | λ | detected after ν | mean delay | bound |
|---|---|---|---|---|---|
| 2.0 | 0.3 | +1.275 | 0.987 in 1,500 | 11 | 13 |
| 1.0 | 0.5 | +0.146 | 0.983 in 3,000 | 72 | 118 |
| 0.5 | 0.8 | +0.0115 | 0.970 in 6,000 | 626 | 1,505 |
| 0.5 | 0.5 | −0.1065 | 0.053 in 6,000 | (missed) | infinite |

- Pin recommendation in the source: κ-grid **{0.2, 0.5, 0.8}** with δ_j unchanged (κ* = 1/m₁ ↔ m₁ = 5, 2, 1.25; covers every shift with m₁ ≥ 1.25; costs ln 3 = 1.10 nats in L, ≈ 6 to 8 percent delay; removes the κ = .5 blind spot, μ = .5 detected at 0.97 with κ = .8). If a single value is mandatory use **κ = 0.8** (μ = 2 gives λ(.8) = .485). Not to ship until H35-1 is repaired: at ρ = .9 the δ guarantee is void.

**Lean plan.** 35a(1): layer cake (`MeasureTheory.lintegral_comp_eq_lintegral_meas_lt_mul`, `integral_Ioi_rpow_of_lt`), ≈ 80 lines (at the time of the memo `L24c_Calibrator` was missing; see Corrections). 35a(2,3): `L24b` and `L24s_Ville` in repo. 35b: bounded-stopping-time optional stopping, same machinery as `ville_finite` (`Submartingale.expected_stoppedValue_mono`), ≈ 100 lines with bounded increments (the Lorden form is not cheaply provable at that Mathlib). 35c: `StrictConcaveOn`, derivative of log, ≈ 40 lines.

---

## L36 — which exchangeability L6/L6′ needs, and why interleaved hold-out breaks it

**Code facts** (`H`, `fcpa/estimator.py` read verbatim). `_CAL_EVERY = 5` (line 34): window j is held out iff `j % 5 == 4` (lines 343, 399); held-out windows are excluded from era statistics (`else` branch at 348 skipped) while windows j−1 and j+1 are training windows, contiguous in time. `window_idx` is global across `partial_fit` calls. `_cal_window` (471–480) appends the 20 rows of each held-out window to `cal_phi` and keeps the newest `_CAL_MAX` = 400 rows (20 windows). `_gamma`/`_shrink` (484–509) and the conformal fit (542–552) both read these rows. `SplitConformal` docstring (`conformal.py:32`): "calibrate on data disjoint from training (whole trajectories, see `trajectory_split`)". `run_primary` builds a 64-trajectory Cal split but passes only Train to the estimator and uses Cal only for the era map.

**Setting.** Windows W_{τ,j} (trajectory τ, index j; J ≈ 19 windows at L = 400, w = 20). Training data T, held-out units H₁..H_n, test window W from a fresh trajectory. Score s = s_T is T-measurable. β(m) = TV(P_{(past,future)}, P_past ⊗ P_future), past and future separated by m windows (gap g = m − 1).

**Statement.**
- **36a (the exact requirement).** Condition (E): given T (and, for Mondrian, the era labels within each era class), (s₁..s_n, s_{n+1}) = (s_T(H₁), .., s_T(H_n), s_T(W)) is exchangeable. Sufficient (Q): H₁..H_n, W i.i.d. and independent of T. With k = ⌈(n+1)(1−α)⌉ and q_k the k-th smallest calibration score: P(s_{n+1} ≤ q_k) ≥ k/(n+1) ≥ 1−α, and ≤ 1−α+1/(n+1) if scores are a.s. distinct. The same condition makes a hold-out loss an unbiased estimate of the fresh-data loss, which the skill gate (`_gamma`, `_MIN_GAIN = 0.2`) needs.
- **36b (interleaving violates (E)).** Protocol P_g: every cal window is separated from every training window of the same trajectory by ≥ g windows (current code g = 0). (i) Let Q keep every block's marginal and make blocks independent. For any event A of (T, H, W): |P(A) − Q(A)| ≤ TV(P,Q) ≤ Σ_τ (k_τ − 1)β(g+1), k_τ the number of time-ordered blocks of trajectory τ with consecutive gaps ≥ g; for the current design k_τ − 1 ≈ 2n_τ, bound ≈ 2nβ(g+1). (ii) The bias is attained, linearly in β: copy process (held-out window C equals the training window T at lag g+1 with probability b = β(g+1), else an independent copy; β(m) = a^m exactly), score s = |C − T|. Coverage = 1 − α/(1−b), deficit αb/(1−b) ≈ αβ(g+1). So the bias lies between αβ/(1−β) (attained) and (k−1)β (bound). (iii) Gaussian linear smoother: s = |C − m̂|, Cov(C, m̂) = c > 0; calibration scale σ_c = √(1 + Var m̂ − 2c) against fresh σ_t = √(1 + Var m̂); coverage at the exact calibration quantile is 2Φ(zσ_c/σ_t) − 1, deficit ≈ 2zφ(z)c/(1 + Var m̂) (0.339c/(1 + Var m̂) at 90 percent). For Gaussian processes |corr| ≤ 2π α_mix ≤ 2πβ (Kolmogorov–Rozanov, cited).
- **36c (trajectory-level hold-out restores (E)).** If held-out units come from trajectories disjoint from every trajectory in T, and trajectories are i.i.d. and stationary, then T, the held-out trajectories and the test trajectory are independent, so (Q) holds for any T-measurable score; no mixing assumption. (i) Exact: one window per held-out trajectory at a uniformly random position, test window at a uniformly random position of a fresh trajectory: the n+1 scores are i.i.d. given T, 36a gives P(cover) ≥ 1−α. (ii) All windows of each of n_c held-out trajectories (clusters of n_w dependent windows): calibrate at level 1−α+t; q* the true (1−α)-quantile of the fresh score law given T. Then P(q̂ ≥ q*) ≥ 1 − exp(−2n_c t²), and on that event F(q̂) ≥ F(q*) ≥ 1−α; with failure probability η, t = √(ln(1/η)/(2n_c)). For the 64-trajectory Cal split and η = 0.05, t = 0.153: too coarse at α = 0.1.
- **36d (coverage error with no trajectory hold-out, gap g).** Under P_g with one score per held-out window: 1−α−B(g) ≤ P(cover) ≤ 1−α+1/(n+1)+B(g), B(g) = Σ_τ (k_τ − 1)β(g+1). For geometric mixing β(m) ≤ Cρ^m, reaching B ≤ ε with n held-out windows and 2n boundaries needs g+1 ≥ ln(2nC/ε)/ln(1/ρ).

**Hypotheses (exact).** 36a: (E) or (Q) as stated; distinct scores for the upper bound. 36b: stationary windows, β as defined. 36c(i): fresh trajectories i.i.d. stationary, random window position. 36c(ii): fixed q* so Hoeffding applies to the mean of n_c independent cluster averages; cluster size does not enter n_c (rows of a window, 20, and windows of a trajectory, 19, are not independent samples). 36d: P_g. Hidden: **H36-1** stationarity: held-out windows sit at j = 4, 9, … while test windows can sit anywhere; (Q) needs the window law not to depend on position (burn-in discarded; the initial law of the first regime from `ns2d.regime_schedule` was not checked). **H36-2** rows are not units: each held-out window contributes 20 adjacent-step rows, and `n_min_ = ⌈1/α⌉ − 1 = 9` rows (less than half a window) declares an era calibrated (`SplitConformal`, line 42). **H36-3** the pool is a FIFO of the newest 400 rows (`estimator.py:479`), i.e. 20 windows; in the real fit (128 trajectories, ≈ 4 held-out windows each) it is the last ≈ 5 trajectories (4 percent), split again by era (Mondrian): a handful of windows per era. **H36-4** the gate comment ("small gains are partly leakage; a large one survives it", `estimator.py:42`) is refuted by the toy. **H36-5** what is NOT broken: P4. ACI (L7) guarantees |avg err − α| ≤ (max(α₁, 1−α₁) + γ)/(γT) for any sequence whatever the base calibration; what breaks is the base set's coverage, set sizes, and the skill gate that decides `gam`, hence E50 (P1, P2). P3 is scored on the router argmin and is not affected by calibration.

**Proof.**
36a. Under (E) and distinct scores the rank of s_{n+1} among the n+1 values is uniform on {1..n+1}, because every permutation of the scores has the same law. s_{n+1} ≤ q_k iff its rank ≤ k. Ties only move the rank down.
36b(i). Order the blocks of one trajectory in time, b₁ < … < b_k. Telescope: TV(P_{b₁..b_k}, ⊗ᵢP_{bᵢ}) ≤ Σ_{i<k} TV(P_{b₁..b_{i+1}}, P_{b₁..b_i} ⊗ P_{b_{i+1}}) ≤ (k−1)β(g+1), each term being a β-mixing coefficient between σ(b₁..bᵢ) and σ(b_{i+1}). Trajectories are independent, so contributions add. A [0,1]-valued function changes by at most TV between two laws.
36b(ii). For continuous marginals the joint of the renewal chain is a^m times the diagonal plus (1 − a^m) times the product, so β(m) = a^m. Calibration scores have an atom of mass b at 0, fresh scores have none; at the calibration quantile of level 1−α, F(q) = (1−α−b)/(1−b), so coverage = 1 − α/(1−b).
36c. Trajectories are generated by independent `simulate` calls with `n_universes = 1` (`run_primary._gen`) and `trajectory_split` assigns whole trajectories, which gives the independence claim. (ii): {q̂ < q*} ⊆ {F̂(q*−) ≥ 1−α+t}; F̂(q*−) is the mean over the n_c independent clusters of the cluster average of 1{s < q*}; F(q*−) ≤ 1−α; Hoeffding.
36d. Under Q the scores are exchangeable (36a), so Q(cover) ∈ [1−α, 1−α+1/(n+1)]; |P − Q| ≤ B(g) by 36b(i).

**Status.** 36a–36d PROVED (Kolmogorov–Rozanov CITED; Hoeffding used in 36c(ii)). CONJECTURES: C36-1 marginal coverage of the all-windows trajectory-level pool is within O(1/n_c) of 1−α; C36-2 β of the observed windows approaches the regime-chain 0.75θˢ when the regime is identifiable from a window; C36-3 the size of the gate optimism on the real estimator (the toy gives mechanism, not magnitude).

**Replacement routes.** **R1 (recommended, prereg-conformant per the memo; compatibility to be checked, see Design decisions):** use the 64-trajectory Cal split as calibration-only chunks. Add a role to `partial_fit` (training or calibration chunk) so all windows of a calibration chunk are held out and none of a training chunk is; `fit_fcpa` passes `XL[tr]` as training and `XL[ca]` as calibration. This removes interleaving, the FIFO truncation and the 4-percent pool at once; use a random window per trajectory (36c(i)) or all windows with a cluster-aware `n_min` (count trajectories, not rows). The gate (`_gamma`) then reads the same trajectory-level rows. **R2** (cheaper, weaker): hold out every 5th trajectory inside the Train split (keeps the API, costs 20 percent of training trajectories). **R3** (embargo): not viable at L = 400.

**Evidence** (`H`, `chase_sanity3.py`, one process, peak 55.9 MB; 60 repetitions per cell, kernel ridge on AR(1) window paths in d = 20, 320 train and 80 held-out windows, y = sin(2u·x) + 0.3 noise, target coverage 0.90).

| window correlation ρ | ridge | coverage, interleaved | coverage, trajectory hold-out | coverage, ideal | gate gain hold-out / fresh: interleaved | trajectory |
|---|---|---|---|---|---|---|
| 0.0 (control, iid) | 0.01 | 0.900 | 0.898 | 0.900 | −0.289 / −0.304 | −0.324 / −0.298 |
| 0.0 | 0.1 | 0.907 | 0.902 | 0.909 | −0.100 / −0.058 | −0.054 / −0.056 |
| 0.9 | 0.01 | 0.730 | 0.894 | 0.901 | +0.377 / −0.481 | −0.607 / −0.509 |
| 0.9 | 0.1 | 0.760 | 0.897 | 0.898 | +0.423 / −0.097 | −0.147 / −0.118 |

With dependent windows the interleaved design under-covers by 0.14 to 0.17 and the gate (`_MIN_GAIN = 0.2`) passes a model that is worse than persistence on fresh trajectories; the trajectory-level design is at 0.894 to 0.897 and the iid control shows no effect. This reproduces the builder observation (gate accepts a model worse than persistence at 16 trajectories). The copy process matches the closed form: b = .5/.25/.125/.0625/.03125 give coverage 0.8004/0.8667/0.8855/0.8938/0.8968 against 1 − α/(1−b) = 0.8000/0.8667/0.8857/0.8933/0.8968.
- Regime chain P = θI + (1−θ)Π, θ = 1 − 4λ_sw/3 = 0.993333: β(s) = 0.75θˢ for s steps of separation (windows g apart: s = 20g + 1); the regime β is a lower bound for the (state, regime) pair β (matrix powers agree with the formula to four decimals):

| gap g (windows) | s (steps) | β (exact) |
|---|---|---|
| 0 (current) | 1 | 0.7450 |
| 1 | 21 | 0.6517 |
| 4 (distance between held-out windows) | 81 | 0.4363 |
| 10 | 201 | 0.1955 |
| 18 | 361 | 0.0670 |
| 19 (first to last window of a trajectory) | 400 | 0.0516 |

  B(g) for n = 20 held-out windows needs β ≤ 2.5e-4 for ε = 0.01, i.e. s ≥ 1,197 steps (60 windows). No embargo inside L = 400 repairs it; only a trajectory-level split does.
- RED tests (`zero/fcpa/tests/test_launch_guards.py`, both fail on the current code; synthetic AR(1) trajectories 30 × 260 × 4, well under 1 GB): `test_f_calibration_windows_are_not_interleaved_with_training_windows_of_their_trajectory` (30 of 30 trajectories supply both held-out and training windows); `test_f_calibration_pool_spans_the_training_trajectories_not_just_the_last_few` (the pool comes from 8 of 30; needs ≥ 15). Fix route for both: R1.

**Lean plan.** `CHANGES.md` lists L6 and L6′ as not in Lean v1, so 36a is the missing piece. 36a: no exchangeability API in that Mathlib; uniform rank by the action of `Equiv.Perm (Fin (n+1))` on a permutation-invariant measure, ≈ 120 lines. 36b(i): TV telescoping on `Measure.prod`/`Measure.map`, ≈ 150 lines; β is a hypothesis. 36b(ii): explicit mixture computation, ≈ 60 lines. 36c(ii): Hoeffding is not in Mathlib of that era (≈ 300 lines to prove); a Chebyshev substitute P(F̂ − F ≥ t) ≤ 1/(4n_ct²) needs t = 0.5/√(ηn_c) (0.31 for n_c = 64, η = 0.05), too coarse, so prefer 36c(i). 36d: composition of the above.

---

## Corrections to earlier claims

1. **Spectral-radius clipping does not control rollout.** Clipping the 2-lag companion to ρ ≤ 1 leaves E50 = 1.346 (r = 1.0) and 1.338 (r = 0.999) (`C`, stage A); L28(e) shows ‖A^j‖ is not a function of ρ, and the measured s_50 moves 2.8% under clipping at ρ = 0.999 (`F`, L28 table). Mechanism: velocity-memory (momentum) mode, not ρ. Source: `F` L28 correction 1 and (e1)–(e2); `C` L31(iii) E4.
2. **The bars do not provably conflict.** The claim "no single γ meets one-step ≤ 0.5 × persistence AND E50 < persistence" is REFUTED: builder witness damping 0.7 with energy gives 0.0114/0.0284 = 0.40 and 0.622 < 0.6335; the toy window [0.524, 0.711] matches Prop 3(iii) exactly. What is true: the one-step optimum (a = α₁) differs from the horizon optimum (a*(50) = 0.55), and the E50 margin is ≈ 0.4% of E² for any velocity-linear predictor, so any win must come from the nonlinear part. Source: `F` L28b.
3. **"Persistence beats every linear map at k = 50" is false in its original form.** Persistence is the linear map M = [I 0], so the best linear k-step map is never worse (L29(a)); what holds is that persistence beats the composed one-step maps A^k and that the room above persistence is Δ_k (≈ 0 on this bed). Source: `F` Corrections 2, L29.
4. **⟨x, B(x,x)⟩ = 0 needs Parseval weights.** In the raw rfft2 coordinates the Euclidean product gives a relative residual 4.6e-2; with M = diag(1 on ky = 0, 2 on ky > 0) it is 1.4e-16 (enstrophy) and 2.6e-17 (energy). Also, d‖x‖²/dt ≤ 2⟨x, Ax+c⟩ is an equality in continuous time and fails by δ² per step in discrete time without E3. Source: `C` F1, L31(iii).
5. **The CTC refusals were harmless because E = 2.** For E = 2 the fixed point is exactly uniform for every c (ω = 0), so the 598/598 REFUSE calls returned the vector the converged loop would have; the loop can only be informative for E ≥ 3. The source verdict on "contracting AND informative" is half right: contraction forces the tilt cap w*_max/w*_min ≤ e^{2q} < e², and for E ≥ 3 the loop is informative by a bounded tilt. Source: `H` L32 (c), H32-4.
6. **κ = 0.5 has negative drift for μ = 0.5.** λ(0.5) = −0.1065 (infinite m), −0.0827 (m = 500, MC); ledger I15 reports −0.068 (0.015 unreconciled); null drift −0.1931 matches ledger I1. At μ = .5 the κ = .5 detector is missed (0.053 in 6,000 steps), while κ = .8 detects at 0.97. Source: `H` L35c and evidence.
7. **The L24\* "no false birth ever" theorem does not cover autocorrelated windows.** It is conditional on P(p_t ≤ u | F_{t−1}) ≤ u. AR(1) windows with exactly uniform marginal p alarm at 0.742 (ρ = 0.9) against δ = 0.05, and a fixed calibration set makes E[e | cal] exceed 1 for 19.3 percent of calibration sets at m = 500. The guarantee holds for the instantiation only after thinning by G or a standardisation (R1, R2) is shown to restore conditional validity. This touches Amendment B (L24\*) and ERRATA 8/14. Source: `H` H35-1.
8. **The `L24c_Calibrator` header reference.** `H` (memo time) reports that the header of `L24b` cites `calibrator_integral_le` in `L24c_Calibrator` and that no such file existed in `zero/lean/FCPA/`, so step 35a(1) was not formalised. At the time of this compilation `zero/lean/FCPA/L24c_Calibrator.lean` is present in the worktree (untracked, written after the memo; no `sorry`; it proves `calibrator_integral_le` from conditional super-uniformity). Its compilation and its inclusion in the build were not verified here. The dangling-reference finding stands for any commit that lacks the file.

Further corrections recorded in the sources:
9. The gate comment "small gains are partly leakage; a large one survives it" (`estimator.py:42`) is refuted: gate gain +0.38 to +0.42 on interleaved hold-out against −0.48 to −0.10 on fresh trajectories (`H` H36-4).
10. The 50-step energy statement |ĝ⁵⁰ − g_k⁵⁰| is not a drift bound against truth (`H` H33-1).
11. The bar "one-step ≤ 0.5 × persistence" cannot be met by any linear 2-lag model on 12 training trajectories (0.80× at best); it belongs to the gated 128-trajectory test, and on the 16-trajectory sample only E50 is a fair RED/GREEN (`C`).
12. The quadratic constraint does not by itself bound the lag model: the A₂ term enters energy at first order in δ (`C` L31(iii) E4).

---

## Design decisions implied (defaults, to be confirmed before code)

Every item below changes a quantity that Amendment 11 pins (κ = 0.5; "only the learning rate of each gradient-trained part is tuned"; the Cal split feeds only the era map, frozen on Cal before Test is read; T = 25,600; the 128/64/64 split) or that the frozen spec fixes. **Any adoption therefore needs a new pre-result prereg amendment with its own FREEZE file (in the manner of 11b/FREEZE3), filed before any eval-seed result; this document does not make that change.** Tuning stays on seed 3, validation split only.

1. **Era model.** Energy-conserving quadratic (L31) on a frozen low-|k| coordinate set J (data-independent, hence exact by L30(a)), plus the dissipative linear part (CONJECTURE: C5 for lag terms, C6 for the drain, T2 not run), plus the energy-shell rescale E3 at inference. Constraints carried: the M^{1/2} pre-scaling for the crossing operator; r ≤ 20 for small eras with the linear fallback (L30(b)); matrix-free CG for r ≥ 40; the guarantee covers era-routed rows only (the TT is outside it). Prerequisite: the one-line check of the L-GIVEN definition in the PREREG (learned-M variant as fallback). Prerequisite tests: T1 (premise), then T2.
2. **Damping** in the L28b window [a₋, a_k^hi], chosen on seed-3 validation. The window is nonempty only if α₁ ≥ √3/2 and g*_k > 0; on the 12 train trajectories g*_50 = −0.10 (infeasible) while on the 4 test trajectories it is 1.23, so the choice must be made on a validation set disjoint from the 4 test trajectories used so far. Expected margin is small (≈ 0.4% of E² for velocity-linear predictors).
3. **CTC normalisation** c = 2q/Δ² with q = 0.8 (L32(e); 102 iterations to 1e-9, tilt cap 4.95×, `max_iter = 500`), `hilbert.iterate(G_q, u, q, tol)` instead of `damped_loop`, blend temperature decoupled from the TT `temp` (H32-2, H32-3). Informative only for E ≥ 3.
4. **κ pin 0.8 plus window thinning (G to be measured).** κ = 0.8 as the single value (L35); the source's primary recommendation is the grid {0.2, 0.5, 0.8} at +ln 3 nats; thinning by G ≥ the decorrelation lag of the novelty score under the null, estimated on the Train stream (R1 of L35; ε(G) and the rescale e/(1+ε) are CONJECTURE C35-1). Gate: do not ship until the RED-then-GREEN null-alarm test (≤ δ on within-era seed-3 windows and on the AR(1) surrogate at the pinned G) passes. Changes the Amendment-11 pin κ = 0.5 and the null-stream definition (200 streams × 2,000 steps).
5. **Trajectory-level calibration (L36 route R1):** Cal-split trajectories as calibration-only chunks, a `partial_fit` role argument, cluster-aware `n_min` (trajectories, not rows). **Prereg compatibility check needed against the freeze files:** the Cal split is documented as feeding only the era map, which is frozen on Cal before Test is read (PREREG_AMENDMENT_11_DRAFT, Era map); using the same Cal trajectories for conformal calibration and the gate must be shown not to leak Test and must be stated explicitly in the new amendment. R2 (hold out every 5th Train trajectory, −20% training) is the fallback that avoids touching the Cal role.
6. **τ re-picked on validation, not test.** τ is the law-consistency threshold `_TAU` in `fcpa/estimator.py` (currently 1.7): a window joins an era iff its mean normalised law residual is ≤ τ. It was chosen by P3 on the seed-3 Test split (P3 0.864, 0.886, 0.928, 0.853, 0.83, 0.755 at τ = 1.4, 1.55, 1.7, 1.85, 2.0, 2.3; Builder-model report). It must be re-picked on the seed-3 validation split.

---

## Open

**CONJECTURES (each with its falsifier where the source gives one).**
- L28: quantitative Schur bound ‖A^j‖ ≤ Σ_{m<n} C(j,m)ρ^{j−m}‖N_schur‖^m (not run against data).
- L28b: the sign of g*_50 (feasibility of Prop 3(iii)) is not statistically resolved on 12/4 trajectories.
- L29: mixing of the NS attractor so that D_lin → √(tr Σ) at large τ.
- L30: C1 data off every quadric (falsified by a rank deficiency of G_c).
- L31: C2 generic intersection dimension of the enstrophy and energy subspaces (rank computation); C3 closure of reduced libraries under S (one commuting-generator example); C4 learned M = Parseval weights (T3); C5 lag-term Lyapunov LMI (no closed form); C6 drain absorbed by the linear part (T1, T2); T1, T2, T3 not run.
- L32: C32-1 sign-correct separation for all q < 1; C32-2 behaviour for q ≥ 1.
- L33: C33-1 mean-reverting projection lowers the floor.
- L35: C35-1 geometric ε(G) for thinning; C35-2 Lai optimality; C35-3 PAC rescale 1/(1+1.645 sd); C35-4 Kesten-exponent delay for λ ≤ 0 < ln E_Q e.
- L36: C36-1 O(1/n_c) coverage of the all-windows pool; C36-2 β of observed windows → regime-chain β; C36-3 magnitude of gate optimism on the real estimator.

**Unmeasured constants and unverified items.**
- Thinning interval G and ε(G); the decorrelation lag of the novelty score under the null.
- The damping a (window bounds on a validation split), Δ_50 as a population quantity (estimated, not computed), per-mode a_i.
- Floor of the energy projection: sd(h₅₀) 5 to 26 percent is an estimate; era 3 rests on 349 transitions from 3 trajectories.
- the energy-rescale margin 2.8× is a property of the true flow, not of the fitted A.
- Ledger discrepancy of 0.015 in E₁ log L (μ = .5, κ = .5): unreconciled. The builder damping rows (0.0114/0.622, 0.0154/0.595) are quoted, not re-run.
- Initial law of the first regime in `ns2d.regime_schedule` (H36-1); the real-estimator size of the gate optimism (C36-3); the L-GIVEN text in the PREREG (L31(iv)); whether `L24c_Calibrator.lean` compiles and is built.
- Lean names: `C`'s are grepped against the pinned Mathlib; `F`'s and `H`'s are recalled and unverified (Picard–Lindelöf availability, `snorm` names, `Submartingale.expected_stoppedValue_mono`). Not in Mathlib at the pin: Kreiss, Schur triangularization, Hoeffding, Minkowski's integral inequality, an exchangeability API, Lorden's inequality.
- Evidence limits: all numbers are seed-3 or toy (6-dimensional, Lorenz-96 N = 8, kernel-ridge AR(1)); none is an eval-seed result; l28 numbers are rms-absolute on 20 samples and not the `bars.rel_err` metric; Prop 3 uses one pooled velocity channel.

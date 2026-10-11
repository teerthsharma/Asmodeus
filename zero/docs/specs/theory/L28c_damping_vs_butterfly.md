# Foreman: L28c (damped increment model vs the frozen butterfly bar S4/K8)

Theory only; nothing new was run. "Quoted" = taken from the brief, PREREG_AMENDMENT_11_DRAFT.md (S4 lines 88-92, K8 line 112), N5, Builder, or my
earlier outputs (`foreman.md`). "Arithmetic" = computed by hand from quoted numbers. "Plug-in" = closed form with idealised inputs.
Status: PROVEN, CONJECTURE (gap stated). Bed: run_primary.LAWS (4 laws, nu in {0.01, 0.02}, x-independent forcings).

## Verdict

Dr House's relation lambda_model = gamma * lambda_data is CORRECT for a model whose increment m is C^1-close to the vector field (it is a
time-rescaling identity), and WRONG for velocity-linear models, whose exponent is a property of the matrix A_gamma, not of the flow. Under
either regime the joint set {one-step <= 0.5 P} ∩ {E50 < baseline} ∩ {S4} is EMPTY or empty-by-a-hair on the quoted numbers; both emptiness
statements are conditional (CONJECTURE where marked). The pull is not between the two frozen accuracy bars; it is between damping (time
rescaling, which S4 reads) and the E50 ceiling.

## L28c-1 (time-rescaling identity)  [PROVEN to O(h^2); exponent-level step CONJECTURE]

Hypotheses. H1: x_{t+1} = Phi_h(x_t), Phi the flow of xdot = G(x) (G smooth), h the snapshot step. H2: the model is
T_gamma(x) = x + gamma*m(x) with m = Phi_h - id + e, e = learned error. H3 (small h): h*||DG|| << 1.
Lambda(G) := Lyapunov exponent per unit time; lambda_data = h*Lambda(G) exactly (Phi_h is the time-h map).

(a) Backward error. With e = 0: T_gamma(x) = x + gamma h G + gamma (h^2/2) Gdot + O(h^3), Gdot = DG.G. The time-gamma*h map of the field
    G~ = G + ((1-gamma) h/2) Gdot has the same expansion to O(h^2) (match the h^2 coefficient: gamma/2 = gamma c + gamma^2/2 => c = (1-gamma)/2).
    So one T_gamma step advances physical time gamma*h of a field that differs from G at relative size (1-gamma) h ||DG||/2.
(b) Mode-wise exponent (commuting tangent, eigenvalue mu = e^{zh}, z = sigma + i omega). The tangent of T_gamma is (1-gamma) I + gamma DPhi_h, with eigenvalue
    1 - gamma + gamma e^{zh}; ln|...| = gamma sigma h + gamma(1-gamma)(sigma^2 - omega^2) h^2 / 2 + O(h^3).
    Proof: ln(1 + gamma(zh + z^2h^2/2 + ...)) = gamma z h + gamma z^2 h^2/2 - gamma^2 z^2 h^2/2 + O(h^3); take the real part.
    So lambda_model = gamma*lambda_data - gamma(1-gamma) omega^2 h^2/2 + ... : the correction is negative (rotation is damped by interpolation toward identity)
    and relative size (1-gamma) omega^2 h/(2 Lambda).  Arithmetic with quoted rms one-step rotation theta = omega h ~ 0.033 (persistence one-step 0.0331 rel):
    gamma(1-gamma) theta^2/2 ~ 1.3e-4 at gamma = 0.6, against gamma*lambda_data = 0.00328 (N5 lambda_data = 0.005462): ~4%. Dominant-mode theta unmeasured, so ±.
(c) Exponent-level claim lambda_model = gamma lambda_data (1 + O((1-gamma)h)) needs Lambda(G~) -> Lambda(G) as the field is perturbed by O(h).
    That is continuity/linear response of the Lyapunov exponent in the vector field: known for uniformly hyperbolic systems, UNPROVEN for truncated NS. CONJECTURE.
(d) Consequence: damping is a time rescaling. A deterministic damped rollout runs at physical speed gamma, so a per-STEP Benettin exponent scales by gamma exactly.
    S4's tolerance 0.25*lambda_data therefore demands gamma(1+eps_gamma) in [0.75, 1.25], eps_gamma ≈ -(1-gamma) theta^2/(2 h lambda_data) ~ -0.03..-0.06 => gamma >= ~0.77-0.80.

## L28c-2 (value error does not control the exponent)  [PROVEN (a); (b) CONJECTURE]

(a) Finite-time perturbation bound. Along a common orbit and common unit directions u_s, let M_s = (1-gamma) I + gamma DPhi_h(x_s) and DT_gamma(x_s) = M_s + E_s.
    If ||M_s u_s|| >= m_0 and ||E_s|| <= gamma*eps_D < m_0, then |ln||DT u_s|| - ln||M_s u_s|||  <= gamma eps_D / (m_0 - gamma eps_D), so
    |lambda_B(model) - lambda_B(ref)| <= gamma eps_D / (m_0 - gamma eps_D).  (Proof: |ln(1+t)| <= |t|/(1-|t|) with t = ||DT u|| / ||M u|| - 1.)
    Arithmetic: to keep this below eta = 0.25*lambda_data = 0.00137, with m_0 ~ 0.9 (per-step contraction ~ 1 - O(0.03)), the learned tangent must satisfy
    eps_D <~ 1.2e-3 per step in operator norm, i.e. derivative accuracy ~6x finer than the quoted one-step VALUE error (0.0075 rel). Value error says nothing about De:
    e small in sup norm allows ||De|| = O(1) (routers, era switches, TT blending). Not resolved by any fit statistic in the repo. CONJECTURE that FCPA meets it.
(b) Caveat: the protocol evolves model and data from the same origin, but their orbits separate (E50 ~ 0.6-1.4), so (a)'s common-orbit hypothesis fails. The comparison is of two
    finite-time exponents on different orbits; the gap includes orbit-to-orbit fluctuation (next section). CONJECTURE that the common-orbit bound is the dominant term.

## L28c-3 (velocity-linear models)  [PROVEN (a)(b); (c) plug-in]

(a) For a linear/affine map A, Benettin with renormalisation telescopes: lambda_B = (1/n) E ln(||A^n u|| / ||u||). Proof: prod_s ||A u_s|| = ||A^n u_0|| with u_{s+1} = A u_s/||A u_s||.
    It converges to ln rho(A) only as n -> infinity; at n = h_B = 50 it contains the transient (L28). For momentum modes the persistence eigenvalue 1 never moves with gamma
    (A_gamma = J + gamma(A - J), J = [[I,0],[I,0]]); a fitted W1 + W2 = s != 0 moves it to 1 + gamma s/(1 - gamma w) (scalar pair: f(1) = -gamma s, f'(1) = 1 - gamma(w+s)).
    So ln rho(A_gamma) ≈ gamma s/(1 - gamma w) ≈ 0 for all gamma < 1/w, NOT gamma*lambda_data. (Quoted rho(LS) = 1.0069 => ln rho = 0.00687 at gamma = 1, diff from 0.005462: 0.00141 ~ eta.)
(b) Pooled momentum channel, v_t = x_t - x_{t-1}, A_a = [[1+a, -a],[1,0]], g_n(a) = a(1-a^n)/(1-a). Then A_a^n u = (u1 + g_n w, u1 + g_{n-1} w), w = u1 - u2,
    and for a uniform unit u in the 2-D lag plane E ln||A^n u|| = ln((sigma_1 + sigma_2)/2) (exact for 2x2: E_theta ln(q1 cos^2 + q2 sin^2) = 2 ln((sqrt q1 + sqrt q2)/2)),
    with sigma_1^2 + sigma_2^2 = (1+g_n)^2 + g_n^2 + (1+g_{n-1})^2 + g_{n-1}^2 and sigma_1 sigma_2 = a^n.
(c) Plug-in at n = 50, target lambda_data = 0.005462 (band [0.0041, 0.0068] for ±25%): the band requires sigma_1 in [2.46, 2.81], i.e. g_50 in [0.62, 0.81], i.e. a in [0.38, 0.45].
    The one-step bar C1 needs a >= 0.524 (g_50 = 1.10, lambda_B ≈ 0.0104 = 1.9 lambda_data). Gap in a: 0.07. CONJECTURE for the fitted 180-dim map (single channel, isotropic perturbation
    in both lag blocks, not run). One evaluation settles it: (1/50) E ln||A_gamma^{50} u|| on the fitted companion for gamma in a grid.

## Joint feasible set

Notation (all from `foreman.md`): C1 = one-step <= 0.5 P: gamma*mu in [c - sqrt(c^2 - 3/4), c + sqrt(c^2 - 3/4)]; C2 = E50 below baseline; C3 = S4.
Momentum-class numbers (test split, a = gamma*w, w ≈ alpha_1 = 0.96-0.977): C1 a >= 0.524 => gamma >= 0.54-0.55 (upper end ~1.4, not binding); C2 a <= 0.712 => gamma <= 0.73-0.74.

Regime A, m C^1-close to the vector field (L28c-1): C3 gives gamma in [0.75, 1.25]/(1 + eps_gamma), and with the sampling budget below, [0.85, 1.15]/(1 + eps_gamma).
  C1 ∩ C2 = [0.54, 0.74] (momentum-class ceiling); Builder FCPA data: 0.7 with energy passes both bars (0.0114 / 0.622 vs 0.0284 / 0.6335), 1.0 fails C2 (1.245 with energy, 1.424 without),
  0.5 without energy misses C1 by 8% (0.0154 > 0.0142). The C2 crossing between 0.7 and 1.0 is not measured. Linear interpolation of the quoted with-energy E50 puts it near 0.71 and the
  curve is convex, so it lies at or above that.
  C2 ∩ C3: empty if the crossing is < 0.77-0.80 (C3 floor after eps_gamma) and < 0.85 with the sampling budget; nonempty only if E50(gamma) stays below the baseline up to gamma ≈ 0.8-0.85,
  which neither quoted point nor the momentum ceiling (0.74) supports. Verdict: EMPTY by 0.03-0.11 under the momentum ceiling; for FCPA itself undecided pending E50 at gamma in {0.75, 0.8, 0.85}.

Regime B, velocity-linear (L28c-3): C1 ∩ C3 = [0.524, ...] ∩ [0.38, 0.45] = EMPTY (plug-in, margin 0.07 in a). The binding conflict is one-step vs butterfly, not E50.
  CONJECTURE for the fitted map; sign of the statement is robust because lambda_B(a) is increasing in a (g_n increasing) while C1 sets a floor.

Why the pull is structural, not tuning: the E50-optimal deterministic output is a conditional mean (shrunk), the butterfly bar reads trajectory-wise sensitivity (unshrunk).
Both come out of one deterministic rollout only if the shrink is applied on the READOUT, not inside the recursion:
  x_hat_k^read = x_0 + s_k (x_hat_k^rollout - x_0), s_k = <D_k, D-hat_k> / ||D-hat_k||^2, E_k(s_k*) = p_k sqrt(1 - c_k^2), c_k = cos(D-hat_k, D_k),
  [PROVEN: Pythagoras, same algebra as Prop 1]. Dynamics stay at gamma = 1, so lambda_model carries only the C^1 error of L28c-2. This removes the S4 conflict but does NOT raise the E50
  ceiling: in the momentum class c_50^2 p^2 equals the same ~0.4% of E^2. Beating the baseline still needs positive displacement alignment c_50 from the nonlinear part.
  It also changes what "FCPA prediction" means in the frozen E50 bar (readout map vs state recursion); that is an amendment question, not mine to settle.
  Replacement route for the killed lever (uniform gamma as the E50 knob): horizon-indexed readout shrink s_k fitted on train origins; measured reason: uniform gamma is a time rescaling (L28c-1d) and moves lambda_B by factor gamma.

## Sampling: is S4 resolvable? (arithmetic from N5)

N5: lambda_data = 0.005462/step, SE = 0.0021 at 8 trajectories => per-trajectory SD ≈ 0.0021*sqrt(8) = 0.0059 ≈ 1.09 lambda_data. eta = 0.00137 < SE: at n = 8 the bar is unresolvable (agreed).
Zero-bias resolvability needs 1.645*0.0059/sqrt(n) <= 0.00137 => n >= 50 trajectories. The frozen test uses 64 trajectories x 5 seeds = 320: SE = 0.00033, 1.645*SE = 0.00055,
leaving a bias allowance of 0.00137 - 0.00055 = 0.00082 = 0.15 lambda_data, i.e. gamma(1 + eps) in [0.85, 1.15] in Regime A (used above).
CONJECTURE: trajectories independent, origins within a trajectory add no variance, and paired model-data differences have SD no larger than the data SD. If the bootstrap unit is the seed (n = 5), SE grows by sqrt(64) and the bar is unresolvable again.
So "S4 may be unresolvable regardless" is true at 8 trajectories, not established at the frozen test size.

## Limits (collected)

- Nothing here was measured on the fitted FCPA tangent; eps_D, c_50, E50(gamma >= 0.75) and the fitted-map lambda_B(gamma) are the four missing numbers.
- L28c-1(c) rests on continuity of the Lyapunov exponent in the vector field; unproven for NS and false in general for non-hyperbolic systems.
- Regime B uses one pooled channel; many channels with different w_i make the 180-dim lambda_B a Sigma-weighted mixture.
- C2 for FCPA is interpolated from two quoted points and the momentum toy; no run was made.
- Lean (Mathlib v4.7.0 era, names recalled, not grepped): (L28c-1b) `Complex.exp_bound`, `Real.abs_log_sub_add_sum_range_le`; (L28c-2a) `Real.log_le_sub_one_of_pos`, `abs_sub_le`; (L28c-3a) `Finset.prod_range_succ`,
  `Real.log_prod`; the 2x2 identity E ln = ln((s1+s2)/2) needs an integral not in Mathlib (replace by the Jensen upper bound for a one-sided statement); (L28c-1a) backward-error expansion needs
  flows/Taylor with remainder (`taylor_mean_remainder_lagrange`), flow existence as a hypothesis; (readout shrink) `norm_sub_sq_real`, quadratic minimisation. Lyapunov-exponent continuity is not Lean-able (no theory in Mathlib).

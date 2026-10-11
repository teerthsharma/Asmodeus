## Amendment 11 (pre-result)
Filed 2026-10-11, before any FCPA result exists. DRAFT for the coordinator: this text is appended to `PREREG.md` only after sign-off. Nothing here has been run.

**Source.** `docs/specs/2026-10-11-fcpa-design.md` §5 (beds and the one primary claim) with Amendments A, B and C. Where this text and the spec differ on a number, this text wins; where it only restates, the spec's lemma is the authority. FCPA enters as a third first-class candidate beside OBSERVER-ATLAS and PHASE-SPHERE (folder rule 2). Nothing is dropped because a side says otherwise; the measurements below pick the winner per claim.

**What the existing arms are, for this amendment.**
- **FCPA** is the reference implementation in `fcpa/` with the pins of section "Pins" below.
- **Global** is one model of the same family as an era model (the JEPA-in-slab of D5), trained on all eras with no router and no birth, at a width chosen so that its parameter count is within ±2% of FCPA's.
- **ASMODEOUS top-m** is the `asmodeous.py` arm with its own learned router and top-m selection, given as many experts as FCPA ends with, trained end to end with no era labels.

## Bed pins (NS-2D multiverse)
All values are frozen in the bed config at the commit that precedes the first eval run. They are not tuned on eval seeds.

- **Solver.** Periodic 2-D vorticity on [0, 2π)², 32 × 32 grid, 2/3 dealiasing, RK4, solver step 0.01, one **step** = one snapshot = 10 solver steps. `fcpa/beds/ns2d.py: simulate`.
- **Encoder (E1).** `ns2d.encode` with kmax = 4: a truncated rfft as a unit complex vector ψ plus the amplitude channel a = ‖z‖. The state x_t is the unnormalised coefficient vector (a·ψ). All arms see the same x_t.
- **Laws.** R = 4 regimes, each a distinct (ν, f) with f time-constant inside a regime (E2): r0 = (0.010, 0.5·cos 4y), r1 = (0.020, 0.5·cos 4y), r2 = (0.010, 0.5·cos 2y), r3 = (0.020, 1.0·cos 2y).
- **Switching.** The regime follows a 4-state Markov chain: per step, stay with probability 1 − λ_sw, otherwise move uniformly to one of the other three. **λ_sw = 0.005 per step.** Regimes recur, so an era is a maximal interval and its class is the regime id (L14′).
- **Window.** w = 20 steps. **λ_sw · w = 0.1**, the bound that Amendment A sets for the identification bar.
- **Trajectories.** Per seed, 256 trajectories, each of length L = 400 steps, each with its own independently drawn switching schedule and initial condition (256 calls of `simulate` with `n_universes = 1`, so no two trajectories share a schedule). A random-field burn-in of 200 solver snapshots precedes step 0 and is discarded.
- **Split, by whole trajectory** (`conformal.trajectory_split`, rng = seed): 128 Train, 64 Cal, 64 Test. A trajectory is never cut.
- **Seeds.** Eval seeds **0, 1, 2, 4, 5**; **tuning seed 3 is never evaluated** (`evaluate` asserts it). Bed data seed = s; tuning bed seed = 2003. This is the convention of the T4a pins (PREREG §2, Appendix F).
- **Test stream.** The Test trajectories concatenated in index order: **T = 64 × 400 = 25,600 steps** per seed. This is the stream ACI runs on.
- **T sizing.** The coverage bar needs T ≥ (max(α₁, 1−α₁) + γ)/(0.02·γ). With α = α₁ = 0.1 and γ = 0.05 this is 0.95/0.001 = **950 steps** (L24*, Amendment B). The pinned T is 26.9 times that. **Precondition (checked in code):** T < 950 voids P4. The L24′ fallback of Kill K2 (γ = 0.0022, ≈ 20,500 steps) also fits inside T.
- **FCPA pins.** α = 0.1, γ = 0.05, α₁ = α, δ = 0.05, κ = 0.5, w = 20, d_c = 8, K_max = 8 eras, Mondrian (per-era) calibration, L6′ newborn inclusion rule. Only the learning rate of each gradient-trained part is tuned (below).
- **Tuning.** As Amendment 9: each gradient-trained arm picks one lr from {1e-3, 3e-3, 1e-2} on tuning seed 3 only, 1,000 steps, Adam, batch 64, score = val E50 (below) after 1,000 steps. ASMODEOUS top-m also picks m (below). Evaluation: 5 eval seeds, 5,000 steps per gradient-trained part, Adam, batch 64.
- **Births.** The primary claim is scored on the **frozen** era set (births disabled in the Test stream), so P1 and P2 compare fixed models. RSL growth is scored only in S2.

## Metric definitions (the code contract)
Every bar below is computed by one module, `fcpa/bars.py` (not yet written; it is written RED-first against synthetic inputs, so each definition has a test that fails before the code exists). A bar whose function is missing or whose tests are not green is NOT RUN, never PASS.

- **Origins.** t₀ ∈ {20, 30, …, 340} in each Test trajectory (33 origins; 20 steps of history, t₀ + 50 ≤ 390 < L).
- **E50(arm, traj).** The mean over the 33 origins of ‖x̂_{t₀+50} − x_{t₀+50}‖₂ / ‖x_{t₀+50}‖₂, where x̂ is the arm's **closed-loop** 50-step rollout from the true history x_{t₀−19..t₀}. Closed loop means the arm feeds on its own predictions: for FCPA, the router, the conformal set and every era choice are recomputed on predicted states, and **no timestamp and no era label is visible** at any step. The error is in the common encoded space, so every arm is scored on identical targets.
- **Seed score.** E50(arm, s) = the mean of E50(arm, traj) over the 64 Test trajectories of seed s.
- **Cluster bootstrap (the one inference tool).** B = 10,000 resamples, seed 20261011, resampling **whole Test trajectories with replacement within each seed**, and recomputing the statistic as the mean over the 5 seeds. "95% one-sided bound" means the 95th percentile (upper) or 5th percentile (lower) of the resampled statistic. The T4a pins record that 5 seeds are the minimum for the bootstrap.
- **Ratio.** ρ(arm) = E50(FCPA) / E50(arm), each as a seed-mean, resampled jointly (paired on trajectory).
- **Era map.** Each learned era is mapped to a true regime id by majority vote of the regime at its Cal steps. The map is frozen on Cal before Test is read. Several learned eras may map to one regime.
- **Compute.** Amendment 7's binding convention (`active_macs`). Inference MACs/step for FCPA = TT + Born/observer + router and conformal scoring over all live eras + Σ_{k∈C_t}(crossing S, zoom, J_k, S†), averaged over the Test stream. Training MACs = every forward MAC, backward counted at 2× forward for gradient-trained parts, closed-form fits (ridge, normal equations, STRidge) counted at their solve cost, including bound estimation, calibration and ACI.

## The primary claim
**One claim.** On the NS-2D multiverse bed, FCPA, scored on all five eval seeds, satisfies **all four** of P1–P4 together with the compute rules C1–C2. The test is an intersection-union test: every component must pass at its own level, so no multiplicity correction applies and none is taken.

**P1. FCPA against Global (equal parameters).**
- **Bar.** ρ(Global) ≤ 0.95, **and** the 95% one-sided upper bound of ρ(Global) is < 1, **and** E50(FCPA, s) < E50(Global, s) on at least 4 of 5 seeds.
- **Parameters.** Global's width is solved by integer search until its parameter count is within ±2% of FCPA's final count (TT, observer V, every era model, crossing generators G_a). Both counts come from one counter and are logged.

**P2. FCPA against ASMODEOUS top-m (compute-matched).**
- **Bar.** Same three conditions as P1 with ρ(ASMODEOUS).
- **m.** m is the largest m ∈ {1, 2, 3, 4} whose mean inference MACs/step is ≤ 1.05 × FCPA's mean inference MACs/step on the tuning seed. If no m ≥ 1 fits, m = 1.
- ASMODEOUS is tuned for lr on seed 3 under the same rule as FCPA's parts. Its tuning budget is therefore equal.

**C1/C2. Compute rules (preconditions of P1 and P2).**
- **C1 (inference).** FCPA's mean inference MACs/step ≤ each baseline's. A failure fails that comparison. It is not a void.
- **C2 (training).** Each baseline is trained, per seed, to a budget equal to FCPA's **total** training MACs on that seed (steps = ⌊B_FCPA / MACs per step⌋). FCPA's total is the sum of every component listed under "Compute".

**P3. Era identification.**
- **Score.** Over Test steps that lie more than w = 20 steps from the nearest true switch (windows straddling a switch are ambiguous by construction), the fraction of steps where the router's top-scoring era maps (era map) to the true regime. Pooled over all trajectories of a seed; the bar uses the mean over the 5 seeds.
- **Bar.** Seed-mean **≥ 0.9**. Its 95% one-sided lower bound and the worst seed are reported.

**P4. ACI long-run coverage.**
- **Score.** On each seed's Test stream (T = 25,600), the set C_t at ACI level α_t, with the Gibbs–Candès boundary convention of Amendment C (`conformal.ACI.update`). err_t = 1[true regime ∉ map(C_t)] (the era map is applied to the learned eras in C_t); coverage = 1 − (1/T) Σ err_t.
- **Bar.** |coverage − 0.9| ≤ 0.02 on **every** one of the 5 seeds.
- **Fact (spec Amendments A, C).** With the boundary convention the deviation is at most 0.95/(0.05 · 25,600) = 0.00074 for every sequence, so P4 is an integrity check on the implementation, not an empirical falsifier. It is in the primary claim because a coverage claim that fails to hold on its own stream voids the router's calibration. Label and trigger use one era definition (the regime id); a mismatch is an integrity fault.

**Reported with the primary, with no bar:** E50 split into within-era origins and origins whose 50 steps cross a true switch; mean |C_t|; per-seed ρ; parameter and MAC tables; the unconstrained-arm comparison; the distribution of learned-era count at the end of fit.

## Secondary bars (Holm-corrected)
**Family.** The five bars S1–S5 form one Holm family at FWER 0.05. Each has a one-sided p-value defined below. Sort ascending, p₍₁₎ ≤ … ≤ p₍₅₎; reject p₍ᵢ₎ if p₍ᵢ'₎ ≤ 0.05/(5 − i' + 1) for every i' ≤ i, that is, step down and stop at the first failure. A bar PASSES only if its null is rejected by Holm **and** its point condition holds. The primary claim is not part of the family. Kill rules use point estimates and are not Holm-gated.

**S1. L-GIVEN: NS-form arm against the learned-operator arm.**
- **Arms.** NS-form = FCPA with D3's flow law θ = (ν, f) fitted by the NS projection (E2′, `bounds.within_slab_estimate`). Learned = FCPA with the NS law replaced by sparse library regression with no fixed terms (`bounds.library`, `bounds.learn_operator`, STRidge), certified by `bounds.certify_sparse` (L-GIVEN lemma). Every other component, pin and seed is identical.
- **Statistic.** ρ_G = E50(NS-form) / E50(learned), paired cluster bootstrap.
- **Bar.** The 95% one-sided upper bound of ρ_G ≤ 1.02 (NS-form beats or ties the learned operator, with a 2% tie margin). Null (rejected for a pass): ρ_G > 1.02. p = P_boot(ρ_G > 1.02), the fraction of bootstrap resamples with ρ_G > 1.02.
- **Reported.** The certified verdict of the learned arm per library term (PRESENT / SMALL / REFUSE at resolution 2η) and the fraction of seeds where only the NS terms are PRESENT.
- **Departs from.** Every PDE-discovery owner fits a fixed term set or discovers one without a comparison against the fixed form on a held-out rollout.

**S2. Anytime-valid false-birth rate ≤ δ.**
- **Null streams.** Per seed, 200 independent null streams of 2,000 steps each, drawn from a **single** regime that the model already has an era for (no new law appears), with a fresh initial condition, scored by the full RSL pipeline (conformal novelty p-values into `conformal.BirthTrigger`, δ = 0.05, κ = 0.5). A **false birth** is any birth that fires in a null stream.
- **Statistic.** X = number of null streams (of n = 1,000 pooled over the 5 seeds) with at least one false birth.
- **Bar.** X/n ≤ δ = 0.05 and the exact one-sided 95% Clopper–Pearson upper bound ≤ 0.05 (the claim is that the rate is demonstrably at or below δ). Null (rejected for a pass): rate > δ. p = P(Bin(1000, 0.05) ≤ X).
- **Reported (no bar).** Detection delay and recall of births on a separate stream where a fifth regime appears; the mean number of births per true new law.
- **Departs from.** ACI-triggered births (L24, L24′) grow linearly in αT; the martingale trigger never reads α_t.

**S3. Certified-change precision.**
- **Event.** A certified change is a time t at which `bounds.certified_change` returns CHANGE (disjoint intervals at per-window coverage δ_w = 0.025, so L11 gives P(false certified) ≤ 2δ_w = 0.05), over windows [t − w, t) and [t, t + w), on the Markov bound (observer outcome kernel) or the NS bound (ν, f). Certified events within 2w of each other merge into one.
- **True positive.** The event has a true regime switch in [t − w, t + w]. Precision = TP / (TP + FP), pooled over all Test trajectories and the 5 seeds. REFUSE is not an event and is not counted against precision; the REFUSE rate is reported.
- **Bar.** Precision ≥ 0.95. Null (rejected for a pass): precision ≤ 0.95. p = P(Bin(n_events, 0.95) ≥ TP). **Precondition:** n_events ≥ 50 pooled, otherwise S3 is VOID (not PASS, not scored in Holm; the family shrinks to four).

**S4. Butterfly: |λ_model − λ_data| ≤ η, inside the predictability horizon.**
- **Definition (Benettin, two-trajectory).** From each origin, perturb the encoded state by a random direction of relative size ε₀ = 1e-6, evolve the perturbed and unperturbed states for h_B steps, renormalising the separation back to ε₀ every step, and take λ = (1/h_B) Σ ln(‖separation_s‖ / ε₀) per step. λ_data uses the true solver; λ_model uses the FCPA closed-loop rollout from the same origin and the same perturbation. Both are averaged over the 33 origins and 64 Test trajectories per seed, then over seeds.
- **Horizon.** H = ln(0.3 / ε₀) / λ_data, computed from the data on tuning seed 3 before any model runs. **h_B = min(50, ⌊H⌋).** The horizon is inside the predictability horizon by construction.
- **Tolerance.** **η = 0.25 · |λ_data|**, frozen on tuning seed 3.
- **Bar.** The 95% one-sided upper bound of |λ_model − λ_data| ≤ η. Null (rejected for a pass): |λ_model − λ_data| > η. p = P_boot(|λ_model − λ_data| > η).
- **Preconditions.** λ_data (tuning seed 3) ≤ 0, or H < 10: S4 is VOID ("bed not chaotic enough to test it"), never PASS.
- **Reported.** The unconstrained arm's λ (no L21 parametrisation), as context. It has no bar.

**S5. Laziness ≤ 1.2 eras per step.**
- **Score.** Mean |C_t| over the Test stream, where C_t is the conformal era set used by the decision rule (|C| = 0 counts as 0 eras consulted; a blend over C counts |C|; newborn inclusion counts).
- **Bar.** Seed-mean |C| ≤ 1.2. Null (rejected for a pass): mean |C| > 1.2. p = P_boot(mean |C| > 1.2).

## Kill rules, each with its replacement route
Kills are point-estimate conditions. A kill removes the named claim. It does not delete code.

| # | Kill | Fires when | Replacement route (each needs its own pre-result amendment before anything is scored) |
|---|---|---|---|
| K1 | FCPA against Global (P1) | ρ(Global) > 1.0 on the seed-mean, or C1/C2 fail against Global | Revert rule below. The diagnostic "is it the router or the era model" is read from the S1 arms and the per-era E50 split, which are already logged. |
| K2 | FCPA against ASMODEOUS (P2) | ρ(ASMODEOUS) > 1.0 on the seed-mean, or C1/C2 fail against ASMODEOUS | Revert rule below. Parts that worked (differentiator, bounds, RSL births) are offered to ASMODEOUS as parts under folder rule 1; the FCPA router is not. |
| K3 | Era identification (P3) | seed-mean < 0.8 | Router replaced by the law-class address of L14′: nearest fingerprint κ plus the HMM forward filter over era classes. The conformal set stays as the calibrated uncertainty layer. |
| K4 | Coverage (P4) | |coverage − 0.9| > 0.02 on any seed | Integrity fault (see P4 Fact). Fix the implementation. If the Gibbs–Candès convention is verified in code and the fault stays, ACI is replaced by fixed-level split conformal L6′ and the coverage claim is restricted to exchangeable segments. |
| K5 | NS form (S1) | ρ_G point > 1.10 (NS-form worse than the learned operator by more than 10%) | The learned operator becomes the default hypothesis class for the flow law. The NS form is reported as "given structure" only. Between 1.02 and 1.10 the bar fails with no kill and the result is likewise reported as "given structure". |
| K6 | Anytime-valid births (S2) | X/n > 2δ = 0.10 | The martingale trigger leaves. Births fall back to L24′: split-conformal novelty with n_min = 50 and γ = 0.0022 (about 20,500 steps, which fits in T). The false-birth claim becomes the αT-linear bound of L24. |
| K7 | Certified change (S3) | precision < 0.90 | Certified change is reported as a REFUSE-style diagnostic only. Era split and merge use a window-invariant segmenter without a certification claim (planimeter-style DSU on windows). |
| K8 | Butterfly (S4) | |λ_model − λ_data| > η (point) | The "butterfly bounded" claim leaves. It shrinks to the proved linear and unitary part (L21 demoted per Amendment A). Nonlinear sensitivity is reported, not claimed. |
| K9 | Laziness (S5) | mean |C_t| > 2.0 | The decision rule is capped at hard top-1 (jump), the blend path is removed, and coverage is claimed over a fixed top-m set rather than the full conformal set. |

## The revert rule
- If the primary claim fails (any of P1, P2, P3, P4, or C1/C2), **FCPA stays behind a feature flag**, off by default. The reshape of `zero/` is reverted to the pre-FCPA default (OBSERVER-ATLAS, PHASE-SPHERE and ASMODEOUS as they stand); `fcpa/` remains in the tree and the `FCPA` flag enables it. No code is deleted.
- Secondary passes cannot rescue a failed primary claim. A secondary kill on a passed primary removes that secondary claim only.
- The language bed (phase 2) and the TempLAMA arm are **not started** unless the primary claim passes.
- The primary bars are not re-run with changed constants. A change needs a new pre-result amendment before the next eval run; a re-run with the same constants is allowed only to confirm a fault, and both outputs are reported.

## Computability statement
- **No beyond-Turing claim is made, anywhere, by any bar in this amendment.** The Gödel-universe figure is the source of the picture (eras as slabs, the CTC as asking an era), not of any computational power.
- **P_CTC = PSPACE** (Aaronson and Watrous, 2009; CITED). The CTC loop D10 is realised as a **computable damped fixed-point search** (L22*/L22†): it either converges within its a-priori iteration budget, or the dispatcher returns REFUSE. FCPA computes nothing a Turing machine with polynomial space cannot.
- Forced Navier–Stokes is Turing-universal (corpus 376; CITED), so no bar here claims anything about the unbounded horizon. The NS-bed bars are **bounded-horizon**: E50 is 50 steps and h_B ≤ 50 steps.

## Departs from
No owner combines a certified era boundary (L11), a calibrated era set under drift (L7, L24*), and non-destructive growth (L23) in one predictor. Prior art for the pieces: switching Kalman and Markov-switching models (no certified boundary), MoE routers (no coverage guarantee), Gibbs–Candès ACI (no growth), mixture-of-world-models (no calibrated abstention). Row IDs for OWNERS.md are added at sign-off.

## Order
Lean v1 Tier A lemmas are committed with `sorry` first and the CI fails on `sorry`. `fcpa/bars.py` and its tests go RED-first before any eval run. Tuning on seed 3 runs first and freezes the lr's, m, h_B and η. Eval runs follow in one batch. P4's precondition (T ≥ 950) and S4's VOID rule are checked before any model is scored.

## Limits
- The bed constants (R = 4, λ_sw = 0.005, w = 20, the four (ν, f) pairs, the 25% tolerance of η, the 2% tie margin of S1, the 0.95 ratio margin of P1 and P2) are pins chosen by the amendment author without a calibration run; they are frozen at the commit before the first eval run. If the bed is not chaotic at these pins, S4 is VOID and says so.
- The primary claim is scored on a frozen era set. It does not show that RSL growth improves prediction; S2 shows only that false births are rare.
- Five eval seeds is the minimum for the bootstrap (T4a pins); the bootstrap resamples trajectories within seed, not seeds, so it understates between-seed variance. The "at least 4 of 5 seeds" condition in P1 and P2 is the guard.
- P4 is a theorem under the boundary convention. Passing it is not evidence of a good router; P3 and S5 are.
- Compute is counted in MACs under Amendment 7's convention, not wall-clock. FCPA's router and conformal scoring are charged; solver cost for generating data is not charged to any arm.
- The baseline ASMODEOUS is given FCPA's final expert count, which FCPA learns; a baseline given a smaller, hand-chosen count is not run.

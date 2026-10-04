# PREREG draft: HYBRID arm

Code: `atlas/hybrid.py` (`HybridAtlas`, built by a nurse and reviewed by Kutner; no other file is edited).
`HybridAtlas(TimeTravelAtlas, CoherentAtlas)` has MRO HybridAtlas -> TimeTravelAtlas -> CoherentAtlas ->
ObserverAtlas. It combines three parts:
- OBSERVER-ATLAS real charts (encoders, predictors, gluing, gate, top-m measurement);
- Arm Q decider: p = |U psi|^2, psi_k = sqrt(softmax(s))_k e^{i theta_k(q)}, U = low-rank Cayley;
- the time-traveller CTC loop (START = preselect M, life-line = candidate futures, HALT = self-consistent set).

Nothing is bound together by hand. Every part has a limit in which it vanishes, and the arm starts at the
first of these limits:
- Cp = 0 and gamma = 0 gives OBSERVER-ATLAS exactly (Cp = 0 is the init; theta then has no effect).
- gamma = 0 gives CoherentAtlas.
- Cp = 0 gives TimeTravelAtlas.

**Two preregistered variants, scored identically. Neither is picked by results.**
- **hybrid-TT** (`HybridAtlas`): the discrete chart-set loop of `atlas/time_traveller.py`. It acts in
  `rollout` (planning). START = top-M; HALT = the set maps to itself or max_loops is reached.
- **hybrid-FP** (`HybridFPAtlas`): Foreman's continuous CTC fixed point (`phase_sphere.fixed_point`) on the
  candidate weights, w <- coherent_probs(s0 - gamma * dev(w)) restricted to the M candidates. It is solved
  without gradient, then one differentiable step is taken from w*. It acts in `rollout` and (if the nurse's
  report confirms it) per step in `forward` with one-step futures. Worst-case `loop_iters` are charged.
- TO2 / TO3 are scored on each variant's own loop. For hybrid-TT the convergence fraction is the share of
  rollouts that halt before max_loops. Both variants are reported in every table. A "hybrid" claim
  holds only for the variant that clears it, and the report names that variant.

**psi interface (TO1 cell, Foreman K§5.3).** `forward(...)['psi']` = the complex chart amplitudes
U(sqrt(softmax(s)) e^{i theta}) [B, T, K] at the same per-step context c the observer routes on.
`readouts.chern_of_arm` reads it. Caveat for the readout owner: `_chern_psi` passes all icosphere vertices
as ONE sequence [1, n, D]. For any arm with a recurrent context (atlas, hybrid), psi at vertex i then
depends on vertices before i in sequence order. The readout should pass [n, 1, D] (one vertex per
sequence). Foreman to decide; until then the hybrid Chern is also reported via `hybrid.chern_s2`
(single-frame context).

## Matched sizes (L-BR)
- Equal active FLOPs (`active_flops`, including the coherent decider) and equal total params (`n_params`,
  including Cp, Cq and theta_keys) against atlas, phase-sphere and every owned arm, matched with
  `arms/match.py` by chart hidden width.
- Planning uses `time_traveller.rollout_flops` / `budget_samples`. The hybrid also pays the coherent START
  decider and the worst-case loop MACs.

## Claims, scored identically for atlas, phase-sphere and hybrid
All three arms get the same beds, seeds (3 + tune seed), metric code and bars. The table below is filled
in once per arm.

| claim | bed | readout | bar (from) |
|---|---|---|---|
| C1 geometry | s2, t2, klein, so3, r2 (lift) | discontinuity, h = 1 / h = 10 error | PROMPT §8 T2 |
| C2 topology (Betti) | same | nerve Betti over Z2 | PROMPT §8 T2 |
| TO1 Chern | Berry bed + trivial control (ORIGIN) | c1 = FHS Chern of the observer amplitude bundle on an icosphere of inputs (`hybrid.chern_s2`, `ref/berry_check`) | ORIGIN TO1 |
| TO7 holonomy | t2, klein | sign holonomy along held-out loops (`phase_sphere.sign_holonomy`) | ORIGIN TO7 |
| TO2 fixed point | all | loop residual and convergence rate; h = 10 error vs one-pass routing | ORIGIN TO2 |
| C3 lazy scale | T3 sweep | loss vs K at fixed active compute | PROMPT §8 T3 |
| C5 time traveller | swingup, pointmaze | success, paired CI | PREREG_DRAFT_t5_t6 |
| T6 (measure only) | t2, swingup | QQ, order, dA, dB vs the untrained init | PREREG_DRAFT_t5_t6 |

Readout rules:
- The Chern and holonomy readouts are eval-only and never enter a loss (L-GIVEN).
- A real-valued arm (the atlas) has a trivial amplitude bundle by construction, so it reads c1 = 0. On the
  Berry bed this counts as a miss for the atlas, not a VOID.

## Hybrid-specific bars and kills
- **Bar (the "best of both" claim):** on each claim the hybrid is >= max(atlas, phase-sphere) - the claim's
  own tolerance, AND it is strictly better than both on at least one preregistered claim, with paired CI
  lower bound > 0. Equal matched sizes throughout.
- **Kill:** on >= 2 claims the hybrid is worse than the better of atlas and phase-sphere, which means
  composition costs more than it adds. The kill also fires if every learned limit collapses: ||Cp|| / ||Cq||
  < 0.01 and the TT loop picks the start set on >= 99% of steps. In that case the hybrid has learned to be
  the atlas and is scored as the atlas.
- **Ablations (preregistered, same budget):** hybrid with Cp frozen at 0; hybrid with gamma = 0; hybrid with
  theta frozen at 0. A hybrid win counts only for the part whose ablation loses it.

Prior (guess): ~10% that the hybrid clears the bar on a claim where both parents fail.

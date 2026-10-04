# PREREG DRAFT: event-driven laziness in time ("observed → exists", T-EV)

Draft by Taub, 2026-10-04. Not committed. It merges into `PREREG.md` only after Inspector review.

## Origin and owner

Origin: Seal's paper `ref/origin/AETHER_qGPU_event_driven.pdf`. A new frame is computed only when
Δ(t) = ‖μ(t) − μ(t−δt)‖₂ ≥ ε(t), so FPS ≈ ‖μ̇‖ / ε, with ε adapted by a learned governor.

Here it runs on a classical GPU only. The adaptive-POVM governor in the paper is not part of this test.

Owners to hunt before any bar (L-OWNER):
- event cameras and event-driven vision [K];
- send-on-delta / level-crossing sampling [K];
- skip RNNs (Campos et al. 2018) [K];
- PonderNet / adaptive computation time [K];
- Mixture-of-Depths 2024 [K];
- delta networks (Neil et al. 2017) [K].

Skip-RNN and delta networks are the closest: both use a learned skip or update threshold in time.
The gap rule applies: novelty is earned only by a decisive test that the owned parts fail.

## Mechanism (`atlas/event_driven.py`)

`EventDriven(arm)` wraps any contract arm: ATLAS, PHASE-SPHERE, HYBRID, or an owned baseline.

- **Trigger.** Per row:
  - t = 0 always triggers.
  - For t ≥ 1 the deviation is Δ_t = ‖o_t − o_{τ}‖₂ / √D, where τ is the last trigger time.
  - The step triggers when Δ_t ≥ ε, with ε = softplus(ρ) learned.
- **Exact semantics.** Frames that do not trigger do not exist for the arm.
  - The arm runs on each row's compacted triggered frames.
  - Its outputs are sample-and-hold in between.
  - A hold step predicts no change.
- **Learning ε.** In training only, ε is learned through a straight-through sigmoid relaxation, plus a
  compute-cost term λ_cost · mean(s). Evaluation always uses the exact compaction path.
- **FLOP accounting.** Executed FLOPs = (#triggers) × arm.active_flops(). Skipped steps cost only the
  Δ check (D MACs, reported separately).

## Contract tests (RED first, `tests/test_event_driven.py`)

- ε = 0 reproduces the always-compute arm exactly.
- ε = ∞ never recomputes after t0.
- Outputs at trigger steps equal the arm run on the hand-compacted subsequence.
- Executed FLOPs = triggers × active_flops, and they are non-increasing in ε.
- The gradient reaches ρ.

## Quality metric (pinned so hold steps cannot game it)

`metrics.pred_error` on the event arm's own held latents g̃ is gameable: a hold makes copy-last
perfect, so npersist improves for free. The event arm is therefore scored against the always-compute
latents of the same weights:

  npersist_ev(h) = MSE(ĝ̃_{t+h}, g^full_{t+h}) / MSE(g^full_t, g^full_{t+h})

Here g^full is the wrapped arm evaluated at ε = 0, which is exact by the test above, and ĝ̃ is the event
arm's h-step prediction. The loss of quality is

  ΔQ = npersist_ev(1) / npersist_full(1) − 1

## Beds, arms, seeds

- Beds: `s2`, `so3`, and the swing-up bed (`test_swingup.py`'s bed: T²×R², actions), all D = 256 lift.
  The neutral bed `r2` runs beside them (L-WALD).
- Arms: ATLAS (K = atlas(M), m = 2), PHASE-SPHERE and HYBRID at their PREREG configs, each wrapped.
- Baselines:
  - the same arm, always-compute;
  - the same arm with a fixed ε swept over Δ-quantiles {0.25, 0.5, 0.75, 0.9} of the training beds;
  - a skip-RNN-style owned baseline: single JEPA with a learned binary update gate, same cost term.
- Seeds 0, 1, 2. λ_cost and lr are tuned on seed 3.

## Bar and kill

- **Bar (per arm):** on ≥ 2 of 3 beds, the learned-ε arm executes ≥ 2× fewer FLOPs per step (mean
  trigger fraction ≤ 0.5) with ΔQ ≤ 0.01. The 3-seed mean must hold the bar, and so must the
  worst seed for the FLOP ratio.
- **Neutral:** on `r2`, ΔQ ≤ 0.01 at the same λ_cost. A reported saving there is a property of the bed's
  dynamics, not a claim.
- **Learned vs fixed:** learned ε must match or beat the best fixed-ε quantile on the
  (executed FLOPs, ΔQ) frontier. If a fixed quantile dominates it, the "learned governor" claim is dead
  and the mechanism is plain send-on-delta [K].
- **Kill:**
  - no ε reaches ≥ 2× fewer FLOPs at ΔQ ≤ 0.01 on ≥ 2 beds; or
  - the skip-RNN baseline reaches the same frontier point (owned, so the result is measurement only).

## T7 coupling (systems, µs/step on the 4060)

Executed FLOPs are not wall time. A data-dependent skip cannot be captured in a CUDA graph, and at
B = 1 every step pays the Δ check plus a host-visible branch.

- **Measure** (bench_t7-style protocol, eager):
  - c_hold: Δ check plus reuse;
  - c_trig: Δ check plus the fused step;
  - µs/step = (1 − r)·c_hold + r·c_trig, at the trigger rates r measured on s2, so3 and swing-up
    validation data.
- **At B > 1:** rows trigger independently, so a step runs the fused kernel on the triggered subset.
  A variable-size gather has no graph.
- **Systems bar:** wall µs/step ≥ 1.5× lower than always-compute (graph mode, its best mode) at the
  measured r on ≥ 2 beds, at B = 1 and B = 64.
- **Prior from bench_t7:** this is likely to fail at B = 1. The atlas step already sits on the ~60 µs
  graph floor, and eager single-launch cost is about the same. Report it either way.

## Open for Inspector

- Action handling across gaps: actions between triggers are dropped in v1, and the upgrade is to
  integrate them. This matters on swing-up, where actions are dense.
- Per-chart triggers (chart k recomputes only when its own input deviates) are deferred. Whole-step
  triggers come first, because they are arm-agnostic.

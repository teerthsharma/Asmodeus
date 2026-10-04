# PREREG_DRAFT_phase_sphere.md — PHASE-SPHERE spec as built, and the readout matrix (Foreman, 2026-10-04)

**Status.** PHASE-SPHERE is one of three first-class candidates, alongside OBSERVER-ATLAS (Chase) and HYBRID (Kutner). It does not replace OBSERVER-ATLAS. All three obey the `common.py` contract, so every bed and metric scores them identically.

**Origin.** `ORIGIN.md` §3, steps 1–8. Each design choice there is tagged as a reading.

**Code** (written by sonnet nurses, reviewed by Foreman):
- `atlas/phase_sphere.py`
- `beds/__init__.py` (`berry`, `berry0`, `closed_loops`)
- `metrics.py` (`chart_transitions`, `orientation_holonomy`)
- `readouts.py`

**Tests:**
- `tests/test_phase_sphere.py`
- `tests/test_beds_phase.py`
- `tests/test_readouts.py`

## 1. Layer (as built)

| step | object | built as |
|---|---|---|
| 1 | state | ψ = normalise(E(o_t)) ∈ C^N, complex64. E is an MLP D→hidden→hidden→2N with GELU. An EMA target encoder `tgt_` is excluded from params. |
| 2 | gauges | K learned chart gauges u_k ∈ C^N, a null gauge u_∅, and per-chart frames V_k ∈ C^{N×d_c}. |
| 3 | decider | Born rule p = \|⟨u_k\|ψ⟩\|² / Σ (tested equal to softmax(2 log\|α\|)). The "observe nothing" outcome gives gate = 1 − p_∅. w_soft is Born over the K charts. w is the top-m of w_soft, renormalised. |
| 4 | lazy charts | Only the top-m charts compute z_k = V_kᴴψ · e^{−i arg α_k}. This gauge fix makes g invariant to the global phase of ψ (tested). |
| 5 | local dynamics | ẑ_k = Cayley(X_k(a) − X_k(a)ᴴ) z_k, which is unitary (tested). The real case equals Kutner's `cayley_apply` (tested). |
| 6 | glue | g = gate · Σ_k w_k φ_k([Re z_k, Im z_k]), with φ_k linear R^{2d_c}→R^{D_g}. |
| 7 | transition phases | χ_jk = arg(α_j ᾱ_k) on top-m pairs (antisymmetric). |
| 8 | CTC observer | w ← Born(future ψ̂(w)), iterated by `fixed_point` to tol or `loop_iters`, then one differentiable step. An amortised one-shot head is trained by KL to w*; `mode="amortised"` runs with no loop. |
| losses | | Per-chart JEPA against an EMA target in the same chart; overlap consistency; SIGReg on g (anti-collapse, mandatory); load balance; task; head KL. |

**Proposed T2 / TO dimensions** (numbers fixed at PREREG commit):
- N = 2n+2 complex dimensions, so S² and T² use C⁶ and SO(3) uses C⁸.
- d_c = n complex dimensions per chart.
- D_g = 2n+2, the same pin as OBSERVER-ATLAS (D_g ≤ 2n+2).
- K ∈ {1,2,3,4,8,16,64} and m ∈ {1,2}, the same grid as the atlas. TO1 and TO7 also run m = 3, since a nerve with β₂ needs m ≥ 3.
- loop_iters = 8, loop_tol = 1e-4.

**FLOP accounting.** Multiply-adds per time step: one complex MAC counts as 4 real MACs, and every CTC-loop iteration is counted. See §3 for the formula as implemented. PHASE-SPHERE is flop- and param-matched against the same reference as the atlas (`sweep_t2.configs`).

## 2. Beds added

| bed | state (EVAL ONLY) | obs | dynamics | meta n / atlas / β_Z2 | use |
|---|---|---|---|---|---|
| `berry` | ψ ∈ S³ ⊂ C², stored as [Re ψ₁, Im ψ₁, Re ψ₂, Im ψ₂] | lift(ψ) | n = Hopf(ψ) follows the s2 overdamped walk; ψ is horizontally transported (SU(2) minimal rotation), so ⟨ψ_t\|ψ_{t+1}⟩ is real | 3 / 2 / (1,0,0,1) | TO1 |
| `berry0` | n ∈ S² (the same n_t and actions as `berry`) | lift(n) | the same | 2 / 2 / (1,0,1) | TO1 control (γ ≡ 0) |
| `closed_loops(t2 \| klein)` | held-out closed loops by class: T² a, b, ab, 0 (all +1); Klein a (−1), b (+1), 0 (+1) | the same lift as the bed | — | — | TO7 |

## 3. FLOPs as implemented

Multiply-adds per time step, from `PhaseSphere.active_flops(mode)`. Notation: Lc = loop_m = min(K, max(4, m)) in loop mode, Lc = m in amortised mode, and h = hidden.

| term | formula | note |
|---|---|---|
| encoder | D·h + h·h + 2N·h | |
| Born scores | 4N·K + 4N | K charts plus the null outcome. This is the only K-dependent term (tested). |
| per candidate chart × Lc | 4N·d_c (Vᴴψ) + 2A·d_c² (X(a)) + 4d_c³ (Cayley) + 4d_c² (Uz) + 4N·d_c (Vẑ) | |
| CTC loop | (loop_iters + 1)·(6·Lc·N + 2N) | the +1 is the differentiable step. Counted at max iterations, which is an upper bound. |
| glue | m·4·d_c·D_g + 2m·D_g | |
| amortised head | 2N·2N + 2N·K | amortised mode only |

**Review notes (Foreman):**
- The loop runs over Lc = max(4, m) candidate charts, not over all K. That keeps it lazy.
- The rows above are counted at loop_iters, not at the realised iterations. An early halt therefore does not get compute credit; it is reported separately as `ctc_iters`.
- In amortised mode `w_soft` is the head's softmax, not Born. Nerve readouts in amortised mode read the head.

## 4. Readout matrix: each claim on each candidate

N/A is used only with a reason.

| claim / test | ATLAS (Chase) | PHASE-SPHERE | HYBRID (Kutner, coherent observer on atlas charts) |
|---|---|---|---|
| **C1 geometry** | disc_rate and fold_rate on g; linear probe vs PCA(obs) control | same, on g (real, gauge-fixed) | same, on g |
| **C2 topology** | nerve Betti (Z2 + Q) of the `w` co-selection at frozen τ; flag and soft-weight (`w_soft`, τ ≤ 1/4) variants; Mapper on the single-JEPA latent as baseline | nerve of Born weights ≥ τ (`w`, `w_soft`), the same code. Chern via `chern_of_arm` is an extra readout. | nerve of the coherent \|Uψ\|² weights ≥ τ, the same code |
| **C3 lazy scale** | loss vs K at fixed active FLOPs, m = 2 | the same. Here K enters only through K·N Born scores (tested), so no √K product keys are needed. | coherent mix is O(K·r) and product keys are refused (Kutner), so C3 is read only up to the K where O(K·r) ≤ the budget; beyond that N/A by construction |
| **C4 collapse** | effective rank of g and of per-chart z (`z`/`zidx`) under SIGReg 0.1× and 0× | the same: per-chart z is the real view of C^{d_c} | the same |
| **C5 time traveller** | `atlas/time_traveller.py` re-scoring vs owned MPC | the CTC loop is the time traveller: loop mode vs amortised vs one-pass, at equal FLOPs including iterations | Kutner's arm runs with `time_traveller.py` |
| **TO1 Chern (Berry bed)** | **N/A**: g, z and the router are real, so they carry no U(1) phase and no Chern class. Substitute readout: probe \|c₁\| from a complex pairing of g's first two PCs. It is reported, but carries no bar. | `chern_of_arm`: Fukui–Hatsugai–Suzuki on ψ over an icosphere of `berry` states; bar \|c₁\| = 1 on berry, 0 on berry0 (sign is convention-bound, Wilson); gauge sensitivity reported | Chern of the coherent amplitude Uψ over the icosphere, if the arm exposes it as `psi`; otherwise N/A (real amplitudes, sign-only phases) |
| **TO2 fixed-point observer** | N/A: one-pass router, no loop (ablation baseline) | convergence fraction (`ctc_residual` < 1e-4 within loop_iters) and h=10 npersist vs one-pass at equal FLOPs | N/A unless Kutner adds a loop |
| **TO3 halt before start** | N/A (no loop) | top-1 agreement of the amortised head with w* | N/A |
| **TO4 butterfly bonded via topology** | λ₁ from rollout Jacobians; flip rate of the nerve Betti vector and orientation holonomy under 1× noise | the same, plus the flip rate of Chern and Berry-sign | the same as ATLAS |
| **TO5 scattering front end** | the encoder is swapped for a Morlet cascade in every arm; equal-FLOPs comparison per arm | the same | the same |
| **TO6 rays through a lattice** | the router is replaced by r learned rays over a √K×√K chart grid | ray selection over the Born scores of the grid | N/A: the coherent mix needs all K amplitudes, so lazy ray selection is incompatible |
| **TO7 holonomy** | `loop_orientation`: sign of det of fitted chart transitions, multiplied along held-out loops (needs `w`, `z`, `zidx`) | `loop_orientation` (on the real view z, a complex-linear transition has det > 0 always, so this reads +1 by construction and is reported as a negative control), plus `loop_berry` (sign of the Berry holonomy of ψ, i.e. the primary readout) | `loop_orientation` |

**TO7 bar** (applies to the primary readout of each candidate):
- `to7_score.min` ≥ 0.95: every loop class must be ≥ 95% correct, with Klein a = −1 and all other classes +1.
- An abstention (None) counts as wrong.
- The bar must hold on 3/3 seeds.
- Why it is per class: in the untrained smoke, a constant +1 scored 0.667 overall, and abstaining on exactly the Klein a loops scored 1.0 under the old None-ignoring accuracy.

**TO7 in context.** The test adds a loop-level, sign-resolved readout. Rational Betti already separates T² from Klein at the global level, so TO7 is not the only test that can.

## 5. Open items before PREREG commit

1. The FLOP formula as implemented, pasted from the nurse report into §3 after review.
2. N, d_c and loop_iters per bed, confirmed by a smoke run (memory at K = 64 on the 4060).
3. HYBRID's interface for `psi`: needed to fill its TO1 cell.

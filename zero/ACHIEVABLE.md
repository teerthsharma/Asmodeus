# ACHIEVABLE.md: THE LIST for this phase (Foreman, 2026-10-04)

## End goal

**Asmodeous (०) is a new kind of AI that can hold a conversation after training.** Text goes in, and text comes out, generated on the RTX 4060. Its layers are the origin's design (ORIGIN.md): a phase state ψ on CP^{N−1}, read lazily by an observer whose decider is the Born rule, with unitary local steps per chart, gluing by gauge links, and a closed self-consistent observer loop. Topology is read out from what the layers learn.

It is not a transformer and not a JEPA. Those are baselines at equal compute, and Asmodeous must beat or tie them on preregistered cells. Taub's END_GOAL.md has not landed. This file states the same requirement and is to be merged with END_GOAL.md when it arrives.

## Compute

| resource | use | budget |
|---|---|---|
| RTX 4060 8 GB (local) | All **deciding** runs (PROMPT security law: "deciding numbers from the local RTX 4060, Kaggle reproduces"): T2, layer ablations, the LM bake-off at small scale, systems/T7 | one GPU job box-wide (Seal); torch threads ≤ 4 |
| Kaggle GPU (CLI configured) | Long LM pretraining of the **already-chosen** architecture, plus reproduction of 4060 deciding cells | per-session time limit [K: ~12 h/session, ~30 GPU-h/week; check the account]; every job resumes via Wilson's checkpoint format |

Measured 4060 cost (Foreman):
- Step time is launch-bound: atlas 46 ms/step, phase-sphere 56, hybrid 64, single JEPA 16 (batch 64×32).
- Two processes on the GPU give 1.18× throughput; three give 1.32×. Hence the one-job rule costs little.

## THE LIST (dependency order)

GPU-h = 4060 hours unless marked Kaggle.

| # | deliverable | owner | the test that proves it | depends on | GPU-h |
|---|---|---|---|---|---|
| 1 | **Reproducibility rule.** T0 cannot be byte-identical on this CPU: one cell differs at 5.0e-16 vs 4.4e-16, an OpenBLAS kernel effect, not numpy. ref stays read-only. Adopt a ≤ 1e-15 tolerance on rounding-level cells; all other lines byte-identical. | Wilson | `tests/test_laws.py` SHA pin, plus a tolerance diff test of identities.out | — | 0 |
| 2 | **PREREG.md commit.** Merge every PREREG_DRAFT_*. Pin CAL (eps/delta/far/fold_delta for 7 beds), cells, 8k-step rule with plateau extension, lr-per-family on s2, per-class TO7 bar, \|c₁\| bar, LM bars (row 9). | Wilson (Inspector audits order) | `test_laws::test_prereg_before_runs` | 1 | 0 |
| 3 | **Code freeze.** Commit zero/*.py; the sweep refuses dirty code. | Wilson | `sweep_t2.dirty_code() == []`; full CPU suite green | 2 | 0 |
| 4 | **Checkpoint/resume.** Atomic, bit-identical resume, and `probe()`. Done (Foreman nurse, Wilson's API). Kaggle session-limit resume uses the same file. | Wilson (queue), Foreman (run.py) | `test_run::test_checkpoint_resume_bit_identical` (green); plus a Kaggle 2-session resume smoke | 3 | 0.1 Kaggle |
| 5 | **T2 tier 1: the decisive cells.** 153 jobs: 3 candidates (ATLAS, PHASE-SPHERE, HYBRID) at K = atlas(M) and the good-cover cell; 4 owned baselines at equal active FLOPs; seed-major so a full seed lands first. | Foreman runs, Wilson queues | PREREG T2 bars: fold/disc vs torn reference, nerve Betti Z2+Q vs frozen Mapper, h=10 npersist vs best owned, neutral r2 within +2% | 3, 4 | **18.1** |
| 6 | **Origin-object verdicts.** TO1 (\|c₁\| on berry vs berry0), TO2/TO3 (loop converges and the amortised head is right), TO7 (per-class holonomy), and the Hopf single-chart control against C1. Beds are calibrated: berry median step 0.175 rad, \|mean n\| < 0.03, phase median 1.25–2.0 rad. | Foreman (PHASE-SPHERE), Kutner (HYBRID/Hopf), Chase (ATLAS) | TO1/TO2/TO3/TO7 bars in PREREG_DRAFT_phase_sphere.md; Hopf kill clause in ASMODEOUS.md | 3, 4 | ~6 |
| 7 | **ASMODEOUS v1 assembled from winners.** Each slot (state, decider = Parseval POVM, gluing = links, loop = damped CPTP, laziness, readouts) is kept only if its ablation loses. | Kutner builds, Thirteen readouts, House attacks | `tests/test_asmodeous.py` contract; per-slot ablation at equal FLOPs, 3 seeds, on t2 + klein + berry | 5, 6 | ~10 |
| 8 | **Language bed + tokenizer.** BPE (`tokenizers` installed), a small English corpus. TinyStories-scale models are reported coherent below ~30M params [K: Eldan & Li 2023, not fetched]. The dataset download needs Seal's yes and ~2 GB of the 81 GB free disk. Splits keyed by name (the lift lesson). | Thirteen (Kaggle bundling), Foreman (bed) | data hash pinned; train/val/test disjoint by document; contamination grep | 2 | 0 |
| 9 | **Language bake-off at equal FLOPs and params (deciding, 4060).** **ASMODEOUS-LM**: tokens → ψ ∈ C^N, chart steps, observer loop, next-token head = Born over a learned vocabulary frame p(v) = \|⟨e_v\|ψ⟩\|², i.e. the POVM as LM head. Baselines: GPT-style transformer and GRU, ~10–20M params, same tokens and budget. | Kutner (model), Cameron (baselines), Foreman (harness/metrics) | Bars (preregistered in 2): val perplexity ≤ 1.05× transformer; 3 seeds; samples graded blind on grammar and coherence (rubric fixed in PREREG) | 7, 8 | ~25 |
| 10 | **Talking model (Kaggle pretrain of the winner from 9).** Same code and checkpoint format, scaled 3–5×. Then instruction tuning on a small dialogue set so it answers prompts. | Thirteen (Kaggle), Wilson (resume), Taub (systems) | It answers held-out prompts. A fixed 50-prompt sheet is scored for on-topic, grammatical, non-repeating replies, against the transformer trained identically. The 4060 reproduces the final eval and serves chat locally (≥ 10 tok/s). | 9 | ~40 Kaggle, 2 local |
| 11 | **Systems: lazy compute pays on the 4060.** Fused, grouped chart kernels or CUDA graphs. Today every arm is launch-bound at ~45 ms/step. | Taub | T7 bar ≥ 1.5× faster than dense at equal quality; tokens/s for row 10 | 7 | ~4 |
| 12 | **T2 tier 2 (non-decisive).** K sweep (P-COUNT onset), m=1 tear cells, param-matched rows, d ∈ {n, 2n}. | Foreman | P-COUNT prediction table | 5 | 43.7 |

**Totals.**
- Decisive path (rows 5, 6, 7, 9, 11): about 63 local GPU-h.
- Talking model: about 40 Kaggle GPU-h, plus about 2 local.
- Tier 2 (row 12) runs after the decisive path, or in idle time. It never blocks the talking model.

## Already green (this session)

Each item below went red → green under TDD:
- **Beds and metrics.** 5 T2 beds plus berry/berry0 and closed-loop sets; metrics `fold_rate` and per-chart erank; Z2/Q homology (Lutz triangulations); frozen-cover Mapper.
- **PHASE-SPHERE layer** and arm-agnostic readouts: Chern prefix-free; TO7 scored per class.
- **Harness and sweep.** Harness with checkpoint/resume; single-GPU tiered sweep.

## Known risks

Each risk is stated with the default already chosen.
- **Hopf control may own C1 on S²** (Kutner's prior: 50%). If it does, C1 retreats to compute and readout claims, and the talking model is unaffected.
- **Launch-bound step times.** If row 11 fails, row 10 trains slower; Kaggle absorbs it.
- **Dataset download needs Seal's yes.** Default request: TinyStories (CDLA-Sharing), the smallest corpus known to give fluent small models [K].

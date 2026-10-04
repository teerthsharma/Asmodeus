2026-10-04 — PREREG v1, committed before any result file exists (`zero/runs/` empty at assembly, checked by Wilson).

# PREREG — Asmodeous ० (OBSERVER-ATLAS)

## Sources and change rules
- Assembled by Wilson from three inputs:
  - `PROMPT.md` §8–§9 (rev. 2, 2026-10-04). Bars and kills are quoted verbatim.
  - `PREREG_DRAFT_facts.md`, cited as **F§n**.
  - Every `PREREG_DRAFT_*.md` present at assembly: `PREREG_DRAFT_beds.md` (Foreman, Appendix A), `PREREG_DRAFT_arms.md` (Cameron, Appendix B), `PREREG_DRAFT_t3_t7.md` (Taub, Appendix C), `PREREG_DRAFT_t4b.md` (Thirteen, Appendix D), `PREREG_DRAFT_t5_t6.md` (Kutner, Appendix E), `PREREG_DRAFT_t4a.md` (Thirteen, Appendix F), `PREREG_DRAFT_atlas.md` (Chase, Appendix G), `PREREG_DRAFT_armQ.md` (Kutner, Appendix H), `PREREG_DRAFT_hybrid.md` (Kutner, Appendix J; **withdrawn**, provenance only), `PREREG_DRAFT_phase_sphere.md` (Foreman, Appendix K), `PREREG_DRAFT_event.md` (Taub, Appendix L), and `ORIGIN.md` (Foreman, Appendix I). Appendices are verbatim copies; each header carries the draft's sha256 prefix at copy time.
- Inline **PENDING** marks name an open cell of §6. Such a cell is NOT RUNNABLE until a pre-result amendment closes it; all other tests are runnable at v1.
- Every departure from PROMPT text is marked **AMENDED: <fact, source>**. Nothing else changes.
- Computable facts in §0 (Betti of `ref/triangulations.json` over Z2 and Q, Betti-only vertex minima, nerve-dimension bounds, two-annulus covers of T² and Klein) are bound by `tests/test_facts.py` (68 tests, independent implementation `ref/betti_check.py`, cross-checked against `metrics.betti_z2`).
- Code references point at files that exist at assembly: `metrics.py`, `atlas/observer_atlas.py`, `arms/*.py`, `beds/*.py`.

## 0. Facts the amendments rest on (from F)

1. **Atlas counts in PROMPT §3 are ball-chart counts B(M).** Rudyak–Schlenk math/0605350 eq. (2): cl+1 ≤ cat ≤ B(M) ≤ m+1, with unnormalised cat [F§1.1]. B(M) = 2 for S², 3 for T² and Klein, 4 for SO(3) ≅ RP³ [F§1.2].
2. **Counts with general charts are smaller.** Here a chart is any open set homeomorphic to an open subset of Rⁿ, which is what E_k: o → R^{d_c} can learn. Per F§1.3: S² 2, T² 2, Klein 2, SO(3) 2 [F§1.3]. SO(3): Gómez-Larrañaga, González-Acuña, Hoste, *Math. Proc. Camb. Phil. Soc.* 109 (1991) 105–115 — a closed 3-manifold needs exactly 2 such charts iff the Bockstein βw₁ = 0 (always true for orientable M³). T² and Klein rest on explicit two-annulus constructions [F§1.3, F§1.5]; no surface citation was located.
3. **Top-m nerves cannot carry higher Betti numbers.** A nerve built from top-m co-selection has dimension ≤ m−1, so H_k = 0 for k ≥ m. With m ∈ {1,2}, β₂ = β₃ = 0 on every input [F§5]. `atlas/observer_atlas.py::nerve` docstring states the same.
4. **Threshold bound.** With partition-of-unity weights, at most ⌊1/τ⌋ charts have w ≥ τ. So β₂ needs τ ≤ 1/3 and β₃ needs τ ≤ 1/4. At τ → 0⁺ a softmax nerve is the full simplex, which is contractible [F§5].
5. **Minimum vertex counts for a nerve.**
   - Betti-only (exact Z2 Betti): S² 4, T² and Klein 5, RP³ 6 [F§3a].
   - Homotopy-correct (covering type ct): S² 4, T² 7, Klein 8, RP³ 10 or 11 [F§2].
6. **Z2 Betti numbers do not separate T² from Klein.** T² is (1,2,1) over both Z2 and Q. Klein is (1,2,1) over Z2 and (1,1,0) over Q [F§1.4].
7. **MoP-JEPA's 0.85 is planAll (graph-search) on pointmaze-medium-stitch, not closed-loop success.** Closed-loop execution appears only in v1 Table 4 (MoP .67/.33 vs MDN .87/.27, med/large). v3 calls its own 3-seed execution check "inconclusive" [F§6].

## 1. Laws
PROMPT §1 laws apply unchanged: L-GIVEN, L-PREREG, L-BR, L-WALD, L-LEARN, L-OWNER, Gap rule, L-REPRO/L-SHA, L-AUDIT, Security.
- **AMENDED: (compute location; coordinator, resolves Taub's rule conflict).**
  - "Kaggle may TRAIN checkpoints (fp16, Wilson's checkpoint format chained across sessions); every score, bar and verdict is computed on the local 4060 from those checkpoints."
  - This refines PROMPT §1 Security ("deciding numbers from the local RTX 4060, Kaggle reproduces").
  - Checkpoint format: `run.save_state` / `run.load_state`, as agreed with Foreman. **PENDING:** fp16 and cross-session chaining on Kaggle are not yet part of that format or its tests.
- **Ground rule (Seal; top of `zero/CLAUDE.md`):** Asmodeous (०) is a new artificial intelligence. OBSERVER-ATLAS, PHASE-SPHERE, the single Hopf-bundle chart and every other piece are **parts** of Asmodeous, never the whole.
- **Every claim is about which parts Asmodeous keeps.** A part that passes its claim is kept and built on; a part that fails is replaced. No part is bound in advance.
- Candidate parts:
  - OBSERVER-ATLAS (`PROMPT.md`);
  - PHASE-SPHERE (`ORIGIN.md`);
  - Hopf single-chart (Kutner);
  - ASMODEOUS assembly (slot-by-slot design, Kutner + Thirteen);
  - further parts as drafted.
- Every claim (C1–C5, TO1–TO7, T-EV) is scored for every candidate part. The verdict per claim names the part(s) that pass and are therefore kept.
- Owned methods (JEPA, MoE, attention, ...) are baselines, not constraints.
- This supersedes the earlier "design follows the origin" and "three first-class candidates" lines.

## 2. Tests, bars, kills

### T0 reproduce
> run identities.py, check SHA pin.

Pin (PROMPT §1): SHA-256 `467c2ae25f92121093a60435c3397bd5f1e9fe246a2f602b4fc03bbace40a3fe`; outputs must match `ref/identities.out` (numpy 2.4.4).

**AMENDED: (a) T0 tolerance.**
- Fact, from Foreman's finding in `house-events.jsonl` (Inspector audit "clean"):
  - "numpy 2.4.4 venv reproduces the same single-cell diff as 2.4.6 (Luders pure max|QQ| 5.00e-16 vs 4.44e-16). Cause is OpenBLAS DYNAMIC_ARCH kernel dispatch, not numpy version".
  - "No kernel on this machine reproduces all 36 lines."
- Rule:
  - SHA-256 pin of `ref/identities.py` must match exactly.
  - Rounding-level cells of `ref/identities.out` match within |Δ| ≤ 1e-15.
  - Every other cell matches exactly.
- Replaces PROMPT L-REPRO's byte match for T0 only.

### T1 owner hunt — DONE (Wilson)
> Full combination owned → measurement only. Part owned → cite, continue.

Result (`OWNERS.md`): **full combination not located.** Partial owners per delta are cited there. T1 does not switch the project to measurement-only.

### T2 geometry (decisive; C1, C2)

PROMPT §8, verbatim:
> Manifolds: S² (spherical pendulum); T² (double-pendulum angles, MuJoCo); Klein bottle (synthetic); SO(3) (free rigid body, MuJoCo); neutral R² (one chart suffices). Observations: 64×64 renders or random nonlinear lift to D=256; manifold, charts, state never given. Arms at equal active FLOPs and total params: single JEPA d ∈ {n, 2n, 2n+2}; MoP-JEPA (K heads, shared encoder); chart-AE-style multi-chart encoder; OBSERVER-ATLAS K ∈ {1,2,3,4,8,16,64}, m ∈ {1,2}, d_c = n. Metrics: latent pred error h=1,10; linear-probe state recovery from g; discontinuity rate (fraction of state pairs within geodesic ε whose glued latents differ by > δ; ε, δ preregistered); nerve Betti over Z2; effective rank per chart; FLOPs/step.
> Predictions: P-COUNT discontinuity ≈ 0 first at K = 2 (S²), 3 (T²), 3 (Klein), 4 (SO(3)). P-BETTI Z2: S² (1,0,1); T² (1,2,1); Klein (1,2,1); SO(3)≅RP³ (1,1,1,1). Neutral: K=1 suffices, no gain.
> Bars: discontinuity ≤ 0.01 at K = atlas(M) on ≥3/4 manifolds over 3 seeds; Betti correct on ≥3/4; pred error ≤ best owned at equal active FLOPs; neutral error within +2% of single JEPA.
> Kills: single JEPA d=2n+2 matches error and discontinuity AND Mapper on its latent recovers same Betti; or MoP-JEPA ties every metric; or Betti wrong on ≥2 manifolds.

**AMENDED: (P-COUNT): stated against both chart notions.** Fact 0.1, 0.2; Rudyak–Schlenk math/0605350 eq. (2); F§1.3.
- **P-COUNT-B (ball charts, PROMPT's numbers):** discontinuity ≈ 0 first at K = B(M) = 2 (S²), 3 (T²), 3 (Klein), 4 (SO(3)).
- **P-COUNT-G (general charts):** discontinuity ≈ 0 first at K = G(M) = 2 (S²), 2 (T²), 2 (Klein), 2 (SO(3)).
- **Reported:** K_first(M), the smallest K in the grid where the 3-seed mean discontinuity is ≤ 0.01, per manifold. The report states which of P-COUNT-B / P-COUNT-G it equals, or neither.
- **Discontinuity bar** unchanged, evaluated at K = B(M). B(M) ≥ G(M), so the bar is satisfiable under either notion.
- `atlas(M)` in the bar text means B(M).
- K grid {1,2,3,4,8,16,64} already contains every G(M) and B(M).

**AMENDED: (metric "nerve Betti over Z2" → Z2 and Q).** Fact 0.6; F§1.4; recomputed on `ref/triangulations.json`.
- Every nerve Betti vector is reported over **Z2 and Q**.
- P-BETTI targets:

  | M | Z2 | Q |
  |---|---|---|
  | S² | (1,0,1) | (1,0,1) |
  | T² | (1,2,1) | (1,2,1) |
  | Klein | (1,2,1) | (1,1,0) |
  | SO(3) | (1,1,1,1) | (1,0,0,1) |
  | R² | (1,0,0) | (1,0,0) |

- "Betti correct" means **both** vectors match.
- Code state at assembly: `metrics.py::betti_z2` and `nerve_betti_z2` are Z2 only, and `nerve_betti_z2` defaults to max_dim=2. A Q-rank path and max_dim=3 for SO(3) are needed before T2 scoring. `metrics.betti_z2` reproduces all six Z2 vectors of `ref/triangulations.json` (checked by Wilson).

**AMENDED: (P-BETTI evaluated on the soft-weight nerve).** Facts 0.3–0.5; F§3, F§5.
- **Nerve.** True nerve (`flag=False`) of the **pre-top-m** softmax weights over all K: `w_soft` from `ObserverAtlas.forward`. Built over held-out inputs of the bed. Gated so that closed-gate inputs add no simplices.
  - The top-m nerve (`w`) is reported as an ablation only. With m ≤ 2 it has β₂ = β₃ = 0 by construction.
  - **RESOLVED (coordinator, conflict 3):** the true pre-top-m nerve is scored. The clique-filled nerve (`flag=True`, Foreman) is reported as a labelled extra and never scored.
- **τ grid — RESOLVED (coordinator ruling, conflict 1).**
  - The full grid τ ∈ {0.05, 0.10, …, 0.50} (Foreman A§4) is **reported**.
  - P-BETTI is **scored only on τ ≤ 1/4**, i.e. {0.05, 0.10, 0.15, 0.20, 0.25}.
  - Fact basis (F§5; `tests/test_facts.py::test_pou_tau_nerve_dim_le_floor_inv_tau_minus_1`): β₂ ≠ 0 needs τ ≤ 1/3 and β₃ ≠ 0 needs τ ≤ 1/4.
- **Reported Betti vector — RESOLVED (coordinator, conflict 2).**
  - **Scored:** τ is frozen on the tuning seed at the midpoint of the longest constant-Betti run within τ ≤ 1/4 (Foreman's rule). The Betti vector at that τ on eval is the scored value.
  - **Secondary, never scored:** the most persistent vector over τ ≤ 1/4 on eval (`metrics.most_persistent`). The full τ sweep is also reported.
- **K at which P-BETTI is scored:**
  - **RESOLVED (coordinator, conflict 10):** the threshold is the good-cover size ct(M): S² 4, T² 7, Klein 8, RP³ taken as 11 (conservative; ct ∈ {10, 11}). Scored at K ∈ {16, 64}, which is ≥ the threshold for every manifold. No scored cell changes.
  - Also reported at K = 8: ≥ the Betti-only minimum for all four manifolds, ≥ ct for S², T² and Klein, and < ct for RP³.
  - Small K cannot carry the targets [F§3a]: K ≤ 3 cannot give S², K ≤ 4 cannot give T² or Klein, K ≤ 5 cannot give RP³. K ≤ 4 is reported but not scored.

**AMENDED: (C2 comparators: router nerves of owned arms).** Cameron's requirement; owned arms expose routing weights: `arms/mop_jepa.py` returns `w_soft = pi`, `arms/chart_ae.py` returns `w_soft = p`.
- **RESOLVED (coordinator, conflict 11):** TO1 is scored for every candidate. The real atlas has c₁ = 0 by construction (trivial bundle): it misses the Berry bar and passes the control. Foreman's PC-pairing probe is reported, no bar.
- **RESOLVED (coordinator, conflict 6):** all router nerves are scored at K ∈ {16, 64}. Cameron's native K (MoP-JEPA 8; chart-AE 2/3/4) is reported as a labelled extra and never scored.
- Under the same nerve, τ rule, Betti rule and K ∈ {16, 64}, the following are scored beside the atlas:
  - (i) Mapper on the best single-JEPA latent (`metrics.mapper_betti`);
  - (ii) the MoP-JEPA router nerve (π over K heads);
  - (iii) the chart-AE chart-prediction nerve (p over K charts).
  - (iv) the lazy chart-AE nerve (`arms/lazy_chart_ae.py`, `w_soft` = p over all K); added by Cameron (Appendix B §B.2).
- **Bar addition:** the atlas has Betti correct on ≥ 3/4 manifolds AND on strictly more manifolds than each of (i)–(iv).
- **Kill addition:** (ii), (iii) or (iv) is correct on as many manifolds as the atlas. This extends PROMPT's "Mapper on its latent recovers same Betti" clause to the three owned router nerves.

Beds, metrics, ε/δ calibration and data/optimiser settings: Appendix A (Foreman). Arm list per test, L-BR matched sizes, MoP/chart-AE deviations and tuning grid: Appendix B (Cameron). Owner proposals that change PROMPT bar text are listed in §6 (conflicts) and are not adopted here. OBSERVER-ATLAS design pins, the shared anti-collapse table (SIGReg λ = 0.9 for every arm, λ_bal = 0.3, overlap 0.1), the scale contract, and the topology-readout pins (nerve on gated `w_soft`, max_dim = 3, the same τ grid): Appendix G (Chase), each pin bound by a named test.

**RESOLVED (coordinator, conflict 4): amendments (b) and (c) confirmed** — tear AND fold; disc ≤ 0.2 × torn reference AND ≤ 0.01.

**AMENDED: (b) C1 scored on tear AND fold.** Fact (Foreman, Appendix A §A.5.1):
- A single JEPA is a continuous map of a continuous lift, so it cannot tear. At d = n it fails by folding.
- `metrics.discontinuity_rate` scores a d = n projection at 0.0000, so PROMPT's kill clause "single JEPA matches discontinuity" fires trivially.
- C1 is therefore scored on both:
  - discontinuity (tear): `metrics.discontinuity_rate`, eps = 0.08·diam, delta = 0.5;
  - fold_rate (non-injectivity): `metrics.fold_rate`, far = 0.30·diam, fold_delta = 0.05.
- Kill text becomes: "single JEPA d=2n+2 matches error AND discontinuity AND fold_rate, AND Mapper on its latent recovers same Betti". The other kill clauses are unchanged.
- Fold bar: fold ≤ 0.01 (Foreman A§3).

**AMENDED: (c) Discontinuity bar relative to the torn-chart reference.** Fact (Foreman A§3): a fully torn one-seam chart scores only 0.018–0.066 depending on eps. At eps = 0.08·diam the torn reference is s2 0.0466, t2 0.0453, klein 0.0565, so3 0.0356, r2 0.0481. An absolute 0.01 alone therefore has no fixed meaning.
- **Deciding rule:** disc ≤ 0.2 × torn-reference(bed, eps) **AND** disc ≤ 0.01. Both are reported.
- At eps = 0.08·diam the relative limit is 0.0093 / 0.0091 / 0.0113 / 0.0071 / 0.0096 (s2/t2/klein/so3/r2).
  - Where it is below 0.01 (s2, t2, so3, r2) the relative clause decides.
  - On klein the absolute 0.01 decides.
- Foreman proposed 0.25×. The coordinator set 0.2×.

**RESOLVED (coordinator, conflict 8): per-chart effective rank** is computed through the optional `z` [B,T,m,d_c] / `zidx` [B,T,m] keys, already present in `common.py` (line 16).

**AMENDED: (d) Prediction error primary = relative to copy-last.** `npersist` = MSE / MSE(copy-last) from `metrics.pred_error_from`. Fact (Foreman A§2): a variance-normalised error is beaten by a trajectory-id latent (nvar = 0, npersist = ∞). `nvar` is reported as secondary. Applies to "pred error ≤ best owned" in the T2 bar and to the T2 kill.

**AMENDED: (e) Linear probe beside its PCA(obs) control.** Fact (Foreman A§3): PCA(obs), with no training, reaches probe R² 0.998 / 0.992 / 0.973 / 0.895 / 0.973 (s2/t2/klein/so3/r2). The probe is reported only with this control beside it, split by trajectory.

**AMENDED: (f) Fairness of D_g.**
- **RESOLVED (coordinator ruling, conflict 9): D_g = 2n+2 per bed** — 6 for s2, t2, klein and r2; 8 for so3.
- A single JEPA at d = D_g joins the single-JEPA arms d ∈ {n, 2n, 2n+2}.
- Reason (PROMPT §3 Whitney caveat): a single JEPA with d ≥ 2n embeds any n-manifold, so a glued space wider than the single-JEPA comparators would be an unequal budget.
- Cameron's matched-size table (Appendix B §B.3) is re-run at D_g = 2n+2 (07:58; r2, s2, t2, klein, so3 rows). Chase's draft now uses D_g = 2n+2 = 6 on S² (Appendix G); `sweep_t2.py:130` pins D_g = 2n+2.

**AMENDED: (g) Physics and actions.**
- Bed dynamics are first-order (overdamped), x_{t+1} = Exp_x(dt·(drift + a_t)). Fact (Foreman A§1, A§5.5): with second-order physics a single frame is not Markov.
- The Klein bed has zero actions. Fact (Foreman A§1, A§5.6): it is non-orientable, so no global action frame exists, and a (u,v)-frame action leaks the sheet.
- Replaces PROMPT's "S² (spherical pendulum)", "free rigid body" and "double-pendulum" dynamics wording for T2. The T5 swing-up bed (Appendix E) stays second-order.

### T3 lazy scaling (C3)
> m=2; K ∈ {16,64,256,1024}, product keys from K=256. Arms: dense JEPA equal active FLOPs; single JEPA with PEER experts. Bars: loss monotone in K, beats dense from K=256; lazy top-m within 1% rel of soft mixture over all K. Kill: flat in K, or PEER-in-one-JEPA matches.

Operational pins: Appendix C §C.2 (Taub). Taub's draft notes it merges only after Inspector review → **PENDING Inspector sign-off**.

**AMENDED: (T3 kill scope) — RESOLVED (coordinator, conflict 7), via the gap rule (novelty only if owned parts fail).** The C3 kill also fires if **PEER-JEPA OR lazy chart-AE OR lazy MoP-JEPA** matches the atlas at equal active FLOPs and equal total params (Cameron, Appendix B §B.2). PROMPT's "PEER-in-one-JEPA matches" is the first of these three.

### T4 collapse (C4)
> (a) SIGReg 0.1× and 0×, single vs atlas, track effective rank of z_k and g; kill: atlas collapses as fast. (b) OGBench pointmaze-stitch with MoP-JEPA protocol; bar success ≥ MoP-JEPA; kill below.

**AMENDED: (T4b: the metric must be named).** Fact 0.7; F§6. MoP-JEPA arXiv 2607.05238 reports three different "success" quantities. Each T4b comparison names its quantity and the MoP version it compares against:
- planAll: v1 Table 2 = v3 S10(a)
- realroute: v1 Table 3 = v3 S10(b)
- closed-loop execution: v1 Table 4 only

The MoP-JEPA "0.85" is planAll on medium-stitch.
- Protocol facts are in F§6.4 (datasets, coordinate features, 20% transitions held out, 5/3 seeds; v3: K=8, 30k updates, 512-d latents).
- Unknown: encoder size, batch size, learning rate; no public code located.
- Thirteen's draft `PREREG_DRAFT_t4b.md` is present and reproduced verbatim in Appendix D. Its status line reads "DRAFT, not preregistered".
- **AMENDED: (T4b bar: realroute + precision as primary metric).** Facts:
  - 0.85 is planAll on the model's own proposed graph [F§6.1].
  - planAll and realroute both score 1.0 for a propose-every-edge control. This follows from the v3 S10 definitions [F§6.1–6.2]; Thirteen measured it (Appendix D §D.6, controls).
  - **RESOLVED (coordinator, conflict 5)**, adopting Appendix D §D.6:
    - Primary: atlas realroute ≥ MoP-JEPA realroute, with precision ≥ MoP precision − 0.05, on ≥ 2/3 mazes over 5 paired split seeds.
    - Kill: below on ≥ 2/3 mazes.
    - Secondary: planAll, reported for **every arm including the atlas**. Closed-loop execution is reported beside it.
    - The primary edge rule is the centred rule (next amendment).
  - Mandatory controls: memoriser, propose-all, codebook, shuffled context. Protocol G and Protocol E are as in Appendix D §D.3–D.4.
  - This replaces PROMPT's undefined "success" only by naming the metric. The direction of the bar ("≥ MoP-JEPA; kill below") is unchanged.
- **AMENDED: (T4b edge rule; decided by coordinator).**
  - **Primary:** for every arm, subtract the train-set mean embedding (computed per arm), then propose c→c′ when cosine > .5. The rule is identical for all arms.
  - **Secondary:** the paper's raw rule, cos(head_k(c), z(c′)) > .5 on uncentred latents (Appendix D §D.3).
  - Fact (Thirteen, 2-step smoke, reported to coordinator; Appendix D §D.7 item 10 states the dependence on latent offset): the raw rule proposed 246/287/580 of 650 medium edges for un-normalised arms vs 0 for MoP. planAll tracked latent offset, not learning.
- **L-LEARN gate for the MoP arm:** Appendix D §D.5, all four conditions on medium-stitch.
- **PENDING Thirteen, open before the bar can be trusted.**
  - Appendix D §D.7.2 reports that a train-edge memoriser, under Thirteen's reconstruction, scores realroute 0.321/0.237/0.419 (med/large/teleport). That is above the paper's MoP .247/.139/.192.
  - Thirteen states that the reconstruction-vs-paper discrepancy "must be settled before any bar is trusted".

T4a arm list: Appendix B §B.2 (Cameron): single JEPA, MoP-JEPA, chart-AE and lazy chart-AE at SIGReg λ × {1, 0.1, 0}. T4a pins: Appendix F (Thirteen, `PREREG_DRAFT_t4a.md`).
- Primary pair: single JEPA (d = D_g) vs atlas (K=8, m=2), with D_g = 2n+2. This is consistent with amendment (f).
- Beds: s2, so3, and r2 (neutral). SIGReg scale s ∈ {1, 0.1, 0}; the primary reading is ratio-preserving, λ = 0.9 / 0.4737 / 0.
- Collapse time T = first step with g_erank < 1.5. Bar: PASS at s = 0 on both s2 and so3. Kill: KILL at s = 0 on either.
- Eval seeds 0, 1, 2, 4, 5; tuning seed 3 is never evaluated (`test_tuning_seed_never_evaluated`). Owner states 5 is the minimum for the bootstrap.
- PROMPT's kill wording "atlas collapses as fast" is operationalised as the CI lower bound of geometric-mean T_atlas/T_single ≤ 1.
- **Calibration done** (Appendix F, "Pre-commit calibration"). It is coordinator-approved, baseline only, on tuning seed 3, with output in the scratchpad, not `runs/`.
  - Single JEPA at s = 0 on s2 did **not** collapse in 5000 steps (censored). g_erank minimum 3.32, ending at 5.02; g_std grew 0.035 → 0.83.
- **AMENDED / RESOLVED (coordinator, conflict 14):**
  - **VOID rule:** a (bed, s) cell where single is censored on ≥ 3/5 eval seeds scores VOID ("baseline does not collapse"), never PASS. The continuous min g_erank/D_g paired difference is reported, with no bar.
  - **Primary decisive cell:** SIGReg 0 AND no EMA target (online stop-grad target), on s2 and so3, identical across arms. PASS and KILL as in Appendix F §F.4 on this cell.
  - **Secondary cell:** EMA (τ = 0.996) + SIGReg 0.
  - Fact basis: single with EMA at s = 0 was censored on s2 (calibration above).
  - **PENDING Thirteen:** single-only recalibration of the primary cell on tuning seed 3.

### T5 time traveller (C5)
> MuJoCo double-pendulum swing-up (T²×R²) and OGBench pointmaze; CEM in glued space with §4.7 re-scoring; arms V-JEPA-2-AC-style single-JEPA MPC, MoP-JEPA, equal planning compute. Bar ≥ best owned + 0.05; kill at or below.

Operational pins: Appendix E (Kutner): swing-up bed, success = tip ≥ 0.75 for ≥ 20 consecutive control steps, 150 paired episodes per arm, equal planning MACs with all re-scoring charged, shared tuning grid on seed 1, L-LEARN gate (probe R² ≥ 0.9, h=10 error ≤ 2× h=1).
- Additions beyond PROMPT text, from the owner draft (not fact-driven amendments):
  - C5 also requires TT > (atlas, M = m) at the same budget.
  - Bed-level VOID conditions.
  - A pre-arm rule that switches the shared energy to running latent L1 if terminal-L1 with a perfect model scores ≤ floor + 0.05.
- PROMPT's bar and kill are unchanged.
- **Calibration filled** (Appendix E; bed only, no arm), swing-up:
  - zero torque 0/50 and uniform random 0/50, so the floor is 0;
  - oracle CEM, warm, running-height cost: 20/20;
  - oracle CEM, cold, height cost: 10/10;
  - oracle CEM, cold, terminal-L1 energy (V-JEPA-2-AC form) with a perfect model: 3/10.
- **Calibration decision (owner, Appendix E, taken before any arm trains).**
  - The terminal-L1 energy is the binding limit: 3/10 vs 10/10 for the running height cost at the same `cem_plan` sampling.
  - The shared T5 energy for every arm is the **running latent energy** E(a) = mean_h ‖g_h − g_goal‖₁.
  - The terminal-L1 V-JEPA-2-AC protocol is reported beside it, with no bar.
- **T5 second bed: OGBench pointmaze** (medium- and large-stitch, official `evaluate()`), Appendix E.
  - Same energy, equal-compute rule and tuning grid. Closed-loop success only; planAll stays T4b.
  - VOID if the true-graph executor ceiling is < 0.8.

### T6 optional (contextuality)
> query pairs both orders, order effect + QQ with bootstrap CIs. Measurement only.

Reference (PROMPT §7): Lüders max|QQ| 4.44e−16; classical+disturbance mean|QQ| 0.0751. Pins: Appendix E §E.T6 (Kutner), statistic validated in `tests/test_t6.py` against `ref/identities.py` §2.

### Candidate parts and scoring matrix (ground rule: parts of Asmodeous; a passing part is kept)

| Candidate | Definition | Owner | State |
|---|---|---|---|
| OBSERVER-ATLAS | PROMPT §4; pins in Appendix G | Chase | draft present (Appendix G) |
| PHASE-SPHERE | ORIGIN.md §3 layer as built (`atlas/phase_sphere.py`, present since 07:58, with `readouts.py`, `tests/test_phase_sphere.py` and `tests/test_readouts.py`): ψ ∈ C^N, Born decider with null gauge, lazy gauge-fixed charts, Cayley-unitary local dynamics, CTC fixed-point loop plus amortised head. Pins and **per-candidate readout matrix**: Appendix K | Foreman | draft present (Appendix K); open items K§5 |
| ASMODEOUS | One architecture, slot-by-slot best feature of every source (replaces the HYBRID subclass merge, withdrawn by Seal). `atlas/hybrid.py` is **outside the scoring matrix** | Kutner + Thirteen | **PENDING** (after `IDEAS_kutner.md` / `IDEAS_thirteen.md`) |
| Hopf single chart | A single Hopf-bundle chart as a candidate part. It is not only a C1 attack: if it passes, it is kept | Kutner | **PENDING Kutner** (no draft present) |

- **Scoring rule.** Every claim is scored for every candidate under the same beds, seeds, equal active FLOPs and params (L-BR), and owned baselines:
  - C1 geometry; C2 topology; C3 lazy scale; C4 collapse (T4a, T4b); C5 time traveller;
  - TO1 Chern; TO2 CTC fixed point; TO3 halt before start; TO4 butterfly; TO5 scattering front end; TO6 ray lattice; TO7 holonomy.
- The verdict line per claim names the candidate part(s) that pass, which are then kept in Asmodeous, or NONE.
- **Where a claim is undefined for a candidate**, it is scored N/A with the reason stated, never as a pass or fail. For example, nerve Betti is undefined for a candidate without a router, and Chern is undefined for a real-only candidate without phases.
- **Per-candidate readout matrix: Appendix K §K.4 (Foreman).** Where its cell wording differs from rulings 2, 3, 6 and 10, the rulings govern. Example: for C2 the scored nerve is the true pre-top-m soft nerve at the tuning-seed-frozen τ; the `w` co-selection and flag variants are labelled extras.

### Origin-derived claims (ORIGIN.md, Foreman)
Source: Appendix I (`ORIGIN.md`, verbatim).

**Status.**
- Transcriptions are [T]. O1–O8 are Foreman's [R] readings.
- Every former [?] item is resolved by Foreman as a committed reading O10–O18 (Appendix I §I.5).
- **TO10–TO18 ADOPTED (coordinator ruling)** as written in Appendix I §I.5: TO10 layer ablation, TO11 observer feedback λ₂, TO13 vortex switching, TO14 bimodal branch bed, TO17 strand linking, TO18 scope ≤ 24 h tier 1. O15 maps to TO7; O16 adds no cell. **TO12 is merged into T-EV** (see T-EV).
- Fact notes (Wilson):
  - TO12 is merged into T-EV per the ruling; the merged bar is in the T-EV section.
  - TO14's bed is the MoP-JEPA multimodal problem; MoP-JEPA is an owned arm (Appendix B).
- **TO1–TO7 are ADOPTED (coordinator ruling; scored for all three candidates per the Laws line).** Their bars and kills are preregistered as written in Appendix I §I.4, with the edits below.
- Shared rules: 3 seeds plus tuning seed 3; equal active FLOPs to the owned arms, with observer-loop iterations counted.

**Edits to the adopted tests**
- **TO7 (coordinator edit).** The clause "This is the test Z2 nerve Betti cannot pass, since both manifolds read (1,2,1)" is **struck**.
  - Fact: rational Betti numbers already separate T² (1,2,1) from Klein (1,1,0) (`tests/test_facts.py::test_z2_does_not_separate_torus_from_klein_but_q_does`).
  - Holonomy stays as its own readout. Bar and kill are unchanged.
- **TO7 scoring pin (owner, Appendix K §K.4; fact-driven tightening).**
  - The bar is evaluated per loop class: `to7_score.min` ≥ 0.95 (every class ≥ 95% correct), with abstentions counted as wrong, on 3/3 seeds.
  - Fact (Foreman, untrained smoke): a constant +1 scored 0.667 overall, and abstaining on exactly the Klein-a loops scored 1.0 under the earlier abstention-ignoring accuracy.
  - The ≥ 95% threshold is unchanged; only the aggregation is fixed. **Flagged for coordinator confirmation.**
- **TO5.** Keeps its L-GIVEN note: a fixed cascade is hand-given structure, so a win scores "representable only" unless the cascade's filters are learned.
- **AMENDED: (TO1 sign of c₁).**
  - Fact: c₁'s sign depends on two conventions, the Berry-connection sign and the surface orientation. `tests/test_facts.py::test_hopf_bundle_chern_number_is_one` gives c₁ = −1 for the spin-½ state under A = i⟨ψ|dψ⟩ with outward orientation, i.e. |c₁| = 1. This agrees with Wikipedia "Tautological bundle": "Its Chern class is −H".
  - The TO1 bar therefore reads **|c₁| = 1 on the Berry bed and c₁ = 0 on the control, on 3/3 seeds, with |c₁ − round(c₁)| < 0.1**. A learned model may realise either orientation. The magnitude is unchanged.
- **TO1 bed (fact note, no change).** The interference channel is cos(γ_t). Its value is the same for γ = +Ω/2 and γ = −Ω/2 (cos is even), so the sign convention of γ does not affect the bed.

**Fact checks, bound by tests** (`ref/berry_check.py`, `tests/test_facts.py`, 78/78)
- **Spin-½ Berry phase.** γ = −½ × enclosed solid angle, under:
  - γ = i∮⟨ψ|dψ⟩, discretised as −arg ∏⟨ψ_j|ψ_{j+1}⟩;
  - loop counter-clockwise seen from outside;
  - ψ = spin-up along n.
  - Tests: 5 caps to 1e-3 (`test_spin_half_berry_phase_is_minus_half_solid_angle`); the triangle Bargmann phase equals −Ω/2 exactly to 1e-9 on 200 random triangles.
  - [V] Wikipedia "Berry connection and curvature": "The Berry phase ... is just half the solid angle subtended by the path" (magnitude only; sign is convention). [K] Berry 1984, Proc. R. Soc. A 392:45, γ_m = −mΩ; not fetched this session.
- **Hopf bundle / spin-½ monopole.** The lattice Chern number is exactly −1 (|c₁| = 1) on outward icospheres at subdivision levels 2, 3 and 4, by the Fukui–Hatsugai–Suzuki construction (arXiv cond-mat/0503172, title verified). Controls: spin-1 gives −2; a constant state gives 0.
  - [V] Wikipedia "Berry connection and curvature": "Integrating the Berry curvature over the complete sphere yields exactly 2π ... the Chern number is unity".
  - [V] Wikipedia "Hopf fibration": "the Hopf fibration is equivalent to the fiber bundle structure of the Dirac monopole".
- **Mutation check.** Mutants are caught: the sign flip in `berry_phase` (9 tests fail), the dropped e^{iφ} phase (9 fail), and inward face orientation (2 fail).
- **[D] Born rule = softmax.** p_k = |α_k|²/Σ|α_j|² = softmax(2·log|α_k|) exactly. Arm Q (Appendix H) implements a Born decider over phased amplitudes.

### Arm Q — coherent (non-IIA) observer
Pins: Appendix H (Kutner, `PREREG_DRAFT_armQ.md`).
- **Code and tests.** `atlas/coherent_observer.py` (`CoherentAtlas`, subclass of `ObserverAtlas`); `tests/test_coherent_observer.py` (owner: 11/11).
- **Mechanism.** ψ_k = √softmax(s)_k·e^{iθ_k(q)}, p = |Uψ|², with U = Cayley(PQᵀ − QPᵀ) real orthogonal of rank 2r.
  - Measurement is top-m of p plus the gate.
  - P = 0 reproduces the exact softmax (init). The gradient of P is nonzero at P = 0, so coherence is learned, not given (L-GIVEN).
- **Spec change, recorded by the owner under the then-current folder rule 2 ("origin wins", since superseded by "no binding"; Arm Q stands as a candidate component).** PROMPT §7 dropped the Born/Lüders decider; Arm Q restores a Born-rule decider, with softmax as its decoherence limit. Source: origin notes "small phases all around the sphere but the decider will be a softmax".
- **Arms.** Softmax atlas; CoherentAtlas r ∈ {2, 4}; ablations P ≡ 0 and θ ≡ 0. L-BR equal active FLOPs and params via `arms/match.py`.
- **Bar** (all required, on ≥ 3/4 curved manifolds, 3 seeds):
  - duplicate-chart rate ≤ 0.5 × softmax;
  - Betti correct on ≥ as many manifolds as softmax;
  - discontinuity and h=10 error ≤ softmax + 2%;
  - r2 within +2%;
  - the P ≡ 0 ablation does not reproduce the gain.
- **Kill** (any one):
  - Arm Q ≤ softmax on the T2 metrics;
  - ‖P‖/‖Q‖ < 0.01 on ≥ 3/4 manifolds;
  - the θ ≡ 0 ablation ties full Arm Q, which kills the phase claim.
- **T6 link** is measurement only.
- Prior (owner guess): ~15%.
- **AMENDED-consistency note (Wilson; fact, not a change).** Arm Q's "Betti correct" inherits this PREREG's Z2+Q, soft-nerve and K ∈ {16, 64} amendments (§T2). The duplicate-chart metric uses the same δ as T2 discontinuity, which is now relative to the torn reference (amendment c). **PENDING Kutner:** confirm δ.

### T-EV event-driven laziness in time ("observed → exists")
Pins: Appendix L (Taub, `PREREG_DRAFT_event.md`; origin `ref/origin/AETHER_qGPU_event_driven.pdf`).
- **Mechanism.** `EventDriven(arm)` wraps any candidate or owned arm. A step triggers when Δ_t = ‖o_t − o_τ‖₂/√D ≥ ε, with ε = softplus(ρ) learned; untriggered frames do not exist for the arm (sample-and-hold).
- **Quality metric** (anti-gaming): npersist_ev is scored against the same weights' always-compute latents. ΔQ = npersist_ev(1)/npersist_full(1) − 1.
- **Bar:** on ≥ 2/3 beds (s2, so3, swing-up), trigger fraction ≤ 0.5 with ΔQ ≤ 0.01 (3-seed mean; worst seed for the FLOP ratio).
  - Learned ε must match or beat the best fixed-ε quantile on the (FLOPs, ΔQ) frontier.
  - r2 must hold ΔQ ≤ 0.01.
- **Kill:** no ε reaches the bar on ≥ 2 beds, or the skip-RNN owned baseline reaches the same frontier point.
- **Systems bar:** wall µs/step ≥ 1.5× lower than always-compute at the measured trigger rate, at B = 1 and B = 64, on ≥ 2 beds.
- **AMENDED: TO12 merged into T-EV (coordinator ruling: one test, the stricter bar of each pair wins).**
  - Quality: ΔQ ≤ 0.01 (T-EV). This is stricter than TO12's "npersist within 2%".
  - Compute: trigger fraction ≤ 0.5 (both).
  - Added from TO12: executed steps correlate with |Δstate|, Spearman ρ ≥ 0.5 (the "∝ tez" claim).
  - Beds: s2, so3 and swing-up, ≥ 2/3 (T-EV's set, which contains TO12's overdamped beds).
  - Kill: union of both kills. No ε reaches the bar on ≥ 2 beds; OR the skip-RNN owned baseline reaches the same frontier point; OR ρ < 0.2.
- **Status:** adopted with TO12 merged (coordinator).
  - Adopted by the coordinator (TO12 merged; see above).
  - **PENDING Inspector:** the draft says "merges only after Inspector review".
  - **L-OWNER done (Wilson; `OWNERS.md` §E).** No located work combines all of:
    - an observation-delta trigger against the last computed frame;
    - a learned ε (softplus, straight-through, compute penalty);
    - sample-and-hold of a world-model/latent-predictor wrapper.
  - Closest partial owners:
    - Skip RNN 1708.06834 (learned gate + budget + hold; gate from state, not Δ);
    - Delta Networks 1612.05571 (Δ ≥ fixed threshold vs last transmitted value);
    - CATS (WACV 2024; gradient-learned temporal thresholds + event penalty, inside CNNs);
    - Eventful Transformers 2308.13494 (hand-set token threshold);
    - DISK 2602.00440 and WorldCache 2603.06331 (training-free skip/reuse in diffusion world models, fixed thresholds).
  - Gap rule applies: a T-EV win is novel only if these owned mechanisms fail the same bar. Delta-network-style fixed ε is already in the draft's fixed-quantile baselines.

### PLAN item 5 — deciding bake-off: Quantum-Filter LM (QF-LM) talks and is new (plan of record `PLAN.md`, Dr House)
**Status.** This is the decisive experiment. Topology claims T2 and TO1–TO18 (and C1/C2) move to PLAN item 8: idle-time only, non-blocking, bars unchanged.

**Arms** (4060; ~10M params; equal FLOPs (L-BR); 3 seeds):
- QF-LM (Kutner, PLAN item 3 primitives);
- GPT and GRU baselines (Cameron);
- ablations: QF-LM + input embedding (write ≠ read); softmax head; no-think; no-null.

**Bed:** TinyStories, BPE |V| = 4096, document-disjoint splits, data hash pinned (PLAN item 2).

**Credit rule:** a slot is credited only if its ablation loses.

**"Talks" bar (PLAN.md):**
- val perplexity ≤ 1.05 × GPT, AND
- QF-LM ties GPT on the blind 50-prompt rubric. **PENDING:** rubric definition, rater, and tie criterion (PLAN item 6).

**"New" bar: all three required (PLAN.md):**
- **(a) write = read.** QF-LM + embed does not beat QF-LM by > 2% perplexity on ≥ 2/3 seeds. Otherwise write = read is dead (an RNN with a Born head).
- **(b) think tracks surprise.** Think steps correlate with surprisal (Spearman ≥ 0.3), AND adaptive beats fixed steps at equal mean FLOPs.
- **(c) flat memory.** Memory is flat in context length.
- **PENDING owner pins:** exact measurement of "flat" in (c), and the comparison metric in (b).

**Kill:** "talks" fails, or (a) fails. "New" is dead if any of (a)–(c) fails.

**Fact (bound, Wilson; `tests/test_facts.py::test_instrument_max_mass_bounded_by_dimension`).**
- For any complete instrument on C^N, Σ_v max_ψ ‖K_vψ‖² = Σ_v σ_max(K_v)² ≤ Σ_v tr(K_v†K_v) = N, with equality iff every K_v has rank 1.
- So at most 2N outcomes can ever reach p ≥ ½. N = 16 gives ≤ 32 of 4096 tokens. This motivates N ≥ 2048 in PLAN.md.
- The diag(√M)·Q head of PLAN item 3 is complete and obeys the bound (`test_plan_diag_q_head_bound_and_talk_threshold`).

**Additional talk bars, stated by the coordinator from Taub's G1 (TinyDialogues, BLiMP) and not in PLAN.md:**
- TinyStories val loss ≤ 1.05 × transformer;
- TinyDialogues perplexity ≤ 1.10 × transformer;
- BLiMP ≥ transformer;
- samples reported, with no bar.
- **PENDING coordinator:** whether these are retained beside PLAN's talk bars.

**Compute location:** Kaggle may train (item 6, 25–30M params); every score is computed on the 4060 (compute-location amendment above).

**QF-LM owner hunt:** `OWNERS.md` §F. The full combination was not located. The closest owner of "measure to remember" is Hidden Quantum Markov Models (1002.2337, 1710.09016).

### T7 systems
> lazy routing + tiny JEPAs on 4060, µs/step vs dense JEPA of equal quality. Bar ≥1.5× faster.

Operational pins: Appendix C §C.3 (Taub). Appendix C §C.4 holds Taub's routing/chart-compute cost measurements, which the draft labels "cost measurements, not T7 results" (bench run under contended CPU, unlocked clocks).

## 3. Kills and priors (PROMPT §9, verbatim)
> T1 finds full combo → measurement only. T2 killed → geometry claim dead; T3–T5 run only if their claims stand without C1, C2. Hand-given structure → "representable only".

| Test | Prior (author, PROMPT §9) |
|---|---|
| T2 | ~30% |
| T3 | ~25% |
| T4a | ~20% |
| T4b | ~35% |
| T5 | ~20% |
| Combination beats owned arms on a neutral bed (PROMPT header) | ~15–20% (guess) |

## 4. Order
PROMPT §10: T0 → T1 → T2 (decides) → T3 ∥ T4 → T5 → T7; T6 optional anytime. T1 is complete (§2).

## 5. Conflicts between drafts and amendments (all 14 RESOLVED by coordinator, 2026-10-04)

Column 3 records the ruling; column 4 keeps the owner-draft position for provenance.

| # | Topic | Ruling | Owner draft (provenance) |
|---|---|---|---|
| 1 | τ grid | **RESOLVED (coordinator):** full grid {0.05…0.50} reported; P-BETTI scored only on τ ≤ 1/4 | Foreman A§4: 0.05, 0.10, …, 0.50 |
| 2 | τ / Betti selection rule | **RESOLVED (coordinator):** Foreman's tuning-seed freeze is scored; most-persistent-on-eval is reported as secondary, never scored | Foreman A§4: τ frozen on the tuning seed at the middle of the longest constant-Betti run |
| 3 | Nerve variant | **RESOLVED (coordinator):** true pre-top-m nerve scored; clique-filled nerve is a labelled extra, never scored | Foreman A§2: `flag=True` clique complex reported, "the only way top-2 routing can show β₂"; frozen separately |
| 4 | Discontinuity bar | **RESOLVED (coordinator):** confirmed: tear AND fold; disc ≤ 0.2× torn reference AND ≤ 0.01 | Foreman A§3 proposed 0.25 × torn reference |
| 5 | T4b primary metric | **RESOLVED (coordinator):** Thirteen: realroute + precision floor primary (centred edge rule); planAll secondary for every arm including the atlas; closed-loop reported beside | Thirteen D§6: realroute primary, planAll secondary. Cameron B§2: atlas "scored on planAll too, with closed-loop reported beside it" |
| 6 | Owned-nerve K for C2 | **RESOLVED (coordinator):** all router nerves scored at K ∈ {16, 64}; native K (MoP 8, chart-AE 2/3/4) is a labelled extra | Cameron B§1–2: MoP-JEPA at K = 8, chart-AE at K = atlas(M) = B(M) (2/3/4). Fact: a nerve on ≤ 8 vertices cannot be homotopy-equivalent to RP³ (ct ≥ 10, F§2). On ≤ 4 vertices no complex has the T², Klein or RP³ Z2 Betti vector (F§3a; `tests/test_facts.py::test_min_vertex_counts_are_minimal_exhaustive`) |
| 7 | T3 kill scope | **RESOLVED (coordinator):** Cameron's extension adopted (gap rule): C3 kill if PEER-JEPA OR lazy chart-AE OR lazy MoP matches the atlas | Cameron B§2 extends it to "any of PEER-JEPA, lazy chart-AE, or lazy MoP matches" |
| 8 | Per-chart effective rank | **RESOLVED (coordinator):** optional `z`/`zidx` key adopted (already in `common.py`) | Foreman A§5.9: `common.py` exposes no per-chart z; proposes an optional `z` key |
| 9 | Atlas D_g | **RESOLVED (coordinator):** D_g = 2n+2 per bed (6 for s2/t2/klein/r2, 8 for so3); single JEPA at d = D_g in the sweep | Chase G§3 and Cameron B§3 used D_g = 8; their configs follow the ruling (PENDING re-size) |
| 10 | K threshold for scoring P-BETTI | **RESOLVED (coordinator):** good-cover sizes; RP³ taken as 11; no scored cell changes | Chase G§4: "minimal triangulation size" S² ≥ 4, T² ≥ 7, Klein ≥ 8, RP³ ≥ 11. Both give K ∈ {16, 64} in practice. For RP³, ct is 10 or 11 while the vertex-minimal triangulation is 11 (F§2) |
| 11 | TO1 Chern for the real-valued atlas | **RESOLVED (coordinator):** Kutner. Chern is computed on the real atlas as defined (trivial bundle, c₁ = 0) and scored, so the atlas misses TO1's Berry bar and passes its control. Foreman's substitute PC-pairing probe is reported beside it, no bar | Kutner J: c₁ = 0 by construction, so "a miss for the atlas, not a VOID". Foreman K§4: N/A ("no U(1) phase and no Chern class"); substitute PC-pairing \|c₁\| probe reported, no bar. *Fact:* a trivial bundle has c₁ = 0, which is defined (`tests/test_facts.py` trivial control). **OPEN — coordinator** |
| 12 | HYBRID C2 readout and claim coverage | **WITHDRAWN with HYBRID (Seal). Ruling text kept for provenance: PREREG rules apply unchanged to the third candidate; claims missing from its owner table are scored from Foreman's matrix** | Kutner J table: "nerve Betti over Z2". C4 (T4a/T4b) and TO3–TO6 are absent from the table. **OPEN — coordinator** (default: PREREG rules apply to the hybrid unchanged; missing claims are scored per the matrix) |
| 13 | TO2 / TO3 for HYBRID | **WITHDRAWN with HYBRID (Seal). Ruling text kept for provenance: loop claims (TO2/TO3) are scored on the third candidate's own loop; N/A only with a stated cost reason** | Foreman K§4: hybrid TO2, TO3 "N/A unless Kutner adds a loop"; TO6 N/A (coherent mix needs all K); C3 N/A beyond the K where O(K·r) exceeds the budget. **OPEN — coordinator** (default: score TO2/TO3 on the hybrid's TT loop) |
| 14 | T4a decisive cell when the baseline never collapses | **RESOLVED (coordinator):** Thirteen's VOID rule adopted. The PRIMARY decisive T4a cell is now SIGReg 0 AND no EMA target (online stop-grad target), identical across arms. The EMA + SIGReg-0 cell is secondary. Thirteen is recalibrating single-only on seed 3 | Thirteen F (post-calibration): VOID if single is censored on ≥ 3/5 seeds; continuous min g_erank/D_g difference reported with no bar; optional harsher stressor. *Fact:* single censored at s = 0 on s2, tuning seed 3, 5000 steps. **OPEN — coordinator** |

## 6. Open cells at v1 (each is NOT RUNNABLE until a pre-result amendment closes it)

Rules for this table:
- A cell listed here blocks only the tests that name it.
- Every test not named here is runnable at the v1 commit. That includes T2 tier 1 on the lift beds, per `sweep_t2.py`, which pins D_g = 2n+2 at line 130.
- An amendment that closes a cell must be committed before that test's first result file exists.
| Section | Owner | State at assembly |
|---|---|---|
| Beds: §6 render-injectivity probe ("pending" in draft) | Foreman | **NOT RUNNABLE until a pre-result amendment** |
| Arms: Inspector review (draft: "merges only after Inspector review"); single-JEPA d ∈ {n,2n,2n+2} re-sized at PREREG time | Cameron / Inspector | **NOT RUNNABLE until a pre-result amendment** |
| T4b memoriser-vs-paper discrepancy (Appendix D §D.7.2); antmaze-teleport out of first pass | Thirteen | **NOT RUNNABLE until a pre-result amendment** |
| T5 pointmaze calibration (random policy; true-graph executor ceiling) recorded before arms run | Kutner | **NOT RUNNABLE until a pre-result amendment** |
| T3/T7 Inspector sign-off | Inspector | **NOT RUNNABLE until a pre-result amendment** |
| L-GIVEN audit, prereg order | Inspector | **NOT RUNNABLE until a pre-result amendment** |
| PLAN item 5: rubric definition/tie criterion; (b)/(c) metric pins; QF-LM model + baselines code (items 3, 5) | Kutner / Cameron / Foreman | **NOT RUNNABLE until a pre-result amendment** |
| PLAN item 5: retain Taub's G1 extra talk bars (TinyDialogues, BLiMP)? | coordinator | **NOT RUNNABLE until a pre-result amendment** |
| Checkpoint format: fp16 + Kaggle cross-session chaining (compute-location amendment) | Foreman / Wilson | **NOT RUNNABLE until a pre-result amendment** |
| Hopf single-chart part draft (definition, matched sizes, which claims it is scored on) | Kutner | **NOT RUNNABLE until a pre-result amendment** |
| TO10–TO18 owner pins (beds, readout code; TO14 bimodal bed) | Foreman | **NOT RUNNABLE until a pre-result amendment** |
| TO1–TO7 owner pins (beds, solvers, readout code) | Foreman / Chase | **NOT RUNNABLE until a pre-result amendment** |
| ASMODEOUS candidate draft (slot-by-slot design; per-claim readouts per Appendix K §K.4) | Kutner + Thirteen | **NOT RUNNABLE until a pre-result amendment** |
| PHASE-SPHERE: N, d_c, loop_iters per bed confirmed by a K=64 memory smoke run on the 4060 (K§5.2); FLOP formula done (K§3) | Foreman | **NOT RUNNABLE until a pre-result amendment** |
| T-EV: Inspector review (adopted; TO12 merged; L-OWNER done) | Inspector | **NOT RUNNABLE until a pre-result amendment** |
| Arm Q: Inspector review (δ_dup = 0.05 × median latent distance pinned by owner, Appendix H) | Inspector | **NOT RUNNABLE until a pre-result amendment** |
| T4a: widths pinned via match.py; single-only recalibration of the primary cell (SIGReg 0, no EMA) on seed 3 | Thirteen | **NOT RUNNABLE until a pre-result amendment** |

## Appendix A — `PREREG_DRAFT_beds.md`, verbatim (title line dropped; sha256 95d08ca48d79)

Draft for `PREREG.md`. No arm has been trained or scored. The calibration below uses only arm-free latents: the embedded state, PCA of the observations, an atan2-type chart, a projection and a constant. Code: `beds/__init__.py`, `metrics.py`, `run.py`, `calibrate.py`. Tests: `tests/test_beds.py`, `tests/test_metrics.py`, `tests/test_topology.py`, `tests/test_run.py`.

### A.1. Beds

`make_bed(name, n_traj, T, obs, seed)` returns the Bed dict defined in `common.py`.

- **state**: the embedded coordinates, used for evaluation only.
  - S² ⊂ R³; T² ⊂ R⁴ as (cos θ₁, sin θ₁, cos θ₂, sin θ₂); Klein ⊂ R⁴ in the standard embedding with R = 1, r = 0.5; SO(3) ⊂ R⁹ as the rotation matrix; R² = [-1,1]².
  - SO(3) uses the rotation matrix, not a quaternion. A quaternion is a double cover of SO(3), so a linear probe and a geodesic computed from it would be sign-ambiguous.
- **obs = 'lift'**: `tanh(s W1 + b1) W2 + N(0, 0.02²)`, with W1 ∈ R^{S×64} and D = 256.
  - The lift weights are seeded by the bed name, not by the seed, so train, eval and tuning splits all see the same observation map (test `test_lift_is_shared_across_seeds`).
  - Over 2000 uniform pairs, the lifted/embedded distance ratio has min/median > 0.2, so the lift is well-conditioned (test `test_lift_is_well_conditioned`).
- **obs = 'render'**: MuJoCo 64×64 grayscale, available for t2 and so3 only. The so3 body is asymmetric: three arms of different length and shape. Lift is the primary observation.
- **Dynamics are first order (overdamped)**: x_{t+1} = Exp_x(dt·(drift + a_t)).
  - s2, klein and r2 are simulated in numpy.
  - t2 is a MuJoCo double pendulum: hinge damping 3, 25 substeps of 2 ms, torque σ = 10.
  - so3 is a MuJoCo ball joint: damping 0.6, gravity 0, body-frame torque σ = 2.4.
  - Check (test `test_first_order_dynamics`): corr(Δx_t, a_t) is 0.80 / 0.98 / 1.00 / 0.95 and |corr(Δx_t, a_{t-1})| is 0.03 / 0.08 / 0.07 / 0.02 for s2 / t2 / so3 / r2.
- **Klein has zero actions.** It is non-orientable and not parallelisable, so no action frame is defined globally. Its flow, (0.5 + 0.3 cos v, 0.5 sin v cos u) plus isotropic noise σ = 0.5, is invariant under the deck group.
- **Coverage** at the eval size (n_traj = 128, T = 64, seed 1000): covering radius / diameter and the fraction of transitions that cross the atan2 seam.

| bed | covering radius / diam | seam-crossing transitions | trajectories crossing a seam | median step |
|---|---|---|---|---|
| s2 | 0.044 | 0.035 | 0.51 | 0.179 |
| t2 | 0.045 | 0.024 | 0.43 | 0.178 |
| klein | 0.043 | 0.031 | 0.65 | 0.191 |
| so3 | 0.114 | 0.019 | 0.49 | 0.290 |
| r2 | 0.021 | 0.030 | 0.45 | 0.115 |

- **geo**:
  - S²: great-circle distance.
  - T²: flat torus distance with wrapping.
  - Klein: flat quotient metric, the minimum over (u,v) → (u + 2πk, (−1)^k v + 2πl).
  - SO(3): angle of R₁ᵀR₂.
  - R²: Euclidean distance.

### A.2. Metrics (`metrics.py`): each one fails a constant latent (tests)

- **pred_error**: reports `npersist` = MSE / MSE(copy-last) (primary) and `nvar` = MSE / Var(g) (spec).
  - A latent that encodes only the trajectory identity gets nvar = 0 but npersist = ∞.
  - A constant latent gets ∞ on both.
- **discontinuity_rate(eps, delta)**: pairs of points within geodesic distance eps, with latent distance divided by the median pairwise latent distance. Constant latent → 1.0.
- **fold_rate(far, fold_delta)** (new): the fraction of points that have a latent neighbour within fold_delta·median while lying more than `far` away on the manifold.
- **linear_probe**: ridge R², trained on 80% of trajectories and tested on the held-out trajectories.
- **effective_rank**: the exponential of the entropy of the normalised singular values.
- **betti_z2 / betti_q**: GF(2) and rational ranks of the boundary matrices.
  - d∘d = 0 is checked.
  - Ground truth comes from Wilson's Lutz triangulations (`ref/triangulations.json`): S² tet/oct, T² 7, RP² 6, Klein 8, RP³ Walkup 11.
  - Also checked: a 4×4 and a 3×5 Klein grid, ∂Δ⁴, ΣRP², and sd(∂16-cell)/±.
  - Q separates T² from Klein: Klein is (1,1,0) over Q.
- **nerve_betti_z2(w, tau, flag)**: builds the nerve from co-selection.
  - `flag=True` is the clique complex of the co-selection graph. It is the only way top-2 routing can show β₂ at all.
- **Mapper**: filter = PCA to min(d, 2n+2) dimensions, a ball cover of one radius, single linkage, the same Z2/Q code.
  - The cover (n_cover, overlap) is tuned label-free on the tuning seed, then frozen.
  - It is evaluated with n_samples equal to the nerve sample count, which is all eval frames.

### A.3. Calibration: eps / delta (calibrate.py, eval bed seed 1000, 4000 of 8192 points)

**Proposed:** delta = 0.5; eps = 0.08 × diameter; far = 0.30 × diameter; fold_delta = 0.05.

| bed | eps | oracle disc | PCA(obs) disc | torn disc | far | oracle fold | PCA fold | fold-latent fold | PCA(obs) probe R² |
|---|---|---|---|---|---|---|---|---|---|
| s2 | 0.251 | 0.0000 | 0.0000 | **0.0466** | 0.942 | 0 | 0 | **0.444** | 0.998 |
| t2 | 0.355 | 0.0000 | 0.0000 | **0.0453** | 1.333 | 0 | 0 | **0.850** | 0.992 |
| klein | 0.355 | 0.0000 | 0.0000 | **0.0565** | 1.333 | 0 | 0 | **0.537** | 0.973 |
| so3 | 0.251 | 0.0000 | 0.0000 | **0.0356** | 0.942 | 0 | 0 | **0.962** | 0.895 |
| r2 | 0.226 | 0.0000 | 0.0000 | **0.0481** | 0.849 | 0 | 0 | **0.547** | 0.973 |

How the settings were chosen:
- The fold latents are: s2 (x,y); t2 (cos θ₁, cos θ₂); klein (x₁,x₂); so3 the first row of R; r2 (x,|y|).
- At delta = 0.25 the noise floor rises at eps ≥ 0.08·diam (PCA up to 0.083 on r2, oracle up to 0.074 on klein). At delta = 0.5 both stay at 0 up to eps = 0.10·diam.
- The torn rate grows linearly with eps: on s2 it is 0.018, 0.031, 0.047 and 0.066 at eps = 0.04, 0.06, 0.08 and 0.10·diam. Eps = 0.08·diam puts one-seam charts at 3.6–5.7× the 0.01 bar.

**Proposed bar wording:**
- disc ≤ 0.01 **and** disc ≤ 0.25 × the torn-reference rate of the same bed at the same eps.
- fold ≤ 0.01.
- probe R² is reported against the PCA(obs) control in the table above, never on its own.

### A.4. Other numbers

- **tau grid**: 0.05, 0.10, …, 0.50.
  - It contains τ ≤ 1/3, which β₂ needs, and τ ≤ 1/4, which β₃ needs (Wilson facts §5).
  - τ is frozen on the tuning seed at the middle of the longest constant-Betti run. The flag variant is frozen separately.
- **Data**: train n_traj = 512, T = 64, window 32. Eval n_traj = 128, T = 64. Readout-tuning bed n_traj = 128, T = 64.
- **Bed seeds**: train data seed s, eval 1000 + s, readout tuning 2000 + 3.
- **Optimisation**: batch 64, AdamW (lr 1e-3, wd 1e-2), 20,000 steps.
  - lr is chosen from {3e-4, 1e-3, 3e-3} on tuning seed 3 only.
- **Arm seeds**: 0, 1, 2; tuning seed 3, never evaluated (`evaluate` asserts this).
- **Records**: every run record carries `mapper.{n_samples, cover, tune_seed}`, `nerve_n_samples`, `seeds`.

### A.5. Spec flaws these beds and metrics expose (for House / Inspector)

1. **Continuity.** A single JEPA is an MLP of a continuous lift, so it is continuous on M by construction. At d = n it fails by folding (Borsuk–Ulam), not by tearing.
   - Discontinuity scores a d = n projection at 0.0000. The kill clause "single JEPA matches discontinuity" therefore fires trivially.
   - C1 needs fold_rate. Only routed arms (hard top-m switching) can tear.
2. **The 0.01 bar has no meaning on its own.** A fully torn chart scores only 0.018–0.066 depending on eps, so the bar must be tied to the torn reference.
3. **Nerve dimension.** A top-m nerve has dimension ≤ m−1. A minimal atlas is not a good cover.
   - P-COUNT (K = atlas) and P-BETTI cannot both hold at the same K. A nerve needs K ≥ 4, 7, 8 and 10–11 for S², T², Klein and RP³ (Wilson facts §3).
   - Z2 cannot separate T² from Klein, so betti_q is reported as well.
4. **Variance-normalised prediction error is beaten by a latent that does not move.** Use npersist.
5. **Second-order physics** makes one frame non-Markov: rollout(obs0) has no velocity. The beds are overdamped to remove this.
6. **Klein actions** cannot exist globally. Any (u,v)-frame action leaks the sheet of the cover.
7. **Lift seed.** A lift seeded by the data seed would give the eval split a different observation function. The lift is keyed by bed name.
8. **Linear probe.**
   - PCA(obs), which needs no training, reaches R² 0.90–1.00, so the probe proves nothing without that control.
   - A frame-level split leaks neighbouring frames; the split is by trajectory.
9. **Per-chart effective rank.** `common.py` exposes no per-chart z, so the per-chart effective rank that T2 and T4a call for cannot be computed through the contract. Proposal: add an optional `z` key.
10. **Mapper must be frozen on the tuning seed.**
    - A sweep on the eval latent hid a bug: a link radius set from the MST omits the largest gap, which broke cycles on fresh samples.
    - A 2-D PCA filter on S² never recovers β₂, so the filter dimension must be ≥ n+1.
11. **Rendering.**
    - A symmetric so3 body would render SO(3)/G.
    - Grayscale orthographic rendering can lose the depth sign.
    - The lift stays primary; render injectivity is in §6.

### A.6. Render injectivity probe

Setup: raw pixels used as the latent, 16 trajectories × 24 frames, seed 7.

- **No folds found.** fold_rate (far = 1.4, fold_delta = 0.05/0.1/0.2) is 0/0/0 for both so3 and t2. The sample is small, so this is weak evidence of injectivity.
- **Pixel L2 is poorly conditioned.**
  - discontinuity_rate (eps = 0.25, delta = 0.5) is 0.83 for so3 and 0.92 for t2.
  - The cause: thin links stop overlapping after a small motion, so pixel distance saturates near the median almost immediately.
  - Render beds therefore test pixel-to-manifold learning, not chart topology. Keep lift as the primary bed, and do not run disc/fold on raw pixels.

## Appendix B — `PREREG_DRAFT_arms.md`, verbatim (title line dropped; sha256 87b7c10973ad)

Draft by Cameron, 2026-10-04. Not committed. Merges into `PREREG.md` only after Inspector review. Every
number below is a sizing computation (`arms/match.py`, meta device, no training), never a result.

### B.1. Arms

| Arm | File | Router w / w_soft | Lazy | Role |
|---|---|---|---|---|
| Single JEPA, d ∈ {n, 2n, 2n+2} (and d = D_g = 8 for matching) | `arms/single_jepa.py` | none | n/a | Whitney baseline (§3), T2 kill arm at d = 2n+2 |
| MoP-JEPA, K = 8 | `arms/mop_jepa.py` | π (softmax over K) / same | no: all K heads every step | closest owner (C4, T4b) |
| MoP-JEPA lazy, K = 8 | `arms/mop_jepa.py` (`lazy=True`) | π / same | inference runs the router-argmax head only | params ≫ FLOPs owner |
| Chart-AE JEPA, K = atlas K | `arms/chart_ae.py` | p (membership over all K) / same | no | owner of separate charts + gluing (1912.10094) |
| Lazy chart-AE JEPA, K, m = atlas K, m | `arms/lazy_chart_ae.py` | top-m renormalised p / p over all K | yes, top-m charts only | **observer ablation**: atlas minus backbone, gate, keys |
| PEER-JEPA | `arms/peer_jepa.py` | none | product-key top-k experts | owned "params grow, compute fixed" route (T3 kill) |
| CEM planner | `arms/planner.py` | n/a | n/a | V-JEPA-2-AC-style MPC over any arm's `rollout` (T5) |

Every arm has SIGReg (`zero/sigreg.py`, shared with the atlas) and an EMA target (τ = 0.996), takes only
`obs, act`, and follows the counting conventions in `arms/__init__.py`. `active_flops` is the MACs per step
of `observe`, and `n_params` excludes EMA copies. The atlas uses the same conventions.

### B.2. Arm list per test

- **T2 / C1 (geometry):**
  - single JEPA at d ∈ {n, 2n, 2n+2}
  - MoP-JEPA K = 8
  - chart-AE JEPA at K = atlas(M)
  - lazy chart-AE JEPA at the atlas's K, m
  - Each arm is run at the matched sizes in §3: one row when both targets are hit, two rows (flop, param) when they are not.
- **T2 / C2 (topology):**
  - `metrics.nerve_betti_z2` on `w_soft` of MoP-JEPA, chart-AE and lazy chart-AE (tested in `tests/test_arms.py::test_nerve_betti_runs_on_router_weights`).
  - `metrics.mapper_betti` on the single JEPA d = 2n+2 latent.
  - The atlas nerve has to beat every one of these, not only Mapper. The same τ sweep (`metrics.sweep_tau`) applies to all of them.
  - Router nerves are scored at K ∈ {16, 64}; each arm's native K (MoP 8, chart-AE atlas(M)) is reported as a labelled extra row (ruling 2026-10-04).
- **T3 (lazy scaling):**
  - dense single JEPA, flop-matched
  - PEER-JEPA, both matched; n_experts is set by `match()`, not fixed to K
  - lazy chart-AE at the same K, both matched
  - MoP lazy, both matched
  - The T3 kill extends to "any of PEER-JEPA, lazy chart-AE, or lazy MoP matches" (adopted, ruling 2026-10-04).
- **T4a (collapse):**
  - single JEPA, MoP-JEPA, chart-AE and lazy chart-AE at SIGReg λ × {1, 0.1, 0}
  - MoP at λ = 0 is the paper-native arm.
  - Each is tracked by `metrics.effective_rank` of z (per chart) and of g.
- **T4b (OGBench pointmaze-stitch):**
  - MoP-JEPA via `arms.mop_jepa.MoPGraph` (fit / embed / predict, the `arms/controls_t4b.py` interface) on `ogbench_bed.cell_transitions`; edges from `ogbench_bed.propose_edges`, scores from `ogbench_bed.graph_metrics`.
  - Primary: realroute + precision (centred edges). Secondary: planAll, for every arm (ruling 2026-10-04). The paper's 0.85 headline is planAll.
  - Also reported: active heads per context (π > 0.5/K) and precision at cos > 0.8. On a 4-context, 2-successor toy, K = 8 collapsed its router to one head per context and still recovered 8/8 edges at cos > 0.5, so the 0.5 threshold cannot tell mode splitting from a head sitting between two successors.
  - Closed loop (`ogbench_bed.evaluate` with `ogbench_bed.executor`) is reported beside the graph scores.
- **T5 (planning):**
  - `cem_plan` over single JEPA, MoP-JEPA (router-argmax rollout) and lazy chart-AE.
  - Planning compute is equal across arms: n_samples × iters × horizon × rollout MACs, with the sample count set per arm to equalise.
  - MoP over all K modes: `cem_plan(..., modes=True)` scores each action sequence by the π-weighted L1 energy over `MoPJEPA.rollout_modes`, which branches on every head at step 1 and then follows the router-argmax head. `arms.planner.mpc_policy` wraps any arm for `beds.swingup.evaluate`.

### B.3. Matched sizes against the atlas (D = 256 lift, D_g = 2n+2 per bed (6 for n=2, 8 for so3), d_h = 64, d_key = 32, hidden = 32, m = min(2, K))

Tolerance ±5% on each axis. **both** = one config hits both targets. **flop** / **param** = the arm cannot
hold the atlas's params/FLOPs ratio, so two baselines run: equal compute with fewer params, and equal params
with more compute. The atlas must beat each one on the axis it is matched on.

- Dense arms have params ≈ FLOPs.
- Top-1-of-K arms have params/FLOPs < K.
- A "both" match whose smallest width is below 32 is degenerate and is replaced by the split. MoP on t2 hit both targets only with a 7-wide encoder feeding 575-wide heads.

| Bed (n, A) | Atlas K | Atlas F / P | Single (d = D_g) | MoP K=8 | MoP lazy | Chart-AE | Lazy chart-AE | PEER |
|---|---|---|---|---|---|---|---|---|
| r2 (2, 2) | 1 | 72,556 / 72,779 | both h134 | both h139/139 | flop h163 (P 90.3k); param h137 (F 57.6k) | both h159/159 | both h159/159 | flop h98 (P 456k); param h15 (F 18.1k) |
| s2 (2, 3) | 2 | 81,624 / 81,685 | both h144 | both h151/151 | flop h177 (P 102k); param h149 (F 64.6k) | both h166/166 | both h166/166 | flop h111 (P 517k); param h17 (F 19.4k) |
| t2 (2, 2) | 3 | 81,592 / 90,463 | flop h144; param h154 (F 89.9k) | both h64/569 | both h174/109 | flop h157 (P 79.5k); param h173 (F 92.4k) | both h149/340 | both h118, N=64 |
| klein (2, 2) | 3 | 81,592 / 90,463 | flop h144; param h154 (F 89.9k) | both h64/569 | both h174/109 | flop h157 (P 79.5k); param h173 (F 92.4k) | both h149/340 | both h118, N=64 |
| t2 (2, 2) | 16 | 82,008 / 205,409 | flop h145 (P 82.7k); param h258 (F 204k) | flop h154 (P 84k); param h299 (F 202k) | both h161/1132 | flop h95 (P 74.5k); param h225 (F 223k) | both h138/354 | both h118, N=289 |
| t2 (2, 2) | 64 | 83,544 / 629,825 | flop h147 (P 84.4k); param h496 (F 629k) | flop h156 (P 85.4k); param h624 (F 624k) | flop h181 (P 103k); param h624 (F 563k) | flop h37 (P 73.7k); param h284 (F 720k) | both h109/352 | both h118, N=1225 |
| so3 (3, 3) | 4 | 82,272 / 100,217 | flop h144 (P 82.8k); param h163 (F 99.3k) | flop h145 (P 84.2k); param h167 (F 98.6k) | both h174/139 | flop h141 (P 78.5k); param h171 (F 105k) | both h149/257 | both h118, N=81 |
| so3 (3, 3) | 16 | 82,656 / 209,057 | flop h145 (P 83.7k); param h259 (F 207k) | flop h145 (P 84.2k); param h290 (F 206k) | both h161/868 | flop h80 (P 73.4k); param h201 (F 233k) | both h138/269 | both h118, N=289 |
| so3 (3, 3) | 64 | 84,192 / 644,417 | flop h146 (P 84.5k); param h500 (F 642k) | flop h147 (P 85.6k); param h617 (F 639k) | both h40/3943 | flop h28 (P 72.2k); param h234 (F 762k) | both h118/266 | both h118, N=1225 |

How to read the table:
- h a/b means encoder hidden / head-or-chart hidden.
- The single JEPA row is at d = D_g. The d ∈ {n, 2n, 2n+2} rows are re-sized by `match("single", ..., d=…)` at PREREG time.
- To regenerate: `python <scratchpad>/match_table.py`, or call `arms.match.match(arm, flops=atlas.active_flops(), params=atlas.n_params(), D=256, A=A, ...)`.
- 75% of the atlas's F is its GRU backbone (Taub's flag in `PREREG_DRAFT_t3_t7.md`). Every owned arm spends that budget on encoder width instead.

### B.4. MoP-JEPA deviations from arXiv 2607.05238v1

1. **SIGReg is added** on the pre-normalisation encoder output (PROMPT §3: anti-collapse is mandatory). The paper has no anti-collapse term; λ = 0 restores it, and T4a runs that arm.
2. **λ_route, λ_bal, encoder width/depth, batch size and lr are unpublished.** The defaults here are 1.0 / 0.0 / MLP-2 GELU / Foreman's `run.py` / tuned (§5).
3. **The balance term Σ γ̄ log Kγ̄ uses the hard γ and so has zero gradient**, as the paper writes it. It is reported, not optimised.
4. **One ghat is read out.** The paper emits all K (u_k, π_k); the contract needs one ghat, so ghat = u at argmax π (the paper's router-top-1 readout). The full set is used by the T4b graph protocol (`MoPGraph.predict`) and by T5 mode planning (`rollout_modes`).
5. **The encoder is an MLP on the D = 256 lift**, where the paper uses a GRU/attention encoder for AV2 and coordinate features for OGBench.
6. **Latent dimension is set by matching**, not the paper's 512. On T4b the paper's v3 setting (K = 8, latent 512, 30k updates) also runs as an unmatched reference row.
7. **Version.** v3 of the arXiv id is retitled Branch-JEPA. Cite v1. There is no public code.

Chart-AE deviations from 1912.10094:
- JEPA losses replace pixel reconstruction. The chart fidelity target is the EMA embedding sg(E_tgt(o)): a detached online target diverged on the toy bed.
- The Lipschitz penalty is skipped.
- Lazy chart-AE adds top-m routing and a Switch-style load balance (λ_bal = 0.01) so unselected charts keep a router gradient.

### B.5. Tuning (seed 3 only; same budget for every arm, the atlas included)

12 configs per arm, a grid sampled without replacement with seed 3, scored on held-out h = 1 `npersist`. Widths are fixed by §3 and never tuned.

| Knob | Values | Arms |
|---|---|---|
| lr (Adam) | 1e-3, 3e-3, 1e-2 | all |
| SIGReg λ | 0.02, 0.05, 0.1 | all |
| EMA τ | 0.99, 0.996 | all |
| λ_route | 0.1, 1.0 | MoP |
| λ_bal | 0, 0.01 (MoP, applied to soft π̄ when > 0); 0.01, 0.1 (lazy chart-AE) | MoP, lazy chart-AE |
| PEER (heads, topk) | (4, 8), (8, 4) | PEER |
| depth | 2, 3 | single, MoP encoder, chart-AE E |

The selected config is frozen and trained on seeds 0, 1, 2.

### B.6. Open for Inspector

- In train mode, the PEER BatchNorm query statistics pool over B·T tokens, so they mix time steps. Eval-mode causality is tested; train-time leakage is batch statistics only and carries no state.
- In chart-AE / lazy chart-AE rollout, membership is re-read from the predicted g. This is valid because the fidelity term places g in E's space, but it is untested on a bed.

## Appendix C — `PREREG_DRAFT_t3_t7.md`, verbatim (title line dropped; sha256 b993cd789cfa)

Draft by Taub, 2026-10-04. Not committed. Merges into `PREREG.md` only after Inspector review; until that
commit lands, every number below is a cost-model or timing measurement, never a T3/T7 result (L-PREREG).

### C.1. Bars, verbatim from PROMPT.md §8

- **T3 lazy scaling (C3):** m=2; K ∈ {16,64,256,1024}, product keys from K=256. Arms: dense JEPA equal
  active FLOPs; single JEPA with PEER experts. Bars: loss monotone in K, beats dense from K=256; lazy top-m
  within 1% rel of soft mixture over all K. Kill: flat in K, or PEER-in-one-JEPA matches.
- **T7 systems:** lazy routing + tiny JEPAs on 4060, µs/step vs dense JEPA of equal quality. Bar ≥1.5× faster.

The bars stay as written. Sections 2 and 3 only pin the words the bars leave open.

### C.2. T3 operational pins

| Word in the bar | Pinned meaning |
|---|---|
| bed | `so3` (n=3, atlas=4, D=256 random lift, `beds.make_bed("so3", obs="lift")`): the largest n and chart count in T2. L-WALD neutral bed beside it: `r2`. |
| loss | held-out h=1 `npersist` from `metrics.pred_error_from` (MSE / MSE of copy-last), computed in each arm's own g. It is comparable across arms because it is normalised by the arm's own copy-last error. A constant latent scores ∞, and so does a latent that encodes only the trajectory id (`tests/test_metrics.py`). Secondary: `metrics.linear_probe` R² of state from g (state is used at evaluation only). |
| seeds | train 0, 1, 2. Every arm, the atlas included, is tuned on seed 3 (L-BR). Steps, optimizer, batch and T come from Foreman's `run.py` and are identical for every arm. |
| atlas config | `ObserverAtlas(D=256, A=a, K, m=2, d_c=3, D_g=8, d_h=64, d_key=32, hidden=32)`, `product_keys = K>=256`. |
| dense JEPA equal active FLOPs | `SingleJEPA(d=D_g)` with `hidden` bisected by `bench_t7.match_flops` to within ±5% of `atlas.active_flops()` at that K. That gives hidden 144–146 at m=2 (table §4). |
| PEER-JEPA | `PEERJEPA(d=D_g, heads=4, topk=min(8,√K), d_key=32)`: `hidden` is bisected to equal active FLOPs and **n_experts is the smallest square whose total params ≥ atlas total params** (L-BR: equal active FLOPs AND equal total params). With n_experts = K, PEER has 4–21× fewer params than the atlas (§4), which is not a fair arm. |
| monotone in K | the 3-seed mean `npersist` drops at every step 16→64→256→1024, and each drop is larger than the pooled seed SD of the two Ks it connects. |
| flat in K (kill) | relative change of the mean `npersist` from K=16 to K=1024 is under 2%. |
| beats dense from K=256 | at K=256 and at K=1024, the atlas has lower `npersist` than dense on 3/3 seed-paired runs. |
| lazy within 1% of soft | the same config is trained with `observe(mode="soft")` (full softmax over all K; costs K charts per step, see §4). The bar holds when \|L_lazy − L_soft\| / L_soft ≤ 0.01 on the 3-seed mean at every K. |
| PEER matches (kill) | PEER-JEPA `npersist` ≤ 1.01 × atlas `npersist` (3-seed means) at K=1024. |

Open flag for Inspector: at this config the GRU backbone (3·d_h·(D+d_h) = 61,440 MACs) is 75% of atlas
active FLOPs, and K·d_key routing adds up to 131k more at K=4096 with flat keys. "Equal active FLOPs" is
therefore dominated by the backbone, not the charts. The draft keeps the contract's definition, which
counts the backbone, and reports the chart-only MACs (m · 9,088) beside it.

### C.3. T7 operational pins

**Dense JEPA of equal quality.**
- Bed and metric are T3's: `so3` lift, held-out h=1 `npersist`, 3 seeds, the same training budget.
- Width sweep: `SingleJEPA(d=D_g, hidden=H)` for H ∈ {64, 128, 256, 512, 1024, 2048, 4096}.
- H* is the smallest H whose 3-seed mean `npersist` ≤ the atlas reference (K=1024, m=2, product keys)
  AND whose `linear_probe` R² ≥ the atlas R² − 0.01.
- If no H ≤ 4096 qualifies, H* = 4096 and the ratio is reported as a lower bound.
- The `r2` neutral bed is run beside it and reported, but it does not decide the verdict.

**µs/step.**
- Measured with `bench_t7.py` at the PREREG commit, using the back-to-back CUDA-event protocol: 25 warmup
  steps, 200 timed, median + IQR.
- Each arm gets its best mode (min over eager and CUDA-graph replay). The atlas may use the parity-tested
  Triton chart kernel (`lazy_tri`).
- fp32, forward only. The step is one `observe` on obs [B, 1, 256].

**Batch sizes.**
- **Primary: B = 1.** One agent stepping is the regime that "computed only when observed" claims.
- B = 64 and B = 1024 are reported but do not decide the verdict.

**Bar.**
- The bar is met when median_dense(H*) / median_atlas ≥ 1.5 AND q1_dense > q3_atlas, at B = 1.
- The bench runs 3 times, non-concurrently with any other GPU lane, and the median of the 3 ratios
  decides.
- **Kill:** the ratio is < 1.5.

**Clocks.** The GPU is a laptop part and clocks cannot be locked without admin. The bench prints SM clock,
utilisation and temperature per batch size. A run where another process holds the GPU (`nvidia-smi
--query-compute-apps`) is void and rerun.

### C.4. Cost model vs measurement

**Provenance.**
- Hardware and software: `bench_t7.py` (uncommitted, branch atlas-zero), RTX 4060 Laptop 8 GB, torch
  2.14.0+cu126, triton-windows 3.7.1.
- Run conditions: one sweep on 2026-10-04 with CPU at 100% from other lanes. SM clock was 1455–2700 MHz.
- Recorded config: `D=256, A=2, d_c=2, D_g=8, d_h=64, d_key=32, hidden=32`.
- All cells below are graph-mode medians in µs (200 steps).
- These are cost measurements, not T7 results.

#### 4.1 Routing: O(K·d) flat keys vs O(√K·d) product keys

The model, in MACs per query (d_key = 32), not counting the shared query projection d_h·d_key = 2,048:
- flat: K·d_key MACs, plus top-m over K, reading K·d_key·4 B of keys
- product keys: √K·d_key MACs, plus 2 top-m over √K, plus top-m over m², reading √K·d_key·4 B of keys

| K | flat MACs | pk MACs | flat B=1 | pk B=1 | flat B=1024 | pk B=1024 |
|---|---|---|---|---|---|---|
| 16 | 512 | 128 | 61.6 | 73.7 | 61.2 | 96.3 |
| 64 | 2,048 | 256 | 68.1 | 68.6 | 60.4 | 89.1 |
| 256 | 8,192 | 512 | 58.6 | 70.8 | 90.1 | 69.6 |
| 1024 | 32,768 | 1,024 | 107.5 | 71.5 | 180.2 | 94.7 |
| 4096 | 131,072 | 2,048 | 58.7 | 73.7 | 589.3 | 106.5 |

**Reading.**
- Up to K=256 at B=1024, both paths sit on the ~60–75 µs graph-launch floor, so
  the O(√K) advantage cannot be seen there.
- Above it, flat keys grow with B·K (589 µs at K=4096, B=1024), while product keys stay at ≤ 107 µs:
  5.5× at K=4096.
- The PROMPT switch point (product keys from K=256) is consistent with this. Product keys cost about
  +10 µs of extra kernels at small K.
- The 107.5 µs for flat at K=1024, B=1 is a single-run outlier; the 3-run rule absorbs it.

#### 4.2 Chart compute: lazy top-m vs all-K vs dense at equal active FLOPs (m = 2)

| K | B | lazy (gather) | lazy_tri | soft (all K) | dense eq-FLOPs (H) | PEER eq-FLOPs |
|---|---|---|---|---|---|---|
| 16 | 1 | 89.1 | 60.5 | 74.1 | 59.4 (144) | 132.1 |
| 1024 | 1 | 84.0 | 81.9 | 448.5 | 59.4 (145) | 126.0 |
| 4096 | 1 | 90.1 | 79.9 | 1,843 | 59.4 (146) | 125.3 |
| 16 | 1024 | 916.5 | 335.9 | 477.2 | 109.6 | 597.0 |
| 1024 | 1024 | 1,156.6 | 428.0 | 24,090 | 118.8 | 873.5 |
| 4096 | 1024 | 1,144.3 | 518.1 | 107,924 | 130.0 | 1,276.9 |

Dense SingleJEPA at fixed width (graph mode, µs):

| B | H=128 | H=512 | H=1024 | H=2048 | H=4096 |
|---|---|---|---|---|---|
| 1 | 57.6 | 74.3 | 105.5 | 173.1 | 561.2 |
| 1024 | 104.4 | 361.5 | 969.7 | 3,231.7 | 11,840.5 |

**What this means for the T7 bar.**
- At equal active FLOPs the atlas never wins:
  - At B = 1 it ties dense within the launch floor, or is up to 1.38× slower (K=1024).
  - At B = 1024 it is 3.1–4.0× slower.
- So T7 can pass only through quality matching, that is, if the quality-matched dense width H* is large.
- At the atlas reference (K=1024, m=2, lazy_tri):
  - At B = 1 the bar of 1.5× needs H* between 1024 and 2048: H=1024 gives 105.5 / 81.9 = 1.29×, which is NO, and
    H=2048 gives 2.11×.
  - At B = 1024 it needs H* between 512 and 1024: H=512 gives 0.84×, H=1024 gives 969.7 / 428.0 = 2.27×.
- The same B=1 thresholds hold at K=4096, where lazy_tri is 79.9 µs.

#### 4.3 Gather cost and the faster path

- **Reference `StackedMLP.forward` is slow because of the gather.** It materialises the gathered weights
  `W[idx]` as an [N, m, i, h] tensor, which for E at B=1024, m=2 is 64 MB fp32. That tensor is written,
  then read again by an einsum.
- **The fused Triton kernel (`bench_t7._chart_mlp`) avoids that tensor.** It uses one program per
  (row, slot) and computes the whole chart MLP in one kernel: silu(x W1[e] + b1[e]) W2[e] + b2[e].
- **Measured gain:**
  - 2.2–2.7× at B = 1024 (916→336, 1,157→428, 1,144→518 µs).
  - 1.0–1.5× at B = 1.
  - Eager mode drops from ~6 ms to ~3 ms because the launch count falls.
- **Correctness:** parity with the gather path is within 1e-4 at all 5 tested shapes, and with the full
  atlas `observe` at K ∈ {16, 256 product-keys}. Removing the silu breaks 7/7 parity tests.
- **Remaining gap at B = 1024 against equal-FLOPs dense (~4×):** the kernel is a per-row GEMV with no
  tensor cores. Dense is a single SGEMM.
- **Next faster path:** sort (row, slot) by expert, then run a segment-grouped `tl.dot` over rows that
  share an expert, which is useful when B·m ≫ K. Reading bf16 weights would halve the
  ~34 KB-per-slot weight traffic.
- These are inference-only paths; training keeps the reference autograd path.

#### 4.4 Eager numbers (not used for any bar)

- Eager steps take 0.4–7 ms on this Windows box under the contended CPU load.
- In eager mode, lazy (6.0 ms) is ~5.5× slower than dense (1.1 ms) at every batch size, because the atlas
  launches more kernels per step and launch overhead dominates.
- CUDA-graph replay removes this. The bar therefore uses each arm's best mode, as pinned in §3.

#### 4.5 Design note (not a result): grouped bf16 kernel and backbone

- Measured with graph-mode medians, run conditions as in §4.1, m = 2.
- **Sort-by-expert grouped kernel (`bench_t7._grouped`, bf16 weights, fp32 accumulate, BM = 16).**
  - A single E call at K=4096, N=1024 runs 89 µs, against 211 µs for the per-row kernel and 782 µs for
    the gather path.
  - Inside a full `observe` step it gains only 1.0–1.13×. On narrow-input calls (P, φ, i ≤ 8) it loses to
    the per-row kernel, so `kind="auto"` uses grouped only for i ≥ 64.
  - Bound on bf16 error: 2e-2·max|ref|. Observed: 3.4–3.9e-3.
- **Backward pass.** `_Fused` recomputes the gather reference in fp32. Grads match the reference to 1e-5;
  for bf16 they are straight-through.
- **The backbone is the B = 1024 bottleneck: the cuDNN `nn.GRU` kernel, not the backbone's size.**
  - `nn.GRU` at T=1 takes 222 µs for d_h = 64, while `nn.GRUCell` with the same weights takes 64.5 µs.
  - Swapping in GRUCell (`lazy_cell`) brings the atlas to 127 / 194 / 221 µs at K = 16 / 1024 / 4096,
    from 337 / 381 / 466.
  - The gap to equal-FLOPs dense shrinks from 3.3–4.3× to 1.25–2.0×.
  - Shrinking d_h to 16 or 32 then changes nothing (185 / 187 / 194 µs at d_h = 16 / 32 / 64,
    K = 1024).
- **At B = 1 and B = 64 nothing changes.** Every arm stays on the 60–120 µs graph-replay floor.
- **Effect on the T7 threshold at B = 1024 (if Chase adopts GRUCell for streaming).** Against the earlier
  width sweep, H* ≥ 512 would suffice: 361.5 / 193.5 = 1.87×. Today it needs H* between 512 and 1024. B = 1
  still needs H* between 1024 and 2048.
- **TF32 asymmetry:** cuDNN runs the GRU in TF32 by default (7e-4 off from fp32), while the bench's
  matmuls are fp32. Pin `torch.backends.cudnn.allow_tf32` in PREREG.

#### 4.6 Design note (not a result): one-launch fused step, with fair baselines

- **The kernel.** `bench_t7.fused_observe` runs routing, all m charts and the mix in one Triton launch
  per T=1 step: query, gate, exact top-m (flat keys for K ≤ 256, otherwise product keys over the
  √K×√K sum), softmax, E, P, φ(z), φ(ẑ), and the gated mix. The GRU stays outside the kernel.
- **Parity.** It matches `observe(mode="lazy")` on 10/10 configs (idx exact, w within 1e-5, g/gh/gate
  within 1e-4, cuDNN TF32 off). Two deliberate breaks were each caught: swapping the weights failed 6/10,
  inverting the router failed 10/10.
- **Fairness.** Dense, PEER and the width sweep also get `torch.compile` (the `*_c` arms). It changes
  nothing for dense (B=1024: 106→105 µs). It helps PEER only at B=1 (123–136 → 100–125 µs). Dense has no
  backbone, so the GRUCell swap has no counterpart there.
- **Graph-mode medians, fused + GRUCell (`lazy_fused_cell`) vs dense at equal active FLOPs:**

| B | K=16 | K=1024 | K=4096 | dense eq-FLOPs (best of plain/compiled) |
|---|---|---|---|---|
| 1 | 60.5 | 60.1 | 62.5 | 59.4–61.4 |
| 64 | 67.6 | 63.5 | 57.9 | 71.7–77.8 |
| 1024 | 119.8 | 131.1 | 169.0 | 105.5–108.5 |

- **Reading.**
  - B=1: the atlas reaches the ~60 µs graph floor (ties dense).
  - B=64: the atlas beats dense by 1.1–1.3×.
  - B=1024: the atlas is 1.1–1.6× slower than dense. Before the fused kernel, GRUCell and bf16 it was
    3.3–4.3× slower.
  - Eager: the fused step takes 0.68–0.84 ms vs 1.0–1.2 ms for dense (fewer launches).
- **T7 consequence.**
  - The bar still rests on H*. In this run the width sweep at B=1 gives 104 µs at H=1024 and 164 µs at
    H=2048, so the atlas is 1.7× faster at H*=1024 and 2.6× faster at H*=2048.
  - At B=1024 it is 2.2× faster at H*=512 (289 vs 131 µs).
  - Run-to-run drift is about 30% at B=1024 (densew_1024: 970 → 696 µs across sweeps), so this is not
    decisive until the 3-run rule is applied.

#### 4.7 Runner pins (`run_t3.py`; decided by the T3 owner, for Inspector)

- **The soft all-K reference runs only at K ∈ {16, 64, 256}.**
  - At K = 1024 it peaks at ≈ 9 GiB on the 8 GB card, about 56 s/step with driver spill (so3, batch
    64×32, 2026-10-04).
  - The "lazy within 1% of soft" bar is therefore checked at K ≤ 256, and K = 1024 is reported as
    "not run (resource)".
  - Per-chart SIGReg in soft mode uses `sigreg_chunked`. Its value and gradient equal `zero/sigreg.py` to
    1e-12 (float64 test), with peak memory O(chunk), chunk = 16,384.
  - Soft K = 256 measured 2.26 GiB peak and ≤ 1.43 s/step. At 20,000 steps that is about 8 h per job,
    and the 12 soft K = 256 jobs (2 beds × 6) total about 95 GPU-hours, about 32 h at 3 concurrent.
  - Inspector to decide: fewer steps for soft, or accept the cost.
- **The scheduler launches only when all hold:**
  - own workers + other GPU compute processes < 3;
  - free RAM > 3 GB;
  - torch threads ≤ 4 per worker;
  - git tree clean (L-REPRO).

## Appendix D — `PREREG_DRAFT_t4b.md`, verbatim (title line dropped; sha256 5a8c0c038041)

Status: DRAFT, not preregistered. Nothing under `runs/` exists for T4b. Bed: `beds/ogbench_bed.py`. Controls: `arms/controls_t4b.py`. Adapters: `arms/t4b_adapters.py`. Runner: `run_t4b.py` (`--dry-run`). Tests in `tests/` (`test_ogbench_bed`, `test_controls_t4b`, `test_t4b_adapters`, `test_run_t4b`) run green in venv `~/.venvs/ogbench`.

### D.1. Source and pins
- Paper: **arXiv 2607.05238v1** (MoP-JEPA, 6 Jul 2026). Cite v1 only: v3 (3 Aug 2026) is retitled Branch-JEPA, reports different campaigns, and calls its own execution check "inconclusive". v2 Table 4 shows large-stitch realroute .139 (K=8, 30k); v1 Table 4 shows .208 for the same cell because it substitutes the matched K=16/60k-step run. Quote .139 for the K=8 protocol.
- No public code (v1–v3 PDFs, abs pages, web search). The hyperparameters below are reconstructed from text.
- ogbench **1.2.1** (PyPI 2026-01-14; repo `seohongpark/ogbench` master `1d4140997f60c52c6fb0702ec100dc988b18c548`). The changelog shows no pointmaze env, dataset or eval change after 1.0.1. The 1.2.0 change affects singletask envs only.
- venv `~/.venvs/ogbench`: mujoco 3.14.0, gymnasium 1.2.0, numpy 2.4.6, torch 2.14.0+cu126. Installing ogbench system-wide would raise the system mujoco from 3.11 to 3.14 and move the T2 beds.
- Dataset SHA-256 values are pinned in `ogbench_bed.SHA256`, and the loader refuses a mismatch. The upstream downloader crashes on Windows (WinError 32, rename of an open file), so the bed fetches the files itself.

### D.2. Beds
`pointmaze-medium-stitch-v0`, `pointmaze-large-stitch-v0`, `pointmaze-teleport-navigate-v0` (state, 2-D). The paper's 4th env, antmaze-teleport (29-D), is out of scope for the first pass.
- Stitch: 5000 episodes × 201 stored states (200 transitions). Navigate: 1000 × 1001. Val: 10% of that.
- obs = xy = state (`meta['obs_is_state']`). Here L-GIVEN reduces to "no cell id, no graph, no maze map reach any arm".
- Free space is a planar domain, so atlas = 1. betti_z2: medium (1,1), large (1,3), teleport (2,2), where the 2nd component is the isolated out-cell (1,7). T4b therefore tests branching and mode collapse only. The C1 geometry argument gives the atlas no edge here.

### D.3. Protocol G — graph (where 0.851 comes from)
- Nodes are free maze cells (unit 4). A transition is a pair of consecutive distinct cells in a dataset episode. Measured successors per cell: 2.08 / 2.09 / 2.20, against the paper's "2.1–2.2", so the reconstruction matches.
- Holdout: 20% of unique directed edges, `round(0.2|E|)`, never a source cell's last edge. The split seed is the run seed.
- Arms see coordinates only and are action-marginalised (v3 S10). The training pair is (source-cell centre xy → successor-cell centre xy) with held-out edges excluded (`cell_transitions(context='cell')`).
- Hyperparameters: K=8, 30k updates, latent 512.
- Edge proposal, PRIMARY (coordinator decision): centre first. Subtract the arm's mean latent over the training points (`embed(ctx ∪ nxt)`) from embeddings and heads, then propose c→c′ when cos > .5 (`ogbench_bed.decode_edges(rule='centred')`). The same rule applies to every arm and control. A test checks that a constant latent offset leaves the edges unchanged. SECONDARY, reported beside it: v1's raw rule, c→c′ when cos(head_k(c), z(c′)) > .5 (`rule='paper'`). Primary readout uses all heads (as for planAll). Secondary readout gates heads on π_k > 0.5/K.
- Metrics (`graph_metrics`), over 300 truly-reachable sampled queries (`sample_queries(seed)`) plus the 5 official task queries:
  - **planAll**: BFS path in the proposal.
  - **realroute**: path after keeping only real edges.
  - **precision**: macro per-source fraction of real edges.
- Seeds: 5.

### D.4. Protocol E — closed-loop execution
- Follows the official OGBench loop (`evaluate`). `reset(options=task_id)`, goal = `info['goal']`, actions clipped to [-1,1], 1000-step limit, terminate at goal, score = final-step `info['success']`.
- Volume: 5 tasks × 50 episodes per seed (the OGBench paper rule; the repo default is 20). Seeds: 5 (the paper used 3).
- Executor: `ogbench_bed.executor(name, edges, replan=True, patience=40)`. The paper's executor is unpublished, so this is a reconstruction. Each step it runs BFS over the proposed edges minus deleted ones and steers at full speed to the next cell centre. A teleport hop steers to its own centre. A hop not reached within 40 steps is deleted and the route is replanned. Mutation tests cover the replanning and the reset between episodes.

### D.5. L-LEARN gate (our MoP-JEPA must reproduce before any comparison)
On medium-stitch, 5 seeds, Protocol G, all four must hold:
- (a) planAll ≥ 0.75 (paper .851±.08, within 0.10).
- (b) realroute in [0.15, 0.35] (paper .247±.08).
- (c) precision in [0.32, 0.66] (paper .42–.56).
- (d) our dense K=1 arm planAll ≤ 0.15 (paper .084). If dense does not collapse, the bed is not reproducing the phenomenon and T4b is VOID, not a pass.

Teleport and large are reported, not gated: large is seed-bimodal (.386±.45). Execution is reported, not gated: paper .67/.33 med/large, likely 5 episodes per seed. 66.7±11.5 over 3 seeds equals {0.6, 0.6, 0.8}, which is an inference.

### D.6. T4b bar (proposal, both metrics as Wilson asked)
- **Primary: realroute, with precision ≥ MoP-JEPA precision − 0.05**, paired over the same 5 split seeds.
  - Bar: atlas realroute ≥ MoP-JEPA on ≥2/3 mazes.
  - Kill: below on ≥2/3 mazes.
  - Paired bootstrap 95% CI over seeds.
- Secondary: planAll, which is weak because hallucinated edges raise it (MDN case).
- Tertiary: execution.
- Mandatory controls, every run:
  - **train-edge memoriser** (`arms/controls_t4b.Memoriser`): proposes exactly the training edges.
  - **propose-all** (`ProposeAll`): realroute = planAll = 1.0, precision .086/.046/.050.
  - context-free codebook, latent space as in v1 (`t4b_adapters.CodebookGraph`, `codebook_latent`): MoP-JEPA with zeros fed to heads and router. A coordinate-space k-means version (`codebook_xy`) runs as a labelled extra.
  - shuffled-context test.
- An arm "works" only if realroute beats the memoriser and precision beats propose-all by ≥ 0.3.

### D.7. Hidden variables found
1. **0.851 is planAll, not success.** It is BFS over the model's own proposal and is inflated by fake edges. MoP precision is .42–.56, so about half of its edges do not exist.
2. **The memoriser beats the paper's verified numbers** (our reconstruction, 5 split seeds):
   - A lookup of training edges, with no generalisation, scores planAll = realroute = **0.321±0.075 / 0.237±0.028 / 0.419±0.131** (med/large/teleport).
   - Paper MoP realroute is .247 / .139 / .192. All three are below the memoriser.
   - Either our query/holdout reconstruction differs from theirs, or the paper's realroute gain is over a baseline weaker than a table lookup. This must be settled before any bar is trusted.
3. **realroute is freeloadable.** Proposing every edge scores 1.0, so precision must be reported beside it. The paper never reports realroute for the codebook.
4. **Context granularity decides whether branching exists.** With the cell-centre context there are about 2.1 successors per context. With the exact exit coordinate the successor is nearly determined and dense would not collapse. The paper says only "coordinate features".
5. **Unclipped actions tunnel through walls.** `PointEnv.step` adds 0.2·a to qpos with no clip, and a mutation test confirms it. The official loop clips, so any CEM or planner path must clip as well.
6. **Global numpy RNG.** Init/goal noise (±1 unit) and the teleport destination come from `np.random`, not `env.np_random`. OGBench impls seed it once, so per-episode noise depends on how many episodes came before. `evaluate` seeds `np.random`, `env.reset(seed)` and `action_space.seed`, and a mutation test confirms this matters.
7. **Teleport trap.** Out-cell (1,7) is isolated, so 1/3 of teleports end the episode as a failure. The replanning executor on the true graph scores 0.493±0.002 on teleport against 1.000 on medium and large (3 seeds × 250). Teleport execution has a low ceiling that no world model can lift: it only just exceeds the published GCIVL score of 45, so it cannot separate arms.
8. **Version, seed and campaign mixing.**
   - v1 Table 4 mixes K=8/30k and K=16/60k cells.
   - v3 says planAll and realroute come from different checkpoint campaigns.
   - Planning uses 5 seeds and execution 3.
   - The large-stitch execution of .33 places "second of seven" only on large. On medium, GCIVL 70, QRL 80 and HIQL 74 all beat MoP's .67.
9. **Raw files store L+1 states per episode.** The last action has no successor. The bed zeros it, so T = 201, not 200.
10. **The cosine edge rule depends on latent geometry.** cos > .5 does not change with scale, but it does change with the latent's offset. Latents that are not normalised and not centred (single, atlas, chart-AE) put most points in one cone. After 2 training steps on medium, they proposed 246 / 287 / 580 of 650 possible edges, which raises planAll, while MoP and the latent codebook proposed 0. Decided: the centred rule is PRIMARY and v1's raw rule is SECONDARY. The precision floor stays.
11. **Unpublished.** Encoder and head widths, LR, batch size, λ_route, λ_bal, the edge-decoding latent (online or EMA encoder), the query sampling RNG, and the replanning executor.

### D.8. Smoke (not results; PREREG not committed)
- Random uniform policy, 5×50 episodes, seed 0: medium **0.000**, large **0.000**, teleport **0.016**.
- Executor on the true graph, 3 seeds × 5 tasks × 50 episodes: medium 1.000, large 1.000, teleport 0.493±0.002 (per task .47/.53/.50/.47/.50).
- Executor on the true graph plus 10% fake edges (3 seeds × 250): replanning gives 1.000 / 1.000 (medium/large); without replanning, 0.133 / 0.200. The paper reports 0.40 → 1.00.

## Appendix E — `PREREG_DRAFT_t5_t6.md`, verbatim (title line dropped; sha256 0d375f78a099)

Draft for merging into `PREREG.md`. Nothing here is a result. Calibration numbers come from the bed alone
(no arm involved) and are filled in before PREREG is committed; they decide VOID conditions, never bars.

### E.T5 bed: `beds/swingup.py`

- Physics: MuJoCo double pendulum, links 0.5 m / 0.35 m, masses 0.1 / 0.07 kg (geometry of `beds.T2_XML`),
  hinge damping 0.02, two motors with gear 0.5 N m and ctrl in [-1, 1]. Holding link 1 horizontal needs
  ~0.71 N m, so the pendulum has to pump energy to swing up. implicitfast, dt 0.01 s, control period 0.05 s.
- State T^2 x R^2: (cos q1, sin q1, cos q2, sin q2, qd1, qd2). This is used for evaluation only (L-GIVEN).
- Observation: `beds.lift("swingup", lift_state(s))` + N(0, 0.02^2), D = 256, with velocities scaled by
  1/20. One frame is Markov. `obs="render"` (64x64) hides velocity, so it is reported separately and never
  pooled with lift.
- Offline data: `make_dataset(n_traj=4096, T=64, seed)`. Starts are uniform on T^2 with qd ~ N(0, 2).
  Actions are clip(beta sign(qd) + OU), beta ~ U(0, 1). About 1% of frames have the tip above 0.75. Seeds:
  train 0, tune 1, eval 100/101/102. No model sees beta, the controller or the state.
- Episode: start at q, qd ~ N(0, 0.05^2) (hanging near rest), 200 control steps (10 s).
- **Success** means the tip height above the pivot is >= 0.75 (max 0.85) for at least 20 consecutive
  control steps (1 s) somewhere in the 200. `evaluate()` returns the success rate, the longest hold and the
  tip trace.
- n = 50 episodes per eval seed x 3 seeds = 150 episodes per arm. Every arm sees the same starts
  (`evaluate` is deterministic in seed). The CI is a paired bootstrap over episodes (10,000 resamples).

#### Calibration (bed only, fill before commit)

| policy | budget | success |
|---|---|---|
| zero torque | none | 0/50 (seed 100) |
| uniform random | none | 0/50 (median max tip -0.11) |
| oracle CEM, true dynamics, cost = running height + 0.1 vel^2 | H 30, 256 samples, 16 elites, 5 iters, warm | 20/20 (median hold 118) |
| oracle CEM, true dynamics, cost = height | H 30, 800/32/10, cold (= `cem_plan` sampling) | 10/10, median hold 164 (rollouts in chunks of 1024, which gives identical physics; 5 blocks x 2 episodes, eval seeds 100-104) |
| oracle CEM, true dynamics, **cost = terminal L1 to goal in obs space** | H 30, 800/32/10, cold | 3/10: reaches the top (median max tip 0.850) but median hold is 18 < 20 |

The last row is the V-JEPA-2-AC energy (`arms/planner.py`) run with a perfect model and an identity
encoder. **If it scores <= floor + 0.05, then the terminal-L1 energy cannot do swing-up even with perfect
foresight. Every learned arm would then tie at the floor, and a T5 "kill" would only reflect the energy.**
In that case the shared energy changes, for all arms, to the running latent energy
sum_h ||g_h - g_goal||_1. That decision rests on this row alone and is taken before any arm is trained.

#### VOID conditions (bed-level, decided before arms run)
- Oracle ceiling (row 3) < 0.80, which means the bed is unsolvable at this budget.
- The chosen shared energy with a perfect model scores <= floor + 0.05, which means the bed cannot be
  read by the energy.
- After the runs: the best owned arm is >= ceiling - 0.05 (no headroom), so the margin is VOID, not a kill.

#### Calibration decision (bed only, taken before any arm trains)
At `cem_plan` sampling (800/32/10, cold) with true dynamics, the running height cost scores 10/10 and the
terminal L1 energy scores 3/10. The terminal energy, not the sampling or the warm start, is the binding
limit. **The shared T5 energy for every arm is therefore the running latent energy**
E(a) = mean_h ||g_h - g_goal||_1 (`t5_plan.cem_running` / `LatentMPC`, cold start, `cem_plan` sampling,
n_samples set by `budget_samples`). The terminal-L1 result (`arms/planner.cem_plan`) is reported beside it
as the owned V-JEPA-2-AC protocol, with no bar.

### E.T5 second bed: OGBench pointmaze
- Bed: `beds/ogbench_bed.py` (Thirteen). pointmaze-medium-stitch-v0 and pointmaze-large-stitch-v0, official
  `evaluate()` (5 tasks x 50 episodes, 1000-step limit, final-step success, seeded global RNG). obs = xy, so
  arms run with D = 2.
- World models train on the `train` split; the `val` split serves the L-LEARN gate.
- Planner: `LatentMPC(arm, None, H, 2, B)` replans every step. g_goal = the arm's encoding of the goal xy.
  Same running energy, the same equal-compute rule and the same tuning grid on seed 1 as swing-up.
- Arms: single JEPA, MoP-JEPA, atlas, atlas + TT, hybrid-TT, hybrid-FP, phase-sphere. MoP-JEPA's graph
  protocol (planAll) stays T4b. T5 is the closed-loop success only.
- Calibration (no arm): random policy, and the `ogbench_bed.executor` over the TRUE cell graph as a
  ceiling. Both are recorded before arms run. VOID if the ceiling is < 0.8.
- Bar / kill as for swing-up (TT >= best owned + 0.05, paired CI over the 250 episodes; kill at or below).

### E.Planning protocol (all arms)

- Receding-horizon CEM (`arms/planner.cem_plan` semantics). Execute the first action and replan every
  control step. Box clip [-1, 1]. g_goal = the arm's encoding of `swingup.goal_obs()` (noise-free, upright,
  at rest).
- **Equal planning compute.** The budget is B = (MACs per control step of the single-JEPA arm at
  H = 30, 800 samples, 10 iters). Every arm gets n_samples = floor(B / (iters x H x arm.active_flops())),
  with iters, H, elites and momentum held fixed. **Every MAC the time traveller spends on preselection,
  candidate rollouts and re-scoring is charged against B** and so reduces its n_samples. Because active
  FLOPs are already matched (L-BR), sample counts stay within a few percent of each other. The realised
  MACs per arm are logged.
- Tuning: the same grid {H in (20, 30, 40), elites in (16, 32), momentum in (0, 0.5)} for every arm on
  tune seed 1, with 20 episodes per cell. The best cell is frozen. Arms do not get different tuning budgets.
- L-LEARN gate: before planning, each arm's linear probe g -> state on the held-out split must reach
  R^2 >= 0.9 at h = 1 and its h = 10 latent error must be <= 2x its h = 1 error. An arm that fails the gate
  is reported as "did not learn" and is not compared.

### E.What counts as "reading predicted futures" fairly (§4.7 vs single-JEPA MPC)

Single-JEPA MPC already reads predicted futures: CEM scores rolled-out latents. For C5 to say anything,
the time traveller's gain has to come from one thing only, which is **re-selecting the chart set S from
the predicted futures**. It must not come from extra compute, extra information or a better energy.
1. Same energy. Re-scoring uses the shared goal energy (and the router logit). There is no learned value
   or reward head unless the same head, trained on the same data, is given to every arm.
2. Same compute. All re-scoring work counts against B, as described above.
3. Same information. No state, no true dynamics, and no access to the eval episode beyond the current obs.
   Preselection uses the static keys at time t only.
4. Native rollouts for owned arms. Single JEPA rolls out its one latent. MoP-JEPA routes at every predicted
   step, which is its native behaviour. MoP mode enumeration is not given to anyone, or else to everyone.
5. Isolating ablation: the atlas with M = m (re-scoring off) at the same B. **C5 credits the time traveller
   only if TT > (atlas, M = m) and TT >= best owned + 0.05**, both with paired-bootstrap lower bound > 0.
   If the atlas with M = m already clears the owned arms, the gain belongs to the atlas (T2/C1), not to
   time travel.
6. Mechanism (`atlas/time_traveller.py`, from the Gödel CTC figure in `ref/origin`). START: the static keys
   at the present frame preselect top-M candidate charts. Life-line: every candidate's predictor rolls the
   CEM action sequence forward. Re-score: s_k = logit_k - gamma * mean_h ||phi_k(z_k,h) - gbar_h||^2,
   where gbar is the weighted consensus future of the current top-m. HALT when the top-m set is a fixed
   point (self-consistent), or after max_loops. Each CEM sample (imagined life-line) gets its own chart
   set. gamma and max_loops are planning hyperparameters, tuned on tune seed 1 alongside the CEM grid
   (gamma in {0.3, 1, 3}, M in {2m, 4m}). The worst-case loop MACs are charged via `rollout_flops`.
7. The CTC "start/halt" loop is capped at one re-selection per control step. More passes cost B.

Bar: TT success >= best owned + 0.05 (absolute), paired 95% CI lower bound > 0, over 150 episodes.
Kill: TT <= best owned. Between the two the result is "no decision". The prior from PROMPT.md is ~20%.

### E.T6: contextuality (measurement only, `t6_contextuality.py`)

- Question X = present segment X after a shared prefix. The answer is yes iff the observer's argmax chart
  at the last frame of X equals chart_X. chart_X is the modal argmax when X is asked first. It is fixed
  before the two orders are compared.
- Contexts: `bed_segments(bed, n=2000, P=8, L=4, seed=100)`. A continues the prefix's trajectory and B comes
  from another trajectory. Beds: T2 lift and swingup lift. Arms: atlas (any m) and MoP-JEPA (router w).
  Single JEPA has no w, so it is N/A.
- Report QQ = P_AB(a != b) - P_BA(a != b), order = P_AB(1,1) - P_BA(1,1), and the marginal disturbances
  dA and dB. Each comes with a paired percentile bootstrap 95% CI (2,000 resamples, seed 0).
- The statistic is validated in `tests/test_t6.py`. It reproduces `ref/identities.py` §2 on the identities
  generators (same rng stream): mean|QQ| 0.0000 / 0.0000 / 0.0751 / 0.0000 and mean|order| 0.0702 /
  0.0397 / 0.0747 / 0.0000. A context-free observer scores exactly 0.
- Null reference: an untrained `ObserverAtlas(D=256, A=2, K=8, m in {1, 2})` (torch seed 0) on swingup
  (dataset seed 1, contexts seed 100, n = 2000) has every CI containing 0. The CI half-width is ~0.013,
  which sets the detection floor at n = 2000. Each trained arm is reported next to its own untrained init.
- Reading: QQ CI containing 0 with an order or dA/dB CI excluding 0 means the order effect satisfies the
  QQ equality. QQ CI excluding 0 means the order effect is not of Lüders form. Neither reading claims
  anything quantum. Hard top-1 answers with chart_A != chart_B make the yes/yes cell structurally rare, so
  the order effect can be ~0 while dA and dB carry the disturbance. That is why dA and dB are reported.

## Appendix F — `PREREG_DRAFT_t4a.md`, verbatim (title line dropped; sha256 334ae9f15104)

Status: DRAFT, not preregistered. Nothing has been run. Runner `run_t4a.py` (`--dry-run`, `--summarise`); tests `tests/test_t4a.py`. Run with system python (mujoco 3.11, the T2 beds).

### F.1. Claim and question
- C4 (PROMPT §6): observer competition reduces representation collapse when anti-collapse is weakened.
- T4a asks one question: when SIGReg is weakened, does OBSERVER-ATLAS lose rank more slowly than a single JEPA?

### F.2. Grid
- **Arms.**
  - Primary pair: single JEPA (d = D_g) against OBSERVER-ATLAS (K=8, m=2, d_c=n, D_g).
  - Extras, reported but not part of the bar: MoP-JEPA (K=8, d=D_g) and lazy chart-AE (K=8, m=2, d_c=n, D_g).
  - D_g = 2n+2 for every arm, so g has the same dimension in every arm: 6 on s2 and r2, 8 on so3.
  - Widths come from `arms/match.py` (L-BR), to be pinned before commit.
- **Beds.** s2 and so3 (decisive); r2 (neutral, L-WALD). All use obs = 'lift' (D=256).
  - Train data: `make_bed(bed, 256, 64, seed)`.
  - Fixed held-out eval bed: `make_bed(bed, 64, 64, seed=10000+seed)`. The lift is keyed by bed name, so it is shared.
- **SIGReg scale s ∈ {1, 0.1, 0}** of the shared table (`sigreg.LAM = 0.9`).
  - Arms mix the loss as (1−λ)·prediction + λ·SIGReg, so "0.1×" has two readings.
  - **ratio (primary):** λ = s·r/(1+s·r) with r = LAM/(1−LAM), so λ = 0.9 / 0.4737 / 0. This weakens SIGReg relative to prediction by exactly 10×.
  - **naive (reported):** λ = s·LAM = 0.9 / 0.09 / 0. This weakens it by about 91×.
- **Training.** Mirrors `run.train`: AdamW lr 1e-3, wd 1e-2, batch 32, window 32. Budget 5000 steps, logged every 50 steps.
- **Seeds:** eval 0, 1, 2, 4, 5, paired across arms. Seed 3 is the tuning seed and is never evaluated.

### F.3. Measurements, every 50 steps, on the eval bed
- `g_erank`: effective rank (`metrics.effective_rank`, Roy–Vetterli exp-entropy of singular values) of g over all eval frames.
- `chart_erank_min`: minimum per-chart effective rank of z_k. Only charts with ≥ max(2·d_c, 1% of rows) rows count, so an unused chart is not scored as collapsed. Single and MoP have one chart.
- `g_std`, `chart_std_min`: root mean per-dimension variance.
- `gate_mean` (atlas only).
- loss.

### F.4. Definitions
- **Collapse time T** = the first logged step with `g_erank` < 1.5. If it never happens, T is censored at the budget and the run is flagged.
- **Ratio** R = T_atlas / T_single, paired by seed.
- **"As fast"**: the 95% bootstrap CI lower bound of the geometric-mean R (10,000 resamples over seeds, rng seed 0) is ≤ 1.
- **Verdict**, per (bed, s):
  - VOID if every single-JEPA run is censored (no collapse to compare).
  - PASS if the CI lower bound is > 1.
  - KILL otherwise.
  - Censored atlas times enter at the budget. This underestimates the slower arm, so it is conservative for PASS.
- **Bar:** PASS at s = 0 on both s2 and so3.
- **Kill (PROMPT §8 T4a):** KILL at s = 0 on either s2 or so3, meaning the atlas collapses as fast.
- s = 0.1 is reported, not decisive (see §5, item 4). r2: the same table is reported; no gain is expected.
- **Sanity:** at s = 1 no arm may collapse. If one does, the λ table is broken and T4a is VOID.
- **Secondary verdicts, same rules:**
  - chart level: `chart_erank_min` < 1.5;
  - scale level: T_shrink = first step with g_std / g_std(step 0) < 0.01.

### F.5. Hidden variables (why a number here could lie)
1. **Effective rank ignores scale.** Uniform shrinkage, the failure `sigreg.py` documents (per-dimension std 0.05–0.25 while the loss fell), leaves erank unchanged. The scale-collapse verdict exists for that reason.
2. **The erank threshold depends on dimension.** For g, dimensions are matched (2n+2), so 1.5 means the same thing for every arm. For charts, d_c = n caps erank at 2 (s2, r2) or 3 (so3), so 1.5 is 75% or 50% of the maximum. Chart-level results are therefore secondary and cannot be compared across beds.
3. **Mixing form.** Under a convex mix, "0.1× λ" gives 10× or 91× depending on the reading. Both are logged, and the primary reading is fixed above before any run.
4. **SIGReg term count differs by arm.**
   - Atlas carries SIGReg per chart and on g, chart-AE carries it on z and g, and single carries one term.
   - At s > 0 the atlas therefore gets more anti-collapse pressure. Only s = 0 isolates "observer competition alone", which is why it is the decisive cell.
5. **EMA and predictor asymmetry resist collapse on their own (BYOL).** All arms use τ = 0.996. At s = 0 the single JEPA may never collapse within 5000 steps, which gives VOID, not PASS. The pre-commit calibration (below) confirmed this: single never collapsed.
6. **The atlas gate is its own collapse route.** g = gate·Σ w φ. A closing gate shrinks g without lowering erank (item 1). `gate_mean` is logged, and a gate below 0.05 at any log counts as scale collapse.
7. **MoP normalises z to the unit sphere.** Its g cannot shrink, so scale collapse is undefined for MoP. erank still applies.
8. **Router dead charts.** Charts below 1% usage are excluded from the chart minimum. Their count is logged, so "collapse by abandonment" stays visible and is not hidden.
9. **Seeds.** With 3 seeds the bootstrap has only 10 distinct resamples, and the CI is meaningless. 5 seeds is the minimum, and 10 is recommended if the s = 0 cell is close.
10. **Equal budgets.** Steps are equal; FLOPs and params per step are not equal unless the `match.py` widths are pinned. Collapse time is reported both in steps and in `active_flops` × steps.

11. **Arms start at different erank.** In a tiny 2-step smoke on so3, step-0-ish g_erank was 4.1 (single), 4.5 (atlas), 4.7 (MoP) and 2.0 (lazy chart-AE). An arm that starts near 1.5 "collapses" trivially. Rule: a run whose step-0 g_erank is < 2·THR = 3.0 is flagged "starts collapsed" and excluded from its verdict. The step-0 erank is reported per arm.
12. **Gate closure and chart abandonment are logged** (`t_gate_closed` at gate < 0.05; `n_charts_used/n_charts_seen` per log). Either counts as a collapse route in the secondary verdict.

### F.Pre-commit calibration (baseline only)
Approved by the coordinator: single JEPA only, SIGReg scale 0, bed s2, tuning seed 3, 5000 steps. No other arm was run. Output went to the scratchpad, not `runs/`. This is bed calibration, not a result.
- **Command** (system python 3.11.9, torch 2.14.0+cu126 on CPU with 4 threads, uncommitted tree, `run_t4a.py` sha256 prefix 1566d144):
  `python -X faulthandler cal.py`
  This calls `run_t4a.run_job(dict(arm="single", bed="s2", scale=0.0, seed=3, ...), "uncommitted", steps=5000, log_every=50)`, writing to `<scratchpad>/t4a_cal/single__s2__lam0__s3.json`. λ = 0.0 and n_params = 68,748.
- **Outcome: censored, no collapse.**
  - t_collapse_g = 5000 and t_shrink_g = 5000, both censored.
  - g_erank started at 4.20, fell to a minimum of 3.32 at step 3450, then rose to 5.02 at step 5000.
  - g_std went from 0.035 to 0.83: the latent expanded about 24×. It did not shrink.
  - Wall time was 495 s under load.
- **Effect on the rules.** Raising the budget would not help, since erank was rising at the end. Single at s = 0 does not collapse on s2, so the collapse-time ratio is undefined there. The following replaces the bar and kill in §4 (proposal):
  - **VOID wording:** "On a (bed, s) cell where the single JEPA is censored on ≥ 3 of 5 eval seeds, there is no collapse for observer competition to prevent. T4a scores that cell VOID ('baseline does not collapse'), never PASS. If every decisive cell (s = 0 on s2 and so3) is VOID, C4's collapse claim is VOID: untestable on these beds with these arms, not supported and not killed."
  - **Continuous measure, reported, not a bar:** per seed, the minimum over the curve of g_erank / D_g, as the paired atlas − single difference with a bootstrap 95% CI. It is descriptive only, because no threshold for it was fixed before the calibration.
  - **Budget** stays at 5000 steps, which equals the calibrated horizon. A longer budget is not justified because erank was rising.
  - **Optional, Seal's call:** add the naive-λ reading (0.09) or EMA τ = 0 (no target smoothing) as a harsher stressor, so the baseline does collapse. Either one has to be calibrated on tuning seed 3 again before commit.
- **Environment hazard.** The first two calibration attempts segfaulted (exit 139) after about 100 steps. During the same window a fresh `import torch` failed with WinError 1455 (paging file too small): the commit charge was 24 of 36 GB, held by other lanes' processes. The third attempt, with `torch.set_num_threads(4)`, completed. Rule: a crashed job leaves no JSON (atomic write) and is re-run on resume. A job is never scored from a partial curve.
- **Seed hygiene.** Seed 3 was in the default eval seeds {0..4}. The eval seeds are now `EVAL_SEEDS = (0, 1, 2, 4, 5)`, and `plan()` raises on any `TUNING_SEEDS` member (test `test_tuning_seed_never_evaluated`).

#### Recalibration: online-target stressor (coordinator ruling)
- **Ruling.** The PRIMARY decisive cell is SIGReg scale 0 with an online target: EMA momentum 0, so the target is the current online encoder under stop-grad. It applies identically to every arm through `build_arm(target="online")` (τ = 0 for single, MoP and chart-AE; `ema` = 0 for the atlas). SECONDARY is scale 0 with the EMA target (the calibration above). The VOID rule is unchanged. The runner adds `--targets ema online`; online keys end in `__online`.
- **Run.** single only, s2, tuning seed 3, 5000 steps, 4 torch threads, one process. Peak RSS 0.91 GB; 4.4 GB was available at launch. `run_t4a.py` sha256 prefix 39616edc.
  Command: `python -X faulthandler cal_online.py`, which calls `run_job(dict(arm="single", bed="s2", scale=0.0, target="online", seed=3, ...), "uncommitted", steps=5000, log_every=50)` and writes `<scratchpad>/t4a_cal/single__s2__lam0__s3__online.json`. Wall time 347 s.
- **Outcome: censored, no collapse.**
  - t_collapse_g = 5000 and t_shrink_g = 5000, both censored.
  - g_erank went from 4.20 to a minimum of 1.998 at step 200, then back to 3.67 at step 5000.
  - g_std went from 0.035 to a maximum of 3.61, then to 2.30. The latent blew up in scale; it did not collapse.
  - Without the EMA lag, the predictor-asymmetry route (no SIGReg) did not collapse a 6-d single JEPA on s2 within the budget. Its minimum erank of 2.0 stayed above THR = 1.5.
- **Effect.** By the VOID rule, the primary cell is VOID on s2 unless single collapses on ≥ 3 of 5 eval seeds; the tuning seed predicts it will not. so3 is uncalibrated: the ruling allowed s2 only. The min-erank-over-curve difference stays descriptive. The stronger classic collapse setting, with no predictor and no target asymmetry, is not proposed, because it would not be a JEPA.

### F.6. Order
Pin widths (match.py) → calibration done (single censored; VOID wording above) → commit PREREG → `run_t4a.py --procs 3`. That is 4 arms × 3 beds × 3 scales × 5 seeds = 180 jobs.

## Appendix G — `PREREG_DRAFT_atlas.md`, verbatim (title line dropped; sha256 416fbb2f5fd3)

Code: `atlas/observer_atlas.py`, `sigreg.py`, `common.py`. Tests: `tests/test_observer_atlas.py` and `tests/test_sigreg.py`. Every pin below is bound by a test named next to it.

### G.1. Shared anti-collapse table, the same for every arm (fairness)
- **SIGReg.** `sigreg.py`: sliced Epps–Pulley test against N(0,1), port of the LeJEPA reference: t ∈ [0,3], 17 knots, 256 slices drawn fresh each call.
  - It drops LeJEPA's N factor, so λ is in N-free units.
  - Grouped (per-chart) mode makes **one** slice draw shared by all groups. It scatters over groups, so cost is O(N) whatever the group count, and the value does not change when charts are relabelled.
  - Groups with fewer than 2 rows are skipped.
  - Tests: `test_grouped_equals_mean_of_groups_with_one_shared_slice_draw`, `test_group_relabelling_invariant`, `test_cost_independent_of_group_count`.
- **Mixing form for every arm:** loss = (1 − λ)·(prediction terms) + λ·Σ(SIGReg terms).
- **Scale anchor (deviation from LeJEPA).** `sigreg` = Epps–Pulley + 0.1·mean over slices of log(E[(x·a)²] + 1e-6)².
  - Reason: Epps–Pulley without the N factor gives almost no restoring force on expansion. For x = s·N(0, I), d/ds is −0.658 at s = 0.3, +0.0076 at s = 15 and ≈0 at s = 30.
  - With the anchor it is +0.152 at s = 15.
  - Without the anchor, the atlas diverged on 2 of 3 seeds: g std 15.2 and 11.3, prediction loss 0.1 → 134.
  - Test: `test_restoring_force_on_expansion`.
- **λ = `sigreg.LAM` = 0.9**, the default of every arm (single, MoP, chart-AE, lazy chart-AE, PEER, atlas).
  - Conditions: grid {0.5, 0.75, 0.9, 0.95}; S² bed; D_g = 2n + 2 = 6; 200 Adam steps (lr 1e-3, batch 32×16); seeds 0, 1, 2; anchor on.

  | λ | Result (5 arms, 3 seeds) |
  |---|---|
  | 0.5 | The atlas fails 2 of 3 seeds (g std 17.7 and 5.1). |
  | 0.75 | 15/15 pass (atlas g std 0.93–1.42). |
  | 0.9 | 15/15 pass (atlas g std 0.94–0.98). |

  - 0.9 is pinned: one grid step above the smallest all-pass value, as margin.
  - Earlier single-seed tables (without the anchor, D_g = 8) are superseded. Their numbers are kept in house-events.jsonl.
- **Chart-AE.** It fails the rank leg of the contract on 3 of 3 seeds at every λ in {0.75, 0.9, 0.95}: erank(g) is 1.57–2.0 against n = 2, while its std passes.
  - This is a learning failure of the sigmoid-chart design, not collapse. It is flagged to Cameron.
  - The test is a strict xfail, so a pass shows up as news.
  - Under L-LEARN, chart-AE cannot be compared until it clears the contract.
- **Scale contract (L-LEARN precondition).** Before any comparison, every arm must pass on the tuning seed:
  - per-dim std of every tensor its SIGReg acts on ∈ [0.3, 3];
  - effective rank of g ≥ n;
  - for top-m routers, at least K − 2 charts used on more than 1% of inputs.
  - Test: `test_scale_contract_every_arm_same_lambda[*]`.
- **T4a ablation levels.** "SIGReg 0.1×" means the ratio λ/(1 − λ) multiplied by 0.1, i.e. λ = 0.4737. "0×" means λ = 0.
- **Load balance λ_bal = `sigreg.LAM_BAL` = 0.3.** It is the Switch form K·Σ f_k P_k, inside the (1 − λ) bracket.
  - It applies to the top-m routers: atlas and lazy chart-AE.
  - Fact (atlas, circle bed): λ_bal = 0.01 left 2–3 of 8 charts dead after 200 steps; 0.3 left 8/8 alive.
  - MoP-JEPA keeps its paper's own balance term (λ_bal = 0, reported).
- **Atlas-only weights**, both inside the (1 − λ) bracket:
  - overlap λ_overlap = 0.1, with stop-grad weights;
  - λ_div = 0 (off): optional chart diversity −λ_div·mean_{i<j} KL(ρ_i‖ρ_j), with ρ_k = chart k's occupancy (a column of w_soft normalised over the batch, ε = 1e-6).
- **λ_div source and sign.** Ported from `ref/origin/AETHER_qGPU_event_driven.pdf` L_Div. The sign is flipped: the paper's +γ·L_Div, minimised, pulls modules together.
- **Smoke comparison** (lr 3e-4, 2 seeds; numbers before the anchor):
  - no guard: 6/8 charts alive;
  - λ_bal = 0.3: 8/8 alive;
  - λ_div = 0.1 alone: 8/8 alive;
  - neither guard changed the scale failure.
- **λ_div cost:** O(N·K²) per training step. It is computed even when off, for reporting; at K = 1024 that is about 2 GMAC per step at N = 2048. T3 should skip it there.

### G.2. Pins common to every arm
- **EMA τ = 0.996.** EMA copies are excluded from `n_params`: arms exclude children named `tgt_*`; the atlas excludes parameters with requires_grad False. L-BR parameter matching uses `n_params()`.
- **TF32 off.** `common.py` sets `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False`. It is imported by `arms/__init__.py` and the atlas. Test: `test_tf32_pinned_off`.
- **Determinism.** Bitwise on CPU under a seed (`test_determinism_under_seed`). On CUDA, the SIGReg scatter uses atomics (nondeterministic at about 1e-7); seeds are compared with a tolerance there.

### G.3. OBSERVER-ATLAS design pins
- **Backbone.** A GRU with d_h = 64. The context c_t has already read o_t, so the router sees the current observation; this is strictly causal (`test_causality`).
- **Streaming path.** `step()` and `rollout()` use an `nn.GRUCell` that shares the same Parameter objects as the GRU. Parity with the sequence path is 1e-12 in float64, and nn.GRU is never called (`test_streaming_step_matches_sequence_and_skips_nn_gru`). T7 times `step()`.
- **Router.**
  - q = W_q c, d_key = 32, logits q·κ/√d_key, then top-m and a renormalising softmax.
  - Flat keys for K < 256. Product keys for K ≥ 256; their top-m is exactly the exhaustive top-m (`test_product_keys_topm_equals_exhaustive_topm`).
- **Gate.** One scalar per step, sigmoid(W c_t + b), multiplying g and ĝ. A closed gate gives g = 0 with a finite loss and finite gradients (`test_gate_closed_outputs_nothing_without_nan`).
- **ĝ_{t+1}** uses the weights, chart set and gate of time t; t+1 weights would leak the target. The target is sg(g_{t+1}).
- **Per-chart prediction loss** is unweighted by w. Overlap uses stop-grad w.
- **Lazy rule.** Only the top-m charts read their weights. Proven three ways:
  - setting unselected weights to NaN leaves outputs bit-identical, and a dense control turns NaN;
  - `FlopCounterMode` equals `active_flops()` exactly;
  - lazy outputs equal the dense top-m mixture.
- **active_flops (MACs per step)** = 3·d_h(D + d_h) + d_h·d_key + d_h + {K·d_key flat | √K·d_key product} + m[D·h + h·d_c + (d_c + A)·h + h·d_c + 2(d_c·h + h·D_g)] + 2m·D_g.
- **Rollout policy.** The chart set, weights and gate are frozen at t = 0; there is no re-routing on predictions.
  - Lazy chart-AE re-routes from ĝ. That asymmetry is declared, not harmonised: re-routing on predictions is the §4.7 time-traveller, which is unbuilt (it raises NotImplementedError).
- **Not tuned:** d_c = n, D_g = 2n + 2 (coordinator ruling; atlas default 6, runners pass it per bed), hidden = 32 for chart MLPs, lr and batch per Appendix B's tuning grid on the tuning seed.

### G.4. Topology readout (C2)
- **Nerve source.** `nerve(w_soft, τ, max_dim=3)` on the pre-top-m softmax over all K, multiplied by the gate so that closed-gate inputs add no simplices. The top-m nerve (`w`) is an ablation only, because its dimension is at most m − 1.
- **τ grid:** {1/4, 1/5, 1/6, 1/8, 1/10, 1/12, 1/16, 1/24, 1/32}, as in PREREG.md §T2. The reported vector is `metrics.most_persistent`. Betti numbers up to b3, so max_dim = 3 for SO(3).
- **K for P-BETTI.** P-BETTI is scored only at K ≥ the minimal triangulation size of each manifold: S² ≥ 4, T² ≥ 7, Klein ≥ 8, RP³ ≥ 11.
  - In practice that means K ∈ {16, 64} (PREREG.md conflict row 6).
  - P-COUNT is scored at K = atlas(M), where a nerve with correct Betti numbers is impossible: two charts on S² give a nerve that is a single edge.
- **Integration.** Atlas `nerve` matches `metrics.betti_z2` and `nerve_betti_z2` (`test_nerve_integrates_with_metrics_betti`).

### G.5. Known limits, reported and not tuned away
- **Atlas sits near the edge of the λ window.** It blew up at λ = 0.75 and is stable at 0.9 and 0.95.
- **The scale contract is a 200-step smoke test, not a convergence guarantee.**
- **Adam momentum still updates unselected chart rows.** Their gradient is exactly zero, so a dead chart drifts but does not learn. Load balance is the only revival path; it reaches every key (`test_grad_reaches_selected_charts_only_and_all_keys`).
- Origin rule (`zero/CLAUDE.md`, `ref/origin/`): the JEPA-shaped losses here (per-chart predictor, EMA target, SIGReg) are the current vocabulary, not a constraint. Any origin-driven redesign of the chart objective is a new arm and must clear the same scale contract (§1) before comparison.

## Appendix H — `PREREG_DRAFT_armQ.md`, verbatim (title line dropped; sha256 abf4dcf081ea)

Code: `atlas/coherent_observer.py` (`CoherentAtlas`, a subclass of `ObserverAtlas`; Chase's file is not
edited). Tests: `tests/test_coherent_observer.py`, 11/11.
Recorded spec change (folder rule 2, origin wins): PROMPT.md §7 dropped the Born/Lüders decider. The origin
notes say "small phases all around the sphere but the decider will be a softmax", so the decider is
restored as a Born rule over phased chart amplitudes. Softmax stays as its decoherence limit.

### H.Mechanism
- Softmax router = Born rule on a decohered state, p = |psi|^2 with psi = sqrt(softmax(s)). It obeys Luce IIA.
- Arm Q: psi_k = sqrt(softmax(s))_k e^{i theta_k(q)}, p = |U psi|^2, with U = Cayley(P Q^T - Q P^T) real
  orthogonal of rank 2r. Woodbury gives O(K r) per query and no KxK matrix is ever built (tested at
  K = 200,000). theta_k(q) = (W_q c) . theta_key_k.
- Measurement = top-m of p + the parent's gate. These are the nonlinear, linearity-breaking steps.
- Limits, all tested: P = 0 reproduces the exact softmax for any theta (the init). theta = 0 gives the real
  sign-only model. grad_P != 0 at P = 0, so coherence is learned and never given (L-GIVEN). The reference
  `born_gibbs` reproduces identities §6: p1/p2 = 1.3499 / 1.5165 / 2.0587 at g = 0 / 0.5 / 1.

### H.Hypothesis
Under softmax IIA, near-duplicate charts split weight (two equal keys each get half). Their top-m slots
are wasted and spurious co-selections show up in the nerve. Coherent coupling lets duplicates interfere
destructively, so one survives. Prediction: fewer duplicate charts and a cleaner nerve at equal compute.

### H.Arms (T2 beds s2, t2, klein, so3 + neutral r2, lift obs; 3 seeds; tuning on a separate seed)
1. Softmax `ObserverAtlas`. 2. `CoherentAtlas` (rank r in {2, 4}). 3. Ablation: Arm Q with P frozen at 0
(same params, softmax behaviour). 4. Ablation: theta frozen at 0 (real-only interference).
Equal active FLOPs and equal total params (L-BR). Arm Q pays K (d_key + 16 r + 2) + 16 r^2 extra MACs per
step, which comes out of the chart hidden width via `arms/match.py`.

### H.Metrics (preregistered definitions)
- Duplicate-chart rate: fraction of chart pairs whose top-m selection masks over the eval set have
  Jaccard >= 0.9 AND whose glued images agree on the co-selected points:
  mean ||phi_j(z_j) - phi_k(z_k)|| <= delta_dup * median pairwise ||g_a - g_b||.
  **delta_dup = 0.05**: the fold_delta scale ("same point"), relative to the median latent distance as in
  `metrics.fold_rate`. It is NOT the discontinuity delta 0.5, which marks a tear (a gap of half the median
  distance is two different points, not a duplicate). Amendment (c) rescales the discontinuity BAR relative
  to the torn reference; it does not change either delta. The duplicate bar (<= 0.5x softmax) is already
  relative, so it needs no torn reference. Sensitivity at delta_dup in {0.02, 0.1} is reported, with no bar.
- Nerve Betti over Z2 vs truth, discontinuity rate, h = 1 and h = 10 latent prediction error (T2 definitions).
- Learned IIA violation: mean |Δ log(p_j / p_k)| for top-2 charts j, k when one other chart's amplitude is
  zeroed and p renormalised. Softmax scores 0 exactly.
- Coupling used: ||P||_F / ||Q||_F at the end of training.

### H.Bar (all required, >= 3/4 curved manifolds, 3 seeds)
Arm Q's duplicate rate <= 0.5x softmax's; Betti correct on >= as many manifolds as softmax; discontinuity
and h = 10 error <= softmax + 2%; neutral r2 error within +2%; the P = 0 ablation does not reproduce the
gain (otherwise the gain comes from params, not coherence).

### H.Kill
- Arm Q <= softmax on the T2 metrics (no improvement in duplicate rate, Betti or discontinuity) means Arm Q is dead.
- Learned coupling stays ~0 (||P|| / ||Q|| < 0.01 on >= 3/4 manifolds) means SGD picks decoherence, the
  coherence is unused, and Arm Q is dead.
- The theta = 0 ablation ties full Arm Q means phases are unused. Arm Q survives only as "real
  interference" and the phase claim is dead.

### H.T6 link (measurement only)
Run `t6_contextuality` (same protocol as PREREG_DRAFT_t5_t6.md) on softmax vs Arm Q, each next to its
untrained init. Report QQ, order, dA and dB with paired bootstrap CIs. Within one step the observer is still
a map of c_t, so order effects come from the GRU context in both arms. The question is whether coherence
changes their form (QQ ≈ 0 with order ≠ 0 is the Lüders signature). A Lüders signature is "consistent
with", never "evidence of", quantum structure. No bar, no kill.

Prior (guess): ~15% that Arm Q clears the bar.

## Appendix I — `ORIGIN.md`, verbatim (title line dropped; sha256 76adf6ad6689)

**Sources**
- `ref/origin/AT_I_notes.pdf`: two handwritten pages, read at 6–10× zoom.
- `ref/origin/godel_ctc_figure.jpg`: the Gödel-universe CTC figure.

**Tags**
- **[T]**: transcription.
- **[?]**: illegible or uncertain word.
- **[R]**: Foreman's reading. A reading is not a fact: Seal confirms or rejects it.
- **[K]**: known result, not fetched this session.
- **[U]**: uncertain.

---

### I.1. Transcription

#### Page 1, drawings (top to bottom)

**D1. "AT" chain.** [T]
- "AT", then four small circles joined by a line, with arcs underneath joining circles 1–2 and 3–4.

**D2. Node chain with a return arc.** [T]
- A node "m" connects by a line labelled "λ"[?] to a large ellipse with an inner ellipse, labelled "ν"/"V"[?].
- The line continues to a node "Us"[?], then to an ellipse "Teeth"[?].
- An arc labelled "y"[?] returns from below the large ellipse to "m".

**D3. Axis plot.** [T]
- A vertical axis with a curve that falls steeply, then flattens.
- An arrow up to the right, labelled "entropy ∝ Tez"[?] (the last word is illegible).

**D4. Large sphere.** [T]
- Many concentric and nested ellipses inside a large closed curve.
- Straight rays converge on a point at the right.
- Heavy red dots and slash marks.
- Small loops labelled "C", "P", "A".
- An arrow pointing out of the sphere labelled "my".
- A mark "∪ ?"[?] at lower right.

**D5. Box of cells.** [T]
- A rectangle split into 2–3 cells, with small circles in the cells.
- A heavy dot at the top-right corner, joined by two lines to an outside circle at the right.
- Labels: "o Surgery"[?] on top, "cost" on the left, "drug (medicals)" underneath.

**D6. Crossing strands.** [T]
- Label "π law".
- Three circles rise diagonally to the upper right. Two strands join them and cross each other in an X below.

**D7. Disc in a flow.** [T]
- A disc with 3–4 small circles inside, joined by a zigzag path with small "π"/"rr"-like tick marks[?].
- About ten long curved lines sweep across the disc from lower left to upper right, like field lines or rays passing an obstacle.

**D8. Lattice of nodes.** [T]
- A small circle and a line, with "λ y"[?].
- A lattice of about 15 ellipses (nodes).
- One node at the left sends 3 straight rays through the lattice: one down, two to the right.
- A doubly circled node sits where two rays meet.

**D9. "What I propose".** [T]
- A 3-D box (a frame with depth).
- Below it, a triangle.
- A dotted line starts at the box's upper left, passes behind and through the box, exits right, curves down, crosses into the triangle, exits at its base and continues to the right.

#### Page 1, text [T]

> "now [O]ne must ask / but guruji why / WTF"
>
> "So what I wanna say is like we start this / fress[?] a new transformation of data / via scattering in to fractal / assymetric bounds and phases aka hose[?] / legfht[?] & it, information travels to / what that clotted [dotted] line is The observer like is / we dont know great mysteries of an / body but we use it / it will be combination transformer / inspired by "JEV"[?] and "JEPA" / but I want to go primal / a dual layer structure was tried but the problem / was on the 'Oberseur' [observer] make decision while / JEPA uses causality side Kinda the / beauty of this is defined in topology / basically we will have jepa as a phase / over higher dimensional S2 sphere / [?] like"

#### Page 2 [T]

**Drawings and text, in order:**
- "Kinda". A sphere cut into about 8 sectors by lines that all meet at one interior point. One sector carries a boxed label "JEPA".
- "So yes a truly topological transforce [transformer] / shall not pollute itself with / JEPA, it will have JEPA / as self attention have softmax / Small phases all around the sphere / but the decider will be a softmax here?"
- Drawing: an outer rounded cylinder or box containing an inner cylinder; an arrow labels it "JEPA".
  - A closed loop, a twisted figure-8 or knot, passes through both, with a small "..." break where it crosses the inner body.
  - Two strands are labelled "Causality" and "Tsoffran[?] observer".
  - Underneath: "bound of ICCR[?] PAPER KINDA".
- "Guruji now have tons for making a plan"
- "→ New design Not JEPA, nor self attention"
- "→ Newer layers"
- "→ New repo so New grounds"
- "→ is inspired by use of topology used in universe and our style of topology"
- "→ Also the internal design of this should also be butterfly effect bonded via topology"
- "→ Aim Fuck ALLEN TURING we Do TOPOLOGY". The signature "Guruji" is underlined twice.

#### Gödel CTC figure [T]

- **Frame.** A vertical cylinder in coordinates (t, r, Φ), with t along the axis.
- **Light cones.** Small light cones tilt more as r grows. The central hourglass is a surface of tipping cones.
- **Labels.**
  - "time-traveller's life-line (time-like curve)": blue, spiralling outward, then back.
  - "closed photon-like curve".
  - "CTC's": horizontal circles of constant t at large r.
  - "r = constant hypersurface": the dashed cylinder.
- **Events on the axis.**
  - "start": at s, an event on the t-axis, upper.
  - "halt": at h, an event on the t-axis **below s**.
- **The life-line's path:**
  1. It leaves s, spirals out beyond the radius where the Φ-circles turn time-like.
  2. It rides them backward in t.
  3. It returns to the axis at h, earlier than s.
- **Reading of the figure.** [R] It is the relativistic-hypercomputation figure of the Andréka–Németi school [K/U, source not fetched]: the halting answer reaches the observer before the computation starts.

---

### I.2. Mathematical objects (each [R])

**O1. "jepa as a phase over higher dimensional S2 sphere / small phases all around the sphere / decider = softmax", with the sector sphere on page 2.**
- The state is a point of projective Hilbert space CP^{N−1}, a unit complex vector ψ ∈ C^N modulo a global phase.
  - CP¹ = S² is the Bloch sphere. "Higher dimensional S²" = CP^{N−1}.
  - The "small phases all around the sphere" are the U(1) fibre: a phase field, i.e. a section of a U(1) bundle over the sphere.
- The sector sphere is a phase gauge cut into patches. The point where every sector line meets is where a single gauge breaks down: a vortex, since no global continuous phase exists on a bundle with non-zero Chern number (the Hopf bundle S³→S² has c₁ = 1).
- Each small model is one local gauge on one patch. Patches are glued by transition phases e^{iχ_jk}.
- The topological readout is the Chern number. It is an integer computed from windings of the transition phases on overlaps, not from nerve Betti numbers.
- "Decider = softmax" = the Born rule. p_k = |⟨u_k,ψ⟩|² / Σ_j |⟨u_j,ψ⟩|², which is exactly softmax(2 log|⟨u_k,ψ⟩|).
- This matches CLAUDE.md rule 3: linear superposition, then a measurement that breaks linearity.

**O2. Nested cylinders with a knotted loop labelled "Causality" / "observer"** (the same geometry as the Gödel figure).
- The inner cylinder ("JEPA") is the causal region, where prediction runs forward in time.
- The loop that leaves and re-enters it is the observer's closed time-like curve.
- Computational object: a self-consistent fixed point (the Deutsch CTC condition, x = F(x) [K]).
  - The observer's choice w must equal the choice it would make after seeing the future that w itself causes: w* = D(P_h(z, w*)).
  - A classical stochastic loop map always has a fixed point (Brouwer / Perron–Frobenius [K]), so there is no grandfather paradox.
- This is the "dual layer" of the text. The decision layer (observer) and the causal layer (JEPA) conflicted; the loop makes them consistent.

**O3. Gödel figure, halt before start.** Two readings:
- (a) **Amortisation.** The decider learns to output the converged fixed point w* in one shot from the start state. The answer exists before the iteration that defines it runs.
- (b) **Phase becomes time.** Beyond the critical radius the angle Φ is time-like. The evolution is then a rotation in phase, and a closed loop in phase returns to its start carrying a holonomy.

**O4. "butterfly effect bonded via topology".**
- The continuous internal state may be chaotic (largest finite-time Lyapunov exponent λ₁ > 0).
- The discrete topological readout (winding, Chern number, holonomy class) cannot change under small perturbations unless a gap closes: topological protection.

**O5. "scattering in to fractal asymmetric bounds and phases"** (D7). Two readings, undecided:
- (a) **Wavelet scattering transform** (Mallat [K]). A multiscale ("fractal") cascade of wavelet-modulus operators. The modulus discards phase, and "phases" are what is kept or removed.
- (b) **Physical scattering.** Rays deflected by a body; the information is in phase shifts.

**O6. Lattice node emitting rays (D8).**
- Lazy selection by ray casting: only lattice nodes on the observer's rays compute ("observed → exists").
- A node hit by two rays is co-selected, i.e. a simplex.
- Mathematically, the selected set is a union of lines in a 2-D grid of charts. That is product-key selection (rows × columns) seen geometrically.

**O7. "What I propose": box + triangle + dotted line ("what that dotted line is: the observer").**
- The cells have different shapes: a cube cell and a simplex cell, so this is a cell complex, not a single chart type.
- The observer is a path through the cells.
- The transformation of the data is transport along that path, i.e. a discrete connection (a group element per cell crossing).
- On a closed path the readout is the holonomy.

**O8. "π law" with crossing strands (D6).**
- Exchanging two strands multiplies by the phase e^{iπ} = −1. This is exchange statistics, or holonomy π, which is orientation reversal.
- It joins O7: transport around the Klein bottle's non-orientable loop has holonomy π (sign −1), and around a torus loop +1.

**O9. Adaptive POVM (Seal, `ref/origin/AETHER_qGPU_event_driven.pdf`, "Quantum GPU Architecture with Adaptive Event-Driven Frame Rendering" [V, read this session]).**
- The paper defines E_k(Φ) = U†(Φ) Π_k U(Φ): a learned unitary rotates fixed projectors Π_k = |k⟩⟨k|.
- Completeness Σ_k E_k = U† I U = I then holds for any Φ.
- Φ is updated by expectation feedback: Φ′ = Φ + λ₂⟨Ψ|O₂|Ψ⟩.
- [R] This is the PHASE-SPHERE Born decider in POVM form, with chart gauges u_k = U†e_k.
- [D] Caveat: completeness by construction holds only for K ≤ N rank-1 projectors. K = 64 > N = 2n+2 needs a Naimark dilation to dimension ≥ K, which costs ≥ K·N per step. That is no cheaper than the current renormalised Born scores (4NK MACs).
- Design consequence: the POVM form is an optional variant for K ≤ N. In that variant the null outcome is I − Σ E_k.
- The event-driven time laziness in the same paper belongs to Taub.

**D1–D5 and D3 ("entropy ∝ Tez[?]").**
- No object can be extracted without Seal's gloss.
- D5 (surgery[?] / drug / cost) reads as a branching-decision example, i.e. mode choice.
- They get no test until clarified.

---

### I.3. Where OBSERVER-ATLAS diverges from the origin

| | OBSERVER-ATLAS (PROMPT §4) | Origin [R] |
|---|---|---|
| State | real chart z_k ∈ R^{d_c}, glued into R^{D_g} | complex amplitude ψ ∈ C^N modulo phase (CP^{N−1}); a phase per patch |
| Gluing | learned real maps φ_k plus an overlap MSE | transition **phases** e^{iχ_jk}; consistency = holonomy, integer winding |
| Decider | softmax(q·κ) over keys, from a backbone context (attention-shaped) | Born rule \|⟨u_k,ψ⟩\|²: linear superposition, nonlinearity only at measurement |
| Dynamics | MLP predictor per chart | linear and phase-preserving: ψ_{t+1} = U(a_t)ψ_t, with U = exp(skew-Hermitian generator(a)) |
| Observer | one feed-forward pass; time-traveller optional (T5) | the observer is a **closed loop**: choice and predicted future solved to a self-consistent fixed point (start → iterate → halt); central, not optional |
| Topology readout | nerve Betti over Z2 (blind to T² vs Klein; dimension ≤ m−1) | Chern number / holonomy: integer, read from a path or loop; separates T² from Klein (holonomy ±1) |
| Lazy compute | top-m keys | rays through a chart lattice (O6) |
| Chaos | not addressed | continuous part may be chaotic; the integer readout is protected (O4) |

**Proposed layer, PHASE-SPHERE (a design implied by O1–O8, to be specified by Chase and attacked by House):**
1. ψ_t = normalise(E(o_t)) ∈ C^N.
2. K chart gauges u_k ∈ C^N, learned.
3. Amplitudes α_k = ⟨u_k, ψ⟩. Born decider p = |α|² / Σ|α|². Keep the top-m, plus a "nothing" outcome: the gated observer from identities.out.
4. Lazy: only the selected charts run a local unitary step U_k(a)α_k.
5. Transition phases χ_jk = arg α_j − arg α_k on co-selected pairs.
6. Readouts:
   - Chern / winding = (1/2π) Σ over chart cycles of wrapped χ.
   - Holonomy of the observer's closed path.
7. Observer loop: w ← Born(U_h(w)ψ), iterated to ‖Δw‖ < ε ("halt"). It is amortised by a one-shot head (O3a).
8. L-GIVEN holds: u_k, the generators and E are SGD-learned. No chart, phase or loop is supplied.

---

### I.4. Falsifiable tests (PREREG style)

Shared rules for every test below:
- 3 seeds, plus tuning seed 3.
- Equal active FLOPs to the owned arms, with the observer-loop iterations counted.
- Bars and kills are fixed before any result.

**TO1, phase sphere / Chern (O1).**
- **Bed: Berry bed.** A direction n_t ∈ S² follows the overdamped random walk of the s2 bed.
  - The observation is lift(n_t) plus an interference channel cos(γ_t), where γ_t = (1/2) × the solid angle swept by the path relative to a fixed reference (spin-½ Berry phase).
  - Control bed: the same with γ ≡ 0, a trivial bundle.
- **Readout:** the learned Chern number c₁ = (1/2π) Σ wrapped transition phases over a triangulated chart cycle.
- **Bar:** c₁ = 1 on the Berry bed and 0 on the control, on 3/3 seeds, with |c₁ − round(c₁)| < 0.1. The h=10 npersist on the interference channel must be ≤ that of the best real-valued arm.
- **Kill:** c₁ wrong on ≥ 2 seeds, or a real-valued OBSERVER-ATLAS matches interference npersist at equal FLOPs.

**TO2, CTC observer as a fixed point (O2).**
- **Bar:**
  - The loop converges (residual < 1e-4 within 20 iterations) on ≥ 99% of eval inputs.
  - h=10 npersist is ≤ 0.95 × that of one-pass routing at equal total FLOPs.
- **Kill:** non-convergence > 5%, or no gain at equal FLOPs.
- **Unit test (RED first):** the solver finds the fixed point of 1,000 random column-stochastic matrices to 1e-8.

**TO3, halt before start (O3a).**
- **Bar:** a one-shot head predicts the converged w* with top-1 agreement ≥ 95% on held-out data, at ≤ 1/5 of the loop's FLOPs.
- **Kill:** agreement < 80%.

**TO4, butterfly bonded via topology (O4).**
- **Measure:** the finite-time largest Lyapunov exponent λ₁ of the latent rollout, from the product of rollout Jacobians over h = 10.
- **Measure:** the flip rate of the integer readout (c₁ or holonomy class) under input perturbations at 1× the observation-noise σ.
- **Bar:** λ₁ > 0 and flip rate < 1%.
- **Kill:**
  - λ₁ ≤ 0: no butterfly, so the claim is vacuous.
  - Or flip rate ≥ 5%: not bonded.

**TO5, scattering front end (O5a).**
- **Arm:** a fixed multiscale Morlet-modulus cascade feeding the layer, against a learned MLP encoder at equal FLOPs.
- **Bar:** ≥ 1.5× fewer steps to reach h=1 npersist ≤ 0.5.
- **Kill:** < 1.1×.
- **L-GIVEN note:** a fixed cascade is hand-given structure, so a win scores "representable only" unless the cascade's filters are learned.

**TO6, rays through a lattice (O6).**
- **Arm:** charts on a √K × √K grid. Selection = cells on r learned lines (rays), at the same active count as top-m.
- **Bar:** loss ≤ product keys at K = 256 and 1024. Compute grows ∝ r·√K, not K.
- **Kill:** loss worse than product keys at both K.

**TO7, holonomy along the observer's path; π law (O7, O8).**
- **Beds:** t2 and klein (lift).
- **Readout:** a learned frame is transported along held-out closed state loops through the selected cells, and its sign holonomy is measured.
- **Bar:** the holonomy sign is correct on ≥ 95% of loops. Klein's non-orientable generator must give −1; every T² loop must give +1. This holds on 3/3 seeds.
  - This is the test Z2 nerve Betti cannot pass, since both manifolds read (1,2,1).
- **Kill:** < 80% correct, or identical sign patterns on T² and Klein.

Every former [?] item is resolved in §5, each as a committed reading. No item is waiting on Seal.


---

### I.5. Committed readings for every former [?] item (O10–O18)

Each item gets a reading, a design slot and a test. The readings are Foreman's best guesses, now treated as committed, and are marked [R]. Facts checked on Wiktionary or Wikipedia are marked [V] (see `house-events.jsonl`).

**O10. "AT" chain (D1).** Four nodes with arcs pairing 1–2 and 3–4.
- [R] The title glyph of "AT I ०": two coupled pairs, i.e. the dual layer of the text (decision pair and causal pair) chained into one line.
- Slot: the two-layer split of PHASE-SPHERE. The observer (Born decider and CTC loop) is one layer; the chart dynamics (unitary steps) is the other.
- **TO10 (layer ablation).** Train with the observer frozen at its initialisation, then train with the chart dynamics frozen.
  - Bar: each ablation raises h=10 npersist by ≥ 10% relative to the full model.
  - Kill: either ablation costs < 2%, which would mean the pair is not coupled.

**O11. Chain m —λ— V —Us— "Teeth" with a return arc y (D2).**
- [R] "Teeeth" reads as "Truth". This is an observation chain: matter m emits at wavelength λ, passes a lens or eye V (the double ellipse), reaches the viewer "Us", and becomes Truth.
- The return arc y from below V back to m is feedback. It matches AETHER's Φ′ = Φ + λ₂⟨Ψ|O₂|Ψ⟩ (O9), where λ is the feedback gain.
- Slot: observer feedback. The decider's gauges u_k get a state-dependent rotation Φ_t driven by the last expectation ⟨ψ_t|O|ψ_t⟩, with O a learned Hermitian operator. This is the "y" arc.
- **TO11.** Compare against the same layer with λ₂ ≡ 0 at equal FLOPs.
  - Bar: h=10 npersist ≤ 0.95× on t2 and so3.
  - Kill: no gain, or a learned λ₂ that collapses to 0 within 10% of training.

**O12. "entropy ∝ Tez" (D3), an axis plot with a curve falling, then flattening.**
- [R] "tez" means fast. [V] The Urdu Wiktionary entry for تیز glosses it "(chiefly) fast, quick" and also "sharp". The Hindi तेज entry as fetched gives only "spirit, vigour", so the Hindi sense is unconfirmed. Reading: entropy grows with the rate of change of the scene.
- The plotted curve is the inter-event interval Δt ≈ ε / |v|, which falls as speed rises. This is AETHER's event-driven rule: render or compute only when accumulated change exceeds ε.
- Slot: event-driven laziness in time. A step runs only when ‖ψ_t − ψ_{last event}‖ > ε_t, and ε_t is learned. This is Taub's lane, recorded here as the origin's reason for it.
- **TO12.**
  - Bar: at equal h=10 npersist (within 2%), executed steps per frame ≤ 0.5 on the overdamped beds.
  - Bar: executed steps correlate with |Δstate| (Spearman ρ ≥ 0.5). This is the "∝ tez" claim.
  - Kill: ρ < 0.2.

**O13. Large sphere (D4).** Nested loops, rays converging on one point, small loops "C", "P", "A", an arrow "my" pointing out, and a mark "∪ ?".
- [R] This is the phase sphere itself. The convergence point is the vortex, where the sector lines of page 2 meet and no single gauge exists. The nested loops are the CTC circles of the Gödel figure, drawn on the sphere.
- "C", "P", "A" read as Causality, Phase, Attention: the three ingredients the text names.
- "my" is the observer's own direction, the query.
- "∪ ?" asks whether the charts' union covers the sphere: the covering question.
- Slot: the vortex is where the readout must switch gauge.
- **TO13.**
  - Bar: the chart switch rate (top-1 change between adjacent states) is ≥ 3× higher within 0.3 rad of the learned ψ's zero-amplitude points (the vortices, |α_k| → 0 for the active chart) than elsewhere.
  - Bar: on the Berry bed there is at least one vortex per gauge.
  - Kill: switching is uniform over the sphere.

**O14. Box of cells with "o Surgery", "cost", "drug (medicals)" (D5).**
- [R] A choice between treatments (surgery or drugs) at a cost. Cells are options, and the outside circle joined to one corner by two lines is the observer committing to one branch.
- In model terms this is multimodal futures: mode choice, the MoP-JEPA problem.
- Slot: a branch bed. A state can evolve into one of two futures depending on a latent coin, and the observer must pick a branch instead of averaging.
- **TO14.** A bimodal bed: overdamped s2 where, at a random 30% of steps, the action sign is flipped by a hidden coin.
  - Bar: best-of-K (K=2) h=1 error ≤ that of MoP-JEPA, and the averaged-prediction error of one-pass routing is ≥ 1.5× the best-of-K.
  - Kill: the PHASE-SPHERE Born decider averages, with best-of-K ≈ mean prediction.

**O15. "π law" with crossing strands (D6).** This is O8, now committed.
- [R] An exchange or loop multiplies the state by e^{iπ} = −1: holonomy π, orientation reversal.
- Slot: the sign holonomy readout (`loop_berry`, `loop_orientation`).
- **Test:** TO7, with the per-class `to7_score.min` ≥ 0.95.

**O16. "JEV".**
- [R] V-JEPA, the video JEPA. The text reads "combination transformer inspired by JEV and JEPA", meaning video and image JEPA.
- [V-negative] No Wikipedia page uses "JEV" for a JEPA. "V-JEPA" redirects to "World model (artificial intelligence)". The reading rests on handwriting alone.
- Slot: none new. V-JEPA-2-AC is already the T5 baseline.
- **Test:** T5's bar applies unchanged; no extra cell.

**O17. "Tsoffran observer".**
- [R] The softmax observer. "T soffran" reads as "softmax" in the same hand that writes "softmax" twice on page 2 ("as self attention have softmax", "a softmax here?").
- The knot drawing therefore labels the two strands "Causality" (the JEPA's forward time) and "softmax observer" (the decider).
- Slot: the Born decider (softmax(2 log|α|), tested). The knot means the two strands are linked: causality feeds the decider and the decider feeds causality.
- **TO17 (linking).** Measure the Jacobian coupling both ways, ‖∂w/∂ẑ‖ and ‖∂ẑ/∂w‖, through the CTC loop.
  - Bar: both are non-zero and within a factor of 10 of each other at convergence.
  - Kill: one direction is < 1e-3 of the other, meaning the strands are unlinked.

**O18. "bound of ICCR PAPER KINDA".**
- [R] "ICCR" reads as ICLR. [V] Wikipedia: "International Conference on Learning Representations", described as "a machine learning conference typically held in late April or early May each year". The knot of causality and observer is "bound" (scoped) to an ICLR-paper-sized claim.
- Slot: scope discipline. Each claim must be decidable by one preregistered cell within the single-GPU budget.
- **TO18.**
  - Bar: every claim C1–C5 and TO1–TO17 has its decisive cell inside tier 1 of the sweep plan, and tier 1 finishes ≤ 24 h on the 4060 at one GPU job.
  - Kill: a claim with no decisive cell gets a VOID verdict.

## Appendix J — `PREREG_DRAFT_hybrid.md`, verbatim (title line dropped; sha256 a393d117f7c1)

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

### J.Matched sizes (L-BR)
- Equal active FLOPs (`active_flops`, including the coherent decider) and equal total params (`n_params`,
  including Cp, Cq and theta_keys) against atlas, phase-sphere and every owned arm, matched with
  `arms/match.py` by chart hidden width.
- Planning uses `time_traveller.rollout_flops` / `budget_samples`. The hybrid also pays the coherent START
  decider and the worst-case loop MACs.

### J.Claims, scored identically for atlas, phase-sphere and hybrid
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

### J.Hybrid-specific bars and kills
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

## Appendix K — `PREREG_DRAFT_phase_sphere.md`, verbatim (title line dropped; sha256 1b5cc362f169)

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

### K.1. Layer (as built)

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

### K.2. Beds added

| bed | state (EVAL ONLY) | obs | dynamics | meta n / atlas / β_Z2 | use |
|---|---|---|---|---|---|
| `berry` | ψ ∈ S³ ⊂ C², stored as [Re ψ₁, Im ψ₁, Re ψ₂, Im ψ₂] | lift(ψ) | n = Hopf(ψ) follows the s2 overdamped walk; ψ is horizontally transported (SU(2) minimal rotation), so ⟨ψ_t\|ψ_{t+1}⟩ is real | 3 / 2 / (1,0,0,1) | TO1 |
| `berry0` | n ∈ S² (the same n_t and actions as `berry`) | lift(n) | the same | 2 / 2 / (1,0,1) | TO1 control (γ ≡ 0) |
| `closed_loops(t2 \| klein)` | held-out closed loops by class: T² a, b, ab, 0 (all +1); Klein a (−1), b (+1), 0 (+1) | the same lift as the bed | — | — | TO7 |

### K.3. FLOPs as implemented

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

### K.4. Readout matrix: each claim on each candidate

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

### K.5. Open items before PREREG commit

1. The FLOP formula as implemented, pasted from the nurse report into §3 after review.
2. N, d_c and loop_iters per bed, confirmed by a smoke run (memory at K = 64 on the 4060).
3. HYBRID's interface for `psi`: needed to fill its TO1 cell.

## Appendix L — `PREREG_DRAFT_event.md`, verbatim (title line dropped; sha256 674c4af81891)

Draft by Taub, 2026-10-04. Not committed. It merges into `PREREG.md` only after Inspector review.

### L.Origin and owner

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

### L.Mechanism (`atlas/event_driven.py`)

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

### L.Contract tests (RED first, `tests/test_event_driven.py`)

- ε = 0 reproduces the always-compute arm exactly.
- ε = ∞ never recomputes after t0.
- Outputs at trigger steps equal the arm run on the hand-compacted subsequence.
- Executed FLOPs = triggers × active_flops, and they are non-increasing in ε.
- The gradient reaches ρ.

### L.Quality metric (pinned so hold steps cannot game it)

`metrics.pred_error` on the event arm's own held latents g̃ is gameable: a hold makes copy-last
perfect, so npersist improves for free. The event arm is therefore scored against the always-compute
latents of the same weights:

  npersist_ev(h) = MSE(ĝ̃_{t+h}, g^full_{t+h}) / MSE(g^full_t, g^full_{t+h})

Here g^full is the wrapped arm evaluated at ε = 0, which is exact by the test above, and ĝ̃ is the event
arm's h-step prediction. The loss of quality is

  ΔQ = npersist_ev(1) / npersist_full(1) − 1

### L.Beds, arms, seeds

- Beds: `s2`, `so3`, and the swing-up bed (`test_swingup.py`'s bed: T²×R², actions), all D = 256 lift.
  The neutral bed `r2` runs beside them (L-WALD).
- Arms: ATLAS (K = atlas(M), m = 2), PHASE-SPHERE and HYBRID at their PREREG configs, each wrapped.
- Baselines:
  - the same arm, always-compute;
  - the same arm with a fixed ε swept over Δ-quantiles {0.25, 0.5, 0.75, 0.9} of the training beds;
  - a skip-RNN-style owned baseline: single JEPA with a learned binary update gate, same cost term.
- Seeds 0, 1, 2. λ_cost and lr are tuned on seed 3.

### L.Bar and kill

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

### L.T7 coupling (systems, µs/step on the 4060)

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

### L.Open for Inspector

- Action handling across gaps: actions between triggers are dropped in v1, and the upgrade is to
  integrate them. This matters on swing-up, where actions are dense.
- Per-chart triggers (chart k recomputes only when its own input deviates) are deferred. Whole-step
  triggers come first, because they are arm-agnostic.

## Amendment 1 (pre-result)
Talk-rubric rules for PLAN item 5 (Adams, `zero/plans/adams.md` step 1). Filed 2026-10-04, before any QF-LM or GPT checkpoint exists. It closes the open cell "PLAN item 5: rubric definition / tie criterion".
- **Frozen sheet.** `zero/ref/prompts_50.json`, 50 prompts. sha256 `ff9a955c3a864d7fa9b4daca3bbae308b524fddffe65c7182986d892de264c7e`, computed over `json.dumps(prompts, ensure_ascii=False)`. This is the rule in `eval_talk.py:21` and the sheet's own "note" field; verified by Wilson. It is not the file's byte hash, which depends on line endings.
- **Tie.** Five samples per prompt per model. The 90% prompt-bootstrap confidence interval of the win rate must lie inside [0.40, 0.60]. This must hold on every one of the 3 seeds.
- **Rater agreement.** The rubric counts only if Krippendorff α ≥ 0.667.
- **Null control.** GPT vs GPT with a different seed must tie under the same rule. Otherwise the rule is void.
- **Reported:** rater side-guessing (can raters tell which model wrote a sample) and train-text overlap of the samples.
- **Wording.** No "talks like a person" wording in any verdict or report.

## Amendment 2 (pre-result)
Filed 2026-10-04, before any PLAN item 5 result exists. It changes PLAN item 5's "new" criteria and adds two pins.
- **Criterion (c) replaced.** "Memory flat in context length" is true by construction for QF-LM (the state is one ψ ∈ C^N) and equally true of the GRU baseline, so it cannot show newness.
  - It is replaced by Adams's **recall-vs-distance curve**: recall of a fact as a function of the token distance since it was stated, for QF-LM vs GPT vs GRU, same beds and seeds.
  - It is reported as a measurement, with no bar. "New" now requires (a) and (b) only.
- **Criterion (b) tightened (Adams).** Think steps are credited only if forcing s = 0 (no think steps) raises val loss. This is in addition to the existing Spearman ≥ 0.3 and adaptive-beats-fixed conditions. If forcing s = 0 does not raise val loss, (b) fails.
- **Completeness bar for the instrument (Chase).** |Σ_v p(v) + p_null − 1| ≤ 1e-6 in float64 and ≤ 1e-5 in fp32, every step. This replaces the single "± 1e-6" of PLAN item 3 for fp32 runs.
- **TO7 control row (Cameron).** The owned w₁-Jacobian baseline (`arms/w1_jacobian.py`: sign of learned chart-transition Jacobians multiplied along loops, after arXiv 2602.22873; contracts in `tests/test_w1_jacobian.py`) is scored on the same held-out loops. Bar and per-class aggregation are as TO7. A candidate part's TO7 pass is credited only if it is at least as accurate as this control on every loop class.
- **Coherence controls (Foreman's finding).**
  - QF-LM's per-step head is classical: the effects K_v†K_v = Q†diag(M_v)Q all commute. Wilson verified this: `tests/test_facts.py::test_diag_q_head_effects_commute`, commutators < 1e-12. Any gain must therefore come from coherence carried across steps.
  - Two deciding control arms are added to item 5, at equal FLOPs and the same seeds:
    - **dephased-HMM**: ψ dephased to the diagonal (Q basis) each step;
    - **real-ψ**: the state is restricted to real amplitudes.
  - **"New" additionally requires** QF-LM to beat the dephased-HMM arm by > 2% val perplexity on ≥ 2/3 seeds.
- **Fact for the record (Wilson, OWNERS.md §F).** arXiv 1711.01416 (Pestun, Terilla, Vlassopoulos 2017) owns "write = read" on language in formalism: the word is a measurement operator and the phrase is the operator product. So criterion (a) measures whether write = read pays; it does not make write = read new.

## Amendment 3 (pre-result)
Filed 2026-10-04, before any PLAN item 5 result exists. It adds two pre-checks and one kill test (Thirteen, `plans/thirteen.md` H1, H3; Park). A pre-check's outcome changes sizes only, never a bar.

- **MM-QFA regular-language kill test (Thirteen H1).**
  - Bed: synthetic regular languages, CPU:
    - parity;
    - "last symbol was a";
    - "a then b within k";
    - Dyck-1 at depth ≤ 4.
  - Arms: QF-LM, QF-LM + embed, GRU, dephased-HMM, at equal FLOPs; 3 seeds.
  - **Bar:** QF-LM is within 0.02 bits/symbol of the true entropy rate on all four languages.
  - **Kill:** QF-LM fails on any language while the GRU succeeds. Write = read is then dead, or needs a classical register.
  - **Fact check (Wilson).**
    - [V] Yamakami 2004 (arXiv quant-ph/0410040, citing Kondacs–Watrous 1997 [32]): "no 1qfa recognizes the regular language Zero" = {x0 | x ∈ {0,1}*}, "and therefore, 1QFA cannot contain REG". "Last symbol was a" is exactly this language.
    - [K, not fetched] Brodsky–Pippenger give a group-theoretic characterisation of measure-once 1QFAs; parity is a group language.
    - [D] QF-LM is not literally a 1qfa: its think unitaries depend on ψ, which makes the map nonlinear in ψ, and it is a generator scored by likelihood, not a bounded-error acceptor. The theorem therefore motivates the test but does not predict its outcome.
- **H3 Zipf-head pre-check (Thirteen).**
  - Procedure: on TinyStories BPE-4096 validation, using a trained small GPT, count:
    - the positions whose top-1 p ≥ ½;
    - the number of distinct tokens at those positions.
  - Reported, no bar. It informs N only.
  - Fact basis: at most 2N outcomes can ever reach p ≥ ½ (`tests/test_facts.py::test_instrument_max_mass_bounded_by_dimension`).
  - Size rule: if the distinct count is ≤ 2N for a smaller N than PLAN's N ≥ 2048, that N may be used, fixed before the bake-off.
- **Holevo capacity pre-check (Park).**
  - Fact: QF-LM's only channel from past to future is one pure ψ ∈ C^N, read by instruments. Any sequence of instruments composes into one instrument (one POVM on the record). So the information the future can extract about the past is I ≤ S(ρ) ≤ ln N nats = log₂ N bits.
  - Evidence:
    - [V] Wikipedia "Holevo's theorem": "the accessible information between X and Y is bounded by the Holevo information"; "given n qubits, only n classical bits of information can actually be retrieved".
    - Bound by `tests/test_facts.py::test_holevo_one_instrument_carries_at_most_ln_N_nats`, including the saturating case.
  - Procedure: before item 5, measure a small GPT's NLL saving on the second half of 200 TinyStories validation documents from seeing the first half. If the saving exceeds the threshold, grow N or move to a mixed-state ψ before the bake-off.
  - **Fact check on the threshold.** The relevant per-document cap for a pure state in C^N with no shared entanglement is **ln N nats**, not 2·ln N. The factor 2 belongs to entanglement-assisted (superdense) coding, which QF-LM does not have.
    - The threshold is therefore pinned at **ln N nats**, which is stricter than 2·ln N.
    - Park's exact definition of the "NLL saving" (per document vs per token, which model, which cut) is **NOT RUNNABLE until a pre-result amendment** pins it.

## Amendment 4 (pre-result)
Filed 2026-10-04, before any PLAN item 5 result exists. It pins Park's NLL-saving definition. This closes the open cell "Park's exact definition of the NLL saving" in Amendment 3, so the Holevo pre-check is runnable.
- **Data:** 200 TinyStories validation documents, BPE-4096 tokenizer, each truncated to 256 tokens. Split at L = 128: the first half is tokens 1–128, the second half tokens 129–256.
- **Model:** the GPT arm.
- **Statistic per document d,** in nats, summed over the 128 second-half tokens:
  S_d = NLL(second half | fresh context, no first half) − NLL(second half | first half in context).
- **Uncertainty:** bootstrap confidence interval of the mean S_d, resampling over documents.
- **Rule:**
  - If the CI lower bound is > ln N nats (Amendment 3 threshold), move to the N-scaling or mixed-state variant before item 5.
  - Otherwise item 5 proceeds at the pinned N.
  - This changes sizes only, never a bar.

## Amendment 5 (pre-result)
Filed 2026-10-04, before any arm has run on these beds. Source: Thirteen, `zero/plans/amendment3_draft.md`, which is copied there in full. **It supersedes Amendment 3's MM-QFA kill-test bars.** Amendment 3's H3 and Holevo items stand.

- **Sources.**
  - [V, PDF text read by Wilson] arXiv quant-ph/0009004 (Ambainis, Kikusts, Valdats), Theorem 1.1, citing Kondacs–Watrous 1997:
    - "All languages recognized by 1-way QFAs are regular";
    - there is a regular language that cannot be recognised by a 1-way QFA with probability ½ + ε.
  - Theorem 1.2 (Brodsky–Pippenger 1999), same paper: the minimal-DFA forbidden construction rules out recognition by a measure-many 1-way QFA.
  - [V, Thirteen] arXiv quant-ph/9802062 (Ambainis–Freivalds), Theorem 2 and Corollary 4: above correctness 7/9, 1-way QFAs equal 1-way reversible automata; a*b* is recognised only with probability ≈ 0.68.
  - The Kondacs–Watrous 1997 primary PDF was **not** obtained; it is cited at second hand only.
  - These proofs use linearity and unitarity. QF-LM's renormalised Kraus update, √-weighted null records and ψ-dependent think need not obey them, so the test decides empirically.
- **Bed** (`beds/reglang.py`). Alphabet {a, b} with BOS/EOS, exact true log-probabilities, length cap 64, train/val/test 20k/2k/2k from generator seeds 0/1/2, data hash pinned. Generators as in the draft's table:
  - **parity**: control only;
  - decisive **last-a** = (a+b)*a;
  - decisive **a\*b\***;
  - decisive **k-gap, k = 3**;
  - decisive **Dyck-1 at depth ≤ 4**.
- **Arms.** All at equal budgets, 3 seeds, tuning seed 3; Adam, 5,000 steps, batch 64; one learning rate per arm from {1e-3, 3e-3, 1e-2} on seed 3.
  - **Deciding:**
    - QF-LM at N ∈ {8, 16, 32};
    - QF-LM + W_v (below);
    - dephased-HMM, which joins when `qflm/controls.py` exists and is NOT RUNNABLE until then;
    - GRU (hidden 16).
  - **Reported only:** GPT (2 layers, d = 32), QF-LM nothink, QF-LM nonull.
- **Honest +embed arm (Kutner, `zero/plans/kutner.md` items 2, 3).**
  - Under K_v = diag(√M_v)Q = Q·√F_v, every token shares W = Q.
  - The "+embed" arm is therefore **K_v = W_v diag(√M_v) Q**, with W_v a token-dependent unitary (low-rank butterfly per token). p(v) is unchanged; only the write differs.
  - This replaces `nn.Embedding` as the +embed arm here and in item 5's criterion (a).
- **Metric.** Excess entropy ε = (Σ model NLL − Σ true NLL) / (number of scored symbols), in bits per symbol, on the test split. Every arm is scored on the same positions after BOS, including EOS. QF-LM also scores the BOS-predicted first position; this bias against QF-LM is recorded.
- **Bars and kills.**
  - **Pass** on a language: ε ≤ 0.02 bits/symbol on ≥ 2/3 seeds.
  - **Validity:** the GRU must pass a language, or that language is VOID for every arm.
  - **QF-LM fails** a language: ε > 0.05 on ≥ 2/3 seeds at every N ∈ {8, 16, 32}, while the GRU passes.
  - **Kill (write = read):** QF-LM fails any of last-a, a\*b\*, k-gap or Dyck-1. Verdict line: "write = read is dead on regular languages, or needs a classical register".
    - If +W_v passes where QF-LM fails, the missing piece is the write path.
    - If dephased-HMM passes, coherence hurts.
  - **Survive:** QF-LM passes all four decisive languages at some N ≤ 32.
  - Parity failure is reported as an optimisation failure, never as a capacity verdict.

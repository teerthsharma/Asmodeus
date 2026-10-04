# PREREG DRAFT: owned arms (L-BR)

Draft by Cameron, 2026-10-04. Not committed. Merges into `PREREG.md` only after Inspector review. Every
number below is a sizing computation (`arms/match.py`, meta device, no training), never a result.

## 1. Arms

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

## 2. Arm list per test

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

## 3. Matched sizes against the atlas (D = 256 lift, D_g = 2n+2 per bed (6 for n=2, 8 for so3), d_h = 64, d_key = 32, hidden = 32, m = min(2, K))

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

## 4. MoP-JEPA deviations from arXiv 2607.05238v1

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

## 5. Tuning (seed 3 only; same budget for every arm, the atlas included)

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

## 6. Open for Inspector

- In train mode, the PEER BatchNorm query statistics pool over B·T tokens, so they mix time steps. Eval-mode causality is tested; train-time leakage is batch statistics only and carries no state.
- In chart-AE / lazy chart-AE rollout, membership is re-read from the predicted g. This is valid because the fidelity term places g in E's space, but it is untested on a bed.

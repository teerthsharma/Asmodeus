# PREREG draft: OBSERVER-ATLAS and the shared anti-collapse table (Chase)

Code: `atlas/observer_atlas.py`, `sigreg.py`, `common.py`. Tests: `tests/test_observer_atlas.py` and `tests/test_sigreg.py`. Every pin below is bound by a test named next to it.

## 1. Shared anti-collapse table, the same for every arm (fairness)
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

## 2. Pins common to every arm
- **EMA τ = 0.996.** EMA copies are excluded from `n_params`: arms exclude children named `tgt_*`; the atlas excludes parameters with requires_grad False. L-BR parameter matching uses `n_params()`.
- **TF32 off.** `common.py` sets `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False`. It is imported by `arms/__init__.py` and the atlas. Test: `test_tf32_pinned_off`.
- **Determinism.** Bitwise on CPU under a seed (`test_determinism_under_seed`). On CUDA, the SIGReg scatter uses atomics (nondeterministic at about 1e-7); seeds are compared with a tolerance there.

## 3. OBSERVER-ATLAS design pins
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

## 4. Topology readout (C2)
- **Nerve source.** `nerve(w_soft, τ, max_dim=3)` on the pre-top-m softmax over all K, multiplied by the gate so that closed-gate inputs add no simplices. The top-m nerve (`w`) is an ablation only, because its dimension is at most m − 1.
- **τ grid:** {1/4, 1/5, 1/6, 1/8, 1/10, 1/12, 1/16, 1/24, 1/32}, as in PREREG.md §T2. The reported vector is `metrics.most_persistent`. Betti numbers up to b3, so max_dim = 3 for SO(3).
- **K for P-BETTI.** P-BETTI is scored only at K ≥ the minimal triangulation size of each manifold: S² ≥ 4, T² ≥ 7, Klein ≥ 8, RP³ ≥ 11.
  - In practice that means K ∈ {16, 64} (PREREG.md conflict row 6).
  - P-COUNT is scored at K = atlas(M), where a nerve with correct Betti numbers is impossible: two charts on S² give a nerve that is a single edge.
- **Integration.** Atlas `nerve` matches `metrics.betti_z2` and `nerve_betti_z2` (`test_nerve_integrates_with_metrics_betti`).

## 5. Known limits, reported and not tuned away
- **Atlas sits near the edge of the λ window.** It blew up at λ = 0.75 and is stable at 0.9 and 0.95.
- **The scale contract is a 200-step smoke test, not a convergence guarantee.**
- **Adam momentum still updates unselected chart rows.** Their gradient is exactly zero, so a dead chart drifts but does not learn. Load balance is the only revival path; it reaches every key (`test_grad_reaches_selected_charts_only_and_all_keys`).
- Origin rule (`zero/CLAUDE.md`, `ref/origin/`): the JEPA-shaped losses here (per-chart predictor, EMA target, SIGReg) are the current vocabulary, not a constraint. Any origin-driven redesign of the chart objective is a new arm and must clear the same scale contract (§1) before comparison.

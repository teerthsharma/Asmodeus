## Amendment 11b (pre-result)
Filed 2026-10-11, before any FCPA evaluation result on seeds 0, 1, 2, 4, 5 under the lag-stack input exists. The seed-0 pilot under the single-state input is disclosed in `PILOT_DISCLOSURE.md`. Appended to Amendments 11 (`FREEZE.md`) and 11a (`FREEZE2.md`); neither is edited. Where this text differs from them on the input, the output bins or the 11a sizing table, this text wins.

**What changes**
1. **Input (all arms).** Every arm (FCPA, Global, ASMODEOUS top-m) receives the same lag-stacked row `[x_t, x_{t−1}]`, 2 × 90 = 180 columns, and predicts x_{t+1} (90 columns). The first row of each trajectory is padded with x_0. In the closed-loop rollout each prediction is pushed back into the stack. FCPA is fitted with `n_lags = 2`. All arms get identical information, so the comparison stays one of architecture.
2. **FCPA output bins.** FCPA predicts increments (`y = None`, prediction x_t + Δ). Its bins come from `FCPARegressor.make_step_edges(X_train, n_lags=2)`, not from absolute-state edges.
3. **Reference columns.** Persistence (x_t) and constant velocity (2x_t − x_{t−1}) are reported per seed beside the bars. They have no pass/fail value and no entry in the verdict.
4. **11a table.** The table in 11a is the sizing at the single-state pins and is superseded as a sizing. Under the 180-wide input, Global stays parameter-matched to FCPA. Every MAC and parameter count in the report is the per-seed value recomputed by `run_primary.fcpa_train_macs`, `baselines.global_macs` and `Asmodeous.active_flops`. The C2′ rule (pinned budgets, MACs reported, not equalised) and C1 are unchanged.

**What does not change.** P1 to P4, C1, C2′, the beds, the eval seeds (0, 1, 2, 4, 5), the tuning seed (3), the 128/64/64 split, T = 25,600, the bootstrap, the Holm family S1 to S5, kill rules K1 to K9, and every pinned constant.

**Why.** On tuning seed 3 the single-state estimator equals persistence exactly (one-step relative error 0.030169 for both), while constant velocity reaches 0.0048 on a 40-trajectory sample. The state alone cannot carry the velocity: the pinned bed is second order in time as observed. The row-wise `predict` contract of scikit-learn (ERRATA 3) admits history only through the features. This is a seed-3 and synthetic-smoke fact; no eval-seed number enters it.

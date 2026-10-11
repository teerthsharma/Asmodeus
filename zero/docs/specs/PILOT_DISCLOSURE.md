# Pilot disclosure (pre-result)

Filed 2026-10-11, before any FCPA evaluation run. This file records every development run that touched an eval-seed bed (seeds 0, 1, 2, 4, 5). Nothing here is an evaluation result. It is listed because the numbers were seen, and estimator work after them may have been influenced by them.

## Runs on eval-seed beds

**Mid run (eval seed 0, 128-trajectory bed).** `evaluate(0, ...)` with `n_traj = 128`, L = 400, burn-in 200, bed data seed 0, split by `three_way_split(128, seed=0)` into 64 Train / 32 Cal / 32 Test (T = 12,800). Tuning was not run for it; the ASMODEOUS arm used a placeholder lr = 3e-3, m = 2. Wall time 208.9 s. Every number seen:

- FCPA: 8 learned eras, 83,232 parameters, training 1.40e9 MACs (1,401,949,460), inference 3.75e5 MACs/step (374,662.6).
- E50 on the seed-0 Test split: FCPA 696.5943, Global 1.9988, ASMODEOUS 0.9521. Within-era origins: 700.1520 / 2.0382 / 0.9519. Origins crossing a switch: 685.5904 / 1.8767 / 0.9526. Non-finite rollout rows: 0 / 0 / 0.
- P3 identification score 0.1384. ACI coverage deviation 0.0003125. Mean |C| 7.8508.
- Parameters: Global 83,250 (width by integer search), ASMODEOUS 16,282. Inference MACs/step: Global 158,220, ASMODEOUS 15,424. Training MACs: Global 1.805e9, ASMODEOUS 1.386e9.
- C2 bookkeeping: budget 1,401,949,460; Global afforded 527 of 25,536 Train pairs (925 used, the floor F + 1); ASMODEOUS 55 steps and 17,415 probe pairs.

**Diagnostic on the same cached seed-0 bed** (FCPA refit on the same 64 Train trajectories, scored on the same 32 Test trajectories):

- One-step relative error: FCPA 0.6889; persistence (x̂ = x_t) 0.03199; Train-mean predictor 0.9880.
- FCPA per top-scoring era, own-era one-step / TT-alone one-step: era 2 (249 rows) 0.5315 / 0.2395; era 5 (6 rows) 0.3865 / 0.1727; era 7 (12,513 rows) 0.5560 / 0.3105.
- FCPA closed-loop relative error at horizon h = 1..5 from origin 20: 0.7222, 0.7696, 0.8256, 0.9358, 0.9756.

**Smoke runs (eval seeds 0 and 1, tiny beds).** `--smoke` as first built used eval seeds 0 and 1 with 8 trajectories per bed, L = 110, burn-in 10, 4 origins, T = 220 (P4 VOID). Seen:

- Tuning on seed 3 (bed 2003), validation E50 for lr 1e-3 / 3e-3 / 1e-2: 1.2029 / 1.3286 / 1.3394.
- Seed 0: 1 era; E50 FCPA 22.7557, Global 4.0916, ASMODEOUS 1.6855; P3 0.000; coverage deviation 0.00909; mean |C| 1.00.
- Seed 1: 1 era; E50 1.3847 / 1.4600 / 1.0465; P3 0.588; deviation 0.00909; mean |C| 0.94.
- Smoke P1 rho 4.3484 (95% upper 5.2576), P2 rho 8.8361.

The smoke runner now uses non-eval bed seeds 7001 and 7002.

## Influence

The seed-0 diagnostic (FCPA one-step error far above persistence; no residual path in the estimator) was reported to the coordinator and motivates a revision of `estimator.py`. Any such revision is therefore informed by one eval-seed bed. The pinned seed-0 bed of the evaluation (256 trajectories) shares the seed and the per-trajectory seeding with the pilot, so its first 128 trajectories are the same trajectories as the pilot bed; the Train/Cal/Test assignment differs because it is drawn over 256 trajectories.

## Rule from here

All development runs, tuning and smoke tests use tuning seed 3 (bed seed 2003) or synthetic non-eval seeds. `run_primary` refuses `--smoke` on an eval seed and refuses a full run on a non-eval seed.

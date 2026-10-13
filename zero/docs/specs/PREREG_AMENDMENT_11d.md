## Amendment 11d (pre-result): the L37 era model
Filed 2026-10-13, before any FCPA evaluation result on seeds 0, 1, 2, 4, 5 exists under the lag-stack input. No eval-seed bed was read for this amendment. Appended to Amendments 11 (`FREEZE.md`), 11a (`FREEZE2.md`), 11b (`FREEZE3.md`) and 11c (`FREEZE4.md`); none is edited. Where this text differs from them on how an FCPA era predicts the next state, this text wins.

**What changes: the model an era answers with.** The pins are `run_primary.FCPA_PINS = dict(damping=1.0, law="ns2d", era="quad_vq", era_lam=0.1, era_pool=inf, energy=True)`.
1. **Features (L37).** An era no longer predicts from φ = Q[x; 1] in its chart. It predicts the increment of the newest block from z = [x_t, x_{t-1}, q(x_t)] / s, built on the 80 independent coordinates of each block (40 complex modes; the (0, 0) mode is zero and the ky = 0, kx < 0 modes are conjugates). Here q is an orthonormal basis of Sym² (3,240 quadratic terms), and s is one scalar: the root mean square of the independent coordinates over the Train rows, fixed before fitting, the same way as the step edges (`symmetry.ModeFeatures`).
2. **Symmetry (L37 Q2, R1).** The ridge is restricted to the maps that commute with the bed's O(2) symmetry: x-shifts and the rotation by π. The era keeps the statistics of the data as seen (Chan-merged, so `partial_fit` equals `fit`, L27′). The solve reads their group average, which equals the statistics of the data augmented by all 34 elements of Γ17 = 17 shifts × {1, C}, each weighted 1/34. `symmetry.average` computes it in closed form from the charge of every feature. It matches the 34-image augmentation to machine precision, and the result is O(2)-equivariant, not only Γ17-equivariant (`tests/test_symmetry.py`).
3. **Ridge.** The penalty is λ n ‖W‖² with λ = 0.1 on every feature but the intercept. There is no rank guard and no λ cut: λ is a pin.
4. **One shared dynamics (era_pool = inf).** The coefficient map W is the solution on the pooled statistics of all eras (exact Chan merge). Each era keeps its own intercept (its group-averaged mean increment minus Wᵀ its mean feature). The eras, the law test, the router, the conformal sets, ACI and the skill gate `_shrink` are unchanged.
5. **Energy projection (energy = True).** In closed loop, an answering era's next newest block is projected onto the sphere |x_{t+1}| = g_k |x_t|. Here g_k is the era's mean norm ratio |x_{t+1}| / |x_t| over its training windows, an additive statistic that already exists (`FCPARegressor(energy=True)`, unchanged). A query no era answers is unchanged.

**What does not change.** P1 to P4, C1, C2′, the beds, the eval seeds (0, 1, 2, 4, 5), the tuning seed (3), the 128/64/64 split, T = 25,600, the lag-stack input of 11b, the calibration of 11c, the bootstrap, the Holm family S1 to S5, kill rules K1 to K9, and every other pinned constant, including α = 0.1, γ = 0.05, w = 20, K_max = 8, `_TAU` = 1.7, `_MIN_GAIN` = 0.2 and the damping pin 1.0. C2′ counts the new era's training MACs analytically (`run_primary.fcpa_train_macs`: per-row feature statistics, the per-solve group average and solve).

**Why.** Before this amendment, FCPA's closed-loop 50-step error was worse than persistence on every run (11c comparison; L28: an undamped linear era cannot beat persistence). L37 identifies the gap: the bed's advection is quadratic in the state, and an unconstrained quadratic ridge on 3,400 features is unstable. Restricted to the symmetric maps, its effective dimension drops and the penalty is invariant.

**Exploratory comparison behind the choice (not evidence for any bar).** All runs used the tuning bed (seed 3, bed seed 2003) and non-eval seed 7001 at full size. Each candidate was fitted as the primary run fits it (`fit_fcpa`: Train, then Cal calibrates) and scored on the Test split (`fcpa/explore_fcpa_era.py`; eval seeds are refused). E50 / persistence E50 and the share of Test trajectories that beat persistence, seed 3 / seed 7001:

| candidate | E50 ratio | wins | one-step error | diverged rollouts |
|---|---|---|---|---|
| φ era (pins before 11d) | 1.37 / 1.55 | 0.11 / 0.06 | 0.0075 / 0.0083 | 0 / 0 |
| φ era + energy | 1.35 / 1.54 | 0.09 / 0.06 | 0.0086 / 0.0090 | 0 / 0 |
| linear V_Q (λ = 1e-3) | 0.94 / 1.00 | 0.72 / 0.56 | 0.0137 / 0.0112 | 0 / 0 |
| linear V_Q + energy | 0.97 / 1.01 | 0.66 / 0.56 | 0.0142 / 0.0115 | 0 / 0 |
| quadratic V_Q per era | 135 / 198 | 0.67 / 0.70 | 0.0117 / 0.0110 | 27 / 58 |
| quadratic V_Q shared | 2.70 / 44.9 | 0.84 / 0.75 | 0.0119 / 0.0107 | 12 / 13 |
| quadratic V_Q per era + energy | 0.82 / 0.74 | 0.88 / 0.89 | 0.0125 / 0.0116 | 0 / 0 |
| **quadratic V_Q shared + energy (11d)** | **0.72 / 0.70** | **0.97 / 0.89** | 0.0125 / 0.0113 | 0 / 0 |

Persistence E50 is 0.6585 / 0.6099 and its one-step error is 0.0302 / 0.0278. Partial pooling (a ridge toward the shared map at strength 0.1 or 1, without energy) diverged like the per-era fit. A standalone single-model exploration on seed 3, with Cal as validation (`fcpa/explore_era_model.py`), showed the same ordering: quadratic V_Q at λ = 0.1 had E50 0.495 against persistence 0.634, while the unconstrained quadratic ridge was no better than persistence at any λ. The quadratic era alone has the best one-step error but diverges in closed loop on a minority of origins. The energy projection alone does not rescue the φ era. Together they beat persistence on both seeds. The pins were chosen on these two seeds' exploratory numbers, not on any bar's value on an eval seed.

## Limits
- The energy projection uses the era's mean norm ratio. It bounds the rollout's growth and is not a physical conservation law of the forced, damped bed.
- With era_pool = inf, the eras differ only in intercept, dissipation rate, law statistics and calibration. The dynamics map is shared. Per-era maps diverged in this comparison.
- Two exploratory seeds; no claim on the eval seeds is made here.
- The damping pin stays 1.0. Damping 0.9 was not compared under the energy projection.

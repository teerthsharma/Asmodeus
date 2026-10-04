# PREREG_DRAFT_beds.md — T2 beds, metrics, calibration (Foreman, 2026-10-04)

Draft for `PREREG.md`. No arm has been trained or scored. The calibration below uses only arm-free latents: the embedded state, PCA of the observations, an atan2-type chart, a projection and a constant. Code: `beds/__init__.py`, `metrics.py`, `run.py`, `calibrate.py`. Tests: `tests/test_beds.py`, `tests/test_metrics.py`, `tests/test_topology.py`, `tests/test_run.py`.

## 1. Beds

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

## 2. Metrics (`metrics.py`): each one fails a constant latent (tests)

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

## 3. Calibration: eps / delta (calibrate.py, eval bed seed 1000, 4000 of 8192 points)

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

## 4. Other numbers

- **tau grid**: 0.05, 0.10, …, 0.50.
  - It contains τ ≤ 1/3, which β₂ needs, and τ ≤ 1/4, which β₃ needs (Wilson facts §5).
  - τ is frozen on the tuning seed at the middle of the longest constant-Betti run. The flag variant is frozen separately.
- **Data**: train n_traj = 512, T = 64, window 32. Eval n_traj = 128, T = 64. Readout-tuning bed n_traj = 128, T = 64.
- **Bed seeds**: train data seed s, eval 1000 + s, readout tuning 2000 + 3.
- **Optimisation**: batch 64, AdamW (lr 1e-3, wd 1e-2), 20,000 steps.
  - lr is chosen from {3e-4, 1e-3, 3e-3} on tuning seed 3 only.
- **Arm seeds**: 0, 1, 2; tuning seed 3, never evaluated (`evaluate` asserts this).
- **Records**: every run record carries `mapper.{n_samples, cover, tune_seed}`, `nerve_n_samples`, `seeds`.

## 5. Spec flaws these beds and metrics expose (for House / Inspector)

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

## 6. Render injectivity probe

Setup: raw pixels used as the latent, 16 trajectories × 24 frames, seed 7.

- **No folds found.** fold_rate (far = 1.4, fold_delta = 0.05/0.1/0.2) is 0/0/0 for both so3 and t2. The sample is small, so this is weak evidence of injectivity.
- **Pixel L2 is poorly conditioned.**
  - discontinuity_rate (eps = 0.25, delta = 0.5) is 0.83 for so3 and 0.92 for t2.
  - The cause: thin links stop overlapping after a small motion, so pixel distance saturates near the median almost immediately.
  - Render beds therefore test pixel-to-manifold learning, not chart topology. Keep lift as the primary bed, and do not run disc/fold on raw pixels.

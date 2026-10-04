# PREREG draft: T5 (time traveller) and T6 (contextuality)

Draft for merging into `PREREG.md`. Nothing here is a result. Calibration numbers come from the bed alone
(no arm involved) and are filled in before PREREG is committed; they decide VOID conditions, never bars.

## T5 bed: `beds/swingup.py`

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

### Calibration (bed only, fill before commit)

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

### VOID conditions (bed-level, decided before arms run)
- Oracle ceiling (row 3) < 0.80, which means the bed is unsolvable at this budget.
- The chosen shared energy with a perfect model scores <= floor + 0.05, which means the bed cannot be
  read by the energy.
- After the runs: the best owned arm is >= ceiling - 0.05 (no headroom), so the margin is VOID, not a kill.

### Calibration decision (bed only, taken before any arm trains)
At `cem_plan` sampling (800/32/10, cold) with true dynamics, the running height cost scores 10/10 and the
terminal L1 energy scores 3/10. The terminal energy, not the sampling or the warm start, is the binding
limit. **The shared T5 energy for every arm is therefore the running latent energy**
E(a) = mean_h ||g_h - g_goal||_1 (`t5_plan.cem_running` / `LatentMPC`, cold start, `cem_plan` sampling,
n_samples set by `budget_samples`). The terminal-L1 result (`arms/planner.cem_plan`) is reported beside it
as the owned V-JEPA-2-AC protocol, with no bar.

## T5 second bed: OGBench pointmaze
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

## Planning protocol (all arms)

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

## What counts as "reading predicted futures" fairly (§4.7 vs single-JEPA MPC)

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

## T6: contextuality (measurement only, `t6_contextuality.py`)

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

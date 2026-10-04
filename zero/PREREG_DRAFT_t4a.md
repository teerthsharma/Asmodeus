# PREREG DRAFT — T4a (representation collapse under weakened anti-collapse)

Status: DRAFT, not preregistered. Nothing has been run. Runner `run_t4a.py` (`--dry-run`, `--summarise`); tests `tests/test_t4a.py`. Run with system python (mujoco 3.11, the T2 beds).

## 1. Claim and question
- C4 (PROMPT §6): observer competition reduces representation collapse when anti-collapse is weakened.
- T4a asks one question: when SIGReg is weakened, does OBSERVER-ATLAS lose rank more slowly than a single JEPA?

## 2. Grid
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

## 3. Measurements, every 50 steps, on the eval bed
- `g_erank`: effective rank (`metrics.effective_rank`, Roy–Vetterli exp-entropy of singular values) of g over all eval frames.
- `chart_erank_min`: minimum per-chart effective rank of z_k. Only charts with ≥ max(2·d_c, 1% of rows) rows count, so an unused chart is not scored as collapsed. Single and MoP have one chart.
- `g_std`, `chart_std_min`: root mean per-dimension variance.
- `gate_mean` (atlas only).
- loss.

## 4. Definitions
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

## 5. Hidden variables (why a number here could lie)
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

## Pre-commit calibration (baseline only)
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

### Recalibration: online-target stressor (coordinator ruling)
- **Ruling.** The PRIMARY decisive cell is SIGReg scale 0 with an online target: EMA momentum 0, so the target is the current online encoder under stop-grad. It applies identically to every arm through `build_arm(target="online")` (τ = 0 for single, MoP and chart-AE; `ema` = 0 for the atlas). SECONDARY is scale 0 with the EMA target (the calibration above). The VOID rule is unchanged. The runner adds `--targets ema online`; online keys end in `__online`.
- **Run.** single only, s2, tuning seed 3, 5000 steps, 4 torch threads, one process. Peak RSS 0.91 GB; 4.4 GB was available at launch. `run_t4a.py` sha256 prefix 39616edc.
  Command: `python -X faulthandler cal_online.py`, which calls `run_job(dict(arm="single", bed="s2", scale=0.0, target="online", seed=3, ...), "uncommitted", steps=5000, log_every=50)` and writes `<scratchpad>/t4a_cal/single__s2__lam0__s3__online.json`. Wall time 347 s.
- **Outcome: censored, no collapse.**
  - t_collapse_g = 5000 and t_shrink_g = 5000, both censored.
  - g_erank went from 4.20 to a minimum of 1.998 at step 200, then back to 3.67 at step 5000.
  - g_std went from 0.035 to a maximum of 3.61, then to 2.30. The latent blew up in scale; it did not collapse.
  - Without the EMA lag, the predictor-asymmetry route (no SIGReg) did not collapse a 6-d single JEPA on s2 within the budget. Its minimum erank of 2.0 stayed above THR = 1.5.
- **Effect.** By the VOID rule, the primary cell is VOID on s2 unless single collapses on ≥ 3 of 5 eval seeds; the tuning seed predicts it will not. so3 is uncalibrated: the ruling allowed s2 only. The min-erank-over-curve difference stays descriptive. The stronger classic collapse setting, with no predictor and no target asymmetry, is not proposed, because it would not be a JEPA.

## 6. Order
Pin widths (match.py) → calibration done (single censored; VOID wording above) → commit PREREG → `run_t4a.py --procs 3`. That is 4 arms × 3 beds × 3 scales × 5 seeds = 180 jobs.

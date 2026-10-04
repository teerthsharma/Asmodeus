# PREREG draft: Arm Q (coherent observer)

Code: `atlas/coherent_observer.py` (`CoherentAtlas`, a subclass of `ObserverAtlas`; Chase's file is not
edited). Tests: `tests/test_coherent_observer.py`, 11/11.
Recorded spec change (folder rule 2, origin wins): PROMPT.md §7 dropped the Born/Lüders decider. The origin
notes say "small phases all around the sphere but the decider will be a softmax", so the decider is
restored as a Born rule over phased chart amplitudes. Softmax stays as its decoherence limit.

## Mechanism
- Softmax router = Born rule on a decohered state, p = |psi|^2 with psi = sqrt(softmax(s)). It obeys Luce IIA.
- Arm Q: psi_k = sqrt(softmax(s))_k e^{i theta_k(q)}, p = |U psi|^2, with U = Cayley(P Q^T - Q P^T) real
  orthogonal of rank 2r. Woodbury gives O(K r) per query and no KxK matrix is ever built (tested at
  K = 200,000). theta_k(q) = (W_q c) . theta_key_k.
- Measurement = top-m of p + the parent's gate. These are the nonlinear, linearity-breaking steps.
- Limits, all tested: P = 0 reproduces the exact softmax for any theta (the init). theta = 0 gives the real
  sign-only model. grad_P != 0 at P = 0, so coherence is learned and never given (L-GIVEN). The reference
  `born_gibbs` reproduces identities §6: p1/p2 = 1.3499 / 1.5165 / 2.0587 at g = 0 / 0.5 / 1.

## Hypothesis
Under softmax IIA, near-duplicate charts split weight (two equal keys each get half). Their top-m slots
are wasted and spurious co-selections show up in the nerve. Coherent coupling lets duplicates interfere
destructively, so one survives. Prediction: fewer duplicate charts and a cleaner nerve at equal compute.

## Arms (T2 beds s2, t2, klein, so3 + neutral r2, lift obs; 3 seeds; tuning on a separate seed)
1. Softmax `ObserverAtlas`. 2. `CoherentAtlas` (rank r in {2, 4}). 3. Ablation: Arm Q with P frozen at 0
(same params, softmax behaviour). 4. Ablation: theta frozen at 0 (real-only interference).
Equal active FLOPs and equal total params (L-BR). Arm Q pays K (d_key + 16 r + 2) + 16 r^2 extra MACs per
step, which comes out of the chart hidden width via `arms/match.py`.

## Metrics (preregistered definitions)
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

## Bar (all required, >= 3/4 curved manifolds, 3 seeds)
Arm Q's duplicate rate <= 0.5x softmax's; Betti correct on >= as many manifolds as softmax; discontinuity
and h = 10 error <= softmax + 2%; neutral r2 error within +2%; the P = 0 ablation does not reproduce the
gain (otherwise the gain comes from params, not coherence).

## Kill
- Arm Q <= softmax on the T2 metrics (no improvement in duplicate rate, Betti or discontinuity) means Arm Q is dead.
- Learned coupling stays ~0 (||P|| / ||Q|| < 0.01 on >= 3/4 manifolds) means SGD picks decoherence, the
  coherence is unused, and Arm Q is dead.
- The theta = 0 ablation ties full Arm Q means phases are unused. Arm Q survives only as "real
  interference" and the phase claim is dead.

## T6 link (measurement only)
Run `t6_contextuality` (same protocol as PREREG_DRAFT_t5_t6.md) on softmax vs Arm Q, each next to its
untrained init. Report QQ, order, dA and dB with paired bootstrap CIs. Within one step the observer is still
a map of c_t, so order effects come from the GRU context in both arms. The question is whether coherence
changes their form (QQ ≈ 0 with order ≠ 0 is the Lüders signature). A Lüders signature is "consistent
with", never "evidence of", quantum structure. No bar, no kill.

Prior (guess): ~15% that Arm Q clears the bar.

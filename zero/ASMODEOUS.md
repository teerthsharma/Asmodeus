# ASMODEOUS: one architecture, best part per slot

**Status.** Draft by Kutner, 2026-10-04. Rows marked [T13] stay open for Thirteen's ideas (IDEAS_thirteen.md) and
are folded in when that file lands.

**What this is.** ASMODEOUS is not a composition of arms. Each slot is chosen on its own merits, and the
model is built as one class: `atlas/asmodeous.py`, built by nurses.

**Fixed constraints.** Seal's laws (L-GIVEN: every structure learned; L-PREREG; L-BR fair baselines) and the
`common.py` contract. Every row's ablation is a preregistered arm at equal active FLOPs and equal params.

**Source keys.**
- AT = OBSERVER-ATLAS (Chase).
- PS = PHASE-SPHERE (Foreman).
- AE = Seal's A.E.T.H.E.R. paper (`ref/origin/AETHER_qGPU_event_driven.pdf`).
- O# = ORIGIN.md object.
- K# = IDEAS_kutner.md idea.
- OW = owned method.

## Slot table

| slot | chosen feature | source | why it wins over the alternatives | ablation that removes it |
|---|---|---|---|---|
| **state** | **Global state.** ψ = normalise(E(o)) ∈ C^N, a point of CP^{N−1}. Phase is part of the state; "small phases all around the sphere".<br>**Encoder.** One-frame MLP. No recurrent backbone, so the frame is Markov on lift beds. | PS, O1 | A complex state is the only option that carries a U(1) phase, so TO1 (Chern) is defined on it. Dropping the GRU removes the order-of-presentation context that would otherwise confound T6. The swing-up lift contains velocity, so one frame suffices. | **real-ψ**: ψ ∈ R^N, everything else the same. TO1 then reads c₁ = 0 by construction. |
| **chart** | z_k = V_k† ψ · e^{−i arg α_k} ∈ C^{d_c}, d_c = n. Each chart has its own learned frame V_k, gauge-fixed by its own amplitude phase. It is computed only when chart k is read. | PS | Charts are cheap (4·N·d_c MACs) and exactly gauge-invariant, so a global phase cancels. Separate per-chart MLP encoders (AT) cost D·h each and still need gluing to agree. | **mlp-charts**: AT's per-chart MLP encoders on o_t, lazy. |
| **decider** | **Parseval-frame POVM.** Chart effects v_k are the rows of an isometry V ∈ C^{(K+1)×N} with V†V = I. The extra row is the "observe nothing" effect. p_k = \|⟨v_k, ψ⟩\|², Σ_k p_k = 1 exactly, with no partition function. gate = 1 − p_null. V is parametrised by a Cayley / QR retraction on the Stiefel manifold, learned. | AE (adaptive POVM E_k = U†Π_kU), K1, O1 (Born), Gleason | AE's form gives completeness only for K ≤ N. ORIGIN O9 concluded that K > N needs a Naimark dilation costing ≥ K·N. **It does not:** a K×N isometry's rows already form a rank-1 POVM with K > N outcomes (a tight frame). Probabilities of any read subset are exact, not renormalised. Gleason makes this the only non-contextual decider. | **born-renorm**: PS's \|⟨u_k,ψ⟩\|² / Σ_j. **softmax**: AT's q·κ. |
| **laziness in K** | Score all K+1 effects (K·N MACs, cheap at N = 16), but run only the top-m charts. Because Σp = 1 holds by construction, the weights of the read charts are exact without the Σ over K. Above K ≈ 4096, a Kronecker frame V = A ⊗ B (still an isometry) prunes candidates; this is a T3 extension. | K1, PS | AT's lazy top-m differs from its soft version by an approximation gap (T3 bar "within 1%"). Here that gap is 0 by construction, which is a unit test, not a bar. | **dense**: all K charts run, weights masked. The parity test requires lazy == dense on the read charts. |
| **laziness in time** | (a) **Event trigger** (AE). Compute only when ‖o_t − o_τ‖ ≥ ε, with ε learned (Taub's `EventDriven` wrapper). (b) **Adiabatic re-route.** Between events, keep the chart set and re-score only the m read effects (m·N MACs). Call the full decider when p_read < 1 − η, η learned, i.e. when the state leaves its charts. | AE, K5 | (a) skips frames; (b) skips routing inside frames. They cover different redundancy. Every other arm routes every step. | **no-event** (ε = 0). **always-route** (η = 0). |
| **gluing** | **Gauge links.** For co-read charts j, k, a learned link G_jk ∈ U(d_c) (Cayley of a skew-Hermitian matrix from [z_j, z_k]) is trained so that z_k ≈ G_jk z_j (cocycle loss). Plaquettes G_jk G_kl G_lj are measured on nerve triangles. The contract latent is g = gate · Σ_read w_k φ_k(z_k), kept so every metric is shared. | K2, O7, PS (χ_jk) | The links are the transition functions themselves, i.e. a learned Čech cocycle. Sign holonomy (TO7) is det(Π links) along a loop, read off with no extra learned frame field. AT's overlap MSE in R^{D_g} only makes the global embedding agree. | **overlap-mse**: AT/PS's overlap term instead of links. TO7 then falls back to PS's sign_holonomy. |
| **dynamics** | ẑ_k = Cayley(X_k(a)) z_k, a unitary local step: one batched linear solve, norm-preserving. | PS, K10 | It stays on the group exactly, which suits long rollouts at the planning horizon h = 30. | **mlp-step**: AT's residual MLP predictor. |
| **observer loop** | **Damped CPTP fixed point** (v1, as built). Over the M candidates, w ← (1−ε)·P w + ε·uniform, where P[d,c] is the probability of candidate d measured on candidate c's normalised predicted state. P is column-stochastic: a classical Deutsch fixed point, a contraction by theorem. The read set = top-m of w* (loop mode). Solved with `phase_sphere.fixed_point` without gradient, then one differentiable step from w*. ε > 0 makes the map a contraction, so the fixed point is unique and convergence is geometric. **Halt before start (O3a):** an amortised head predicts w* in one shot and is distilled from the loop. | K3, O2, O3, PS loop, KT time-traveller | PS's loop is undamped and only reports non-convergence. The TT loop halts on set equality and can cycle. Here TO2's convergence bar holds as a theorem, so the test measures accuracy at equal FLOPs. | **one-pass** (no loop). **undamped** (ε = 0, PS loop). **discrete-TT**. |
| **readout** | Nerve Betti over Z2 (soft nerve, the PREREG pins). Chern c₁ from ψ over an icosphere. Sign holonomy = det of the link product. T6: QQ / order / dA / dB, plus the exchange phase θ = arg⟨ψ_AB\|ψ_BA⟩. **All eval-only.** | AT, PS, K2, K16, t6 | Each readout is the one that can fail where the others cannot: Betti cannot tell T² from Klein, holonomy can; c₁ needs phase; θ catches phase-only contextuality. | None; readouts never train. Control: each readout on the untrained init. |
| **anti-collapse** | SIGReg on g (shared LAM table) and a **Fubini–Study / Haar test on ψ**: sliced \|⟨u, ψ⟩\|² vs Beta(1, N−1) for random u. Plus the EMA target encoder. | OW (LeJEPA), K14 | A Gaussian target is the wrong reference for a unit complex vector; Haar on CP^{N−1} is the uniform law on that space. | **sigreg-only**. |
| **training** | Prediction loss = **fidelity** 1 − \|⟨ψ̂_{t+1}, sg ψ^{tgt}_{t+1}⟩\|². It is phase-invariant and lives on CP^{N−1}, where ψ̂_{t+1} = normalise(Σ_read w_k V_k ẑ_k e^{i arg α_k}). Also: per-chart gauge-fixed MSE (PS), the link cocycle loss, load balance, and distillation of the amortised head. | O1, PS, CLAUDE.md rule 1 (not JEPA) | An L2 JEPA loss on z is gauge-dependent and penalises a pure phase shift. Fidelity is the natural loss on CP^{N−1}. | **l2-jepa**: PS's gauge-fixed MSE only. |
| [T13] | Folded in: see "Thirteen's slot additions" below. Each addition is a row with its own ablation. v1 builds the rows marked v1; v2 rows wait for the v1 contract to be GREEN. | IDEAS_thirteen.md | | |

## Thirteen's slot additions (IDEAS_thirteen.md, T#)

| slot | addition | source | why it wins | ablation | build |
|---|---|---|---|---|---|
| **gluing + observer loop** | **Kuramoto gauge relaxation.** Each read chart's gauge phase θ_k (= arg α_k) is relaxed on the nerve inside the loop: θ_j ← θ_j + η Σ_k w_jk sin(θ_k − θ_j − χ_jk), where χ_jk = arg det G_jk comes from the links. Halt when the energy V = −Σ w_jk cos(·) stops falling. Pairs below a coherence threshold do not couple (T18). | T1, T18, O1, O4 | Topology becomes what the loop *cannot* relax away: a nontrivial bundle leaves a vortex. The continuous θ may be chaotic, but the vortex count is protected (O4, literally). The loop gets a Lyapunov function and costs O(m) with mean-field coupling. | **no-relax**: χ read statically (PS). | v1 |
| **laziness in time (c)** | **Per-chart send-on-delta.** A read chart recomputes only when its own input projection drifts by ‖V_k†(ψ_t − ψ_ref,k)‖ ≥ ε_k (learned, with spike-frequency adaptation). Otherwise it holds ẑ_k. | T4, AE (event-driven), event camera, LIF | (a) gates whole frames and (b) gates routing. (c) gates each chart, so a chart over a static region never fires (AER-style). | **global-event** only. | v1 |
| **anti-collapse (router)** | **Sinkhorn equipartition targets** in training only: the batch effect-assignment target is Sinkhorn-balanced (3 iterations), and V is trained by CE toward it. It replaces the Switch load balance. Inference is unchanged. | T3, OT | It makes dead charts impossible by construction during training. Switch only nudges. | **switch-lb**. | v1 |
| **dynamics init** | **DMDc initialisation** of each chart's Cayley generator X_k(a) from its first observed rows. SGD trains it from there (L-GIVEN). | T8 (Koopman) | It gives a reason why a unitary local step suffices (Koopman eigenfunctions per chart), with faster time to L-LEARN. | **random-init**. | v1 |
| **training** | **Rare-mode replay.** A buffer of rows whose winning chart had p_k* < 1/K or a high loss is replayed at a fixed batch fraction. | T5, hippocampal replay | Fixes the WTA rare-mode ceiling that MoP v1 names (37% full recovery at 4-way cells). | **no-replay**. | v1 |
| **readout** | **Vortex c₁** (residual winding after relaxation), **Morse χ** (critical-point census of the loop energy on a kNN graph of g), **sheaf w₁** (smallest eigenvalue of the orientation-sheaf Laplacian on the links), and the **nerve-theorem gate** (overlaps must have H₀ = 1, H₁ = 0, otherwise the readout is VOID). All eval-only. | T1, T6, T10, T11 | χ + w₁ classify every closed surface, so the readouts are complete on the T2 beds. The gate prevents a confident wrong Betti or c₁ from a chart that wraps. | Control: each readout on the untrained init and on a random f (Morse). | v1 (readouts.py extension) |
| **gluing (contract g)** | **Precision-weighted glue**: g = (Σ Λ_k)⁻¹ Σ Λ_k φ_k(z_k), with low total precision as the "observe nothing" signal. | T9, Kalman information form, predictive coding | It weights *trust per coordinate*, not only which chart. It gives a calibrated 1-step variance. | **w-glue** (Kutner's row). | v2 |
| **laziness in K (huge K)** | Learned **SimHash shortlist** in front of the POVM above K ≈ 4096, and a **Bélády-scored VRAM cache** of chart parameters for K beyond 8 GB, using the loop's predicted routing as the reuse oracle. | T13, T14 | The Kronecker frame forces a factorisation; LSH adapts to the key distribution. The cache makes "observed → exists" a memory guarantee. | **kronecker**, **all-resident**. | v2 (T3/T7) |
| **training (K itself)** | **MDL growth.** Copy-on-write chart splitting and pruning, stopped by two-part code length. | T12 | P-COUNT becomes emergent: K = atlas(M) found, not swept. | **fixed K**. | v2 |
| **honesty** | **LTSA-glued charts** as a closed-form rival to learned links (alongside Kutner's Hopf arm). | T17 | If one eigenproblem matches learned gluing, the gluing claim is owned. | — (control arm) | v1 control |

## Contract and invariants (RED tests, `tests/test_asmodeous.py`)

- **Contract.** `forward(obs, act)` → `loss, g, ghat, w [B,T,K], gate, psi (complex), ctc_iters, ctc_residual`, plus
  `phase` [B,T,m] = arg α_k, `chi` [B,T,m,m] = wrapped transition phases, `zidx`, and `link` [B,T,m,m,d_c,d_c]
  (unitary, G_kk = I), for Thirteen's readouts. atlas/asmodeous.py must not import `beds` (L-GIVEN audit).
  Also `rollout`, `active_flops`, and `n_params` (EMA target excluded).
- **Decider.**
  - V†V = I to 1e-6 after every optimiser step (retraction).
  - Σ p = 1, p ≥ 0.
  - Lazy == dense on the read charts, to 1e-6.
- **Loop.** Contraction: the residual decays at rate ≤ (1−ε) on 1,000 random instances, with a unique fixed
  point from 10 random starts.
- **Gauge.**
  - g, w and the losses are invariant to ψ → e^{iφ}ψ.
  - With real-ψ (ablation), c₁ = 0.
- **Links.** G_jk is unitary, and det of the link product is ±1 for real links.
- **L-GIVEN.** No bed state, chart identity or manifold enters the model. A test greps the forward path for
  `state` access, as in tests/test_lgiven.py.

## Controls attacking our own claims (honesty)

- **Hopf single-chart arm (K8; owned-style control against C1).**
  - Latent ψ ∈ S³ ⊂ C², read out by the Hopf map p(z₀,z₁) = (2z₀z̄₁, \|z₀\|² − \|z₁\|²) ∈ S². One chart,
    no observer, equal FLOPs.
  - On S² (and, via unit quaternions, on SO(3)), if it reaches discontinuity 0 at the T2 ε/δ, then C1's
    "the atlas is needed" is owned by a bundle trick for those manifolds, and C1 must retreat to readout or
    compute claims.
  - It enters PREREG as a control arm with its own bar: discontinuity ≤ the ASMODEOUS value on s2 and so3.
- All ablation rows above run at equal active FLOPs and params, 3 seeds, tune seed 3.
- A win credits a slot only if that slot's ablation loses it.

## Build plan (sonnet nurses, Kutner reviews every diff)

1. **Primitives** (`atlas/asmodeous.py`, functions part): `stiefel_retract`, `povm_probs`, damped
   `cptp_fixed_point` (wrapping `phase_sphere.fixed_point`), `link_cayley`, `fidelity`, `haar_cp_stat`.
   Each gets its RED unit test.
2. **Contract tests** for `Asmodeous` (RED), written in parallel with step 1.
3. **Class** `Asmodeous`: GREEN against both test sets, with lazy parity and the L-GIVEN grep.
4. **Hopf control arm** (`arms/hopf.py`): RED → GREEN, owned-style.

## Hopf chart: an ASMODEOUS part (preregistered before any run; Wilson merges)

**Framing (Seal, zero/CLAUDE.md).** The atlas is one part of ASMODEOUS. If one Hopf chart covers S² (and
CP^{N−1}) continuously, it becomes the ASMODEOUS chart for those spaces, and that is a win.

**Arm `atlas/hopf_chart.py` (`HopfChart`).**
- State and dynamics:
  - one-frame encoder o → ψ ∈ C^N, normalised, a point of S^{2N−1};
  - unitary action step ψ_{t+1} = Cayley(X(a, ψ)) ψ;
  - g = the real traceless part of ψψ† ∈ R^{N²−1}. For N = 2 this is exactly the Hopf map to S², the Bloch
    vector. It is invariant to the global phase.
- Training:
  - fidelity loss against an EMA target;
  - SIGReg on g, the shared table;
  - `psi` is exposed so TO1 can read it.
- Contract: no observer and no charts, so w = None.

**Beds.** s2, berry, berry0 (lift), 3 seeds, tune seed 3.

**Matching (L-BR).** Equal active FLOPs and equal params against ATLAS (K = atlas(M), m = 2) and
PHASE-SPHERE, matched by encoder hidden width.

**Bars.**
- C1, on s2 and berry0:
  - Discontinuity (tear) ≤ 0.2 × torn reference AND ≤ 0.01, with fold_rate ≤ 0.01, using the PREREG
    eps/delta/far/fold_delta.
  - h = 10 error ≤ ATLAS + 2%.
  - All of this on 3/3 seeds.
- TO1:
  - \|c₁\| = 1 on berry and 0 on berry0, with \|c₁ − round(c₁)\| < 0.1, on 3/3 seeds.
  - The gauge-sensitivity control (random per-vertex phase) is reported.

**Kill.** Discontinuity or fold above the bar on either bed for ≥ 2 seeds. Then one bundle chart does
not suffice in practice, and the atlas keeps the chart slot for S².

**Wiring smoke.** One run on tune seed 3, s2 only, with a short step count. It checks wiring only; its
numbers are never reported as results.

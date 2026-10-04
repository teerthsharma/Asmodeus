# IDEAS: Thirteen (computation, neuro and topology-of-data half of the hunt)

**Sources.** About 50 English Wikipedia articles, fetched 2026-10-04 by 8 sonnet nurses. Each fact is
summary-level. A fact marked **[K]** is standard background that the nurse could not find on the page;
check the primary source before citing it in a paper.

Kutner holds physics and geometry (`IDEAS_kutner.md`). Where we overlap, I say so and name the split.

**What each idea is measured against:**
- softmax/Born top-m (atlas, phase-sphere) and Kutner's POVM isometry;
- free gluing maps φ_k into R^{D_g}, and Kutner's gauge links;
- MLP and Cayley chart predictors;
- FP/TT/CPTP observer loops;
- nerve Betti, sign holonomy and Chern;
- SIGReg plus the Switch load balance.

**Fields in each entry:** slot · source (URL, one fact) · mechanism · why it beats today · test (bar / kill) ·
prior (a guess).

---

## Top six (my picks for ASMODEOUS)

### 1. Kuramoto phase-locking as the gluing dynamics; vortices as the Chern readout
- **Slot:** gluing, observer loop, readout.
- **Source:** https://en.wikipedia.org/wiki/Kuramoto_model. Coupled phase oscillators
  dθ_j/dt = ω_j + (K/N)Σ sin(θ_k − θ_j) synchronise above a critical coupling K_c. The order parameter is
  r·e^{iψ} = (1/N)Σ e^{iθ_j}.
- **Mechanism:**
  - Each co-selected chart carries a U(1) phase: its gauge, which is phase-sphere's "small phases all
    around the sphere".
  - The observer loop is a few Kuramoto steps on the nerve. Phases relax toward agreement on overlaps,
    with the learned transition phases χ_jk as the frustration offsets:
    θ_j ← θ_j + η Σ_k w_jk sin(θ_k − θ_j − χ_jk).
  - On a trivial bundle the phases lock (r → 1).
  - On a bundle with c₁ ≠ 0, a global phase cannot exist (ORIGIN O1). Relaxation must leave a **vortex**.
    The winding count over nerve triangles after relaxation is the Chern number.
- **Why it beats today:**
  - Chern is today a sum of wrapped transition phases read off a static triangulation.
  - Here the phases are produced by a dynamics with a Lyapunov function,
    V = −Σ w_jk cos(θ_k − θ_j − χ_jk). Topology becomes what the loop *cannot* relax away, which is ORIGIN
    O4 ("butterfly bonded via topology") literally: θ can be chaotic, while the vortex count is protected.
  - The loop's halt rule is physical: dV < ε.
- **Test (Berry bed, TO1):**
  - Bar: residual vortex count = 1 on Berry and 0 on control, on 3/3 seeds, after ≤ 20 relaxation steps.
  - Bar: r ≥ 0.95 on the control.
  - Kill: vortex count wrong on ≥ 2 seeds, or r < 0.8 on the control (no locking means the gluing failed).
- **Prior:** 30%.

### 2. Grid-module phase codes for the glued state: exponential range from a few periodic charts
- **Slot:** state, gluing, chart.
- **Sources:**
  - https://en.wikipedia.org/wiki/Grid_cell. A hexagonal lattice with 60° symmetry. Modules share
    orientation and spacing with random phase offsets. Spacing grows from about 25 cm to about 3 m along
    the dorsal-ventral axis.
  - The about-1.4 module ratio is **[K]** (Stensola 2012), not on the page.
- **Mechanism:**
  - The glued space is a product of tori, ∏_i T² with M modules. Each module's coordinate is a phase pair,
    and module i has a learned period λ_i (initialised geometric, free under SGD).
  - Charts write their prediction as phase increments, i.e. path integration, Δθ_i = 2π v/λ_i.
  - Decoding a position is a residue-number problem. The range grows like ∏λ_i / resolution, which is
    exponential in M, while each module stays a tiny 2-D phase chart.
- **Why it beats today:**
  - R^{D_g} gluing has range linear in D_g and no notion of phase.
  - Phases make the state live on tori, so holonomy and winding readouts are native.
  - Prediction becomes addition of phases, which is linear and Koopman-like (idea 8).
- **Test (r2, pointmaze-large):** at equal D_g, the h = 30 path-integration error is ≤ 0.7× the R^{D_g} glue.
  Linear-probe recovery of position is ≥ the R^{D_g} glue.
- **Kill:** ≥ 0.95×.
- **Prior:** 30%.

### 3. Sinkhorn equipartition router: no dead charts without an auxiliary loss
- **Slot:** anti-collapse (router), decider training.
- **Sources:**
  - https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem: alternating row and column rescaling makes a
    positive matrix doubly stochastic, unique up to scale.
  - https://en.wikipedia.org/wiki/Optimal_transport: entropic OT, min Σγc + εγ log γ, is solved by
    Sinkhorn.
- **Mechanism:**
  - During training only, the chart-assignment targets for a batch are the Sinkhorn-balanced version of the
    router scores (3 iterations, as in SwAV/BASE layers **[K]**). The decider is trained by cross-entropy
    toward them.
  - At inference the decider is unchanged: plain Born/softmax, lazy.
- **Why it beats today:**
  - The Switch loss n·Σf_iP_i only nudges.
  - Equipartition makes "every chart gets ~B/K rows" a constraint. Dead charts (the router collapse in
    T4a's chart-abandonment log) become impossible by construction during training.
- **Test (T4a protocol, chart level):** at SIGReg 0, the fraction of charts used (≥ 1% rows) at step 5000 is
  ≥ 0.9, against Switch-loss runs. The T2 h = 1 error is no worse than +2%.
- **Kill:** used fraction ≤ the Switch fraction, or error > +2%.
- **Prior:** 45%.

### 4. Event-driven charts: per-chart send-on-delta with reference reset (event camera + LIF)
- **Slot:** laziness in time, per chart.
- **Sources:**
  - https://en.wikipedia.org/wiki/Event_camera: a pixel emits an event when |log I − log I_ref| ≥ C, then
    resets I_ref. 3–15 µs latency, 120 dB dynamic range.
  - https://en.wikipedia.org/wiki/Spiking_neural_network: LIF integrates input, spikes at threshold and
    resets. Spike-frequency adaptation raises the threshold while firing.
- **Mechanism:**
  - Each *selected* chart keeps a reference of its input projection and its last output.
  - It recomputes only when its own input drift ‖P_k o_t − P_k o_ref‖ crosses a learned threshold ε_k
    (softplus, straight-through). Otherwise it holds.
  - An adaptation term raises ε_k after each fire, which damps chatter.
- **Why it beats today:**
  - `EventDriven` gates the whole arm with one ε.
  - Kutner #5 gates the router only.
  - This makes time-laziness local, as in neuromorphic AER: a chart over a static region of the state
    never fires, while a chart on a moving region fires every step.
  - Compute ∝ (charts × events), not charts × frames.
- **Test (T2 beds, T-EV protocol):** at ≤ +1% h = 1 error, chart-evaluations per frame are ≤ 0.5× the global
  EventDriven, at equal routing.
- **Kill:** ≥ 0.8×.
- **Prior:** 40%.

### 5. Replay of rare transitions: fix the WTA rare-mode ceiling
- **Slot:** training.
- **Sources:**
  - https://en.wikipedia.org/wiki/Hippocampal_replay: forward, reverse and pre-play sequences.
  - https://en.wikipedia.org/wiki/Sharp_wave%E2%80%93ripple_complexes: 140–220 Hz ripples, 30–100 ms
    windows; ripple suppression impairs new memory.
  - The compression factor (about 10–20×) is **[K]**.
- **Mechanism:**
  - Keep a small buffer of (context, successor) pairs whose winning chart had low router mass (π_k* < 1/K)
    or high loss.
  - Replay them at a fixed fraction of each batch, and reverse-replay sequences for credit assignment.
  - MoP-JEPA v1 names this as untested ("replay biased toward rare modes remain untested").
- **Why it beats today:** v1 recovers all four modes in only 37% of four-way cells. Hard WTA starves rare
  modes; replay feeds them.
- **Test (T4b medium-stitch, 4-way cells):**
  - Bar: full-mode recovery ≥ 0.6, against the reproduced MoP baseline, at equal steps.
  - Bar: realroute ≥ the baseline.
- **Kill:** ≤ +0.05 recovery.
- **Prior:** 45%.

### 6. Morse census: Euler characteristic and chart count from a learned scalar observer field
- **Slot:** readout, chart.
- **Sources:**
  - https://en.wikipedia.org/wiki/Morse_theory: topology is constant between critical values, an index-γ
    critical point attaches a γ-cell, and the strong Morse inequalities give C^γ ≥ b_γ.
  - https://en.wikipedia.org/wiki/Reeb_graph: Morse functions give a finite graph with min/max of degree
    1 and saddles of degree 3. Mapper is the discretised Reeb graph.
- **Mechanism:**
  - The observer's top logit, or the Kuramoto energy V of idea 1, is a scalar field f on the data.
  - On a kNN graph of eval states, count discrete critical points by lower-link connectivity (minima,
    saddles, maxima).
  - χ = Σ(−1)^γ C^γ is exact for a Morse function, whatever the counts. Chart basins under −∇f form a
    Morse–Smale cover.
- **Why it beats today:**
  - Nerve Betti needs simplices up to dimension m−1, and its dimension is capped by m.
  - The census reads χ from any f in O(n·k). It separates S² (χ = 2) from T² and Klein (χ = 0).
  - Kutner's sign holonomy then separates T² from Klein. χ plus w₁ classify all closed surfaces, so the two
    readouts together are complete for the T2 beds.
- **Test (T2 beds):**
  - Bar: χ correct on s2, t2 and klein on 3/3 seeds, and this must hold for both the learned f and a random
    control f.
  - The random control is required: the census is exact for *any* Morse f. So the claim is "the learned
    representation is faithful", and the test is honest only if the kNN graph is built on the arm's g,
    not on state.
- **Kill:** χ wrong on ≥ 2 beds.
- **Prior:** 35%.

---

## Further ideas

### 7. Modern-Hopfield energy for the observer loop: guaranteed descent, β sets the effective m
- **Slot:** observer loop, decider.
- **Source:** https://en.wikipedia.org/wiki/Hopfield_network (dense associative memory section). With an
  exponential interaction F = e^x, capacity is ≈ 2^{N_f/2}. The log-sum-exp Lagrangian gives attention.
- **Mechanism:**
  - Observer update q ← Σ_k softmax(β κ_k·q) κ_k, where κ_k is a chart key augmented by that chart's
    predicted future.
  - E = −lse(β K q) + ½‖q‖² decreases monotonically, and halt is ΔE < ε.
  - Small β gives metastable mixtures (many charts); large β gives one chart.
- **Why it beats today:**
  - It gives an energy where FP/TT have none.
  - It complements Kutner #3, which gives a contraction on ρ. Mine needs no density matrix (O(K) not O(K²)).
  - β is learned, so m can become data-driven.
- **Test:** converges in ≤ 3 iterations on ≥ 99% of inputs (TO2 bar). h = 10 error ≤ the FP loop.
- **Kill:** > 1.05× FP.
- **Prior:** 30%.

### 8. Koopman/DMDc chart dynamics: linear (unitary) per chart, with a closed-form initialisation
- **Slot:** dynamics.
- **Sources:**
  - https://en.wikipedia.org/wiki/Composition_operator (Koopman): C_φ f = f∘φ is linear even when φ is
    not.
  - https://en.wikipedia.org/wiki/Dynamic_mode_decomposition: reduced operator S̃ = UᵀV₂WΣ⁻¹ at cost
    O(MN²). DMDc handles control.
- **Mechanism:**
  - Chart step z′ = A_k(a) z with A_k(a) = exp(skew(G_k a)), as on phase-sphere.
  - Initialise each chart's generator from DMDc on that chart's first observed rows. SGD still trains it
    (L-GIVEN).
- **Why it beats today:**
  - It gives a reason why linear unitary local dynamics suffice: Koopman eigenfunctions on a chart.
  - The eigenvalue spectrum (|λ| ≈ 1, arg λ = rotation per step) is a readout of the chart's period.
- **Test:** h = 10 error ≤ the MLP chart predictor, and time-to-L-LEARN ≥ 1.5× faster with DMDc init.
- **Kill:** no speed-up.
- **Prior:** 30%.

### 9. Precision-weighted (information-form Kalman) gluing
- **Slot:** gluing, uncertainty.
- **Sources:**
  - https://en.wikipedia.org/wiki/Kalman_filter: the information filter fuses estimates by adding inverse
    covariances.
  - https://en.wikipedia.org/wiki/Predictive_coding: errors are weighted by precision (inverse variance).
- **Mechanism:**
  - g = (Σ_k Λ_k)⁻¹ Σ_k Λ_k φ_k(z_k), with Λ_k = diag softplus(learned) per chart, per input.
  - When charts disagree, the most certain chart wins locally. Σ Λ_k is a calibrated confidence. A low
    total precision is the "observe nothing" signal, with no gate parameter.
- **Why it beats today:** the softmax weight mixes *which* chart; precision mixes *how much to trust* each
  coordinate. On overlaps that removes the overlap-MSE fight.
- **Test:**
  - Bar: T2 discontinuity rate ≤ softmax gluing at equal FLOPs.
  - Bar: the NLL calibration slope of the 1-step error against predicted variance is in [0.8, 1.2].
- **Kill:** the slope is outside [0.5, 2].
- **Prior:** 30%.

### 10. Sheaf-Laplacian cohomology readout on Kutner's gauge links
- **Slot:** readout.
- **Sources:**
  - https://en.wikipedia.org/wiki/Sheaf_(mathematics): locality and gluing; sheaf cohomology is the
    obstruction to local-to-global.
  - Cellular sheaf Laplacian L = δᵀδ, ker L = H⁰ **[K]** (Hansen–Ghrist; the WP page lacks it).
- **Mechanism:**
  - Stalks = chart frames, restriction maps = Kutner's links g_jk.
  - dim ker L₀ = number of global sections. A twisted (Klein) O(1) sheaf has H⁰ = 0 for the orientation
    line bundle, against 1 on T².
- **Why it beats today:**
  - Sign holonomy needs held-out *closed loops* drawn from state (eval-only, but loop-picking is a choice).
  - The sheaf spectrum reads w₁ from the learned nerve alone: no loops, no state.
- **Test:** on t2 against klein, the smallest eigenvalue of L₀ on the orientation sheaf is < 0.05 on T² and
  > 0.2 on Klein, on 3/3 seeds.
- **Kill:** no separation.
- **Prior:** 30% (needs Kutner #2).

### 11. Nerve-theorem admission gate: the topology readout is valid only if overlaps are contractible
- **Slot:** readout validity, regulariser.
- **Source:** https://en.wikipedia.org/wiki/Nerve_of_a_covering. The Leray/Borsuk nerve theorem: if every
  non-empty finite intersection is contractible, then N(C) ≃ ∪C.
- **Mechanism:**
  - For each nerve edge or triangle, take the eval points co-selected by those charts and compute H₀ and H₁
    of their kNN complex (gudhi is installed).
  - A non-contractible overlap (two components, i.e. a chart wrapping around) is flagged.
  - Betti or Chern readouts are reported only on nerves that pass. Otherwise the result is VOID, not a
    pass.
- **Why it beats today:** today the nerve Betti readout is trusted blindly. A chart that wraps a circle
  gives a nerve with the wrong H₁ while every number still prints.
- **Test (unit):** a planted 2-chart cover of S¹ whose overlap has 2 components is flagged; a good 3-chart
  cover passes.
- **Bed test:** the fraction of passing nerves is reported per arm.
- **Prior:** 70% as a tool, n/a as a claim.

### 12. MDL chooses K: P-COUNT becomes emergent, not swept
- **Slot:** training and model selection, anti-overfit.
- **Source:** https://en.wikipedia.org/wiki/Minimum_description_length: two-part code, min L(H) + L(D|H).
  It is formally identical to BIC for large n.
- **Mechanism:**
  - Grow charts by copy-on-write splitting: the worst-error chart forks, and its parameters are shared until
    they diverge.
  - Prune charts whose removal lowers L(params) + L(residuals | Gaussian at the learned precision).
  - Stop when no split lowers the description length.
- **Why it beats today:** PROMPT C1 predicts tearing stops at K = atlas(M), but the sweep supplies K. If MDL
  stops at atlas(M) without being told, that is learned structure (L-GIVEN), and it is a stronger claim
  than any sweep.
- **Test:** the MDL-stopped K equals atlas(M) on ≥ 3 of {s2: 2, t2: 3, klein: 3, so3: 4}, and K = 1 on r2,
  on 3/3 seeds.
- **Kill:** off by ≥ 2 on ≥ 2 beds.
- **Prior:** 20%.

### 13. LSH (SimHash) chart shortlist: collision probability *is* an angle
- **Slot:** laziness in K.
- **Source:** https://en.wikipedia.org/wiki/Locality-sensitive_hashing. Random hyperplanes give
  Pr[h(x) = h(y)] = 1 − θ/π. L tables of k bits give sublinear O(n^ρ) query.
- **Mechanism:**
  - Hash the decider query (or ψ) into L = 4 tables of k = log₂K bits.
  - The candidate charts are those in colliding buckets, and the exact Born/POVM scores are computed only on
    candidates.
  - The hyperplanes are learned (SGD on the bucket-recall loss), not random.
- **Why it beats today:** product keys force a √K×√K factorisation of the key space; LSH adapts to the key
  distribution. It complements Kutner #1, which needs a MIPS front end, and this is one.
- **Test (T3):** at K = 1024 and 4096, recall@m of true top-m ≥ 0.95 at ≤ 0.5× product-key routing FLOPs.
- **Kill:** recall < 0.9.
- **Prior:** 35%.

### 14. Bélády-scored chart cache: K beyond VRAM
- **Slot:** laziness in K (systems, T7).
- **Source:** https://en.wikipedia.org/wiki/Cache_replacement_policies. Bélády evicts the item used
  farthest in the future; ARC adapts between LRU and LFU; S3-FIFO filters one-hit wonders.
- **Mechanism:**
  - Chart parameters live in host RAM. A VRAM cache holds the hot set.
  - The observer's predicted future routing (the CTC loop already computes it) is the reuse oracle, i.e. an
    approximate Bélády.
- **Why it beats today:** on the 8 GB 4060, K is capped by VRAM. A lazy atlas whose charts *exist only when
  observed* should not pay VRAM for unobserved ones. That is "observed → exists" as a systems guarantee.
- **Test (T7):** K = 2^20 tiny charts with hit rate ≥ 0.95 on T3 traces, and µs/step ≤ 1.5× an all-resident
  K = 2^16 run.
- **Kill:** hit rate < 0.8.
- **Prior:** 35%.

### 15. Theta sweep: the observer's look-ahead as a phase-ordered sequence inside one tick
- **Slot:** observer loop, planning (T5).
- **Sources:**
  - https://en.wikipedia.org/wiki/Phase_precession: spikes move earlier in the theta cycle across a field,
    turning spatial order into temporal order within one cycle.
  - https://en.wikipedia.org/wiki/Theta_wave: 6–10 Hz in active rodents, with gamma nested in theta.
- **Mechanism:**
  - One tick = one sweep. The observer visits the chart *sequence* along the predicted path (chart t, t+1,
    …, t+h) in phase order, reading each chart once. The sweep length adapts to speed (the step norm).
  - It is the time-traveller's life-line, compiled into a single pass whose cost is linear in h and
    independent of K.
- **Why it beats today:** the TT loop re-scores M candidates × h. The sweep follows one path, and only
  branches where the chart router is uncertain (the entropy gate). At equal planning compute that buys a
  longer h.
- **Test (T5 swing-up):** success ≥ the TT loop at ≤ 0.5× planning MACs.
- **Kill:** lower success at equal MACs.
- **Prior:** 25%.

### 16. Particle-filter decider for stochastic beds: resample charts, don't average them
- **Slot:** decider on branching dynamics (T4b).
- **Source:** https://en.wikipedia.org/wiki/Particle_filter. Weight by likelihood, then resample when the
  weights degenerate. Rao-Blackwellised PF puts a Kalman filter inside each particle. Bias and variance
  are ≤ c/N uniformly in time.
- **Mechanism:**
  - Keep P hypotheses (chart id, latent).
  - Each step every hypothesis proposes its K successor heads; weights are the router mass.
  - Resample to P when the ESS drops below P/2.
  - Planning (planAll) runs over the particle set, not over a single argmax.
- **Why it beats today:** MoP's argmax-π readout throws away the branches it learned. The particle set keeps
  them at fixed cost P.
- **Test (T4b teleport):** realroute ≥ MoP at P = K = 8, and execution ≥ MoP.
- **Kill:** ≤ MoP.
- **Prior:** 25%.

### 17. Owed to honesty: spectral gluing (LTSA / diffusion maps) as a closed-form rival to learned gluing
- **Slot:** gluing (adversarial baseline).
- **Sources:**
  - https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction: LTSA aligns local tangent coordinates
    into global ones by one eigenproblem.
  - https://en.wikipedia.org/wiki/Diffusion_map: eigenvectors of the random-walk Laplacian as coordinates.
- **Mechanism:** fit charts as usual, then glue by LTSA on the chart outputs instead of the learned φ_k.
- **Why it matters:**
  - If closed-form alignment matches learned gluing, our gluing claim is owned, and post-hoc at that.
  - It is the C2-style "post-hoc Mapper" baseline, transposed to gluing. Better we run it than a reviewer.
- **Test:** T2 discontinuity rate of LTSA-glued charts against learned gluing at equal charts.
- **Reading:** learned must win by ≥ 20% relative, or the gluing claim moves to "online, no eigenproblem".
- **Prior:** 50% that LTSA ties, which is bad for us.

### 18. Communication-through-coherence: charts exchange messages only when phase-aligned
- **Slot:** gluing (which pairs talk), laziness in the loop.
- **Sources:**
  - https://en.wikipedia.org/wiki/Neural_oscillation: binding by synchrony at about 40 Hz. Information
    passes between populations when their oscillations are phase-aligned. One theta cycle sequences 4–8
    ensembles.
  - https://en.wikipedia.org/wiki/Kuramoto_model: r ∝ √((K−K_c)/K_c) just above K_c, and the mean-field
    update is O(N).
- **Mechanism:**
  - In the idea-1 loop, a pair (j, k) contributes to the gluing step only if cos(θ_k − θ_j − χ_jk) > c.
  - Coupling uses the mean field (r, ψ) of the selected set: O(m) per step, not O(m²).
- **Why it beats today:** the overlap term couples every co-selected pair. Coherence-gating drops pairs that
  disagree, so a wrong chart cannot drag the glue. The loop then costs O(m).
- **Test:** T2 discontinuity rate ≤ ungated Kuramoto at ≤ 0.5× loop MACs.
- **Kill:** worse at equal MACs.
- **Prior:** 25%.

### 19. SOM neighbour coupling on the chart lattice: makes TO6's rays meaningful
- **Slot:** laziness in K (lattice), anti-collapse.
- **Source:** https://en.wikipedia.org/wiki/Self-organizing_map. W_v ← W_v + θ(u,v,s)·α(s)·(x − W_v), with
  a shrinking grid neighbourhood θ. A folded map is a detectable topological defect.
- **Mechanism:**
  - Charts sit on a √K×√K lattice (TO6).
  - The winner's lattice neighbours get a decaying share of its key update, so adjacent lattice cells cover
    adjacent state regions.
  - A ray through the lattice then selects a *contiguous* strip of state space, not random charts.
- **Why it beats today:** TO6's rays only beat product keys if the lattice is ordered, and today nothing
  orders it.
- **Test (TO6, K = 256 and 1024):** loss with SOM coupling ≤ product keys, while without coupling it is >
  product keys. The coupling is the reason rays work.
- **Kill:** no difference with and without coupling.
- **Prior:** 30%.

---

## Proposed ASMODEOUS slot fill (my half, to merge with Kutner's)

| Slot | Pick | Runner-up |
|---|---|---|
| State | complex ψ on CP^{N−1} (ORIGIN O1), read as grid-module phases (2) | real R^{D_g} |
| Decider | Kutner's POVM isometry (exact, lazy), with an LSH shortlist (13) | Born top-m |
| Laziness in K | POVM + LSH (13); Bélády cache for K ≫ VRAM (14) | product keys |
| Laziness in time | per-chart send-on-delta (4) plus Kutner's Zeno gap at router level | global EventDriven |
| Chart dynamics | Koopman unitary per chart, DMDc init (8); Kutner's Cayley/rotor is the same object | MLP |
| Gluing | Kutner's gauge links, precision-weighted (9); phases relaxed by Kuramoto (1) | φ_k into R^{D_g} |
| Observer loop | Kuramoto relaxation = halting dynamics (1); Hopfield energy (7) or CPTP (Kutner #3) for chart weights | FP/TT |
| Anti-collapse | Sinkhorn equipartition (3) + Fubini–Study target (Kutner #14) | SIGReg + Switch |
| Training | rare-mode replay (5); MDL growth (12) | fixed K |
| Readouts | vortex Chern (1), Morse χ (6), sheaf w₁ (10), nerve-gate validity (11) | nerve Betti |
| Honesty arms | LTSA gluing (17), Kutner's Hopf single chart (#8) | — |

One sentence: **ASMODEOUS** is a phase state on CP^{N−1}, measured lazily by an exact POVM.
- Only observed charts compute, and each one recomputes only on its own events.
- Each chart steps unitarily.
- Charts glue through gauge links whose phases are relaxed by a Kuramoto loop that halts on an energy.
- Topology is read as whatever that loop cannot relax away: vortices (c₁), Euler characteristic (χ) and the
  orientation sheaf (w₁).

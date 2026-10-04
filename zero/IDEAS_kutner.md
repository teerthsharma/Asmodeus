# IDEAS: Kutner (physics and geometry half of the hunt)

**Sources.** 40 English Wikipedia articles, fetched 2026-10-04 by 4 sonnet nurses. Each fact is
summary-level (the fetch tool summarises the page), so check the article before citing it in a paper.
Thirteen holds the computation, neuro and topology-data half.

**What each idea is measured against.** These are today's options in `zero/`:
- the softmax/Born top-m decider;
- the GRU context;
- the free gluing maps phi_k into R^{D_g};
- Cayley local steps;
- the TT and FP observer loops;
- nerve Betti and sign holonomy.

**Fields in each entry:** slot · source (URL, one fact) · why it beats today · one falsifiable test ·
prior (a guess).

---

## Top five (my picks for ASMODEOUS)

### 1. POVM decider from an isometry: exact probabilities with no partition function, so lazy in K
- **Slot:** decider, laziness in K.
- **Source:** https://en.wikipedia.org/wiki/POVM. A POVM generalises projective measurement, and its
  outcome count may exceed the Hilbert dimension. Prob(i) = tr(ρF_i), F_i ⪰ 0, Σ F_i = I.
  Also https://en.wikipedia.org/wiki/Gleason%27s_theorem: for dimension ≥ 3, the Born rule is the only
  non-contextual additive probability on projections.
- **Mechanism:**
  - Make chart keys the rows of an isometry V ∈ C^{K×N} with V†V = I. This costs O(K N²) per step by
    QR or Cayley, and N is about 16.
  - Then F_k = v_k v_k† and p_k = |v_k† ψ|².
  - Σ_k p_k = ‖ψ‖² = 1 holds exactly, so **no Z is summed over K**.
  - Top-m by approximate max-inner-product search, or product keys, then gives *exact* probabilities for
    the charts it reads. "Observe nothing" falls out as 1 − Σ_{read} p_k.
- **Why it beats today:**
  - Softmax needs Z over all K. Product keys only make the top-m *selection* sub-linear: the
    renormalised weights are wrong in absolute terms, and T3's "lazy within 1% of soft" is an
    approximation gap.
  - Here the gap is zero by construction.
  - Gleason says this is the *only* non-contextual decider. T6 then has a clean null: any order effect
    has to come from the context dynamics, not from the decider.
- **Test:** at K ∈ {256, 1024, 4096}, |p_k(lazy) − p_k(dense)| = 0 to float eps for every read chart
  (unit test). The T3 loss of POVM-lazy must be ≤ softmax-soft + 0.5%, at routing FLOPs ∝ √K.
- **Kill:** T3 loss more than 1% worse than softmax at equal FLOPs.
- **Prior:** 40%.

### 2. Gauge-link gluing: transition functions on nerve edges instead of a global embedding
- **Slot:** gluing, readout.
- **Sources:**
  - https://en.wikipedia.org/wiki/Connection_(principal_bundle): a connection is a Lie-algebra-valued
    1-form, and its curvature is Ω = dω + ½[ω∧ω].
  - https://en.wikipedia.org/wiki/Wilson_loop: Wilson (1974) built lattice gauge theory from link
    matrices, with plaquette traces as gauge-invariant observables.
- **Mechanism:**
  - Each co-selected chart pair (j, k) learns a link g_jk(z) ∈ O(d_c) or U(d_c), via Cayley of a skew
    matrix.
  - The state is transported chart to chart by links. There is no φ_k into a shared R^{D_g}.
  - Plaquette W_jkl = g_jk g_kl g_lj on nerve triangles. Holonomy along loops is the product of links.
- **Why it beats today:**
  - Σ w_k φ_k(z_k) into R^{D_g} *is* a global embedding, which is Whitney's route (PROMPT §3 caveat).
    The atlas claims to be needed exactly where a global embedding is the alternative, so its gluing
    should not quietly be one.
  - Links keep only local frames (a Čech cocycle).
  - The TO7 sign holonomy becomes det of the link product around the loop: no separate learned frame
    field, and no hand-built fixture.
- **Test (t2 vs klein, lift):**
  - Cocycle residual ‖W_jkl − I‖ < 0.05 on contractible triangles.
  - det(Π links) = −1 on Klein's orientation-reversing generator and +1 on every T² loop, on ≥ 95% of
    held-out loops over 3 seeds.
- **Kill:** sign accuracy < 80%, or the same sign pattern on T² and Klein.
- **Prior:** 35%.

### 3. CTC loop as a CPTP map with ε-depolarisation: a fixed point always exists and is unique
- **Slot:** observer loop.
- **Sources:**
  - https://en.wikipedia.org/wiki/Closed_timelike_curve: under Deutsch's prescription the page states
    classical and quantum computation coincide (PSPACE). Deutsch's ρ_CTC = Tr_sys[U(ρ_in ⊗ ρ_CTC)U†] is a
    fixed point of a CPTP map. That last point is standard background; the nurse noted the page omits the
    formula.
  - https://en.wikipedia.org/wiki/Novikov_self-consistency_principle: only globally self-consistent
    histories are allowed (Novikov, mid-1980s).
- **Mechanism:**
  - The loop acts on a K×K (low-rank) density matrix over charts, not a weight vector:
    ρ ← (1−ε) Σ_a M_a ρ M_a† + ε I/K.
  - M_a are Kraus operators built from the candidates' predicted futures.
  - CPTP plus ε > 0 makes the map a strict contraction in trace distance, so Banach gives a unique
    fixed point and geometric convergence at rate ≤ (1−ε).
- **Why it beats today:**
  - Foreman's loop is an undamped iteration that *reports* non-convergence (`ctc_residual`). The TT loop
    halts on set equality, which can cycle.
  - Here convergence is a theorem, so TO2's "≥ 99% converge" bar is met by construction. The open
    question becomes accuracy, not stability.
- **Test:**
  - Unit test: 1,000 random Kraus sets converge to 1e-8 within ⌈log(1e-8)/log(1−ε)⌉ iterations.
  - Bed test: T2 h = 10 error ≤ Foreman FP loop at equal loop FLOPs.
- **Kill:** h = 10 error > 1.05× the FP loop.
- **Prior:** 30%.

### 4. Amplitude amplification inside the observer loop: quadratically fewer iterations
- **Slot:** observer loop, laziness in time.
- **Source:** https://en.wikipedia.org/wiki/Amplitude_amplification. Brassard and Høyer (1997)
  generalised Grover. About π/(4θ) iterations suffice, success = sin²((2n+1)θ), and each step is two
  reflections.
- **Mechanism:**
  - Each loop step is Q = −S_ψ S_good on the candidate amplitudes. S_good flips the sign of candidates
    whose predicted futures agree with the consensus (dev < τ).
  - For a consistent chart with prior weight p, this takes O(1/√p) iterations, against O(1/p)-like
    re-weighting.
  - Overshoot is fixed by the fixed-point (π/3) variant [K: Yoder–Low–Chuang 2014, not fetched].
- **Why it beats today:**
  - The TT/FP loops re-weight multiplicatively. When the right chart starts unlikely, which is exactly
    the time-traveller's use case of a rare future-consistent chart, they need many iterations.
- **Test (synthetic, unit-level):** log-log slope of iterations-to-w_good ≥ 0.9 against prior p ∈
  [1e-4, 1e-1] is ≈ −0.5 for Q and ≈ −1 for FP/TT.
- **Bed test:** on swing-up, equal planning MACs and a success rate ≥ TT.
- **Kill:** slope > −0.7.
- **Prior:** 25% on beds, 90% on the synthetic test.

### 5. Zeno / adiabatic laziness in time: re-route only when the chart gap closes
- **Slot:** laziness in time.
- **Sources:**
  - https://en.wikipedia.org/wiki/Quantum_Zeno_effect: frequent measurement suppresses transitions, with
    short-time survival ≈ 1 − (t/τ)².
  - https://en.wikipedia.org/wiki/Adiabatic_theorem: Born and Fock (1928) showed the state follows its
    instantaneous eigenstate if the change is slow relative to the gap.
- **Mechanism:**
  - Keep the chart set S_t. Each step, compute only the m read charts' own Born scores (O(m), not O(K)).
  - Run the full decider when the gap (top read score minus an O(1) learned exit threshold) falls below
    0, i.e. near a chart boundary.
  - The adiabatic condition sets the skip length.
- **Why it beats today:** every arm routes every step. On smooth trajectories most steps stay in-chart,
  so this is lazy in time on top of lazy in K.
- **Test (T2 beds):**
  - Full router calls per step ≤ 0.3, with h = 1 error ≤ +1% vs routing every step.
  - Zeno check: chart dwell time does not grow without bound as the measurement rate rises (no
    freezing). It is reported against routing rates 1, ½ and ¼.
- **Kill:** to stay within +1% error, more than 0.6 of steps need the router.
- **Prior:** 45%.

---

## Further ideas

### 6. Weak-measurement decider: a differentiable collapse with learned strength
- **Slot:** decider.
- **Source:** https://en.wikipedia.org/wiki/Weak_measurement. Every measurement disturbs (Busch). Weak
  measurement trades information for disturbance, with Kraus M ≈ I + λ·O.
- **Mechanism:** among the M preselected charts, ψ ← Mψ/‖Mψ‖ with learned λ ≥ 0. As λ → ∞ this recovers
  the projective top-m.
- **Beats:** the hard top-m has no gradient through the selection.
- **Test:** T2 h = 10 error at equal FLOPs ≤ top-m, and the learned λ is reported.
- **Kill:** if λ → ∞ on ≥ 3/4 beds, SGD prefers collapse and the idea is dead.
- **Prior:** 20%.

### 7. Lindblad observer: learned per-chart dephasing decides where coherence pays
- **Slot:** decider dynamics, anti-collapse.
- **Sources:**
  - https://en.wikipedia.org/wiki/Lindbladian: GKSL dynamics are trace-preserving and completely positive
    when the rates are ≥ 0.
  - https://en.wikipedia.org/wiki/Quantum_decoherence: environment entanglement kills off-diagonals,
    giving a pointer basis.
- **Mechanism:** ρ over charts evolves as −i[H(c), ρ] + Σ γ_k(P_k ρ P_k − ½{P_k, ρ}), with γ_k = softplus.
- **Beats:** Arm Q has one global coupling. This learns *which* charts stay coherent.
- **Test (prediction):** the learned γ is low for near-duplicate chart pairs and high elsewhere, with a
  Spearman correlation between pairwise γ and the duplicate score of < −0.3.
- **Kill:** every γ → large, in which case softmax suffices and Arm Q is dead as well.
- **Prior:** 20%.

### 8. Hopf-bundle state: one smooth chart on the total space
- **Slot:** state, chart. This is an adversarial idea against C1.
- **Sources:**
  - https://en.wikipedia.org/wiki/Hopf_fibration: Hopf (1931), S³ → S² with S¹ fibres,
    p(z₀, z₁) = (2z₀z̄₁, |z₀|² − |z₁|²).
  - https://en.wikipedia.org/wiki/Bloch_sphere: pure states are the sphere and mixed states are the
    interior.
- **Mechanism:** the latent is ψ ∈ S³ ⊂ C², read out through the Hopf map. S² needs 2 charts, but its
  bundle S³ is a Lie group (SU(2)) with a global frame.
- **Why it matters:** if one bundle chart is continuous on S² (and unit quaternions mod ± on SO(3)), then
  P-COUNT's "the atlas is needed" has an owned alternative. Better we find it than a reviewer.
- **Test:** on the S² bed, a single Hopf-latent arm reaches a discontinuity score of 0 at the T2 ε and δ,
  at equal FLOPs.
- **Reading the result:** success weakens C1 for S², so the claim has to move to readout or compute.
  Failure is evidence for C1.
- **Prior:** 50% that it works, which is bad for C1.

### 9. Symplectic latent predictor for second-order beds
- **Slot:** dynamics.
- **Source:** https://en.wikipedia.org/wiki/Symplectic_integrator. Verlet is second-order symplectic:
  half kick, drift, half kick.
- **Mechanism:** the chart latent is (q, p), with p += h/2 F(q); q += h p; p += h/2 F(q). F is a learned
  MLP, and the damping is a learned scalar.
- **Beats:** residual MLP predictors drift in energy over the h = 30 planning horizon. T5 swing-up is
  exactly energy pumping.
- **Test (swingup):**
  - h = 30 latent error ≤ 0.7× the residual-MLP predictor at equal FLOPs.
  - T5 success ≥ the residual predictor.
- **Kill:** ≥ 0.95×.
- **Prior:** 35%.

### 10. Lie-group (Cayley) chart steps with rotor parametrisation for SO(3)
- **Slot:** dynamics.
- **Sources:**
  - https://en.wikipedia.org/wiki/Lie_group_integrator: Y_{n+1} = exp(hΩ_n) Y_n stays on the group exactly.
  - https://en.wikipedia.org/wiki/Cayley_transform: Q = (I−A)(I+A)⁻¹ is orthogonal for skew A.
  - https://en.wikipedia.org/wiki/Geometric_algebra: a rotor R = exp(−B/2) acts by sandwich.
- **Mechanism:** the SO(3) chart state is a rotor (Spin(3) ≅ S³). The predictor outputs a bivector, and
  Cayley gives the step.
- **Beats:** an MLP in R^{d_c} leaves the group, so its error compounds.
- **Test:** SO(3) bed h = 10 geodesic error ≤ 0.8× the MLP chart predictor at equal FLOPs.
- **Prior:** 30%.

### 11. Syndrome-gated recompute (stabiliser code)
- **Slot:** laziness in time, readout.
- **Sources:**
  - https://en.wikipedia.org/wiki/Toric_code (served the Surface code page): stabilisers A_v and B_p
    commute, and violations are anyons.
  - https://en.wikipedia.org/wiki/Quantum_error_correction: syndromes infer errors without reading the
    logical data.
- **Mechanism:** a cheap parity check is the cocycle residual on the current nerve triangle (idea 2). The
  expensive gluing or observer loop is recomputed only when the syndrome is nonzero.
- **Test:** ≥ 2× fewer loop invocations at ≤ +1% h = 10 error.
- **Kill:** the syndrome is nonzero on > 50% of steps.
- **Prior:** 25% (needs idea 2).

### 12. Renormalisation-group charts: coarse charts refine on demand
- **Slot:** laziness in K.
- **Source:** https://en.wikipedia.org/wiki/Renormalization_group. Couplings run with scale (QED
  1/137 → 1/127), and block-spin coarse-graining flows to fixed points.
- **Mechanism:** a tree of charts. A fine child chart is read only when the coarse chart's predicted
  error exceeds a learned threshold.
- **Beats:** product keys split the key space and ignore error. This spends K where prediction is hard.
- **Test:** at K = 1024, loss ≤ product keys at ≤ 0.5× routing FLOPs.
- **Kill:** worse at equal FLOPs.
- **Prior:** 20%.

### 13. Light-cone mask for the observer loop
- **Slot:** laziness in the loop.
- **Sources:**
  - https://en.wikipedia.org/wiki/Light_cone: influence lies only inside the cone.
  - https://en.wikipedia.org/wiki/G%C3%B6del_metric: CTCs exist only beyond a critical radius, r_c.
- **Mechanism:** a candidate chart enters the TT/FP re-score only if it is reachable within h nerve hops
  from the start set (the cone). Loops run only where the predicted rollout returns near its start (a
  closed curve, i.e. "r > r_c").
- **Test:** halt sets identical to the full loop on ≥ 99% of rollouts, at ≤ 0.5× loop MACs.
- **Prior:** 30%.

### 14. Fubini–Study geometry for the complex state
- **Slot:** anti-collapse, chart.
- **Source:** https://en.wikipedia.org/wiki/Fubini%E2%80%93Study_metric. It is the natural metric on
  CP^n = S^{2n+1}/U(1), with distance = arccos|⟨ψ|φ⟩|.
- **Mechanism:** SIGReg's Gaussian target is wrong for a unit complex state. Use a Haar-uniform target on
  CP^n instead: a sliced test of |⟨u|ψ⟩|² against Beta(1, n−1) for random u.
- **Test:** the effective rank of ψ under the 0.1× anti-collapse weight is ≥ 1.2× the Gaussian-SIGReg
  arm's (T4a protocol).
- **Prior:** 25%.

### 15. Quantum walk on the nerve for long-range chart transport (planning)
- **Slot:** dynamics across charts.
- **Source:** https://en.wikipedia.org/wiki/Quantum_walk. Spread is linear in t against √t classically,
  with i dψ/dt = Lψ.
- **Mechanism:** goal-directed chart proposals for pointmaze come from e^{−iLt} on the learned nerve
  graph, read by Born.
- **Test:** pointmaze-large planAll ≥ BFS on the same learned graph at equal compute.
- **Prior:** 10%.

### 16. Anyonic exchange phase as the T6 order statistic
- **Slot:** readout.
- **Sources:**
  - https://en.wikipedia.org/wiki/Anyon: exchange multiplies the wavefunction by e^{iθ}, with θ between
    boson and fermion.
  - https://en.wikipedia.org/wiki/Braid_group: B_3 is infinite and non-abelian, and isomorphic to the
    trefoil knot group.
- **Mechanism:** for a complex-state arm, θ = arg⟨ψ_AB|ψ_BA⟩ after the two question orders. θ = 0 means
  order commutes in amplitude, even when probabilities agree.
- **Beats:** QQ and order work on probabilities only. θ catches phase-only contextuality that
  probabilities miss.
- **Test:** a context-free observer gives θ = 0 exactly (unit test). Report θ with a bootstrap CI for the
  hybrid and phase-sphere arms. Measurement only.
- **Prior:** n/a.

---

## For the ASMODEOUS design session

Slot fill as proposed:
- **State:** complex ψ (phase-sphere).
- **Decider:** POVM isometry (1), which is exact and lazy.
- **Gluing:** gauge links (2), whose cocycle syndromes gate recompute (11).
- **Observer loop:** CPTP + ε (3), with amplitude amplification (4) when the prior is small.
- **Laziness in time:** Zeno / adiabatic gap (5).
- **Dynamics:** Cayley/rotor (10), and symplectic (9) on second-order beds.
- **Readouts:** Chern from ψ, sign holonomy from links, θ for T6.

Ideas 8 and 16 are owed to honesty. Idea 8 is the strongest attack on our own C1.

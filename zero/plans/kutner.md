# Kutner — plan for after the 3-hour session (2026-10-04)

Stance: the cheap, falsifiable experiment. Sources: English Wikipedia, fetched 2026-10-04.
`Born_machine` returns 404 (no article); Born-machine facts below come from the Tensor network and MPS articles.

## Findings (URL, fact, implication for QF-LM)

1. https://en.wikipedia.org/wiki/Matrix_product_state
   Left-canonical MPS tensors satisfy Σ_s A^{s†} A^{s} = I.
   **Implication:** this is exactly the completeness condition Σ_v K_v†K_v = I. QF-LM with no think step and no null
   *is* a translation-invariant left-canonical MPS Born machine with bond dimension D = N (Miller et al. 2021 own this).
   The no-think/no-null ablation is the owned baseline, and the claim of novelty rests on the think and null slots alone.

2. https://en.wikipedia.org/wiki/POVM
   Kraus operators are not unique: M_i = W√F_i gives the same statistics for any unitary W but a different post-measurement state.
   **Implication:** K_v = diag(√M_v)Q = Q·√F_v with F_v = Q†diag(M_v)Q, so W = Q for every token. A token-dependent W_v
   is the most expressive write that keeps p(v) unchanged. It is also an input embedding in disguise, so the honest "+embed"
   arm is K_v = W_v diag(√M_v) Q. That arm, and not nn.Embedding, is the one write=read has to beat.

3. https://en.wikipedia.org/wiki/POVM (same page)
   An extremal POVM on C^d has at most d² outcomes.
   **Implication:** an extremal head on N = 64 could carry 4096 outcomes, but the diag(√M)Q head is far from extremal. All its
   effects F_v commute because they are diagonal in the Q basis. Within one step the measurement is classical (a Bayes filter on
   |Qψ|²), and every bit of non-classicality comes from Q and think *between* steps. Test: replace Q by a permutation at
   equal N. If ppl does not move, the "quantum" part is doing nothing.

4. https://en.wikipedia.org/wiki/Quantum_instrument
   An instrument is CP maps {E_x} with Σ_x E_x trace-preserving; p(x|ρ) = tr E_x(ρ); ρ_x = E_x(ρ)/tr E_x(ρ).
   **Implication:** the per-token step is fully specified. The only open design choice is the post-emission state when
   emission is a mixture over s null rounds. Under the instrument formalism that state is the *mixed* state Σ_s E_x∘E_null^s(ρ),
   not a renormalised superposition of the branches. A pure-state implementation of that mixture is an approximation and has
   to be named as one (see step 3).

5. https://en.wikipedia.org/wiki/Quantum_Zeno_effect
   Under frequent measurement, the transition probability per interval scales as t², so evolution freezes as the interval goes to 0.
   **Implication:** "null = still thinking" with a small-angle think unitary measured every round is a Zeno trap. A state
   with p_null near 1 stays there, and the s ≤ 4 cap then forces emission from an unchanged state. Falsifier: plot p_null(s)
   against the think-angle scale on the trained model. Flat p_null across s means think is frozen and the adaptive-compute
   claim (b) dies.

6. https://en.wikipedia.org/wiki/Unitary_matrix and https://en.wikipedia.org/wiki/Givens_rotation
   U(N) has dimension N² (real parameters). A general N×N rotation needs N(N−1)/2 Givens rotations, each costing O(N).
   **Implication:** a single butterfly with N/2·log₂N complex 2×2 blocks has about 2N log₂N real parameters, under 1% of U(N) at N = 2048.
   Q is a severely restricted unitary. Whether that hurts is an empirical question. Dense Q at N = 256 is a cheap control
   (one matmul per token).

7. https://en.wikipedia.org/wiki/Clos_network (the Beneš section; the Beneš_network URL redirects there)
   A Beneš network has 2·log₂N − 1 stages of N/2 2×2 switches and is rearrangeably non-blocking: it realises any permutation.
   The butterfly network (https://en.wikipedia.org/wiki/Butterfly_network) has log₂N stages with diameter log₂N and cannot.
   **Implication:** a log₂N-stage butterfly Q cannot even express every basis permutation. A Beneš-depth Q costs 2× and can.
   That is a one-line ablation: stages = log₂N vs 2log₂N − 1.

8. https://en.wikipedia.org/wiki/One-way_quantum_computer
   MBQC is universal using only measurements on a resource state, because later measurement bases depend on earlier outcomes
   (feed-forward) and outcome-dependent Pauli "byproduct" corrections are applied.
   **Implication:** "measure to remember" with ψ-dependent think angles is owned vocabulary: adaptive measurement plus
   byproduct correction. It is a strong precedent that measurement alone can carry computation. It also says that the
   token-dependent correction W_v (finding 2) is the standard way to make the scheme deterministic, so it needs its own arm.

9. https://en.wikipedia.org/wiki/Quantum_walk
   Discrete-time quantum walks (coin plus shift unitaries) spread ballistically (σ ∝ t, against √t classically) and never converge to a stationary distribution.
   **Implication:** think steps are a quantum walk on N basis states. They mix fast but cannot forget, because they are unitary.
   All forgetting in QF-LM must come from the Lüders filter, and the state never decays toward a prior. Readout: the
   entropy of |Qψ|² against t across long contexts. A monotone collapse to low entropy means the filter is over-committing,
   and that predicts the long-context ppl gap against the GRU.

10. https://en.wikipedia.org/wiki/Born_rule
    For POVMs, p(i) = tr(ρF_i) = ⟨ψ|F_i|ψ⟩, and Gleason's theorem derives the Born rule from the Hilbert-space structure of measurements.
    **Implication:** any complete Kraus head is Born-form, so the head is not where novelty lives. Together with the House bound
    (Σ_v max_ψ p(v) ≤ N), this means N must scale with |V| (N ≥ |V|/2 for the diag head), which fixes the cost floor of the per-token step.

11. https://en.wikipedia.org/wiki/Tensor_network
    MPS/tensor-train models are used for generative modelling, and generic PEPS contraction is #P-complete.
    **Implication:** the chain structure keeps exact likelihood at O(T·N log N). No 2-D or tree generalisation of
    QF-LM should be proposed: exact likelihood would be lost.

## Plan after the session (≤ 10 steps, each one falsifiable and cheap)

1. Land PLAN item 3 GREEN (zero/qflm, tests/test_qflm.py), with mutation checks recorded in house-events.jsonl.
2. Add the owned-baseline arm explicitly: no-think + no-null QF-LM = uniform MPS Born machine (finding 1). It is a separate
   row in the item-5 bake-off and not merely an "ablation".
3. Replace nn.Embedding in the "+embed" arm with a token-dependent unitary W_v (finding 2; a low-rank butterfly per token).
   Criterion (a) is judged against this arm.
4. Commutation control (finding 3): Q → learned permutation (Sinkhorn) at equal N. If ppl is within 2%, the quantum
   interference is not doing the work.
5. Depth control (findings 6–7): Q stages log₂N against 2log₂N − 1 (Beneš), plus dense Q at N = 256. Run 1 seed on fixture TinyStories, 30 CPU-min each.
6. Zeno check (finding 5): on the M0 checkpoint, report p_null(s) for s = 0..4 and the think-angle scale. Bar: mean p_null
   must drop by ≥ 0.1 across s on ≥ 30% of positions, or the null slot is uncredited.
7. Exact-mixture check (finding 4): compare the pure-state post-emission rule with the exact density-matrix mixture at N = 16
   on 1k tokens. If NLL differs by > 1%, use the mixture (O(N²) only at small N) or cap s at 1.
8. Memory readouts on M0, as one CPU script: entropy H(|Qψ_t|²) against t (finding 9), the pentimento probe comparing phase and modulus (H1), and the Preisach wipe-out curve (H2). Run it before any long-context claim.
9. Write the T1 owner rows for MPS Born-machine LMs, MBQC feed-forward and HQMMs into OWNERS.md before any novelty claim. The item-7 exchange-phase readout carries a shuffled-pair control (H3).
10. Run the item-5 bake-off with arms {QF-LM, MPS-baseline, +W_v, softmax head, no-think, no-null, GPT, GRU} under the PREREG bars.

## Wander (no agenda)

Done on 2026-10-04, about 20 min, en.wikipedia.org only, 16 articles. Each hop followed whatever pulled hardest; there was no search goal.

**Path.** Three branches from Physics:
- **Palimpsest branch (deepest, 5 hops):** Physics → Archimedes_Palimpsest → Palimpsest → Novgorod_Codex → Dendrochronology → Miyake_event.
  Side branch: Palimpsest → Pentimento.
- **Catastrophe branch (4 hops):** Physics → Catastrophe_theory → ADE_classification → Monstrous_moonshine → Leech_lattice.
  Side branch: Catastrophe_theory → Hysteresis → Preisach_model_of_hysteresis, and Hysteresis → Kuleshov_effect.
- **Turbulence branch:** Physics → Turbulence → Swing_bowling.

### 5 most surprising things
1. **Novgorod Codex.** It is a "hyper-palimpsest": the stylus cut through the wax into the wood, so every erased text is still there as scratches under the current one. Zaliznyak recovered layered texts from it. Hardly anyone else has learned to repeat the reading, so only small parts are peer-reviewed. A memory that looks erased is still there, but reading it back is so costly that no one can check it.
2. **Reverse swing.** An old cricket ball swings toward its shiny side, which is the reverse of normal swing. Roughness decides which side's boundary layer turns turbulent first. Golf-ball dimples work the same way: added roughness lowers drag.
3. **Miyake events.** A one-year cosmic-ray ¹⁴C spike (774 CE, 993 CE and three others) is recorded in every tree alive at the time, worldwide. It gives one synchronisation mark that dates the Viking site at L'Anse aux Meadows to exactly AD 1021.
4. **Preisach model.** Hysteresis, a persistent rate-independent memory, is modelled as a *linear superposition* of two-state relays. The memory state is a staircase line in a triangle. When a later, larger input excursion dominates the earlier history, it wipes that history out.
5. **Kuleshov effect.** The same neutral face reads as hungry, grieving or desiring depending on the shot placed next to it. No original footage survives. The 1992 replication (137 participants) failed, and the effect was later backed by Mobbs et al. 2006 (fMRI). Also on this branch: 196884 = 196883 + 1. McKay's coincidence turned out to be the Monster group.

### Instinct connections to Asmodeous. Every item here is a hunch, not a finding.
- **H1 (pentimento/palimpsest), hunch.** The Lüders write overwrites |Qψ|², which is what p can see. The phase of ψ may keep earlier tokens anyway, the way scratches stay in the wood, and a later think unitary could make them visible again (paint going transparent). Falsifier: on the M0 checkpoint, fit linear probes for token t−k from ψ_t using (i) the moduli |Qψ| only and (ii) the full complex ψ. If (ii) beats (i) by more than 5 points at k ≥ 4, memory lives in phase. If it does not, the "quantum" state is a classical belief vector and the MPS-baseline reading (finding 1) is the whole story.
- **H2 (Preisach wiping-out), hunch.** The diagonal Lüders filter acts like a bank of relays, so a confident emission (p > 0.9) should wipe earlier history the way a large excursion clears the Preisach staircase. This is the mechanism behind the Zeno/over-commit risk (findings 5 and 9). Falsifier: compare probe accuracy for tokens before a confident emission with the accuracy for tokens before an unconfident one. If the drop is larger after confident emissions, think has to undo the wipe-out, and that is a reason for the null slot to exist.
- **H3 (Kuleshov), hunch.** Meaning set by the neighbouring shot is non-commutativity. The item-7 exchange-phase readout arg⟨ψ_AB|ψ_BA⟩ is a Kuleshov measurement. The 1992 failed replication is the warning: the exchange phase has to beat a shuffled-pair control (random A,B with no clause relation) before anyone reports it.
- **H4 (Miyake/cross-dating), weak hunch.** The per-token think-step count is a "ring-width" series. Two paraphrases should cross-date, meaning their step series align. This is a cheap item-7 readout, not a claim.
- **H5 (reverse swing / dimples), weak hunch.** Deliberate asymmetric roughness can reverse the direction of drift. A small random phase jitter on Q during training might break repeat loops better than larger think angles. Not tested. Only worth a look if the repeat-escape test is borderline.

### Plan changes after the wander
Step 8 becomes: "Memory readouts on M0. Entropy H(|Qψ_t|²) vs t (finding 9), the pentimento phase-vs-modulus probe (H1), and the Preisach wipe-out curve (H2). One CPU script, run before any long-context claim."
Step 9 gains: "the item-7 exchange-phase readout carries a shuffled-pair control (H3)."
All other steps are unchanged.

## Difference (Seal's direction, 2026-10-04): move away from every owner, using Seal's own prior art

**Owner map for QF-LM today** (OWNERS.md rows 184, 185, 210, 220):
- **write = read.** Owned by Pestun–Terilla–Vlassopoulos 2017 (1711.01416), in formalism and with no experiments.
- **Unit-norm chain with no think step.** This is an MPS Born machine (finding 1).
- **Halting likelihood summed over steps.** This sits close to PonderNet (2107.05407) and ACT (1603.08983). OWNERS separates them only because QF-LM sums over *null measurement records* inside an instrument.
- **The M0 smoke shows that gap is thin in practice.** Every token ran exactly 3 rounds. So null-halting currently behaves like a fixed-depth PonderNet with a flat halting probability.

**What to push.** The parts nobody owns, taken from Seal's sources. Every row below needs a pre-result amendment and is not adopted until Wilson files it.

| # | Differentiating choice | Seal's prior-art source | Owner it moves away from | Falsifier (cheap) |
|---|---|---|---|---|
| D1 | **CTC halting replaces null-mass halting.** A think round iterates the self-consistency map p ← Born(U(p)ψ): the angles depend on the *anticipated* outcome distribution p, i.e. the future. "Still thinking" means the outcome has not reached its own fixed point. Halt when ‖p_{s+1} − p_s‖₁ < ε. Null keeps its instrument role, carrying the probability that the loop has not closed. | ORIGIN O2 (Deutsch-CTC fixed point x = F(x)), O3a, Gödel figure "halt before start"; TO2/TO3 | PonderNet/ACT halt on a learned scalar. Here halting is a convergence property of the state, so it cannot be flat by construction. | Rounds-run vs surprisal Spearman ≥ 0.3 (PLAN 5b), where the M0 smoke gave constant 3.0. Kill: the fixed-point residual fails to converge on > 5% of tokens (TO2 bar). |
| D2 | **Halt before start.** A one-shot head predicts the CTC fixed point p* from ψ before any round runs. Rounds then run only when the head's predicted residual exceeds ε, so compute fires on surprise. | ORIGIN O3a; A.E.T.H.E.R. event-driven compute (PLAN; Aether line) | ACT/PonderNet pay for every step up to the halt | TO3 bar: top-1 agreement with p* ≥ 95% at ≤ 1/5 of loop FLOPs; kill at < 80% |
| D3 | **The observer chooses the measurement basis at every round, round 0 included.** K_v(ψ) = diag(√M_v)·Q·B(\|ψ\|²), where B is a butterfly whose angles come from \|ψ\|². For fixed ψ the instrument is still complete, so Σp = 1 holds, but the instrument is now ψ-dependent: a nonlinear filter. | PRIOR_QUANTUM_NOTES L-Q1 ("no circuit chooses what to measure based on an earlier outcome", never exercised); aether-link basis-rotation feedback loop (R-A1) | MPS, HQMM and 1711.01416 all use a fixed linear dictionary of Kraus operators | Arm "fixed basis" (B = I) vs D3 at equal MACs. Kill if within 1% ppl. |
| D4 | **Butterfly effect bonded via topology.** The think dynamics are deliberately chaotic: angles drawn Uniform(−π, π) (already the init), measured λ₁ > 0. The emitted token, the integer readout, must still be stable. | ORIGIN O4 / TO4 | Unitary RNNs (uRNN) aim at λ = 0 (norm-preserving and stable) and claim no chaos-plus-stable readout | Bar: finite-time λ₁ > 0 over the think rounds, and an argmax-token flip rate < 1% under ψ perturbations at 1e-3 |
| D5 | **Phase becomes time.** Report the accumulated geometric phase Σ_s arg⟨ψ_s\|ψ_{s+1}⟩ of each think loop next to its rounds. The prediction is that it is quantised near multiples of π on halted tokens (π law). This is a readout only, not a training signal, and it carries a shuffled-pair control (H3). | ORIGIN O3b, O6 (π law / exchange holonomy), O7/O8 | None (unowned readout) | Kill: the phase histogram on halted tokens is indistinguishable from the shuffled control (KS p > 0.05) |
| D6 | **Certified halt.** If the CTC map of D1 is a contraction (σ_max of its Jacobian < 1), halting is guaranteed and the round count is bounded by log(ε)/log(ρ). If it is not certified, the token runs to s_max and is flagged, never silently emitted early. | aether-link/sigmoid `coupling.rs` contraction certificate (R-A5); Epsilon certify-or-refuse (R-E1) | Every halting owner halts on a threshold without a certificate | Fraction of tokens certified on M0. Kill if < 50%. |

**Plan steps updated (still ≤ 10).**
- **Step 6 (Zeno check) becomes D1 + D6.** Build CTC halting and its certificate, RED-first. Run the flat-null check on the TAU arm and the CTC arm side by side.
- **Step 4 (commutation control) gains D3.** The fixed-basis vs observer-chosen-basis arm runs at equal MACs.
- **Step 8 readouts gain D4 and D5:** λ₁ with flip rate, and the phase-vs-π histogram with the shuffled control.
- **New pre-result amendments, requested from Wilson:** D1–D6 as arms and readouts. The deciding QF-LM arm stays as filed (Amendment 6) until those amendments land.

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
8. Entropy readout (finding 9): plot H(|Qψ_t|²) against t on 512-token contexts, and fold it into item 7's origin readouts.
9. Write the T1 owner rows for MPS Born-machine LMs, MBQC feed-forward and HQMMs into OWNERS.md before any novelty claim.
10. Run the item-5 bake-off with arms {QF-LM, MPS-baseline, +W_v, softmax head, no-think, no-null, GPT, GRU} under the PREREG bars.

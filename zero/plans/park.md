# Park — decoding and talking, for a QF-LM whose next token is a measurement outcome

Source crawl: Wikipedia, fetched 2026-10-04 (WebFetch). Each finding = URL, fact, implication for QF-LM.
Code that this plan refers to: `qflm/generate.py`, `chat.py`, `tests/test_generate.py`, `tests/test_chat.py`.

## Findings

1. **Repeated measurement gives the same outcome.** https://en.wikipedia.org/wiki/Measurement_in_quantum_mechanics
   — "If a quantum measurement is repeated in quick succession, the same outcome will occur both times";
   general update via Kraus operators, E_i = A_i†A_i.
   *Implication:* if a learned K_v is close to projective, the Lüders write ψ ← K_vψ/‖K_vψ‖ makes
   p(v | just saw v) ≈ 1. Degenerate repetition is not a decoding accident in QF-LM, it is the physics.
   Only the think unitaries between outcomes move ψ off the last outcome. Readout to add: self-repeat rate
   P(x_t = x_{t-1}) vs the corpus rate, per arm.

2. **Zeno: escape probability is quadratic in the evolution time.** https://en.wikipedia.org/wiki/Quantum_Zeno_effect
   — transition probability after short time t ∝ t²; frequent measurement freezes evolution; slower
   measurement can speed decay (anti-Zeno).
   *Implication:* a think step with small butterfly angle θ leaves ψ with escape mass ~θ². A budget of s ≤ 4
   small-angle steps cannot break the loop in finding 1. Measure the angle norm per think step; if the
   loop survives, the null budget is mis-sized, not the sampler.

3. **Temperature is the Boltzmann/softmax inverse-temperature knob; T → 0 is argmax, T → ∞ uniform.**
   https://en.wikipedia.org/wiki/Softmax_function
   *Implication:* renormalised Born is softmax(2 log|α|), so p^{1/T} = softmax(2 log|α| / T): temperature on
   QF-LM is exactly the owned softmax temperature. No novelty claim for it. It biases which outcome is
   recorded; the Lüders write must still use the physical K_v. Any sample drawn at T ≠ 1 is no longer a
   Born record, so likelihood/holonomy readouts must use T = 1 records only.

4. **Nucleus sampling picks the smallest set whose cumulative mass exceeds p (Holtzman, Choi et al. 2019);
   it adapts to model confidence, top-k does not; sampling can still repeat.** https://en.wikipedia.org/wiki/Top-p_sampling
   *Implication:* the nucleus size per step is a free readout of instrument sharpness. The House bound
   (Σ_v max p ≤ N) caps how sharp the instrument can be, so the nucleus size distribution at N = 2048 vs
   N = 16 is a direct, cheap check of whether the bound bites in practice. The null outcome is included in
   the nucleus (joint over V+1 outcomes), so "think" competes with tokens on equal terms.

5. **Beam search is neither complete nor optimal; width 1 is hill-climbing (greedy).**
   https://en.wikipedia.org/wiki/Beam_search , https://en.wikipedia.org/wiki/Greedy_algorithm
   — greedy takes the locally optimal choice and may miss the global optimum.
   *Implication:* beam search also breaks QF-LM's one-state memory claim: a beam of width b holds b
   collapsed states and b different Lüders histories. Skip beam search; greedy is only a determinism test.

6. **The most probable sequence is generally not typical.** https://en.wikipedia.org/wiki/Typical_set
   — the all-ones string under p(1) = 0.9 is the single most likely string, but it is not typical.
   *Implication:* argmax/greedy QF-LM text will be atypical (degenerate) even if the model is perfect.
   The "talks" sheet must be sampled at T = 1 (with a pre-registered top-p), as the TinyStories rubric already
   does. Greedy transcripts are never evidence for or against talking.

7. **Perplexity = exp of average cross-entropy, units by log base; per-word normalisation; weak predictor of
   downstream (speech) quality.** https://en.wikipedia.org/wiki/Perplexity
   *Implication:* QF-LM perplexity must marginalise the null records exactly (Σ over s ≤ 4), on the same BPE-4096
   tokens as GPT. Otherwise the ppl ratio bar (≤ 1.05×) is not comparable. A ppl tie is still not "talks":
   the blind rubric stays a separate bar.

8. **BLEU is tokenizer-dependent ("scores ... incomparable"), built for multiple references and MT; ROUGE is
   recall-oriented n-gram/LCS overlap for summarisation.** https://en.wikipedia.org/wiki/BLEU ,
   https://en.wikipedia.org/wiki/ROUGE_(metric)
   *Implication:* neither metric measures open dialogue against one reference. Do not add them as talk bars;
   it would be a metric no one can defend. Use ppl + blind rubric + distinct-n / self-repeat (findings 1, 6).

9. **Chat formatting = role markers ("User:" / "Assistant:") plus a system prompt; instruction tuning is
   supervised fine-tuning to follow instructions (InstructGPT), RLHF on top.**
   https://en.wikipedia.org/wiki/Large_language_model
   *Implication:* the template tokens are measurement outcomes written into ψ like any other token. Two
   consequences. First, the post-system-prompt ψ is one N-dim complex vector, so the "prompt cache" is O(N),
   not a KV cache. Second, in SFT the prompt outcomes are forced, not sampled: their nll is masked from the
   loss, but their Lüders writes still happen. `chat.py` uses the plain `User:/Assistant:` template, which
   suits TinyStories-Instruct.

10. **Fine-tuning degrades robustness to distribution shift; fix = linearly interpolate fine-tuned and base
    weights.** https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)
    *Implication:* linear interpolation of Q (Stiefel isometry / butterfly) leaves the manifold, which breaks
    Σp + p_null = 1. Interpolate the butterfly angles and M's logits instead, or retract (QR/polar) after
    mixing. This needs a RED test before Instruct fine-tuning (PLAN item 6).

11. **ELIZA (1966) made people attribute understanding to clue-word pattern matching; modern chatbots
    hallucinate.** https://en.wikipedia.org/wiki/Chatbot
    *Implication:* a blind judge can be fooled by fluent shallow text. The 50-prompt sheet needs a calibration
    arm: a TinyStories 4-gram model. If the judge cannot separate the 4-gram model from GPT, the rubric
    tie bar means nothing.

12. **A dialogue system keeps a dialogue state in a dialogue manager.** https://en.wikipedia.org/wiki/Dialogue_system
    *Implication:* in QF-LM, ψ is the whole dialogue state. Eval readout: |⟨ψ_turn k | ψ_turn k+1⟩| across
    turns, which should be high within a topic and drop at a topic switch. This is cheap and fits item 7.

## Plan after this session (≤ 10 steps)

1. Gate `qflm/generate.py` + `chat.py` (built this session) on Kutner's real QF-LM: give the model
   `reset/dist/observe` (dist = (‖K_vψ‖², p_null)); `chat.load` already returns native steppers unwrapped.
2. RED test: on a trained QF-LM, the sampled record at T = 1, top_p = 1 reproduces the model nll (Born
   record = training distribution), within MC error. Binds finding 3.
3. Add readouts to the M0 eval: self-repeat rate, nucleus size, think steps per token, at T = 1 (findings 1, 2, 4).
4. If self-repeat > 2× corpus: log the think angle norms. Zeno diagnosis (finding 2) comes before any sampler change.
5. PREREG amendment (pre-result): the talks sheet is sampled at T = 1, top_p = 0.9, no repetition guard, 3 seeds.
   Greedy is banned as evidence (finding 6). The guard is reported as a separate arm.
6. Add the 4-gram calibration arm to the blind sheet (finding 11). Thirteen bundles it for Kaggle.
7. RED test for the Instruct fine-tune: Stiefel-safe interpolation keeps Σp + p_null = 1 ± 1e-6 (finding 10).
8. SFT loss masks prompt outcomes but keeps their Lüders writes. Test: the masked-prompt nll gradient is
   nonzero through ψ (finding 9).
9. Cache ψ after the system prompt in `chat.py`, assert memory is constant across 1k turns (criterion (c)).
10. Report the cross-turn fidelity curve on 20 transcripts (finding 12) with item 7. No bar.

### Status at the end of the session
- Done: the stepper protocol (`qflm/generate.py`), `chat.py`, `lm/ngram.py` (the 4-gram calibration arm, which plugs into `chat.reply`; `eval_talk.py` is untouched, and the calibration bar was proposed to Adams in `house-events.jsonl`), and `qflm/zeno_probe.py`. 26/26 tests pass across the five test files.
- Step 4 has been run on an **untrained** `qflm.model.QFLM` (N=256, V=64, seeds 0 and 1, 20 prompts) through a scalar adapter (observe(NULL) = null Lüders write, then think). p_repeat came out at 0.016–0.017 (= 1/V) for s = 0..4 at every think-angle scale (0, 0.016, 0.16, 1.56 rad). The initial instrument is nearly uniform, so there is no Zeno trap at init. On the sharp reference instrument, θ=0 holds p_repeat at 0.992. θ=0.05 escapes anyway (0.992 → 0.061 over 4 thinks), because rotations compound coherently. Finding 2's "θ² per step" overstates the trap, and the real test is the M0 checkpoint.
- Kutner's `QFLM` API is batched: `dist()` → ([B,V], [B]), `observe(tensor)`, `think()`. Wrapping it for `generate()` needs a 10-line adapter, currently in the scratchpad. It should move into `qflm/generate.py` as `QFLMStepper`.

## Wander (no agenda)

Path, about 20 min, 17 articles: Geometry → Whitehead's point-free geometry → Pointless topology → Stone duality, with a side hop from point-free geometry to Process and Reality → (jump) Quantum logic → Gleason's theorem → Linear logic (linked from quantum logic) → (jump) No-cloning theorem → Geometry of interaction (linked from linear logic) → Kochen–Specker theorem → Game semantics (linked from linear logic/GoI) → back to Geometry's links: Theorema Egregium → Erlangen program → Plimpton 322 → (jump) Einselection → Holevo's theorem.
Longest linked chain: Geometry → Whitehead → Pointless topology → Stone duality (3 hops). Quantum logic → Linear logic → Geometry of interaction → Game semantics adds 3 more.

### Five most surprising things
1. **A space with no points.** The locale of surjections ℕ → ℝ is a perfectly good topological object with no points at all. Tychonoff holds for locales without choice. https://en.wikipedia.org/wiki/Pointless_topology
2. **Born is forced, not chosen.** In dimension ≥ 3, any non-contextual probability assignment on projectors *is* the Born rule (Gleason 1957). Dimension 2 has counterexamples, but the POVM version (Busch) covers qubits too. https://en.wikipedia.org/wiki/Gleason%27s_theorem
3. **A continuum of states, but only log₂N readable bits.** n qubits give at most n bits of accessible classical information. The bound is χ = S(Σpᵢρᵢ) − Σpᵢ S(ρᵢ) (Holevo 1973). https://en.wikipedia.org/wiki/Holevo%27s_theorem
4. **The best universal cloner reaches fidelity 5/6** (Bužek–Hillery 1996). That is far above what "no cloning" suggests. https://en.wikipedia.org/wiki/No-cloning_theorem
5. **Quantum logic has no material conditional.** Distributivity fails for non-commuting tests, and the article says the logic "struggles with temporal reasoning". https://en.wikipedia.org/wiki/Quantum_logic
   (Runner-up: one entry on Plimpton 322, c. 1800 BC, is the *square* of its correct value.)

### Instinct connections to Asmodeous

- **H1 (hunch, strong; possibly the most important thing in this file).** QF-LM's context capacity is ≤ log₂N bits.
  - Why: in QF-LM the whole future record (text generated or scored after time t) is produced from ψ_t by a fixed instrument plus outcome-driven updates. So it is a single POVM on ψ_t.
  - By Holevo, under the model's own joint distribution, I(past; entire future) ≤ log₂N. That is **11 bits = 7.62 nats at N = 2048**, and 16 bits even at N = 65536.
  - This is the owned "excess entropy ≤ quantum statistical complexity ≤ log N" result from quantum computational mechanics (Gu et al. 2012, unverified here). It is the flip side of claim (c), "memory flat in context".
  - Real stories reuse names and objects across hundreds of tokens, and a GPT with a KV cache has no such cap.
  - Cheap decisive test: total nll saved on the second half of a held-out TinyStories doc by conditioning on its first half.
    - Under the model's own samples, QF-LM is bounded by 7.62 nats at N=2048. On data it is only approximately bounded, since that saving is measured against q(·|ψ0), not the mixture.
    - Measure the same saving for GPT. If GPT saves ≫ 8 nats per doc, the "talks at ppl ≤ 1.05× GPT" bar is already in trouble before training, and N must grow, or ψ must become a mixed/multi-register state.
- **H2 (hunch).** Linear logic and no-cloning give the same warning from the other side. The Lüders write consumes ψ, and there is no "!" (contraction) to re-use a token. Prediction: QF-LM fails copy/induction (… A B … A → B) far below GPT at equal ppl. Test: a synthetic induction task at N = 256 on CPU, 10 minutes.
- **H3 (hunch).** Einselection: K_v = diag(√M_v)·Q makes every write a partial dephasing in the Q basis, so the Q-basis states are the pointer states. If the think steps are weak, ψ decoheres to a pointer state and QF-LM is a classical HMM (owned). Readout: l1 coherence of ψ in the Q basis over 200 tokens, against the MPS/no-think arm.
- **H4 (hunch, weak).** Gleason/Kochen–Specker: at N ≥ 3 the Born head is the only non-contextual head over shared effects. The softmax-head ablation is then the *contextual* arm, and `t6_contextuality.py` could read that out. Not on the critical path.
- **H5 (hunch, weak).** Point-free reading: tokens are regions (effects E_v = K_v†K_v), not points (no embedding). The House bound says only rank-1 effects can be sharp, so "points" = tokens whose E_v reach rank 1. Readout: participation ratio of E_v against token frequency on M0.

### Plan update (replaces steps 9–10; steps 1–8 stand)
9. **H1 first, before the item-5 bake-off spends GPU hours.** Compute the Holevo cap log₂N next to GPT's measured first-half → second-half nll saving on 200 val docs. Then bound the QF-LM saving on M0 by sampling. Bar (PREREG Amendment 3, 343e1e6; S_d definition = Amendment 4): if the CI lower bound of GPT's mean saving > ln N nats (7.62 at N=2048), run the N-scaling or mixed-state variant first. ln N, not 2 ln N: the factor 2 is entanglement-assisted capacity, which a pure ψ without entanglement lacks (Wilson, test-bound).
10. H2 induction test plus the H3 coherence readout, both CPU, on M0 and the no-think arm. The cross-turn fidelity and ψ-cache work from the old steps 9–10 fold into item 7.

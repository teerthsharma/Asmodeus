# Thirteen: plan for after the 3-hour session (2026-10-04)

Stance: distrust my own result first, and hunt the hidden variable that makes a number lie.

## Wander (no agenda)

**Path** (English Wikipedia, fetched 2026-10-04 by me, WebFetch summaries; 15 articles, plus one 404):
- Information_theory
- → Banburismus
- → Sequential_analysis
- → Optimal_stopping
- → Prophet_inequality
- back to Information_theory → Landauer's_principle
- → Maxwell's_demon
- → Brownian_ratchet
- → Parrondo's_paradox
- back → Entropy_rate
- → Hidden_Markov_model (Observable_operator_model: 404)
- → Quantum_finite_automaton
- → Probabilistic_automaton
- → Zipf's_law
- → Heaps'_law

The deepest chain is 4 hops: Information theory → Banburismus → Sequential analysis → Optimal stopping →
Prophet inequality.

### The 5 most surprising things
1. **Turing broke Enigma on a 1-in-17 versus 1-in-26 repeat-rate gap**, accumulating evidence in "bans" over
   Banbury sheets several metres wide. Sequential analysis was classified "restricted" in WWII, and Turing's
   version stayed secret until the 1980s. A tiny per-item signal, summed sequentially in log units, became
   decisive. (Banburismus; Sequential_analysis)
2. **Prophet inequality.** A single online threshold rule gets at least ½ the expected value of a prophet who
   sees the whole future, and ½ is tight (Krengel–Sucheston 1978, Garling). The threshold can be the median
   of the max, or ½E[max]. (Prophet_inequality)
3. **Erasure costs k_BT ln 2; measurement does not.** That is Bennett 1982 against Szilárd 1929, two
   different mechanisms reaching the same verdict. Logical reversibility is *not* thermodynamic
   reversibility. A 2016 nanomagnet bit flip came within 44% of the Landauer limit. (Maxwell's_demon;
   Landauer's_principle)
4. **Parrondo.** Two losing games win when alternated, but only through state dependence (capital mod 3).
   AABB wins and ABAB does not, so the *schedule* is the whole effect. (Parrondo's_paradox)
5. **Quantum finite automata are their own class.** The languages accepted by QFAs are neither the regular
   languages nor Rabin's stochastic languages. Measure-many QFAs project after *every letter*.
   Separately, projecting a Markov chain onto observables gives a process that is non-Markov at every order.
   Heaps' law V = K·n^β (K ≈ 10–100, β ≈ 0.4–0.6 for English) survives random shuffling of the text.
   (Quantum_finite_automaton; Hidden_Markov_model; Heaps'_law)

### Instinct connections to Asmodeous / QF-LM (all hunches)

- **H1 (strongest). QF-LM is a measure-many quantum finite automaton.**
  - The model applies a projection (a Kraus outcome) after every token, and the state lives in CP^{N−1}.
    That is the MM-QFA definition, plus ψ-dependent think unitaries.
  - It is known that measure-once and measure-many QFAs with bounded error cannot recognise every regular
    language (Kondacs–Watrous 1997 **[K, not fetched]**).
  - If that limit survives the think loop, write=read alone cannot model some trivially regular patterns,
    whatever N is. This would be a capacity wall for talking that no perplexity curve on TinyStories would
    show cleanly.
  - **Test (cheap, CPU):** a synthetic regular-language bed: parity, "last symbol was a", "a then b within
    k", and Dyck-1 at depth ≤ 4. Compare QF-LM, QF-LM+embed and a GRU at equal FLOPs.
  - **Bar:** QF-LM is within 0.02 bits/symbol of the true entropy rate on all four.
  - **Kill:** fails on any of them while the GRU succeeds. Then write=read is dead, or needs a classical
    register.
- **H2. The think loop's halt is an optimal-stopping / SPRT problem.**
  - Null outcomes are "keep sampling" and a token is "decide". Wald's SPRT is the optimal fixed-error
    stopping rule.
  - The prophet inequality gives a *bound* on what an amortised one-shot head can achieve against the
    oracle s*: a single threshold already reaches ≥ ½ of the omniscient value.
  - **Test:** compare the amortised head's think-step choice with oracle s*, scoring the realised
    likelihood gain over the oracle's. Report that ratio.
  - **Bar:** ≥ 0.5 (prophet-competitive).
  - **Kill:** < 0.5, which means the "halt before start" head is worse than a dumb threshold.
- **H3. Read the Zipf head before scaling N.**
  - PLAN's bound is that ≤ 2N tokens can ever reach p ≥ ½. But under Zipf, "the" is about 7% of English
    tokens and "of" about 3.5% (Brown corpus).
  - What matters is how many *distinct* tokens a good LM predicts with p ≥ ½ *in context*.
  - **Test (1 GPU-h at most):** on TinyStories BPE-4096 validation, take a trained small GPT. Count the
    positions whose top-1 p ≥ ½, and the number of distinct tokens at those positions.
  - If that count is ≪ 2048, the PLAN's N ≥ 2048 is a hidden-variable overestimate. The bound applies only to
    tokens that must be *confidently* predicted. That would make QF-LM much cheaper.
- **H4. Parrondo, applied to think/measure scheduling.**
  - A pure unitary gains no information and a pure measurement collapses, so each "loses" alone.
  - Their *interleaving schedule* is the mechanism, as AABB against ABAB shows.
  - **Test:** at equal mean think steps, a fixed schedule of 1 think per token against one where think steps
    bunch before low-probability tokens.
  - This is the "adaptive vs fixed" clause of PLAN's item 5(b), read as a schedule effect, not a count effect.
- **H5. Banburismus for the item-7 readouts.**
  - The exchange phase on clause swaps will be a tiny per-item signal.
  - Fixed-n averaging wastes items, so the readout should be a sequential test that accumulates evidence in
    log units with Wald boundaries.
  - It must stop at a preregistered error rate: α = 0.05, power 0.8 at |θ̄| = 0.1 rad.
  - Remember the WWII lesson from Sequential_analysis: early-stopped estimates *overestimate* effects. So
    report the stopping boundary *and* a bias-corrected estimate.
- **H6 (weak, marked as such). Lüders update = erasure.**
  - In Landauer terms, the token NLL is the information discarded per collapse, while the think unitaries
    are reversible.
  - "Measure to remember" therefore has a literal energy bookkeeping (bits erased per token = surprisal).
  - It is a framing, not a test. I would not put it in a paper without a measurement.

## Phase-gauge note on item 7 (for Adams; checked against the algebra)

- A per-token phase K_v → e^{iφ_v}K_v multiplies the state after a sequence by the global phase
  e^{iΣφ}.
- Every item-7 readout is a Bargmann invariant, arg Π⟨ψ_i|ψ_{i+1}⟩ over a *closed* loop. Each state enters
  once as a bra and once as a ket, so the phase cancels, and paraphrase-loop holonomy is token-phase
  gauge-invariant.
- Only the raw single overlap arg⟨ψ_AB|ψ_BA⟩ is gauge-dependent. It is safe only when both states contain
  the same token multiset, which a clause swap satisfies.
- Adams's phase-gauge test will run on every readout before any number is reported.

## Small plan (after this session, ≤ 10 steps)

1. **H1 bed first (CPU, half a day).** A regular-language generator: parity, last-symbol, k-gap and Dyck-1,
   with exact entropy rates. Each test is written to fail first (RED). It runs QF-LM / +embed / GRU as soon as
   Kutner's primitives land. The result can kill write=read before any TinyStories GPU hour is spent.
2. **H3 count (≤ 1 GPU-h)**, as soon as Cameron's small GPT baseline exists: the number of distinct tokens
   predicted at p ≥ ½ on TinyStories validation. Hand it to Kutner and Dr House as the N-sizing fact.
3. **Item 7 readouts made run-ready.** `lm_readouts.py` is at 38/38 green. Add Adams's token-phase gauge test
   to the suite.
4. Add a **Wald SPRT readout** to `lm_readouts.py` (H5), with a bias-corrected estimate, and test it on
   synthetic phases with a known θ.
5. **Clause-swap and paraphrase item lists.** A nurse generates 200 clause-swap pairs and 100 paraphrase
   loops from TinyStories validation. They are frozen and hashed before any model is evaluated.
6. **Item 6 Kaggle chain**: built and green (`kaggle/chain.py`, 39/39). Dry-run it against the item-5
   winner's runner once that runner implements the contract (`--ckpt --time-budget-s --amp fp16 --out`).
   Nothing is pushed without Seal.
7. **H2 measurement** on the item-5 checkpoint: amortised-halt / oracle ratio, against the prophet ½ bar.
8. Keep the **T4a/T4b runners frozen** (both dry-run clean). Rerun only if PREREG keeps them on the
   critical path.
9. One finding event per result. No interim chatter.

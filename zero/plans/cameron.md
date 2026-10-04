# Cameron: findings and plan (QF-LM baselines lane), 2026-10-04

Stance: optimist. Build the strongest owned baselines. If QF-LM only beats weak ones, the result is worth nothing.

## Wander (no agenda)

I started at Biology and followed whatever pulled; 16 Wikipedia articles over 6 hops:
- **Hop 1:** Biology → Homeostasis → Allostasis.
- **Hop 2:** Allostasis → Hibernation, after the "repaying the debt" line.
- **Hop 3:** Hibernation → Place cell → Grid cell, via sleep debt to sleep to hippocampus.
- **Hop 4:** Grid cell → Slime mold → Quorum sensing → Hydra, via "cognition without neurons". The quorum-quenching example in Quorum sensing is a Hydra.
- **Hop 5:** Hydra → Predictive coding → Saccade → Chronostasis, from nerve net to how a brain predicts.
- **Hop 6:** Chronostasis → Octopus → Tardigrade → Sparse distributed memory, from distributed bodies to distributed memory.

**The 5 most surprising things**
1. **Chronostasis is backdating.** After a saccade, the brain fills the blind interval with what it saw *after* landing, and projects that percept backwards in time, up to about half a second. That is why a clock's second hand seems to freeze. Perception literally writes the past from the future.
2. **Allostasis: stability through change, by prediction.** The body *anticipates* demand and moves its set point; it does not wait for error. In exercise, muscles need about 18× more oxygen while the heart can rise only about 3.5×, so the brain borrows blood from gut and kidney and "repays the debt" later. Fever is a *raised* set point, not a malfunction.
3. **Preplay.** Hippocampal place-cell sequences fire for routes *before* the animal has taken them. Neighbouring place cells have unrelated place fields, and a colour change can globally remap all of them at once.
4. **Octopus.** Two-thirds of its neurons are in the arms, and a severed arm still responds. Its skin carries opsins, so it matches colour it cannot see with its eyes.
5. **Cognition with no neurons, and memory outside the body.** *Physarum* finds the shortest path through a maze and rebuilds the Tokyo rail network. It learns periodic bad conditions and stores memory as slime trails in the world. Bacteria act only above a density threshold (quorum: about 10¹¹ cells/ml before *Vibrio* glows), and some eavesdrop on the signal without paying for it.

Runners-up:
- Hibernators wake up periodically at large energy cost, perhaps to pay a sleep debt.
- Hydra does not appear to age at all.
- Tardigrades suspend metabolism for years in the tun state.
- Kanerva's memory activates only the hard locations within a Hamming radius of the address.

**Instinct connections to Asmodeous (all hunch, none tested)**
- *(hunch)* **Chronostasis is the time-traveller, observed in a living brain.** The CTC observer's "halt before start" answer is backdated: the percept after the saccade is written into the blind interval before it. That makes saccade + chronostasis a neural precedent for w* = D(P_h(z, w*)): commit to the state the future confirms, then present it as if it had always been there.
  - A testable form for QF-LM: allow the null "thinking" steps to *revise* the last emitted state's representation (not the token), i.e. backdate. Then measure whether surprisal after a think burst drops for the *preceding* token's context.
- *(hunch)* **Allostasis is the right word for the event clock.** A.E.T.H.E.R. fires on deviation, which is homeostasis. An allostatic version fires on *predicted* deviation: think before the surprise lands, using the amortised head. PLAN criterion (b) (think steps correlate with surprisal) should be scored on the *next* token's surprisal as well as the current one. A predictive clock would correlate with the future, a reactive one only with the present.
- *(hunch)* **Global remapping is the missing operation in the atlas story.** Place cells remap all at once on a context switch, with no smooth interpolation. The observer that picks charts is local and smooth. A "remap" event, where the whole chart set swaps when p_null spikes, could be what the null outcome is *for*, rather than "nothing to see".
- *(hunch)* **Octopus arms are the lazy charts, and they are more autonomous than ours.** Two-thirds of the compute sits in the periphery and works when cut off. Our charts compute only when called. The octopus suggests charts that *keep running cheaply on their own*, like tardigrade-level metabolism, and are *read* only when called.
  - That is a different laziness: lazy reading, not lazy computing. It may be the better trade on a GPU, where per-chart compute is cheap and memory traffic is not.
- *(hunch)* **Quorum sensing is the honest version of top-m.** Bacteria act only when enough of them agree (a density threshold), and cheaters eavesdrop. A router that commits only when k charts agree above a threshold, otherwise emitting null, could be the cheapest guard against single-chart overconfidence.
  - The cheater is a warning: a baseline that reads the routing signal without paying for charts is exactly the "propose-all" freeloader T4b already controls for.
- *(hunch)* **Physarum stores memory in the world, as trails.** QF-LM stores memory in ψ (O(N)). The slime mould says some memory can live in the *emitted record* itself. QF-LM's context = holonomy of the token path already half-says this. The claim to test is that nothing beyond ψ is needed, and the +embed ablation is that test.

Source tags:
- [V]: page fetched or search hit read this session.
- [K]: known, not fetched; fetch before citing.

## Findings

1. **An owner of QF-LM's core loop. Read it before item 5's bars are final.**
   - Source: Pestun, Terilla & Vlassopoulos, *Language as a Matrix Product State*, arXiv [1711.01416](https://arxiv.org/abs/1711.01416) (2017) [V, abstract only]. It models language "in complex matrices" with a translation-invariant probability measure "via the Born rule from a translation invariant matrix product state".
   - Related: Bradley, Stoudenmire & Terilla, *Modeling Sequences with Quantum States* (2019), and *Language Modeling with Reduced Densities* arXiv [2007.03834](https://arxiv.org/abs/2007.03834) [V, search hit].
   - Implication: a translation-invariant MPS sampled left to right has the same shape as QF-LM. Token-indexed operators act on one state vector, the emission probability is a squared norm, and the emitted operator conditions the state. That is p(v) = ‖K_v ψ‖² followed by the Lüders update.
     - So "tokens are measurement outcomes, write = read" is likely owned as a model class.
     - What remains new: the null outcome with input-free butterfly "thinking", and adaptive think steps on surprise.
     - Criteria (a) and (b) of PLAN item 5 test exactly those, so they stand. The paper's novelty sentence must not claim the Born/MPS LM itself.
     - Missing comparison: an MPS Born-machine LM with no null and no think. That is QF-LM with think=0 and null=0, which the no-think and no-null ablations already give. Label that row "MPS Born LM (Pestun et al. 2017)".

2. **Criterion (c), "memory flat in context length", is not a novelty test.**
   - Source: [Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) [V]: a hidden state updated each step is constant memory. [Mamba](https://arxiv.org/abs/2312.00752) [V] matches transformers on language with linear time and constant state.
   - Implication: the GRU baseline also passes (c). The paper can claim (c) only against GPT. A strong RNN-class owner beside the GRU (Mamba-style selective SSM, or at least an LSTM) is needed, or reviewers will say QF-LM was only compared against an old RNN.
   - Proposal: report (c) as "inherited from the recurrent class, not new", and add a Mamba-style arm in v2 if time allows.

3. **Perplexity is comparable only on identical tokenization.**
   - Source: [Perplexity](https://en.wikipedia.org/wiki/Perplexity) [V]: perplexity is b^{H(p̃,q)} per test event, so it is per *token* here.
   - Implication: QF-LM, GPT and GRU must share Foreman's BPE-4096 tokenizer (`beds/lang.py`); they do by construction.
   - Also report bits per byte (total NLL / UTF-8 bytes of the val text). This keeps the number comparable with TinyStories literature that uses other tokenizers, and makes the 1.05× bar robust to any tokenizer change.

4. **TinyStories says ~10M parameters is enough to talk, so the bar is fair.**
   - Source: Eldan & Li, arXiv [2305.07759](https://arxiv.org/abs/2305.07759) [V]: models "below 10 million total parameters", even "one transformer block", give "almost perfect grammar"; evaluation is GPT-4 grading of grammar, creativity and consistency.
   - Implication: a ~10M GPT on TinyStories is a strong, talking reference, and QF-LM tying it is a meaningful bar.
   - The paper's rubric (grammar / consistency / creativity) should copy their three axes, so results are comparable to an owned protocol rather than a home-made one.

5. **Weight tying belongs in the baselines.**
   - Source: Press & Wolf, arXiv [1608.05859](https://arxiv.org/abs/1608.05859) (EACL 2017) [V]: tying input and output embeddings gives "a significant reduction in perplexity" across neural LMs.
   - Implication:
     - QF-LM's write = read is a tied embedding in spirit: the same operator K_v reads and writes.
     - The GPT and GRU baselines must tie too (`arms/lm_baselines.py` does, tie=True), or QF-LM gets credit for a 2017 trick.
     - The "+embed" ablation (write ≠ read) is then QF-LM untied. Expect it to *lose*, as untied GPTs lose. Criterion (a) says +embed must not beat QF-LM by more than 2%, which is the tying result again. Item 5's write-up should cite this, so a pass on (a) is not oversold.

6. **Count embedding parameters, and say so.**
   - Source: [Neural scaling law](https://en.wikipedia.org/wiki/Neural_scaling_law) [V]: "Kaplan et al. did not count parameters in the token embedding layer", which biased small-scale coefficients; Chinchilla counts everything.
   - Implication: at |V| = 4096 and d = 256, the tied embedding is about 1M of a ~10M model, or 10%. "~10M params" must state whether embeddings count. Recommendation: count everything, Chinchilla-style. The matcher (`match_lm`) counts all parameters, with tied weights once.

7. **The token budget for a fair bake-off.**
   - Source: Chinchilla, via [Neural scaling law](https://en.wikipedia.org/wiki/Neural_scaling_law) [V]: about 20 tokens per parameter, C ≈ 6ND.
   - Implication: a 10M model wants about 200M tokens, roughly 6·10M·2e8 ≈ 1.2e16 FLOPs. On a 4060 at a realistic ~5 TFLOP/s effective, that is about 40 min per GPT run, so 3 seeds × 3 arms fit item 5's ~20 GPU-h only if QF-LM's per-token cost is near GPT's.
   - Pin the token budget per arm in PREREG: equal tokens AND equal FLOPs, both reported.

8. **Pre-LN is the safe owned recipe.**
   - Source: [Transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) [V]: pre-LN "stabilizes training without requiring learning rate warmup" (2020), whereas the original 2017 model was post-LN.
   - Implication: the GPT baseline is pre-LN (as specified), so it cannot lose to QF-LM through a warmup accident. The same lr schedule (cosine, short warmup) is used for every arm and tuned on the tune seed for each arm separately (L-BR).

9. **Data-limited small models have owned tricks.**
   - Source: BabyLM findings, arXiv [2412.05149](https://arxiv.org/abs/2412.05149) [V]: on 10M words, "a hybrid causal-masked language model" won, and distillation helped.
   - Implication: if the TinyStories download is not approved and the bake-off runs on a small corpus, BabyLM is the owned reference. QF-LM should then be compared only to identically trained baselines, never to BabyLM leaderboard numbers.

10. **History gives a positioning line.**
    - Source: [Language model](https://en.wikipedia.org/wiki/Language_model) [V]: Jelinek's IBM statistical LMs and perplexity (1970s); Bengio et al. 2003 neural LM; Mikolov et al. 2010 RNN LMs "substantially outperform" n-grams.
    - Implication: the story arc is n-gram (count), then neural (embed), then recurrent (carry a state), then attention (look back at everything), then QF-LM (measure a state).
    - It lines up with the Turing/observer intro and is accurate history. Each stage is one baseline row: a Kneser-Ney 5-gram costs nothing, so add it as the floor.

## Plan after the 3 hours (≤10 steps)

1. **Done:** `arms/lm_baselines.py` (pre-LN tied GPT, tied GRU) and `match_lm` are in, with 17/17 tests passing. The test failed first (import error), and a non-causal-attention mutant was caught.
   - Proposed full size at V=4096, T=256:
     - GPT d=352, L=6, H=4: 10.48M params, 10.91M MACs/token.
     - GRU d_emb=512, hidden=832, L=2: 10.04M params, 10.03M MACs/token.
   - The final sizes come from `match_lm` against QF-LM once Kutner's model exposes `active_flops` and `n_params`.
   - Item 7 below becomes: score PLAN criterion (b) on next-token surprisal too, to test the allostasis hunch.
2. Size GPT and GRU against Kutner's QF-LM at item-5 scale: `match_lm`, both-or-split, the same rule as `PREREG_DRAFT_arms.md` §3. Write the table into the prereg amendment.
3. Add two cheap owned rows:
   - a Kneser–Ney 5-gram floor on the same BPE tokens;
   - the "MPS Born LM" label on QF-LM no-think + no-null (finding 1).
4. Add a bits-per-byte metric next to per-token perplexity in Foreman's harness (finding 3).
5. Hand Wilson the owner hit (1711.01416, 2007.03834) for T1, fetching the full text first (L-OWNER). Rewrite the QF-LM novelty sentence to rest on null/think/adaptive (criteria a and b) only.
6. Smoke both baselines on the fixture corpus through the harness, then on TinyStories once Seal approves the download. This checks wiring only; no numbers are reported.
7. Tune lr and warmup per arm on the tune seed, with an equal trial budget per arm, then freeze.
8. Run the 3-seed GPT/GRU runs in the item-5 queue behind QF-LM, at an equal token budget (finding 7).
9. If time remains: add a Mamba-style selective-SSM arm as the strongest constant-memory owner (finding 2).
10. README and paper: rebuild the hero around the talking QF-LM once item 5 has a verdict. Until then, the LM section is a skeleton with no numbers.

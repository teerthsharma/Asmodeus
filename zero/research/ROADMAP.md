# Roadmap: how a new ML architecture is researched, and where this one stands

Updated 2026-10-13. Phase 0 is stages 0–4, phase 1 is stages 5–7, and phase 2 and later is stages 8–13.

**Where we are:** stages 0–4 are done, so phase 0 is closed. Stage 5 (Lean) is nearly done, and phase 1 is next.

A stage is **done** only when its exit gate closes, on fresh seeds or in Lean. A promising number is not enough.

```
PHASE 0  research          0 ✓  1 ✓  2 ✓  3 ✓  4 ✓
PHASE 1  build and prove   5 ◐  6 ○  7 ○                  <- next
PHASE 2+ compare and scale 8 ○  9 ○  10 ○  11 ○  12 ○  13 ○   (Seal, later)
✓ done   ◐ in progress   ○ not started
```

## Why the stages run in this order

Every architecture that lasted took the same path:

- **Transformer (2017):** won by training in parallel on GPUs, then by scaling predictably.
- **Mixture-of-experts (1991; sparse 2017; Switch 2021):** won once load balancing stopped router collapse.
- **S4 → Mamba (2021–2023):** had theory first. It competed only after a hardware-aware parallel scan made it fast.
- **What stalled:** Neural Turing Machines and capsule networks stalled for lack of hardware fit and scaling evidence.

The order runs from cheap and decisive to expensive. An idea dies early, when it is cheap to kill. It is never scaled before its mechanism is shown to work.

How GPT itself evolved (figures from memory, approximate):

| Year | Step | Bottleneck it removed |
|---|---|---|
| 1948–1995 | n-gram counts, smoothing | none; it is the baseline: counts cannot share strength across contexts |
| 2003 | neural LM with embeddings | similar words share statistics |
| 1997–2014 | LSTM, RNN LMs, seq2seq, attention | long context, but training is sequential |
| 2017 | Transformer | parallel training: fits GPUs |
| 2018 | GPT-1 (≈117M) | pretrain once, fine-tune per task |
| 2019 | GPT-2 (1.5B) | zero-shot tasks from the prompt |
| 2020 | scaling laws (Kaplan) | small runs predict big ones |
| 2020 | GPT-3 (175B) | in-context learning appears with scale |
| 2022 | Chinchilla | ≈20 tokens per parameter is compute-optimal |
| 2022 | InstructGPT → ChatGPT | RLHF makes it usable |
| 2023 | GPT-4 | loss predicted from runs ≈1000× smaller |

After 2017, GPT's gains came from scale, data and tuning, not from a new mechanism. A new mechanism must therefore win on a different axis, and then show it scales.

---

## Phase 0: research

### Stage 0. Frame the idea ✓
- **Purpose:** state what the mechanism is, why it should help, and how it could fail.
- **Activities:**
  - Define every term from its source.
  - Survey prior art.
  - Write the core hypothesis.
  - Write one kill rule per claim.
- **Deliverables:** a program document; definitions; claims with kill rules.
- **Exit gate:** each claim can be falsified by a named experiment.
- **Common failure:** vague terms that let any result count as success.
- **Ours:**
  - [`PROGRAM.md`](PROGRAM.md): hypothesis H-core, claims C1–C6, Addenda 1–3.
  - [`origin_notes_transcription.md`](origin_notes_transcription.md): Seal's handwritten definitions.
  - **TT** is the softmax decider that answers alone and goes back to an era when it needs that era's knowledge.
  - **Chronal fluid density** is w_t, its weight over eras: multiply by the likelihood, normalise, with momentum.

### Stage 1. Formal objects and theory sketch ✓
- **Purpose:** write the maths so each component has a rule and a lemma it rests on.
- **Activities:**
  - Define the state, update, decider, births and merges.
  - List the lemmas and the bounds they promise.
  - Note which lemma predicts each experiment.
- **Deliverables:** the lemma list and its formal statements.
- **Exit gate:** no component is left without a lemma.
- **Common failure:** theory written after the results, fitted to them.
- **Ours:** the FCPA lemma chain L1–L37 in `zero/lean/`, plus the era lemmas L38–L41.
  - L32 predicted that a contracting CTC loop cannot be a sharp decider.
  - The tournament confirmed it (CTC added nothing).

### Stage 2. Toy beds and owned baselines ✓
- **Purpose:** show the mechanism works where it should, before spending compute.
- **Activities:**
  - Build synthetic beds designed so the advantage can show.
  - Build an oracle as the regret reference.
  - Run baselines we own and tune them equally.
  - Freeze the bars by sha256 before any result exists.
  - Evaluate on fresh seeds only.
- **Deliverables:** beds, arms, frozen bars, results, a verdict.
- **Exit gate:** the preregistered bars pass on fresh seeds.
- **Common failure:** tuning on the evaluation seeds; baselines tuned less than the new method.
- **Ours:** E0, E0b and E0c ([`VERDICT_E0.md`](VERDICT_E0.md)).
  - Regret was 0.028 nats/token against 0.305 for the best owned learner, about 11× lower.
  - It showed no forgetting on re-entry and 0 false eras on stationary streams.
  - Exactly 4 eras formed for 4 laws on 8 of 10 seeds.

### Stage 3. Ablations and component tournament ✓
- **Purpose:** find which parts matter and pick the best version of each.
- **Activities:**
  - Implement every candidate variant.
  - Tune each on held-out seeds.
  - Evaluate on fresh seeds.
  - Compare them paired, seed by seed.
  - Keep the winner.
- **Deliverables:** an ablation table and one winner per component.
- **Exit gate:** the winner beats the incumbent on ≥ 8 of 10 paired seeds and passes every gate.
- **Common failure:** picking the winner on the same seeds used to tune it.
- **Ours:** E0d and E0e ([`VERDICT_TOURNAMENT.md`](VERDICT_TOURNAMENT.md)).
  - MOM (momentum along a learned era kernel) beat SOFT on 10 of 10 seeds.
  - CTC added nothing.
  - HARD answered with one era on 46% of steps at no loss.

### Stage 4. Strong-learner controls at equal size ✓
- **Purpose:** check that the gain is not something a standard learner gets for free.
- **Activities:**
  - Run GRUs trained online and meta-trained.
  - Run a meta-trained transformer that infers the regime from context.
  - Give the competitors the advantage of seeing the generating family.
- **Deliverables:** a head-to-head with frozen bars.
- **Exit gate:** E1 bars B1 (REC) and B2 (CYC-S). ERA's regret must be ≤ 0.5× the best neural arm's on ≥ 8 of 10 seeds.
- **Common failure:** under-trained competitors. This must be stated when it happens.
- **Ours:** E1 ([`VERDICT_E1.md`](VERDICT_E1.md)) passes both bars on 10 of 10 fresh seeds.
  - On REC, ERA's regret is 0.027 against 0.463 for the best competitor, a meta-trained 0.86M transformer. That is 15× lower.
  - The gain comes from memory beyond the context window.
  - Limit: the meta-trained GRU never learned in context.

## Phase 1: build and prove

### Stage 5. Decide every lemma in Lean ◐
- **Purpose:** turn claims into guarantees. This sets the design apart, because GPT has none.
- **Activities:**
  - Prove or refute each lemma.
  - Machine-check the proofs.
  - Record counterexamples.
  - Tie the proved constants back to the code.
- **Deliverables:** Lean files that compile with no `sorry`; a status table.
- **Exit gate:** every lemma is decided and the Lean build is clean.
- **Common failure:** proving a cleaner model than the code actually runs.
- **Ours:** L38–L41 are proved in Lean (PR #26). Two items are still open:
  - the learned-Q rate in L41;
  - a merge threshold with a finite-sample guarantee. Today's BF > 1 means δ = 1; the candidate is log BF > log(1/δ).

### Stage 6. Engineering build ○
- **Purpose:** make the design fast and exact enough to run on real data.
- **Activities:**
  - A batched torch module for the era bank and TT decider.
  - The forward filter as a parallel scan in log space.
  - Fused count updates and mixed precision.
  - Profiling, deterministic runs and checkpointing.
  - Monitoring era usage, so one era never takes all the weight.
- **Deliverables:** the engine, a parity test suite, a throughput report.
- **Exit gate:** exact parity with the numpy reference on E0 seeds, and a stated throughput target met.
- **Common failure:** a fast engine that silently differs from the research code.

### Stage 7. Real-data pilot, no GPT comparison ○
- **Purpose:** prove the engine runs on real text end to end.
- **Activities:**
  - A tokenizer.
  - A data pipeline pinned by hash.
  - Document-disjoint splits and contamination checks.
  - Era heads as closed-form ridge readouts on features, because count tables do not generalise to language.
- **Deliverables:** reproducible runs on TinyStories and a loss report against our own n-gram baseline.
- **Exit gate:** a reproducible run that beats the owned n-gram baseline.
- **Common failure:** data leakage across splits.

## Phase 2 and later (Seal runs these)

### Stage 8. Equal-compute comparison with small GPTs ○
- **What gets done:** nanoGPT-style baselines rebuilt from scratch, on the same data, tokenizer and FLOP budget, tuned equally.
- **Exit gate:** a win, a tie, or a clearly stated niche.

### Stage 9. Scaling study ○
- **What gets done:** 3–5 model sizes; fit loss against parameters, tokens and compute; find the compute-optimal frontier.
- **Exit gate:** a slope at least as good as the baseline's.

### Stage 10. Robustness and failure analysis ○
- **What gets done:** distribution shift, long context, out-of-distribution inputs, seed variance, and a reading of what each era holds.

### Stage 11. Composition and downstream use ○
- **What gets done:** era heads as a layer inside an existing model, the way MoE slots in; fine-tuning, instruction tuning and standard benchmarks.

### Stage 12. Release and replication ○
- **What gets done:** open code, a model card, a reproducibility package and a paper.
- **Exit gate:** someone outside reproduces the main result.

### Stage 13. Adoption ○
- **What gets done:** tools, integrations, and a demo people use.

## Risks carried forward

- **Synthetic evidence:** every win so far is on synthetic beds built to favour eras. Stage 7 starts to fix that.
- **Scale of E1 competitors:** the best context reader was a 0.86M transformer, and the meta-trained GRU never learned in context. A larger in-context control belongs to phase 2.
- **Sequential filter:** the filter runs token by token. The fix is the stage 6 parallel scan.
- **Proof and code drift:** Lean proves the maths, not the Python. Property tests must pin the code's constants to each lemma's hypotheses; for example, the NOREC gate is guaranteed for MOM when α ≤ 0.0078.

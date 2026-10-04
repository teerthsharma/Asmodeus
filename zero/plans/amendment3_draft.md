# Amendment 3 (pre-result), DRAFT by Thirteen, for Wilson to file

**Status.** Filed before any arm has run on these beds. It adds a deciding capacity test for write = read
(regular-language bed) and one measurement (H3).

## Source check (verified from fetched text, 2026-10-04)
- **arXiv quant-ph/0009004** (Ambainis, Kikusts, Valdats), §1, Theorem 1.1, citing Kondacs–Watrous 1997
  (FOCS'97, pp. 66–75):
  1. "All languages recognized by 1-way QFAs are regular."
  2. "There is a regular language that cannot be recognized by a 1-way QFA with probability ½ + ε for any
     ε > 0."
- **Theorem 1.2** (Brodsky–Pippenger 1999), same paper: if the minimal DFA has states q1 ≠ q2 and words x, y
  with q1 →x q2, q2 →x q2 and q2 →y q1, then no 1-way QFA (Kondacs–Watrous measure-many model) recognises L.
- **arXiv quant-ph/9802062** (Ambainis–Freivalds), Theorem 2 and Corollary 4:
  - with correctness > 7/9, 1-way QFAs are only as powerful as 1-way reversible automata;
  - a*b* contains a forbidden construction and is recognised only with probability about 0.68.
- **Not obtained:** the Kondacs–Watrous primary PDF (the TR1350 download returned a different paper). The
  result is cited at second hand from the two papers above.

**Why this applies to QF-LM, and why it might not.**
- QF-LM measures after every token, so it is measure-many.
- But its state is not a linear unitary QFA state, for three reasons:
  - the Lüders update renormalises a non-unitary Kraus image (K_v = diag(√M_v)Q is a contraction);
  - the post-state is a √weight mixture over null records;
  - the think unitaries' angles depend on ψ.
- The Kondacs–Watrous / Brodsky–Pippenger proofs use linearity and unitarity, so they need not transfer.
  **This test decides it empirically.**

## Bed (`beds/reglang.py`)
- Five stochastic sources over the alphabet {a, b}, framed as BOS … EOS.
- Every source has an exact per-position true log-probability, so the excess entropy is exact.

| language | generator (probabilities per state) | DFA states | QFA theory |
|---|---|---|---|
| **parity** (control) | a, b: p = 0.45 each. EOS: p = 0.1, allowed only when #a is even; otherwise the mass is renormalised over a, b. | 2 | group (reversible) language: QFA-friendly |
| **last-a** = (a+b)*a | a, b: 0.45 each. EOS: 0.1, allowed only if the last symbol was a. | 2 | Brodsky–Pippenger construction with x = a, y = b: no 1-way QFA |
| **a\*b\*** | State A: a 0.45, b 0.45, EOS 0.1. State B (after the first b): b 0.8, EOS 0.2, a 0. | 2 (+ dead) | Ambainis–Freivalds: probability ≤ about 0.68 only |
| **k-gap** (k = 3) | x_t = x_{t−3} with p = 0.9, else flipped. The first 3 symbols are uniform. EOS has hazard 1/64 per step. | 8 | definite language: needs forgetting |
| **Dyck-1, depth ≤ 4** | At depth d: "(" 0.4 if d < 4; ")" 0.4 if d > 0; EOS 0.2 if d = 0. Probabilities are renormalised over the allowed symbols. | 5 | bounded counter: non-reversible at the walls |

- Lengths are capped at 64 symbols (EOS forced at the cap, scored as part of the true model).
- Splits: train 20k, val 2k, test 2k sequences, from generator seeds 0 / 1 / 2. The data hash is pinned.

## Arms
All arms share the same tokens (a, b, EOS, plus BOS as the context start), equal budgets, 3 seeds, and tune
seed 3.

**Deciding arms**
- **QF-LM** (`qflm.model.QFLM`, arm `qflm`) at N ∈ {8, 16, 32}. The N sweep removes the "too small"
  excuse.
- **QF-LM + embed** (arm `embed`).
- **dephased-HMM**: Amendment 2's control. It joins when `qflm/controls.py` exists; until then this row is
  PENDING, not dropped.
- **GRU** (`arms/lm_baselines.GRULM`, hidden 16).

**Reported, not deciding**
- GPT (2 layers, d = 32, context 64).
- QF-LM arms `nothink` and `nonull`.

**Training.** Adam, 5,000 steps, batch 64. One learning rate is tuned per arm on seed 3 over
{1e-3, 3e-3, 1e-2}.

## Metric
- **Excess entropy** ε = (Σ model NLL − Σ true NLL) / (number of scored symbols) in bits per symbol, on the
  test split.
- Scored symbols are every position after BOS, including EOS. Each arm is scored on the same positions.
- QF-LM's forward also scores the BOS-predicted first position. That is a small bias against QF-LM, and it
  is recorded.

## Bars and kills
- **Pass on a language:** ε ≤ 0.02 bits/symbol on ≥ 2 of 3 seeds.
- **Bed validity:** the GRU must pass on a language, or that language is VOID for every arm.
- **QF-LM fails a language:** ε > 0.05 on ≥ 2 of 3 seeds at *every* N ∈ {8, 16, 32}, while the GRU passes.
- **Kill (write = read):** QF-LM fails any of last-a, a\*b\*, k-gap or Dyck-1. The verdict line then reads:
  "write = read is dead on regular languages, or needs a classical register".
  - If QF-LM + embed passes where QF-LM fails, the missing piece is the write path, not the instrument.
  - If dephased-HMM passes where QF-LM fails, coherence hurts.
- **Survive:** QF-LM passes all four non-reversible languages at some N ≤ 32. The verdict line then reads:
  "the Kondacs–Watrous / Brodsky–Pippenger QFA limit does not bind QF-LM; the nonlinearity (renormalised
  Kraus plus ψ-dependent think) escapes it".
- **Parity** is a control. A QF-LM failure on parity is an optimisation failure, reported as such and not
  counted as a capacity verdict.
- **Prior (Thirteen, a guess):**
  - 55% that QF-LM passes all;
  - 30% that it fails last-a or a\*b\* only;
  - 15% that it fails more.

## H3 (measurement, no bar; runs after TinyStories is available)
- On TinyStories BPE-4096 val, using the item-5 GPT baseline (seed 0), report:
  - the fraction of positions whose top-1 probability is ≥ ½;
  - the number of *distinct* tokens at those positions;
  - the same count at ≥ ¾.
- **Pre-stated reading.** PLAN's bound allows ≤ 2N tokens to ever reach p ≥ ½. If the distinct count is
  < 512, the N ≥ 2048 sizing argument is revisited, so a smaller N becomes admissible. No verdict changes
  from this measurement.

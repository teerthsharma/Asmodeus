# END GOAL: an Asmodeous model that talks, trained on one 8 GB GPU plus Kaggle

Draft by Taub, 2026-10-04. Readers: Foreman, who writes ACHIEVABLE.md, then Dr House.

Every source number below was fetched by sonnet nurses on 2026-10-04 and is cited with its URL.
- [unv] means it could not be opened.
- [est] means it is my estimate, to be replaced by a measurement at milestone M0.

## Recommendation

**G1, Asmodeous-LM.** A 15–30M-parameter language model:
- Its sequence core is built from Asmodeous parts:
  - a phase-state recurrence;
  - a POVM-routed lazy chart FFN;
  - event-driven token compute.
- Training order: pretrain on TinyStories, then fine-tune on TinyStories-Instruct and TinyDialogues.
- It is compared with a same-size transformer trained on the same tokens and the same budget.
- Pretraining runs on Kaggle. Every deciding number is computed on the 4060 from the checkpoint.

It is the fastest route to a model that talks and is still new, because TinyStories already shows that
models under 10M params write coherent English.

## 0. What "talks" means, measurably

These are the same five checks for every candidate. The model must pass 1–3 to be called "talking".

1. **Language modelling.** Held-out TinyStories validation loss (nats/token, fixed tokenizer).
   - Bar: within 5% of a same-size GPT-style transformer trained on the same tokens with the same
     optimiser budget (L-BR: equal params, and equal active FLOPs as a second baseline).
2. **Dialogue.** Held-out perplexity on TinyDialogues test turns (MIT licence).
   - Bar: within 10% of the transformer.
3. **Grammar.** BLiMP zero-shot (67 × 1000 minimal pairs, log-prob comparison, runs on 8 GB).
   - Reference: the OPT-125M baseline on BabyLM strict-small scores 0.63
     (https://arxiv.org/abs/2504.08165).
   - Bar: ≥ the same-size transformer.
4. **Judged samples.** The TinyStories rubric: grammar, consistency and creativity out of 10, 10
   completions per prompt at T = 1 (https://arxiv.org/abs/2305.07759).
   - Judge: an external LLM, blind to the arm.
   - Reported, no bar.
5. **Transcripts.** 20 fixed multi-turn dialogue prompts, fixed seeds, published verbatim for both
   arms. These are reported for inspection, no bar.

## 1. Candidates, ranked by impact × probability

| # | Goal | One result that shows it | Compute (4060 / Kaggle GPU-h) | Critical path | Earliest milestone | Impact | P | I×P |
|---|---|---|---|---|---|---|---|---|
| **G1** | **Asmodeous-LM**: the Asmodeous core replaces attention in a 15–30M LM, TinyStories → Instruct → TinyDialogues | Checks 1–3 met **and** ≥ 1.5× fewer active FLOPs/token than the transformer at equal validation loss (laziness pays) | ~20 local (M0, ablations, all evals, µs/token) + 25–40 Kaggle (2 arms × 2 seeds × ~1B tokens) [est] | T3 (lazy scaling, now K experts per token), T-EV (token skip), T7 (µs/token), TO2 (bounded observer loop) | **M0, 1 GPU-h:** 3–5M Asmodeous char/BPE LM on a TinyStories subset writes readable sentences; its loss gap to a same-size transformer is measured | 5 | 0.35 talks at parity (0.85 talks at all; 0.20 also wins the FLOP bar) | **1.75** |
| G2 | **BabyLM strict-small entry** using G1's core on a 10M-word budget | BLiMP ≥ 0.63 (OPT-125M baseline) at ≤ 30M params, against a public leaderboard | ~6 local [est] | G1's M0–M1 | G1 M1 checkpoint, retrained on 10M words | 4 | 0.30 | 1.20 |
| G3 | **Phase-state LM**: the complex unitary recurrence and POVM readout only (no charts, no events) | Checks 1–3 at parity | ~10 local + 15 Kaggle [est] | T7 | Same M0 with the chart FFN replaced by a dense MLP | 2 (close to LRU/SSM, owned) | 0.50 | 1.00 |
| G4 | **Asmodeous memory on a frozen tiny LM** (roneneldan/TinyStories-33M, MIT): the atlas acts as a lazy episodic memory/router adapter | Long-story consistency (judge score) up at equal params | ~8 local [est] | T3 | Day 1: the frozen model already talks | 1 (the talking is owned) | 0.60 | 0.60 |
| G5 | **Grounded talker**: G1 plus a caption channel on the T2 beds that narrates its own chart switches ("over the top now") from programmatic captions | Captions name the true region on held-out trajectories at ≥ 90%, and narrated chart changes align with seam crossings | G1 + ~10 local [est] | T2, G1 | After G1 M1 | 5 | 0.10 | 0.50 |

**Why G1 ranks first.**
- Talking is close to certain at this scale. TinyStories reports coherent output below 10M params,
  with all its models trainable on one V100 in ≤ 30 h (https://arxiv.org/abs/2305.07759).
- The uncertain part, parity and the FLOP win, is exactly what makes it new.
- G2 and G5 reuse its checkpoints, so G1 is on every path that proves something.

**Why the probabilities are not higher.**
- Non-attention cores reach parity only at ≥ 125M params and 15–300B tokens: Mamba is 5.1% below Llama
  in perplexity at ~165M, RWKV-4 4.8% above (https://arxiv.org/abs/2405.04517).
- No source compares such cores on TinyStories at ≤ 30M params.
- A recurrent core can lag at this size.

## 2. G1 design: the parts as a sequence core

Every part is learned (L-GIVEN). Each part keeps its PREREG ablation.

- **Tokenizer.** Byte-level BPE, 4–8k vocab, trained on TinyStories. TinyStories itself used the top 10K
  GPT-Neo tokens.
- **State.** ψ_t ∈ C^N, the "phase" state. It is updated by a gauge-invariant unitary Cayley step driven
  by the token. This is a linear recurrence, so training uses a parallel scan.
  - Owned neighbours: LRU, S4/Mamba [K]. The new part is the CP^{N−1} state with a fidelity loss.
- **Decider.** A Parseval-frame POVM over K chart experts per token: Σp = 1 exactly, plus an "observe
  nothing" effect. Top-m charts run (K = 64–1024, m = 2).
  - Owned: MoE, PEER. The exact lazy = dense weights property comes from ASMODEOUS.md.
- **Event-driven time.** A token's chart FFN recomputes only when ‖ψ_t − ψ_τ‖ ≥ ε, with ε learned.
  - This saves executed FLOPs.
  - It does not save wall time at B = 1 (see §4).
- **Observer loop.** A damped fixed point over the m read charts, with ≤ 3 iterations and an amortised
  one-shot head.
  - Owned: DEQ, PonderNet [K].
  - It is optional in v1, behind an ablation flag.
- **Readout.** Logits come from the glued g_t.
  - Topology readouts (nerve Betti over token contexts) are eval-only extras, not talking checks.

## 3. Where compute runs

| Side | Runs there | GPU-h | Precision | Limits that bind |
|---|---|---|---|---|
| **Local RTX 4060 Laptop (8 GB)** | M0; all ablations at 3–5M params and ~100M tokens; every deciding eval (checks 1–4) on downloaded checkpoints; T3, T-EV and T7 µs/token benches; tokenizer training | ~20 [est] | bf16 autocast (sm_89) | One GPU job at a time. RAM is tight: ≤ 2 GB and 4 threads per process, free RAM > 3 GB before launch. Laptop clocks are unlocked. |
| **Kaggle (T4×2 or P100, private kernels)** | Long pretraining (~1B tokens per arm): G1 and the transformer, 2 seeds each, plus the Instruct and TinyDialogues fine-tunes | 25–40 [est], about 1–1.5 weeks of the 30 GPU-h/week quota (https://www.kaggle.com/docs/efficient-gpu-usage) | **fp16 + GradScaler.** T4 and P100 have no bf16: T4 emulates it, and P100 cuBLAS bf16 fails. Both arms use the same precision on the same platform. | 12 h per session; /kaggle/working ≤ 20 GB persistent; 29 GB RAM, 4 CPUs (https://www.kaggle.com/docs/notebooks). Phone verification is required for GPU and internet. |

**Data upload.**
- TinyStories is 2.14M stories, CDLA-Sharing-1.0 (https://huggingface.co/datasets/roneneldan/TinyStories),
  about 474M GPT-2 tokens [unv].
- Pre-tokenised uint16 shards come to about 1 GB. They go into one private dataset, ≤ 50 top-level
  files, well under the 200 GB per-dataset limit (https://www.kaggle.com/docs/datasets).
- TinyDialogues is 316 MB, MIT (https://huggingface.co/datasets/styfeng/TinyDialogues).
- DailyDialog is CC BY-NC-SA. Use it only if the non-commercial licence is acceptable.

**Checkpoint and resume across 12 h sessions.**
- Wilson's `run.save_state` / `load_state` format is reused: atomic `os.replace`, params + EMA, AdamW
  state, step, torch/cuda/numpy RNG.
- It is extended with LR-scheduler state, the GradScaler state and the data cursor (shard, offset).
- Save every 30 min to /kaggle/working and keep the last 2. A 30M model plus Adam is about 0.4 GB, far
  under 20 GB.
- The next kernel attaches the previous one's output through `kernel_sources` and resumes from the
  latest checkpoint.
- Run `kaggle quota` (CLI ≥ 2.2.1) before every push.
- Thirteen owns `zero/kaggle/` bundling. Nothing is pushed until the plan says so.

**Rule conflict for Seal.**
- PROMPT §1 says "deciding numbers from the local RTX 4060, Kaggle reproduces", and
  `zero/kaggle/README.txt` says Kaggle "never decides anything".
- G1 needs Kaggle to *produce* the pretrained checkpoints.
- Proposed amendment: Kaggle may train checkpoints from a committed sha, and every score is computed
  on the 4060 from the downloaded checkpoint at a committed sha.
- Both arms train on the same platform with the same precision.
- **Pending Seal.**

## 4. What the measured numbers so far say about G1

- **T7.**
  - Generation at B = 1 sits on a ~60 µs per-step CUDA-graph floor on the 4060, so talking latency is not
    a problem: a 0.2–1 ms step is 1–5k tokens/s [est].
  - At equal active FLOPs, the fused atlas step ties dense at B = 1 and is 1.1–1.6× slower at B = 1024.
  - So training throughput, not inference, is where the core can lose. M0 must measure tokens/s for both
    arms.
- **T-EV.**
  - Real trigger rates on lift data are 0.25–0.85.
  - Executed-FLOP savings are real, but a data-dependent skip costs 0.2–2 ms in eager mode against
    16–27 µs graphed.
  - For the LM, the FLOP saving is reported as FLOPs. A wall-time claim needs a graph-capturable masked
    step.
- **T3 runner.** The soft all-K reference is feasible only for K ≤ 256 on 8 GB. The LM chart FFN therefore
  starts at K ≤ 256 locally, with K = 1024 only on Kaggle (16 GB).

## 5. Milestones (earliest first)

- **M0 (1 GPU-h, local).**
  - 3–5M-param G1 core vs a same-size transformer.
  - Data: char or 4k-BPE, TinyStories subset of ~50M tokens.
  - Measure both arms' tokens/s on the 4060, the validation-loss gap and 5 samples each.
  - **Kill switch for G1:** the gap is > 20% at M0 after lr tuning on a separate seed. In that case fall
    back to G3, the phase-state core alone.
- **M1 (local, ~10 GPU-h).** 15M models, 1 epoch of TinyStories, checks 1, 3 and 4. Ablations on the
  K, m, ε and loop flags.
- **M2 (Kaggle, ~30 GPU-h).** 30M models at ~1B tokens, 2 seeds, Instruct + TinyDialogues fine-tune. Then
  checks 1–5 on the 4060.
- **M3.** The G2 BabyLM strict-small rerun and the G5 grounded talker, both from M2 checkpoints.

## 6. Limits

- Every time and GPU-hour figure is [est] until M0 measures tokens/s on the 4060. The nearest sourced run
  is a 15M model on one L40S in ~50 min for 655M tokens
  (https://huggingface.co/brianbellx/ternary15M), a far faster card.
- The 4060 Laptop's dense fp16/bf16 tensor throughput was not found on a fetched page.
- TinyStories token counts are third-party [unv].
- Probabilities are author guesses, in the same spirit as PROMPT §9's priors.
- The LLM judge in check 4 is external and is never a bar.

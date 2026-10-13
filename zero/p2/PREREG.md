# Phase 2 comparison: the era LM against from-scratch small LMs (frozen before any result)

Frozen 2026-10-13, before any baseline or era-LM training. The sha256 of this file, `bench.py` and `frontier.py` are in `FREEZE.txt`. Changes after this commit are numbered pre-result amendments (filed before the result they touch exists) or post-hoc sections.

## Question

At equal training FLOPs and at equal parameters, is the era LM's next-token loss on TinyStories lower than that of the small models people build from scratch: a nanoGPT-style decoder and a GRU?

## Data and tokenizer

- Corpus: `zero/data/TinyStories-train.txt`, sha256 `c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f`.
- Split: `beds.lang.split_of`, salt `lang-v1`, 90/5/5 train/val/test, keyed by the sha256 of the normalised document.
- Tokenizer: byte-level BPE, 4096 tokens, trained on the train split only by `beds.lang.build_tokenizers`. The file is `p2/tok/tok_4096.json`, sha256 `80eafd2752301aac0d0398a4c8028d6c93231d6ae1eb6d543b1f220c3ff957a4`. Every arm uses this exact file.
- `p2/prep.py` writes each split as uint16 tokens, each document followed by `<|endoftext|>`. Measured: train 429,572,717 tokens (1,908,038 docs), val 23,824,441 (105,925), test 23,744,679 (105,526). 200M training tokens is under half an epoch.
- Eval sets: the first 4096 non-overlapping 257-token windows (1,052,672 tokens) of val (curves, LR tuning) and of test (the bars). Test is read once per eval-seed cell, after training.

## Arms

| Arm | Class | Shape | Params |
|---|---|---|---|
| gpt-s | `arms.lm_baselines:GPTLM` | d 128, 4 layers, 4 heads | 1.35M |
| gpt-m | GPTLM | d 192, 6 layers, 6 heads | 3.51M |
| gpt-l | GPTLM | d 320, 8 layers, 8 heads | 11.26M |
| gru-m | `arms.lm_baselines:GRULM` | emb 256, hidden 512, 2 layers | 3.94M |
| era | the era LM, per `zero/research/phase2/INTERFACE.md` | its main size | measured |

All arms tie the embedding and head. The GPT is the nanoGPT recipe: learned positions, pre-LN, GELU 4x MLP, init 0.02, residual projections scaled by 1/sqrt(2L).

## Recipe (identical for every arm)

- Context 256, batch 64 (16,384 tokens per step), 200M training tokens (12,207 steps).
- Batches are random 257-token windows of train.bin, drawn by a PCG64 generator seeded with the cell seed. The same seed gives every arm the same batches.
- AdamW, betas (0.9, 0.95), weight decay 0.1 on matrices and 0 on vectors, grad-norm clip 1.0.
- LR: linear warmup over 1% of steps, then cosine decay to 0.1x.
- bf16 autocast with TF32 off. An arm whose INTERFACE states it needs fp32 trains in fp32, and its throughput is reported as measured.
- 20 evaluations on the val set at evenly spaced steps.

## LR tuning (the same allowance for every arm)

- Tuning seed 0, 25M tokens, grid {5e-4, 1e-3, 2e-3, 4e-3, 8e-3}. The LR with the lowest final val loss is used for that arm.
- If the best LR is at a grid edge, the grid extends by one factor of 2 in that direction, and this repeats until the best is interior.
- The tuning seed is never an eval seed.

## Eval seeds

100, 101 and 102, for every arm. A cell is one (arm, seed) pair.

## Metrics

- **Primary:** test loss (mean next-token nll, nats per token) and its perplexity. An arm whose `loss` carries more than next-token nll must return `nll` [B, T], and evaluation uses that.
- **FLOPs:** forward FLOPs per token = max(FlopCounterMode count on a 2×257 batch, 2 × the arm's declared `active_flops`). The counter sees matmuls and attention but not cuDNN RNNs or elementwise work. That is why the larger count wins. Training FLOPs = 3 × forward FLOPs × tokens, plus any fitting cost outside gradient steps (closed-form heads, counts), which INTERFACE must declare.
- **Reported, not gated:** training tokens/s (wall-clock excludes evaluation), peak GPU memory (`max_memory_allocated`), loss against tokens, against training FLOPs and against wall-clock. All measured on one RTX 4060 Laptop GPU.

## Bars (the era LM against the GPT ladder)

The GPT frontier is the per-size mean test loss over seeds 100–102, interpolated linearly in log10(x) through gpt-s, gpt-m and gpt-l. x is training FLOPs for B1 and parameters for B2. Margin δ = max(0.02 nats, 2 × the largest per-size seed standard deviation in the ladder).

- **B1, equal FLOPs:** Δ = mean era test loss − frontier at the era's training FLOPs.
- **B2, equal params:** the same at the era's parameter count.
- For each bar, the result is **win** if Δ ≤ −δ and the era is below the frontier on at least 2 of 3 seeds. It is **lose** if Δ ≥ +δ. Otherwise it is a **tie**.
- **No extrapolation.** If the era's x is outside the ladder, a GPT of the same recipe matched to within 5% of that x is tuned and trained on seeds 100–102 before the era's test loss is read. That point joins the ladder.
- **GRU:** reported next to the era. If the era's parameter count is within ±25% of gru-m's, Δ against gru-m's mean is reported with the same δ rule. It does not gate.

`frontier.py` computes all of the above from `results/results.jsonl`, and `test_p2.py` pins its rules.

## Verdict rule

- B1 win → the era LM beats from-scratch transformers at equal compute on TinyStories.
- B1 lose → it does not, at this scale. That is reported as the result, and the era-LM design goes back to the era-LM thread.
- B2 is reported alongside. A B2 win with a B1 loss means parameter-efficient but compute-hungry, and is stated that way.

## Limits

One corpus (TinyStories), one context length (256), one token budget (200M), small models (1–11M), one GPU. LR is tuned at 25M tokens and used at 200M. That favours higher LRs equally for every arm. The ladder is trained at a fixed token budget, not compute-optimally, so the frontier is an equal-data frontier.

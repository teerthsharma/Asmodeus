# E1 preregistration: era bank vs context-reading models (pre-result)

Filed 2026-10-13. Claim **C5** (PROGRAM.md): at the eras' home turf, models that infer the regime from context do not match the era bank. If they do, eras are redundant with context.

## Arms

| Arm | What | Params | Training |
|---|---|---|---|
| **ERA** | architecture of record: E0c era bank + MOM decider (α = 0.003, E0e pick); CPU, numpy | ≈ 1–2k counts | none; online from the stream itself |
| GRU-ONLINE-S | 1-layer GRU, hidden 16, prequential: predict a 32-token chunk, then one Adam step on it, carried state | ≈ 1.9k | online on the stream; lr tuned |
| GRU-ONLINE-L | same, hidden 64 | ≈ 17k | online; lr tuned |
| GRU-META | 1-layer GRU, hidden 256, **meta-trained** on streams from the bed family (fresh laws per stream), frozen at test, state carried across the whole stream | ≈ 0.2M | 50M tokens |
| TF-META | causal transformer, 4 layers, d = 128, 4 heads, context 512, meta-trained likewise, frozen; eval with a sliding window (stride 256, every token sees ≥ 256 context) | ≈ 0.8M | 50M tokens |

The meta arms see the generator family (they learn the prior over laws); ERA does not. Both choices favour the competitor on purpose.

- **Meta-training data:** seeds 100000+ drawn equally from REC, CYC-S and NOREC (V = 16, T = 20,000 per stream). Validation streams use seeds 90000–90019. AdamW, lr 1e-3, cosine schedule, batch 64 × 512, best validation checkpoint.
- **Online lr grid:** {1e-3, 3e-3, 1e-2}, tuned on REC 100–104 + CYC-S 200–204 + NOREC 300–304 (mean loss).

## Beds and seeds

REC, CYC-S and NOREC, as in E0d/E0e. **Fresh evaluation seeds 50–59.** ORACLE is the regret reference.

## Bars

- **E1-B1 (REC):** ERA regret ≤ 0.5 × the best regret among the four neural arms on ≥ 8/10 seeds.
- **E1-B2 (CYC-S):** the same rule on CYC-S.
- **Kill:** if B1 or B2 fails, C5 is withdrawn for that bed. That bed's regime is recoverable from context, so eras are not needed there.
- **Reported, no bar:** NOREC regrets; parameters; training and evaluation wall time (4060 for the neural arms, CPU for ERA).

## Run

- **Code:** `zero/research/e1/` (torch ≥ 2.0).
- **Where:** the neural arms run on the RTX 4060 laptop; ERA and ORACLE run on CPU.
- **Order:** the bars are frozen in `e0/FREEZE.txt` before any E1 number exists.

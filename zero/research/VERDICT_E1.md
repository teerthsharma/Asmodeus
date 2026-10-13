# E1 verdict: era bank vs context-reading models (post-result)

Written 2026-10-13. Bars: `E1_PREREG.md` (sha256 d4a85646…) and `E1_AMEND1.md` (9d344487…, run location), both frozen in `e0/FREEZE.txt` before any full-scale neural number existed. Numbers: `e1/out/RESULTS_E1.md` and `e1/out/score.json`. Fresh seeds 50–59.

## Bottom line

- **C5 holds on both beds.** E1-B1 (REC) and E1-B2 (CYC-S) each pass on **10 of 10 seeds**.
- **REC:** ERA's regret averages **6%** of the best neural arm's, 15× lower (worst seed 7%).
- **CYC-S:** it averages **22%** (worst seed 40%, under the bar's 50%).
- The best context reader was the meta-trained transformer: 0.86M parameters, 50M tokens of the bed family. It ends at 0.46 nats/token regret on REC; ERA ends at 0.027.

| arm | params | REC | CYC-S | NOREC |
|---|---|---|---|---|
| **ERA** (E0c bank + MOM) | counts | **0.027** | **0.148** | **0.019** |
| TF-META | 0.86M | 0.463 | 0.705 | 0.182 |
| GRU-ONLINE-L | 17k | 0.688 | 0.660 | 0.750 |
| GRU-ONLINE-S | 1.9k | 0.740 | 0.717 | 0.776 |
| GRU-META | 0.21M | 1.023 | 1.048 | 0.714 |

Regret vs ORACLE, nats/token, mean over seeds.

## Why: memory beyond the context window

- Each regime is a fresh 16 × 16 transition table, so 256 probabilities must be learned again.
- TF-META sees 512 tokens. On CYC-S that is about 5 segments of about 100 tokens each, too few to estimate a table well. So its regret is worst on the fast-switching bed (0.71), not on REC (0.46).
- ERA keeps every regime's counts for the whole stream and returns to them on re-entry. This is the TT going back to an era for its knowledge.
- The two models differ in what they can remember, not in how well they were tuned.

## Limits (stated, not hidden)

- **GRU-META barely learned.** Its validation loss stayed at 2.69 nats against the oracle's 1.78. Under the frozen recipe (lr 1e-3, 1,526 steps) it never learned the tables in context, so it is a weak competitor.
- **The bars did not depend on it.** TF-META and GRU-ONLINE-L were the best arms on every seed.
- **Scale is untested.** GPT-3 showed in-context learning grows with size. A larger transformer, or one with a longer context, could close the gap. That comparison is phase 2 (`ROADMAP.md` stage 8 and stage 9).
- **Beds are still synthetic** order-1 Markov switches built to favour eras.
- **Run location:** everything ran on the container CPU, per the amendment. Wall times: ERA 7.8 s per stream; TF-META 52 min of meta-training.

## What changes

- Stage 4 of `ROADMAP.md` is done. Phase 0 is closed.
- Phase 1 (stages 5–7: Lean, engine, real-text pilot) is next.

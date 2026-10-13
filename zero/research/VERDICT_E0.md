# E0 / E0b / E0c verdict (post-result)

Written 2026-10-12 after both runs. Bars: `PROGRAM.md` §6 (E0) and `E0b_PREREG.md`, hashes in `e0/FREEZE.txt`. Numbers: `e0/RESULTS.md`, `e0/RESULTS_E0b.md`. Code: `e0/` (numpy only, 10/10 sanity tests).

## Bottom line

- **The era mechanism pays, by a lot, on a chronal fluid.** On recurring regimes the era bank sits 0.028 nats/token above an oracle that knows the true regime, against 0.305 for the best owned learner (BOCPD): **11× lower regret, 10/10 fresh seeds.**
- **It does not forget.** Re-entering a seen regime costs 0.084 nats over the oracle, against 0.70 for BOCPD and 0.86 for EMA.
- **It is nearly free when nothing recurs** once BOCPD is its novelty expert (1.17× BOCPD's regret, bar 1.25×).
- **It never invents eras on stationary data:** 0 spurious births over 20 null seeds. The Ville bound holds with online-learned eras.
- **It over-split laws in E0b** (4 to 8 eras for 4 laws). **E0c fixed it** on fresh seeds 20–29: a global pairwise Bayes-factor merge plus 64 transitions of evidence before deciding gives exactly 4 eras on 8/10 seeds and 5 on the other 2, with no loss cost (regret 0.0277).

## Bars

| Bar | E0 (seeds 0–9) | E0b, SOFT+BM (fresh seeds 10–19) |
|---|---|---|
| B1 regret ≤ 0.5 × best owned | PASS, 10/10 (ratio 0.11) | PASS, 10/10 (ratio 0.09) |
| B2 no cost without recurrence | FAIL (3.71× BOCPD) | PASS (1.17×) |
| B3 birth control | FAIL (7–11 eras; 0 null births) | FAIL (4–8 eras; 0 null births) |
| B4 no forgetting | PASS (0.086 vs 0.347 allowed) | PASS (0.084 vs 0.351 allowed) |

E0c, SOFT+BMX (fresh seeds 20–29, `E0c_PREREG.md`): **all bars pass.** B3″ 10/10 seeds ≤ 6 eras (8/10 exactly 4), 0 null births; B1′ 10/10 (0.0277 vs 0.3056); B2′ 1.18×; B4′ 0.084 vs 0.347 allowed. Numbers: `e0/RESULTS_E0c.md`.

Claims status after E0c: **C1, C2, C3 and C4 all stand** for the arm SOFT+BMX (fixed-share filter over the era bank + BOCPD novelty expert + global merge + minimum evidence). C3's "era count tracks the number of laws" was withdrawn after E0b and restored by E0c on fresh seeds.

## What it means (diagnosis)

- **The winning predictor is the forward filter, not the hard router.** SOFT (fixed share over eras) beats HARD (active era alone) 0.034 vs 0.125. The origin note says the architecture acts "after multiplication and normalisation": that is exactly the filter's Bayes step, posterior ∝ prior × likelihood, then normalise. The TT should answer with the filtered blend, and jump hard only for compute.
- **This agrees with L32.** A contracting CTC loop cannot route sharply (tilt cap e^{2q}). Sharp routing comes from the likelihood product over time, which the filter provides and the loop does not.
- **Over-splitting came from the address step** (short, mixed segments; the E0b merge only checked the active era). E0c's evidence wait costs nothing in loss because the filter, not the active era, makes the prediction. Era purity of the active era is 0.80–0.84 per step, the gap being the lag at switches.

## Honest limits

- The bed was built to favour eras: order-1 laws, sharp rows, 4 recurring laws. E0 is a **necessary** condition. It says nothing yet about language.
- The oracle and every count model share the generating prior a = 0.2.
- Tuned δ = 0.05, w = 32 and n_min = 64 all sit on the edge of their frozen grids; wider grids may do better.
- The era bank holds 4–12 count tables (1–3k counts). GLOBAL-o2 holds 4,352 counts and loses (regret 0.42), so parameter count does not explain the gain.
- No in-context learner was run. That is the real competitor (E1).

## Next (ranked)

1. **E1, the real test (GPU):** same bed, plus a GRU and a tiny transformer at equal parameters that can infer the regime from context. If they match the era bank, eras are redundant with context. Needs torch on the 4060.
2. **E2, language:** TinyStories / Instruct / TinyDialogues interleaved as eras; era heads with closed-form statistics on a frozen small LM.

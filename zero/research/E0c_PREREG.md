# E0c preregistration (post-E0b, pre-E0c-result)

Filed 2026-10-12. E0b's primary arm SOFT+BM passed B1′, B2′ and B4′ and failed B3′ (REC: 4–8 eras for K = 4 laws, 5/10 seeds within K + 2; NULL: 0 births). E0 and E0b bars and verdicts are not edited.

## Diagnosis (from E0b numbers)

- The merge compares only the active era with the others, so two stale duplicates never meet.
- A birth can be decided on a very short segment (median switch delay 13 tokens), where a fresh era's marginal likelihood is noisy and can win against the true era.

## Changes (both in arm SOFT+BMX)

1. **Global merge.** After every address decision, repeat: among all era pairs take the one with the largest Dirichlet–multinomial log Bayes factor log ML(C_i + C_j) − log ML(C_i) − log ML(C_j); merge it if that is > 0; stop when none is.
2. **Minimum evidence.** After an alarm, the address decision waits until the segment since the CUSUM change-time estimate holds ≥ n_min transitions. The detector keeps running meanwhile; prediction is unchanged.

δ, w, λ, hazard as in E0b. α and n_min ∈ {0, 32, 64} are tuned on seeds 100–104 (REC). **Tuning objective (fixed now):** among settings whose mean tuning loss is ≤ 1.02 × SOFT+BM's tuning loss, take the one with the smallest mean |eras − 4|; ties go to the lower loss.

## Seeds

Fresh evaluation seeds **20–29** for REC, NOREC and NULL. The owned arms are rerun there with their E0 picks.

## Bars (arm SOFT+BMX)

- B3″: NULL spurious births summed over 10 seeds ≤ 2; REC final era count ≤ 6 on ≥ 8/10 seeds.
- Non-regression: B1′ (REC regret ≤ 0.5 × best owned on ≥ 8/10), B2′ (NOREC ≤ 1.25 × BOCPD), B4′ (re-entry excess ≤ 0.5 × BOCPD's) all pass.

**Reported, no bar:** era count exactly 4 per seed; era purity (share of tokens whose era's majority regime is their true regime).

Verdict: all pass → SOFT+BMX is the E0 architecture, and C3 ("era count tracks the number of laws") is restored. Any failure is reported as is.

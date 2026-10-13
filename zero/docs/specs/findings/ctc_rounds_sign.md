# Finding: ctc rounds measure commitment stiffness, not uncertainty

Filed 2026-10-13. **Recorded as a finding by owner decision (Seal, option C).** This is a diagnostic: it is not under PREREG, and it changes no bar or kill. PREREG.md is unchanged.

**Closes** the open item at `zero/house-events.jsonl:549-551` (Kutner). The capped ctc arm gave Spearman(rounds, surprisal) = −0.485; the distance-travelled hunch was NOT SUPPORTED; the inverted sign was marked unexplained.

**Statement.** Under contraction halting (Amendment 8 D1, Amendment 10 cap), the number of think rounds tracks how stiffly the loop has committed to its answer, not how uncertain the token is. Tokens the loop settles on confidently take *more* rounds. The negative sign is a property of the halting rule, not a bug.

**Bearing on D1.** The D1 bar (Spearman ≥ 0.3, Amendment 10 item 3) stands as written and is expected to fail for this structural reason. No alternative halting rule is adopted here.

## Verdict
- **Reproduced:** the sign is negative on 5/5 seeds (−0.07 to −0.34, mean −0.22).
- **Explained:** rounds are set by two per-token quantities. Both are larger on confident tokens:
  1. **d0 = ‖F(p0) − p0‖₁**, how far the first think step moves the prediction.
  2. **λ, the local contraction rate at the fixed point** (how slowly the residual shrinks).
- Control for d0 and λ together and the sign goes away on 5/5 seeds (partial ρ −0.07 to +0.24).
- **This is why Kutner's test failed:** it controlled for d0 only. With d0 alone, seeds 1 and 2 keep partial ρ ≈ −0.25, the same pattern Kutner saw (−0.45). λ is the missing variable, and it is the stronger one: ρ(λ, rounds) = +0.62 to +0.77 on 4/5 seeds.
- **Not a pre-loop difficulty effect:** against surprisal *before* thinking (s0), ρ(rounds, s0) is −0.24 to +0.15 (mean −0.03). In plain terms, rounds do not track how hard a token is.

## Mechanism (mental model)
- At this scale the loop does most of the predicting: surprisal before the loop is 3.0–3.7 nats against a uniform 3.85; after the loop it is 1.8–2.0.
- So "easy" means "the loop reached a sharp, self-consistent answer", not "the token was easy to begin with".
- A sharp answer is a stiff attractor:
  - Low entropy at the fixed point goes with more rounds (ρ(rounds, H) is negative on 5/5 seeds, −0.24 to −0.43).
  - The residual near a committed answer decays slowly (ρ(λ, surprisal) is negative on 4/5 seeds, −0.26 to −0.41).
- The halting test, absolute ℓ₁ residual < 1e-4, therefore measures **how stiff the commitment is, not how uncertain the token is**.
- **Hunch, not tested:** the learned angle map W·p acts as positive feedback near commitment (critical slowing down). The next check is ‖J_F(p*)‖ (`point_jacobian_l1`) against surprisal.

## Ranked hypotheses
| # | Hypothesis | Status |
|---|---|---|
| 1 | λ channel: confident fixed points contract slower | **Supported** (5 seeds) |
| 2 | Travel channel (d0): the loop moves p further on tokens it ends up confident on | **Partly supported**; it alone does not explain the sign (matches Kutner) |
| 3 | Post-loop surprisal is endogenous: the loop makes tokens easy | **Supported** as the framing; vs s0 there is no consistent sign |
| 4 | Entropy-scaled or relative halting test fixes the sign | **Refuted**: a relative test (residual/d0) stays negative (−0.10 to −0.51) because λ remains |
| 5 | Artefact of the s_max = 4 cap / binary rounds | **Unlikely**: 0 tokens capped; uncensored rounds give the same ρ |

## Caveats
- This is a reconstruction, not Kutner's script (`rounds_capped.py` / `taskA.py` were never committed).
  - Setup: char-level V = 47, tiny_corpus, N = 64, r = 16, s_max = 4, Adam lr 3e-2, batch 8×32, 150 steps, 256 held-out tokens per seed.
  - Train/heldout split: 90/10.
- λ is estimated from the residual ratio over rounds 1→3, √(r₃/r₁).
- 256 tokens × 5 seeds, and rounds are effectively binary {3, 4}.
- The ctc_hinge arm ended uncertified (ρ_raw ≈ 1.01) on all 4 completed seeds, so every token ran s_max and its Spearman is undefined. Kutner's run landed at 0.992; the reason for the difference was not checked.

## Per-seed numbers (ctc arm)
| seed | ρ(R,surpr) | ρ(R,s0) | ρ(R,d0) | ρ(R,λ) | ρ(λ,surpr) | partial \| d0 | partial \| d0,λ |
|---|---|---|---|---|---|---|---|
| 0 | −0.107 | +0.136 | 0.367 | 0.246 | +0.096 | −0.015 | −0.036 |
| 1 | −0.344 | −0.046 | 0.278 | 0.765 | −0.407 | −0.256 | +0.038 |
| 2 | −0.252 | −0.244 | 0.078 | 0.622 | −0.259 | −0.248 | −0.072 |
| 3 | −0.066 | +0.153 | 0.610 | 0.745 | −0.283 | −0.044 | +0.237 |
| 4 | −0.316 | −0.170 | 0.527 | 0.643 | −0.302 | −0.071 | +0.135 |

Reproduce from `zero/docs/specs/findings/`: `python rounds_probe.py train ctc 0,1,2,3,4`, then `python rounds_probe.py analyse` (needs torch and scipy). CPU: about 4 min per seed at 1 thread.

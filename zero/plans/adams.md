# Adams — is the "talks" claim fair to the reader? (plan for after the 3-hour session)

Stance: PLAN.md says QF-LM "talks" iff val ppl ≤ 1.05× GPT **and** it "ties GPT on the blind 50-prompt rubric".
PREREG item 5 still has the rubric, rater and tie criterion PENDING. As written, the tie bar can be passed by a
model that is clearly worse, because a non-significant difference on 50 items is being read as a tie.

## Findings (Wikipedia crawl, 2026-10-04)

| # | URL | Fact | Implication for Asmodeous |
|---|---|---|---|
| 1 | https://en.wikipedia.org/wiki/Equivalence_test | A p-value above alpha does not show the absence of an effect; equivalence needs TOST against a pre-set margin (smallest effect size of interest). | "Tie" must be TOST on win-rate with a margin fixed now, not "sign test p ≥ 0.05". |
| 2 | https://en.wikipedia.org/wiki/Sign_test | Ties are dropped; exact binomial; efficiency 0.637 vs paired t, 0.67 vs Wilcoxon. | With 50 prompts the two-sided sign test only rejects at ≥ 33/50 wins; QF-LM can lose 31–19 and still "tie". Power is the problem, not the test. |
| 3 | https://en.wikipedia.org/wiki/Pseudoreplication | Treating correlated samples as independent gives p-values that are too small; found in ~48% of studies (Hurlbert). | Several samples per prompt, or 3 seeds × 50 prompts, are not 150 independent items. Resample by prompt (cluster bootstrap); the prompt is the unit. |
| 4 | https://en.wikipedia.org/wiki/Cohen%27s_kappa | κ = (p_o − p_e)/(1 − p_e); worked example p_o 0.7, p_e 0.5 gives κ 0.4; prevalence/bias paradox; weighted κ for ordered categories. | Use quadratic-weighted κ for Likert fields and report raw agreement plus marginals beside κ (paradox). Pinned in the tests: the 0.4 example. |
| 5 | https://en.wikipedia.org/wiki/Krippendorff%27s_alpha | α = 1 − D_o/D_e; handles missing ratings and nominal/ordinal/interval levels; α ≥ 0.800 reliable, 0.667–0.800 tentative only, < 0.667 discard. | Rubric scores count only if inter-rater α ≥ 0.667 per field, with α ≥ 0.8 required to claim "talks" outright. Below 0.667 the rubric does not decide anything. |
| 6 | https://en.wikipedia.org/wiki/Fleiss%27_kappa | Landis & Koch bands "may be more harmful than helpful"; κ is higher with fewer categories. | Do not quote "moderate/substantial" labels. Use the α thresholds from finding 5, fixed before any ratings exist. |
| 7 | https://en.wikipedia.org/wiki/Likert_scale | Single items are ordinal; central-tendency and acquiescence biases. | Use 1–5 per field with anchored descriptions; ordinal α / weighted κ and Wilcoxon, not means and t-tests. The pairwise preference is the primary end-point because it avoids the scale-use bias. |
| 8 | https://en.wikipedia.org/wiki/Blinded_experiment | Unblinding before the end reintroduces bias; CONSORT asks who was blinded, to what, and how well; success is rarely reported. | Raters see shuffled, side-randomised pairs with opaque ids (eval_talk.blind_pairs). Afterwards each rater guesses which side was QF-LM, and the guess accuracy is reported. Length is equalised by a shared max_new_tokens and stop rule. |
| 9 | https://en.wikipedia.org/wiki/ELIZA_effect | Brief exposure to a simple program made "normal people" think it understood them (Weizenbaum 1966). | Fluent TinyStories prose will be over-read. The rubric scores prompt-conditioned content (on-topic, names kept, no loops), never "seems to understand". Report the n-gram baseline in the same sheet as a floor. |
| 10 | https://en.wikipedia.org/wiki/Turing_test | The test rewards human-likeness and deception (Goostman: 33% of judges, persona with excuses); GPT-4.5 was judged human 73% of the time with a persona. | "Talks" must not mean "passes as human". It is relative to an identically trained GPT on fixed prompts. Claim nothing beyond that. |
| 11 | https://en.wikipedia.org/wiki/Leakage_(machine_learning) | Duplicate rows across splits and group leakage inflate results; leakage hit ≥ 294 papers in 17 fields. | The 50 prompts are written in TinyStories register, and some openings ("Once upon a time … Lily") are surely in the training text. Report the 13-gram overlap of each prompt with the train split. Contamination is symmetric across both arms, so it is fair to the comparison but not to an absolute "it talks" claim. |
| 12 | https://en.wikipedia.org/wiki/Preregistration_(science) | Prevents HARKing and p-hacking; about three-quarters of preregistered studies deviate at least once; registered reports publish more null results. | The sheet hash (ff9a955c…) and the tie rule go into PREREG as a pre-result amendment now. Any deviation gets logged as a dated amendment, never a silent edit. |
| 13 | https://en.wikipedia.org/wiki/Replication_crisis | Average power in psychology is 33–36% against the 80% norm; replication 36% of 97 studies (2015). | Three seeds and 50 prompts are underpowered. Compute the power of the tie test before rating, and publish it. |
| 14 | https://en.wikipedia.org/wiki/Grammaticality | Grammaticality is categorical while acceptability is gradient. Chomsky's "Colorless green ideas" is grammatical but meaningless. | "Grammatical" and "on-topic" stay as separate fields. The automatic grammar proxy is not a grammaticality judge. Wikipedia has no BLiMP page (404), so the BLiMP bar in PREREG rests on the paper, not on this crawl. |

## Arithmetic behind the tie rule (prompt = unit)
- Equivalence at margin ±0.10 on the non-tie win-rate needs about n ≈ (1.645 + 1.282)² · 0.25 / 0.10² ≈ 214 non-tie
  judgements for 80% power when the truth is 0.5. Fifty prompts with one sample each cannot reach this.
- So: keep the 50 fixed prompts and draw k = 5 samples per prompt per model (seeded), giving 250 pairs. Treat the prompt as the
  cluster: a cluster bootstrap over the 50 prompts gives the CI. If the 90% CI of the win-rate lies inside [0.40, 0.60],
  that is a tie (TOST at α = 0.05). If not, it is not a tie, whatever the sign-test p.

## Plan (≤ 10 steps)
1. Pre-result amendment to PREREG item 5. Contents: sheet sha256 ff9a955c3a864d7fa9b4daca3bbae308b524fddffe65c7182986d892de264c7e;
   the primary end-point is the pairwise preference; tie = TOST, 90% cluster-bootstrap CI of the QF-LM win-rate ⊂ [0.40, 0.60];
   k = 5 samples per prompt, temperature 0.8, top-p off, max_new_tokens 128, identical for both models.
2. Freeze the anchored rubric text: 1–5 for on_topic, grammatical and non_repeating, with one example per anchor, written before any sample exists.
3. Raters: two independent raters (two different LLM judges, plus Seal on a 50-pair subset if Seal chooses). Both get only the
   `blind_pairs` output. Gate: Krippendorff α ≥ 0.667 per field, otherwise the rubric is void and is reported as void.
4. Contamination report: 13-gram overlap of each prompt with the TinyStories train split, plus a check that no prompt is an exact
   substring of the val split used for ppl.
5. Floors and ceilings on the same sheet: a bigram/n-gram model (floor) and GPT vs GPT with a different seed (ceiling of "tie").
   The tie procedure must declare GPT-vs-GPT a tie. If it does not, the rule is broken, not the model.
6. Automatic proxy grader (eval_talk.proxy_scores) runs on every sample. Report it beside the human/LLM scores and never use it in place of them.
   Report its rank correlation with the rater scores.
7. Blinding check: after rating, each rater guesses QF-LM's side, and the accuracy is reported with a binomial CI.
8. Power statement before rating: given 250 pairs in 50 clusters, the simulated probability of declaring a tie when the truth is 0.5 and
   when it is 0.35. Published in the results whatever it says.
9. Report all three seeds; a "talks" claim needs a tie on every seed. Never pick the best seed.
10. Write-up language: "matches an identically trained GPT on 50 fixed TinyStories prompts", never "talks like a person".

## Status after Amendment 1 (adopted by the coordinator)
- `eval_talk.equivalence_tie` is in place: prompt-cluster bootstrap, TOST [0.40, 0.60], 90% CI. `tests/test_eval_talk.py` passes 17/17.
  An independent mutation (`tie = lo <= 0.5 <= hi`) turns 2 tests RED; after restoring, 17/17 again.
- Step 8 power (simulated, 200 runs, 50 prompts × 5 samples, n_boot = 500), where prompt_sd is the per-prompt heterogeneity of the true win-rate:

  | true win-rate | prompt_sd | P(declared tie) |
  |---|---|---|
  | 0.50 | 0 | 0.86 |
  | 0.50 | 0.15 | 0.69 |
  | 0.55 | 0.15 | 0.40 |
  | 0.35 | 0 | 0.00 |

  An equal model is declared tied 69–86% of the time; a clearly worse one is never declared tied. The GPT-vs-GPT seed control
  (step 5) measures the real prompt_sd. If its P(tie) < 0.8, raise k to 10 by amendment before any QF-LM rating.

## Wander (no agenda)
Path (19 articles, 7 hops deep): Philosophy of mind → Inverted spectrum → Opponent process → Impossible color;
Philosophy of mind → Mary's room → Epiphenomenalism → Bereitschaftspotential → (free will) Laplace's demon → Maxwell's demon →
Landauer's principle → Reversible computing → Holevo's theorem; Laplace (self-reference) → Halting problem → Busy beaver →
Gödel's incompleteness theorems → Orchestrated objective reduction; Gödel → Gödel metric → Closed timelike curve → Quantum Zeno effect.
All on en.wikipedia.org/wiki/<title>.

### 5 most surprising things
1. The inverted spectrum is *not* undetectable. Hardin: there are more distinguishable shades between red and blue than between green and yellow,
   and dark yellow turns brown while dark blue stays blue. So a red↔green swap would show up in behaviour.
   A symmetry is only invisible when the structure around it is symmetric too.
2. "Olo" (April 2025): a laser stimulating only M cones gave 5 subjects a blue-green that lies outside the normal gamut. Crane & Piantanida had already
   reported colours outside CIE 1931 in 1983.
3. Jackson abandoned his own Mary's-room argument (2023). The "Wow!" he used against physicalism also refutes the epiphenomenalism he drew from it.
4. Huxley compared consciousness to the bell of a clock: it rings but does not keep time. Libet found readiness activity about 0.35 s before
   reported intent, and kept "free won't", a last-moment veto.
5. BB(5) = 47,176,870 was proved in 2024 by an online amateur collective in Rocq. A 745-state machine halts iff ZFC is inconsistent.
   Turing never wrote the word "halt"; that is Davis, 1952.
   (Runners-up: Bennett's reversible TM simulates any TM; the erasure, not the measurement, pays k_B T ln 2, about 0.018 eV;
   Zeno freezing, first confirmed by Itano 1990, has an anti-Zeno twin.)

### Instinct connections to Asmodeous (all **hunch**, none tested)
- **hunch: the phase readouts may be pure gauge.** K_v → e^{iφ_v} K_v leaves every p(v) and the whole likelihood unchanged, so training
  cannot fix φ_v (inverted spectrum with no Hardin asymmetry). arg⟨ψ_AB|ψ_BA⟩ on a clause *swap* is safe, because both paths pick up the same
  phase factors. **Holonomy on paraphrase loops is not safe:** paraphrases use different tokens, so the readout shifts by an arbitrary Σφ.
  Only gauge-invariant quantities may be reported (Bargmann invariants arg⟨a|b⟩⟨b|c⟩⟨c|a⟩ over states from one path set, or loops
  that return through the same tokens). PLAN item 7 is at risk of reporting noise as "origin science".
- **hunch: think steps may be Huxley's bell.** Spearman(think steps, surprisal) ≥ 0.3 can hold with zero causal effect. Only the
  intervention counts: force s = 0 on high-surprisal tokens and NLL must rise, and adaptive must beat fixed at equal FLOPs.
  Correlation alone should never be credited to (b).
- **hunch: Zeno can kill thinking.** Repeated null outcomes are repeated measurements. If K_null is close to a projector, ψ freezes
  (Zeno) and the think loop does nothing. Cheap test: ‖ψ_s − ψ_0‖ against s along null records. If it is flat, the "still thinking" story is false.
- **hunch: criterion (c) is true by construction.** A fixed CP^{N−1} state means flat memory always holds. Landauer and Bennett say the cost
  sits in erasure: the Lüders update must forget. The honest measure is a recall-vs-distance curve, not bytes.
- **hunch: Holevo is a second capacity bound.** An N-dim state yields at most log₂N bits of accessible information about its preparation
  (N = 2048 gives 11 bits, while |V| = 4096 is 12 bits). It probably does not bind on TinyStories, whose per-token entropy is low, but it sits beside
  Wilson's mass bound and should be checked, not assumed.
- **hunch: halting rule from Deutsch's CTC.** The origin's CTC figure and "halt before start" suggest Deutsch's self-consistency.
  Halt the think loop at a fixed point, ‖U(ψ)ψ − ψ‖ < ε, as a parameter-free baseline for the learned s* head.
- **hunch: no system certifies itself** (Gödel II; Wolpert: no two inference devices fully predict each other). The team cannot be its own rater.
  The external rater plus the GPT-vs-GPT control is the "stronger system", which reinforces step 3/5.

### Plan changes (still ≤ 10 steps; replaces steps 6 and 7 of the plan above)
- Step 6 (was: proxy grader correlation) adds a gauge test: every reported phase readout must be unchanged under random
  K_v → e^{iφ_v}K_v. RED first; a readout that fails is dropped from item 7. The proxy-vs-rater correlation is folded into step 3.
- Step 7 (blinding check kept, merged into step 3) is replaced by two cheap probes on M0: the Zeno probe (‖ψ_s − ψ_0‖ vs s) and the
  s = 0 intervention for (b). Both run before any "new" claim is credited.

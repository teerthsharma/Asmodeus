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

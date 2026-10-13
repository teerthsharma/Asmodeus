# Research program: TT on a chronal fluid

Drafted 2026-10-12 (UTC). Sources: Asmodeus `zero/` (FCPA spec D1–D11, L1–L37, Amendments A–D, ERRATA 1–18, AUDIT_2, PILOT_DISCLOSURE, README plan) and the `math` fork (an external corpus of manuscripts, already cited by the spec as "corpus NNN"; it is a reference library, not Seal's own prior work).

## 0. Terms, as the sources define them

- **TT (time traveller).** A softmax predictor π(y | h) that streams forward and stores only the *results* of history (FCPA D7). Phase 1 TT is an LDA softmax (ERRATA 2); candidate H2 is an attention head whose KV multiset is its sufficient statistic (ERRATA 16). It answers ALONE, JUMPs to an era, or BLENDs.
- **Chronal fluid data density.** FCPA D1: the data is an ensemble of trajectories x^(u)_t ("multiverse"). ρ_t is its density at time t. Its governing law L_t (Markov kernel P, flow law θ) is piecewise constant in t. A maximal constant-law interval is an **era** (D5). The change of law ΔE between eras is also the era's **address** (L13–L14).
- **The one fixed point that connects every part.** Every FCPA mechanism is a fixed-point statement on ρ:
  - an era is a fixed point of the law estimator (its sufficient statistics stop moving);
  - a birth fires when no existing era is a fixed point for the incoming data (L23–L24*);
  - the CTC loop is the fixed point w* = softmax(βAw*/T) of the TT's own routing (L22H, L32).
  So the architecture is **"learn the fixed points of a moving density, and route between them."**

## 1. What the record already proves (no re-derivation needed)

| Fact | Source | Consequence for the program |
|---|---|---|
| Exact lazy weights: Born probabilities need no partition function | L1, README | Lazy top-m costs no approximation. Keep. |
| Non-destructive birth: old era probabilities unchanged when a new era is born | L23, D2′ | Zero forgetting **by construction**. This is the strongest structural claim. |
| Exact `partial_fit` = `fit` from additive statistics | L27′ | Online = batch. Keep. |
| Contracting CTC loop has tilt cap w*_max/w*_min ≤ e^{2q} < e² | L32(b) | **A contracting CTC loop cannot be a sharp decider.** It can only regularise a blend. Routing sharpness must come from elsewhere (forward filter, law fingerprint). |
| CTC is uninformative at E = 2 eras (w* = uniform) | L32(c), H32-4 | The loop matters only at E ≥ 3. |
| Zoom cannot discriminate eras | ERRATA 17 | Discrimination comes from the law fingerprint, not from geometry. |
| Linear era model cannot beat persistence at 50 steps | L28, L29 | The NS bed's bottleneck is the era *regressor* (needs the quadratic term), not the era *mechanism*. |
| Closed-loop E50 (lag-stack) 1.329 vs persistence 0.6585 | ERRATA 18(a), unbound | FCPA currently loses on its own primary bed. |

**Diagnosis.** The NS-2D primary claim tests the era mechanism only through a hard function-approximation problem (quadratic advection over 50 steps). If FCPA fails there, we will not know whether the *chronal-fluid idea* failed or the *regressor* failed. The program needs a test that isolates the mechanism.

## 2. Core hypothesis (H-core)

When the data law is piecewise stationary with a finite set of **recurring** laws (a chronal fluid), a learner that
1. keeps memory as eras of additive sufficient statistics,
2. switches eras with an anytime-valid change detector,
3. addresses eras by law fingerprint (likelihood of the recent window),
4. births non-destructively,

reaches lower cumulative loss than every owned single-model or changepoint learner, forgets nothing on re-entry, and grows its era count only with the number of true laws.

**Owned neighbours (vocabulary, not novelty):** BOCPD (Adams–MacKay 2007), fixed share / mixing past posteriors (Herbster–Warmuth 1998; Bousquet–Warmuth 2002), sticky HDP-HMM, e-detectors (Shin–Ramdas–Rinaldo). The new combination is: exact non-destructive birth + Ville-bounded birth + exact partial_fit + lazy exact routing, as one architecture that a TT sits on.

**Real competitor for language:** in-context inference. A long-context model can identify the regime from its context window without any era machinery. Any language claim must beat that, at equal parameters.

## 3. Formal objects

- Stream x_1..x_T over a vocabulary of size V. Regime r_t ∈ {1..K}. Law P_r ∈ Δ_V^V (order-1 kernel in E0).
- Era k: count table C_k ∈ N^{V×V}. Predictive p_k(x | x′) = (C_k[x′,x] + a)/(Σ_y C_k[x′,y] + aV).
- Detector on the active era A: e_t = F_t(x_t)/A_t(x_t), with F_t = λA_t + (1−λ)W_t and W_t the window predictive. E[e_t | past] = 1 under "x_t ~ A_t", so the Shin–Ramdas–Rinaldo recursion S_t = (S_{t−1} + 1/(t(t+1)))·e_t gives M_t = S_t + 1/(t+1), a test martingale; Ville bounds a false alarm by δ.
- Address on alarm: compare the window's log-likelihood under each bank era with a fresh era's prequential (marginal) likelihood. Best bank era wins → JUMP; fresh wins → BIRTH.
- Forgetting: re-entry excess = mean log-loss over the first 100 tokens after re-entering a seen regime, minus the oracle's on the same tokens.

## 4. Falsifiable claims and kill rules

| ID | Claim | Kill rule |
|---|---|---|
| C1 | Era factoring pays: regret vs oracle far below every owned baseline on the recurring bed | Fails B1 → the mechanism does not pay even on its most favourable bed. Stop building on it. |
| C2 | The gain comes from recurrence, not from changepoint detection | Gain over BOCPD does not shrink on the no-recurrence bed → the claim of "recurrence memory" is not what is measured. |
| C3 | Birth is controlled | Spurious births on stationary streams above the bar → the Ville guarantee does not survive estimated laws (as on NS, E1). |
| C4 | No forgetting | Re-entry excess not below BOCPD's → L23 does not translate into behaviour. |
| C5 | Beats in-context inference at equal params (E1, needs GPU) | A same-size GRU/transformer matches it → eras are redundant with context. |
| C6 | Transfers to language (E2) | Era heads on a frozen TT do not cut domain-shift loss on a TinyStories-family stream. |

## 5. Ranked experiment plan (impact × probability, cheapest decisive first)

| # | Experiment | Decides | Compute | Impact | P(pass) | I×P |
|---|---|---|---|---|---|---|
| **E0** | Switching Markov sources (V = 16, K = 4, recurring) vs Global-o1/o2, EMA, BOCPD, oracle | C1–C4 | CPU, minutes | 4 | 0.7 | **2.8** |
| E1 | Same bed + in-context controls (GRU, tiny transformer, equal params) | C5 | 4060, < 1 GPU-h | 5 | 0.4 | 2.0 |
| E2 | Language: TinyStories / Instruct / TinyDialogues interleaved as eras; era heads (closed-form ridge or n-gram) on a frozen small LM | C6 | 4060, ~2 GPU-h | 5 | 0.35 | 1.75 |
| E3 | NS-2D primary claim with the V_Q quadratic era ridge | Amendment 11 P1–P4 | owned by "Unblock FCPA fluid prediction" | 4 | 0.3 | 1.2 |
| E4 | CTC as a regulariser, not a decider: E ≥ 3 blends with q = 0.8 vs forward-filter weights | L32 consequence | CPU | 2 | 0.5 | 1.0 |

E0 is a **necessary-condition test**. The bed is chosen to favour eras. Passing it proves little alone; failing it kills the mechanism cheaply.

## 6. E0 bars (pre-result)

Frozen before any E0 number exists. The sha256 of this file up to the end of this section is recorded in `e0/FREEZE.txt`.

**Beds.** Generator `e0/bed.py`. V = 16, order-1 kernels with rows ~ Dirichlet(0.2·1). Segment length = 50 + Geometric(mean 350). T = 20,000.
- REC (recurring): K = 4 laws; each new segment draws a law ≠ the current one.
- NOREC: every segment draws a fresh law.
- NULL: one law, no switches.

**Seeds.** Tuning: 100–104 (REC). Evaluation: 0–9 for each bed. No eval-seed number is looked at before the bars below are frozen.

**Arms.** Shared Dirichlet prior a = 0.2 for every count model.
- ORACLE: one count table per true regime (lower bound).
- GLOBAL-o1: one count table.
- GLOBAL-o2: order-2 table backed off to order-1 (hierarchical Dirichlet, strength β tuned).
- EMA: decayed counts, decay λ tuned.
- BOCPD: run-length posterior, Dirichlet-categorical per run, hazard tuned, run length capped at 1,500.
- FCPA-HARD: era bank + e-detector (δ, window w tuned; λ = 0.5) + address/birth; predicts with the active era (TT ALONE).
- FCPA-SOFT: same bank and births; predicts with a fixed-share forward filter over bank eras plus the window model (L14′ HMM filter; switch rate α tuned).
- Tuning: each arm's grid is scored by mean log-loss on seeds 100–104 only; the best setting is then fixed for all eval seeds and beds.

**Bars.** "Owned" = {GLOBAL-o1, GLOBAL-o2, EMA, BOCPD}. Regret = arm mean log-loss − ORACLE mean log-loss (nats/token).
- **B1 (C1).** On REC, the better FCPA arm's regret ≤ 0.5 × the best owned regret, on ≥ 8 of 10 eval seeds.
- **B2 (C2).** Ratio BOCPD regret / FCPA regret (better arm) is larger on REC than on NOREC, in the mean over seeds; and on NOREC, FCPA regret ≤ 1.25 × BOCPD regret (it must not pay much for its memory when there is nothing to remember).
- **B3 (C3).** NULL: total births beyond the first era, summed over 10 seeds, ≤ 2. REC: final era count ≤ K + 2 = 6 on ≥ 8 of 10 seeds.
- **B4 (C4).** REC: FCPA re-entry excess ≤ 0.5 × BOCPD re-entry excess, in the mean over seeds.

**Reported, no bar.** Parameters held, wall time, per-switch detection delay, NOREC era count.

**Verdict rule.** All four bars pass → H-core survives E0; proceed to E1. B1 fails → kill the mechanism at its most favourable bed and report it. Any other single failure → the named claim is withdrawn, the rest stands.

---

## Addendum 1 (post-freeze, does not change §6)

- **Origin notes reconciled** ("That if we combine", page 3, read at 400 dpi): "our softmax is a TT (Time T[raveller]) … is smart alone, but when he needs knowledge of a particular era he goes back in time … contains only results of history … the era contains knowledge … like a JEPA"; "fluid chrononal predictive architecture (after multiplication and normalisation)"; "JEPAs are eras inside … time traveller will have all knowledge but going to an era expands"; "LEMMA and LEAN → BUILD → PROVEN". This matches FCPA D7 (TT = softmax) over the atlas reading (TT = CTC loop); the CTC loop is the TT's *return trip*, not the TT.
- **"Chronal fluid data density" is this program's formalisation** of FCPA D1 plus the note's "liquid of JEPA": the stream's law density ρ_t, piecewise stationary, with eras as its fixed points. The exact phrase appears nowhere in Seal's files (prior-work catalog).
- E0/E0b verdict: `VERDICT_E0.md`.

## Addendum 2 (post-freeze): grounded in Seal's handwritten notes

Source: `/mnt/project-files/prior-work/handwritten-notes-transcription.md` (both notes, full prose). §6 bars and the E0–E0c verdicts are unchanged.

**Definitions, replaced by Seal's own**
- **TT** (unchanged): "our softmax is a TT … smart alone … when he needs knowledge of a particular era he goes back in time … contains only results of history."
- **The fluid is the liquid of eras, not the data.** "We will provide the time travel a dimensional phase; its momentum will be like in a dimension of liquid … path of softmax … so we will get liquid of JEPA." My Addendum 1 definition (the data's law density over time) is **withdrawn**.
- **Chronal fluid data density := w_t**, the TT's softmax weight over the eras at time t: where the TT stands in the liquid. Its path through time is the "path of softmax", and it should carry **momentum**. Births add new eras to the liquid; merges fuse two of them.
- **"After multiplication and normalisation"** is the update w_{t+1} ∝ w_t ⊙ p(x_t | era), then normalise. The E0 filter already computes exactly this w_t.

**Where the plan changes**
1. **E0's winning arm is the user's density, not a side result.** The fixed-share filter weight is w_t. Its share rate α is a uniform "viscosity" with no momentum.
2. **New experiment E0d (CPU, next after E1):** momentum. A learned era-to-era transition kernel (FCPA D3's Markov law over observer outcomes) moves w_t along learned paths, against uniform fixed share. It runs on a bed whose regimes follow structured cycles. Prediction: it wins there and ties on E0's uniform-switch bed.
3. **"Proven" now means lemma plus Lean**, per the note's research → LEMMA and LEAN → BUILD → PROVEN. Until then the E0 results are *measured*, not *proven*. Lemmas to write:
   - L38: regret of the filter over a bank that grows by births;
   - L39: consistency of the Bayes-factor merge;
   - L40: no forgetting, as a corollary of L23 for count eras.
4. **"Butterfly effect bonded via topology" becomes a measured bound.** One token can move log w_t by at most log(max p / min p) plus the share term. Report it per step. The topology readout is the era transition graph, the nerve of w_t's path.
5. **Eras may be any predictor except JEPA or self-attention** ("JEPA as a phase … the decider will be a softmax"). For E2 (language), eras stay closed-form statistical heads, and the TT stays the softmax decider.

**Ranking now:** E1 (context control, GPU) → E0d (momentum, CPU) → L38–L40 (Lean) → E2 (language).

## Addendum 3 (post-freeze): tournament outcome

- E0d/E0e decider tournament (`VERDICT_TOURNAMENT.md`): **MOM wins**, 10/10 fresh seeds vs SOFT, all gates. Architecture of record = E0c era bank + MOM decider.
- Next, ranked: E1 (in-context controls on the 4060) → E2 (language eras) → lemmas L38–L41 in Lean (L41: MOM regret as an HMM filter with learned transitions). Next fresh seeds start at 50.

# CHASE: corpus dependence memo (P1-P6), feeds Amendment E

## Verdict

The math corpus does not close or upgrade any of P1-P6. It contains 0 of 372 families on e-values, anytime-valid inference, conformal prediction, Hoeffding/Azuma/Bernstein for dependent data, mixing coefficients, changepoint detection, or model collapse. The L35 hole is repaired anyway, by a lemma that needs no corpus result and no mixing assumption (E1 below, MC-checked). Every non-corpus citation below is labelled CITED-UNVERIFIED unless I proved it here.

Provenance of "0 of 372" (three independent nurse sweeps, all haiku, raw files in `scratchpad/chase_corpus/`):
1. `families.txt` (the 372 family lines of `CONTENTS.md`): clusters P1..P6 return 9/7/5/11/2/3 keyword hits (`hits_P1_P3.txt`, `hits_P2_P5.txt`, `hits_P4_P6.txt`); every P1/P2/P3/P5/P6 hit is a false positive by title ("conformal" = complex analysis or QFT, "mixing" = ergodic theory or shuffles, "Ville" inside "Liouville", "e-value" inside "squarefree-values").
2. Full text of `CONTENTS.md` (abstracts included), 49 patterns: 42 patterns have zero hits (test martingale, anytime-valid, always-valid, confidence sequence, conformal prediction, split conformal, exchangeab, Hoeffding, Azuma, McDiarmid, beta/alpha/phi-mixing, Doeblin, changepoint, CUSUM, Lorden, hidden Markov, HMM, model collapse, self-consuming, MCMC, minimax, hypothesis test, ...). The only real hits: mixing time in 131, 227, 238; spectral gap in 227 (plus Hamiltonian gaps in 265, 268, 269) (`hits_fulltext.txt`).
3. 722 preprint directory names, `README.md`, `overview.tex`: no sequential-statistics, conformal or concentration title (nurse list of 126 substring-matches, all geometry, algebra, QFT, complexity).
The family list was also read by me end to end (372 titles): the corpus is a collection of resolved conjectures in pure mathematics. Corpus papers are 2026 preprints, unrefereed.

Two findings that change the repair strategy (both mine, both checked):
- K1. The bed has no sigma-field mixing within an era. `fcpa/beds/ns2d.py` lines 169-258: `simulate` is a deterministic RK4 pseudo-spectral integration; randomness is the initial vorticity (lines 211-216) and, only if `markov_P` is given, the forcing-state chain (lines 132-137). Within a fixed-law era the null windows are a deterministic function of the initial condition. So every route that needs beta/phi/alpha-mixing of the windows (Berbee coupling, thinning with an error term, Chase's own C35-1, L36 embargo R3) rests on a coefficient that is 1 when the window contains the full field (Proposition E0). Independence on the bed exists only across trajectories.
- K2. The regime chain carries about 1.34 independent samples per 400-step trajectory (E2). Any per-trajectory concentration for it is vacuous at n = 400; the unit of concentration is the trajectory, as L36c already says.

---

## P1. Concentration for mixing Markov chains and beta-mixing sequences (Lean L10 OPEN)

### Corpus candidates (none closes it)

**Family 093, Dimension-free logarithmic Sobolev inequality for subgaussian log-concave measures.** PDF `preprints/A-dimension-free-logarithmic-Sobolev-inequality-for-subgaussian-log-concave-measures-September-23-2026/paper.pdf`, Theorem 1.1, p. 1 (Corollary 1.2, p. 2). Statement (nurse, pdftotext plus page render): absolute C such that for every centered log-concave measure with Lebesgue density on R^n and every a with sup_theta E exp(<X,theta>^2/a^2) <= 2, Ent(f^2) <= C a^2 integral |Df|^2 dmu. Hypotheses: log-concave density on R^n, centered, linearly a-subgaussian. HOLDS on our bed: NO. The regime-conditional law of a window feature is a finite mixture over four laws (multimodal, hence not log-concave), and the regime chain itself is a 4-atom measure. The result is also about one measure, not a dependent sample. Via Herbst it gives Gaussian concentration of Lipschitz functions of one log-concave vector, which is not the Markov plug-in statement L10 needs. Upgrades: nothing.

**Family 145, Rokhlin's multiple-mixing problem.** PDF `preprints/Rokhlins-multiple-mixing-problem-for-one-transformation-September-23-2026/paper.pdf`, Theorem 1.1, p. 2. Statement: every invertible mixing probability-preserving transformation is mixing of every finite order (convergence mu(A_1 ∩ T^-n1 A_2 ∩ ...) -> product as gaps -> infinity). Hypotheses: invertible, measure-preserving, mixing. The nurse found 0 hits for "quantitative" and no rate anywhere in the paper (only "entropy rate"). HOLDS on our bed: unknown (no mixing proof for forced 2-D NS windows exists in the corpus; the two NS manuscripts, family 376, have 0 hits for ergodic, mixing, invariant measure, attractor). Even if it held, a qualitative limit yields no epsilon(G) and no concentration. Upgrades: nothing. Its use is negative: do not write "the windows mix" as an assumption without a rate.

Swept and rejected: 131, 227, 238 (mixing of specific chains, see P4); 273 ("mixing" = beam-splitter), 150, 154 (billiards, ergodic averages, qualitative), 149, 353 (word collisions).

### What actually closes L10 for the bed (not corpus)

L10 as written (`2026-10-11-fcpa-design.md` line 57) cites Paulin 2015 and Chung et al. for a generic chain. For the bed's regime chain the exact computation below makes the citation unnecessary (E2). For a generic reversible chain the standard route is the Poisson equation plus Freedman's martingale Bernstein: CITED-UNVERIFIED (Freedman 1975, Ann. Probab. 3; Paulin 2015, EJP 20; Kontorovich-Ramanan 2008, Ann. Probab. 36; none retrieved this session).

New era lemma enabled: **E2** (below).

---

## P2. E-values, test martingales, anytime-valid inference and change detection under dependence (the L35 hole)

### Corpus candidates

None. The only family touched by "likelihood ratio" is 218 (Ising comparison argument, not a test). Family 376 (forced Navier-Stokes computation) is the nearest dynamical-systems neighbour of the bed and is checked here so nobody cites it by mistake:

**Family 376, Universal computation in forced Navier-Stokes flows.** PDFs `preprints/Universal-Computation-with-Eventually-Stationary-Navier-Stokes-Forcing-September-27-2026/manuscript.pdf` (Theorem 1.1, p. 2) and `preprints/Velocity-Field-Detection-of-Computation-in-Forced-Navier-Stokes-Flows-September-27-2026/manuscript.pdf` (Theorem 1.1, p. 2). Statement: for each positive computable viscosity there is an effective map from Turing machines to smooth body forces on the flat three-torus (second paper: also R^2 x T), zero initial velocity, such that a fixed event occurs iff the machine halts. Hypotheses: dimension 3, constructed non-generic forces, exact trajectories ("no stability under perturbations of the force ... is asserted", second paper p. 2). HOLDS on our bed: NO (bed is 2-D, forcing is a fixed x-independent shear mode `(0,ky,amp)`, and nothing is claimed about statistics). Upgrades: nothing; it neither supports nor refutes mixing of the bed windows (0 hits for ergodic/mixing).

### The hole, restated and sharpened

L24*/L35a needs P(p_t <= u | F_{t-1}) <= u. On the bed this is not merely unverified, it is not provable by any mixing-coefficient route (K1):

**Proposition E0 (beta is 1 for a deterministic full-state window process).** Let W_j be windows of a deterministic flow given the initial condition, each window containing the full state at its last step, and let the law of W_{j+g} be non-atomic. Then beta(sigma(W_1..W_j), sigma(W_{j+g})) = 1 for every gap g.
Proof. W_{j+g} = F(W_j) for a measurable F, so the joint law is carried by the graph Gamma = {(a, F(a))}. The product law gives Gamma mass integral P(W_{j+g} = F(a)) dP(a) = 0 because the marginal has no atoms. A set of mass 1 for the joint and 0 for the product gives TV = 1. QED.
Consequence: the Berbee-coupling bound P(false birth) <= delta + (K-1) beta(G) is vacuous on the bed; the L36b/L36d term B(g) = sum (k_tau - 1) beta(g+1) is vacuous; embargo R3 is dead for every window length, not only L = 400. If the windows are a lossy feature map (low-dimensional phi), E0 does not apply and beta may be below 1: UNVERIFIED, and not estimable from one trajectory.

For the AR(1) surrogate (not the bed) the coupling bound is valid and computable (E1b below).

### Top results (all mine, in order of strength)

**E1. Trajectory-conformal calibration of the detector's running maximum.** Source: Chase L36a (exchangeable rank), applied to the statistic Z = max_t M_t. Not a corpus result. Detail in the "New era maths" section. HOLDS on our bed: the hypotheses are met by construction (independent `simulate` calls give i.i.d. trajectories, see L36c proof), provided calibration trajectories reproduce the test stretch law (H-E1-1..4).

**Union of marginal e-values (KILL, with reason).** CITED-UNVERIFIED (Vovk and Wang 2021, Ann. Statist. 49: the arithmetic mean of e-values is an e-value under arbitrary dependence). Applying it needs only E[e_t] <= 1, which does hold under marginal super-uniformity. But each single e-value is bounded: e_t <= kappa (m+1)^(1-kappa) = 0.5 sqrt(501) = 11.2 at m = 500, kappa = .5 (B = 2.41 in L35b), below 1/delta = 20. An average of e-values bounded by 11.2 can never reach 20, so this route cannot alarm at delta = .05. Replacement: E1, which keeps the product's power.

**Thinned e-detector, deflated (valid for stochastic surrogates only).** Lemma E1b below. Needs beta (or phi) mixing; void on the bed by E0.

### Hypothesis check of E1 on the bed (the dependence check)

Computed in `chase_sanity4.py` (this session, one process, numpy; null AR(1) with exactly uniform marginal p, kappa = .5, delta = .05, T = 400, nu = 200, 3,000 calibration sets of n = 64 null stretches each, one fresh null stretch per set):

| rho | fixed threshold 1/delta = 20: null alarm | E1: null alarm (guarantee <= 0.0462) | median calibrated threshold |
|---|---|---|---|
| 0.0 | 0.0251 | 0.0437 | 12.3 |
| 0.9 | 0.7455 | 0.0500 | 5.5e15 |
| 0.99 | 0.6914 | 0.0430 | 4.0e71 |

(rho = 0.9 value 0.0500 is 1 Monte Carlo sd above the bound; sd = 0.0038 at 3,000 repetitions.) Power after the change (2,000 streams): rho = 0.9, shift mu = 2 sd: detected after nu 0.976, mean delay 48 steps, pre-change false alarm 0.022, against the fixed threshold's pre-change false alarm 0.571. rho = 0.9, mu = 1: 0.488 detected (iid control rho = 0: 0.955). rho = 0.99: mu = 2 gives 0.571, mu = 1 gives 0.141. So the price of exactness under strong dependence is power at small shifts, not validity.

### Which FCPA lemma it repairs

L35a and Lean `L24b` ("no false birth, ever"): replace by the finite-horizon, per-trajectory statement E1. It also removes the fixed-calibration defect H35-1(1) (E[e|cal] > 1 for 19.3 percent of calibration sets), because the p-value calibration set is part of the pipeline and the guarantee is on the pipeline output.

---

## P3. Conformal prediction under dependence, block and cluster conformal (L36)

### Corpus candidates

None. The five "conformal" family hits (071, 072, 218, 280, 282) are conformal maps and conformal field theory.

### Non-corpus status of L36

- L36a (exact rank) and L36c(i) (one window per held-out trajectory) need no dependence assumption: they stand. Mine, proved.
- L36b/L36d (embargo bound B(g) = sum (k_tau - 1) beta(g+1)): by E0 the bound is vacuous when windows hold the full state, which strengthens Chase's earlier conclusion (R3 not viable at L = 400) to "not viable at any L".
- The general non-exchangeable form (coverage >= 1 - alpha - sum of weighted TV distances): CITED-UNVERIFIED (Barber, Candes, Ramdas, Tibshirani 2023, Ann. Statist. 51, "Conformal prediction beyond exchangeability"). Split conformal under beta-mixing: CITED-UNVERIFIED (Oliveira, Orenstein, Ramos, Romano 2024, JMLR, "Split conformal prediction and non-exchangeable data"). Permutation-based time-series conformal under stationarity and weak dependence: CITED-UNVERIFIED (Chernozhukov, Wuthrich, Zhu 2018, JASA). All three carry a TV/beta term of the same type as B(g) and are therefore vacuous on the bed by E0 for full-state windows.
- L36c(ii) needs Hoeffding over n_c = 64 independent trajectories. Not in Mathlib v4.7.0. Replacement: use (i) (rank only, no concentration) or the exact Beta law: for a fixed calibration set the realised miscoverage of the split-conformal rule is Beta(n+1-k, k)-distributed (standard, CITED-UNVERIFIED; the rank argument of L36a gives its mean). With n = 64, k = 62: mean 3/65 = 0.0462, sd = sqrt(3*62/(65^2 * 66)) = 0.0258 (my arithmetic, not simulated).

### Integration

E1 is L36a applied to the detector maximum, so L35 and L36 now share one exchangeability hypothesis (i.i.d. trajectories): a single guard test covers both.

---

## P4. Mixing time and spectral gap for reversible chains (Lean L12 OPEN; regime chain theta = 1 - 4 lambda_sw/3)

### Corpus candidates (none applies)

**Family 227, Critical SK autocorrelation processes ... (spectral-gap member).** PDF `preprints/A-spectral-gap-throughout-the-high-temperature-Sherrington-Kirkpatrick-phase-September-24-2026/main.pdf`, Theorem 1.1, p. 3. Statement (verbatim fragment, 12 words): "the discrete-time heat-bath chain that chooses a uniform site has spectral gap at least 1/(Cn)". Hypotheses: Gaussian SK couplings, 0 < beta < 1 fixed, constant C_beta not explicit, conclusion only "with probability tending to one" over the disorder. HOLDS on our bed: NO (the state space is {-1,1}^n spins with random couplings; ours is a 4-state chain with a known matrix; the constant is not explicit). Upgrades: nothing.

**Family 131, Rapid mixing of graph switches.** PDF folder `preprints/Polynomial-Mixing-of-the-Switch-Chain-for-Every-Graphical-Degree-Sequence-September-25-2026/` (`main.pdf`), Theorem 1.1, p. 1: t_mix(d) <= 2 n^8, spectral gap >= [24 n^2 C(n,4)]^-1. Chain on graphs with a given degree sequence. HOLDS: NO (different state space). Upgrades: nothing.

**Family 238, Thorp shuffle.** Not extracted (a permutation chain; no bearing). UNVERIFIED relevance, rejected by title.

Also noted, not extracted: family 229 (reconstruction thresholds for 3- and 4-state symmetric channels on trees). The 4-state symmetric kernel is exactly our regime chain's kernel, but the family is about branching trees; a path (our case) has information decaying as theta^s, computed exactly below. UNVERIFIED, not needed.

### Exact closure for the bed chain (mine)

See **E2**: spectrum, TV distance, mixing time, relaxation time and variance of occupation sums in closed form. It also shows that L12 as designed is violated: window w = 20 steps against t_mix(1/4) = 165 steps (factor 8.2).

---

## P5. Changepoint and regime detection: optimality, delay lower bounds, multiple changepoints

### Corpus candidates

None (0 hits for changepoint, change-point, CUSUM, Lorden, quickest detection, Pollak, Shiryaev, hidden Markov; the two P5 keyword hits, 225 and 371, are "regime" in the physics sense).

### Non-corpus status

- Optimality and lower bounds: CITED-UNVERIFIED (Lorden 1971, Ann. Math. Stat. 42; Moustakides 1986, Ann. Statist. 14: CUSUM Lorden-optimal; Pollak 1985, Ann. Statist. 13; Lai 1998, IEEE Trans. IT 44, "Information bounds and quick detection of parameter changes in stochastic systems", the statement for dependent data under a quick-convergence condition on the log-likelihood-ratio). Shiryaev-Roberts-type e-detectors with an ARL guarantee: CITED-UNVERIFIED (Shin, Ramdas, Rinaldo 2022, "E-detectors"). None retrieved.
- Hidden hypothesis in all of these: the delay bounds assume the conditional law of the post-change increments is known or estimable, and the false-alarm side (ARL or P(ever)) assumes the validity that fails under dependence. Our L35b delay theorem (bounded increments, drift lambda) is proved and does not need independence of the p-values given the alternative provided H2 (conditional drift >= lambda) holds; H2 is a conditional statement and is as unverified as H35-1.
- E1 interacts with L35b: E1 changes the alarm level from 1/delta_j to the calibrated c. L35b's delay bound becomes E[(tau - nu)^+] <= (ln c + ln((nu+1)(nu+2)) + B)/lambda, i.e. replace ln(1/delta_j) by ln c (proof unchanged: only the level changes). MC in the P2 table: ln c = 36.2 at rho = .9, against ln 20 = 3.0: the delay penalty is a factor of about 12 on that term, consistent with the observed 48 versus 18 steps at mu = 2 (my reading; delay model not otherwise verified for dependent data).

No new era lemma from P5 beyond that substitution.

---

## P6. Model collapse and recursive self-training with guarantees (RSL layer; provenance as anti-collapse, L25')

### Corpus candidates

None (0 hits for model collapse, self-consuming, recursive training, retrain, synthetic data; "recursive" in 233 and 242 means recursive sets and recursive enumerability, "collapse" in 357 is "noncollapsed" RCD spaces).

### Non-corpus status

All CITED-UNVERIFIED, none retrieved: Shumailov et al. 2024 (Nature, "AI models collapse when trained on recursively generated data"): replace-with-synthetic loses tails; Gerstgrasser et al. 2024 ("Is model collapse inevitable?"): accumulating real plus synthetic data keeps linear-regression test error bounded by a constant independent of the number of iterations; Bertrand et al. 2024 ("On the stability of iterative retraining of generative models on their own data", ICLR): stability under a local-Lipschitz condition when a fraction of fresh real data is kept. The accumulate-with-provenance form is the one that matches L25' (provenance tags keep real data identifiable). A proof sketch I trust but have not written: with accumulation the n-th generation is trained on at least n real-weighted samples, so the error recursion e_{n+1} <= e_n (1 - c/n) + sigma^2/n^2-type terms converges; I did not derive constants. UNVERIFIED; no Amendment E lemma offered for P6.

---

## New era maths (lemmas ready for Amendment E)

### Lemma E1 (L35'): finite-horizon birth guarantee valid under arbitrary within-stretch dependence

Setting. A birth pipeline maps a stretch W = (w_1..w_T) of windows to detector values M_1..M_T (any measurable map, in particular the L35 e-detector with p-values from a calibration set C, any kappa grid, any online updates, all frozen as a function of training data D). Let Z(W) = max_{t<=T} M_t(W). Null stretches: trajectories of a fixed-law era, same burn-in, same length T, generated by independent `simulate` calls.
Hypotheses. (H1) Given (D, C), the null stretches W_1..W_n and the test stretch W_{n+1} are i.i.d. (true by construction: L36c proof). (H2) The same pipeline, including every online update, is run on calibration and test stretches. (H3) n >= ceil(1/delta) - 1 (n = 19 for delta = .05).
Statement. Let k = ceil((n+1)(1-delta)) and c = Z_(k) (the k-th smallest of Z(W_1..W_n)). Raise a birth alarm at tau = min{t <= T : M_t > c}. If the test stretch is null, then P(tau <= T) <= 1 - k/(n+1) <= delta, the probability being over calibration and test stretches together. No assumption on temporal dependence inside a stretch, on the marginal law of the p-values, or on super-uniformity of any conditional law. Ties are harmless.
Proof. By H1 the n+1 values Z(W_i) are exchangeable, so the rank of Z(W_{n+1}) (ties broken at random) is uniform on {1..n+1}. {Z(W_{n+1}) > Z_(k)} forces rank > k, which has probability (n+1-k)/(n+1) <= delta. {tau <= T} = {max_t M_t > c} = {Z(W_{n+1}) > c}. QED. This is L36a with the score replaced by Z.
Anytime-validity on [0,T]: the event is about the running maximum, so the guarantee holds for every stopping time <= T simultaneously.
Check. MC table in P2: 0.0437 / 0.0500 / 0.0430 for rho = 0 / .9 / .99 against the bound 0.0462 (3,000 repetitions, sd 0.0038); the unrepaired detector alarms 0.7455 at rho = .9.
Limits (collected once). (a) Finite horizon T; a stream longer than the calibration horizon needs recalibration. (b) The guarantee is per trajectory, marginal over the calibration draw; for a fixed calibration set the realised false-alarm rate is Beta(3, 62)-distributed at n = 64, delta = .05 (mean 0.046, sd 0.026). (c) Pooling the four laws gives a guarantee averaged over the law mixture; per-law (Mondrian) validity needs n >= 19 trajectories per law, and 64/4 = 16 per law gives a floor delta >= 1/17 = 0.059. (d) Test stretches must have the law of the calibration stretches including the onset state; stretches that begin at a regime switch (state carried over from another law) are not exchangeable with stretches that begin at a random initial vorticity, and need their own calibration set. (e) Threshold inflation under strong dependence costs power at small shifts (rho = .99, mu = 1: 0.141 detected versus 0.955 at rho = 0).
Lean-ability: reuse the L36a rank proof (about 120 lines in v4.7.0); E1 adds a three-line corollary. UNVERIFIED in Lean.
Replaces: Lean `L24b` hypothesis "conditionally super-uniform" by "i.i.d. trajectories"; retains Ville only for the independent-p-value toy.

### Lemma E1b (valid for stochastic surrogates only): thinned coupling bound

Statement. Let q_k = p_{kG} for k = 1..K, and let beta(G) be the beta-mixing coefficient between sigma(q_1..q_k) and sigma(q_{k+1}) (at least G steps apart), each q_k super-uniform. Then P(exists k <= K : M_k >= 1/delta) <= delta + (K-1) beta(G).
Proof. Let Q be the law with the same marginals and independent q_k. Under Q, the e-detector is a supermartingale (L35a, hypotheses hold trivially), so Ville gives <= delta. The event is a function of (q_1..q_K), so |P - Q| <= TV(P, Q) <= (K-1) beta(G) by the telescoping of L36b(i). QED.
Numbers (AR(1) rho = .9, T = 400, exact Gaussian beta by quadrature, `chase_sanity4.py`): G = 20: beta = 3.9e-2, K = 20, bound 0.79 (vacuous); G = 50: beta = 1.6e-3, K = 8, bound 0.0615; G = 100: beta = 8.5e-6, K = 4, bound 0.0500. So thinning yields a theorem only at G >= 50 (eight tests in 400 steps), although the empirical alarm at G = 20 is already 0.025 (L35 Evidence). By E0 the bound is void on the bed. Use for the AR toy and as a sanity bridge only.

### Lemma E2 (L10 and L12 closed for the bed's regime chain; two hidden hypotheses exposed)

Setting. Regime chain with kernel P = theta I + (1-theta) Pi on 4 states, Pi the rank-one projector onto the uniform law, theta = 1 - 4 lambda_sw/3 = 0.993333 (lambda_sw = .005).
(a) Powers. P^s = theta^s I + (1 - theta^s) Pi. Proof: Pi is idempotent and P Pi = Pi P = Pi; induct. Spectrum: eigenvalue 1 once, eigenvalue theta three times; P is symmetric, hence reversible; absolute spectral gap 1 - theta = 0.006667, t_rel = 150.
(b) Exact TV. max_x TV(P^s(x,.), pi) = 0.75 theta^s. t_mix(1/4) = ceil(ln 3 / ln(1/theta)) = 165. It satisfies the generic bounds quoted in L12: (t_rel - 1) ln 2 = 103 <= 165 <= t_rel ln(4/pi_min) = 150 ln 16 = 416 (arithmetic).
(c) Occupation sums. For f = 1[X = 0] under stationarity, Cov(f(X_0), f(X_s)) = theta^s sigma^2, sigma^2 = 3/16, so Var(S_n) = sigma^2 [ n(1+theta)/(1-theta) - 2 theta (1 - theta^n)/(1-theta)^2 ]. At n = 400: 14,620.5 exact; Monte Carlo (40,000 chains) 14,644. An iid sequence would have 75. Hence n_eff = n (1-theta)/(1+theta) = 1.34 independent samples per 400-step trajectory. Chebyshev: P(|S_n/n - 1/4| >= eps) <= 0.0914/eps^2, vacuous below eps = 0.30.
Consequences for the design (hidden hypotheses): (i) L12 asks the window to exceed the mixing time; the bed has w = 20 steps against t_mix = 165, so the premise fails by a factor 8.2 by construction. (ii) No per-trajectory plug-in interval (L10 form) can be informative at L = 400; the usable replication unit is the trajectory (128 independent ones), where plain i.i.d. Chebyshev with the exact variance above is available and Hoeffding is not needed.
Lean-ability (v4.7.0, mine, UNVERIFIED): (a) is induction on 4x4 real matrices; (b) is a finite sum; (c) is a geometric series; none needs Mathlib's missing Hoeffding or reversible-chain spectral theory. Closes L12 and the Chebyshev form of L10 for this chain; the generic CITED lemmas stay cited.
Beta of the regime process alone is 0.75 theta^s (L36 table), a lower bound for the beta of the observed windows; E0 shows the observed windows can be worse.

### Ranking of what to adopt

1. E1 (it is the repaired birth guarantee; needs only the trajectory split already adopted as L36 R1).
2. E0 (retires every mixing-coefficient route on the bed; prevents a wrong theorem from entering Amendment E).
3. E2 (closes two Lean OPEN items for the regime chain in elementary linear algebra and exposes the L12 premise failure).

## Open numbers I did not obtain
- Whether the bed's windows are full-state (E0 applies) or lossy features (E0 silent): needs a read of `fcpa` window construction; not done.
- Power of E1 on real NS windows (only AR(1) surrogate tested).
- Every CITED-UNVERIFIED reference above.

# Era lemmas L38–L41: statements, proofs, status

Lean 4 (`leanprover/lean4:v4.7.0`, Mathlib v4.7.0 tag `a45ae63`), files `FCPA/L38_*`, `L39_*`, `L39c_*`, `L40_*`, `L41_*`.
`python check_no_sorry.py`: 0 sorry. Objects are the code in `zero/research/e0/` (`arms.py::fcpa`, `deciders.py::Soft`,
`deciders.py::Momentum`), not an idealisation of it.

## Status

| Lemma | Claim | Status | Lean |
|---|---|---|---|
| L38 | Regret of the TT filter over a bank that grows by births and shrinks by merges, any linear decider (SOFT, MOM) | **PROVED**, pathwise | `L38.bank_regret`, `bank_regret_simple`, `bank_regret_fixed`, `window_regret`, `butterfly` |
| L39 (exact) | Merge rule = address rule applied to a whole era; both directions agree; BF is an e-value | **PROVED** | `L39.chain_rule`, `preq_eq`, `merge_iff_address`, `address_symm`, `false_merge_le`, `seq_mass_one` |
| L39 (consistency) | `log BF = -Div + Gain ± K`: divergence linear in evidence, Occam gain logarithmic | **PROVED** (deterministic); sampling bridge CITED | `L39.log_bf_sandwich`, `refuse_merge`, `accept_merge`, `same_law_merges`, `div_row_nonneg` |
| L40 | No forgetting for count eras | **PROVED** (settled tokens); unconditional form FALSE | `L40.no_forgetting`, `untouched_era_unchanged`, `count_birth_old_unchanged` |
| L41 | MOM is an HMM forward filter; switch cost `-log((α/2)Q(j,i))`, learned `Q` included | **PROVED** | `L41.mom_is_move`, `mom_eq_hmm_predict`, `mom_switch_cost`, `mom_regret_tail` |

## L38: regret over a growing bank

**Model (exactly the code).** Pool `s_t` (eras, window, BOCPD). Per token:
- move `v_t(i) = (1-α) w_t(i) + ρ/|s_t| + Σ_j w_t(j) K_t(j,i)`;
- predict `p_t = Σ_i v_t(i) P_t(i)`;
- Bayes `u_t(i) = v_t(i) P_t(i)/p_t` ("multiply by likelihood, then normalise");
- between tokens: `birth` (`w ↦ (1-α)w + α·[new]`) and `fold i j` (weights add).
- SOFT: `ρ = α, K = 0`. MOM: `ρ = α/2, K = (α/2)Q̃` (L41).

**Theorem (`bank_regret`).** For every token sequence, every expert prediction sequence, every birth/merge schedule and
every comparator path `c_t ∈ s_t`:

`Σ_{t≤T} -log p_t ≤ Σ_{t≤T} -log P_t(c_t) + log(|s_0|/(1-α+ρ)) + Σ_{t<T} cost_t`

- stay (comparator follows its era through merges): `cost = -log(1-α+K(c,c)) + b_t log(1/(1-α))`;
- switch `j → i`: `cost = -log max(ρ/N, K(j,i)(1-α)^{b_t})` ≤ `log(N/ρ)`;
- `b_t` = births after token `t`, `N` ≥ pool size.

**Readable forms.**
- `bank_regret_simple`: `regret ≤ log(|s_0|/(1-α+ρ)) + S log(N/ρ) + (T+B) log(1/(1-α))`.
- `bank_regret_fixed` (vs BOCPD, never folded): `≤ log(|s_0|/(1-α+ρ)) + T log(1/(1-α+κ)) + B log(1/(1-α))`, `κ` = BOCPD's
  self-kernel (`0` SOFT, `α/2` MOM).
- `window_regret`: from any start time, entering a path costs at most `log(N/ρ)` (weight floor `ρ/N`).
- `butterfly`: one token moves `log w(i)` by at most `log(max P / min P)` (Addendum 2, item 4).

**Proof idea.** Telescoping: `-log p_t = -log P_t(c_t) - log v_t(c_t) + log u_t(c_t)`, and every transport step keeps
`v_{t+1}(c_{t+1}) ≥ γ_t u_t(c_t)`.

**Checked against the code (E0e config, α = 0.003, seeds 40–42, REC / NOREC / CYC-S).** Bound holds on all 18 runs.
On NOREC it is nearly tight:

| | regret vs BOCPD per token (measured) | L38 bound |
|---|---|---|
| SOFT | 0.0024–0.0029 | 0.0031 |
| MOM | 0.0012–0.0015 | 0.0016 |

**What it explains.**
- B2/G2 ("era memory is nearly free when nothing recurs") is now a theorem: overhead vs BOCPD ≤ `α/(1-α)` per token (SOFT),
  `≈ α/2` (MOM).
- Design rule: G2 (NOREC ≤ 1.25 × BOCPD) is guaranteed when the overhead ≤ 0.25 × BOCPD's regret (0.0156 → α ≤ 0.0039
  SOFT, α ≤ 0.0078 MOM). E0d round 1 picked α = 0.01, outside the guaranteed range, and five of six deciders failed G2.
  E0e's α = 0.003 is inside it for both, and both passed. (The rule is sufficient, not necessary.)

**Not covered.** Regret against the *oracle* (needs era ≈ regime estimation: L10/L12, CITED). Non-linear answers
(CTC, TEMP, HARD) and tempered likelihoods (GBAYES) are outside the model.

## L39: the merge rule

**Objects.** Order-1 count tables `C : R → X → ℕ`, Dirichlet(`a`) rows, `ML` = Dirichlet–multinomial marginal. Merge
fires when `log BF = log ML(C+D) - log ML(C) - log ML(D) > 0`.

**Exact part (`L39_MergeRule`).**
- `chain_rule`: `ML(C+D) = ML(C) · ML(D | C)`.
- `preq_eq`: the address loop's prequential product (`ll += log pred(E)(x); E += x`) equals `ML(counts | C)`, for any
  token order.
- `merge_iff_address`: **merge(C, D) ⇔ era C beats a fresh era on D's tokens in the address step.** The merge rule is the
  address rule applied to a whole era.
- `address_symm`: C addressing D's data ⇔ D addressing C's data. The rules cannot disagree.
- `false_merge_le` / `false_split_le`: BF is an e-value: under the separate-eras model, `P(BF ≥ 1/δ) ≤ δ`; symmetric for
  splits. `seq_mass_one`: the sequential Dirichlet model is a probability model, so this applies.
- **Finding:** the code's threshold `BF > 1` is `δ = 1`: no finite-sample guarantee (`threshold_one_vacuous`). Candidate
  for a tournament: `log BF > log(1/δ)`.

**Consistency part (`L39c_MergeConsistency`).** Exact per-row decomposition
`log ML(c) = maxll(c) + occam(c) + Θ(c)`:
- `maxll(c) = Σ c_x log(c_x/n)`;
- `occam(c) = (a-½) Σ log c_x - (Va-½) log n`;
- `|Θ(c)| ≤ Vκ(a) + κ(Va)`, `κ(b) = |log b| + 1 + b + 2b²` (Stirling, `theta_bounds`).

`log_bf_sandwich`: `|log BF - (-Div + Gain)| ≤ K = 3|R|(Vκ(a) + κ(Va))`.
- `Div ≥ 0` (Gibbs, `div_row_nonneg`) = `N ×` weighted Jensen–Shannon divergence: linear in evidence when laws differ.
- `Gain` = Occam gain, logarithmic. Same empirical law, full support: `Div = 0` and
  `Gain = ((V-1)/2) log(n_c n_d/N) + (½-a) Σ log ĉ_x` exactly (`proportional_div`, `proportional_gain`).
- `refuse_merge`: `Div > Gain + K ⇒` no merge. `accept_merge`: `Div + K < Gain ⇒` merge. `same_law_merges`.

**Strongest true statement:** different laws are refused once `Div` (∝ N) beats `O(log N) + K`; same laws are merged once
the Occam gain beats `K`. **Cited, not formalised:** sampling bridge (`Div/N → JS > 0` a.s. by the strong law; `2Div → χ²`
by Wilks).

**False versions (rejected).** "BF > 1 never merges different laws once each era has ≥ n_min tokens": false for every
n_min (two laws close enough give identical counts with probability → 1). n_min = 64 is a tuned value, not derived.

**Limit.** `K` is loose: V = 16, a = 0.2, |R| = 16 gives K ≈ 3.5·10³ nats. The theorem fixes rates (linear vs log), not
thresholds. Tightening needs Gautschi-type bounds on `Γ(a+k)/k!`.

## L40: no forgetting

**Model.** Ledger of credited transitions `(time, context, symbol, era)`; an era's table counts its tokens. Ops = all the
code does: `credit`, `birth`, `fold i j`, `readdress τ b` (the address step takes tokens after the change-time estimate
`τ` out of their eras and re-credits them). A token is *settled* at `τ₀` if `time ≤ τ₀`; the code's `det.tmin` is never
before the last decision, so tokens seen before a decision stay settled.

- `count_birth_old_unchanged`: birth leaves every old table unchanged (L23′ for count eras).
- `settled_kept`: no op ever loses a settled token; each is relabelled only by the composed merge map `Φ`.
- `no_forgetting`: `table_after(Φ ℓ) ≥` settled counts of `ℓ` before, for every era and transition.
- `untouched_era_unchanged`: an era no op touches keeps exactly its table (hence exactly its predictive) through any
  excursion of any length.
- `ema_forgets`: contrast: EMA keeps `λ^L` of an era's counts after `L` steps away (→ 0).
- Filter side: `L38.window_regret`. Re-entering an era costs at most `log(N/ρ)` once, plus the stay costs.

**False version (rejected).** "An era's counts change only through tokens assigned to it": false. A re-address removes
unsettled tokens (credited after the change-time estimate) from the era they were credited to, and a merge adds another
era's tokens. Counterexample: credit a token at time 10 to era 0, then `readdress 5 1`; era 0 loses it. The proved form
restricts to settled tokens and lets merges add.

## L41: MOM (tournament winner)

**MOM's move step (code):** `v = (1-α)w + (α/2)/n + (α/2)(w Q̃)`; `Q̃` = learned `Q` on eras (`Q` = switch counts + 1,
zero diagonal, row-normalised), identity on tail experts.

- `mom_is_move`: MOM = L38 with `ρ = α/2, K = (α/2)Q̃`. Every L38 theorem applies.
- `mom_transition_stochastic`, `mom_eq_hmm_predict`: `A = (1-α)I + (α/2)J/n + (α/2)Q̃` is row-stochastic and
  `v = wA`: **MOM is an HMM forward filter** (Bayes step = posterior by L14′ `hmm_filter_is_posterior`).
- `mom_switch_cost`: switch `j → i` along the kernel costs `≤ -log((α/2)Q̃(j,i)) + b log(1/(1-α))`; any switch
  `≤ log(2N/α)`.
- `momQ_pos`, `momQ_row_sum`: the coded `Q` is positive off the diagonal and stochastic.
- `mom_regret_tail`: vs BOCPD, MOM pays `log(1/(1-α/2))` per token, half of SOFT.
- `soft_is_move`, `mom_identity_is_soft` (`Q̃ = I` is SOFT at α/2), `mom_uniform_is_soft` (`Q̃ = J/n` is SOFT at α).
- **Learned Q:** the bound is pathwise with the kernel in force at each step, so learned `Q` is covered as is. Still open:
  how fast the learned `Q` approaches the true switch law.

**What it explains (bound-level, not a loss identity).** Per switch along a learned cycle (`Q̃ ≤ 0.9` = `p_cycle`,
α = 0.003), MOM pays `log(2/(αQ̃)) ≈ 6.6` nats, SOFT pays `log(N/α)`, `N` = eras + 2 tail experts ≈ 6–9. Bound-level saving
per switch `log(NQ̃/2)` ≈ 1.0–1.4 nats. Segment length = `seg_min + Geom` with mean `seg_mean` (`bed.py`), T = 20 000:

| bed | switches | predicted saving (nats/token) | measured SOFT − MOM (E0e) |
|---|---|---|---|
| CYC-S (mean 100) | ≈ 199 | 0.010–0.014 | 0.0077 |
| CYC (mean 350) | ≈ 57 | 0.0028–0.0040 | 0.0022 |

- Measured is 55–80% of the bound-level saving: bounds over-predict, as expected (learned `Q̃` starts flat).
- The ratio matches: switches 199/57 = 3.5; measured gaps 0.0077/0.0022 = 3.5. The saving scales with the number of
  switches, as the theorem says.

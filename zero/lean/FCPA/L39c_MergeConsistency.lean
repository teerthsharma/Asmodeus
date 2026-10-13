import Mathlib.Analysis.SpecialFunctions.Stirling
import Mathlib.NumberTheory.Harmonic.Bounds
import Mathlib.Topology.Order.MonotoneConvergence
import FCPA.L39_MergeRule

/-!
L39 (research program, Addendum 2), consistency part: a two-sided, finite-sample, fully explicit sandwich for the merge
test of `L39_MergeRule`.

For an order-1 count table, row by row (`V = |X|`, Dirichlet(`a`), `n` = row total):
`log ML(c) = maxll(c) + occam(c) + Θ(c)` **exactly**, with
* `maxll(c) = Σ_x c_x log(c_x / n)` — the maximised log-likelihood (`maxll_eq`),
* `occam(c) = (a - 1/2) Σ_x log c_x - (V a - 1/2) log n` — the Occam term,
* `|Θ(c)| ≤ V κ(a) + κ(V a)`, `κ(b) = |log b| + 1 + b + 2b²` (Stirling, `theta_bounds`).

So for two eras `C, D` (`log_bf_sandwich`):
`| log BF(C, D) - ( -Div(C, D) + Gain(C, D) ) | ≤ 3 |R| (V κ(a) + κ(V a))`, where
* `Div = Σ_rows [maxll(c) + maxll(d) - maxll(c+d)] ≥ 0` (`div_nonneg`, Gibbs) is `N ·` the weighted Jensen–Shannon
  divergence of the two eras' empirical laws: it grows **linearly** in the evidence when the laws differ;
* `Gain = Σ_rows [occam(c+d) - occam(c) - occam(d)]` grows only **logarithmically**; for two rows of the same empirical
  law with full support it is exactly `((V-1)/2) log(n_c n_d / (n_c + n_d)) + (1/2 - a) Σ_x log(c_x/n_c)`
  (`proportional_gain`, `proportional_div`).

Consequences (`refuse_merge`, `accept_merge`, `same_law_merges`): the merge fires iff, up to the explicit constant,
`Div < Gain`. Two eras of different laws (`Div ≥ ε N`) are never merged once `ε N` exceeds `O(log N)`; two eras holding
the same law (`Div = 0`) are always merged once the Occam gain `((V-1)/2) log(n_c n_d / N)` exceeds the constant.
The statistical bridge (sampled counts: `Div/N → JS(f, g) > 0` a.s. for different laws by the strong law; `2 Div →
χ²` for equal laws by Wilks, so `Div = O_P(1)`) is **cited**, not formalised here.
-/
open BigOperators Real Finset

namespace FCPA
namespace L39

/-! ### Stirling bounds for `log k!` and `log rise b k` -/

theorem log_factorial_bounds (k : ℕ) (hk : 1 ≤ k) :
    log π / 2 + log 2 / 2 + log k / 2 + k * log k - k ≤ log (k.factorial : ℝ) ∧
      log (k.factorial : ℝ) ≤ 1 + log k / 2 + k * log k - k := by
  obtain ⟨m, rfl⟩ : ∃ m, k = m + 1 := ⟨k - 1, by omega⟩
  have hσpos := Stirling.stirlingSeq'_pos m
  have hanti := Stirling.stirlingSeq'_antitone
  have hup : Stirling.stirlingSeq (m + 1) ≤ exp 1 / Real.sqrt 2 := by
    have := hanti (Nat.zero_le m)
    simpa using this
  have hlim : Filter.Tendsto (Stirling.stirlingSeq ∘ Nat.succ) Filter.atTop (nhds (Real.sqrt π)) :=
    Stirling.tendsto_stirlingSeq_sqrt_pi.comp (Filter.tendsto_add_atTop_nat 1)
  have hlo : Real.sqrt π ≤ Stirling.stirlingSeq (m + 1) := hanti.le_of_tendsto hlim m
  have hform := Stirling.log_stirlingSeq_formula (m + 1)
  have hk0 : (0 : ℝ) < ((m + 1 : ℕ) : ℝ) := by positivity
  have h2k : log (2 * ((m + 1 : ℕ) : ℝ)) = log 2 + log ((m + 1 : ℕ) : ℝ) := log_mul (by norm_num) hk0.ne'
  have hke : log (((m + 1 : ℕ) : ℝ) / exp 1) = log ((m + 1 : ℕ) : ℝ) - 1 := by
    rw [log_div hk0.ne' (exp_pos 1).ne', log_exp]
  have hlu : log (Stirling.stirlingSeq (m + 1)) ≤ 1 - log 2 / 2 := by
    have := log_le_log hσpos hup
    rwa [log_div (exp_pos 1).ne' (by positivity), log_exp, log_sqrt (by norm_num)] at this
  have hll : log π / 2 ≤ log (Stirling.stirlingSeq (m + 1)) := by
    have := log_le_log (Real.sqrt_pos.mpr Real.pi_pos) hlo
    rwa [log_sqrt Real.pi_pos.le] at this
  rw [h2k, hke] at hform
  push_cast at hform hll hlu ⊢
  constructor <;> nlinarith [hform, hll, hlu]

theorem sum_inv_sq_le_two (m : ℕ) : ∑ i in range m, 1 / ((i : ℝ) + 1) ^ 2 ≤ 2 := by
  have aux : ∀ m : ℕ, ∑ i in range (m + 1), 1 / ((i : ℝ) + 1) ^ 2 ≤ 2 - 1 / ((m : ℝ) + 1) := by
    intro m
    induction m with
    | zero => norm_num
    | succ m ih =>
      rw [sum_range_succ]
      have h1 : (0 : ℝ) < (m : ℝ) + 1 := by positivity
      have key : 1 / (((m + 1 : ℕ) : ℝ) + 1) ^ 2 ≤ 1 / ((m : ℝ) + 1) - 1 / (((m + 1 : ℕ) : ℝ) + 1) := by
        push_cast
        rw [div_sub_div _ _ h1.ne' (by positivity), div_le_div_iff (by positivity) (by positivity)]
        nlinarith
      push_cast at key ⊢
      linarith
  cases m with
  | zero => simp
  | succ m =>
    have := aux m
    have : (0 : ℝ) ≤ 1 / ((m : ℝ) + 1) := by positivity
    linarith

/-- `log(1+y)` between `y - y²` and `y` for `y ≥ 0`. -/
theorem log_one_add_bounds {y : ℝ} (hy : 0 ≤ y) : y - y ^ 2 ≤ log (1 + y) ∧ log (1 + y) ≤ y := by
  have h1 : 0 < 1 + y := by linarith
  constructor
  · have := log_le_sub_one_of_pos (inv_pos.mpr h1)
    rw [log_inv] at this
    have h2 : y - y ^ 2 ≤ 1 - (1 + y)⁻¹ := by
      rw [show 1 - (1 + y)⁻¹ = y / (1 + y) by field_simp]
      rw [le_div_iff h1]
      nlinarith [sq_nonneg y, mul_nonneg hy (sq_nonneg y)]
    linarith
  · have := log_le_sub_one_of_pos h1
    linarith

theorem log_rise_bounds {b : ℝ} (hb : 0 < b) (m : ℕ) :
    log b + log (m.factorial : ℝ) + b * ∑ i in range m, 1 / ((i : ℝ) + 1)
        - b ^ 2 * ∑ i in range m, 1 / ((i : ℝ) + 1) ^ 2 ≤ log (rise b (m + 1)) ∧
      log (rise b (m + 1)) ≤ log b + log (m.factorial : ℝ) + b * ∑ i in range m, 1 / ((i : ℝ) + 1) := by
  induction m with
  | zero => simp [rise]
  | succ m ih =>
    have hm : (0 : ℝ) < (m : ℝ) + 1 := by positivity
    have hr : 0 < rise b (m + 1) := rise_pos hb _
    have hstep : rise b (m + 1 + 1) = rise b (m + 1) * (b + ((m + 1 : ℕ) : ℝ)) := rfl
    have hfac : ((m + 1).factorial : ℝ) = ((m : ℝ) + 1) * (m.factorial : ℝ) := by
      push_cast [Nat.factorial_succ]; ring
    have hfpos : (0 : ℝ) < (m.factorial : ℝ) := by exact_mod_cast Nat.factorial_pos m
    have hb1 : b + ((m + 1 : ℕ) : ℝ) = ((m : ℝ) + 1) * (1 + b / ((m : ℝ) + 1)) := by
      push_cast; field_simp; ring
    have hy : 0 ≤ b / ((m : ℝ) + 1) := by positivity
    have hlog1 := log_one_add_bounds hy
    have hpos1 : 0 < 1 + b / ((m : ℝ) + 1) := by positivity
    rw [hstep, log_mul hr.ne' (by positivity), hb1, log_mul hm.ne' hpos1.ne', hfac, log_mul hm.ne' hfpos.ne',
      sum_range_succ, sum_range_succ]
    have hsq : (b / ((m : ℝ) + 1)) ^ 2 = b ^ 2 * (1 / ((m : ℝ) + 1) ^ 2) := by field_simp
    have hli : b / ((m : ℝ) + 1) = b * (1 / ((m : ℝ) + 1)) := by ring
    constructor <;> nlinarith [ih.1, ih.2, hlog1.1, hlog1.2]

/-- Remainder of the rising factorial against its Stirling main term. -/
noncomputable def theta (b : ℝ) (k : ℕ) : ℝ := log (rise b k) - ((k + b - 1 / 2) * log k - k)

theorem theta_zero (b : ℝ) : theta b 0 = 0 := by simp [theta]

theorem theta_bounds {b : ℝ} (hb : 0 < b) {k : ℕ} (hk : 1 ≤ k) :
    log b + log π / 2 + log 2 / 2 - 2 * b ^ 2 ≤ theta b k ∧ theta b k ≤ log b + 1 + b := by
  obtain ⟨m, rfl⟩ : ∃ m, k = m + 1 := ⟨k - 1, by omega⟩
  have hR := log_rise_bounds hb m
  have hF := log_factorial_bounds (m + 1) (by omega)
  have hS := sum_inv_sq_le_two m
  have hH1 : log ((m + 1 : ℕ) : ℝ) ≤ ∑ i in range m, 1 / ((i : ℝ) + 1) := by
    have := log_add_one_le_harmonic m
    simpa [harmonic, Rat.cast_sum] using this
  have hH2 : ∑ i in range m, 1 / ((i : ℝ) + 1) ≤ 1 + log ((m + 1 : ℕ) : ℝ) := by
    have h := harmonic_le_one_add_log m
    have : (harmonic m : ℝ) = ∑ i in range m, 1 / ((i : ℝ) + 1) := by simp [harmonic, Rat.cast_sum]
    rw [this] at h
    have hmono : log (m : ℝ) ≤ log ((m + 1 : ℕ) : ℝ) := by
      rcases Nat.eq_zero_or_pos m with h0 | h0
      · subst h0; simp
      · exact log_le_log (by exact_mod_cast h0) (by push_cast; linarith)
    linarith
  have hfac : (((m + 1).factorial : ℕ) : ℝ) = ((m + 1 : ℕ) : ℝ) * (m.factorial : ℝ) := by
    push_cast [Nat.factorial_succ]; ring
  have hk0 : (0 : ℝ) < ((m + 1 : ℕ) : ℝ) := by positivity
  have hfpos : (0 : ℝ) < (m.factorial : ℝ) := by exact_mod_cast Nat.factorial_pos m
  have hlf : log (((m + 1).factorial : ℕ) : ℝ) = log ((m + 1 : ℕ) : ℝ) + log (m.factorial : ℝ) := by
    rw [hfac, log_mul hk0.ne' hfpos.ne']
  have hb2 : 0 ≤ b ^ 2 := sq_nonneg b
  have hS' : b ^ 2 * ∑ i in range m, 1 / ((i : ℝ) + 1) ^ 2 ≤ 2 * b ^ 2 := by nlinarith
  have e1 : b * log ((m + 1 : ℕ) : ℝ) ≤ b * ∑ i in range m, 1 / ((i : ℝ) + 1) := mul_le_mul_of_nonneg_left hH1 hb.le
  have e2 : b * ∑ i in range m, 1 / ((i : ℝ) + 1) ≤ b * (1 + log ((m + 1 : ℕ) : ℝ)) :=
    mul_le_mul_of_nonneg_left hH2 hb.le
  unfold theta
  have hexp : (((m + 1 : ℕ) : ℝ) + b - 1 / 2) * log ((m + 1 : ℕ) : ℝ) =
      ((m + 1 : ℕ) : ℝ) * log ((m + 1 : ℕ) : ℝ) + b * log ((m + 1 : ℕ) : ℝ) - log ((m + 1 : ℕ) : ℝ) / 2 := by ring
  rw [hexp]
  constructor <;> nlinarith [hR.1, hR.2, hF.1, hF.2, hlf]

/-- Uniform bound on the remainder. -/
noncomputable def kappa (b : ℝ) : ℝ := |log b| + 1 + b + 2 * b ^ 2

theorem abs_theta_le {b : ℝ} (hb : 0 < b) (k : ℕ) : |theta b k| ≤ kappa b := by
  unfold kappa
  rcases Nat.eq_zero_or_pos k with h | h
  · subst h; rw [theta_zero, abs_zero]; positivity
  · have hθ := theta_bounds hb (k := k) h
    have hπ : 0 ≤ log π := log_nonneg (by linarith [Real.pi_gt_three])
    have h2 : 0 ≤ log 2 := log_nonneg (by norm_num)
    rw [abs_le]
    constructor <;> nlinarith [abs_nonneg (log b), le_abs_self (log b), neg_abs_le (log b), sq_nonneg b]

/-! ### Exact row decomposition -/

variable {R X : Type*} [Fintype R] [Fintype X] [DecidableEq R] [DecidableEq X]

/-- Log marginal likelihood of one row. -/
noncomputable def logMLrow (a : ℝ) (c : X → ℕ) : ℝ :=
  ∑ x, log (rise a (c x)) - log (rise (Fintype.card X * a) (rowSum c))

/-- Maximised log-likelihood `Σ c log c - n log n` (`= Σ c log(c/n)`, `maxll_eq`). -/
noncomputable def maxll (c : X → ℕ) : ℝ := ∑ x, (c x : ℝ) * log (c x) - (rowSum c : ℝ) * log (rowSum c)

/-- Occam term. -/
noncomputable def occam (a : ℝ) (c : X → ℕ) : ℝ :=
  (a - 1 / 2) * ∑ x, log (c x : ℝ) - (Fintype.card X * a - 1 / 2) * log (rowSum c : ℝ)

/-- Remainder. -/
noncomputable def bigTheta (a : ℝ) (c : X → ℕ) : ℝ := ∑ x, theta a (c x) - theta (Fintype.card X * a) (rowSum c)

theorem rowSum_cast (c : X → ℕ) : (rowSum c : ℝ) = ∑ x, (c x : ℝ) := by simp [rowSum]

/-- **Exact decomposition** `log ML = maxll + occam + Θ`. -/
theorem logMLrow_decomp (a : ℝ) (c : X → ℕ) : logMLrow a c = maxll c + occam a c + bigTheta a c := by
  unfold logMLrow maxll occam bigTheta theta
  have hn := rowSum_cast c
  simp only [sum_sub_distrib, sub_mul, add_mul, sum_add_distrib, ← sum_mul, ← mul_sum]
  rw [← hn]
  ring

theorem maxll_eq (c : X → ℕ) : maxll c = ∑ x, (c x : ℝ) * log ((c x : ℝ) / rowSum c) := by
  unfold maxll
  rw [rowSum_cast, sum_mul, ← sum_sub_distrib]
  apply sum_congr rfl
  intro x _
  rcases Nat.eq_zero_or_pos (c x) with h | h
  · simp [h]
  · have hn : (0 : ℝ) < ∑ y, (c y : ℝ) :=
      lt_of_lt_of_le (by exact_mod_cast h) (single_le_sum (f := fun y => (c y : ℝ)) (fun y _ => by positivity)
        (mem_univ x))
    rw [log_div (by positivity) hn.ne']
    ring

theorem abs_bigTheta_le {a : ℝ} (ha : 0 < a) [Nonempty X] (c : X → ℕ) :
    |bigTheta a c| ≤ Fintype.card X * kappa a + kappa (Fintype.card X * a) := by
  have hV : (0 : ℝ) < Fintype.card X := Nat.cast_pos.mpr Fintype.card_pos
  unfold bigTheta
  have h1 : |∑ x, theta a (c x)| ≤ Fintype.card X * kappa a := by
    calc |∑ x, theta a (c x)| ≤ ∑ x, |theta a (c x)| := abs_sum_le_sum_abs _ _
      _ ≤ ∑ _x : X, kappa a := sum_le_sum fun x _ => abs_theta_le ha _
      _ = Fintype.card X * kappa a := by simp
  have h2 := abs_theta_le (b := Fintype.card X * a) (by positivity) (rowSum c)
  calc |∑ x, theta a (c x) - theta (Fintype.card X * a) (rowSum c)|
      ≤ |∑ x, theta a (c x)| + |theta (Fintype.card X * a) (rowSum c)| := abs_sub _ _
    _ ≤ _ := add_le_add h1 h2

/-! ### Gibbs: the divergence term is nonnegative -/

/-- `maxll(c) ≥ Σ c_x log m_x` for any sub-probability `m` positive where `c` is. -/
theorem gibbs_row (c : X → ℕ) (m : X → ℝ) (hm0 : ∀ x, 0 ≤ m x) (hmpos : ∀ x, 0 < c x → 0 < m x)
    (hm1 : ∑ x, m x ≤ 1) : ∑ x, (c x : ℝ) * log (m x) ≤ maxll c := by
  rw [maxll_eq]
  have hn0 : (0 : ℝ) ≤ rowSum c := Nat.cast_nonneg _
  have key : ∀ x, (c x : ℝ) * log (m x) + ((c x : ℝ) - rowSum c * m x) ≤ (c x : ℝ) * log ((c x : ℝ) / rowSum c) := by
    intro x
    rcases Nat.eq_zero_or_pos (c x) with h | h
    · simp only [h, Nat.cast_zero, zero_mul, zero_sub, zero_add, zero_div, log_zero, mul_zero]
      have := mul_nonneg hn0 (hm0 x)
      linarith
    · have hc : (0 : ℝ) < c x := by exact_mod_cast h
      have hmx := hmpos x h
      have hn : (0 : ℝ) < rowSum c := by
        rw [rowSum_cast]
        exact lt_of_lt_of_le hc (single_le_sum (f := fun y => (c y : ℝ)) (fun y _ => by positivity) (mem_univ x))
      -- `log(c/(n m)) ≥ 1 - n m / c`
      have hl := log_le_sub_one_of_pos (show 0 < rowSum c * m x / c x by positivity)
      have e : log ((c x : ℝ) / rowSum c) - log (m x) = - log (rowSum c * m x / c x) := by
        rw [log_div hc.ne' hn.ne', log_div (by positivity) hc.ne', log_mul hn.ne' hmx.ne']; ring
      have : (c x : ℝ) * (rowSum c * m x / c x - 1) = rowSum c * m x - c x := by field_simp
      nlinarith
  have hsum := sum_le_sum fun x (_ : x ∈ univ) => key x
  rw [sum_add_distrib, sum_sub_distrib, ← mul_sum, ← rowSum_cast] at hsum
  have : (rowSum c : ℝ) - rowSum c * ∑ x, m x ≥ 0 := by nlinarith
  linarith

theorem div_row_nonneg (c d : X → ℕ) : 0 ≤ maxll c + maxll d - maxll (c + d) := by
  set N : ℝ := (rowSum (c + d) : ℝ)
  have hN : N = (rowSum c : ℝ) + rowSum d := by simp [N, rowSum, sum_add_distrib]
  let m : X → ℝ := fun x => ((c + d) x : ℝ) / N
  have hm0 : ∀ x, 0 ≤ m x := fun x => div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  have hm1 : ∑ x, m x ≤ 1 := by
    simp only [m, ← sum_div]
    rw [← rowSum_cast]
    exact div_self_le_one _
  have hpos : ∀ (e : X → ℕ), (∀ x, e x ≤ (c + d) x) → ∀ x, 0 < e x → 0 < m x := by
    intro e he x hx
    have h1 : (0 : ℝ) < ((c + d) x : ℝ) := by exact_mod_cast lt_of_lt_of_le hx (he x)
    have h2 : (0 : ℝ) < N := by
      simp only [N]; rw [rowSum_cast]
      exact lt_of_lt_of_le h1 (single_le_sum (f := fun y => (((c + d) y : ℕ) : ℝ)) (fun y _ => by positivity)
        (mem_univ x))
    exact div_pos h1 h2
  have g1 := gibbs_row c m hm0 (hpos c (fun x => by simp)) hm1
  have g2 := gibbs_row d m hm0 (hpos d (fun x => by simp)) hm1
  have e : maxll (c + d) = ∑ x, (c x : ℝ) * log (m x) + ∑ x, (d x : ℝ) * log (m x) := by
    rw [maxll_eq, ← sum_add_distrib]
    apply sum_congr rfl
    intro x _
    simp only [m, Pi.add_apply, Nat.cast_add]
    ring
  linarith

/-! ### Tables -/

theorem log_ml_eq {a : ℝ} (ha : 0 < a) [Nonempty X] (D : R → X → ℕ) : log (ml a D) = ∑ r, logMLrow a (D r) := by
  have hV : (0 : ℝ) < Fintype.card X := Nat.cast_pos.mpr Fintype.card_pos
  unfold ml mlPost logMLrow
  rw [log_prod _ _ (fun r _ => (div_pos (prod_pos fun x _ => rise_pos (by positivity) _)
    (rise_pos (by positivity) _)).ne')]
  apply sum_congr rfl
  intro r _
  have h0 : rowSum ((0 : R → X → ℕ) r) = 0 := by simp [rowSum]
  simp only [Pi.zero_apply, Nat.cast_zero, add_zero, h0]
  rw [log_div (prod_pos fun x _ => rise_pos ha _).ne' (rise_pos (by positivity) _).ne',
    log_prod _ _ (fun x _ => (rise_pos ha _).ne')]

/-- Divergence term of two eras (`N ·` weighted Jensen–Shannon divergence of their empirical laws). -/
noncomputable def divT (C D : R → X → ℕ) : ℝ := ∑ r, (maxll (C r) + maxll (D r) - maxll ((C + D) r))

/-- Occam gain of merging. -/
noncomputable def gainT (a : ℝ) (C D : R → X → ℕ) : ℝ :=
  ∑ r, (occam a ((C + D) r) - occam a (C r) - occam a (D r))

theorem divT_nonneg (C D : R → X → ℕ) : 0 ≤ divT C D :=
  sum_nonneg fun r _ => by simpa using div_row_nonneg (C r) (D r)

/-- Explicit constant of the sandwich. -/
noncomputable def sandwichK (a : ℝ) (R X : Type*) [Fintype R] [Fintype X] : ℝ :=
  3 * Fintype.card R * (Fintype.card X * kappa a + kappa (Fintype.card X * a))

/-- **L39 (consistency sandwich).** `log BF = -Div + Gain + E` with `|E| ≤ 3|R|(Vκ(a) + κ(Va))`. -/
theorem log_bf_sandwich {a : ℝ} (ha : 0 < a) [Nonempty X] (C D : R → X → ℕ) :
    |log (bf a C D) - (-divT C D + gainT a C D)| ≤ sandwichK a R X := by
  have hC := ml_pos ha C
  have hD := ml_pos ha D
  have hCD := ml_pos ha (C + D)
  have hlog : log (bf a C D) = log (ml a (C + D)) - log (ml a C) - log (ml a D) := by
    unfold bf; rw [log_div hCD.ne' (mul_pos hC hD).ne', log_mul hC.ne' hD.ne']; ring
  rw [hlog, log_ml_eq ha, log_ml_eq ha, log_ml_eq ha]
  have hE : ∑ r, logMLrow a ((C + D) r) - ∑ r, logMLrow a (C r) - ∑ r, logMLrow a (D r) -
      (-divT C D + gainT a C D) =
      ∑ r, (bigTheta a ((C + D) r) - bigTheta a (C r) - bigTheta a (D r)) := by
    unfold divT gainT
    simp only [logMLrow_decomp, ← sum_sub_distrib, ← sum_add_distrib, ← sum_neg_distrib]
    apply sum_congr rfl
    intro r _
    ring
  rw [hE]
  have hV : (0 : ℝ) < Fintype.card X := Nat.cast_pos.mpr Fintype.card_pos
  set B := (Fintype.card X : ℝ) * kappa a + kappa (Fintype.card X * a)
  calc |∑ r, (bigTheta a ((C + D) r) - bigTheta a (C r) - bigTheta a (D r))|
      ≤ ∑ r, |bigTheta a ((C + D) r) - bigTheta a (C r) - bigTheta a (D r)| := abs_sum_le_sum_abs _ _
    _ ≤ ∑ _r : R, 3 * B := by
        apply sum_le_sum
        intro r _
        have h1 := abs_bigTheta_le ha ((C + D) r)
        have h2 := abs_bigTheta_le ha (C r)
        have h3 := abs_bigTheta_le ha (D r)
        calc |bigTheta a ((C + D) r) - bigTheta a (C r) - bigTheta a (D r)|
            ≤ |bigTheta a ((C + D) r) - bigTheta a (C r)| + |bigTheta a (D r)| := abs_sub _ _
          _ ≤ |bigTheta a ((C + D) r)| + |bigTheta a (C r)| + |bigTheta a (D r)| := by
              linarith [abs_sub (bigTheta a ((C + D) r)) (bigTheta a (C r))]
          _ ≤ 3 * B := by linarith
    _ = sandwichK a R X := by simp [sandwichK, B]; ring

/-- **No merge** when the divergence beats the Occam gain by the constant. -/
theorem refuse_merge {a : ℝ} (ha : 0 < a) [Nonempty X] (C D : R → X → ℕ)
    (h : gainT a C D + sandwichK a R X < divT C D) : log (bf a C D) < 0 := by
  have := (abs_le.mp (log_bf_sandwich ha C D)).2
  linarith

/-- **Merge** when the Occam gain beats the divergence by the constant. -/
theorem accept_merge {a : ℝ} (ha : 0 < a) [Nonempty X] (C D : R → X → ℕ)
    (h : divT C D + sandwichK a R X < gainT a C D) : 0 < log (bf a C D) := by
  have := (abs_le.mp (log_bf_sandwich ha C D)).1
  linarith

/-! ### Same empirical law: exact Occam gain, zero divergence -/

section Proportional
variable (a : ℝ) (c d : X → ℕ) (hc : ∀ x, 0 < c x) (hd : ∀ x, 0 < d x)
  (hprop : ∀ x, (c x : ℝ) * rowSum d = (d x : ℝ) * rowSum c)

theorem rowSum_pos_of (e : X → ℕ) (he : ∀ x, 0 < e x) [Nonempty X] : (0 : ℝ) < rowSum e := by
  rw [rowSum_cast]
  obtain ⟨x⟩ := ‹Nonempty X›
  exact lt_of_lt_of_le (by exact_mod_cast he x) (single_le_sum (f := fun y => (e y : ℝ)) (fun y _ => by positivity)
    (mem_univ x))

theorem proportional_logs [Nonempty X] (x : X) :
    log (((c + d) x : ℕ) : ℝ) = log (c x : ℝ) + log (rowSum (c + d) : ℝ) - log (rowSum c : ℝ) ∧
      log (d x : ℝ) = log (c x : ℝ) + log (rowSum d : ℝ) - log (rowSum c : ℝ) := by
  have hnc := rowSum_pos_of c hc
  have hnd := rowSum_pos_of d hd
  have hcx : (0 : ℝ) < c x := by exact_mod_cast hc x
  have hdx : (0 : ℝ) < d x := by exact_mod_cast hd x
  have hN : (rowSum (c + d) : ℝ) = rowSum c + rowSum d := by simp [rowSum, sum_add_distrib]
  have h1 : (((c + d) x : ℕ) : ℝ) * rowSum c = (c x : ℝ) * rowSum (c + d) := by
    rw [hN]; push_cast; nlinarith [hprop x]
  have h2 : (d x : ℝ) * rowSum c = (c x : ℝ) * rowSum d := by linarith [hprop x]
  have hcd : (0 : ℝ) < (((c + d) x : ℕ) : ℝ) := by push_cast; linarith
  have hNp : (0 : ℝ) < rowSum (c + d) := by rw [hN]; linarith
  constructor
  · have := congrArg log h1
    rw [log_mul hcd.ne' hnc.ne', log_mul hcx.ne' hNp.ne'] at this
    linarith
  · have := congrArg log h2
    rw [log_mul hdx.ne' hnc.ne', log_mul hcx.ne' hnd.ne'] at this
    linarith

/-- For two rows of the same empirical law (full support), the Occam gain of merging is exactly
`((V-1)/2) log(n_c n_d / N) + (1/2 - a) Σ log(c_x / n_c)`. -/
theorem proportional_gain [Nonempty X] :
    occam a (c + d) - occam a c - occam a d =
      ((Fintype.card X : ℝ) - 1) / 2 * (log (rowSum c : ℝ) + log (rowSum d : ℝ) - log (rowSum (c + d) : ℝ)) +
        (1 / 2 - a) * ∑ x, log ((c x : ℝ) / rowSum c) := by
  have hnc := rowSum_pos_of c hc
  have hl := fun x => proportional_logs c d hc hd hprop x
  have hcx : ∀ x, (0 : ℝ) < c x := fun x => by exact_mod_cast hc x
  have e1 : ∑ x, log (((c + d) x : ℕ) : ℝ) =
      ∑ x, log (c x : ℝ) + Fintype.card X * (log (rowSum (c + d) : ℝ) - log (rowSum c : ℝ)) := by
    rw [sum_congr rfl (fun x _ => (hl x).1)]
    simp only [add_sub_assoc, sum_add_distrib, sum_const, card_univ, nsmul_eq_mul]
  have e2 : ∑ x, log (d x : ℝ) = ∑ x, log (c x : ℝ) + Fintype.card X * (log (rowSum d : ℝ) - log (rowSum c : ℝ)) := by
    rw [sum_congr rfl (fun x _ => (hl x).2)]
    simp only [add_sub_assoc, sum_add_distrib, sum_const, card_univ, nsmul_eq_mul]
  have e3 : ∑ x, log ((c x : ℝ) / rowSum c) = ∑ x, log (c x : ℝ) - Fintype.card X * log (rowSum c : ℝ) := by
    rw [sum_congr rfl (fun x _ => log_div (hcx x).ne' hnc.ne')]
    simp [sum_sub_distrib]
  unfold occam
  rw [e1, e2, e3]
  ring

/-- Same empirical law ⇒ zero divergence. -/
theorem proportional_div [Nonempty X] : maxll c + maxll d - maxll (c + d) = 0 := by
  have hl := fun x => proportional_logs c d hc hd hprop x
  have hN : (rowSum (c + d) : ℝ) = rowSum c + rowSum d := by simp [rowSum, sum_add_distrib]
  unfold maxll
  have e1 : ∑ x, (((c + d) x : ℕ) : ℝ) * log (((c + d) x : ℕ) : ℝ) =
      ∑ x, (((c + d) x : ℕ) : ℝ) * log (c x : ℝ) +
        (rowSum (c + d) : ℝ) * (log (rowSum (c + d) : ℝ) - log (rowSum c : ℝ)) := by
    rw [sum_congr rfl (fun x _ => by rw [(hl x).1])]
    rw [rowSum_cast, sum_mul, ← sum_add_distrib]
    apply sum_congr rfl; intro x _; ring
  have e2 : ∑ x, (d x : ℝ) * log (d x : ℝ) =
      ∑ x, (d x : ℝ) * log (c x : ℝ) + (rowSum d : ℝ) * (log (rowSum d : ℝ) - log (rowSum c : ℝ)) := by
    rw [sum_congr rfl (fun x _ => by rw [(hl x).2])]
    rw [rowSum_cast d, sum_mul, ← sum_add_distrib]
    apply sum_congr rfl; intro x _; ring
  have e3 : ∑ x, (((c + d) x : ℕ) : ℝ) * log (c x : ℝ) =
      ∑ x, (c x : ℝ) * log (c x : ℝ) + ∑ x, (d x : ℝ) * log (c x : ℝ) := by
    rw [← sum_add_distrib]; apply sum_congr rfl; intro x _; push_cast; ring
  rw [e1, e2, e3, hN]
  ring

end Proportional

/-- **Same law ⇒ merge.** If every row of the two eras has the same empirical law with full support, then
`log BF ≥ Σ_rows [((V-1)/2) log(n_c n_d / N) + (1/2 - a) Σ_x log(c_x / n_c)] - K`; the first term grows without bound
as both eras gather evidence, so the merge fires (`accept_merge`). -/
theorem same_law_merges {a : ℝ} (ha : 0 < a) [Nonempty X] (C D : R → X → ℕ)
    (hc : ∀ r x, 0 < C r x) (hd : ∀ r x, 0 < D r x)
    (hprop : ∀ r x, (C r x : ℝ) * rowSum (D r) = (D r x : ℝ) * rowSum (C r)) :
    ∑ r, (((Fintype.card X : ℝ) - 1) / 2 *
        (log (rowSum (C r) : ℝ) + log (rowSum (D r) : ℝ) - log (rowSum ((C + D) r) : ℝ)) +
        (1 / 2 - a) * ∑ x, log ((C r x : ℝ) / rowSum (C r))) - sandwichK a R X ≤ log (bf a C D) := by
  have hdiv : divT C D = 0 := by
    unfold divT
    apply sum_eq_zero
    intro r _
    exact proportional_div (C r) (D r) (hc r) (hd r) (hprop r)
  have hgain : gainT a C D = ∑ r, (((Fintype.card X : ℝ) - 1) / 2 *
        (log (rowSum (C r) : ℝ) + log (rowSum (D r) : ℝ) - log (rowSum ((C + D) r) : ℝ)) +
        (1 / 2 - a) * ∑ x, log ((C r x : ℝ) / rowSum (C r))) := by
    unfold gainT
    apply sum_congr rfl
    intro r _
    exact proportional_gain a (C r) (D r) (hc r) (hd r) (hprop r)
  have := (abs_le.mp (log_bf_sandwich ha C D)).1
  rw [hdiv, hgain] at this
  linarith

end L39
end FCPA

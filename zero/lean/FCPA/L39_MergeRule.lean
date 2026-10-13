import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
L39 (research program, Addendum 2), exact part: the E0c merge rule (`research/e0/arms.py::fcpa`, `merge="global"`).

Eras are order-1 count tables `C : R → X → ℕ` (`R` = context = previous symbol, `X` = symbol, `V = |X|`) with a
Dirichlet(`a`) prior on every row. The code merges eras `i, j` when the log Bayes factor
`log ML(C_i + C_j) - log ML(C_i) - log ML(C_j) > 0`, and addresses a segment to the bank era whose **prequential**
log-likelihood of the segment (`ll += log pred(E)(x); E += x`) beats a fresh era's.

* `chain_rule`: `ML(C + D) = ML(C) · ML(D | C)` (rising-factorial identity, exact).
* `preq_eq`: the prequential product of a token list equals `ML(counts | C)`; it depends on the counts only (order-free).
* `merge_iff_address`: the merge test on `(C, D)` holds **iff** era `C`, used as an address candidate for era `D`'s tokens
  (in any order), beats a fresh era on them. The merge rule IS the address rule applied to a whole era.
* `bf_symm` / `address_symm`: the decision is the same in both directions (`C` addressing `D`'s data ⇔ `D` addressing
  `C`'s data). The two rules can never disagree.
* `false_merge_le` / `false_split_le`: the Bayes factor is an e-value: under any probability model `q₀` ("separate") the
  chance that `q₁/q₀ ≥ 1/δ` is `≤ δ`, and symmetrically. The code's threshold is `BF > 1` (`δ = 1`), which carries **no**
  finite-sample guarantee; threshold `log BF > log(1/δ)` would carry `δ`.
* `seq_mass_one`: any sequential predictor whose rows sum to one (the Dirichlet predictive `pred_sum_one` is one) gives
  total mass one over all symbol sequences, so `ML` and the merged/separate models are probability models and the e-value
  statement applies to them.
-/
open BigOperators Finset

namespace FCPA
namespace L39

/-- Rising factorial `b (b+1) ⋯ (b+n-1)` (so `Γ(b+n)/Γ(b)`). -/
noncomputable def rise (b : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => rise b n * (b + n)

@[simp] theorem rise_zero (b : ℝ) : rise b 0 = 1 := rfl

theorem rise_add (b : ℝ) (m : ℕ) : ∀ n, rise b (m + n) = rise b m * rise (b + m) n
  | 0 => by simp
  | n + 1 => by
    rw [← add_assoc, rise, rise, rise_add b m n]
    push_cast
    ring

theorem rise_succ' (b : ℝ) (n : ℕ) : rise b (n + 1) = b * rise (b + 1) n := by
  rw [add_comm n 1, rise_add]
  simp [rise]

theorem rise_pos {b : ℝ} (hb : 0 < b) : ∀ n, 0 < rise b n
  | 0 => one_pos
  | n + 1 => mul_pos (rise_pos hb n) (by positivity)

variable {R X : Type*} [Fintype R] [Fintype X] [DecidableEq R] [DecidableEq X]

/-- Row total. -/
def rowSum (c : X → ℕ) : ℕ := ∑ x, c x

/-- Marginal likelihood of the counts `D` under the Dirichlet(`a`) posterior after counts `C` (row by row). -/
noncomputable def mlPost (a : ℝ) (C D : R → X → ℕ) : ℝ :=
  ∏ r, ((∏ x, rise (a + C r x) (D r x)) / rise (Fintype.card X * a + rowSum (C r)) (rowSum (D r)))

/-- Dirichlet–multinomial marginal likelihood `ML(D)`. -/
noncomputable def ml (a : ℝ) (D : R → X → ℕ) : ℝ := mlPost a 0 D

/-- Bayes factor of "one era" against "two eras". -/
noncomputable def bf (a : ℝ) (C D : R → X → ℕ) : ℝ := ml a (C + D) / (ml a C * ml a D)

theorem mlPost_pos {a : ℝ} (ha : 0 < a) [Nonempty X] (C D : R → X → ℕ) : 0 < mlPost a C D := by
  have hV : (0 : ℝ) < Fintype.card X := Nat.cast_pos.mpr Fintype.card_pos
  apply prod_pos
  intro r _
  apply div_pos
  · exact prod_pos fun x _ => rise_pos (by positivity) _
  · exact rise_pos (by positivity) _

theorem ml_pos {a : ℝ} (ha : 0 < a) [Nonempty X] (D : R → X → ℕ) : 0 < ml a D := mlPost_pos ha 0 D

/-- **Chain rule** `ML(C + D) = ML(C) · ML(D | C)`. -/
theorem chain_rule (a : ℝ) (C D : R → X → ℕ) : ml a (C + D) = ml a C * mlPost a C D := by
  unfold ml mlPost
  rw [← prod_mul_distrib]
  apply prod_congr rfl
  intro r _
  have hs : rowSum ((C + D) r) = rowSum (C r) + rowSum (D r) := by
    simp [rowSum, sum_add_distrib]
  simp only [Pi.zero_apply, Nat.cast_zero, add_zero, rowSum, Finset.sum_const_zero] at hs ⊢
  rw [hs, rise_add]
  have hx : ∀ x, rise a ((C + D) r x) = rise a (C r x) * rise (a + C r x) (D r x) := fun x => by
    simp only [Pi.add_apply]
    exact rise_add a _ _
  simp_rw [hx, prod_mul_distrib]
  rw [mul_div_mul_comm]

/-- The Bayes factor is "how much better era `C`'s posterior predicts `D` than a fresh era does". -/
theorem bf_eq {a : ℝ} (ha : 0 < a) [Nonempty X] (C D : R → X → ℕ) : bf a C D = mlPost a C D / ml a D := by
  unfold bf
  rw [chain_rule]
  exact mul_div_mul_left _ _ (ml_pos ha C).ne'

theorem bf_symm (a : ℝ) (C D : R → X → ℕ) : bf a C D = bf a D C := by
  unfold bf
  rw [add_comm C D, mul_comm (ml a C)]

/-! ### Prequential form (what the address step computes) -/

/-- Dirichlet predictive of symbol `x` in context `r`. -/
noncomputable def pred (a : ℝ) (C : R → X → ℕ) (r : R) (x : X) : ℝ :=
  (C r x + a) / (rowSum (C r) + Fintype.card X * a)

/-- Add one transition `(r, x)`. -/
def bump (C : R → X → ℕ) (r : R) (x : X) : R → X → ℕ :=
  fun r' x' => C r' x' + if r' = r ∧ x' = x then 1 else 0

/-- Prequential product along a token list, updating the era after each token (exactly the address loop). -/
noncomputable def preq (a : ℝ) : (R → X → ℕ) → List (R × X) → ℝ
  | _, [] => 1
  | C, z :: zs => pred a C z.1 z.2 * preq a (bump C z.1 z.2) zs

/-- Transition counts of a token list. -/
def counts : List (R × X) → R → X → ℕ
  | [] => 0
  | z :: zs => bump (counts zs) z.1 z.2

theorem pred_sum_one {a : ℝ} (ha : 0 < a) [Nonempty X] (C : R → X → ℕ) (r : R) : ∑ x, pred a C r x = 1 := by
  have hV : (0 : ℝ) < Fintype.card X := Nat.cast_pos.mpr Fintype.card_pos
  have hd : (0 : ℝ) < rowSum (C r) + Fintype.card X * a := by positivity
  unfold pred
  rw [← sum_div, div_eq_one_iff_eq hd.ne', sum_add_distrib]
  simp [rowSum, Finset.card_univ]

theorem rowSum_bump_self (C : R → X → ℕ) (r : R) (x : X) : rowSum (bump C r x r) = rowSum (C r) + 1 := by
  simp [rowSum, bump, sum_add_distrib]

theorem rowSum_bump_ne (C : R → X → ℕ) {r r' : R} (h : r' ≠ r) (x : X) : rowSum (bump C r x r') = rowSum (C r') := by
  simp [rowSum, bump, h]

/-- One step of the prequential recursion, in closed form. -/
theorem mlPost_bump (a : ℝ) (C D : R → X → ℕ) (r : R) (x : X) :
    mlPost a C (bump D r x) = pred a C r x * mlPost a (bump C r x) D := by
  unfold mlPost
  have hrow : ∀ r', (∏ x', rise (a + C r' x') (bump D r x r' x')) /
      rise (Fintype.card X * a + rowSum (C r')) (rowSum (bump D r x r')) =
      (if r' = r then pred a C r x else 1) *
      ((∏ x', rise (a + bump C r x r' x') (D r' x')) /
        rise (Fintype.card X * a + rowSum (bump C r x r')) (rowSum (D r'))) := by
    intro r'
    by_cases hr : r' = r
    · subst hr
      simp only [if_true]
      rw [rowSum_bump_self, rowSum_bump_self, rise_succ']
      have hnum : ∀ x', rise (a + C r' x') (bump D r' x r' x') =
          (if x' = x then a + C r' x else 1) * rise (a + bump C r' x r' x') (D r' x') := by
        intro x'
        by_cases hx : x' = x
        · subst hx
          simp only [bump, and_self, if_true]
          rw [rise_succ']
          push_cast
          ring_nf
        · simp [bump, hx]
      simp_rw [hnum, prod_mul_distrib, prod_ite_eq', mem_univ, if_true]
      have hden : rise ((Fintype.card X : ℝ) * a + ((rowSum (C r') + 1 : ℕ) : ℝ)) (rowSum (D r')) =
          rise ((Fintype.card X : ℝ) * a + (rowSum (C r') : ℝ) + 1) (rowSum (D r')) := by
        congr 1; push_cast; ring
      rw [hden, mul_div_mul_comm]
      unfold pred
      congr 1
      ring
    · simp only [hr, if_false, one_mul]
      rw [rowSum_bump_ne _ hr, rowSum_bump_ne _ hr]
      congr 1
      apply prod_congr rfl
      intro x' _
      simp [bump, hr]
  simp_rw [hrow, prod_mul_distrib, prod_ite_eq', mem_univ, if_true]

/-- **Prequential = closed form**: the address loop's product depends on the token counts only. -/
theorem preq_eq {a : ℝ} (ha : 0 < a) [Nonempty X] : ∀ (zs : List (R × X)) (C : R → X → ℕ),
    preq a C zs = mlPost a C (counts zs)
  | [], C => by
    simp only [preq, counts, mlPost]
    symm
    apply prod_eq_one
    intro r _
    simp [rowSum]
  | z :: zs, C => by
    simp only [preq, counts]
    rw [mlPost_bump, preq_eq ha zs]

/-- **L39 (exact).** The merge test on `(C, D)` holds iff era `C` beats a fresh era on `D`'s tokens in the address
step (any order of the tokens). -/
theorem merge_iff_address {a : ℝ} (ha : 0 < a) [Nonempty X] (C : R → X → ℕ) (zs : List (R × X)) :
    1 < bf a C (counts zs) ↔ preq a 0 zs < preq a C zs := by
  rw [bf_eq ha, preq_eq ha, preq_eq ha, one_lt_div (ml_pos ha _)]
  rfl

/-- Log form, as the code tests it. -/
theorem merge_iff_address_log {a : ℝ} (ha : 0 < a) [Nonempty X] (C : R → X → ℕ) (zs : List (R × X)) :
    0 < Real.log (ml a (C + counts zs)) - Real.log (ml a C) - Real.log (ml a (counts zs)) ↔
      preq a 0 zs < preq a C zs := by
  rw [← merge_iff_address ha]
  have h1 := ml_pos ha (C + counts zs)
  have h2 := ml_pos ha C
  have h3 := ml_pos ha (counts zs)
  unfold bf
  rw [← Real.log_div h1.ne' h2.ne', ← Real.log_div (div_pos h1 h2).ne' h3.ne', div_div,
    Real.log_pos_iff (div_pos h1 (mul_pos h2 h3))]

/-- **Both directions agree**: `C` addressing `D`'s data beats fresh ⇔ `D` addressing `C`'s data beats fresh. -/
theorem address_symm {a : ℝ} (ha : 0 < a) [Nonempty X] (ys zs : List (R × X)) :
    preq a 0 zs < preq a (counts ys) zs ↔ preq a 0 ys < preq a (counts zs) ys := by
  rw [← merge_iff_address ha, ← merge_iff_address ha, bf_symm]

/-! ### Error control: the Bayes factor is an e-value -/

/-- Under the "separate eras" model `q₀`, the probability that the evidence for merging, `q₁/q₀`, reaches `1/δ` is at
most `δ` (Markov). `q₁` is the merged model. Stated without division: the event is `q₀ ≤ δ q₁`. -/
theorem false_merge_le {Ω : Type*} [Fintype Ω] (q₀ q₁ : Ω → ℝ) (_h0 : ∀ ω, 0 ≤ q₀ ω) (h1 : ∀ ω, 0 ≤ q₁ ω)
    (hs1 : ∑ ω, q₁ ω ≤ 1) {δ : ℝ} (hδ : 0 < δ) :
    ∑ ω in univ.filter (fun ω => q₀ ω ≤ δ * q₁ ω), q₀ ω ≤ δ := by
  calc ∑ ω in univ.filter (fun ω => q₀ ω ≤ δ * q₁ ω), q₀ ω
      ≤ ∑ ω in univ.filter (fun ω => q₀ ω ≤ δ * q₁ ω), δ * q₁ ω := sum_le_sum fun ω hω => (mem_filter.mp hω).2
    _ ≤ ∑ ω, δ * q₁ ω := sum_le_sum_of_subset_of_nonneg (filter_subset _ _)
        (fun ω _ _ => mul_nonneg hδ.le (h1 ω))
    _ = δ * ∑ ω, q₁ ω := by rw [mul_sum]
    _ ≤ δ * 1 := mul_le_mul_of_nonneg_left hs1 hδ.le
    _ = δ := mul_one δ

/-- Symmetric: under the merged model, the chance of evidence `≥ 1/δ` for splitting is at most `δ`. -/
theorem false_split_le {Ω : Type*} [Fintype Ω] (q₀ q₁ : Ω → ℝ) (h0 : ∀ ω, 0 ≤ q₀ ω) (h1 : ∀ ω, 0 ≤ q₁ ω)
    (hs0 : ∑ ω, q₀ ω ≤ 1) {δ : ℝ} (hδ : 0 < δ) :
    ∑ ω in univ.filter (fun ω => q₁ ω ≤ δ * q₀ ω), q₁ ω ≤ δ :=
  false_merge_le q₁ q₀ h1 h0 hs0 hδ

/-- Threshold `δ = 1` (the code's `BF > 1`) gives only the trivial bound: the e-value statement is vacuous there. -/
theorem threshold_one_vacuous {Ω : Type*} [Fintype Ω] (q₀ q₁ : Ω → ℝ) (h0 : ∀ ω, 0 ≤ q₀ ω)
    (hs0 : ∑ ω, q₀ ω = 1) : ∑ ω in univ.filter (fun ω => q₀ ω ≤ 1 * q₁ ω), q₀ ω ≤ 1 := by
  rw [← hs0]
  exact sum_le_sum_of_subset_of_nonneg (filter_subset _ _) (fun ω _ _ => h0 ω)

/-! ### Sequential predictors are probability models -/

/-- Joint mass of a continuation `xs` after history `h` under a sequential predictor `q`. -/
def seqMass {Y : Type*} (q : List Y → Y → ℝ) : List Y → List Y → ℝ
  | _, [] => 1
  | h, y :: ys => q h y * seqMass q (h ++ [y]) ys

/-- **Total mass one** over all sequences of every length, for any predictor whose rows sum to one. -/
theorem seq_mass_one {Y : Type*} [Fintype Y] (q : List Y → Y → ℝ) (hq : ∀ h, ∑ y, q h y = 1) :
    ∀ (n : ℕ) (h : List Y), ∑ ys : Fin n → Y, seqMass q h (List.ofFn ys) = 1
  | 0, h => by simp [seqMass]
  | n + 1, h => by
    rw [← (Equiv.piFinSucc n Y).symm.sum_comp, Fintype.sum_prod_type]
    simp only [Equiv.piFinSucc_symm_apply, Fin.insertNth_zero', List.ofFn_succ, Fin.cons_zero, Fin.cons_succ,
      seqMass]
    simp_rw [← mul_sum, seq_mass_one q hq n, mul_one, hq]

end L39
end FCPA

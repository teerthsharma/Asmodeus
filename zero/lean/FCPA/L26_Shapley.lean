import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Data.Fintype.Perm
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp

/-!
L26 Retire / merge by Shapley (spec §3.10, Chase form). `v : Finset ι → ℝ` is the validation gain of a set of eras
(with `v ∅ = 0`). The Shapley value is the average, over all `n!` orderings `σ`, of the marginal gain of `i` when it
joins the eras that precede it. Efficiency: `Σ_i φ_i = v(univ) - v(∅) = v(univ)`.
Leave-one-out is NOT efficient: two duplicate eras each get 0 (`loo_not_efficient`), so both would be retired.
-/
open BigOperators

namespace FCPA

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The set of items in positions `< m` of the ordering `σ`. -/
def pre (σ : ι ≃ Fin (Fintype.card ι)) (m : ℕ) : Finset ι :=
  Finset.univ.filter (fun j => ((σ j : Fin (Fintype.card ι)) : ℕ) < m)

theorem pre_zero (σ : ι ≃ Fin (Fintype.card ι)) : pre σ 0 = ∅ := by
  ext j; simp [pre]

theorem pre_full (σ : ι ≃ Fin (Fintype.card ι)) : pre σ (Fintype.card ι) = Finset.univ := by
  ext j; simp [pre]

/-- Marginal gains along one ordering telescope to `v(univ) - v(∅)`. -/
theorem marginal_telescope (v : Finset ι → ℝ) (σ : ι ≃ Fin (Fintype.card ι)) :
    ∑ i, (v (pre σ ((σ i : ℕ) + 1)) - v (pre σ (σ i : ℕ))) = v Finset.univ - v ∅ := by
  rw [Equiv.sum_comp σ (fun k : Fin (Fintype.card ι) => v (pre σ ((k : ℕ) + 1)) - v (pre σ (k : ℕ)))]
  rw [Fin.sum_univ_eq_sum_range (fun k : ℕ => v (pre σ (k + 1)) - v (pre σ k)) (Fintype.card ι)]
  rw [Finset.sum_range_sub (fun k : ℕ => v (pre σ k)) (Fintype.card ι), pre_full, pre_zero]

/-- Shapley value of `i`: average marginal gain over all orderings. -/
noncomputable def shapley (v : Finset ι → ℝ) (i : ι) : ℝ :=
  ((Fintype.card ι).factorial : ℝ)⁻¹ *
    ∑ σ : ι ≃ Fin (Fintype.card ι), (v (pre σ ((σ i : ℕ) + 1)) - v (pre σ (σ i : ℕ)))

/-- **Efficiency**: the Shapley values sum to the total gain `v(univ) - v(∅)`. -/
theorem shapley_efficiency (v : Finset ι → ℝ) :
    ∑ i, shapley v i = v Finset.univ - v ∅ := by
  unfold shapley
  rw [← Finset.mul_sum, Finset.sum_comm]
  simp_rw [marginal_telescope]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_equiv (Fintype.equivFin ι),
    nsmul_eq_mul]
  have : ((Fintype.card ι).factorial : ℝ) ≠ 0 := by positivity
  field_simp
  ring

theorem shapley_efficiency_zero (v : Finset ι → ℝ) (hv : v ∅ = 0) :
    ∑ i, shapley v i = v Finset.univ := by
  rw [shapley_efficiency, hv, sub_zero]

/-- Leave-one-out value. -/
def loo (v : Finset ι → ℝ) (i : ι) : ℝ := v Finset.univ - v (Finset.univ.erase i)

/-- Two duplicate eras: `v S = 1` as soon as `S` is nonempty. Each leave-one-out value is `0`, the total is `0 ≠ v(univ) = 1`,
so leave-one-out is not efficient (and would retire both duplicates). -/
theorem loo_not_efficient :
    ∃ v : Finset Bool → ℝ, v ∅ = 0 ∧ (∑ i, loo v i) ≠ v Finset.univ := by
  refine ⟨fun S => if S = ∅ then 0 else 1, by simp, ?_⟩
  have h1 : ({true, false} : Finset Bool).erase true = {false} := by decide
  have h2 : ({true, false} : Finset Bool).erase false = {true} := by decide
  have h3 : ({true, false} : Finset Bool) ≠ ∅ := by decide
  simp [loo, Fintype.sum_bool, h1, h2, h3]
  norm_num

end FCPA

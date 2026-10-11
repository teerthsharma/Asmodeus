import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
L14′ (Amendment A, tie-break by HMM forward filter over era classes): the filter update
`b'_j = (∑_i b_i P_ij) ℓ_j / Z`, `Z = ∑_j (∑_i b_i P_ij) ℓ_j`, is a probability vector whenever `b` is one, `P` is row-stochastic,
`ℓ ≥ 0` and `Z > 0`. (Chase: L14′ needed its own normalisation lemma.)
-/
open BigOperators

namespace FCPA

theorem hmm_filter_prob {K : ℕ} (b ℓ : Fin K → ℝ) (P : Fin K → Fin K → ℝ)
    (hb : ∀ i, 0 ≤ b i) (hP : ∀ i j, 0 ≤ P i j)
    (hl : ∀ j, 0 ≤ ℓ j) (hZ : 0 < ∑ j, (∑ i, b i * P i j) * ℓ j) :
    (∀ j, 0 ≤ (∑ i, b i * P i j) * ℓ j / ∑ j', (∑ i, b i * P i j') * ℓ j') ∧
    ∑ j, (∑ i, b i * P i j) * ℓ j / (∑ j', (∑ i, b i * P i j') * ℓ j') = 1 := by
  constructor
  · intro j
    apply div_nonneg _ hZ.le
    exact mul_nonneg (Finset.sum_nonneg (fun i _ => mul_nonneg (hb i) (hP i j))) (hl j)
  · rw [← Finset.sum_div, div_self hZ.ne']

theorem hmm_predict_prob {K : ℕ} (b : Fin K → ℝ) (P : Fin K → Fin K → ℝ)
    (hb1 : ∑ i, b i = 1) (hP1 : ∀ i, ∑ j, P i j = 1) :
    ∑ j, ∑ i, b i * P i j = 1 := by
  rw [Finset.sum_comm]
  simp_rw [← Finset.mul_sum, hP1, mul_one, hb1]

/-- **Bayes**: the filter update IS the posterior `P(state j | observation)` of the joint model
`q(i,j) = b_i · P_ij · ℓ_j` (previous state `i`, next state `j`, observation likelihood `ℓ_j`):
`P(j | x) = (Σ_i q(i,j)) / (Σ_{j'} Σ_i q(i,j')) = b'_j`. -/
theorem hmm_filter_is_posterior {K : ℕ} (b ℓ : Fin K → ℝ) (P : Fin K → Fin K → ℝ) (j : Fin K) :
    (∑ i, b i * P i j * ℓ j) / (∑ j', ∑ i, b i * P i j' * ℓ j')
      = (∑ i, b i * P i j) * ℓ j / ∑ j', (∑ i, b i * P i j') * ℓ j' := by
  simp_rw [← Finset.sum_mul]

end FCPA

import FCPA.L01_BornExactness
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
E3 (Amendment A): Born probabilities are 2-Lipschitz (ℓ¹ vs ℓ²): for unit `ψ, ψ'` and a Parseval frame `V`,
`∑_k |p_k(ψ) - p_k(ψ')| ≤ 2 ‖ψ - ψ'‖` where `p_k = |(Vψ)_k|²` (gives the radius `R` of L5).
-/
open Matrix BigOperators

namespace FCPA

theorem born_lipschitz {K N : ℕ} (V : Matrix (Fin (K + 1)) (Fin N) ℂ) (hV : Vᴴ * V = 1)
    (ψ ψ' : Fin N → ℂ) (hψ : ∑ i, Complex.normSq (ψ i) = 1) (hψ' : ∑ i, Complex.normSq (ψ' i) = 1) :
    ∑ k, |Complex.normSq ((V *ᵥ ψ) k) - Complex.normSq ((V *ᵥ ψ') k)|
      ≤ 2 * Real.sqrt (∑ i, Complex.normSq (ψ i - ψ' i)) := by
  set a : Fin (K + 1) → ℂ := V *ᵥ ψ with ha
  set b : Fin (K + 1) → ℂ := V *ᵥ ψ' with hb
  set S : ℝ := ∑ i, Complex.normSq (ψ i - ψ' i) with hS
  have hS0 : 0 ≤ S := Finset.sum_nonneg (fun i _ => Complex.normSq_nonneg _)
  set u : Fin (K + 1) → ℝ := fun k => ‖a k - b k‖ with hu
  set v : Fin (K + 1) → ℝ := fun k => ‖a k‖ + ‖b k‖ with hv
  have hpt : ∀ k, |Complex.normSq (a k) - Complex.normSq (b k)| ≤ u k * v k := by
    intro k
    rw [Complex.normSq_eq_norm_sq, Complex.normSq_eq_norm_sq]
    have e : ‖a k‖ ^ 2 - ‖b k‖ ^ 2 = (‖a k‖ - ‖b k‖) * (‖a k‖ + ‖b k‖) := by ring
    rw [e, abs_mul, abs_of_nonneg (by positivity : 0 ≤ ‖a k‖ + ‖b k‖)]
    exact mul_le_mul_of_nonneg_right (abs_norm_sub_norm_le _ _) (by positivity)
  have hu2 : ∑ k, u k ^ 2 = S := by
    have e1 : ∀ k, u k ^ 2 = Complex.normSq (a k - b k) := fun k => by
      rw [hu]; simp only; rw [Complex.normSq_eq_norm_sq]
    simp_rw [e1]
    have : ∀ k, a k - b k = (V *ᵥ (ψ - ψ')) k := fun k => by
      rw [mulVec_sub]; rfl
    simp_rw [this]
    exact parseval V hV (ψ - ψ')
  have hv2 : ∑ k, v k ^ 2 ≤ 4 := by
    have : ∀ k, v k ^ 2 ≤ 2 * Complex.normSq (a k) + 2 * Complex.normSq (b k) := fun k => by
      rw [hv]; simp only; rw [Complex.normSq_eq_norm_sq, Complex.normSq_eq_norm_sq]
      nlinarith [sq_nonneg (‖a k‖ - ‖b k‖)]
    calc ∑ k, v k ^ 2 ≤ ∑ k, (2 * Complex.normSq (a k) + 2 * Complex.normSq (b k)) :=
          Finset.sum_le_sum (fun k _ => this k)
      _ = 2 * ∑ k, Complex.normSq (a k) + 2 * ∑ k, Complex.normSq (b k) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
      _ = 4 := by rw [ha, hb, parseval V hV ψ, parseval V hV ψ', hψ, hψ']; norm_num
  have hCS := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ u v
  rw [hu2] at hCS
  have hX : ∑ k, |Complex.normSq (a k) - Complex.normSq (b k)| ≤ ∑ k, u k * v k :=
    Finset.sum_le_sum (fun k _ => hpt k)
  have hsq : (∑ k, u k * v k) ^ 2 ≤ 4 * S := by
    calc (∑ k, u k * v k) ^ 2 ≤ S * ∑ k, v k ^ 2 := hCS
      _ ≤ S * 4 := mul_le_mul_of_nonneg_left hv2 hS0
      _ = 4 * S := by ring
  have hs0 := Real.sqrt_nonneg S
  have hss := Real.sq_sqrt hS0
  have hfin : ∑ k, u k * v k ≤ 2 * Real.sqrt S := by
    by_contra hlt
    push_neg at hlt
    nlinarith [mul_pos (by linarith : 0 < ∑ k, u k * v k - 2 * Real.sqrt S)
      (by linarith : 0 < ∑ k, u k * v k + 2 * Real.sqrt S)]
  linarith

end FCPA

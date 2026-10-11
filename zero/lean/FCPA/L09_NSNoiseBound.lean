import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
L9 NS noise bound (spec §3.3, Chase form): error in the residual only. `R = ν • L + e`, `ν̂ = ⟪R,L⟫/⟪L,L⟫` ⇒ `|ν̂ - ν| ≤ ‖e‖/‖L‖`.
E2′ (Amendment B): the demeaned within-slab estimator `ν̂ = ∑_t ⟪R̃_t, Δ̃_t⟫ / ∑_t ‖Δ̃_t‖²` satisfies
`|ν̂ - ν| ≤ √(∑‖e_t‖²) / √(∑‖Δ̃_t‖²)` (Cauchy–Schwarz); if `∑‖Δ̃_t‖² = 0` the estimator REFUSEs (hypothesis `hD`).
-/
open BigOperators

namespace FCPA

theorem ns_noise {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (R L e : E) (ν : ℝ)
    (hL : L ≠ 0) (h : R = ν • L + e) :
    |(inner R L : ℝ) / (inner L L : ℝ) - ν| ≤ ‖e‖ / ‖L‖ := by
  have hLn : 0 < ‖L‖ := norm_pos_iff.mpr hL
  have hLL : (inner L L : ℝ) = ‖L‖ ^ 2 := real_inner_self_eq_norm_sq L
  have e1 : (inner R L : ℝ) / (inner L L : ℝ) - ν = (inner e L : ℝ) / ‖L‖ ^ 2 := by
    rw [h, inner_add_left, real_inner_smul_left, hLL]
    field_simp
    ring
  rw [e1, abs_div, abs_of_pos (by positivity : (0 : ℝ) < ‖L‖ ^ 2),
    div_le_div_iff (by positivity) hLn]
  have := abs_real_inner_le_norm e L
  nlinarith [mul_le_mul_of_nonneg_right this hLn.le]

theorem ns_noise_slab {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {n : ℕ}
    (R D e : Fin n → E) (ν : ℝ) (hD : 0 < ∑ t, ‖D t‖ ^ 2) (h : ∀ t, R t = ν • D t + e t) :
    |(∑ t, (inner (R t) (D t) : ℝ)) / (∑ t, ‖D t‖ ^ 2) - ν|
      ≤ Real.sqrt (∑ t, ‖e t‖ ^ 2) / Real.sqrt (∑ t, ‖D t‖ ^ 2) := by
  set S : ℝ := ∑ t, ‖D t‖ ^ 2 with hS
  set E2 : ℝ := ∑ t, ‖e t‖ ^ 2 with hE2
  have hnum : (∑ t, (inner (R t) (D t) : ℝ)) = ν * S + ∑ t, (inner (e t) (D t) : ℝ) := by
    have : ∀ t, (inner (R t) (D t) : ℝ) = ν * ‖D t‖ ^ 2 + inner (e t) (D t) := by
      intro t
      rw [h t, inner_add_left, real_inner_smul_left, real_inner_self_eq_norm_sq]
    simp_rw [this]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum]
  have key : (∑ t, (inner (R t) (D t) : ℝ)) / S - ν = (∑ t, (inner (e t) (D t) : ℝ)) / S := by
    rw [hnum]; field_simp; ring
  have hE0 : 0 ≤ E2 := Finset.sum_nonneg (fun t _ => by positivity)
  have hCS : |∑ t, (inner (e t) (D t) : ℝ)| ≤ Real.sqrt E2 * Real.sqrt S := by
    calc |∑ t, (inner (e t) (D t) : ℝ)| ≤ ∑ t, |(inner (e t) (D t) : ℝ)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ t, ‖e t‖ * ‖D t‖ := Finset.sum_le_sum (fun t _ => abs_real_inner_le_norm _ _)
      _ ≤ Real.sqrt (E2 * S) := by
          refine le_trans (le_abs_self _) (Real.abs_le_sqrt ?_)
          have := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun t => ‖e t‖) (fun t => ‖D t‖)
          simpa [hS, hE2] using this
      _ = Real.sqrt E2 * Real.sqrt S := Real.sqrt_mul hE0 S
  have hsq : 0 < Real.sqrt S := Real.sqrt_pos.mpr hD
  have hss : Real.sqrt S * Real.sqrt S = S := Real.mul_self_sqrt hD.le
  rw [key, abs_div, abs_of_pos hD, div_le_div_iff hD hsq]
  calc |∑ t, (inner (e t) (D t) : ℝ)| * Real.sqrt S ≤ Real.sqrt E2 * Real.sqrt S * Real.sqrt S :=
        mul_le_mul_of_nonneg_right hCS hsq.le
    _ = Real.sqrt E2 * S := by rw [mul_assoc, hss]

/-! ### E2′ with the demeaning step (Amendment B) -/

/-- Time mean of a finite sequence. -/
noncomputable def meanVec {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {n : ℕ}
    (v : Fin n → E) : E := ((n : ℝ)⁻¹) • ∑ t, v t

theorem meanVec_affine {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {n : ℕ}
    (hn : 0 < n) (D e : Fin n → E) (f : E) (ν : ℝ) :
    meanVec (fun t => ν • D t + f + e t) = ν • meanVec D + f + meanVec e := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  unfold meanVec
  simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    ← Finset.smul_sum, smul_add, nsmul_eq_smul_cast ℝ, smul_smul, smul_comm ν]
  rw [inv_mul_cancel hn', one_smul]

/-- **E2′ with demeaning.** Observations `R_t = ν Δ_t + f + e_t` with a time-constant forcing `f`. Demean `R` and `Δ`;
the estimator `ν̂ = ∑⟨R̃_t, Δ̃_t⟩ / ∑‖Δ̃_t‖²` satisfies `|ν̂ - ν| ≤ ‖ẽ‖_F / ‖Δ̃‖_F` (`ẽ` is the demeaned error;
`f` drops out). If `∑‖Δ̃_t‖² = 0` the estimator REFUSEs (hypothesis `hD`). -/
theorem ns_noise_demeaned {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {n : ℕ}
    (hn : 0 < n) (R D e : Fin n → E) (f : E) (ν : ℝ)
    (hD : 0 < ∑ t, ‖D t - meanVec D‖ ^ 2) (h : ∀ t, R t = ν • D t + f + e t) :
    |(∑ t, (inner (R t - meanVec R) (D t - meanVec D) : ℝ)) / (∑ t, ‖D t - meanVec D‖ ^ 2) - ν|
      ≤ Real.sqrt (∑ t, ‖e t - meanVec e‖ ^ 2) / Real.sqrt (∑ t, ‖D t - meanVec D‖ ^ 2) := by
  have hR : R = fun t => ν • D t + f + e t := funext h
  have hm : meanVec R = ν • meanVec D + f + meanVec e := by rw [hR]; exact meanVec_affine hn D e f ν
  refine ns_noise_slab (fun t => R t - meanVec R) (fun t => D t - meanVec D)
    (fun t => e t - meanVec e) ν hD (fun t => ?_)
  simp only
  rw [h t, hm, smul_sub]
  abel

/-- Forcing estimate `f̂ = mean R - ν̂ · mean Δ` has error `f̂ - f = mean e - (ν̂ - ν) · mean Δ`. -/
theorem forcing_error {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {n : ℕ}
    (hn : 0 < n) (R D e : Fin n → E) (f : E) (ν νhat : ℝ) (h : ∀ t, R t = ν • D t + f + e t) :
    (meanVec R - νhat • meanVec D) - f = meanVec e - (νhat - ν) • meanVec D := by
  have hm : meanVec R = ν • meanVec D + f + meanVec e := by
    rw [show R = fun t => ν • D t + f + e t from funext h]; exact meanVec_affine hn D e f ν
  rw [hm, sub_smul]
  abel

end FCPA

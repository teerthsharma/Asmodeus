import Mathlib.Analysis.NormedSpace.Exponential
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Tactic.NoncommRing
import Mathlib.Tactic.Linarith

/-!
L16 holonomy, part 1: second-order remainder of `exp` in a complete normed algebra and the three-factor expansion.
(No Baker–Campbell–Hausdorff lemma exists in Mathlib; everything is derived from the exponential series.)

* `exp_remainder_le`: `‖X‖ ≤ 1 ⇒ ‖exp X - (1 + X + ½X²)‖ ≤ ‖X‖³`.
* `triple_expansion`: for `‖Xᵢ‖ ≤ δ`, `‖wᵢ‖ ≤ 2δ²`, `δ ≤ 1`:
  `‖(1+X₁+w₁)(1+X₂+w₂)(1+X₃+w₃) - (1 + ΣXᵢ + Σwᵢ + Σ_{i<j}XᵢXⱼ)‖ ≤ 51 δ³`.
-/
open NormedSpace BigOperators

namespace FCPA

variable {𝔸 : Type*} [NormedRing 𝔸] [NormedAlgebra ℝ 𝔸] [CompleteSpace 𝔸]

theorem factorial_ge (i : ℕ) : (6 : ℝ) * 2 ^ i ≤ ((i + 3).factorial : ℝ) := by
  induction i with
  | zero => norm_num [Nat.factorial]
  | succ i ih =>
    have h : ((i + 1 + 3).factorial : ℝ) = ((i + 4 : ℕ) : ℝ) * ((i + 3).factorial : ℝ) := by
      rw [show i + 1 + 3 = (i + 3) + 1 by ring, Nat.factorial_succ]; push_cast; ring
    rw [h]
    have h4 : (2 : ℝ) ≤ ((i + 4 : ℕ) : ℝ) := by push_cast; linarith [(Nat.cast_nonneg i : (0:ℝ) ≤ i)]
    calc (6 : ℝ) * 2 ^ (i + 1) = 2 * (6 * 2 ^ i) := by ring
      _ ≤ ((i + 4 : ℕ) : ℝ) * ((i + 3).factorial : ℝ) :=
          mul_le_mul h4 ih (by positivity) (by linarith)

theorem exp_remainder_le (X : 𝔸) (hX : ‖X‖ ≤ 1) :
    ‖exp ℝ X - (1 + X + (1 / 2 : ℝ) • X ^ 2)‖ ≤ ‖X‖ ^ 3 := by
  have hs : Summable (fun n : ℕ => ((n.factorial : ℝ)⁻¹) • X ^ n) := expSeries_summable' (𝕂 := ℝ) X
  have hsum : HasSum (fun n : ℕ => ((n.factorial : ℝ)⁻¹) • X ^ n) (exp ℝ X) :=
    exp_series_hasSum_exp' (𝕂 := ℝ) X
  have h1 := sum_add_tsum_nat_add 3 hs
  have h2 : ∑' n, ((n.factorial : ℝ)⁻¹) • X ^ n = exp ℝ X := hsum.tsum_eq
  rw [h2] at h1
  have h3 : ∑ i in Finset.range 3, ((i.factorial : ℝ)⁻¹) • X ^ i = 1 + X + (1 / 2 : ℝ) • X ^ 2 := by
    simp [Finset.sum_range_succ, Nat.factorial]
  rw [h3] at h1
  have h4 : exp ℝ X - (1 + X + (1 / 2 : ℝ) • X ^ 2) = ∑' i, ((((i + 3).factorial : ℝ)⁻¹) • X ^ (i + 3)) := by
    rw [← h1]; abel
  rw [h4]
  have hg : HasSum (fun i : ℕ => ‖X‖ ^ 3 * (1 / 6) * ((1 : ℝ) / 2) ^ i) (‖X‖ ^ 3 * (1 / 6) * 2) :=
    hasSum_geometric_two.mul_left (‖X‖ ^ 3 * (1 / 6))
  refine le_trans (tsum_of_norm_bounded hg (fun i => ?_)) (by nlinarith [pow_nonneg (norm_nonneg X) 3])
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
  have hp : ‖X ^ (i + 3)‖ ≤ ‖X‖ ^ (i + 3) := norm_pow_le' X (by omega)
  have hpi : ‖X‖ ^ i ≤ 1 := pow_le_one i (norm_nonneg X) hX
  have hf := factorial_ge i
  have hfpos : (0 : ℝ) < ((i + 3).factorial : ℝ) := by positivity
  calc ((((i + 3).factorial : ℝ))⁻¹) * ‖X ^ (i + 3)‖
      ≤ ((((i + 3).factorial : ℝ))⁻¹) * (‖X‖ ^ 3 * ‖X‖ ^ i) := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        calc ‖X ^ (i + 3)‖ ≤ ‖X‖ ^ (i + 3) := hp
          _ = ‖X‖ ^ 3 * ‖X‖ ^ i := by ring
    _ ≤ ((((i + 3).factorial : ℝ))⁻¹) * (‖X‖ ^ 3) := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        nlinarith [pow_nonneg (norm_nonneg X) 3]
    _ ≤ (1 / (6 * 2 ^ i)) * ‖X‖ ^ 3 := by
        rw [inv_eq_one_div]
        exact mul_le_mul_of_nonneg_right (one_div_le_one_div_of_le (by positivity) hf) (by positivity)
    _ = ‖X‖ ^ 3 * (1 / 6) * ((1 : ℝ) / 2) ^ i := by
        rw [one_div_pow]; field_simp

theorem norm_mul_le_of {a b : 𝔸} {p q : ℝ} (ha : ‖a‖ ≤ p) (hb : ‖b‖ ≤ q) : ‖a * b‖ ≤ p * q :=
  (norm_mul_le a b).trans (mul_le_mul ha hb (norm_nonneg _) ((norm_nonneg _).trans ha))

theorem pair_bound {X Y w v : 𝔸} {δ : ℝ} (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1)
    (hX : ‖X‖ ≤ δ) (hY : ‖Y‖ ≤ δ) (hw : ‖w‖ ≤ 2 * δ ^ 2) (hv : ‖v‖ ≤ 2 * δ ^ 2) :
    ‖X * v + w * Y + w * v‖ ≤ 8 * δ ^ 3 := by
  have h1 := norm_mul_le_of hX hv
  have h2 := norm_mul_le_of hw hY
  have h3 := norm_mul_le_of hw hv
  have hd3 : δ ^ 4 ≤ δ ^ 3 := pow_le_pow_of_le_one hδ0 hδ (by norm_num)
  calc ‖X * v + w * Y + w * v‖ ≤ ‖X * v + w * Y‖ + ‖w * v‖ := norm_add_le _ _
    _ ≤ ‖X * v‖ + ‖w * Y‖ + ‖w * v‖ := by gcongr; exact norm_add_le _ _
    _ ≤ δ * (2 * δ ^ 2) + 2 * δ ^ 2 * δ + 2 * δ ^ 2 * (2 * δ ^ 2) := by linarith
    _ ≤ 8 * δ ^ 3 := by nlinarith [pow_nonneg hδ0 3]

theorem triple_expansion (X1 X2 X3 w1 w2 w3 : 𝔸) {δ : ℝ} (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1)
    (h1 : ‖X1‖ ≤ δ) (h2 : ‖X2‖ ≤ δ) (h3 : ‖X3‖ ≤ δ)
    (g1 : ‖w1‖ ≤ 2 * δ ^ 2) (g2 : ‖w2‖ ≤ 2 * δ ^ 2) (g3 : ‖w3‖ ≤ 2 * δ ^ 2) :
    ‖(1 + X1 + w1) * (1 + X2 + w2) * (1 + X3 + w3)
      - (1 + (X1 + X2 + X3) + (w1 + w2 + w3) + (X1 * X2 + X1 * X3 + X2 * X3))‖ ≤ 51 * δ ^ 3 := by
  have e : (1 + X1 + w1) * (1 + X2 + w2) * (1 + X3 + w3)
      - (1 + (X1 + X2 + X3) + (w1 + w2 + w3) + (X1 * X2 + X1 * X3 + X2 * X3))
      = (X1 * w2 + w1 * X2 + w1 * w2) + (X1 * w3 + w1 * X3 + w1 * w3)
        + (X2 * w3 + w2 * X3 + w2 * w3) + (X1 + w1) * (X2 + w2) * (X3 + w3) := by
    noncomm_ring
  rw [e]
  have p12 := pair_bound hδ0 hδ h1 h2 g1 g2
  have p13 := pair_bound hδ0 hδ h1 h3 g1 g3
  have p23 := pair_bound hδ0 hδ h2 h3 g2 g3
  have hd2 : δ ^ 2 ≤ δ := by nlinarith
  have u1 : ‖X1 + w1‖ ≤ 3 * δ := (norm_add_le _ _).trans (by nlinarith)
  have u2 : ‖X2 + w2‖ ≤ 3 * δ := (norm_add_le _ _).trans (by nlinarith)
  have u3 : ‖X3 + w3‖ ≤ 3 * δ := (norm_add_le _ _).trans (by nlinarith)
  have t3 : ‖(X1 + w1) * (X2 + w2) * (X3 + w3)‖ ≤ 27 * δ ^ 3 := by
    have := norm_mul_le_of (norm_mul_le_of u1 u2 |>.trans_eq' rfl) u3
    have e27 : 3 * δ * (3 * δ) * (3 * δ) = 27 * δ ^ 3 := by ring
    exact (norm_mul_le_of (a := (X1 + w1) * (X2 + w2)) (b := X3 + w3)
      (norm_mul_le_of u1 u2) u3).trans_eq e27
  calc _ ≤ ‖(X1 * w2 + w1 * X2 + w1 * w2) + (X1 * w3 + w1 * X3 + w1 * w3)
        + (X2 * w3 + w2 * X3 + w2 * w3)‖ + ‖(X1 + w1) * (X2 + w2) * (X3 + w3)‖ := norm_add_le _ _
    _ ≤ (‖(X1 * w2 + w1 * X2 + w1 * w2) + (X1 * w3 + w1 * X3 + w1 * w3)‖
          + ‖X2 * w3 + w2 * X3 + w2 * w3‖) + ‖(X1 + w1) * (X2 + w2) * (X3 + w3)‖ := by
        gcongr; exact norm_add_le _ _
    _ ≤ ((‖X1 * w2 + w1 * X2 + w1 * w2‖ + ‖X1 * w3 + w1 * X3 + w1 * w3‖)
          + ‖X2 * w3 + w2 * X3 + w2 * w3‖) + ‖(X1 + w1) * (X2 + w2) * (X3 + w3)‖ := by
        gcongr; exact norm_add_le _ _
    _ ≤ 51 * δ ^ 3 := by linarith

end FCPA

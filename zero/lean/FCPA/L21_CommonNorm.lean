import Mathlib.Data.Complex.Basic
import Mathlib.LinearAlgebra.Matrix.PosDef

/-!
L21 Common-norm product bound (spec §3.8, Chase constants form; demoted by Amendment A to the linear part).
`H` is sandwiched `c₁ ‖x‖² ≤ xᴴ H x ≤ c₂ ‖x‖²` (so `c₂/c₁ = κ(H)`), and every step is an `H`-contraction
`(A x)ᴴ H (A x) ≤ xᴴ H x`. Then `‖A_n ⋯ A_1 x‖² ≤ κ(H) ‖x‖²`, i.e. `‖A_n ⋯ A_1‖₂ ≤ √κ(H)`.
`hcontr_of_psd`: the Loewner form `Aᴴ H A ⪯ H` implies the pointwise `H`-contraction.
-/
open Matrix
open scoped ComplexOrder

namespace FCPA

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- `qf H x = Re (xᴴ H x)`. -/
noncomputable def qf (H : Matrix n n ℂ) (x : n → ℂ) : ℝ := (star x ⬝ᵥ H *ᵥ x).re

/-- Product `A_{k-1} ⋯ A_0`. -/
noncomputable def prodMat (A : ℕ → Matrix n n ℂ) : ℕ → Matrix n n ℂ
  | 0 => 1
  | k + 1 => A k * prodMat A k

theorem qf_prod_le (H : Matrix n n ℂ) (A : ℕ → Matrix n n ℂ)
    (hA : ∀ t x, qf H (A t *ᵥ x) ≤ qf H x) (k : ℕ) (x : n → ℂ) :
    qf H (prodMat A k *ᵥ x) ≤ qf H x := by
  induction k with
  | zero => simp [prodMat]
  | succ k ih =>
    have : prodMat A (k + 1) *ᵥ x = A k *ᵥ (prodMat A k *ᵥ x) := by
      rw [prodMat, ← mulVec_mulVec]
    rw [this]
    exact le_trans (hA k _) ih

theorem common_norm (H : Matrix n n ℂ) (c₁ c₂ : ℝ) (hc : 0 < c₁)
    (hH1 : ∀ x, c₁ * (star x ⬝ᵥ x).re ≤ qf H x) (hH2 : ∀ x, qf H x ≤ c₂ * (star x ⬝ᵥ x).re)
    (A : ℕ → Matrix n n ℂ) (hA : ∀ t x, qf H (A t *ᵥ x) ≤ qf H x) (k : ℕ) (x : n → ℂ) :
    (star (prodMat A k *ᵥ x) ⬝ᵥ (prodMat A k *ᵥ x)).re ≤ (c₂ / c₁) * (star x ⬝ᵥ x).re := by
  have h1 := hH1 (prodMat A k *ᵥ x)
  have h2 := qf_prod_le H A hA k x
  have h3 := hH2 x
  rw [div_mul_eq_mul_div, le_div_iff hc]
  linarith

theorem hcontr_of_psd (H A : Matrix n n ℂ) (h : (H - Aᴴ * H * A).PosSemidef) (x : n → ℂ) :
    qf H (A *ᵥ x) ≤ qf H x := by
  have h1 := h.2 x
  have e : star x ⬝ᵥ (H - Aᴴ * H * A) *ᵥ x
      = star x ⬝ᵥ H *ᵥ x - star (A *ᵥ x) ⬝ᵥ H *ᵥ (A *ᵥ x) := by
    rw [sub_mulVec, dotProduct_sub, star_mulVec, ← mulVec_mulVec, ← mulVec_mulVec,
      dotProduct_mulVec (star x) Aᴴ]
  rw [e] at h1
  have := (Complex.le_def.mp h1).1
  simp only [Complex.zero_re, Complex.sub_re] at this
  unfold qf
  linarith

end FCPA

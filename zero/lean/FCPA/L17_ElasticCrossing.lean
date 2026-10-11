import FCPA.L15_CrossingComposition
import Mathlib.LinearAlgebra.Matrix.DotProduct

/-!
L17 Elastic crossing (spec §3.5, Chase: REAL coefficients needed; with a complex coefficient, N=1, G = i, c = -i gives
`exp(1)` which is not unitary). `G_a` skew-Hermitian, `x` real ⇒ `S(x)` unitary, hence norm preserving.
-/
open Matrix BigOperators NormedSpace

namespace FCPA

variable {n : Type*} [Fintype n] [DecidableEq n] {d : ℕ}

theorem skew_sum (G : Fin d → Matrix n n ℂ) (hG : ∀ a, (G a)ᴴ = -G a) (x : Fin d → ℝ) :
    (∑ a, (x a : ℂ) • G a)ᴴ = -∑ a, (x a : ℂ) • G a := by
  rw [Matrix.conjTranspose_sum, ← Finset.sum_neg_distrib]
  apply Finset.sum_congr rfl
  intro a _
  rw [Matrix.conjTranspose_smul, hG a, smul_neg, Complex.star_def, Complex.conj_ofReal]

theorem crossing_unitary (G : Fin d → Matrix n n ℂ) (hG : ∀ a, (G a)ᴴ = -G a) (x : Fin d → ℝ) :
    (crossing G x)ᴴ * crossing G x = 1 := by
  unfold crossing
  have h := skew_sum G hG x
  generalize (∑ a, (x a : ℂ) • G a) = A at h ⊢
  rw [← Matrix.exp_conjTranspose ℂ A, h,
    ← Matrix.exp_add_of_commute ℂ _ _ ((Commute.refl A).neg_left)]
  simp

theorem crossing_norm (G : Fin d → Matrix n n ℂ) (hG : ∀ a, (G a)ᴴ = -G a) (x : Fin d → ℝ)
    (ψ : n → ℂ) :
    star (crossing G x *ᵥ ψ) ⬝ᵥ (crossing G x *ᵥ ψ) = star ψ ⬝ᵥ ψ := by
  rw [star_mulVec, ← dotProduct_mulVec, mulVec_mulVec, crossing_unitary G hG x, one_mulVec]

end FCPA

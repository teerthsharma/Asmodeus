import Mathlib.Data.Complex.Basic
import Mathlib.Analysis.Complex.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.DotProduct

/-!
L1 Born exactness (spec §3.1, Chase form). `V : (K+1) × N` Parseval frame (`Vᴴ V = 1`), `ψ` a unit vector;
the sum runs over ALL `K+1` outcomes (including the null one). `parseval` is the general isometry used by E3.
-/
open Matrix BigOperators

namespace FCPA

theorem dot_eq_sum_normSq {N : ℕ} (φ : Fin N → ℂ) :
    star φ ⬝ᵥ φ = ((∑ i, Complex.normSq (φ i) : ℝ) : ℂ) := by
  simp only [dotProduct, Pi.star_apply, Complex.star_def]
  rw [show ((∑ i, Complex.normSq (φ i) : ℝ) : ℂ) = ∑ i, ((Complex.normSq (φ i) : ℝ) : ℂ) from
    by exact_mod_cast RCLike.ofReal_sum (K := ℂ) Finset.univ (fun i => Complex.normSq (φ i))]
  apply Finset.sum_congr rfl
  intro i _
  rw [Complex.normSq_eq_conj_mul_self]

theorem parseval {K N : ℕ} (V : Matrix (Fin (K + 1)) (Fin N) ℂ) (hV : Vᴴ * V = 1)
    (φ : Fin N → ℂ) :
    ∑ k, Complex.normSq ((V *ᵥ φ) k) = ∑ i, Complex.normSq (φ i) := by
  have h1 : star (V *ᵥ φ) ⬝ᵥ (V *ᵥ φ) = star φ ⬝ᵥ φ := by
    rw [star_mulVec, ← dotProduct_mulVec, mulVec_mulVec, hV, one_mulVec]
  have h2 := dot_eq_sum_normSq (V *ᵥ φ)
  have h3 := dot_eq_sum_normSq φ
  rw [h1, h3] at h2
  exact_mod_cast h2.symm

theorem born {K N : ℕ} (V : Matrix (Fin (K + 1)) (Fin N) ℂ) (hV : Vᴴ * V = 1)
    (ψ : Fin N → ℂ) (hψ : star ψ ⬝ᵥ ψ = 1) :
    ∑ k, Complex.normSq ((V *ᵥ ψ) k) = 1 := by
  rw [parseval V hV ψ]
  have := dot_eq_sum_normSq ψ
  rw [hψ] at this
  exact_mod_cast this.symm

end FCPA

import Mathlib.LinearAlgebra.Matrix.PosDef

/-!
L23′ Non-destructive birth (spec §3.10 + Amendment A; Lean form without matrix square roots).
Given a Hermitian `S` with `S * S = E∅` (`S = E∅^{1/2}`; existence is the functional calculus, not needed here) and `0 ⪯ Q ⪯ 1`, set
`E_new = S Q S`, `E∅′ = S (1-Q) S`. Then `E_new + E∅′ = E∅`, both are PSD, so every old effect is untouched,
`Σ_k E_k + E_new + E∅′ = 1` still holds, and each `p_k(ψ)` is unchanged.
The vacuity remark (Chase): if `E∅` has rank one, `E_new = c E∅` is forced; this file proves only the positive direction.
-/
open Matrix BigOperators
open scoped ComplexOrder

namespace FCPA

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem birth_split (S Q : Matrix n n ℂ) (hS : Sᴴ = S) (hQ : Q.PosSemidef) (hQ' : (1 - Q).PosSemidef) :
    (S * Q * S).PosSemidef ∧ (S * (1 - Q) * S).PosSemidef ∧
    S * Q * S + S * (1 - Q) * S = S * S := by
  refine ⟨?_, ?_, ?_⟩
  · have := hQ.conjTranspose_mul_mul_same S
    rwa [hS] at this
  · have := hQ'.conjTranspose_mul_mul_same S
    rwa [hS] at this
  · noncomm_ring

theorem birth_preserves_sum {m : Type*} [Fintype m] (E : m → Matrix n n ℂ) (E0 Enew E0' : Matrix n n ℂ)
    (hsum : ∑ k, E k + E0 = 1) (hsplit : Enew + E0' = E0) :
    ∑ k, E k + Enew + E0' = 1 := by
  rw [add_assoc, hsplit, hsum]

/-- Born probabilities of the old effects do not depend on the birth (they are the same matrices); the two new effects
carry nonnegative mass summing to the old null mass. -/
theorem birth_born (E0 Enew E0' : Matrix n n ℂ) (h1 : Enew.PosSemidef) (h2 : E0'.PosSemidef)
    (hsplit : Enew + E0' = E0) (ψ : n → ℂ) :
    (star ψ ⬝ᵥ Enew *ᵥ ψ).re + (star ψ ⬝ᵥ E0' *ᵥ ψ).re = (star ψ ⬝ᵥ E0 *ᵥ ψ).re ∧
    0 ≤ (star ψ ⬝ᵥ Enew *ᵥ ψ).re ∧ 0 ≤ (star ψ ⬝ᵥ E0' *ᵥ ψ).re := by
  refine ⟨?_, ?_, ?_⟩
  · rw [← hsplit, add_mulVec, dotProduct_add, Complex.add_re]
  · simpa using (Complex.le_def.mp (h1.2 ψ)).1
  · simpa using (Complex.le_def.mp (h2.2 ψ)).1

/-- Born probability of outcome `k` in state `ψ`. -/
noncomputable def bornP {m : Type*} (E : m → Matrix n n ℂ) (ψ : n → ℂ) (k : m) : ℝ :=
  (star ψ ⬝ᵥ E k *ᵥ ψ).re

/-- The effect family after a birth: the old effects, then `E_new` (`true`) and `E∅′` (`false`). -/
def afterBirth {m : Type*} (E : m → Matrix n n ℂ) (Enew E0' : Matrix n n ℂ) :
    m ⊕ Bool → Matrix n n ℂ :=
  Sum.elim E (fun b => cond b Enew E0')

/-- **Old `p_k` unchanged** (spec L23′): every pre-existing outcome keeps exactly its Born probability, for every `ψ`. -/
theorem birth_old_unchanged {m : Type*} (E : m → Matrix n n ℂ) (Enew E0' : Matrix n n ℂ)
    (ψ : n → ℂ) (k : m) :
    bornP (afterBirth E Enew E0') ψ (Sum.inl k) = bornP E ψ k := rfl

theorem quad_sum {m : Type*} (E : m → Matrix n n ℂ) (ψ : n → ℂ) (s : Finset m) :
    star ψ ⬝ᵥ (∑ k in s, E k) *ᵥ ψ = ∑ k in s, star ψ ⬝ᵥ E k *ᵥ ψ := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert ha ih => rw [Finset.sum_insert ha, Finset.sum_insert ha, add_mulVec, dotProduct_add, ih]

/-- After a birth the Born probabilities still sum to one for every unit `ψ` (`Σ = I` is preserved). -/
theorem birth_total_born {m : Type*} [Fintype m] (E : m → Matrix n n ℂ) (E0 Enew E0' : Matrix n n ℂ)
    (hsum : ∑ k, E k + E0 = 1) (hsplit : Enew + E0' = E0) (ψ : n → ℂ) (hψ : star ψ ⬝ᵥ ψ = 1) :
    ∑ x, bornP (afterBirth E Enew E0') ψ x = 1 := by
  have h1 := birth_preserves_sum E E0 Enew E0' hsum hsplit
  have h2 : star ψ ⬝ᵥ (∑ k, E k + Enew + E0') *ᵥ ψ = 1 := by
    rw [h1, one_mulVec, hψ]
  rw [Fintype.sum_sum_type, Fintype.sum_bool]
  simp only [bornP, afterBirth, Sum.elim_inl, Sum.elim_inr]
  simp only [cond_true, cond_false]
  rw [add_mulVec, add_mulVec, dotProduct_add, dotProduct_add, quad_sum] at h2
  have := congrArg Complex.re h2
  simp only [Complex.add_re, Complex.re_sum, Complex.one_re] at this
  linarith

end FCPA

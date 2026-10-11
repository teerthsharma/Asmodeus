import Mathlib.Analysis.InnerProductSpace.Basic

/-!
L8 NS identifiability (spec §3.3, Chase form): `R = ν • L` with `L ≠ 0` ⇒ `ν = ⟪R, L⟫ / ⟪L, L⟫` (orthogonal projection).
With `f` unknown ν is NOT identifiable from a single snapshot (E2: `(ν', f + (ν-ν')Δω)` gives the same ω).
Identifiability across time (E2, stated as "if" per Chase): see `ns_identifiable_time`.
-/

namespace FCPA

theorem ns_ident {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (R L : E) (ν : ℝ)
    (hL : L ≠ 0) (h : R = ν • L) : ν = (inner R L : ℝ) / (inner L L : ℝ) := by
  subst h
  have : (inner L L : ℝ) ≠ 0 := inner_self_ne_zero.mpr hL
  rw [real_inner_smul_left]
  field_simp

/-- E2 (if-direction): if two parameter pairs `(ν, f)`, `(ν', f')` with a TIME-CONSTANT forcing give the same residual at two
times with `Δω(t) ≠ Δω(t')`, then `ν = ν'`. -/
theorem ns_identifiable_time {E : Type*} [AddCommGroup E] [Module ℝ E] (ν ν' : ℝ) (f f' L L' : E)
    (h : ν • L + f = ν' • L + f') (h' : ν • L' + f = ν' • L' + f') (hne : L ≠ L') : ν = ν' := by
  have h1 : ν • (L - L') = ν' • (L - L') := by
    calc ν • (L - L') = (ν • L + f) - (ν • L' + f) := by rw [smul_sub]; abel
      _ = (ν' • L + f') - (ν' • L' + f') := by rw [h, h']
      _ = ν' • (L - L') := by rw [smul_sub]; abel
  have h2 : (ν - ν') • (L - L') = 0 := by rw [sub_smul, h1, sub_self]
  rcases smul_eq_zero.mp h2 with h3 | h3
  · linarith
  · exact absurd (sub_eq_zero.mp h3) hne

end FCPA

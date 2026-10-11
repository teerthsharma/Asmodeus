import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Tactic.Linarith

/-!
L-GIVEN (Amendment B): library-term classification. `η = ‖e‖/σ_min(A)` bounds the coefficient error `|x*_j - x⁰_j| ≤ η`.
PRESENT if `|x⁰_j| > η` (then `x*_j ≠ 0`); SMALL if `|x⁰_j| ≤ η` (then `|x*_j| ≤ 2η`). For a single scalar coordinate the two
cases are exhaustive; the REFUSE band appears when η is itself only estimated and two thresholds are used.
-/

namespace FCPA

theorem present_sound {x x0 η : ℝ} (h : |x - x0| ≤ η) (hp : η < |x0|) : x ≠ 0 := by
  rintro rfl
  simp only [zero_sub, abs_neg] at h
  linarith

theorem small_sound {x x0 η : ℝ} (h : |x - x0| ≤ η) (hs : |x0| ≤ η) : |x| ≤ 2 * η := by
  rw [abs_le] at *
  constructor <;> linarith [h.1, h.2, hs.1, hs.2]

/-- `η = ‖e‖/σ`: if `A v = e` and `σ ‖v‖ ≤ ‖A v‖` (σ > 0, σ is the least singular value), then `‖v‖ ≤ ‖e‖/σ`. -/
theorem eta_bound {E F : Type*} [NormedAddCommGroup E] [NormedAddCommGroup F] (A : E → F) (v : E) (e : F)
    {σ : ℝ} (hσ : 0 < σ) (hA : A v = e) (hmin : σ * ‖v‖ ≤ ‖A v‖) : ‖v‖ ≤ ‖e‖ / σ := by
  rw [le_div_iff hσ]
  rw [hA] at hmin
  linarith

/-- **Structure learned at resolution `2η`** (Amendment B): if every coefficient estimate is within `η` of the truth, only
the NS terms are PRESENT (`|x⁰_j| > η`) and all other library terms are SMALL (`|x⁰_j| ≤ η`), then the NS terms are truly
nonzero and every other term is truly at most `2η`. -/
theorem structure_learned {J : Type*} (NS : Set J) (x x0 : J → ℝ) (η : ℝ)
    (hclose : ∀ j, |x j - x0 j| ≤ η)
    (hpresent : ∀ j ∈ NS, η < |x0 j|) (hsmall : ∀ j ∉ NS, |x0 j| ≤ η) :
    (∀ j ∈ NS, x j ≠ 0) ∧ (∀ j ∉ NS, |x j| ≤ 2 * η) :=
  ⟨fun j hj => present_sound (hclose j) (hpresent j hj),
   fun j hj => small_sound (hclose j) (hsmall j hj)⟩

end FCPA

import Mathlib.Analysis.NormedSpace.MatrixExponential
import Mathlib.Analysis.Complex.Basic

/-!
L15 Crossing composition (spec §3.4). `S(x) = exp(∑_a x_a G_a)`. If the generators pairwise commute then
`S(y) S(x) = S(x + y)`, so `S(Δ_{jk}) S(Δ_{ij}) = S(Δ_{ik})` (with `Δ_{ab} = e_b - e_a`) and every loop of eras returns to `1`
(trivial holonomy; the abelian special case of L16).
-/
open Matrix BigOperators NormedSpace

namespace FCPA

variable {n : Type*} [Fintype n] [DecidableEq n] {d : ℕ}

/-- Crossing operator `S(x) = exp(∑_a x_a G_a)`. -/
noncomputable def crossing (G : Fin d → Matrix n n ℂ) (x : Fin d → ℝ) : Matrix n n ℂ :=
  exp ℂ (∑ a, (x a : ℂ) • G a)

theorem crossing_comp (G : Fin d → Matrix n n ℂ) (hG : ∀ a b, Commute (G a) (G b))
    (x y : Fin d → ℝ) : crossing G y * crossing G x = crossing G (x + y) := by
  unfold crossing
  have hc : Commute (∑ a, (y a : ℂ) • G a) (∑ a, (x a : ℂ) • G a) := by
    apply Commute.sum_left
    intro a _
    apply Commute.sum_right
    intro b _
    exact ((hG a b).smul_left _).smul_right _
  rw [← Matrix.exp_add_of_commute ℂ _ _ hc]
  congr 1
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro a _
  rw [← add_smul]
  congr 1
  rw [Pi.add_apply]
  push_cast
  ring

theorem crossing_zero (G : Fin d → Matrix n n ℂ) : crossing G 0 = 1 := by
  simp [crossing]

theorem crossing_loop (G : Fin d → Matrix n n ℂ) (hG : ∀ a b, Commute (G a) (G b))
    {ι : Type*} (e : ι → Fin d → ℝ) (i j k : ι) :
    crossing G (e i - e k) * crossing G (e k - e j) * crossing G (e j - e i) = 1 := by
  rw [mul_assoc, crossing_comp G hG, show (e j - e i) + (e k - e j) = e k - e i by abel,
    crossing_comp G hG, show (e k - e i) + (e i - e k) = 0 by abel, crossing_zero]

end FCPA

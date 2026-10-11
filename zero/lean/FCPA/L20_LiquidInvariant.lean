import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Mathlib.Tactic.LinearCombination

/-!
L20 Discrete liquid invariant interval (spec §3.7, Chase form, scalar / componentwise; `F` acts as a diagonal gain).
`x_{n+1} = x_n + h(-(1/τ + F_n) x_n + A F_n)`, `0 ≤ F ≤ B`, `h ≤ τ/(1+τB)`; the box `[min 0 A, max 0 A]` is invariant.

L20′ (Amendment A): with jumps `x_{n+1} = step(x_n) + J_n`, `|J_n| ≤ δ`, the distance `d(x) = dist x box` obeys
`d(x_{n+1}) ≤ (1 - h/τ) d(x_n) + |J_n|`, hence `d(x_n) ≤ ∑_{j<n} (1-h/τ)^{n-1-j}|J_j|` and `d(x_n) ≤ δ τ/h` when `x_0 ∈ box`.
-/
open BigOperators

namespace FCPA

/-- Core algebra shared by all steps: `a = 1 - h(1/τ + F) ≥ 0` under the step-size hypothesis. -/
theorem liquid_coeff_nonneg (τ h B F : ℝ) (hτ : 0 < τ) (hh : 0 < h) (hB : 0 ≤ B)
    (hs : h ≤ τ / (1 + τ * B)) (hF : 0 ≤ F ∧ F ≤ B) : 0 ≤ 1 - h * (1 / τ + F) := by
  have hs' : h * (1 + τ * B) ≤ τ := by
    rwa [le_div_iff (by positivity)] at hs
  have h1 : h * (1 / τ + F) ≤ h * (1 / τ + B) :=
    mul_le_mul_of_nonneg_left (by linarith [hF.2]) hh.le
  have h2 : h * (1 / τ + B) ≤ 1 := by
    have e : h * (1 / τ + B) = h * (1 + τ * B) / τ := by
      rw [eq_div_iff hτ.ne']
      have := one_div_mul_cancel hτ.ne'
      linear_combination h * this
    rw [e, div_le_iff hτ]
    linarith
  linarith

theorem liquid_step_mem (τ h B A x F : ℝ) (hτ : 0 < τ) (hh : 0 < h) (hB : 0 ≤ B)
    (hs : h ≤ τ / (1 + τ * B)) (hF : 0 ≤ F ∧ F ≤ B)
    (hx : min 0 A ≤ x ∧ x ≤ max 0 A) :
    min 0 A ≤ x + h * (-(1 / τ + F) * x + A * F) ∧ x + h * (-(1 / τ + F) * x + A * F) ≤ max 0 A := by
  have ha := liquid_coeff_nonneg τ h B F hτ hh hB hs hF
  have hu : 0 ≤ h / τ := by positivity
  have hf : 0 ≤ h * F := mul_nonneg hh.le hF.1
  have hm1 : min 0 A ≤ 0 := min_le_left _ _
  have hm2 : min 0 A ≤ A := min_le_right _ _
  have hM1 : 0 ≤ max 0 A := le_max_left _ _
  have hM2 : A ≤ max 0 A := le_max_right _ _
  have e1 : x + h * (-(1 / τ + F) * x + A * F) = (1 - h * (1 / τ + F)) * x + (h * F) * A := by ring
  have e2 : h * (1 / τ) = h / τ := by ring
  rw [e1]
  constructor
  · nlinarith [mul_le_mul_of_nonneg_left hx.1 ha, mul_le_mul_of_nonneg_left hm2 hf,
      mul_nonneg hu (neg_nonneg.mpr hm1)]
  · nlinarith [mul_le_mul_of_nonneg_left hx.2 ha, mul_le_mul_of_nonneg_left hM2 hf,
      mul_nonneg hu hM1]

theorem liquid_invariant (τ h B A : ℝ) (hτ : 0 < τ) (hh : 0 < h) (hB : 0 ≤ B)
    (hs : h ≤ τ / (1 + τ * B)) (x F : ℕ → ℝ) (hF : ∀ n, 0 ≤ F n ∧ F n ≤ B)
    (hx : ∀ n, x (n + 1) = x n + h * (-(1 / τ + F n) * x n + A * F n))
    (h0 : min 0 A ≤ x 0 ∧ x 0 ≤ max 0 A) : ∀ n, min 0 A ≤ x n ∧ x n ≤ max 0 A := by
  intro n
  induction n with
  | zero => exact h0
  | succ n ih => rw [hx n]; exact liquid_step_mem τ h B A (x n) (F n) hτ hh hB hs (hF n) ih

/-- Distance from `x` to the box `[min 0 A, max 0 A]`. -/
noncomputable def boxDist (A x : ℝ) : ℝ := max (max (min 0 A - x) (x - max 0 A)) 0

theorem boxDist_nonneg (A x : ℝ) : 0 ≤ boxDist A x := le_max_right _ _

theorem boxDist_zero_of_mem (A x : ℝ) (hx : min 0 A ≤ x ∧ x ≤ max 0 A) : boxDist A x = 0 := by
  unfold boxDist
  apply le_antisymm _ (le_max_right _ _)
  exact max_le (max_le (by linarith [hx.1]) (by linarith [hx.2])) le_rfl

theorem liquid_dist_step (τ h B A x F J : ℝ) (hτ : 0 < τ) (hh : 0 < h) (hB : 0 ≤ B)
    (hs : h ≤ τ / (1 + τ * B)) (hF : 0 ≤ F ∧ F ≤ B) :
    boxDist A (x + h * (-(1 / τ + F) * x + A * F) + J) ≤ (1 - h / τ) * boxDist A x + |J| := by
  have ha := liquid_coeff_nonneg τ h B F hτ hh hB hs hF
  have hu : 0 ≤ h / τ := by positivity
  have hf : 0 ≤ h * F := mul_nonneg hh.le hF.1
  have hm1 : min 0 A ≤ 0 := min_le_left _ _
  have hm2 : min 0 A ≤ A := min_le_right _ _
  have hM1 : 0 ≤ max 0 A := le_max_left _ _
  have hM2 : A ≤ max 0 A := le_max_right _ _
  have e1 : x + h * (-(1 / τ + F) * x + A * F) + J
      = (1 - h * (1 / τ + F)) * x + (h * F) * A + J := by ring
  have e2 : h * (1 / τ) = h / τ := by ring
  have hd0 := boxDist_nonneg A x
  have hd1 : min 0 A - x ≤ boxDist A x := le_trans (le_max_left _ _) (le_max_left _ _)
  have hd2 : x - max 0 A ≤ boxDist A x := le_trans (le_max_right _ _) (le_max_left _ _)
  obtain ⟨a, hadef⟩ : ∃ a, a = 1 - h * (1 / τ + F) := ⟨_, rfl⟩
  rw [← hadef] at e1 ha
  have hac : a ≤ 1 - h / τ := by
    have : a = 1 - h / τ - h * F := by rw [hadef]; ring
    rw [this]; linarith
  have hcd : a * boxDist A x ≤ (1 - h / τ) * boxDist A x := mul_le_mul_of_nonneg_right hac hd0
  have hJ1 := neg_abs_le J
  have hJ2 := le_abs_self J
  have habc : a + h * F + h / τ = 1 := by rw [hadef]; ring
  have hmid : min 0 A = a * min 0 A + h * F * min 0 A + h / τ * min 0 A := by
    calc min 0 A = (a + h * F + h / τ) * min 0 A := by rw [habc, one_mul]
      _ = _ := by ring
  have hMid : max 0 A = a * max 0 A + h * F * max 0 A + h / τ * max 0 A := by
    calc max 0 A = (a + h * F + h / τ) * max 0 A := by rw [habc, one_mul]
      _ = _ := by ring
  rw [e1]
  generalize boxDist A x = d at hd0 hd1 hd2 hcd ⊢
  unfold boxDist
  refine max_le (max_le ?_ ?_) ?_
  · linarith [mul_le_mul_of_nonneg_left hd1 ha, mul_nonneg hu (neg_nonneg.mpr hm1),
      mul_nonneg hf (sub_nonneg.mpr hm2)]
  · linarith [mul_le_mul_of_nonneg_left hd2 ha, mul_nonneg hu hM1,
      mul_nonneg hf (sub_nonneg.mpr hM2)]
  · linarith [mul_nonneg (le_trans ha hac) hd0, abs_nonneg J]

theorem liquid_jump_bound (τ h B A δ : ℝ) (hτ : 0 < τ) (hh : 0 < h) (hB : 0 ≤ B)
    (hs : h ≤ τ / (1 + τ * B)) (hδ : 0 ≤ δ) (x F J : ℕ → ℝ) (hF : ∀ n, 0 ≤ F n ∧ F n ≤ B)
    (hJ : ∀ n, |J n| ≤ δ)
    (hx : ∀ n, x (n + 1) = x n + h * (-(1 / τ + F n) * x n + A * F n) + J n)
    (h0 : min 0 A ≤ x 0 ∧ x 0 ≤ max 0 A) (n : ℕ) :
    boxDist A (x n) ≤ ∑ j in Finset.range n, (1 - h / τ) ^ (n - 1 - j) * |J j| ∧
    boxDist A (x n) ≤ δ * τ / h := by
  have hc0 : 0 ≤ 1 - h / τ := by
    have := liquid_coeff_nonneg τ h B 0 hτ hh hB hs ⟨le_rfl, hB⟩
    have e : h * (1 / τ + 0) = h / τ := by ring
    linarith
  constructor
  · induction n with
    | zero => simp [boxDist_zero_of_mem A _ h0]
    | succ n ih =>
      have hstep := liquid_dist_step τ h B A (x n) (F n) (J n) hτ hh hB hs (hF n)
      rw [← hx n] at hstep
      rw [Finset.sum_range_succ]
      have hsum : (1 - h / τ) * ∑ j in Finset.range n, (1 - h / τ) ^ (n - 1 - j) * |J j|
          = ∑ j in Finset.range n, (1 - h / τ) ^ (n + 1 - 1 - j) * |J j| := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro j hj
        have hjn : j < n := Finset.mem_range.mp hj
        have : n + 1 - 1 - j = (n - 1 - j) + 1 := by omega
        rw [this, pow_succ]; ring
      have hlast : (1 - h / τ) ^ (n + 1 - 1 - n) * |J n| = |J n| := by simp
      have := mul_le_mul_of_nonneg_left ih hc0
      linarith
  · induction n with
    | zero =>
      rw [boxDist_zero_of_mem A _ h0]; positivity
    | succ n ih =>
      have hstep := liquid_dist_step τ h B A (x n) (F n) (J n) hτ hh hB hs (hF n)
      rw [← hx n] at hstep
      have e : (1 - h / τ) * (δ * τ / h) + δ = δ * τ / h := by field_simp; ring
      have := mul_le_mul_of_nonneg_left ih hc0
      linarith [hJ n]

end FCPA

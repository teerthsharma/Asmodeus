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

/-!
L7 ACI under drift (spec §3.2), CORRECTED (Chase): the spec statement is false without the nested-set boundary convention
(`α_t < 0 ⇒ err_t = 0`, `α_t > 1 ⇒ err_t = 1`): with `err ≡ 0`, `α_t` drifts to `+∞` and the average error is `0`.
The initial condition is relaxed from `α 0 ∈ [0,1]` to `α 0 ∈ [-γ, 1+γ]` (needed by L24 for a stretch starting mid-run).
-/
open BigOperators

namespace FCPA

theorem aci_invariant (γ a : ℝ) (hγ : 0 < γ) (ha : 0 ≤ a ∧ a ≤ 1) (α err : ℕ → ℝ)
    (hrec : ∀ t, α (t + 1) = α t + γ * (a - err t)) (he : ∀ t, err t = 0 ∨ err t = 1)
    (hlo : ∀ t, α t < 0 → err t = 0) (hhi : ∀ t, 1 < α t → err t = 1)
    (h0 : -γ ≤ α 0 ∧ α 0 ≤ 1 + γ) : ∀ t, -γ ≤ α t ∧ α t ≤ 1 + γ := by
  intro t
  induction t with
  | zero => exact h0
  | succ t ih =>
    rw [hrec t]
    have hga : 0 ≤ γ * a := mul_nonneg hγ.le ha.1
    have hga1 : γ * a ≤ γ := by nlinarith [ha.2]
    rcases he t with h | h
    · have hle : α t ≤ 1 := by
        by_contra hc
        have := hhi t (not_le.mp hc)
        rw [h] at this; norm_num at this
      rw [h]
      constructor <;> nlinarith [ih.1, ih.2]
    · have hge : 0 ≤ α t := by
        by_contra hc
        have := hlo t (not_le.mp hc)
        rw [h] at this; norm_num at this
      rw [h]
      constructor <;> nlinarith [ih.1, ih.2]

theorem aci (γ a : ℝ) (hγ : 0 < γ) (ha : 0 ≤ a ∧ a ≤ 1) (α err : ℕ → ℝ)
    (hrec : ∀ t, α (t + 1) = α t + γ * (a - err t)) (he : ∀ t, err t = 0 ∨ err t = 1)
    (hlo : ∀ t, α t < 0 → err t = 0) (hhi : ∀ t, 1 < α t → err t = 1)
    (h0 : -γ ≤ α 0 ∧ α 0 ≤ 1 + γ) (T : ℕ) (hT : 0 < T) :
    |(∑ t in Finset.range T, err t) / T - a| ≤ (max (α 0) (1 - α 0) + γ) / (γ * T) := by
  have hinv := aci_invariant γ a hγ ha α err hrec he hlo hhi h0
  have hsum : ∀ n : ℕ, α n = α 0 + γ * (n * a - ∑ t in Finset.range n, err t) := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      rw [hrec n, ih, Finset.sum_range_succ]
      push_cast
      ring
  have hTpos : (0 : ℝ) < T := by exact_mod_cast hT
  have hgT : 0 < γ * T := mul_pos hγ hTpos
  have e : (∑ t in Finset.range T, err t) / T - a = (α 0 - α T) / (γ * T) := by
    rw [hsum T]
    field_simp
    ring
  rw [e, abs_div, abs_of_pos hgT, div_le_div_right hgT, abs_le]
  have h1 := hinv T
  have m1 := le_max_left (α 0) (1 - α 0)
  have m2 := le_max_right (α 0) (1 - α 0)
  constructor <;> linarith [h1.1, h1.2, h0.1, h0.2]

end FCPA

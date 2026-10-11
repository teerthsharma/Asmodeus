import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
L4 Post-selection temperature (spec §3.1, Chase form). `w` lives on `Option S` (`none` = mass outside `S`).
Needs a positive entry in `S` (otherwise the normaliser is 0 and Lean's `x/0 = 0` breaks the sum), and any `T > 0`.
-/
open BigOperators

namespace FCPA

theorem post_select {ι : Type*} [Fintype ι] [DecidableEq ι] (p : ι → ℝ) (h0 : ∀ i, 0 ≤ p i)
    (h1 : ∑ i, p i = 1) (S : Finset ι) {T : ℝ} (hT : 0 < T) (hpos : ∃ j ∈ S, 0 < p j) :
    ∃ w : Option S → ℝ,
      (∀ k : S, w (some k) = (∑ j in S, p j) * p k ^ (1 / T) / ∑ j in S, p j ^ (1 / T)) ∧
      w none = 1 - ∑ j in S, p j ∧
      (∀ x, 0 ≤ w x) ∧ ∑ x, w x = 1 ∧
      ∀ i j : S, (p i ≤ p j ↔ w (some i) ≤ w (some j)) := by
  obtain ⟨j0, hj0, hj0p⟩ := hpos
  have hT' : 0 < 1 / T := by positivity
  set SS : ℝ := ∑ j in S, p j with hSS
  set D : ℝ := ∑ j in S, p j ^ (1 / T) with hD
  have hDpos : 0 < D :=
    Finset.sum_pos' (fun j _ => Real.rpow_nonneg (h0 j) _) ⟨j0, hj0, Real.rpow_pos_of_pos hj0p _⟩
  have hSSpos : 0 < SS := Finset.sum_pos' (fun j _ => h0 j) ⟨j0, hj0, hj0p⟩
  have hSS1 : SS ≤ 1 := by
    rw [← h1]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ S) (fun i _ _ => h0 i)
  refine ⟨fun x => match x with
    | some k => SS * p k ^ (1 / T) / D
    | none => 1 - SS, fun k => rfl, rfl, ?_, ?_, ?_⟩
  · intro x
    cases x with
    | none => simp only; linarith
    | some k =>
      simp only
      have := Real.rpow_nonneg (h0 k) (1 / T)
      positivity
  · rw [Fintype.sum_option]
    simp only
    rw [Finset.sum_coe_sort S (fun k => SS * p k ^ (1 / T) / D), ← Finset.sum_div, ← Finset.mul_sum,
      ← hD, mul_div_assoc, div_self hDpos.ne']
    ring
  · intro i j
    simp only
    have hc : 0 < SS / D := div_pos hSSpos hDpos
    have e : ∀ k : S, SS * p k ^ (1 / T) / D = (SS / D) * p k ^ (1 / T) := fun k => by ring
    rw [e, e, mul_le_mul_left hc, Real.rpow_le_rpow_iff (h0 i) (h0 j) hT']

end FCPA

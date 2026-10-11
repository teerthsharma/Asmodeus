import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
import Mathlib.Analysis.MeanInequalitiesPow

/-!
L3 Tail certificate (spec §3.1).
* `tail_cert`: for `q = 1/T ≥ 1` (T ≤ 1): `∑_{k∉S} p_k^q ≤ M^(q-1) · ∑_{k∉S} p_k` for any upper bound `M ≥ p_k` on the tail
  (Chase form: an upper bound `M`, not a `max`, since the max of an empty tail does not exist).
* `no_Kfree_bound`: for `0 < q < 1` (T > 1) no bound independent of the tail length exists (counterexample).
* `tail_power_mean` (L3′, Amendment B): for `0 < q < 1`, `∑_{k∉S} p_k^q ≤ m^(1-q) · r^q`, `m` = tail length, `r` = tail mass.
-/
open BigOperators

namespace FCPA

theorem tail_cert {ι : Type*} [Fintype ι] [DecidableEq ι] (S : Finset ι) (p : ι → ℝ)
    (h0 : ∀ i, 0 ≤ p i) {q M : ℝ} (hq : 1 ≤ q) (hM : ∀ i ∉ S, p i ≤ M) :
    ∑ i in Sᶜ, p i ^ q ≤ M ^ (q - 1) * ∑ i in Sᶜ, p i := by
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i hi
  have hiS : i ∉ S := by simpa using hi
  have h1 : p i ^ q = p i ^ (q - 1) * p i := by
    have : q = (q - 1) + 1 := by ring
    conv_lhs => rw [this]
    rcases (h0 i).eq_or_lt with h | h
    · rw [← h]; simp [Real.zero_rpow (by linarith : q ≠ 0)]
    · rw [Real.rpow_add h, Real.rpow_one]
  rw [h1]
  exact mul_le_mul_of_nonneg_right
    (Real.rpow_le_rpow (h0 i) (hM i hiS) (by linarith)) (h0 i)

theorem no_Kfree_bound {q : ℝ} (hq0 : 0 < q) (hq1 : q < 1) (B r : ℝ) (hr : 0 < r) :
    ∃ (n : ℕ) (p : Fin n → ℝ), (∀ i, 0 ≤ p i) ∧ ∑ i, p i = r ∧ B < ∑ i, p i ^ q := by
  have hrq : 0 < r ^ q := Real.rpow_pos_of_pos hr q
  have ht := (tendsto_rpow_atTop (show 0 < 1 - q by linarith)).comp tendsto_nat_cast_atTop_atTop
  obtain ⟨n, hn1, hn2⟩ := ((ht.eventually_gt_atTop (B / r ^ q)).and
    (Filter.eventually_ge_atTop 1)).exists
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn2
  refine ⟨n, fun _ => r / n, fun _ => by positivity, ?_, ?_⟩
  · simp [Finset.sum_const]; field_simp; ring
  · simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    have : (n : ℝ) * (r / n) ^ q = (n : ℝ) ^ (1 - q) * r ^ q := by
      rw [Real.div_rpow hr.le hnpos.le, Real.rpow_sub hnpos, Real.rpow_one]
      field_simp
    rw [this]
    have h2 : B / r ^ q < (n : ℝ) ^ (1 - q) := hn1
    rw [div_lt_iff hrq] at h2
    exact h2

theorem tail_power_mean {ι : Type*} [Fintype ι] [DecidableEq ι] (S : Finset ι) (p : ι → ℝ)
    (h0 : ∀ i, 0 ≤ p i) {q : ℝ} (hq0 : 0 < q) (hq1 : q < 1) :
    ∑ i in Sᶜ, p i ^ q ≤ ((Sᶜ).card : ℝ) ^ (1 - q) * (∑ i in Sᶜ, p i) ^ q := by
  by_cases hm : (Sᶜ).card = 0
  · have : Sᶜ = ∅ := Finset.card_eq_zero.mp hm
    rw [this]
    simp only [Finset.sum_empty, Finset.card_empty, Nat.cast_zero]
    positivity
  · have hmpos : (0 : ℝ) < ((Sᶜ).card : ℝ) := by exact_mod_cast Nat.pos_of_ne_zero hm
    set m : ℝ := ((Sᶜ).card : ℝ) with hmdef
    set r : ℝ := ∑ i in Sᶜ, p i with hr
    have hw : ∀ i ∈ Sᶜ, 0 ≤ (fun _ : ι => 1 / m) i := fun _ _ => by positivity
    have hw1 : ∑ i in Sᶜ, (fun _ : ι => 1 / m) i = 1 := by
      simp only [Finset.sum_const, nsmul_eq_mul]
      rw [← hmdef]; field_simp
    have hz : ∀ i ∈ Sᶜ, 0 ≤ (fun i => p i ^ q) i := fun i _ => Real.rpow_nonneg (h0 i) q
    have key := Real.arith_mean_le_rpow_mean (Sᶜ) (fun _ : ι => 1 / m) (fun i => p i ^ q) hw hw1 hz
      (p := 1 / q) (by rw [le_div_iff hq0]; linarith)
    have e1 : ∀ i ∈ Sᶜ, (1 / m) * (p i ^ q) ^ (1 / q) = (1 / m) * p i := by
      intro i _
      rw [← Real.rpow_mul (h0 i), mul_one_div_cancel hq0.ne', Real.rpow_one]
    simp only at key
    rw [Finset.sum_congr rfl e1, ← Finset.mul_sum, ← Finset.mul_sum, one_div_one_div] at key
    rw [← hr] at key
    have hr0 : 0 ≤ r := Finset.sum_nonneg (fun i _ => h0 i)
    rw [Real.mul_rpow (by positivity) hr0] at key
    have e2 : (1 / m) ^ q = m ^ (-q) := by
      rw [one_div, Real.inv_rpow hmpos.le, ← Real.rpow_neg hmpos.le]
    rw [e2] at key
    have e3 : m ^ (1 - q) = m * m ^ (-q) := by
      rw [sub_eq_add_neg, Real.rpow_add hmpos, Real.rpow_one]
    rw [e3]
    have : (∑ i in Sᶜ, p i ^ q) = m * ((1 / m) * ∑ i in Sᶜ, p i ^ q) := by
      field_simp; ring
    rw [this, mul_assoc]
    exact mul_le_mul_of_nonneg_left key hmpos.le

end FCPA

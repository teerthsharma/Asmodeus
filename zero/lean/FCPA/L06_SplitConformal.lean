import Mathlib.MeasureTheory.Integral.Lebesgue
import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic
import Mathlib.MeasureTheory.Group.Arithmetic
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.Order.Floor
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

/-!
L6 / L6′ split-conformal coverage (spec §3.2, Chase's ties-safe counting lemma).
Scores `S_0,…,S_n` (`N = n+1` of them; index `0` is the test score, the others are calibration scores) that are
exchangeable. The conformal p-value of the test score is `p = #{i : S_i ≥ S_0}/N` (the `+1` of the usual
`(1 + #{calibration i : S_i ≥ S_0})/(n+1)` is the term `i = 0`); the prediction set contains the true label iff `p > α`.
* `conformal_count` (ties-safe counting lemma): for any real vector `a` and `m ≤ N`, at least `m` indices `j` have fewer
  than `m` entries strictly below `a_j`.
* `conformal_coverage`: exchangeability + counting ⇒ `P(rank(S_0) < m) ≥ m/N`.
* `split_conformal_coverage`: `P(α < p) ≥ 1 - α`, for every `α ∈ [0,1]`, with ties allowed.
* `rank_le_iff` (L6′ newborn rule): the quantile rank `⌈(n+1)(1-α)⌉` is `≤ n` iff `n ≥ 1/α - 1`; below that threshold the set
  contains every label (the newborn era is included in every `C`).
Mondrian (per-era) calibration is this theorem applied inside each class (condition `μ` on the class event).
-/
open BigOperators MeasureTheory
open scoped ENNReal

namespace FCPA

/-- Number of entries of `a` strictly below `a j`. -/
noncomputable def rankBelow {N : ℕ} (a : Fin N → ℝ) (j : Fin N) : ℕ :=
  (Finset.univ.filter (fun i => a i < a j)).card

/-- **Ties-safe counting lemma.** At least `m` indices have rank `< m`. -/
theorem conformal_count {N : ℕ} (a : Fin N → ℝ) (m : ℕ) (hm : m ≤ N) :
    m ≤ (Finset.univ.filter (fun j => rankBelow a j < m)).card := by
  by_contra hlt
  push_neg at hlt
  set T := Finset.univ.filter (fun j => rankBelow a j < m) with hT
  set U := Finset.univ.filter (fun j => m ≤ rankBelow a j) with hU
  have hsum : T.card + U.card = N := by
    have := Finset.filter_card_add_filter_neg_card_eq_card (s := (Finset.univ : Finset (Fin N)))
      (fun j => rankBelow a j < m)
    simp only [not_lt, Finset.card_univ, Fintype.card_fin] at this
    exact this
  have hUne : U.Nonempty := by
    rw [Finset.nonempty_iff_ne_empty]
    intro h
    rw [h] at hsum
    simp at hsum
    omega
  obtain ⟨J, hJU, hJmin⟩ := Finset.exists_min_image U a hUne
  have hJ : m ≤ rankBelow a J := (Finset.mem_filter.mp hJU).2
  have hsub : Finset.univ.filter (fun i => a i < a J) ⊆ T := by
    intro i hi
    have hlt' : a i < a J := (Finset.mem_filter.mp hi).2
    rw [hT, Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    by_contra hnot
    have : i ∈ U := Finset.mem_filter.mpr ⟨Finset.mem_univ _, not_lt.mp hnot⟩
    exact absurd (hJmin i this) (not_le.mpr hlt')
  have := Finset.card_le_card hsub
  unfold rankBelow at hJ
  omega

variable {Ω : Type*} [MeasurableSpace Ω]

theorem measurable_rankBelow {N : ℕ} (S : Fin N → Ω → ℝ) (hS : ∀ i, Measurable (S i)) (j : Fin N) :
    Measurable (fun ω => rankBelow (fun i => S i ω) j) := by
  have : (fun ω => rankBelow (fun i => S i ω) j)
      = fun ω => ∑ i, (if S i ω < S j ω then 1 else 0) := by
    funext ω; unfold rankBelow; rw [Finset.card_filter]
  rw [this]
  refine Finset.measurable_sum _ (fun i _ => ?_)
  exact Measurable.ite (measurableSet_lt (hS i) (hS j)) measurable_const measurable_const

theorem measurableSet_rank_lt {N : ℕ} (S : Fin N → Ω → ℝ) (hS : ∀ i, Measurable (S i)) (j : Fin N)
    (m : ℕ) : MeasurableSet {ω | rankBelow (fun i => S i ω) j < m} :=
  measurable_rankBelow S hS j (measurableSet_lt measurable_id measurable_const)

/-- Exchangeability + counting ⇒ the test index is covered with probability `≥ m/N`. -/
theorem conformal_coverage (μ : Measure Ω) [IsProbabilityMeasure μ] {N : ℕ} [NeZero N]
    (S : Fin N → Ω → ℝ) (hS : ∀ i, Measurable (S i)) (m : ℕ) (hm : m ≤ N)
    (hexch : ∀ j : Fin N, μ {ω | rankBelow (fun i => S i ω) j < m}
      = μ {ω | rankBelow (fun i => S i ω) 0 < m}) :
    (m : ℝ) / N ≤ (μ {ω | rankBelow (fun i => S i ω) 0 < m}).toReal := by
  set A : Fin N → Set Ω := fun j => {ω | rankBelow (fun i => S i ω) j < m} with hA
  have hAm : ∀ j, MeasurableSet (A j) := fun j => measurableSet_rank_lt S hS j m
  have hpt : ∀ ω, (m : ℝ≥0∞) ≤ ∑ j, (A j).indicator (fun _ => (1 : ℝ≥0∞)) ω := by
    intro ω
    have h := conformal_count (fun i => S i ω) m hm
    have e : ∑ j, (A j).indicator (fun _ => (1 : ℝ≥0∞)) ω
        = ((Finset.univ.filter (fun j => rankBelow (fun i => S i ω) j < m)).card : ℝ≥0∞) := by
      rw [Finset.card_filter]
      push_cast
      apply Finset.sum_congr rfl
      intro j _
      simp [hA, Set.indicator]
    rw [e]
    exact_mod_cast h
  have hint : ∫⁻ ω, ∑ j, (A j).indicator (fun _ => (1 : ℝ≥0∞)) ω ∂μ = ∑ j, μ (A j) := by
    rw [lintegral_finset_sum _ (fun j _ => (measurable_const.indicator (hAm j)))]
    apply Finset.sum_congr rfl
    intro j _
    exact lintegral_indicator_const (hAm j) 1 |>.trans (by simp)
  have hge : (m : ℝ≥0∞) ≤ ∑ j, μ (A j) := by
    rw [← hint]
    calc (m : ℝ≥0∞) = ∫⁻ _ω, (m : ℝ≥0∞) ∂μ := by simp
      _ ≤ _ := lintegral_mono hpt
  have hconst : ∑ j, μ (A j) = (N : ℝ≥0∞) * μ (A 0) := by
    rw [Finset.sum_congr rfl (fun j _ => hexch j)]
    simp
  rw [hconst] at hge
  have hfin : μ (A 0) ≠ ∞ := measure_ne_top _ _
  have hN : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have := ENNReal.toReal_mono (ENNReal.mul_ne_top (ENNReal.nat_ne_top N) hfin) hge
  rw [ENNReal.toReal_mul] at this
  simp only [ENNReal.toReal_nat] at this
  rw [div_le_iff hN]
  linarith

/-- Conformal p-value of the test score `a 0`: the fraction of scores at least as large. -/
noncomputable def pval {N : ℕ} (a : Fin N → ℝ) (j : Fin N) : ℝ :=
  ((Finset.univ.filter (fun i => a j ≤ a i)).card : ℝ) / N

theorem pval_gt_iff {N : ℕ} [NeZero N] (a : Fin N → ℝ) (j : Fin N) (α : ℝ) :
    α < pval a j ↔ (rankBelow a j : ℝ) < N * (1 - α) := by
  have hN : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have hsum : (Finset.univ.filter (fun i => a j ≤ a i)).card + rankBelow a j = N := by
    unfold rankBelow
    have := Finset.filter_card_add_filter_neg_card_eq_card (s := (Finset.univ : Finset (Fin N)))
      (fun i => a j ≤ a i)
    simp only [not_le] at this
    simpa using this
  have hc : ((Finset.univ.filter (fun i => a j ≤ a i)).card : ℝ) = N - rankBelow a j := by
    have : ((Finset.univ.filter (fun i => a j ≤ a i)).card : ℝ) + rankBelow a j = N := by
      exact_mod_cast hsum
    linarith
  unfold pval
  rw [hc, lt_div_iff hN]
  constructor <;> intro h <;> nlinarith [h]

/-- **Split-conformal coverage** `P(true label in C) = P(α < p) ≥ 1 - α` under exchangeability, ties allowed. -/
theorem split_conformal_coverage (μ : Measure Ω) [IsProbabilityMeasure μ] {N : ℕ} [NeZero N]
    (S : Fin N → Ω → ℝ) (hS : ∀ i, Measurable (S i)) {α : ℝ} (hα0 : 0 ≤ α)
    (hexch : ∀ j : Fin N, μ {ω | rankBelow (fun i => S i ω) j < ⌈(N : ℝ) * (1 - α)⌉₊}
      = μ {ω | rankBelow (fun i => S i ω) 0 < ⌈(N : ℝ) * (1 - α)⌉₊}) :
    1 - α ≤ (μ {ω | α < pval (fun i => S i ω) 0}).toReal := by
  have hN : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have hm : ⌈(N : ℝ) * (1 - α)⌉₊ ≤ N := by
    rw [Nat.ceil_le]; nlinarith
  have h := conformal_coverage μ S hS _ hm hexch
  have hset : {ω | α < pval (fun i => S i ω) 0}
      = {ω | rankBelow (fun i => S i ω) 0 < ⌈(N : ℝ) * (1 - α)⌉₊} := by
    ext ω
    simp only [Set.mem_setOf_eq]
    rw [pval_gt_iff, Nat.lt_ceil]
  rw [hset]
  refine le_trans ?_ h
  rw [le_div_iff hN]
  have := Nat.le_ceil ((N : ℝ) * (1 - α))
  nlinarith

/-- Event-level exchangeability follows from permutation invariance of the joint law. -/
theorem exch_of_perm_invariant (μ : Measure Ω) {N : ℕ} [NeZero N] (S : Fin N → Ω → ℝ)
    (hS : ∀ i, Measurable (S i)) (m : ℕ)
    (hperm : ∀ (π : Equiv.Perm (Fin N)) (B : Set (Fin N → ℝ)), MeasurableSet B →
      μ ((fun ω i => S (π i) ω) ⁻¹' B) = μ ((fun ω i => S i ω) ⁻¹' B)) (j : Fin N) :
    μ {ω | rankBelow (fun i => S i ω) j < m} = μ {ω | rankBelow (fun i => S i ω) 0 < m} := by
  set π : Equiv.Perm (Fin N) := Equiv.swap 0 j with hπ
  set B : Set (Fin N → ℝ) := {a | rankBelow a 0 < m} with hB
  have hBm : MeasurableSet B := by
    have hmeas : Measurable (fun a : Fin N → ℝ => rankBelow a 0) := by
      have : (fun a : Fin N → ℝ => rankBelow a 0) = fun a => ∑ i, (if a i < a 0 then 1 else 0) := by
        funext a; unfold rankBelow; rw [Finset.card_filter]
      rw [this]
      refine Finset.measurable_sum _ (fun i _ => ?_)
      exact Measurable.ite (measurableSet_lt (measurable_pi_apply i) (measurable_pi_apply 0))
        measurable_const measurable_const
    exact hmeas (measurableSet_lt measurable_id measurable_const)
  have h1 := hperm π B hBm
  have e1 : (fun ω i => S (π i) ω) ⁻¹' B = {ω | rankBelow (fun i => S i ω) j < m} := by
    ext ω
    simp only [Set.mem_preimage, hB, Set.mem_setOf_eq]
    have : rankBelow (fun i => S (π i) ω) 0 = rankBelow (fun i => S i ω) j := by
      unfold rankBelow
      have hj : π 0 = j := by simp [hπ]
      refine Finset.card_congr (fun i _ => π i) (fun i hi => ?_) (fun i1 i2 _ _ h => π.injective h)
        (fun b hb => ⟨π.symm b, ?_, by simp⟩)
      · simpa [hj] using hi
      · simpa [hj] using hb
    rw [this]
  have e2 : (fun ω i => S i ω) ⁻¹' B = {ω | rankBelow (fun i => S i ω) 0 < m} := rfl
  rw [← e1, ← e2]
  exact h1

/-- L6′ newborn rule: the quantile rank is attainable (`≤ n`) iff `n ≥ 1/α - 1`. -/
theorem rank_le_iff {α : ℝ} (hα : 0 < α) (n : ℕ) :
    ⌈((n : ℝ) + 1) * (1 - α)⌉₊ ≤ n ↔ 1 / α - 1 ≤ n := by
  rw [Nat.ceil_le, div_sub_one hα.ne', div_le_iff hα]
  constructor <;> intro h <;> nlinarith

end FCPA

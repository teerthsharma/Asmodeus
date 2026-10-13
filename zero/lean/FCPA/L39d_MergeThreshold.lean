import FCPA.L24b_BirthEProcess
import FCPA.L39_MergeRule

/-!
L39 (threshold). The code merges when `BF > 1` (`δ = 1`, no guarantee: `L39.threshold_one_vacuous`). This file decides
what threshold can carry a finite-sample guarantee, and proves the strongest one that exists.

**Anytime guarantee (proved).** Test a forming era `D` against an old era `C` as `D`'s tokens arrive.
* `bf_eq_preq_ratio`: `BF(C, counts zs) = preq_C(zs) / preq_∅(zs)`, the ratio of the address loop's prequential
  products; `preqRatio_cons`: it is the running product of one-token ratios `pred_{C+D_t}(x) / pred_{D_t}(x)`.
* `pred_ratio_evalue`: each one-token ratio has mean exactly `1` under the separate-era predictive, so it is a
  conditional e-value, and the running BF is a nonnegative test martingale.
* `prodE_supermartingale`, `prodE_ville`: a running product of conditional e-values never reaches `1/δ` with
  probability more than `δ`, **at any time** (Ville; optional stopping is free).
* `merge_false_alarm`: with one test per (old era, new era) pair, indexed `j = 0, 1, 2, …` as pairs appear, the rule
  **merge iff `log BF ≥ log((j+1)(j+2)/δ)`** makes a false merge, of any pair, at any time, with probability `≤ δ`.
  The probability is under the separate-era model the code itself assumes (Dirichlet rows).

**No uniform guarantee (proved).** `no_uniform_merge_guarantee`: for any merge rule on any fixed sample sizes, the
merge probability is continuous in the two eras' laws. If the rule merges two same-law eras with probability
`≥ 1-β`, then eras with *different* laws close enough merge with probability `> 1-β-ε`. A guarantee that holds for every
pair of different laws cannot coexist with merging same-law eras: the prior-predictive guarantee above is the strongest
kind available. A uniform guarantee needs a separation margin `Δ` between laws (not formalised: needs method-of-types
concentration).
-/
open MeasureTheory BigOperators Finset
open scoped ENNReal

namespace FCPA
namespace L39d

/-! ### The running Bayes factor is a product of conditional e-values -/

section Combinatorial
open L39
variable {R X : Type*} [Fintype R] [Fintype X] [DecidableEq R] [DecidableEq X]

/-- The merge statistic is the ratio of the two prequential products the address loop computes. -/
theorem bf_eq_preq_ratio {a : ℝ} (ha : 0 < a) [Nonempty X] (C : R → X → ℕ) (zs : List (R × X)) :
    bf a C (counts zs) = preq a C zs / preq a 0 zs := by
  rw [bf_eq ha, preq_eq ha, preq_eq ha]
  rfl

/-- Running product of one-token ratios, era `C` (old) vs era `D` (forming). -/
noncomputable def preqRatio (a : ℝ) : (R → X → ℕ) → (R → X → ℕ) → List (R × X) → ℝ
  | _, _, [] => 1
  | C, D, z :: zs => pred a C z.1 z.2 / pred a D z.1 z.2 * preqRatio a (bump C z.1 z.2) (bump D z.1 z.2) zs

theorem pred_pos {a : ℝ} (ha : 0 < a) [Nonempty X] (C : R → X → ℕ) (r : R) (x : X) : 0 < pred a C r x := by
  have hV : (0 : ℝ) < Fintype.card X := Nat.cast_pos.mpr Fintype.card_pos
  unfold pred
  positivity

theorem preq_pos {a : ℝ} (ha : 0 < a) [Nonempty X] : ∀ (zs : List (R × X)) (C : R → X → ℕ), 0 < preq a C zs
  | [], _ => by simp [preq]
  | z :: zs, C => by
    simp only [preq]
    exact mul_pos (pred_pos ha _ _ _) (preq_pos ha zs _)

/-- The ratio of prequential products factors token by token. -/
theorem preqRatio_cons {a : ℝ} (ha : 0 < a) [Nonempty X] :
    ∀ (zs : List (R × X)) (C D : R → X → ℕ), preqRatio a C D zs = preq a C zs / preq a D zs
  | [], _, _ => by simp [preqRatio, preq]
  | z :: zs, C, D => by
    simp only [preqRatio, preq]
    rw [preqRatio_cons ha zs]
    have h1 := pred_pos ha D z.1 z.2
    have h2 := preq_pos ha zs (bump D z.1 z.2)
    field_simp

/-- So the merge statistic of `C` against the forming era is the running product of one-token ratios. -/
theorem bf_running_product {a : ℝ} (ha : 0 < a) [Nonempty X] (C : R → X → ℕ) (zs : List (R × X)) :
    bf a C (counts zs) = preqRatio a C 0 zs := by
  rw [bf_eq_preq_ratio ha, preqRatio_cons ha]

/-- **Each one-token ratio is a conditional e-value**: under the forming era's own predictive, its mean is `1`. -/
theorem pred_ratio_evalue {a : ℝ} (ha : 0 < a) [Nonempty X] (C D : R → X → ℕ) (r : R) :
    ∑ x, pred a D r x * (pred a C r x / pred a D r x) = 1 := by
  rw [← pred_sum_one ha C r]
  apply sum_congr rfl
  intro x _
  have := pred_pos ha D r x
  field_simp
  ring

end Combinatorial

/-! ### Anytime validity: Ville for a product of conditional e-values -/

section Measure
variable {Ω : Type*} {m0 : MeasurableSpace Ω} {μ : Measure Ω} {ℱ : Filtration ℕ m0}

/-- Running product `E_0 = 1`, `E_{t+1} = E_t · e_{t+1}`. -/
noncomputable def prodE (e : ℕ → Ω → ℝ) : ℕ → Ω → ℝ
  | 0 => fun _ => 1
  | t + 1 => fun ω => prodE e t ω * e (t + 1) ω

theorem prodE_nonneg {e : ℕ → Ω → ℝ} (hnn : ∀ t ω, 0 ≤ e t ω) (t : ℕ) (ω : Ω) : 0 ≤ prodE e t ω := by
  induction t with
  | zero => simp [prodE]
  | succ t ih => exact mul_nonneg ih (hnn _ _)

theorem prodE_meas {e : ℕ → Ω → ℝ} (hmeas : ∀ t, Measurable[ℱ t] (e t)) (t : ℕ) :
    Measurable[ℱ t] (prodE e t) := by
  induction t with
  | zero => exact measurable_const
  | succ t ih =>
    have h1 : Measurable[ℱ (t + 1)] (prodE e t) := ih.mono (ℱ.mono (Nat.le_succ t)) le_rfl
    exact h1.mul (hmeas (t + 1))

theorem prodE_integrable [IsFiniteMeasure μ] {e : ℕ → Ω → ℝ}
    (hmeas : ∀ t, Measurable[ℱ t] (e t)) (hnn : ∀ t ω, 0 ≤ e t ω)
    (hval : ∀ t s, MeasurableSet[ℱ t] s → ∫⁻ ω in s, ENNReal.ofReal (e (t + 1) ω) ∂μ ≤ μ s)
    (t : ℕ) : Integrable (prodE e t) μ := by
  induction t with
  | zero => simp [prodE]
  | succ t ih =>
    have hXm : Measurable[ℱ t] (prodE e t) := prodE_meas hmeas t
    have hXn : ∀ ω, 0 ≤ prodE e t ω := prodE_nonneg hnn t
    have hem : Measurable (e (t + 1)) := (hmeas (t + 1)).mono (ℱ.le _) le_rfl
    have hXm0 : Measurable (prodE e t) := hXm.mono (ℱ.le t) le_rfl
    have hε : Measurable (fun ω => ENNReal.ofReal (e (t + 1) ω)) := hem.ennreal_ofReal
    have hXe : Measurable[ℱ t] (fun ω => ENNReal.ofReal (prodE e t ω)) :=
      @Measurable.ennreal_ofReal Ω (ℱ t) _ hXm
    have key := lintegral_mul_le (ℱ := ℱ) (t := t) hε (hval t) hXe Set.univ MeasurableSet.univ
    simp only [Measure.restrict_univ] at key
    show Integrable (fun ω => prodE e t ω * e (t + 1) ω) μ
    refine ⟨(hXm0.mul hem).aestronglyMeasurable, ?_⟩
    rw [hasFiniteIntegral_iff_ofReal (ae_of_all _ (fun ω => mul_nonneg (hXn ω) (hnn _ _)))]
    have e1 : ∀ ω, ENNReal.ofReal (prodE e t ω * e (t + 1) ω)
        = ENNReal.ofReal (prodE e t ω) * ENNReal.ofReal (e (t + 1) ω) :=
      fun ω => ENNReal.ofReal_mul (hXn ω)
    simp_rw [e1]
    exact lt_of_le_of_lt key ((hasFiniteIntegral_iff_ofReal (ae_of_all _ hXn)).mp ih.hasFiniteIntegral)

theorem prodE_supermartingale [IsFiniteMeasure μ] {e : ℕ → Ω → ℝ}
    (hmeas : ∀ t, Measurable[ℱ t] (e t)) (hnn : ∀ t ω, 0 ≤ e t ω)
    (hval : ∀ t s, MeasurableSet[ℱ t] s → ∫⁻ ω in s, ENNReal.ofReal (e (t + 1) ω) ∂μ ≤ μ s) :
    Supermartingale (prodE e) ℱ μ := by
  refine supermartingale_of_set_integral_succ_le ?_ (prodE_integrable hmeas hnn hval) ?_
  · intro t
    exact (prodE_meas hmeas t).stronglyMeasurable
  · intro t s hs
    have hem : Measurable (e (t + 1)) := (hmeas (t + 1)).mono (ℱ.le _) le_rfl
    show ∫ ω in s, prodE e t ω * e (t + 1) ω ∂μ ≤ ∫ ω in s, prodE e t ω ∂μ
    exact set_integral_mul_le (ℱ := ℱ) (t := t) hem (hnn (t + 1)) (hval t) (prodE_meas hmeas t)
      (prodE_nonneg hnn t) (prodE_integrable hmeas hnn hval t) s hs

/-- **Anytime validity.** `P(∃ t, E_t ≥ 1/δ) ≤ δ`. -/
theorem prodE_ville [IsProbabilityMeasure μ] {e : ℕ → Ω → ℝ}
    (hmeas : ∀ t, Measurable[ℱ t] (e t)) (hnn : ∀ t ω, 0 ≤ e t ω)
    (hval : ∀ t s, MeasurableSet[ℱ t] s → ∫⁻ ω in s, ENNReal.ofReal (e (t + 1) ω) ∂μ ≤ μ s)
    {δ : ℝ} (hδ : 0 < δ) :
    μ {ω | ∃ t, 1 / δ ≤ prodE e t ω} ≤ ENNReal.ofReal δ := by
  have h := ville (prodE_supermartingale hmeas hnn hval) (fun t ω => prodE_nonneg hnn t ω)
    (one_div_pos.mpr hδ)
  have h1 : ∫ ω, prodE e 0 ω ∂μ = 1 := by simp [prodE]
  rw [h1] at h
  have hfin : μ {ω | ∃ t, 1 / δ ≤ prodE e t ω} ≠ ∞ := measure_ne_top _ _
  rw [← ENNReal.ofReal_toReal hfin]
  apply ENNReal.ofReal_le_ofReal
  set x := (μ {ω | ∃ t, 1 / δ ≤ prodE e t ω}).toReal
  calc x = δ * (1 / δ * x) := by field_simp; ring
    _ ≤ δ * 1 := mul_le_mul_of_nonneg_left h hδ.le
    _ = δ := mul_one δ

/-- **L39 threshold rule.** One running BF per (old era, forming era) pair, pairs indexed `j = 0, 1, …` as they
appear. Merging iff `BF_j ≥ (j+1)(j+2)/δ` merges some pair falsely, at some time, with probability `≤ δ`. -/
theorem merge_false_alarm [IsProbabilityMeasure μ] (e : ℕ → ℕ → Ω → ℝ)
    (hmeas : ∀ j t, Measurable[ℱ t] (e j t)) (hnn : ∀ j t ω, 0 ≤ e j t ω)
    (hval : ∀ j t s, MeasurableSet[ℱ t] s → ∫⁻ ω in s, ENNReal.ofReal (e j (t + 1) ω) ∂μ ≤ μ s)
    {δ : ℝ} (hδ : 0 < δ) :
    μ {ω | ∃ (j : ℕ) (t : ℕ), (((j : ℝ) + 1) * ((j : ℝ) + 2)) / δ ≤ prodE (e j) t ω} ≤ ENNReal.ofReal δ := by
  set E : ℕ → Set Ω := fun j => {ω | ∃ t, (((j : ℝ) + 1) * ((j : ℝ) + 2)) / δ ≤ prodE (e j) t ω} with hE
  have hset : {ω | ∃ (j : ℕ) (t : ℕ), (((j : ℝ) + 1) * ((j : ℝ) + 2)) / δ ≤ prodE (e j) t ω} = ⋃ j, E j := by
    ext ω; simp [hE]
  have hEj : ∀ j, μ (E j) ≤ ENNReal.ofReal (δ * (1 / (((j : ℝ) + 1) * ((j : ℝ) + 2)))) := by
    intro j
    have hδj : 0 < δ * (1 / (((j : ℝ) + 1) * ((j : ℝ) + 2))) := by positivity
    have h := prodE_ville (hmeas j) (hnn j) (hval j) hδj
    have e1 : 1 / (δ * (1 / (((j : ℝ) + 1) * ((j : ℝ) + 2)))) = (((j : ℝ) + 1) * ((j : ℝ) + 2)) / δ := by
      field_simp
    rw [e1] at h
    exact h
  rw [hset]
  calc μ (⋃ j, E j) ≤ ∑' j : ℕ, μ (E j) := measure_iUnion_le _
    _ ≤ ∑' j : ℕ, ENNReal.ofReal (δ * (1 / (((j : ℝ) + 1) * ((j : ℝ) + 2)))) :=
        ENNReal.tsum_le_tsum hEj
    _ = ENNReal.ofReal (∑' j : ℕ, δ * (1 / (((j : ℝ) + 1) * ((j : ℝ) + 2)))) := by
        rw [ENNReal.ofReal_tsum_of_nonneg (fun j => by positivity)
          (hasSum_inv_mul_succ.mul_left δ).summable]
    _ = ENNReal.ofReal δ := by
        rw [(hasSum_inv_mul_succ.mul_left δ).tsum_eq, mul_one]

end Measure

/-! ### No uniform guarantee over all pairs of different laws -/

section Impossible
variable {X : Type*} [Fintype X] [DecidableEq X]

/-- Probability that the merge rule fires (`A` = samples on which it merges), era `C` sampling `n` tokens from `θ.1`
and era `D` sampling `m` tokens from `θ.2`. -/
noncomputable def mergeProb (n m : ℕ) (A : Finset ((Fin n → X) × (Fin m → X))) (θ : (X → ℝ) × (X → ℝ)) : ℝ :=
  ∑ uw in A, (∏ i, θ.1 (uw.1 i)) * ∏ k, θ.2 (uw.2 k)

theorem mergeProb_continuous (n m : ℕ) (A : Finset ((Fin n → X) × (Fin m → X))) :
    Continuous (mergeProb n m A) := by
  unfold mergeProb
  refine continuous_finset_sum _ fun uw _ => Continuous.mul ?_ ?_
  · exact continuous_finset_prod _ fun i _ => (continuous_apply (uw.1 i)).comp continuous_fst
  · exact continuous_finset_prod _ fun k _ => (continuous_apply (uw.2 k)).comp continuous_snd

/-- **No uniform guarantee.** If a merge rule merges two eras of the same law `θ₀` with probability `≥ 1-β`, then for
every `ε > 0` there is a neighbourhood of `θ₀` such that eras with laws `θ₀` and any `θ_D` in it, different laws
included, merge with probability `> 1-β-ε`. -/
theorem no_uniform_merge_guarantee (n m : ℕ) (A : Finset ((Fin n → X) × (Fin m → X))) (θ₀ : X → ℝ) {β ε : ℝ}
    (hε : 0 < ε) (hpow : 1 - β ≤ mergeProb n m A (θ₀, θ₀)) :
    ∃ η > 0, ∀ θD : X → ℝ, dist θD θ₀ < η → 1 - β - ε < mergeProb n m A (θ₀, θD) := by
  have hc := (mergeProb_continuous n m A).continuousAt (x := (θ₀, θ₀))
  rcases Metric.continuousAt_iff.mp hc ε hε with ⟨η, hη, h⟩
  refine ⟨η, hη, fun θD hD => ?_⟩
  have hd : dist (θ₀, θD) (θ₀, θ₀) < η := by
    rw [Prod.dist_eq, dist_self]
    exact max_lt hη hD
  have := h hd
  rw [Real.dist_eq] at this
  have := neg_abs_le (mergeProb n m A (θ₀, θD) - mergeProb n m A (θ₀, θ₀))
  linarith [abs_lt.mp ‹|mergeProb n m A (θ₀, θD) - mergeProb n m A (θ₀, θ₀)| < ε›]

end Impossible

end L39d
end FCPA

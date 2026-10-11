import FCPA.L24b_BirthEProcess
import Mathlib.MeasureTheory.Integral.Layercake
import Mathlib.Analysis.SpecialFunctions.ImproperIntegrals

/-!
Power calibrator (Amendment B): a super-uniform p-value `p` (`ν{p ≤ u} ≤ u ν(univ)` for `u ∈ [0,1]`) gives an e-value
`e = κ p^{κ-1}`, `0 < κ < 1`: `∫⁻ ofReal e ≤ ν(univ)`. Proof by the layer-cake formula:
`∫⁻ κ f = κ ∫_{t>0} ν{f > t} dt` with `f = p^{κ-1}`, `{f > t} ⊆ {p ≤ t^{1/(κ-1)}}` for `t > 1`, and
`∫_1^∞ t^{1/(κ-1)} dt = (1-κ)/κ`. Applied with `ν = μ|_s` (`s ∈ F_t`) this is exactly the hypothesis `hval` of
`mixM_supermartingale`, so conditional super-uniformity of `p_t` suffices.
-/
open MeasureTheory Set
open scoped ENNReal

namespace FCPA

theorem rpow_tail_integrableOn {e : ℝ} (he : e < -1) :
    IntegrableOn (fun t : ℝ => t ^ e) (Ioi 1) :=
  integrableOn_Ioi_rpow_of_lt he (show (0 : ℝ) < 1 by norm_num)

theorem rpow_tail_integral {e : ℝ} (he : e < -1) :
    ∫ t : ℝ in Ioi 1, t ^ e = -1 / (e + 1) := by
  rw [integral_Ioi_rpow_of_lt he (show (0 : ℝ) < 1 by norm_num), Real.one_rpow]

theorem calibrator_integral_le {Ω : Type*} [MeasurableSpace Ω] (ν : Measure Ω) [IsFiniteMeasure ν]
    {p : Ω → ℝ} (hp : Measurable p) (hp0 : ∀ ω, 0 ≤ p ω) (hp1 : ∀ ω, p ω ≤ 1)
    (hsu : ∀ u : ℝ, 0 ≤ u → u ≤ 1 → ν {ω | p ω ≤ u} ≤ ENNReal.ofReal u * ν univ)
    {κ : ℝ} (hκ0 : 0 < κ) (hκ1 : κ < 1) :
    ∫⁻ ω, ENNReal.ofReal (κ * p ω ^ (κ - 1)) ∂ν ≤ ν univ := by
  obtain ⟨e, he⟩ : ∃ e : ℝ, e = 1 / (κ - 1) := ⟨_, rfl⟩
  have hκ1' : κ - 1 < 0 := by linarith
  have he_lt : e < -1 := by
    rw [he, div_lt_iff_of_neg hκ1']
    linarith
  set f : Ω → ℝ := fun ω => p ω ^ (κ - 1) with hf
  have hfm : Measurable f := hp.pow_const _
  have hfn : ∀ ω, 0 ≤ f ω := fun ω => Real.rpow_nonneg (hp0 ω) _
  -- layer cake with g ≡ κ
  have hlc := lintegral_comp_eq_lintegral_meas_lt_mul ν (f := f) (g := fun _ => κ)
    (ae_of_all _ hfn) hfm.aemeasurable
    (fun t _ => intervalIntegrable_const)
    (ae_of_all _ (fun _ => hκ0.le))
  have hlhs : ∀ ω, ENNReal.ofReal (∫ t in (0 : ℝ)..f ω, κ) = ENNReal.ofReal (κ * p ω ^ (κ - 1)) := by
    intro ω
    rw [intervalIntegral.integral_const]
    simp [hf, mul_comm]
  simp_rw [hlhs] at hlc
  rw [hlc]
  -- bound the measure of the super-level sets
  have hset : ∀ t : ℝ, 1 < t → ν {ω | t < f ω} ≤ ENNReal.ofReal (t ^ e) * ν univ := by
    intro t ht
    have ht0 : 0 < t := by linarith
    refine le_trans (measure_mono ?_) (hsu (t ^ e) (Real.rpow_nonneg ht0.le _)
      (Real.rpow_le_one_of_one_le_of_nonpos ht.le (by linarith)))
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    have hpos : 0 < p ω := by
      rcases (hp0 ω).eq_or_lt with h | h
      · have h0 : f ω = 0 := by
          simp only [hf]; rw [← h]; exact Real.zero_rpow hκ1'.ne
        linarith
      · exact h
    have h1 : (p ω ^ (κ - 1)) ^ e < t ^ e := by
      exact Real.rpow_lt_rpow_of_neg ht0 hω (by linarith)
    have hmul : (κ - 1) * e = 1 := by rw [he, mul_one_div, div_self hκ1'.ne]
    rw [← Real.rpow_mul hpos.le, hmul, Real.rpow_one] at h1
    exact h1.le
  -- split ∫_{Ioi 0}
  have hsplit : Ioi (0 : ℝ) = Ioc 0 1 ∪ Ioi 1 := by
    ext t; simp only [mem_Ioi, mem_union, mem_Ioc]
    constructor
    · intro h; by_cases h1 : t ≤ 1
      · exact Or.inl ⟨h, h1⟩
      · exact Or.inr (not_le.mp h1)
    · rintro (⟨h, _⟩ | h)
      · exact h
      · linarith
  have hdisj : Disjoint (Ioc (0 : ℝ) 1) (Ioi 1) := by
    rw [Set.disjoint_left]; intro t ht ht'; exact absurd ht'.out (not_lt.mpr ht.2)
  rw [hsplit, lintegral_union measurableSet_Ioi hdisj]
  have hA : ∫⁻ t in Ioc (0 : ℝ) 1, ν {ω | t < f ω} * ENNReal.ofReal κ ≤ ν univ * ENNReal.ofReal κ := by
    calc ∫⁻ t in Ioc (0 : ℝ) 1, ν {ω | t < f ω} * ENNReal.ofReal κ
        ≤ ∫⁻ t in Ioc (0 : ℝ) 1, ν univ * ENNReal.ofReal κ :=
          lintegral_mono (fun t => mul_le_mul_right' (measure_mono (subset_univ _)) _)
      _ = ν univ * ENNReal.ofReal κ := by simp
  have hint1 := rpow_tail_integrableOn he_lt
  have hval0 := rpow_tail_integral he_lt
  have hne : κ - 1 ≠ 0 := hκ1'.ne
  have h2 : e + 1 = κ / (κ - 1) := by
    rw [he, div_add_one hne]; congr 1; ring
  have hcalc : -1 / (e + 1) = (1 - κ) / κ := by
    rw [h2]; field_simp [hne, hκ0.ne']
  have hval1 := hval0.trans hcalc
  have hB : ∫⁻ t in Ioi (1 : ℝ), ν {ω | t < f ω} * ENNReal.ofReal κ
      ≤ ENNReal.ofReal ((1 - κ) / κ) * ν univ * ENNReal.ofReal κ := by
    calc ∫⁻ t in Ioi (1 : ℝ), ν {ω | t < f ω} * ENNReal.ofReal κ
        ≤ ∫⁻ t in Ioi (1 : ℝ), (ENNReal.ofReal (t ^ e) * ν univ) * ENNReal.ofReal κ :=
          set_lintegral_mono' measurableSet_Ioi (fun t ht =>
            mul_le_mul_right' (hset t ht) _)
      _ = (∫⁻ t in Ioi (1 : ℝ), ENNReal.ofReal (t ^ e)) * ν univ * ENNReal.ofReal κ := by
          have hg : Measurable (fun t : ℝ => ENNReal.ofReal (t ^ e)) :=
            ENNReal.measurable_ofReal.comp (measurable_id.pow_const e)
          rw [lintegral_mul_const _ (hg.mul_const _), lintegral_mul_const _ hg]
      _ = ENNReal.ofReal ((1 - κ) / κ) * ν univ * ENNReal.ofReal κ := by
          rw [← ofReal_integral_eq_lintegral_ofReal hint1
            ((ae_restrict_iff' measurableSet_Ioi).mpr (ae_of_all _ (fun t ht =>
              Real.rpow_nonneg (by have : (1 : ℝ) < t := ht; linarith) _))), hval1]
  refine le_trans (add_le_add hA hB) (le_of_eq ?_)
  have hq0 : 0 ≤ (1 - κ) / κ := div_nonneg (by linarith) hκ0.le
  have hsum : ENNReal.ofReal κ + ENNReal.ofReal ((1 - κ) / κ) * ENNReal.ofReal κ = 1 := by
    rw [← ENNReal.ofReal_mul hq0, ← ENNReal.ofReal_add hκ0.le (mul_nonneg hq0 hκ0.le)]
    have : κ + (1 - κ) / κ * κ = 1 := by field_simp
    rw [this]; simp
  calc ν univ * ENNReal.ofReal κ + ENNReal.ofReal ((1 - κ) / κ) * ν univ * ENNReal.ofReal κ
      = ν univ * (ENNReal.ofReal κ + ENNReal.ofReal ((1 - κ) / κ) * ENNReal.ofReal κ) := by ring
    _ = ν univ := by rw [hsum, mul_one]

/-- Conditional version: if `p` is super-uniform on every `s ∈ F_t`, then `κ p^{κ-1}` is a conditional e-value. -/
theorem calibrator_cond {Ω : Type*} {m0 : MeasurableSpace Ω} (μ : Measure Ω) [IsFiniteMeasure μ]
    {p : Ω → ℝ} (hp : Measurable p) (hp0 : ∀ ω, 0 ≤ p ω) (hp1 : ∀ ω, p ω ≤ 1)
    {s : Set Ω}
    (hsu : ∀ u : ℝ, 0 ≤ u → u ≤ 1 → μ ({ω | p ω ≤ u} ∩ s) ≤ ENNReal.ofReal u * μ s)
    {κ : ℝ} (hκ0 : 0 < κ) (hκ1 : κ < 1) :
    ∫⁻ ω in s, ENNReal.ofReal (κ * p ω ^ (κ - 1)) ∂μ ≤ μ s := by
  have hsu' : ∀ u : ℝ, 0 ≤ u → u ≤ 1 →
      (μ.restrict s) {ω | p ω ≤ u} ≤ ENNReal.ofReal u * (μ.restrict s) univ := by
    intro u hu0 hu1
    have hm : MeasurableSet {ω | p ω ≤ u} := hp measurableSet_Iic
    rw [Measure.restrict_apply hm, Measure.restrict_apply_univ]
    exact hsu u hu0 hu1
  have h := calibrator_integral_le (μ.restrict s) hp hp0 hp1 hsu' hκ0 hκ1
  rwa [Measure.restrict_apply_univ] at h

/-- **L24\* from p-values.** Conditionally super-uniform p-values `p_{j,t}` (per era `j`, adapted to `ℱ`) fed through the
power calibrator `e = κ p^{κ-1}` and the mixture e-process give: with probability `≥ 1 - δ` no era ever fires a false
birth at level `δ_j = δ/((j+1)(j+2))`. -/
theorem birth_false_alarm_pvalues {Ω : Type*} {m0 : MeasurableSpace Ω} {μ : Measure Ω}
    [IsProbabilityMeasure μ] {ℱ : Filtration ℕ m0} (p : ℕ → ℕ → Ω → ℝ)
    (hpm : ∀ j t, Measurable[ℱ t] (p j t)) (hp0 : ∀ j t ω, 0 ≤ p j t ω) (hp1 : ∀ j t ω, p j t ω ≤ 1)
    (hsu : ∀ j t s, MeasurableSet[ℱ t] s → ∀ u : ℝ, 0 ≤ u → u ≤ 1 →
      μ ({ω | p j (t + 1) ω ≤ u} ∩ s) ≤ ENNReal.ofReal u * μ s)
    {κ : ℝ} (hκ0 : 0 < κ) (hκ1 : κ < 1) {δ : ℝ} (hδ : 0 < δ) :
    μ {ω | ∃ (j : ℕ) (t : ℕ), (((j : ℝ) + 1) * ((j : ℝ) + 2)) / δ ≤
        mixM (fun t ω => κ * p j t ω ^ (κ - 1)) t ω} ≤ ENNReal.ofReal δ := by
  refine birth_false_alarm (ℱ := ℱ) (fun j t ω => κ * p j t ω ^ (κ - 1)) ?_ ?_ ?_ hδ
  · intro j t
    exact ((hpm j t).pow_const _).const_mul κ
  · intro j t ω
    exact mul_nonneg hκ0.le (Real.rpow_nonneg (hp0 j t ω) _)
  · intro j t s hs
    exact calibrator_cond μ ((hpm j (t + 1)).mono (ℱ.le _) le_rfl) (hp0 j (t + 1)) (hp1 j (t + 1))
      (hsu j t s hs) hκ0 hκ1

end FCPA

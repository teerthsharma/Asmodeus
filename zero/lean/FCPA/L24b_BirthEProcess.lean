import FCPA.L24s_Ville
import Mathlib.Probability.Martingale.Basic

/-!
L24* birth-specific guarantee (spec Amendment B, ERRATA 14).
The mixture-over-change-time e-process
`S_0 = 0`, `S_t = (S_{t-1} + w_t) · e_t`, `w_k = 1/(k(k+1))`, `M_t = S_t + 1/(t+1)`
is a nonnegative supermartingale with `M_0 = 1`, provided each `e_t` is a conditional e-value
(`E[e_t | F_{t-1}] ≤ 1`; stated as `∫⁻_s ofReal e_t ≤ μ s` for every `F_{t-1}`-measurable `s`, which is the same
thing). This is the case for `e_t = κ p_t^{κ-1}` with `p_t` conditionally super-uniform (`calibrator_integral_le`
in `L24c_Calibrator`). Composing with Ville (`L24s_Ville`) and the union bound over `δ_j = δ/((j+1)(j+2))` gives
`P(some era j ever has M^{(j)}_t ≥ 1/δ_j) ≤ δ`: no false birth, ever.
-/
open MeasureTheory BigOperators
open scoped ENNReal

namespace FCPA

variable {Ω : Type*} {m0 : MeasurableSpace Ω} {μ : Measure Ω} {ℱ : Filtration ℕ m0}

/-- Key step: an `F_t`-measurable `X ≥ 0` can be pulled out of a conditional-expectation bound `E[ε | F_t] ≤ 1`. -/
theorem lintegral_mul_le {t : ℕ} {ε : Ω → ℝ≥0∞} (hε : Measurable ε)
    (hv : ∀ s, MeasurableSet[ℱ t] s → ∫⁻ ω in s, ε ω ∂μ ≤ μ s)
    {X : Ω → ℝ≥0∞} (hX : Measurable[ℱ t] X) :
    ∀ s, MeasurableSet[ℱ t] s → ∫⁻ ω in s, X ω * ε ω ∂μ ≤ ∫⁻ ω in s, X ω ∂μ := by
  refine @Measurable.ennreal_induction Ω (ℱ t)
    (fun X => ∀ s, MeasurableSet[ℱ t] s → ∫⁻ ω in s, X ω * ε ω ∂μ ≤ ∫⁻ ω in s, X ω ∂μ)
    ?_ ?_ ?_ X hX
  · intro c A hA s hs
    have hA' : MeasurableSet A := ℱ.le t _ hA
    have e1 : ∀ ω, (A.indicator (fun _ => c)) ω * ε ω = A.indicator (fun ω => c * ε ω) ω := by
      intro ω; by_cases h : ω ∈ A <;> simp [Set.indicator, h]
    simp_rw [e1]
    rw [lintegral_indicator _ hA', Measure.restrict_restrict hA', lintegral_const_mul c hε,
      lintegral_indicator_const hA', Measure.restrict_apply hA']
    exact mul_le_mul_left' (hv _ (hA.inter hs)) c |>.trans_eq (by rw [Set.inter_comm])
  · intro f g _ hf hg hPf hPg s hs
    have hf' : Measurable f := hf.mono (ℱ.le t) le_rfl
    have hg' : Measurable g := hg.mono (ℱ.le t) le_rfl
    simp only [Pi.add_apply, add_mul]
    rw [lintegral_add_left (hf'.mul hε), lintegral_add_left hf']
    exact add_le_add (hPf s hs) (hPg s hs)
  · intro f hf hmono hP s hs
    have hf' : ∀ n, Measurable (f n) := fun n => (hf n).mono (ℱ.le t) le_rfl
    have e2 : ∀ ω, (⨆ n, f n ω) * ε ω = ⨆ n, f n ω * ε ω := fun ω => ENNReal.iSup_mul
    simp_rw [e2]
    rw [lintegral_iSup (fun n => (hf' n).mul hε) (fun n m h ω => mul_le_mul_right' (hmono h ω) _),
      lintegral_iSup hf' hmono]
    exact iSup_mono (fun n => hP n s hs)

/-- Mixture weights `w_k = 1/(k(k+1))`. -/
noncomputable def mixW (k : ℕ) : ℝ := 1 / ((k : ℝ) * ((k : ℝ) + 1))

/-- The mixture statistic `S_t`. -/
noncomputable def mixS (e : ℕ → Ω → ℝ) : ℕ → Ω → ℝ
  | 0 => fun _ => 0
  | t + 1 => fun ω => (mixS e t ω + mixW (t + 1)) * e (t + 1) ω

/-- The mixture e-process `M_t = S_t + 1/(t+1)`. -/
noncomputable def mixM (e : ℕ → Ω → ℝ) (t : ℕ) : Ω → ℝ := fun ω => mixS e t ω + 1 / ((t : ℝ) + 1)

theorem mixW_nonneg (k : ℕ) : 0 ≤ mixW k := by unfold mixW; positivity

theorem mixS_nonneg {e : ℕ → Ω → ℝ} (hnn : ∀ t ω, 0 ≤ e t ω) (t : ℕ) (ω : Ω) : 0 ≤ mixS e t ω := by
  induction t with
  | zero => simp [mixS]
  | succ t ih => exact mul_nonneg (add_nonneg ih (mixW_nonneg _)) (hnn _ _)

theorem mixS_meas {e : ℕ → Ω → ℝ} (hmeas : ∀ t, Measurable[ℱ t] (e t)) (t : ℕ) :
    Measurable[ℱ t] (mixS e t) := by
  induction t with
  | zero => exact measurable_const
  | succ t ih =>
    have h1 : Measurable[ℱ (t + 1)] (mixS e t) := ih.mono (ℱ.mono (Nat.le_succ t)) le_rfl
    exact (h1.add measurable_const).mul (hmeas (t + 1))

theorem mixS_integrable [IsFiniteMeasure μ] {e : ℕ → Ω → ℝ}
    (hmeas : ∀ t, Measurable[ℱ t] (e t)) (hnn : ∀ t ω, 0 ≤ e t ω)
    (hval : ∀ t s, MeasurableSet[ℱ t] s → ∫⁻ ω in s, ENNReal.ofReal (e (t + 1) ω) ∂μ ≤ μ s)
    (t : ℕ) : Integrable (mixS e t) μ := by
  induction t with
  | zero => simp [mixS]
  | succ t ih =>
    have hX : Integrable (fun ω => mixS e t ω + mixW (t + 1)) μ := ih.add (integrable_const _)
    have hXm : Measurable[ℱ t] (fun ω => mixS e t ω + mixW (t + 1)) :=
      (mixS_meas hmeas t).add measurable_const
    have hXn : ∀ ω, 0 ≤ mixS e t ω + mixW (t + 1) :=
      fun ω => add_nonneg (mixS_nonneg hnn t ω) (mixW_nonneg _)
    have hem : Measurable (e (t + 1)) := (hmeas (t + 1)).mono (ℱ.le _) le_rfl
    have hXm0 : Measurable (fun ω => mixS e t ω + mixW (t + 1)) := hXm.mono (ℱ.le t) le_rfl
    have hε : Measurable (fun ω => ENNReal.ofReal (e (t + 1) ω)) := hem.ennreal_ofReal
    have hXe : Measurable[ℱ t] (fun ω => ENNReal.ofReal (mixS e t ω + mixW (t + 1))) :=
      @Measurable.ennreal_ofReal Ω (ℱ t) _ hXm
    have key := lintegral_mul_le (ℱ := ℱ) (t := t) hε (hval t) hXe Set.univ MeasurableSet.univ
    simp only [Measure.restrict_univ] at key
    show Integrable (fun ω => (mixS e t ω + mixW (t + 1)) * e (t + 1) ω) μ
    refine ⟨(hXm0.mul hem).aestronglyMeasurable, ?_⟩
    rw [hasFiniteIntegral_iff_ofReal (ae_of_all _ (fun ω =>
      mul_nonneg (hXn ω) (hnn _ _)))]
    have e1 : ∀ ω, ENNReal.ofReal ((mixS e t ω + mixW (t + 1)) * e (t + 1) ω)
        = ENNReal.ofReal (mixS e t ω + mixW (t + 1)) * ENNReal.ofReal (e (t + 1) ω) :=
      fun ω => ENNReal.ofReal_mul (hXn ω)
    simp_rw [e1]
    exact lt_of_le_of_lt key ((hasFiniteIntegral_iff_ofReal
      (ae_of_all _ hXn)).mp hX.hasFiniteIntegral)

theorem set_integral_mul_le [IsFiniteMeasure μ] {t : ℕ} {e1 X : Ω → ℝ}
    (he1m : Measurable e1) (he1n : ∀ ω, 0 ≤ e1 ω)
    (hv : ∀ s, MeasurableSet[ℱ t] s → ∫⁻ ω in s, ENNReal.ofReal (e1 ω) ∂μ ≤ μ s)
    (hXm : Measurable[ℱ t] X) (hXn : ∀ ω, 0 ≤ X ω) (hXi : Integrable X μ)
    (s : Set Ω) (hs : MeasurableSet[ℱ t] s) :
    ∫ ω in s, X ω * e1 ω ∂μ ≤ ∫ ω in s, X ω ∂μ := by
  have hXm0 : Measurable X := hXm.mono (ℱ.le t) le_rfl
  have hε : Measurable (fun ω => ENNReal.ofReal (e1 ω)) := he1m.ennreal_ofReal
  have hXe : Measurable[ℱ t] (fun ω => ENNReal.ofReal (X ω)) :=
    @Measurable.ennreal_ofReal Ω (ℱ t) _ hXm
  have key := lintegral_mul_le (ℱ := ℱ) (t := t) hε hv hXe s hs
  have hfin : ∫⁻ ω in s, ENNReal.ofReal (X ω) ∂μ ≠ ∞ := hXi.integrableOn.lintegral_lt_top.ne
  rw [integral_eq_lintegral_of_nonneg_ae (ae_of_all _ (fun ω => mul_nonneg (hXn ω) (he1n ω)))
      (hXm0.mul he1m).aestronglyMeasurable,
    integral_eq_lintegral_of_nonneg_ae (ae_of_all _ hXn) hXm0.aestronglyMeasurable]
  apply ENNReal.toReal_mono hfin
  have e1 : ∀ ω, ENNReal.ofReal (X ω * e1 ω) = ENNReal.ofReal (X ω) * ENNReal.ofReal (e1 ω) :=
    fun ω => ENNReal.ofReal_mul (hXn ω)
  simp_rw [e1]
  exact key

theorem mixM_supermartingale [IsFiniteMeasure μ] {e : ℕ → Ω → ℝ}
    (hmeas : ∀ t, Measurable[ℱ t] (e t)) (hnn : ∀ t ω, 0 ≤ e t ω)
    (hval : ∀ t s, MeasurableSet[ℱ t] s → ∫⁻ ω in s, ENNReal.ofReal (e (t + 1) ω) ∂μ ≤ μ s) :
    Supermartingale (mixM e) ℱ μ := by
  have hint : ∀ t, Integrable (mixM e t) μ := fun t =>
    (mixS_integrable hmeas hnn hval t).add (integrable_const _)
  refine supermartingale_of_set_integral_succ_le ?_ hint ?_
  · intro t
    exact ((mixS_meas hmeas t).add measurable_const).stronglyMeasurable
  · intro t s hs
    have hXi : Integrable (fun ω => mixS e t ω + mixW (t + 1)) μ :=
      (mixS_integrable hmeas hnn hval t).add (integrable_const _)
    have hXm : Measurable[ℱ t] (fun ω => mixS e t ω + mixW (t + 1)) :=
      (mixS_meas hmeas t).add measurable_const
    have hXn : ∀ ω, 0 ≤ mixS e t ω + mixW (t + 1) :=
      fun ω => add_nonneg (mixS_nonneg hnn t ω) (mixW_nonneg _)
    have hem : Measurable (e (t + 1)) := (hmeas (t + 1)).mono (ℱ.le _) le_rfl
    have hM1 : mixM e (t + 1) = fun ω =>
        (mixS e t ω + mixW (t + 1)) * e (t + 1) ω + 1 / ((t : ℝ) + 2) := by
      funext ω; simp only [mixM, mixS]
      rw [show (((t + 1 : ℕ) : ℝ) + 1) = (t : ℝ) + 2 by push_cast; ring]
      rfl
    have hM0 : mixM e t = fun ω => (mixS e t ω + mixW (t + 1)) + 1 / ((t : ℝ) + 2) := by
      funext ω; simp only [mixM, mixW]; push_cast; field_simp; ring
    have hXe : Integrable (fun ω => (mixS e t ω + mixW (t + 1)) * e (t + 1) ω) μ :=
      mixS_integrable hmeas hnn hval (t + 1)
    rw [hM1, hM0, integral_add hXe.integrableOn (integrable_const _).integrableOn,
      integral_add hXi.integrableOn (integrable_const _).integrableOn]
    have := set_integral_mul_le (ℱ := ℱ) (t := t) hem (hnn (t + 1)) (hval t) hXm hXn hXi s hs
    linarith

/-- The mixture e-process starts at `M_0 = 1`. -/
theorem mixM_zero (e : ℕ → Ω → ℝ) : mixM e 0 = fun _ => 1 := by
  funext ω; simp [mixM, mixS]

theorem mixM_nonneg {e : ℕ → Ω → ℝ} (hnn : ∀ t ω, 0 ≤ e t ω) : 0 ≤ mixM e := by
  intro t ω
  have := mixS_nonneg hnn t ω
  simp only [mixM, Pi.zero_apply]
  positivity

/-- Ville for the mixture e-process: `P(∃ t, M_t ≥ c) ≤ 1/c`. -/
theorem mixM_ville [IsProbabilityMeasure μ] {e : ℕ → Ω → ℝ}
    (hmeas : ∀ t, Measurable[ℱ t] (e t)) (hnn : ∀ t ω, 0 ≤ e t ω)
    (hval : ∀ t s, MeasurableSet[ℱ t] s → ∫⁻ ω in s, ENNReal.ofReal (e (t + 1) ω) ∂μ ≤ μ s)
    {c : ℝ} (hc : 0 < c) :
    c * (μ {ω | ∃ t, c ≤ mixM e t ω}).toReal ≤ 1 := by
  have h := ville (mixM_supermartingale hmeas hnn hval) (mixM_nonneg hnn) hc
  have h1 : ∫ ω, mixM e 0 ω ∂μ = 1 := by
    rw [mixM_zero]; simp
  rw [h1] at h
  exact h

theorem hasSum_inv_mul_succ : HasSum (fun j : ℕ => 1 / (((j : ℝ) + 1) * ((j : ℝ) + 2))) 1 := by
  have hnn : ∀ j : ℕ, 0 ≤ 1 / (((j : ℝ) + 1) * ((j : ℝ) + 2)) := fun j => by positivity
  rw [hasSum_iff_tendsto_nat_of_nonneg hnn]
  have hpart : ∀ N : ℕ, ∑ j in Finset.range N, 1 / (((j : ℝ) + 1) * ((j : ℝ) + 2))
      = 1 - 1 / ((N : ℝ) + 1) := by
    intro N
    induction N with
    | zero => simp
    | succ N ih =>
      rw [Finset.sum_range_succ, ih]
      push_cast
      field_simp
      ring
  simp_rw [hpart]
  have := (tendsto_one_div_add_atTop_nhds_zero_nat : Filter.Tendsto (fun n : ℕ => 1 / ((n : ℝ) + 1)) Filter.atTop (nhds 0))
  simpa using this.const_sub 1

/-- **L24\* birth guarantee.** For a family of e-processes (one per era `j`), each built from conditional e-values,
the probability that any era's mixture e-process ever reaches `1/δ_j`, `δ_j = δ/((j+1)(j+2))`, is at most `δ`:
with probability `≥ 1 - δ` there is no false birth, ever. -/
theorem birth_false_alarm [IsProbabilityMeasure μ] (e : ℕ → ℕ → Ω → ℝ)
    (hmeas : ∀ j t, Measurable[ℱ t] (e j t)) (hnn : ∀ j t ω, 0 ≤ e j t ω)
    (hval : ∀ j t s, MeasurableSet[ℱ t] s →
      ∫⁻ ω in s, ENNReal.ofReal (e j (t + 1) ω) ∂μ ≤ μ s)
    {δ : ℝ} (hδ : 0 < δ) :
    μ {ω | ∃ (j : ℕ) (t : ℕ), (((j : ℝ) + 1) * ((j : ℝ) + 2)) / δ ≤ mixM (e j) t ω} ≤ ENNReal.ofReal δ := by
  set E : ℕ → Set Ω := fun j => {ω | ∃ t, (((j : ℝ) + 1) * ((j : ℝ) + 2)) / δ ≤ mixM (e j) t ω}
    with hE
  have hset : {ω | ∃ (j : ℕ) (t : ℕ), (((j : ℝ) + 1) * ((j : ℝ) + 2)) / δ ≤ mixM (e j) t ω} = ⋃ j, E j := by
    ext ω; simp [hE]
  have hEj : ∀ j, μ (E j) ≤ ENNReal.ofReal (δ * (1 / (((j : ℝ) + 1) * ((j : ℝ) + 2)))) := by
    intro j
    have hpos : 0 < (((j : ℝ) + 1) * ((j : ℝ) + 2)) / δ := by positivity
    have h := mixM_ville (hmeas j) (hnn j) (hval j) hpos
    have hfin : μ (E j) ≠ ∞ := measure_ne_top _ _
    rw [← ENNReal.ofReal_toReal hfin]
    apply ENNReal.ofReal_le_ofReal
    have hq : 0 < (((j : ℝ) + 1) * ((j : ℝ) + 2)) := by positivity
    rw [div_mul_eq_mul_div, div_le_one hδ] at h
    rw [mul_one_div, le_div_iff hq]
    nlinarith [h]
  rw [hset]
  calc μ (⋃ j, E j) ≤ ∑' j : ℕ, μ (E j) := measure_iUnion_le _
    _ ≤ ∑' j : ℕ, ENNReal.ofReal (δ * (1 / (((j : ℝ) + 1) * ((j : ℝ) + 2)))) :=
        ENNReal.tsum_le_tsum hEj
    _ = ENNReal.ofReal (∑' j : ℕ, δ * (1 / (((j : ℝ) + 1) * ((j : ℝ) + 2)))) := by
        rw [ENNReal.ofReal_tsum_of_nonneg (fun j => by positivity)
          (hasSum_inv_mul_succ.mul_left δ).summable]
    _ = ENNReal.ofReal δ := by
        rw [(hasSum_inv_mul_succ.mul_left δ).tsum_eq, mul_one]

end FCPA

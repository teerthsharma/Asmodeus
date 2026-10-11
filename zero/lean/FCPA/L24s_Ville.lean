import Mathlib.Probability.Martingale.OptionalStopping

/-!
L24* (Amendment B, Tier B): Ville's inequality for a nonnegative supermartingale, the anytime-valid birth trigger.
`c · P(∃ n, M_n ≥ c) ≤ E[M_0]`. Proved here (v4.7.0 Mathlib has Doob's maximal inequality only for submartingales):
optional stopping on `-M` (a submartingale) at the hitting time of `[c, ∞)` gives the finite-horizon bound, then continuity
from below of the measure gives the infinite-horizon one. With `c = 1/δ`: `P(sup_n M_n ≥ 1/δ) ≤ δ · E[M_0]`.
-/
open MeasureTheory Filter Set
open scoped ENNReal

namespace FCPA

variable {Ω : Type*} {m0 : MeasurableSpace Ω} {μ : Measure Ω} [IsFiniteMeasure μ]
  {𝒢 : Filtration ℕ m0} {M : ℕ → Ω → ℝ}

/-- Finite-horizon Ville: `c · P(∃ k ≤ N, c ≤ M_k) ≤ E[M_0]`. -/
theorem ville_finite (hM : Supermartingale M 𝒢 μ) (hnn : 0 ≤ M) {c : ℝ} (hc : 0 < c) (N : ℕ) :
    c * (μ {ω | ∃ k ≤ N, c ≤ M k ω}).toReal ≤ ∫ ω, M 0 ω ∂μ := by
  set τ : Ω → ℕ := hitting M {y : ℝ | c ≤ y} 0 N with hτdef
  have hτ : IsStoppingTime 𝒢 τ := hitting_isStoppingTime hM.adapted measurableSet_Ici
  have hτN : ∀ ω, τ ω ≤ N := fun ω => hitting_le ω
  have e1 : stoppedValue (-M) (fun _ => 0) = -(M 0) := by
    ext ω; simp [stoppedValue]
  have e2 : stoppedValue (-M) τ = -(stoppedValue M τ) := by
    ext ω; simp [stoppedValue]
  have hopt : ∫ ω, stoppedValue M τ ω ∂μ ≤ ∫ ω, M 0 ω ∂μ := by
    have h := (hM.neg).expected_stoppedValue_mono (isStoppingTime_const 𝒢 0) hτ
      (fun _ => Nat.zero_le _) hτN
    rw [e1, e2] at h
    simp only [Pi.neg_apply, integral_neg] at h
    linarith
  have hint : Integrable (stoppedValue M τ) μ := by
    have := ((hM.neg).integrable_stoppedValue hτ hτN).neg
    rw [e2] at this
    simpa using this
  set A : Set Ω := {ω | ∃ k ≤ N, c ≤ M k ω} with hAdef
  have hAmeas : MeasurableSet A := by
    have : A = ⋃ k ∈ {k : ℕ | k ≤ N}, {ω | c ≤ M k ω} := by
      ext ω; simp [hAdef]
    rw [this]
    refine MeasurableSet.biUnion (Set.to_countable _) (fun k _ => ?_)
    exact measurableSet_le measurable_const ((hM.stronglyMeasurable k).measurable.le (𝒢.le k))
  have hA : ∀ ω ∈ A, c ≤ stoppedValue M τ ω := by
    intro ω hω
    obtain ⟨k, hk, hck⟩ := hω
    have : stoppedValue M τ ω ∈ {y : ℝ | c ≤ y} :=
      stoppedValue_hitting_mem ⟨k, ⟨Nat.zero_le _, hk⟩, hck⟩
    exact this
  have h1 := set_integral_ge_of_const_le hAmeas (measure_ne_top _ _) hA hint.integrableOn
  have h2 : ∫ ω in A, stoppedValue M τ ω ∂μ ≤ ∫ ω, stoppedValue M τ ω ∂μ :=
    set_integral_le_integral hint (ae_of_all _ fun ω => hnn _ _)
  linarith

/-- **Ville's inequality** for a nonnegative supermartingale. -/
theorem ville {Ω : Type*} {m0 : MeasurableSpace Ω} {μ : Measure Ω} [IsFiniteMeasure μ]
    {𝒢 : Filtration ℕ m0} {M : ℕ → Ω → ℝ} (hM : Supermartingale M 𝒢 μ) (hnn : 0 ≤ M)
    {c : ℝ} (hc : 0 < c) :
    c * (μ {ω | ∃ n, c ≤ M n ω}).toReal ≤ ∫ ω, M 0 ω ∂μ := by
  set A : ℕ → Set Ω := fun N => {ω | ∃ k ≤ N, c ≤ M k ω} with hA
  have hmono : Monotone A := fun N N' h ω ⟨k, hk, hck⟩ => ⟨k, hk.trans h, hck⟩
  have hU : {ω | ∃ n, c ≤ M n ω} = ⋃ N, A N := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, hA]
    exact ⟨fun ⟨n, h⟩ => ⟨n, n, le_rfl, h⟩, fun ⟨_, k, _, h⟩ => ⟨k, h⟩⟩
  have hI := fun N => ville_finite hM hnn hc N
  set I : ℝ := ∫ ω, M 0 ω ∂μ with hIdef
  have hlim : Tendsto (fun N => μ (A N)) atTop (nhds (μ (⋃ N, A N))) :=
    tendsto_measure_iUnion hmono
  have hle : ∀ N, ENNReal.ofReal c * μ (A N) ≤ ENNReal.ofReal I := by
    intro N
    have h := hI N
    have hfin : μ (A N) ≠ ∞ := measure_ne_top _ _
    rw [← ENNReal.ofReal_toReal hfin, ← ENNReal.ofReal_mul hc.le]
    exact ENNReal.ofReal_le_ofReal h
  have hlim2 : ENNReal.ofReal c * μ (⋃ N, A N) ≤ ENNReal.ofReal I :=
    le_of_tendsto' (ENNReal.Tendsto.const_mul hlim (Or.inr ENNReal.ofReal_ne_top)) hle
  rw [hU]
  have hfin : μ (⋃ N, A N) ≠ ∞ := measure_ne_top _ _
  have hI0 : 0 ≤ I := by
    have := hI 0
    have h0 : 0 ≤ c * (μ (A 0)).toReal := by positivity
    linarith
  rw [← ENNReal.ofReal_toReal hfin, ← ENNReal.ofReal_mul hc.le] at hlim2
  exact (ENNReal.ofReal_le_ofReal_iff hI0).mp hlim2

end FCPA

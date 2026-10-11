import Mathlib.MeasureTheory.Integral.Bochner

/-!
L18* floor (Amendment A, with Chase's hypothesis). Let `ζ ≤ 1` be the zoom gain and suppose `E[ζ] ≥ s`, where
`s = Σ_{i ≤ d_c} λ_i(ρ_k)`. (Ky Fan only gives `E[ζ] ≤ s`; equality, hence this hypothesis, holds when `V_k` is the
top-`d_c` eigenspace of `ρ_k`.) Then for `ζ₀ < 1`, `P(ζ ≤ ζ₀) ≤ (1 - s)/(1 - ζ₀)` (Markov on `1 - ζ`).
The Ky Fan ceiling L18 itself is OPEN here (no Ky Fan principle in v4.7.0 Mathlib).
-/
open MeasureTheory

namespace FCPA

theorem zeta_floor {Ω : Type*} {m0 : MeasurableSpace Ω} {μ : Measure Ω} [IsProbabilityMeasure μ]
    {ζ : Ω → ℝ} (hint : Integrable ζ μ) (hle : ∀ ω, ζ ω ≤ 1) {s ζ0 : ℝ}
    (hs : s ≤ ∫ ω, ζ ω ∂μ) (hζ0 : ζ0 < 1) :
    (μ {ω | ζ ω ≤ ζ0}).toReal ≤ (1 - s) / (1 - ζ0) := by
  have hpos : 0 < 1 - ζ0 := by linarith
  have hint' : Integrable (fun ω => 1 - ζ ω) μ := (integrable_const (1 : ℝ)).sub hint
  have hnn : 0 ≤ᵐ[μ] fun ω => 1 - ζ ω := ae_of_all _ (fun ω => by simp only [Pi.zero_apply]; linarith [hle ω])
  have hmk := mul_meas_ge_le_integral_of_nonneg hnn hint' (1 - ζ0)
  have hset : {ω | 1 - ζ0 ≤ 1 - ζ ω} = {ω | ζ ω ≤ ζ0} := by
    ext ω; simp only [Set.mem_setOf_eq]; constructor <;> intro h <;> linarith
  rw [hset] at hmk
  have hI : ∫ ω, (1 - ζ ω) ∂μ = 1 - ∫ ω, ζ ω ∂μ := by
    rw [integral_sub (integrable_const _) hint]; simp
  rw [hI] at hmk
  rw [le_div_iff hpos]
  nlinarith [hmk]

end FCPA

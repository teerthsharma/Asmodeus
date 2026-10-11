import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
L22† (orbit-local certification, Amendment B). The worst-case global Lipschitz constant of the CTC map is vacuous at
practical settings, so certify along the actual orbit: if every step of the orbit contracts the previous step by `c < 1`
(e.g. the Jacobian norm along the orbit is `≤ c`), then in a complete space the orbit converges to a fixed point of the
(continuous) map, geometrically, with an a-priori iteration budget. This is the `contract` branch of the dispatcher.
The Krasnosel'skii–Mann branch (averaging for merely nonexpansive maps) is OPEN (see CHANGES.md).
-/
open Filter Topology

namespace FCPA

variable {X : Type*} [MetricSpace X]

/-- Orbit step bound: `dist (u (n+1)) (u n) ≤ d₀ cⁿ` from the orbit-local contraction. -/
theorem orbit_step_le (u : ℕ → X) (c : ℝ) (hc0 : 0 ≤ c)
    (hc : ∀ n, dist (u (n + 2)) (u (n + 1)) ≤ c * dist (u (n + 1)) (u n)) (n : ℕ) :
    dist (u (n + 1)) (u n) ≤ dist (u 1) (u 0) * c ^ n := by
  induction n with
  | zero => simp
  | succ n ih =>
    calc dist (u (n + 1 + 1)) (u (n + 1)) ≤ c * dist (u (n + 1)) (u n) := hc n
      _ ≤ c * (dist (u 1) (u 0) * c ^ n) := mul_le_mul_of_nonneg_left ih hc0
      _ = dist (u 1) (u 0) * c ^ (n + 1) := by ring

/-- **L22† (contract branch).** In a complete metric space, a continuous map `f` whose orbit `uₙ₊₁ = f uₙ` satisfies the
orbit-local contraction `dist uₙ₊₂ uₙ₊₁ ≤ c · dist uₙ₊₁ uₙ`, `c < 1`, has an orbit converging to a fixed point `x*`,
with `dist uₙ x* ≤ d₀ cⁿ/(1-c)` and residual `dist uₙ (f uₙ) ≤ d₀ cⁿ`, where `d₀ = dist (f u₀) u₀`. -/
theorem orbit_local_converges [CompleteSpace X] (f : X → X) (hf : Continuous f) (u : ℕ → X)
    (hu : ∀ n, u (n + 1) = f (u n)) (c : ℝ) (hc0 : 0 ≤ c) (hc1 : c < 1)
    (hc : ∀ n, dist (u (n + 2)) (u (n + 1)) ≤ c * dist (u (n + 1)) (u n)) :
    ∃ x, f x = x ∧ Tendsto u atTop (𝓝 x) ∧
      (∀ n, dist (u n) x ≤ dist (u 1) (u 0) * c ^ n / (1 - c)) ∧
      (∀ n, dist (u n) (f (u n)) ≤ dist (u 1) (u 0) * c ^ n) := by
  have hstep := orbit_step_le u c hc0 hc
  have hstep' : ∀ n, dist (u n) (u (n + 1)) ≤ dist (u 1) (u 0) * c ^ n := fun n => by
    rw [dist_comm]; exact hstep n
  have hcau : CauchySeq u := cauchySeq_of_le_geometric c (dist (u 1) (u 0)) hc1 hstep'
  obtain ⟨x, hx⟩ := cauchySeq_tendsto_of_complete hcau
  have hfx : f x = x := by
    have h1 : Tendsto (fun n => u (n + 1)) atTop (𝓝 x) := hx.comp (tendsto_add_atTop_nat 1)
    have h2 : Tendsto (fun n => u (n + 1)) atTop (𝓝 (f x)) := by
      simp_rw [hu]
      exact (hf.tendsto x).comp hx
    exact tendsto_nhds_unique h2 h1
  refine ⟨x, hfx, hx, fun n => ?_, fun n => ?_⟩
  · have := dist_le_of_le_geometric_of_tendsto c (dist (u 1) (u 0)) hc1 hstep' hx n
    simpa [mul_div_assoc] using this
  · rw [← hu n]; exact hstep' n

/-- A-priori iteration budget: with `0 < c < 1`, `d₀ > 0` and `ε > 0`, every `n ≥ ⌈log(ε/d₀)/log c⌉` has
`d₀ cⁿ ≤ ε`. -/
theorem iteration_budget {c d0 ε : ℝ} (hc0 : 0 < c) (hc1 : c < 1) (hd : 0 < d0) (hε : 0 < ε) (n : ℕ)
    (hn : Real.log (ε / d0) / Real.log c ≤ n) : d0 * c ^ n ≤ ε := by
  have hlog : Real.log c < 0 := Real.log_neg hc0 hc1
  have h1 : (n : ℝ) * Real.log c ≤ Real.log (ε / d0) := by
    have := mul_le_mul_of_nonpos_right hn hlog.le
    rwa [div_mul_cancel₀ _ hlog.ne] at this
  have h2 : c ^ n ≤ ε / d0 := by
    rw [← Real.log_le_log_iff (pow_pos hc0 n) (div_pos hε hd), Real.log_pow]
    exact h1
  rw [le_div_iff hd] at h2
  linarith

end FCPA

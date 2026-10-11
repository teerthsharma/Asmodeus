import Mathlib.MeasureTheory.Measure.MeasureSpace

/-!
L11 Certified change soundness (spec §3.3, Chase form). Under H0 (both windows share the true law `θ`), if each window's
confidence set misses `θ` with probability ≤ δ, then P(the two sets are disjoint, i.e. a change is certified) ≤ 2δ. Union bound; no
independence needed. Overlap means REFUSE, never "no change".
-/
open MeasureTheory

namespace FCPA

theorem certify_sound {Ω X : Type*} [MeasurableSpace Ω] (μ : Measure Ω) (IA IB : Ω → Set X) (θ : X)
    {δ : ENNReal} (hA : μ {ω | θ ∉ IA ω} ≤ δ) (hB : μ {ω | θ ∉ IB ω} ≤ δ) :
    μ {ω | Disjoint (IA ω) (IB ω)} ≤ 2 * δ := by
  calc μ {ω | Disjoint (IA ω) (IB ω)}
      ≤ μ ({ω | θ ∉ IA ω} ∪ {ω | θ ∉ IB ω}) := by
        apply measure_mono
        intro ω hω
        by_contra hc
        simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_not] at hc hω
        exact Set.disjoint_left.mp hω hc.1 hc.2
    _ ≤ μ {ω | θ ∉ IA ω} + μ {ω | θ ∉ IB ω} := measure_union_le _ _
    _ ≤ δ + δ := add_le_add hA hB
    _ = 2 * δ := (two_mul δ).symm

end FCPA

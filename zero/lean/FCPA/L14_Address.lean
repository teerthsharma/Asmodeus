import Mathlib.Topology.MetricSpace.Basic

/-!
L14 Address (spec §3.4, Chase form): fingerprints `κ` pairwise `s`-separated, estimate `x` within `< s/2` of `κ k` ⇒ the nearest
fingerprint is `κ k`. (The tie-break between eras sharing a law needs the query to carry its index; see L14′ in `L14b`.)
-/

namespace FCPA

theorem address {X ι : Type*} [MetricSpace X] (κ : ι → X) (s : ℝ)
    (hsep : ∀ j k, j ≠ k → s ≤ dist (κ j) (κ k)) (x : X) (k : ι) (hx : dist x (κ k) < s / 2) :
    ∀ j, j ≠ k → dist x (κ k) < dist x (κ j) := by
  intro j hj
  have h1 := hsep k j (Ne.symm hj)
  have h2 := dist_triangle (κ k) x (κ j)
  rw [dist_comm (κ k) x] at h2
  linarith

end FCPA

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Group.Abs
import Mathlib.Data.Finset.Basic
import Mathlib.Tactic.Linarith

/-!
L5 Certified top-m (spec §3.2): if the score boxes `[D_i ± R_i]` of `S` strictly dominate those outside,
then `S` is the top set of every true score vector in the box. (REFUSE is the policy when the hypothesis fails; the converse is false.)
-/

namespace FCPA

theorem certified_topm {ι : Type*} (S : Finset ι) (D R s : ι → ℝ)
    (hsep : ∀ i ∈ S, ∀ j ∉ S, D j + R j < D i - R i)
    (hs : ∀ i, |s i - D i| ≤ R i) :
    ∀ i ∈ S, ∀ j ∉ S, s j < s i := by
  intro i hi j hj
  have h1 := abs_le.mp (hs i)
  have h2 := abs_le.mp (hs j)
  have h3 := hsep i hi j hj
  linarith [h1.1, h2.2]

end FCPA

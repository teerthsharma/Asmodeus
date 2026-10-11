import Mathlib.Algebra.Group.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
L13 Cocycle (spec §3.4): `ΔE_{i→k} = ΔE_{i→j} + ΔE_{j→k}` for additive law coordinates (any abelian group), and for the
log-ratio chart of a positive Markov law.
-/

namespace FCPA

theorem cocycle {G : Type*} [AddCommGroup G] (a b c : G) : (b - a) + (c - b) = c - a := by
  abel

theorem cocycle_logratio {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    Real.log (b / a) + Real.log (c / b) = Real.log (c / a) := by
  rw [Real.log_div hb.ne' ha.ne', Real.log_div hc.ne' hb.ne', Real.log_div hc.ne' ha.ne']
  ring

end FCPA

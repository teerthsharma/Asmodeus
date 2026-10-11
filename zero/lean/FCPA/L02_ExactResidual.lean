import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Mathlib.Algebra.Order.Field.Basic

/-! L2 Exact residual (spec §3.1): the residual `1 - ∑_{T} p` is the exact mass outside `T` and is nonnegative. -/
open BigOperators

namespace FCPA

theorem residual {ι : Type*} [Fintype ι] [DecidableEq ι] (p : ι → ℝ) (h0 : ∀ i, 0 ≤ p i)
    (h1 : ∑ i, p i = 1) (S : Finset ι) :
    1 - ∑ i in S, p i = ∑ i in Sᶜ, p i ∧ 0 ≤ 1 - ∑ i in S, p i := by
  have hc := Finset.sum_compl_add_sum S p
  have hn : 0 ≤ ∑ i in Sᶜ, p i := Finset.sum_nonneg (fun i _ => h0 i)
  constructor <;> linarith

end FCPA

import Mathlib.Order.WellFounded
import Mathlib.Logic.Relation

/-!
L25′ Grounding / anti-collapse (spec §3.10, Chase form). A dataflow DAG: nodes `ι` with a well-founded `edge j i` ("j feeds i");
each node computes `val i = F i val` where `F i` reads only predecessors. If node `i` has no `TT` ancestor (including itself), and two
dataflows agree on every non-TT node's local rule, then they assign `i` the same value, i.e. the era training data is invariant to
whatever the TT nodes (the TT's outputs, depth, recursion) compute. This is a MODELLING statement: it holds exactly when the
dataflow graph really has no TT -> era-training edge (birth/slab selection must be a function of data and D4 alone, see Chase L25).
-/

namespace FCPA

theorem grounding {ι β : Type*} (edge : ι → ι → Prop) (wf : WellFounded edge) (TT : ι → Prop)
    (F₁ F₂ : ι → (ι → β) → β)
    (loc₁ : ∀ i f g, (∀ j, edge j i → f j = g j) → F₁ i f = F₁ i g)
    (loc₂ : ∀ i f g, (∀ j, edge j i → f j = g j) → F₂ i f = F₂ i g)
    (same : ∀ i, ¬ TT i → F₁ i = F₂ i)
    (v₁ v₂ : ι → β) (h₁ : ∀ i, v₁ i = F₁ i v₁) (h₂ : ∀ i, v₂ i = F₂ i v₂)
    (i : ι) (hclean : ∀ j, Relation.ReflTransGen edge j i → ¬ TT j) : v₁ i = v₂ i := by
  induction i using wf.induction with
  | _ i ih =>
    have hnt : ¬ TT i := hclean i Relation.ReflTransGen.refl
    rw [h₁ i, h₂ i, ← same i hnt]
    apply loc₁
    intro j hj
    exact ih j hj (fun k hk => hclean k (hk.trans (Relation.ReflTransGen.single hj)))

/-! ### L25′ provenance tags (Amendment A): no TT-generated datum ever reaches era training -/

/-- Provenance of a datum. -/
inductive Prov | grounded | generated
  deriving DecidableEq

/-- A tagged datum. -/
structure Datum (α : Type*) where
  val : α
  prov : Prov

/-- Births and updates admit only grounded data. -/
def admit {α : Type*} (xs : List (Datum α)) : List (Datum α) :=
  xs.filter (fun d => decide (d.prov = Prov.grounded))

theorem admit_grounded {α : Type*} (xs : List (Datum α)) : ∀ d ∈ admit xs, d.prov = Prov.grounded := by
  intro d hd
  simpa [admit] using (List.mem_filter.mp hd).2

/-- Appending any amount of TT-generated data leaves the admitted (era-training) data unchanged. -/
theorem admit_append_generated {α : Type*} (xs ys : List (Datum α))
    (hy : ∀ d ∈ ys, d.prov = Prov.generated) : admit (xs ++ ys) = admit xs := by
  unfold admit
  rw [List.filter_append]
  have : ys.filter (fun d => decide (d.prov = Prov.grounded)) = [] := by
    apply List.filter_eq_nil.mpr
    intro d hd
    simp [hy d hd]
  rw [this, List.append_nil]

end FCPA

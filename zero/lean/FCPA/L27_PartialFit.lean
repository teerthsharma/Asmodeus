import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Mathlib.Algebra.Order.Field.Basic

/-!
L27′ Exact `partial_fit` (spec §3.11, Chase's correction + Amendment A).
The spec's L27 is FALSE for transition counts: chunks `[a]`, `[b]` lose the `a -> b` boundary transition.
Correct state = (counts, first element, last element), the buffer. `merge` of the states of two chunks equals the state of the
concatenation, so `partial_fit` over chunks = `fit` on the concatenation (in exact arithmetic).
Gram matrices / normal equations are plain sums and are exactly additive (`gram_append`).
-/
open BigOperators

namespace FCPA

variable {α : Type*} [DecidableEq α]

/-- Consecutive pairs of a list. -/
def trans : List α → List (α × α)
  | [] => []
  | [_] => []
  | x :: y :: t => (x, y) :: trans (y :: t)

-- explicit BEq: the default instance on α × α is instBEqProd, which Mathlib's List.count_append does not match
def count (xs : List α) (a b : α) : ℕ := @List.count (α × α) (instBEq : BEq (α × α)) (a, b) (trans xs)

def bnd (xs ys : List α) : List (α × α) :=
  match xs.getLast?, ys.head? with
  | some a, some b => [(a, b)]
  | _, _ => []

theorem trans_append (xs ys : List α) : trans (xs ++ ys) = trans xs ++ bnd xs ys ++ trans ys := by
  induction xs with
  | nil => simp [trans, bnd]
  | cons x t ih =>
    cases t with
    | nil => cases ys <;> simp [trans, bnd]
    | cons y r =>
      have ih' : trans (y :: (r ++ ys)) = trans (y :: r) ++ bnd (y :: r) ys ++ trans ys := ih
      simp [trans, bnd, ih']

theorem count_bnd (xs ys : List α) (a b : α) :
    @List.count (α × α) (instBEq : BEq (α × α)) (a, b) (bnd xs ys) = if xs.getLast? = some a ∧ ys.head? = some b then 1 else 0 := by
  unfold bnd
  rcases xs.getLast? with _ | a' <;> rcases ys.head? with _ | b'
  · simp
  · simp
  · simp
  · by_cases h1 : a' = a <;> by_cases h2 : b' = b <;> simp [List.count_cons, h1, h2, eq_comm]

theorem count_append (xs ys : List α) (a b : α) :
    count (xs ++ ys) a b = count xs a b + count ys a b +
      (if xs.getLast? = some a ∧ ys.head? = some b then 1 else 0) := by
  unfold count
  rw [trans_append, List.count_append, List.count_append, count_bnd]
  ring

structure State (α : Type*) where
  cnt : α → α → ℕ
  first : Option α
  last : Option α

def State.ofList (xs : List α) : State α := ⟨count xs, xs.head?, xs.getLast?⟩

def State.merge (s t : State α) : State α :=
  ⟨fun a b => s.cnt a b + t.cnt a b + (if s.last = some a ∧ t.first = some b then 1 else 0),
   (s.first <|> t.first), (t.last <|> s.last)⟩

theorem head?_append' (xs ys : List α) : (xs ++ ys).head? = (xs.head? <|> ys.head?) := by
  cases xs <;> simp

theorem getLast?_append' (xs ys : List α) :
    (xs ++ ys).getLast? = (ys.getLast? <|> xs.getLast?) := by
  rcases ys with _ | ⟨y, t⟩
  · simp
  · rw [List.getLast?_append_of_ne_nil xs (List.cons_ne_nil y t)]
    obtain ⟨z, hz⟩ := Option.isSome_iff_exists.mp (List.getLast?_isSome.mpr (List.cons_ne_nil y t))
    simp [hz]

theorem state_append (xs ys : List α) :
    State.ofList (xs ++ ys) = (State.ofList xs).merge (State.ofList ys) := by
  have hc : count (xs ++ ys) = fun a b => count xs a b + count ys a b +
      (if xs.getLast? = some a ∧ ys.head? = some b then 1 else 0) := by
    funext a b
    exact count_append xs ys a b
  simp only [State.ofList, State.merge, hc, head?_append', getLast?_append']

theorem count_not_additive :
    ∃ (xs ys : List ℕ) (a b : ℕ), count (xs ++ ys) a b ≠ count xs a b + count ys a b := by
  refine ⟨[0], [1], 0, 1, ?_⟩
  decide

theorem gram_append {γ : Type*} (f : γ → ℝ) (xs ys : List γ) :
    ((xs ++ ys).map f).sum = (xs.map f).sum + (ys.map f).sum := by
  rw [List.map_append, List.sum_append]

/-! ### The full L27′ state: sufficient statistics, w-sample buffer, open-segment statistics

Mirror of `fcpa/estimator.py::_State`. A stream is cut into windows of `w` samples; a window is consumed by a deterministic
`step` that updates the closed-segment statistics `S` and the open-segment statistics `O`. The incomplete tail (fewer than
`w` samples) is the buffer. `partial_fit` consumes `buf ++ chunk`. -/

section Stream

variable {β S O : Type*}

/-- Cut a stream into complete windows of size `w` and the remainder (the w-sample buffer). -/
def split (w : ℕ) : List β → List (List β) × List β
  | xs =>
    if h : 0 < w ∧ w ≤ xs.length then
      let r := split w (xs.drop w)
      ((xs.take w) :: r.1, r.2)
    else ([], xs)
termination_by xs => xs.length
decreasing_by
  rename_i xs0
  have h2 : w ≤ xs0.length := h.2
  simp_wf
  omega

theorem split_pos (w : ℕ) (xs : List β) (h : 0 < w ∧ w ≤ xs.length) :
    split w xs = ((xs.take w) :: (split w (xs.drop w)).1, (split w (xs.drop w)).2) := by
  rw [split]
  simp [h]

theorem split_neg (w : ℕ) (xs : List β) (h : ¬ (0 < w ∧ w ≤ xs.length)) :
    split w xs = ([], xs) := by
  rw [split]
  simp [h]

theorem split_append (w : ℕ) (hw : 0 < w) (xs ys : List β) :
    split w (xs ++ ys) = ((split w xs).1 ++ (split w ((split w xs).2 ++ ys)).1,
                          (split w ((split w xs).2 ++ ys)).2) := by
  induction' hn : xs.length using Nat.strong_induction_on with n ih generalizing xs
  by_cases h : w ≤ xs.length
  · have h1 : w ≤ (xs ++ ys).length := by rw [List.length_append]; omega
    have e1 : (xs ++ ys).take w = xs.take w := List.take_append_of_le_length h
    have e2 : (xs ++ ys).drop w = xs.drop w ++ ys := List.drop_append_of_le_length h
    have hlen : (xs.drop w).length < n := by rw [List.length_drop]; omega
    have ih' := ih _ hlen (xs.drop w) rfl
    rw [split_pos w (xs ++ ys) ⟨hw, h1⟩, e1, e2, ih', split_pos w xs ⟨hw, h⟩]
    simp
  · have h0 : ¬ (0 < w ∧ w ≤ xs.length) := fun h' => h h'.2
    rw [split_neg w xs h0]
    simp

/-- State after consuming a stream: statistics, open-segment statistics, buffer. -/
structure PFState (S O β : Type*) where
  stats : S
  opn : O
  buf : List β

/-- Consume the complete windows with `step`. -/
def consume (step : S × O → List β → S × O) (so : S × O) (ws : List (List β)) : S × O :=
  ws.foldl step so

/-- `fit` on a whole stream. -/
def fitS (w : ℕ) (step : S × O → List β → S × O) (init : S × O) (xs : List β) : PFState S O β :=
  let r := split w xs
  let so := consume step init r.1
  ⟨so.1, so.2, r.2⟩

/-- `partial_fit` of one chunk: consume `buffer ++ chunk`, keep the new remainder. -/
def partialFitS (w : ℕ) (step : S × O → List β → S × O) (st : PFState S O β) (chunk : List β) :
    PFState S O β :=
  let r := split w (st.buf ++ chunk)
  let so := consume step (st.stats, st.opn) r.1
  ⟨so.1, so.2, r.2⟩

/-- **Exact `partial_fit` (L27′)**: with the buffer and the open-segment statistics in the state,
`partial_fit (fit xs) ys = fit (xs ++ ys)`. -/
theorem partialFit_exact (w : ℕ) (hw : 0 < w) (step : S × O → List β → S × O) (init : S × O)
    (xs ys : List β) :
    partialFitS w step (fitS w step init xs) ys = fitS w step init (xs ++ ys) := by
  unfold partialFitS fitS consume
  simp only [split_append w hw xs ys, List.foldl_append]

/-- Any chunking of the stream gives the same state as `fit` on the concatenation. -/
theorem partialFit_chunks (w : ℕ) (hw : 0 < w) (step : S × O → List β → S × O) (init : S × O)
    (xs : List β) (chunks : List (List β)) :
    chunks.foldl (partialFitS w step) (fitS w step init xs) = fitS w step init (xs ++ chunks.join) := by
  induction chunks generalizing xs with
  | nil => simp
  | cons c cs ih =>
    rw [List.foldl_cons, partialFit_exact w hw, ih]
    simp [List.join, List.append_assoc]

/-- Without the buffer (dropping the remainder after every chunk) the result differs: windows 2, one sample per chunk. -/
theorem no_buffer_inexact :
    ∃ (step : ℕ × Unit → List ℕ → ℕ × Unit),
      (fitS 2 step (0, ()) ([0] ++ [1])).stats = 1 ∧
      ((fitS 2 step (0, ()) [0]).stats + (fitS 2 step (0, ()) [1]).stats = 0) := by
  refine ⟨fun so _ => (so.1 + 1, ()), ?_, ?_⟩ <;>
    simp [fitS, consume, split_pos, split_neg]

end Stream

/-! ### Normal-equation statistics (Gram matrix and `Xᵀy`) are exactly additive over chunks -/

section Gram

variable {ι : Type*} [Fintype ι]

/-- `XᵀX = Σ x xᵀ`. -/
def gramList (xs : List (ι → ℝ)) : Matrix ι ι ℝ := (xs.map fun x => Matrix.vecMulVec x x).sum

/-- `Xᵀy = Σ y x`. -/
def xtyList (xs : List ((ι → ℝ) × ℝ)) : ι → ℝ := (xs.map fun p => p.2 • p.1).sum

theorem gram_append_matrix (xs ys : List (ι → ℝ)) :
    gramList (xs ++ ys) = gramList xs + gramList ys := by
  simp [gramList, List.map_append, List.sum_append]

theorem xty_append (xs ys : List ((ι → ℝ) × ℝ)) :
    xtyList (xs ++ ys) = xtyList xs + xtyList ys := by
  simp [xtyList, List.map_append, List.sum_append]

end Gram

end FCPA

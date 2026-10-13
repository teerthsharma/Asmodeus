import Mathlib.Data.List.Basic
import Mathlib.Data.List.Count
import Mathlib.Analysis.SpecificLimits.Basic

/-!
L40 (research program, Addendum 2): **no forgetting** for count eras, as a corollary of non-destructive birth (L23′ for
count tables instead of effects).

Model of `research/e0/arms.py::fcpa` bookkeeping. Every transition the bank has seen is a ledger token
`(time, context, symbol, era)`; era `ℓ`'s count table is the number of ledger tokens labelled `ℓ` per transition.
Ops (all the code does to the tables):
* `credit z` — a new token is counted in its era (`eras[k][prev, cur] += 1`);
* `birth new` — a fresh, empty era (no token changes);
* `fold i j` — era `j` merged into era `i` (`eras[i] += eras[j]`; tokens of `j` relabelled `i`);
* `readdress τ b` — the address step: tokens after the change-time estimate `τ` are taken out of their eras and
  re-credited to era `b` (`eras[kk][p, c] -= 1`, then `eras[best][p, c] += 1`).

A token is *settled* at `τ₀` if its time is `≤ τ₀`; the code's `det.tmin` is always `≥` the last decision time, so every
token seen before a decision is settled for all later re-addresses.

* `count_birth_old_unchanged` — birth leaves every old table (hence every old predictive) exactly unchanged (L23′ analogue).
* `settled_kept` — no op ever loses a settled token: after any op sequence whose re-addresses start at `τ ≥ τ₀`, the
  settled tokens reappear, each relabelled only by the composed merge map `Φ`.
* `no_forgetting` — counts: `table_after(Φ ℓ) ≥ settled counts of ℓ before`, for every era `ℓ` and transition.
* `untouched_era_unchanged` — exact zero forgetting: an era that no op touches (no credit, no merge, no re-address into
  it, all its tokens settled) has the same table, hence the same predictive, after any excursion of any length.
* `ema_forgets` — contrast: an exponentially decayed table keeps `λ^L` of an era's counts after `L` steps away (`→ 0`).
The filter side of re-entry (the era's weight never drops below `α/N`, so returning to it costs at most `log(N/α)`) is
`L38.window_regret`.
-/
namespace FCPA
namespace L40

/-- One credited transition. -/
structure Tok (L C X : Type*) where
  time : ℕ
  ctx : C
  sym : X
  era : L

variable {L C X : Type*} [DecidableEq L] [DecidableEq C] [DecidableEq X]

/-- Relabel a token's era. -/
def Tok.reera (z : Tok L C X) (ℓ : L) : Tok L C X := { z with era := ℓ }

@[simp] theorem Tok.reera_era (z : Tok L C X) (ℓ : L) : (z.reera ℓ).era = ℓ := rfl
@[simp] theorem Tok.reera_time (z : Tok L C X) (ℓ : L) : (z.reera ℓ).time = z.time := rfl
@[simp] theorem Tok.reera_ctx (z : Tok L C X) (ℓ : L) : (z.reera ℓ).ctx = z.ctx := rfl
@[simp] theorem Tok.reera_sym (z : Tok L C X) (ℓ : L) : (z.reera ℓ).sym = z.sym := rfl
@[simp] theorem Tok.reera_self (z : Tok L C X) : z.reera z.era = z := by cases z; rfl
@[simp] theorem Tok.reera_reera (z : Tok L C X) (ℓ m : L) : (z.reera ℓ).reera m = z.reera m := rfl

/-- Era `ℓ`'s count table. -/
def table (led : List (Tok L C X)) (ℓ : L) (r : C) (x : X) : ℕ :=
  led.countP (fun z => decide (z.era = ℓ ∧ z.ctx = r ∧ z.sym = x))

/-- Bank operations on the ledger. -/
inductive LOp (L C X : Type*)
  | credit (z : Tok L C X)
  | birth (new : L)
  | fold (i j : L)
  | readdress (τ : ℕ) (b : L)

namespace LOp

def relabel : LOp L C X → L → L
  | fold i j => fun k => if k = j then i else k
  | _ => id

def apply : LOp L C X → List (Tok L C X) → List (Tok L C X)
  | credit z, led => led ++ [z]
  | birth _, led => led
  | fold i j, led => led.map (fun z => z.reera (if z.era = j then i else z.era))
  | readdress τ b, led => led.map (fun z => if τ < z.time then z.reera b else z)

/-- Re-addresses never reach back before `τ₀`. -/
def Floor (τ₀ : ℕ) : LOp L C X → Prop
  | readdress τ _ => τ₀ ≤ τ
  | _ => True

end LOp

def applyAll : List (LOp L C X) → List (Tok L C X) → List (Tok L C X)
  | [], led => led
  | o :: os, led => applyAll os (o.apply led)

def relabelAll : List (LOp L C X) → L → L
  | [], k => k
  | o :: os, k => relabelAll os (o.relabel k)

/-- Settled at `τ₀`. -/
def settled (τ₀ : ℕ) (z : Tok L C X) : Bool := decide (z.time ≤ τ₀)

/-- Relabel a token by a label map. -/
def reeraBy (f : L → L) (z : Tok L C X) : Tok L C X := z.reera (f z.era)

theorem map_reeraBy_of_id (f : L → L) (hf : ∀ k, f k = k) :
    ∀ l : List (Tok L C X), l.map (reeraBy f) = l
  | [] => rfl
  | z :: zs => by simp [reeraBy, hf, map_reeraBy_of_id f hf zs]

theorem countP_settled (τ₀ : ℕ) (P : Tok L C X → Prop) [DecidablePred P] :
    ∀ led : List (Tok L C X), led.countP (fun z => decide (P z ∧ z.time ≤ τ₀)) =
      (led.filter (settled τ₀)).countP (fun z => decide (P z))
  | [] => rfl
  | z :: zs => by
    rw [List.countP_cons, List.filter_cons, countP_settled τ₀ P zs]
    by_cases h : z.time ≤ τ₀ <;> by_cases hp : P z <;> simp [settled, h, hp, List.countP_cons]

/-- **Birth is non-destructive** for count eras: every table, hence every era predictive, is unchanged. -/
theorem count_birth_old_unchanged (led : List (Tok L C X)) (new ℓ : L) (r : C) (x : X) :
    table ((LOp.birth new : LOp L C X).apply led) ℓ r x = table led ℓ r x := rfl

theorem settled_reeraBy (τ₀ : ℕ) (f : L → L) : settled τ₀ ∘ reeraBy (C := C) (X := X) f = settled τ₀ := by
  funext z; rfl

/-- One op keeps every settled token (relabelled by the op's merge map). -/
theorem op_settled_kept (τ₀ : ℕ) (o : LOp L C X) (ho : o.Floor τ₀) (led : List (Tok L C X)) :
    ((led.filter (settled τ₀)).map (reeraBy o.relabel)).Sublist ((o.apply led).filter (settled τ₀)) := by
  cases o with
  | credit z =>
    simp only [LOp.apply, LOp.relabel, List.filter_append]
    rw [map_reeraBy_of_id id (fun _ => rfl)]
    exact List.sublist_append_left _ _
  | birth new =>
    simp only [LOp.apply, LOp.relabel]
    rw [map_reeraBy_of_id id (fun _ => rfl)]
  | fold i j =>
    simp only [LOp.apply, LOp.relabel]
    rw [List.map_filter]
    have : settled τ₀ ∘ (fun z : Tok L C X => z.reera (if z.era = j then i else z.era)) = settled τ₀ := by
      funext z; rfl
    rw [this]
    exact List.Sublist.refl _
  | readdress τ b =>
    simp only [LOp.Floor] at ho
    simp only [LOp.apply, LOp.relabel]
    rw [List.map_filter]
    have h1 : settled τ₀ ∘ (fun z : Tok L C X => if τ < z.time then z.reera b else z) = settled τ₀ := by
      funext z; simp only [Function.comp]; split_ifs <;> rfl
    rw [h1]
    have h2 : (led.filter (settled τ₀)).map (fun z : Tok L C X => if τ < z.time then z.reera b else z) =
        (led.filter (settled τ₀)).map (reeraBy id) := by
      apply List.map_congr
      intro z hz
      have hz' : z.time ≤ τ₀ := by
        have := (List.mem_filter.mp hz).2
        simpa [settled] using this
      have : ¬ τ < z.time := by omega
      simp [this, reeraBy]
    rw [h2, map_reeraBy_of_id id (fun _ => rfl)]

/-- **Settled tokens are never lost** through any sequence of credits, births, merges and re-addresses. -/
theorem settled_kept (τ₀ : ℕ) : ∀ (os : List (LOp L C X)), (∀ o ∈ os, o.Floor τ₀) → ∀ (led : List (Tok L C X)),
    ((led.filter (settled τ₀)).map (reeraBy (relabelAll os))).Sublist ((applyAll os led).filter (settled τ₀))
  | [], _, led => by
    rw [map_reeraBy_of_id (relabelAll ([] : List (LOp L C X))) (fun _ => rfl)]
    exact List.Sublist.refl _
  | o :: os, hos, led => by
    simp only [relabelAll, applyAll]
    have hcomp : (led.filter (settled τ₀)).map (reeraBy (fun k => relabelAll os (o.relabel k))) =
        ((led.filter (settled τ₀)).map (reeraBy o.relabel)).map (reeraBy (relabelAll os)) := by
      rw [List.map_map]
      rfl
    rw [hcomp]
    have h1 := (op_settled_kept τ₀ o (hos o (List.mem_cons_self _ _)) led).map (reeraBy (relabelAll os))
    have h2 := settled_kept τ₀ os (fun o' h => hos o' (List.mem_cons_of_mem _ h)) (o.apply led)
    exact h1.trans h2

/-- **L40 (no forgetting), counts.** After any op sequence that never re-addresses before `τ₀`, the era that `ℓ` became
(`Φ ℓ`, `Φ` = composed merge map) holds at least every settled count `ℓ` held. -/
theorem no_forgetting (τ₀ : ℕ) (os : List (LOp L C X)) (hos : ∀ o ∈ os, o.Floor τ₀) (led : List (Tok L C X))
    (ℓ : L) (r : C) (x : X) :
    led.countP (fun z => decide ((z.era = ℓ ∧ z.ctx = r ∧ z.sym = x) ∧ z.time ≤ τ₀)) ≤
      table (applyAll os led) (relabelAll os ℓ) r x := by
  have hsub := settled_kept τ₀ os hos led
  unfold table
  calc led.countP (fun z => decide ((z.era = ℓ ∧ z.ctx = r ∧ z.sym = x) ∧ z.time ≤ τ₀))
      = (led.filter (settled τ₀)).countP (fun z => decide (z.era = ℓ ∧ z.ctx = r ∧ z.sym = x)) :=
        countP_settled τ₀ (fun z => z.era = ℓ ∧ z.ctx = r ∧ z.sym = x) led
    _ ≤ (led.filter (settled τ₀)).countP
          (fun z => decide ((reeraBy (relabelAll os) z).era = relabelAll os ℓ ∧
            (reeraBy (relabelAll os) z).ctx = r ∧ (reeraBy (relabelAll os) z).sym = x)) := by
        apply List.countP_mono_left
        intro z _ hz
        simp only [decide_eq_true_eq, reeraBy, Tok.reera_era, Tok.reera_ctx, Tok.reera_sym] at hz ⊢
        exact ⟨by rw [hz.1], hz.2⟩
    _ = ((led.filter (settled τ₀)).map (reeraBy (relabelAll os))).countP
          (fun z => decide (z.era = relabelAll os ℓ ∧ z.ctx = r ∧ z.sym = x)) := by
        rw [List.countP_map]
        rfl
    _ ≤ ((applyAll os led).filter (settled τ₀)).countP
          (fun z => decide (z.era = relabelAll os ℓ ∧ z.ctx = r ∧ z.sym = x)) := hsub.countP_le _
    _ ≤ (applyAll os led).countP (fun z => decide (z.era = relabelAll os ℓ ∧ z.ctx = r ∧ z.sym = x)) :=
        (List.filter_sublist _).countP_le _

/-! ### Exact zero forgetting for an era nobody touches -/

/-- `o` does not touch era `ℓ`: no credit to it, no merge involving it, no birth reusing its label, no re-address
into it and none reaching back before `τ₀`. -/
def Untouched (ℓ : L) (τ₀ : ℕ) : LOp L C X → Prop
  | LOp.credit z => z.era ≠ ℓ
  | LOp.birth _ => True
  | LOp.fold i j => i ≠ ℓ ∧ j ≠ ℓ
  | LOp.readdress τ b => τ₀ ≤ τ ∧ b ≠ ℓ

def eraTokens (ℓ : L) (led : List (Tok L C X)) : List (Tok L C X) := led.filter (fun z => decide (z.era = ℓ))

/-- A map that fixes era-`ℓ` tokens and never sends another token into era `ℓ` leaves era `ℓ`'s tokens unchanged. -/
theorem eraTokens_map (ℓ : L) (g : Tok L C X → Tok L C X) : ∀ (led : List (Tok L C X)),
    (∀ z ∈ led, z.era = ℓ → g z = z) → (∀ z ∈ led, (g z).era = ℓ → z.era = ℓ) →
    eraTokens ℓ (led.map g) = eraTokens ℓ led
  | [], _, _ => rfl
  | z :: zs, hfix, hin => by
    have ih := eraTokens_map ℓ g zs (fun w hw => hfix w (List.mem_cons_of_mem _ hw))
      (fun w hw => hin w (List.mem_cons_of_mem _ hw))
    simp only [List.map_cons, eraTokens, List.filter_cons] at ih ⊢
    by_cases hz : z.era = ℓ
    · rw [hfix z (List.mem_cons_self _ _) hz]
      simp [hz, ih]
    · have hg : ¬ (g z).era = ℓ := fun h => hz (hin z (List.mem_cons_self _ _) h)
      simp [hz, hg, ih]

theorem table_eq (ℓ : L) (r : C) (x : X) : ∀ l : List (Tok L C X),
    table l ℓ r x = (eraTokens ℓ l).countP (fun z => decide (z.ctx = r ∧ z.sym = x))
  | [] => rfl
  | z :: zs => by
    have ih := table_eq ℓ r x zs
    unfold table eraTokens at ih ⊢
    rw [List.countP_cons, List.filter_cons, ih]
    by_cases h1 : z.era = ℓ <;> by_cases h2 : z.ctx = r <;> by_cases h3 : z.sym = x <;>
      simp [h1, h2, h3, List.countP_cons]

theorem op_untouched (ℓ : L) (τ₀ : ℕ) (o : LOp L C X) (ho : Untouched ℓ τ₀ o) (led : List (Tok L C X))
    (hset : ∀ z ∈ led, z.era = ℓ → z.time ≤ τ₀) :
    eraTokens ℓ (o.apply led) = eraTokens ℓ led ∧ ∀ z ∈ o.apply led, z.era = ℓ → z.time ≤ τ₀ := by
  cases o with
  | credit z =>
    simp only [Untouched] at ho
    simp only [LOp.apply, eraTokens, List.filter_append, List.filter_cons, List.filter_nil, ho, decide_False]
    refine ⟨by simp, ?_⟩
    intro w hw hwe
    rcases List.mem_append.mp hw with hw | hw
    · exact hset w hw hwe
    · simp at hw; subst hw; exact absurd hwe ho
  | birth new => exact ⟨rfl, hset⟩
  | fold i j =>
    simp only [Untouched] at ho
    simp only [LOp.apply]
    refine ⟨eraTokens_map ℓ _ led ?_ ?_, ?_⟩
    · intro z _ hz
      have hj : z.era ≠ j := by rw [hz]; exact Ne.symm ho.2
      simp [hj]
    · intro z _ hz
      by_cases h : z.era = j
      · simp [h] at hz; exact absurd hz ho.1
      · simpa [h] using hz
    · intro w hw hwe
      obtain ⟨z, hz, rfl⟩ := List.mem_map.mp hw
      by_cases h : z.era = j
      · simp [h] at hwe; exact absurd hwe ho.1
      · simp [h] at hwe; simpa using hset z hz hwe
  | readdress τ b =>
    simp only [Untouched] at ho
    simp only [LOp.apply]
    refine ⟨eraTokens_map ℓ _ led ?_ ?_, ?_⟩
    · intro z hz he
      have : ¬ τ < z.time := by have := hset z hz he; omega
      simp [this]
    · intro z _ hz
      by_cases h : τ < z.time
      · simp [h] at hz; exact absurd hz ho.2
      · simpa [h] using hz
    · intro w hw hwe
      obtain ⟨z, hz, rfl⟩ := List.mem_map.mp hw
      by_cases h : τ < z.time
      · simp [h] at hwe; exact absurd hwe ho.2
      · simp [h] at hwe ⊢; exact hset z hz hwe

/-- **L40 (exact zero forgetting).** An era that no op touches keeps exactly its table (so exactly its predictive)
through any excursion, of any length, through any number of other eras' births, credits, merges and re-addresses. -/
theorem untouched_era_unchanged (ℓ : L) (τ₀ : ℕ) : ∀ (os : List (LOp L C X)), (∀ o ∈ os, Untouched ℓ τ₀ o) →
    ∀ (led : List (Tok L C X)), (∀ z ∈ led, z.era = ℓ → z.time ≤ τ₀) →
    ∀ r x, table (applyAll os led) ℓ r x = table led ℓ r x
  | [], _, led, _, r, x => rfl
  | o :: os, hos, led, hset, r, x => by
    have ⟨h1, h2⟩ := op_untouched ℓ τ₀ o (hos o (List.mem_cons_self _ _)) led hset
    have ih := untouched_era_unchanged ℓ τ₀ os (fun o' h => hos o' (List.mem_cons_of_mem _ h)) (o.apply led) h2 r x
    simp only [applyAll]
    rw [ih]
    rw [table_eq, table_eq, h1]

/-! ### Contrast: exponential decay forgets -/

/-- An EMA table keeps `λ^L c` of an era's counts after `L` steps away; it tends to `0` (the era bank keeps `c`). -/
theorem ema_forgets {lam : ℝ} (h0 : 0 ≤ lam) (h1 : lam < 1) (c : ℝ) :
    Filter.Tendsto (fun L : ℕ => lam ^ L * c) Filter.atTop (nhds 0) := by
  simpa using (tendsto_pow_atTop_nhds_zero_of_lt_one h0 h1).mul_const c

end L40
end FCPA

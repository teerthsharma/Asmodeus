import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.GCongr

/-!
L38 (research program, Addendum 2): regret of the TT's filter over an era bank that **grows by births and shrinks by
merges**, for every decider of the E0d/E0e tournament that moves `w_t` linearly (`zero/research/e0/deciders.py`).

The density `w_t` (chronal fluid data density) is updated exactly as the code does:
* move   `v_t(i) = (1-α) w_t(i) + ρ/|s_t| + Σ_{j∈s_t} w_t(j) K_t(j,i)`   (over the live pool `s_t`, any kernel `K_t ≥ 0`),
* predict `p_t = Σ_{i∈s_t} v_t(i) P_t(i)`                                 (`P_t(i)` = probability expert `i` gave `x_t`),
* Bayes  `u_t(i) = v_t(i) P_t(i) / p_t`                                   ("multiply by likelihood, then normalise"),
* between steps, a list of ops: `birth new` (`w ↦ (1-α) w + α·[new]`) and `fold i j` (era `j` merged into era `i`,
  weights added).
SOFT (fixed share) is `ρ = α, K = 0`; MOM (momentum) is `ρ = α/2, K = (α/2) Q̃` (see `L41_Momentum`). `K_t` may depend on
the past in any way, so a **learned** kernel is covered.

`bank_regret`: for **every** realised sequence and **every** comparator path `c_t ∈ s_t` (any era, window or BOCPD expert
at each step; a "stay" follows the comparator through merges),
`Σ -log p_t ≤ Σ -log P_t(c_t) + log(|s_0|/(1-α+ρ)) + Σ_t cost_t`, with
`cost_t = (1 + births_t) log(1/(1-α))` on a stay, and `-log max(ρ/N, K(j,i)(1-α)^births_t)` on a switch `j → i`.
Pathwise (no probability), any expert predictions, any birth/merge schedule.
Corollaries: `bank_regret_simple` (`S` switches, `B` births), `bank_regret_fixed` (vs one expert that is never folded,
e.g. BOCPD: `≤ log|s_0| + (T-1+B) α/(1-α)` for SOFT), `window_regret` (any start time: re-entry cost, used by L40),
`butterfly` (one token moves `log w` by at most `log(max P / min P)`).
-/
open BigOperators Real Finset

namespace FCPA
namespace L38

/-! ### The telescoping core -/

/-- Telescoping along a comparator path. `u t`, `v t` are the posterior and prior weight of the comparator at step `t`,
`p t` the mixture probability, `P t` the comparator's probability, `γ t` a lower bound for the transport of weight. -/
theorem telescope (p P u v γ : ℕ → ℝ)
    (hp : ∀ t, 0 < p t) (hP : ∀ t, 0 < P t) (hu : ∀ t, 0 < u t) (hu1 : ∀ t, u t ≤ 1)
    (hγ : ∀ t, 0 < γ t) (hbayes : ∀ t, p t * u t = v t * P t)
    (h0 : γ 0 ≤ v 0) (hstep : ∀ t, γ (t + 1) * u t ≤ v (t + 1)) (T : ℕ) :
    ∑ t in range T, -log (p t) ≤ ∑ t in range T, -log (P t) + ∑ t in range T, -log (γ t) := by
  have hv : ∀ t, 0 < v t := fun t => by
    have h1 : 0 < p t * u t := mul_pos (hp t) (hu t)
    rw [hbayes t] at h1
    exact pos_of_mul_pos_left h1 (hP t).le
  have key : ∀ t, -log (p t) = -log (P t) - log (v t) + log (u t) := fun t => by
    have := congrArg log (hbayes t)
    rw [log_mul (hp t).ne' (hu t).ne', log_mul (hv t).ne' (hP t).ne'] at this
    linarith
  have aux : ∀ n, ∑ t in range (n + 1), -log (p t) - log (u n) ≤
      ∑ t in range (n + 1), -log (P t) + ∑ t in range (n + 1), -log (γ t) := by
    intro n
    induction n with
    | zero =>
      show ∑ t in range 1, -log (p t) - log (u 0) ≤ ∑ t in range 1, -log (P t) + ∑ t in range 1, -log (γ t)
      simp only [sum_range_one]
      rw [key 0]
      have := log_le_log (hγ 0) h0
      linarith
    | succ n ih =>
      show ∑ t in range (n + 1 + 1), -log (p t) - log (u (n + 1)) ≤
        ∑ t in range (n + 1 + 1), -log (P t) + ∑ t in range (n + 1 + 1), -log (γ t)
      rw [sum_range_succ, sum_range_succ (fun t => -log (P t)), sum_range_succ (fun t => -log (γ t)), key (n + 1)]
      have h := log_le_log (mul_pos (hγ (n + 1)) (hu n)) (hstep n)
      rw [log_mul (hγ (n + 1)).ne' (hu n).ne'] at h
      linarith
  cases T with
  | zero => simp
  | succ n =>
    have := aux n
    have : log (u n) ≤ 0 := log_nonpos (hu n).le (hu1 n)
    show ∑ t in range (n + 1), -log (p t) ≤ ∑ t in range (n + 1), -log (P t) + ∑ t in range (n + 1), -log (γ t)
    linarith

/-! ### The operations of the era bank -/

variable {ι : Type*} [DecidableEq ι]

/-- The move step of the decider: stay with `1-α`, spread `ρ` uniformly, and carry mass along the kernel `K`. -/
noncomputable def move (α ρ : ℝ) (s : Finset ι) (K : ι → ι → ℝ) (w : ι → ℝ) : ι → ℝ :=
  fun i => (1 - α) * w i + ρ / s.card + ∑ j in s, w j * K j i

/-- Mixture probability of the realised token. -/
noncomputable def mix (s : Finset ι) (P v : ι → ℝ) : ℝ := ∑ j in s, v j * P j

/-- Bayes step: multiply by the likelihood, then normalise. -/
noncomputable def bayes (s : Finset ι) (P v : ι → ℝ) : ι → ℝ :=
  fun i => v i * P i / mix s P v

/-- Bank operations between two tokens. -/
inductive Op (ι : Type*)
  | birth (new : ι)
  | fold (i j : ι)

namespace Op

/-- `birth new`: every weight times `(1-α)`, the newborn gets `α` (its old weight is `0` when the label is fresh, as in
the code). `fold i j`: era `j` is merged into era `i`, weights add. -/
noncomputable def apply (α : ℝ) : Op ι → (ι → ℝ) → (ι → ℝ)
  | birth new, u => fun k => (1 - α) * u k + if k = new then α else 0
  | fold i j, u => fun k => if k = i then u i + u j else if k = j then 0 else u k

/-- Where a comparator that stays on its era goes through the op. -/
def relabel : Op ι → ι → ι
  | birth _ => id
  | fold i j => fun k => if k = j then i else k

def births : Op ι → ℕ
  | birth _ => 1
  | fold _ _ => 0

theorem apply_nonneg {α : ℝ} (h0 : 0 ≤ α) (h1 : α ≤ 1) (o : Op ι) {u : ι → ℝ} (hu : ∀ k, 0 ≤ u k) (k : ι) :
    0 ≤ o.apply α u k := by
  cases o with
  | birth new =>
    simp only [apply]
    have : 0 ≤ (1 - α) * u k := mul_nonneg (by linarith) (hu k)
    split_ifs <;> linarith
  | fold i j =>
    simp only [apply]
    split_ifs
    · exact add_nonneg (hu i) (hu j)
    · exact le_rfl
    · exact hu k

theorem apply_lb {α : ℝ} (h0 : 0 ≤ α) (o : Op ι) {u : ι → ℝ} (hu : ∀ k, 0 ≤ u k) (k : ι) :
    (1 - α) ^ o.births * u k ≤ o.apply α u (o.relabel k) := by
  cases o with
  | birth new =>
    simp only [apply, relabel, births, pow_one, id]
    split_ifs <;> linarith
  | fold i j =>
    simp only [apply, relabel, births, pow_zero, one_mul]
    by_cases hkj : k = j
    · subst hkj
      simp only [if_true]
      linarith [hu i, hu k]
    · simp only [hkj, if_false]
      by_cases hki : k = i
      · subst hki; simp only [if_true]; linarith [hu j]
      · simp [hki, hkj]

end Op

noncomputable def applyAll (α : ℝ) : List (Op ι) → (ι → ℝ) → (ι → ℝ)
  | [], u => u
  | o :: os, u => applyAll α os (o.apply α u)

def relabelAll : List (Op ι) → ι → ι
  | [], k => k
  | o :: os, k => relabelAll os (o.relabel k)

def birthsAll : List (Op ι) → ℕ
  | [] => 0
  | o :: os => o.births + birthsAll os

theorem applyAll_nonneg {α : ℝ} (h0 : 0 ≤ α) (h1 : α ≤ 1) (os : List (Op ι)) :
    ∀ {u : ι → ℝ}, (∀ k, 0 ≤ u k) → ∀ k, 0 ≤ applyAll α os u k := by
  induction os with
  | nil => intro u hu k; exact hu k
  | cons o _ ih => intro u hu k; exact ih (fun k => o.apply_nonneg h0 h1 hu k) k

/-- A comparator that stays on its era through births and merges keeps at least `(1-α)^births` of its weight. -/
theorem applyAll_lb {α : ℝ} (h0 : 0 ≤ α) (h1 : α ≤ 1) (os : List (Op ι)) :
    ∀ {u : ι → ℝ}, (∀ k, 0 ≤ u k) → ∀ k,
      (1 - α) ^ birthsAll os * u k ≤ applyAll α os u (relabelAll os k) := by
  induction os with
  | nil => intro u _ k; simp [birthsAll, applyAll, relabelAll]
  | cons o os ih =>
    intro u hu k
    simp only [birthsAll, applyAll, relabelAll, pow_add]
    have hnn := fun k => o.apply_nonneg h0 h1 hu k
    have h2 := ih hnn (o.relabel k)
    have h3 := o.apply_lb h0 hu k
    have hp : 0 ≤ (1 - α) ^ birthsAll os := pow_nonneg (by linarith) _
    calc (1 - α) ^ o.births * (1 - α) ^ birthsAll os * u k
        = (1 - α) ^ birthsAll os * ((1 - α) ^ o.births * u k) := by ring
      _ ≤ (1 - α) ^ birthsAll os * o.apply α u (o.relabel k) := mul_le_mul_of_nonneg_left h3 hp
      _ ≤ _ := h2

/-! ### The run -/

/-- A run of the era bank: rates `α, ρ`, live pool `s t`, kernel `K t` in force at step `t`, the probability `P t i`
that expert `i` gave the token at step `t`, and the ops applied after step `t`. -/
structure Run (ι : Type*) where
  α : ℝ
  ρ : ℝ
  s : ℕ → Finset ι
  K : ℕ → ι → ι → ℝ
  P : ℕ → ι → ℝ
  ops : ℕ → List (Op ι)

/-- Standing hypotheses: rates in range, kernel and likelihoods nonnegative, likelihoods positive on the live pool. -/
structure Run.Valid (R : Run ι) : Prop where
  α_pos : 0 < R.α
  α_lt : R.α < 1
  ρ_pos : 0 < R.ρ
  K_nonneg : ∀ t i j, 0 ≤ R.K t i j
  P_nonneg : ∀ t i, 0 ≤ R.P t i
  P_pos : ∀ t, ∀ i ∈ R.s t, 0 < R.P t i

/-- The prior weights `v_t` (after the move step) of the filter. -/
noncomputable def prior (R : Run ι) : ℕ → ι → ℝ
  | 0 => move R.α R.ρ (R.s 0) (R.K 0) (fun _ => 1 / (R.s 0).card)
  | t + 1 => move R.α R.ρ (R.s (t + 1)) (R.K (t + 1))
      (applyAll R.α (R.ops t) (bayes (R.s t) (R.P t) (prior R t)))

/-- The filter's probability of the token at step `t`. -/
noncomputable def pmix (R : Run ι) (t : ℕ) : ℝ := mix (R.s t) (R.P t) (prior R t)

/-- The posterior (after "multiplication and normalisation") at step `t`. -/
noncomputable def post (R : Run ι) (t : ℕ) : ι → ℝ := bayes (R.s t) (R.P t) (prior R t)

/-- Comparator stays at step `t → t+1` (follows its era through births and merges). -/
def Stay (R : Run ι) (c : ℕ → ι) (t : ℕ) : Prop := c (t + 1) = relabelAll (R.ops t) (c t)

/-- The weight-transport factor of step `t → t+1` along the comparator. -/
noncomputable def gam (R : Run ι) (c : ℕ → ι) (N : ℕ) (t : ℕ) : ℝ := by
  classical
  exact if Stay R c t then (1 - R.α + R.K (t + 1) (c (t + 1)) (c (t + 1))) * (1 - R.α) ^ birthsAll (R.ops t)
    else max (R.ρ / N) (if relabelAll (R.ops t) (c t) ∈ R.s (t + 1) then
      R.K (t + 1) (relabelAll (R.ops t) (c t)) (c (t + 1)) * (1 - R.α) ^ birthsAll (R.ops t) else 0)

/-- Cost of step `t → t+1` in the regret bound. -/
noncomputable def cost (R : Run ι) (c : ℕ → ι) (N : ℕ) (t : ℕ) : ℝ := -log (gam R c N t)

section Lemmas
variable {R : Run ι} (hR : R.Valid)

theorem move_nonneg {s : Finset ι} {K : ι → ι → ℝ} (hK : ∀ i j, 0 ≤ K i j) {w : ι → ℝ} (hw : ∀ k, 0 ≤ w k)
    (k : ι) : 0 ≤ move R.α R.ρ s K w k := by
  unfold move
  have : 0 ≤ (1 - R.α) * w k := mul_nonneg (by linarith [hR.α_lt]) (hw k)
  have : 0 ≤ R.ρ / s.card := div_nonneg hR.ρ_pos.le (Nat.cast_nonneg _)
  have : 0 ≤ ∑ j in s, w j * K j k := sum_nonneg fun j _ => mul_nonneg (hw j) (hK j k)
  linarith

theorem move_floor {s : Finset ι} {K : ι → ι → ℝ} (hK : ∀ i j, 0 ≤ K i j) {w : ι → ℝ} (hw : ∀ k, 0 ≤ w k)
    (k : ι) : R.ρ / s.card ≤ move R.α R.ρ s K w k := by
  unfold move
  have : 0 ≤ (1 - R.α) * w k := mul_nonneg (by linarith [hR.α_lt]) (hw k)
  have : 0 ≤ ∑ j in s, w j * K j k := sum_nonneg fun j _ => mul_nonneg (hw j) (hK j k)
  linarith

theorem move_stay {s : Finset ι} {K : ι → ι → ℝ} (hK : ∀ i j, 0 ≤ K i j) {w : ι → ℝ} (hw : ∀ k, 0 ≤ w k)
    (k : ι) : (1 - R.α) * w k ≤ move R.α R.ρ s K w k := by
  unfold move
  have : 0 ≤ R.ρ / s.card := div_nonneg hR.ρ_pos.le (Nat.cast_nonneg _)
  have : 0 ≤ ∑ j in s, w j * K j k := sum_nonneg fun j _ => mul_nonneg (hw j) (hK j k)
  linarith

theorem move_kernel {s : Finset ι} {K : ι → ι → ℝ} (hK : ∀ i j, 0 ≤ K i j) {w : ι → ℝ} (hw : ∀ k, 0 ≤ w k)
    {j : ι} (hj : j ∈ s) (k : ι) : w j * K j k ≤ move R.α R.ρ s K w k := by
  unfold move
  have : 0 ≤ (1 - R.α) * w k := mul_nonneg (by linarith [hR.α_lt]) (hw k)
  have : 0 ≤ R.ρ / s.card := div_nonneg hR.ρ_pos.le (Nat.cast_nonneg _)
  have : w j * K j k ≤ ∑ j' in s, w j' * K j' k :=
    single_le_sum (f := fun j' => w j' * K j' k) (fun j' _ => mul_nonneg (hw j') (hK j' k)) hj
  linarith

theorem move_self {s : Finset ι} {K : ι → ι → ℝ} (hK : ∀ i j, 0 ≤ K i j) {w : ι → ℝ} (hw : ∀ k, 0 ≤ w k)
    {k : ι} (hk : k ∈ s) : (1 - R.α + K k k) * w k ≤ move R.α R.ρ s K w k := by
  unfold move
  have hsum : w k * K k k ≤ ∑ j in s, w j * K j k :=
    single_le_sum (f := fun j => w j * K j k) (fun j _ => mul_nonneg (hw j) (hK j k)) hk
  have : 0 ≤ R.ρ / s.card := div_nonneg hR.ρ_pos.le (Nat.cast_nonneg _)
  linarith

theorem prior_nonneg : ∀ t k, 0 ≤ prior R t k := by
  intro t
  induction t with
  | zero => intro k; exact move_nonneg hR (hR.K_nonneg 0) (fun _ => div_nonneg zero_le_one (Nat.cast_nonneg _)) k
  | succ t ih =>
    intro k
    apply move_nonneg hR (hR.K_nonneg _)
    apply applyAll_nonneg hR.α_pos.le hR.α_lt.le
    intro j
    exact div_nonneg (mul_nonneg (ih j) (hR.P_nonneg t j)) (sum_nonneg fun i _ => mul_nonneg (ih i) (hR.P_nonneg t i))

theorem post_nonneg (t : ℕ) (k : ι) : 0 ≤ post R t k := by
  have hv := prior_nonneg hR t
  exact div_nonneg (mul_nonneg (hv k) (hR.P_nonneg t k)) (sum_nonneg fun i _ => mul_nonneg (hv i) (hR.P_nonneg t i))

/-- The transported weights before the move step. -/
theorem transported_nonneg (t : ℕ) (k : ι) : 0 ≤ applyAll R.α (R.ops t) (post R t) k :=
  applyAll_nonneg hR.α_pos.le hR.α_lt.le _ (post_nonneg hR t) k

/-- **Weight floor**: every live expert keeps at least `ρ/|s_t|` prior weight at every step. -/
theorem prior_floor (t : ℕ) (k : ι) : R.ρ / (R.s t).card ≤ prior R t k := by
  cases t with
  | zero => exact move_floor hR (hR.K_nonneg 0) (fun _ => div_nonneg zero_le_one (Nat.cast_nonneg _)) k
  | succ t => exact move_floor hR (hR.K_nonneg _) (transported_nonneg hR t) k

theorem prior_pos (t : ℕ) (k : ι) (hk : k ∈ R.s t) : 0 < prior R t k := by
  have hcard : (0 : ℝ) < (R.s t).card := Nat.cast_pos.mpr (card_pos.mpr ⟨k, hk⟩)
  exact lt_of_lt_of_le (div_pos hR.ρ_pos hcard) (prior_floor hR t k)

theorem single_le_mix (t : ℕ) (k : ι) (hk : k ∈ R.s t) : prior R t k * R.P t k ≤ pmix R t :=
  single_le_sum (f := fun j => prior R t j * R.P t j)
    (fun j _ => mul_nonneg (prior_nonneg hR t j) (hR.P_nonneg t j)) hk

theorem pmix_pos (t : ℕ) (k : ι) (hk : k ∈ R.s t) : 0 < pmix R t :=
  lt_of_lt_of_le (mul_pos (prior_pos hR t k hk) (hR.P_pos t k hk)) (single_le_mix hR t k hk)

theorem post_pos (t : ℕ) (k : ι) (hk : k ∈ R.s t) : 0 < post R t k :=
  div_pos (mul_pos (prior_pos hR t k hk) (hR.P_pos t k hk)) (pmix_pos hR t k hk)

theorem post_le_one (t : ℕ) (k : ι) (hk : k ∈ R.s t) : post R t k ≤ 1 :=
  (div_le_one (pmix_pos hR t k hk)).mpr (single_le_mix hR t k hk)

theorem bayes_id (t : ℕ) (k : ι) (hk : k ∈ R.s t) : pmix R t * post R t k = prior R t k * R.P t k := by
  show pmix R t * (prior R t k * R.P t k / pmix R t) = _
  field_simp [(pmix_pos hR t k hk).ne']
  ring

/-- One step of transport along any comparator path. -/
theorem step_lb (N : ℕ) (hN : ∀ t, (R.s t).card ≤ N) (c : ℕ → ι) (hc : ∀ t, c t ∈ R.s t) (t : ℕ) :
    gam R c N t * post R t (c t) ≤ prior R (t + 1) (c (t + 1)) := by
  classical
  have hcardpos : ∀ t, (0 : ℝ) < (R.s t).card := fun t => Nat.cast_pos.mpr (card_pos.mpr ⟨c t, hc t⟩)
  have hNpos : (0 : ℝ) < N := lt_of_lt_of_le (hcardpos 0) (Nat.cast_le.mpr (hN 0))
  have hpost := post_nonneg hR t
  have hw := transported_nonneg hR t
  have hlb := applyAll_lb hR.α_pos.le hR.α_lt.le (R.ops t) hpost (c t)
  have hu1 := post_le_one hR t _ (hc t)
  have hu0 := post_nonneg hR t (c t)
  show gam R c N t * post R t (c t) ≤
    move R.α R.ρ (R.s (t + 1)) (R.K (t + 1)) (applyAll R.α (R.ops t) (post R t)) (c (t + 1))
  unfold gam
  have hcmp : R.ρ / N ≤ R.ρ / (R.s (t + 1)).card :=
    div_le_div_of_nonneg_left hR.ρ_pos.le (hcardpos _) (Nat.cast_le.mpr (hN _))
  have hfl := move_floor hR (hR.K_nonneg (t + 1)) hw (s := R.s (t + 1)) (c (t + 1))
  have hfloor : R.ρ / N * post R t (c t) ≤
      move R.α R.ρ (R.s (t + 1)) (R.K (t + 1)) (applyAll R.α (R.ops t) (post R t)) (c (t + 1)) :=
    calc R.ρ / N * post R t (c t) ≤ R.ρ / N * 1 := mul_le_mul_of_nonneg_left hu1 (div_nonneg hR.ρ_pos.le hNpos.le)
      _ ≤ _ := by linarith
  split_ifs with hs hj
  · rw [← hs] at hlb
    have hst := move_self hR (hR.K_nonneg (t + 1)) hw (hc (t + 1)) (s := R.s (t + 1))
    have hK0 := hR.K_nonneg (t + 1) (c (t + 1)) (c (t + 1))
    calc (1 - R.α + R.K (t + 1) (c (t + 1)) (c (t + 1))) * (1 - R.α) ^ birthsAll (R.ops t) * post R t (c t)
        = (1 - R.α + R.K (t + 1) (c (t + 1)) (c (t + 1))) * ((1 - R.α) ^ birthsAll (R.ops t) * post R t (c t)) := by
          ring
      _ ≤ (1 - R.α + R.K (t + 1) (c (t + 1)) (c (t + 1))) * applyAll R.α (R.ops t) (post R t) (c (t + 1)) :=
        mul_le_mul_of_nonneg_left hlb (by linarith [hR.α_lt])
      _ ≤ _ := hst
  · rcases le_total (R.ρ / N)
      (R.K (t + 1) (relabelAll (R.ops t) (c t)) (c (t + 1)) * (1 - R.α) ^ birthsAll (R.ops t)) with hm | hm
    · rw [max_eq_right hm]
      have hk := move_kernel hR (hR.K_nonneg (t + 1)) hw hj (s := R.s (t + 1)) (c (t + 1))
      have hK0 := hR.K_nonneg (t + 1) (relabelAll (R.ops t) (c t)) (c (t + 1))
      calc R.K (t + 1) (relabelAll (R.ops t) (c t)) (c (t + 1)) * (1 - R.α) ^ birthsAll (R.ops t) * post R t (c t)
          = R.K (t + 1) (relabelAll (R.ops t) (c t)) (c (t + 1)) *
              ((1 - R.α) ^ birthsAll (R.ops t) * post R t (c t)) := by ring
        _ ≤ R.K (t + 1) (relabelAll (R.ops t) (c t)) (c (t + 1)) *
              applyAll R.α (R.ops t) (post R t) (relabelAll (R.ops t) (c t)) :=
            mul_le_mul_of_nonneg_left hlb hK0
        _ = applyAll R.α (R.ops t) (post R t) (relabelAll (R.ops t) (c t)) *
              R.K (t + 1) (relabelAll (R.ops t) (c t)) (c (t + 1)) := by ring
        _ ≤ _ := hk
    · rw [max_eq_left hm]; exact hfloor
  · have : (0 : ℝ) ≤ R.ρ / N := div_nonneg hR.ρ_pos.le hNpos.le
    rw [max_eq_left this]; exact hfloor

theorem gam_pos (N : ℕ) (hN0 : 0 < N) (c : ℕ → ι) (t : ℕ) : 0 < gam R c N t := by
  classical
  unfold gam
  have hNpos : (0 : ℝ) < N := Nat.cast_pos.mpr hN0
  split_ifs
  · exact mul_pos (by linarith [hR.α_lt, hR.K_nonneg (t + 1) (c (t + 1)) (c (t + 1))]) (pow_pos (by linarith [hR.α_lt]) _)
  · exact lt_of_lt_of_le (div_pos hR.ρ_pos hNpos) (le_max_left _ _)
  · exact lt_of_lt_of_le (div_pos hR.ρ_pos hNpos) (le_max_left _ _)

/-- Start of the path: the uniform start keeps `(1-α+ρ)/|s_0|` (`= 1/|s_0|` for SOFT and MOM). -/
theorem start_lb (k : ι) : (1 - R.α + R.ρ) / (R.s 0).card ≤ prior R 0 k := by
  show _ ≤ move R.α R.ρ (R.s 0) (R.K 0) (fun _ => 1 / (R.s 0).card) k
  unfold move
  have : 0 ≤ ∑ j in R.s 0, 1 / ((R.s 0).card : ℝ) * R.K 0 j k :=
    sum_nonneg fun j _ => mul_nonneg (div_nonneg zero_le_one (Nat.cast_nonneg _)) (hR.K_nonneg 0 j k)
  have e : (1 - R.α + R.ρ) / ((R.s 0).card : ℝ) = (1 - R.α) * (1 / (R.s 0).card) + R.ρ / (R.s 0).card := by ring
  linarith

end Lemmas

section Main
variable {R : Run ι} (hR : R.Valid)

/-- **L38.** Regret of the growing-bank filter against any comparator path, over the tokens `0..T`. -/
theorem bank_regret (N : ℕ) (hN : ∀ t, (R.s t).card ≤ N) (c : ℕ → ι) (hc : ∀ t, c t ∈ R.s t) (T : ℕ) :
    ∑ t in range (T + 1), -log (pmix R t) ≤
      ∑ t in range (T + 1), -log (R.P t (c t)) + log ((R.s 0).card / (1 - R.α + R.ρ)) +
        ∑ t in range T, cost R c N t := by
  have hcardpos : ∀ t, (0 : ℝ) < (R.s t).card := fun t => Nat.cast_pos.mpr (card_pos.mpr ⟨c t, hc t⟩)
  have hN0 : 0 < N := Nat.cast_pos.mp (lt_of_lt_of_le (hcardpos 0) (Nat.cast_le.mpr (hN 0)))
  have h1αρ : 0 < 1 - R.α + R.ρ := by linarith [hR.α_lt, hR.ρ_pos]
  let γ : ℕ → ℝ := fun t => match t with
    | 0 => (1 - R.α + R.ρ) / (R.s 0).card
    | t + 1 => gam R c N t
  have hγ : ∀ t, 0 < γ t := by
    intro t
    cases t with
    | zero => exact div_pos h1αρ (hcardpos 0)
    | succ t => exact gam_pos hR N hN0 c t
  have main := telescope (pmix R) (fun t => R.P t (c t)) (fun t => post R t (c t)) (fun t => prior R t (c t)) γ
    (fun t => pmix_pos hR t _ (hc t)) (fun t => hR.P_pos t _ (hc t)) (fun t => post_pos hR t _ (hc t))
    (fun t => post_le_one hR t _ (hc t)) hγ (fun t => bayes_id hR t _ (hc t)) (start_lb hR (c 0))
    (fun t => step_lb hR N hN c hc t) (T + 1)
  have hγsum : ∑ t in range (T + 1), -log (γ t) =
      log ((R.s 0).card / (1 - R.α + R.ρ)) + ∑ t in range T, cost R c N t := by
    rw [sum_range_succ', add_comm]
    have e0 : -log (γ 0) = log ((R.s 0).card / (1 - R.α + R.ρ)) := by
      show -log ((1 - R.α + R.ρ) / ((R.s 0).card : ℝ)) = _
      rw [← log_inv, inv_div]
    rw [e0]
    rfl
  linarith

/-- A switch costs at most `log(N/ρ)` (the uniform floor), whatever the kernel. -/
theorem cost_switch_le (N : ℕ) (hN0 : 0 < N) (c : ℕ → ι) (t : ℕ) (hs : ¬ Stay R c t) :
    cost R c N t ≤ log (N / R.ρ) := by
  classical
  have hNpos : (0 : ℝ) < N := Nat.cast_pos.mpr hN0
  unfold cost
  have hg : R.ρ / N ≤ gam R c N t := by
    unfold gam; rw [if_neg hs]; exact le_max_left _ _
  have := log_le_log (div_pos hR.ρ_pos hNpos) hg
  rw [log_div hR.ρ_pos.ne' hNpos.ne'] at this
  rw [log_div hNpos.ne' hR.ρ_pos.ne']
  linarith

/-- A switch along the kernel from a live era costs at most `-log K(j,i) + births · log(1/(1-α))`. -/
theorem cost_kernel_le (N : ℕ) (c : ℕ → ι) (t : ℕ) (hs : ¬ Stay R c t)
    (hj : relabelAll (R.ops t) (c t) ∈ R.s (t + 1))
    (hK : 0 < R.K (t + 1) (relabelAll (R.ops t) (c t)) (c (t + 1))) :
    cost R c N t ≤ -log (R.K (t + 1) (relabelAll (R.ops t) (c t)) (c (t + 1))) +
      birthsAll (R.ops t) * log (1 / (1 - R.α)) := by
  classical
  have h1α : 0 < 1 - R.α := by linarith [hR.α_lt]
  unfold cost
  have hg : R.K (t + 1) (relabelAll (R.ops t) (c t)) (c (t + 1)) * (1 - R.α) ^ birthsAll (R.ops t) ≤ gam R c N t := by
    unfold gam; rw [if_neg hs, if_pos hj]; exact le_max_right _ _
  have hpos : 0 < R.K (t + 1) (relabelAll (R.ops t) (c t)) (c (t + 1)) * (1 - R.α) ^ birthsAll (R.ops t) :=
    mul_pos hK (pow_pos h1α _)
  have := log_le_log hpos hg
  rw [log_mul hK.ne' (pow_pos h1α _).ne', log_pow] at this
  rw [one_div, log_inv]
  linarith

theorem cost_stay (N : ℕ) (c : ℕ → ι) (t : ℕ) (hs : Stay R c t) :
    cost R c N t = -log (1 - R.α + R.K (t + 1) (c (t + 1)) (c (t + 1))) +
      birthsAll (R.ops t) * log (1 / (1 - R.α)) := by
  classical
  have h1 : 0 < 1 - R.α + R.K (t + 1) (c (t + 1)) (c (t + 1)) := by
    linarith [hR.α_lt, hR.K_nonneg (t + 1) (c (t + 1)) (c (t + 1))]
  have h2 : 0 < 1 - R.α := by linarith [hR.α_lt]
  unfold cost gam
  rw [if_pos hs, log_mul h1.ne' (pow_pos h2 _).ne', log_pow, one_div, log_inv]
  ring

/-- A stay costs at most `(1 + births) log(1/(1-α))`. -/
theorem cost_stay_le (N : ℕ) (c : ℕ → ι) (t : ℕ) (hs : Stay R c t) :
    cost R c N t ≤ (1 + birthsAll (R.ops t)) * log (1 / (1 - R.α)) := by
  rw [cost_stay hR N c t hs]
  have h2 : 0 < 1 - R.α := by linarith [hR.α_lt]
  have := log_le_log h2 (show 1 - R.α ≤ 1 - R.α + R.K (t + 1) (c (t + 1)) (c (t + 1)) by
    linarith [hR.K_nonneg (t + 1) (c (t + 1)) (c (t + 1))])
  rw [one_div, log_inv]
  linarith

theorem log_inv_one_sub_le : log (1 / (1 - R.α)) ≤ R.α / (1 - R.α) := by
  have h1α : 0 < 1 - R.α := by linarith [hR.α_lt]
  have := log_le_sub_one_of_pos (one_div_pos.mpr h1α)
  have e : 1 / (1 - R.α) - 1 = R.α / (1 - R.α) := by field_simp
  linarith

open Classical in
/-- **L38, readable form.** `S` switches and `B` births over `T + 1` tokens:
`regret ≤ log(|s_0|/(1-α+ρ)) + S log(N/ρ) + (T + B) log(1/(1-α))`. -/
theorem bank_regret_simple (N : ℕ) (hN : ∀ t, (R.s t).card ≤ N) (c : ℕ → ι) (hc : ∀ t, c t ∈ R.s t) (T : ℕ) :
    ∑ t in range (T + 1), -log (pmix R t) ≤
      ∑ t in range (T + 1), -log (R.P t (c t)) + log ((R.s 0).card / (1 - R.α + R.ρ)) +
        (((range T).filter (fun t => ¬ Stay R c t)).card : ℝ) * log (N / R.ρ) +
        ((T : ℝ) + ∑ t in range T, (birthsAll (R.ops t) : ℝ)) * log (1 / (1 - R.α)) := by
  classical
  have hcardpos : (0 : ℝ) < (R.s 0).card := Nat.cast_pos.mpr (card_pos.mpr ⟨c 0, hc 0⟩)
  have hN0 : 0 < N := Nat.cast_pos.mp (lt_of_lt_of_le hcardpos (Nat.cast_le.mpr (hN 0)))
  have hL : 0 ≤ log (1 / (1 - R.α)) := log_nonneg (by
    rw [le_div_iff (by linarith [hR.α_lt])]; linarith [hR.α_pos])
  have main := bank_regret hR N hN c hc T
  have hcost : ∑ t in range T, cost R c N t ≤ ∑ t in range T,
      ((if ¬ Stay R c t then log (N / R.ρ) else 0) + (1 + (birthsAll (R.ops t) : ℝ)) * log (1 / (1 - R.α))) := by
    apply sum_le_sum
    intro t _
    by_cases hs : Stay R c t
    · rw [if_neg (not_not.mpr hs), zero_add]
      exact cost_stay_le hR N c t hs
    · rw [if_pos hs]
      have := cost_switch_le hR N hN0 c t hs
      have : 0 ≤ (1 + (birthsAll (R.ops t) : ℝ)) * log (1 / (1 - R.α)) := mul_nonneg (by positivity) hL
      linarith
  have hsplit : ∑ t in range T,
      ((if ¬ Stay R c t then log (N / R.ρ) else 0) + (1 + (birthsAll (R.ops t) : ℝ)) * log (1 / (1 - R.α))) =
      (((range T).filter (fun t => ¬ Stay R c t)).card : ℝ) * log (N / R.ρ) +
        ((T : ℝ) + ∑ t in range T, (birthsAll (R.ops t) : ℝ)) * log (1 / (1 - R.α)) := by
    rw [sum_add_distrib, sum_ite, sum_const_zero, add_zero, sum_const, nsmul_eq_mul, ← sum_mul, sum_add_distrib]
    simp
  linarith

open Classical in
/-- **L38 vs one fixed expert** (e.g. the BOCPD novelty expert, never folded): no switch is ever paid, and each step
costs `log(1/(1-α+κ))` where `κ ≤ K_t(e,e)` is the expert's self-kernel (`κ = 0` for SOFT, `κ = α/2` for MOM's tail
experts), plus `log(1/(1-α))` per birth. -/
theorem bank_regret_fixed (N : ℕ) (hN : ∀ t, (R.s t).card ≤ N) (e : ι) (he : ∀ t, e ∈ R.s t)
    (hfix : ∀ t, relabelAll (R.ops t) e = e) (κ : ℝ) (hκ0 : 0 ≤ κ) (hκ : ∀ t, κ ≤ R.K t e e) (T : ℕ) :
    ∑ t in range (T + 1), -log (pmix R t) ≤
      ∑ t in range (T + 1), -log (R.P t e) + log ((R.s 0).card / (1 - R.α + R.ρ)) +
        T * log (1 / (1 - R.α + κ)) + (∑ t in range T, (birthsAll (R.ops t) : ℝ)) * log (1 / (1 - R.α)) := by
  have h := bank_regret hR N hN (fun _ => e) he T
  have h1 : 0 < 1 - R.α + κ := by linarith [hR.α_lt]
  have hc : ∑ t in range T, cost R (fun _ => e) N t ≤
      ∑ t in range T, (log (1 / (1 - R.α + κ)) + (birthsAll (R.ops t) : ℝ) * log (1 / (1 - R.α))) := by
    apply sum_le_sum
    intro t _
    have hs : Stay R (fun _ => e) t := (hfix t).symm
    rw [cost_stay hR N _ t hs]
    have := log_le_log h1 (show 1 - R.α + κ ≤ 1 - R.α + R.K (t + 1) e e by linarith [hκ (t + 1)])
    simp only [one_div, log_inv]
    linarith
  rw [sum_add_distrib, sum_const, card_range, nsmul_eq_mul, ← sum_mul] at hc
  linarith

/-- SOFT form: `regret ≤ log|s_0| + (T + B) α/(1-α)` against any never-folded expert. -/
theorem bank_regret_fixed_simple (N : ℕ) (hN : ∀ t, (R.s t).card ≤ N) (e : ι) (he : ∀ t, e ∈ R.s t)
    (hfix : ∀ t, relabelAll (R.ops t) e = e) (T : ℕ) :
    ∑ t in range (T + 1), -log (pmix R t) ≤
      ∑ t in range (T + 1), -log (R.P t e) + log ((R.s 0).card / (1 - R.α + R.ρ)) +
        ((T : ℝ) + ∑ t in range T, (birthsAll (R.ops t) : ℝ)) * (R.α / (1 - R.α)) := by
  have h := bank_regret_fixed hR N hN e he hfix 0 le_rfl (fun t => hR.K_nonneg t e e) T
  rw [add_zero] at h
  have hB : (0 : ℝ) ≤ (T : ℝ) + ∑ t in range T, (birthsAll (R.ops t) : ℝ) := by positivity
  have := mul_le_mul_of_nonneg_left (log_inv_one_sub_le hR) hB
  nlinarith

/-- **Window regret** (any start `t₀`): the comparator's weight never drops below `ρ/N`, so entering any path at any
time costs at most `log(N/ρ)`. This is the filter half of L40 (re-entering an old era). -/
theorem window_regret (N : ℕ) (hN : ∀ t, (R.s t).card ≤ N) (c : ℕ → ι) (hc : ∀ t, c t ∈ R.s t) (t₀ T : ℕ) :
    ∑ t in range (T + 1), -log (pmix R (t₀ + t)) ≤
      ∑ t in range (T + 1), -log (R.P (t₀ + t) (c (t₀ + t))) + log (N / R.ρ) +
        ∑ t in range T, cost R c N (t₀ + t) := by
  have hcardpos : ∀ t, (0 : ℝ) < (R.s t).card := fun t => Nat.cast_pos.mpr (card_pos.mpr ⟨c t, hc t⟩)
  have hN0 : 0 < N := Nat.cast_pos.mp (lt_of_lt_of_le (hcardpos 0) (Nat.cast_le.mpr (hN 0)))
  have hNpos : (0 : ℝ) < N := Nat.cast_pos.mpr hN0
  let γ : ℕ → ℝ := fun t => match t with
    | 0 => R.ρ / N
    | t + 1 => gam R c N (t₀ + t)
  have hγ : ∀ t, 0 < γ t := by
    intro t
    cases t with
    | zero => exact div_pos hR.ρ_pos hNpos
    | succ t => exact gam_pos hR N hN0 c _
  have h0 : γ 0 ≤ prior R (t₀ + 0) (c (t₀ + 0)) := by
    show R.ρ / N ≤ _
    exact le_trans (div_le_div_of_nonneg_left hR.ρ_pos.le (hcardpos _) (Nat.cast_le.mpr (hN _)))
      (prior_floor hR _ _)
  have main := telescope (fun t => pmix R (t₀ + t)) (fun t => R.P (t₀ + t) (c (t₀ + t)))
    (fun t => post R (t₀ + t) (c (t₀ + t))) (fun t => prior R (t₀ + t) (c (t₀ + t))) γ
    (fun t => pmix_pos hR _ _ (hc _)) (fun t => hR.P_pos _ _ (hc _)) (fun t => post_pos hR _ _ (hc _))
    (fun t => post_le_one hR _ _ (hc _)) hγ (fun t => bayes_id hR _ _ (hc _)) h0
    (fun t => step_lb hR N hN c hc (t₀ + t)) (T + 1)
  have hγsum : ∑ t in range (T + 1), -log (γ t) = log (N / R.ρ) + ∑ t in range T, cost R c N (t₀ + t) := by
    rw [sum_range_succ', add_comm]
    have e0 : -log (γ 0) = log (N / R.ρ) := by
      show -log (R.ρ / (N : ℝ)) = _
      rw [← log_inv, inv_div]
    rw [e0]
    rfl
  linarith

/-- **Butterfly bound** (Addendum 2, item 4): one token moves an expert's log-weight by at most `log(max P/min P)`
in the Bayes step, when the prior is a probability vector on the live pool. -/
theorem butterfly (t : ℕ) (m M : ℝ) (hm : 0 < m) (hP : ∀ i ∈ R.s t, m ≤ R.P t i ∧ R.P t i ≤ M)
    (hsum : ∑ i in R.s t, prior R t i = 1) (k : ι) (hk : k ∈ R.s t) :
    |log (post R t k) - log (prior R t k)| ≤ log M - log m := by
  have hv := prior_pos hR t k hk
  have hp := pmix_pos hR t k hk
  have hPk := hR.P_pos t k hk
  have hlo : m ≤ pmix R t := by
    have : ∑ i in R.s t, prior R t i * m ≤ ∑ i in R.s t, prior R t i * R.P t i :=
      sum_le_sum fun i hi => mul_le_mul_of_nonneg_left (hP i hi).1 (prior_nonneg hR t i)
    rw [← sum_mul, hsum, one_mul] at this
    exact this
  have hhi : pmix R t ≤ M := by
    have : ∑ i in R.s t, prior R t i * R.P t i ≤ ∑ i in R.s t, prior R t i * M :=
      sum_le_sum fun i hi => mul_le_mul_of_nonneg_left (hP i hi).2 (prior_nonneg hR t i)
    rw [← sum_mul, hsum, one_mul] at this
    exact this
  have e : log (post R t k) - log (prior R t k) = log (R.P t k) - log (pmix R t) := by
    show log (prior R t k * R.P t k / pmix R t) - _ = _
    rw [log_div (mul_pos hv hPk).ne' hp.ne', log_mul hv.ne' hPk.ne']; ring
  rw [e, abs_le]
  have h1 := log_le_log hm (hP k hk).1
  have h2 := log_le_log hPk (hP k hk).2
  have h3 := log_le_log hm hlo
  have h4 := log_le_log hp hhi
  constructor <;> linarith

end Main


end L38
end FCPA

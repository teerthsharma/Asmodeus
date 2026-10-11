import FCPA.L22H_HilbertLipschitz
import Mathlib.Topology.MetricSpace.Contracting
import Mathlib.Analysis.NormedSpace.FiniteDimension

/-!
L22H, fixed-point half (ERRATA 15; closes the gap left in `L22H_HilbertLipschitz`).
No completeness of the Hilbert metric on the open simplex is needed. Log-coordinates modulo constants are embedded isometrically
(`Eemb z = (k,l) ↦ z k - z l`, whose sup norm is the oscillation `max - min`) into the finite-dimensional, hence complete,
space `ι × ι → ℝ`; the image `range Eemb` is a linear subspace, hence closed, hence complete. On it the CTC map
`w ↦ softmax((β/T) A w)` becomes a `K`-contraction with `K = βD/(4T)`, and Mathlib's `ContractingWith` applies.

Main theorem `hilbert_ctc_fixed_point`: if `βD/4 < T` the undamped CTC loop has a unique fixed point among positive
probability vectors and `hdist (G^n w₀) w* ≤ K^n · hdist w₀ w*` (also the a-priori form with `1/(1-K)`),
where `hdist w w' = max_{k,l} [log(w_k/w'_k) - log(w_l/w'_l)]`-oscillation of `log w - log w'`.
-/
open BigOperators

namespace FCPA

/-! ### Abstract Banach step on a complete subset -/

theorem banach_subset {α : Type*} [MetricSpace α] (C : Set α) (hC : IsComplete C) (f : α → α)
    (hm : Set.MapsTo f C C) (K : NNReal) (hK : K < 1)
    (hf : ∀ x ∈ C, ∀ y ∈ C, dist (f x) (f y) ≤ K * dist x y) (x0 : α) (hx0 : x0 ∈ C) :
    ∃ y ∈ C, f y = y ∧ (∀ y' ∈ C, f y' = y' → y' = y) ∧
      (∀ x ∈ C, ∀ n : ℕ, dist (f^[n] x) y ≤ (K : ℝ) ^ n * dist x y) ∧
      (∀ x ∈ C, dist x y ≤ dist x (f x) / (1 - K)) := by
  haveI : CompleteSpace C := hC.completeSpace_coe
  haveI : Nonempty C := ⟨⟨x0, hx0⟩⟩
  set g : C → C := hm.restrict f C C with hg
  have hgv : ∀ z : C, ((g z : C) : α) = f z := fun z => rfl
  have hcontr : ContractingWith K g := by
    refine ⟨hK, LipschitzWith.of_dist_le_mul (fun x y => ?_)⟩
    rw [Subtype.dist_eq, Subtype.dist_eq, hgv, hgv]
    exact hf x x.2 y y.2
  set w : C := ContractingWith.fixedPoint g hcontr with hw
  have hwfix : g w = w := ContractingWith.fixedPoint_isFixedPt (hf := hcontr)
  have hwf : f (w : α) = w := by
    have := congrArg Subtype.val hwfix
    exact this
  have hiter : ∀ (x : C) (n : ℕ), (((g^[n]) x : C) : α) = (f^[n]) (x : α) := by
    intro x n
    induction n with
    | zero => rfl
    | succ n ih => rw [Function.iterate_succ_apply', Function.iterate_succ_apply', hgv, ih]
  refine ⟨w, w.2, hwf, ?_, ?_, ?_⟩
  · intro y' hy' hfix
    have : (⟨y', hy'⟩ : C) = w := by
      apply ContractingWith.fixedPoint_unique (hf := hcontr)
      exact Subtype.ext hfix
    exact congrArg Subtype.val this
  · intro x hx n
    have hlip := (hcontr.toLipschitzWith.iterate n).dist_le_mul ⟨x, hx⟩ w
    have hfixn : (g^[n]) w = w := (Function.IsFixedPt.iterate hwfix n)
    rw [hfixn, Subtype.dist_eq, Subtype.dist_eq, hiter] at hlip
    simpa using hlip
  · intro x hx
    have := ContractingWith.dist_le_of_fixedPoint (hf := hcontr) (⟨x, hx⟩ : C) hwfix
    rw [Subtype.dist_eq, Subtype.dist_eq, hgv] at this
    simpa using this

/-! ### Log-coordinates modulo constants -/

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- `z ↦ (k,l) ↦ z k - z l`; its sup norm is the oscillation `max z - min z`. -/
def Eemb (z : ι → ℝ) : ι × ι → ℝ := fun p => z p.1 - z p.2

theorem Eemb_sub (z z' : ι → ℝ) : Eemb z - Eemb z' = Eemb (fun k => z k - z' k) := by
  funext p; simp only [Eemb, Pi.sub_apply]; ring

theorem Eemb_add_const (z : ι → ℝ) (c : ℝ) : Eemb (fun k => z k + c) = Eemb z := by
  funext p; simp only [Eemb]; ring

theorem dist_Eemb (z z' : ι → ℝ) : dist (Eemb z) (Eemb z') = ‖Eemb (fun k => z k - z' k)‖ := by
  rw [dist_eq_norm, Eemb_sub]

theorem le_norm_Eemb (v : ι → ℝ) (k l : ι) : v k - v l ≤ ‖Eemb v‖ := by
  have h := norm_le_pi_norm (Eemb v) (k, l)
  rw [Real.norm_eq_abs] at h
  exact le_trans (le_abs_self _) h

theorem norm_Eemb_le (v : ι → ℝ) {δ : ℝ} (hδ : 0 ≤ δ) (h : ∀ k l, v k - v l ≤ δ) :
    ‖Eemb v‖ ≤ δ := by
  rw [pi_norm_le_iff_of_nonneg hδ]
  intro p
  rw [Real.norm_eq_abs, abs_le]
  constructor
  · have := h p.2 p.1; simp only [Eemb]; linarith
  · exact h p.1 p.2

theorem Eemb_eq_iff (i0 : ι) {z z' : ι → ℝ} (h : Eemb z = Eemb z') :
    z' = fun k => z k + (z' i0 - z i0) := by
  funext k
  have := congrFun h (k, i0)
  simp only [Eemb] at this
  linarith

/-- The Hilbert (oscillation) distance of positive vectors. -/
noncomputable def hdist (w w' : ι → ℝ) : ℝ := ‖Eemb (fun k => Real.log (w k) - Real.log (w' k))‖

/-- `range Eemb` is complete: a linear subspace of a finite-dimensional space. -/
theorem isComplete_range_Eemb : IsComplete (Set.range (Eemb : (ι → ℝ) → ι × ι → ℝ)) := by
  let L : (ι → ℝ) →ₗ[ℝ] (ι × ι → ℝ) :=
    { toFun := Eemb
      map_add' := fun z z' => by funext p; simp [Eemb]; ring
      map_smul' := fun c z => by funext p; simp [Eemb, mul_sub] }
  have h : IsClosed (LinearMap.range L : Set (ι × ι → ℝ)) :=
    Submodule.closed_of_finiteDimensional _
  have h2 : (LinearMap.range L : Set (ι × ι → ℝ)) = Set.range (Eemb : (ι → ℝ) → ι × ι → ℝ) :=
    LinearMap.range_coe L
  rw [← h2]
  exact h.isComplete

/-! ### Softmax facts -/

theorem softmax_sum [Nonempty ι] (z : ι → ℝ) : ∑ k, softmax z k = 1 := by
  unfold softmax
  rw [← Finset.sum_div]
  exact div_self (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty).ne'

theorem softmax_add_const [Nonempty ι] (z : ι → ℝ) (c : ℝ) :
    softmax (fun k => z k + c) = softmax z := by
  funext i
  have hS : 0 < ∑ j, Real.exp (z j) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  unfold softmax
  simp only [Real.exp_add]
  rw [← Finset.sum_mul]
  have := (Real.exp_pos c).ne'
  field_simp
  ring

theorem softmax_sub_const [Nonempty ι] (z : ι → ℝ) (c : ℝ) :
    softmax (fun k => z k - c) = softmax z := by
  simp only [sub_eq_add_neg]
  exact softmax_add_const z (-c)

theorem softmax_log [Nonempty ι] (w : ι → ℝ) (hw : ∀ k, 0 < w k) (hs : ∑ k, w k = 1) :
    softmax (fun k => Real.log (w k)) = w := by
  funext i
  unfold softmax
  simp only [Real.exp_log (hw _)]
  rw [hs, div_one]

theorem log_softmax [Nonempty ι] (z : ι → ℝ) (k : ι) :
    Real.log (softmax z k) = z k - Real.log (∑ j, Real.exp (z j)) := by
  have hy : 0 < ∑ j, Real.exp (z j) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  unfold softmax
  rw [Real.log_div (Real.exp_pos _).ne' hy.ne', Real.log_exp]

theorem Eemb_log_softmax [Nonempty ι] (z : ι → ℝ) :
    Eemb (fun k => Real.log (softmax z k)) = Eemb z := by
  have : (fun k => Real.log (softmax z k)) = fun k => z k + (-Real.log (∑ j, Real.exp (z j))) := by
    funext k; rw [log_softmax]; ring
  rw [this, Eemb_add_const]

/-! ### The CTC map in log-coordinates -/

variable [Nonempty ι]

/-- CTC map on probability vectors. -/
noncomputable def Gmap (A : ι → ι → ℝ) (β T : ℝ) (w : ι → ℝ) : ι → ℝ :=
  softmax (fun i => β / T * ∑ k, A i k * w k)

/-- Logit map in log-coordinates. -/
noncomputable def Phi (A : ι → ι → ℝ) (β T : ℝ) (z : ι → ℝ) : ι → ℝ :=
  fun i => β / T * ∑ k, A i k * softmax z k

theorem Phi_sub_const (A : ι → ι → ℝ) (β T : ℝ) (z : ι → ℝ) (c : ℝ) :
    Phi A β T (fun k => z k - c) = Phi A β T z := by
  unfold Phi
  rw [softmax_sub_const]

theorem Phi_log (A : ι → ι → ℝ) (β T : ℝ) (w : ι → ℝ) (hw : ∀ k, 0 < w k) (hs : ∑ k, w k = 1) :
    Phi A β T (fun k => Real.log (w k)) = fun i => β / T * ∑ k, A i k * w k := by
  funext i
  simp only [Phi, softmax_log w hw hs]

theorem Gmap_pos (A : ι → ι → ℝ) (β T : ℝ) (w : ι → ℝ) (k : ι) : 0 < Gmap A β T w k :=
  softmax_pos _ k

theorem Gmap_sum (A : ι → ι → ℝ) (β T : ℝ) (w : ι → ℝ) : ∑ k, Gmap A β T w k = 1 :=
  softmax_sum _

/-- The embedded map: `f (Eemb z) = Eemb (Phi z)`. -/
noncomputable def fmap (A : ι → ι → ℝ) (β T : ℝ) (i0 : ι) (y : ι × ι → ℝ) : ι × ι → ℝ :=
  Eemb (Phi A β T (fun k => y (k, i0)))

theorem fmap_Eemb (A : ι → ι → ℝ) (β T : ℝ) (i0 : ι) (z : ι → ℝ) :
    fmap A β T i0 (Eemb z) = Eemb (Phi A β T z) := by
  unfold fmap
  have : (fun k => Eemb z (k, i0)) = fun k => z k - z i0 := rfl
  rw [this, Phi_sub_const]

/-- Contraction estimate for `Phi` in the oscillation norm. -/
theorem Phi_osc_le (A : ι → ι → ℝ) (β T D : ℝ) (hT : 0 < T) (hβ : 0 ≤ β)
    (hD : ∀ i j k l, A i k - A j k - A i l + A j l ≤ D) (z z' : ι → ℝ) :
    ‖Eemb (fun k => Phi A β T z k - Phi A β T z' k)‖
      ≤ (β * D / 4 / T) * ‖Eemb (fun k => z k - z' k)‖ := by
  have hD0 : 0 ≤ D := by have := hD (Classical.arbitrary ι) (Classical.arbitrary ι)
                             (Classical.arbitrary ι) (Classical.arbitrary ι); linarith
  have hK : 0 ≤ β * D / 4 / T := by positivity
  apply norm_Eemb_le _ (mul_nonneg hK (norm_nonneg _))
  intro i j
  have key := softmax_hilbert_lipschitz A β T D hT hβ hD (softmax z) (softmax z')
    (softmax_pos z) (softmax_pos z') (softmax_sum z) (softmax_sum z') (‖Eemb (fun k => z k - z' k)‖)
    (fun k l => by
      have := softmax_ratio z z' k l
      have h2 := le_norm_Eemb (fun k => z k - z' k) k l
      rw [softmax_ratio]
      simp only at h2
      linarith) i j
  have e : Real.log (softmax (fun i => β / T * ∑ k, A i k * softmax z k) i
        / softmax (fun i => β / T * ∑ k, A i k * softmax z' k) i)
      - Real.log (softmax (fun i => β / T * ∑ k, A i k * softmax z k) j
        / softmax (fun i => β / T * ∑ k, A i k * softmax z' k) j)
      = (Phi A β T z i - Phi A β T z' i) - (Phi A β T z j - Phi A β T z' j) := by
    rw [softmax_ratio]; rfl
  rw [e] at key
  exact key

/-- Orbit link: the log-coordinates of `G w` are `f` of those of `w`. -/
theorem Eemb_log_Gmap (A : ι → ι → ℝ) (β T : ℝ) (i0 : ι) (w : ι → ℝ) (hw : ∀ k, 0 < w k)
    (hs : ∑ k, w k = 1) :
    Eemb (fun k => Real.log (Gmap A β T w k))
      = fmap A β T i0 (Eemb (fun k => Real.log (w k))) := by
  rw [fmap_Eemb, Phi_log A β T w hw hs]
  exact Eemb_log_softmax _

theorem Gmap_iter_prop (A : ι → ι → ℝ) (β T : ℝ) (w0 : ι → ℝ) (hp : ∀ k, 0 < w0 k)
    (hs : ∑ k, w0 k = 1) (m : ℕ) :
    (∀ k, 0 < ((Gmap A β T)^[m] w0) k) ∧ ∑ k, ((Gmap A β T)^[m] w0) k = 1 := by
  induction m with
  | zero => exact ⟨hp, hs⟩
  | succ m _ =>
    rw [Function.iterate_succ_apply']
    exact ⟨Gmap_pos A β T _, Gmap_sum A β T _⟩

theorem Eemb_log_iter (A : ι → ι → ℝ) (β T : ℝ) (i0 : ι) (w0 : ι → ℝ) (hp : ∀ k, 0 < w0 k)
    (hs : ∑ k, w0 k = 1) (m : ℕ) :
    Eemb (fun k => Real.log (((Gmap A β T)^[m] w0) k))
      = (fmap A β T i0)^[m] (Eemb (fun k => Real.log (w0 k))) := by
  induction m with
  | zero => rfl
  | succ m ih =>
    have h := Gmap_iter_prop A β T w0 hp hs m
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', ← ih,
      ← Eemb_log_Gmap A β T i0 _ h.1 h.2]

/-- **L22H, fixed-point form.** If `βD/4 < T` the undamped CTC loop `w ↦ softmax((β/T) A w)` has a unique fixed
point among positive probability vectors; it is approached geometrically in the Hilbert (oscillation) distance. -/
theorem hilbert_ctc_fixed_point (A : ι → ι → ℝ) (β T D : ℝ) (hT : 0 < T) (hβ : 0 ≤ β)
    (hD : ∀ i j k l, A i k - A j k - A i l + A j l ≤ D) (hκ : β * D / 4 < T) :
    ∃ w : ι → ℝ, (∀ k, 0 < w k) ∧ ∑ k, w k = 1 ∧ Gmap A β T w = w ∧
      (∀ w' : ι → ℝ, (∀ k, 0 < w' k) → ∑ k, w' k = 1 → Gmap A β T w' = w' → w' = w) ∧
      (∀ w0 : ι → ℝ, (∀ k, 0 < w0 k) → ∑ k, w0 k = 1 → ∀ n : ℕ,
        hdist ((Gmap A β T)^[n] w0) w ≤ (β * D / 4 / T) ^ n * hdist w0 w) ∧
      (∀ w0 : ι → ℝ, (∀ k, 0 < w0 k) → ∑ k, w0 k = 1 →
        hdist w0 w ≤ hdist w0 (Gmap A β T w0) / (1 - β * D / 4 / T)) := by
  have hD0 : 0 ≤ D := by
    have := hD (Classical.arbitrary ι) (Classical.arbitrary ι)
      (Classical.arbitrary ι) (Classical.arbitrary ι)
    linarith
  set i0 : ι := Classical.arbitrary ι with hi0
  set K0 : ℝ := β * D / 4 / T with hK0
  have hK0nn : 0 ≤ K0 := by positivity
  have hK1 : K0 < 1 := by rw [hK0, div_lt_one hT]; exact hκ
  set K : NNReal := ⟨K0, hK0nn⟩ with hK
  have hKc : (K : ℝ) = K0 := rfl
  set f := fmap A β T i0 with hf
  set C : Set (ι × ι → ℝ) := Set.range Eemb with hC
  have hm : Set.MapsTo f C C := fun y _ => ⟨_, rfl⟩
  have hcontr : ∀ x ∈ C, ∀ y ∈ C, dist (f x) (f y) ≤ K * dist x y := by
    rintro _ ⟨z, rfl⟩ _ ⟨z', rfl⟩
    rw [hf, fmap_Eemb, fmap_Eemb, dist_Eemb, dist_Eemb, hKc]
    exact Phi_osc_le A β T D hT hβ hD z z'
  obtain ⟨y, hyC, hfy, huniq, hrate, hap⟩ := banach_subset C isComplete_range_Eemb f hm K
    (by rw [← NNReal.coe_lt_coe]; simpa [hKc] using hK1) hcontr (Eemb (fun _ => 0)) ⟨_, rfl⟩
  obtain ⟨zs, rfl⟩ := hyC
  set w : ι → ℝ := softmax zs with hwdef
  have hwlog : Eemb (fun k => Real.log (w k)) = Eemb zs := Eemb_log_softmax zs
  have hwpos : ∀ k, 0 < w k := softmax_pos zs
  have hwsum : ∑ k, w k = 1 := softmax_sum zs
  refine ⟨w, hwpos, hwsum, ?_, ?_, ?_, ?_⟩
  · -- fixed point
    have h1 : Eemb (Phi A β T zs) = Eemb zs := by rw [← fmap_Eemb A β T i0 zs]; exact hfy
    have h2 := Eemb_eq_iff i0 h1.symm
    have h3 : Gmap A β T w = softmax (Phi A β T zs) := rfl
    rw [h3, h2, softmax_add_const]
  · intro w' hw'p hw's hfix
    have hy' : f (Eemb (fun k => Real.log (w' k))) = Eemb (fun k => Real.log (w' k)) := by
      rw [hf, ← Eemb_log_Gmap A β T i0 w' hw'p hw's, hfix]
    have := huniq _ ⟨_, rfl⟩ hy'
    have h4 : (fun k => Real.log (w' k)) = fun k => zs k + (Real.log (w' i0) - zs i0) :=
      Eemb_eq_iff i0 this.symm
    have h5 := softmax_log w' hw'p hw's
    rw [h4, softmax_add_const] at h5
    rw [← h5]
  · intro w0 hw0p hw0s n
    have := hrate (Eemb (fun k => Real.log (w0 k))) ⟨_, rfl⟩ n
    rw [← Eemb_log_iter A β T i0 w0 hw0p hw0s n, ← hwlog] at this
    unfold hdist
    rw [dist_Eemb, dist_Eemb] at this
    rw [hKc] at this
    exact this
  · intro w0 hw0p hw0s
    have := hap (Eemb (fun k => Real.log (w0 k))) ⟨_, rfl⟩
    rw [hf, ← Eemb_log_Gmap A β T i0 w0 hw0p hw0s, ← hwlog] at this
    unfold hdist
    rw [dist_Eemb, dist_Eemb] at this
    have e : (1 : ℝ) - (K : ℝ) = 1 - β * D / 4 / T := rfl
    rw [e] at this
    exact this

end FCPA

import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.BigOperators.Ring

/-!
L22H (ERRATA 15, Hilbert-metric CTC). The Hilbert (oscillation) distance of positive vectors is
`d_H(w,w') = max_{k,l} [log(w_k/w'_k) - log(w_l/w'_l)]`; below it appears as an explicit bound `d` on all those differences.
* `softmax_ratio`: `softmax` is an exact `d_H`-isometry up to its linear argument (the normaliser cancels).
* `hilbert_lipschitz`: for probability vectors `w, w'` and `c` of range `≤ r`:
  `|∑ c_k (w_k - w'_k)| ≤ r · d/4` (total variation `≤ d/4` plus the range bound).
* `softmax_hilbert_lipschitz` (L22H): `G_T(w) = softmax((β/T) A w)` satisfies
  `d_H(G_T w, G_T w') ≤ (κ_H / T) d_H(w, w')`, with `κ_H = (β/4) D`, `D ≥ A_ik - A_jk - A_il + A_jl` (sharp for the max).
  For `T > κ_H` the undamped CTC loop is a Hilbert contraction (Banach on the open simplex, which is `d_H`-complete; not proved here).
-/

open BigOperators

namespace FCPA

/-- `tanh(t/2) ≤ t/2` in the form `2(x-1) ≤ (x+1) log x` for `x ≥ 1`. -/
theorem two_mul_sub_one_le_log (x : ℝ) (hx : 1 ≤ x) : 2 * (x - 1) ≤ (x + 1) * Real.log x := by
  let f : ℝ → ℝ := fun x => (x + 1) * Real.log x - 2 * (x - 1)
  have hd : ∀ y ∈ Set.Ioi (1:ℝ), HasDerivAt f (Real.log y + (y + 1) / y - 2) y := by
    intro y hy
    have hy0 : y ≠ 0 := by have : (1:ℝ) < y := hy; linarith
    have h1 := ((hasDerivAt_id y).add_const 1).mul (Real.hasDerivAt_log hy0)
    have h2 := ((hasDerivAt_id y).sub_const 1).const_mul 2
    have := h1.sub h2
    convert this using 1
    simp only [id]
    field_simp
  have hmono : MonotoneOn f (Set.Ici 1) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici 1)
    · intro y hy
      have hy' : (1:ℝ) ≤ y := hy
      have hcont : ContinuousOn f (Set.Ici 1) :=
        ((continuousOn_id.add continuousOn_const).mul (Real.continuousOn_log.mono
          (fun z hz => ne_of_gt (lt_of_lt_of_le one_pos (Set.mem_Ici.mp hz))))).sub
          (continuousOn_const.mul (continuousOn_id.sub continuousOn_const))
      exact hcont y hy
    · rw [interior_Ici]
      intro y hy
      exact (hd y hy).differentiableAt.differentiableWithinAt
    · rw [interior_Ici]
      intro y hy
      rw [(hd y hy).deriv]
      have hy1 : (1:ℝ) < y := hy
      have hy0 : 0 < y := by linarith
      have h := Real.log_le_sub_one_of_pos (inv_pos.mpr hy0)
      rw [Real.log_inv] at h
      have e : (y + 1) / y = 1 + y⁻¹ := by field_simp
      rw [e]
      linarith
  have := hmono (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hx) hx
  simp only [f, Real.log_one] at this
  linarith

theorem exp_ineq (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    4 * ((Real.exp a - 1) * (Real.exp b - 1)) ≤ (a + b) * (Real.exp (a + b) - 1) := by
  have h1 := two_mul_sub_one_le_log (Real.exp a) (Real.one_le_exp ha)
  have h2 := two_mul_sub_one_le_log (Real.exp b) (Real.one_le_exp hb)
  rw [Real.log_exp] at h1 h2
  rw [Real.exp_add]
  have hu : 1 ≤ Real.exp a := Real.one_le_exp ha
  have hv : 1 ≤ Real.exp b := Real.one_le_exp hb
  generalize Real.exp a = u at *
  generalize Real.exp b = v at *
  have h3 : 2 * (u - 1) * (v + 1) ≤ a * (u + 1) * (v + 1) := by
    nlinarith [mul_le_mul_of_nonneg_right h1 (by linarith : (0:ℝ) ≤ v + 1)]
  have h4 : 2 * (v - 1) * (u + 1) ≤ b * (v + 1) * (u + 1) := by
    nlinarith [mul_le_mul_of_nonneg_right h2 (by linarith : (0:ℝ) ≤ u + 1)]
  have h5 : 4 * (u * v - 1) ≤ (a + b) * ((u + 1) * (v + 1)) := by nlinarith [h3, h4]
  have huv : 0 ≤ u * v - 1 := by nlinarith
  have h6 := mul_le_mul_of_nonneg_right h5 huv
  have hpos : 0 < (u + 1) * (v + 1) := by positivity
  refine le_of_mul_le_mul_right ?_ hpos
  nlinarith [h6, sq_nonneg (u - v)]

/-- Total-variation bound in Hilbert distance: `∑_{w ≥ q} (w - q) ≤ d/4`. -/
theorem tv_le {ι : Type*} [Fintype ι] [DecidableEq ι] (w q : ι → ℝ) (hq : ∀ k, 0 < q k)
    (hw : ∀ k, 0 < w k) (hw1 : ∑ k, w k = 1) (hq1 : ∑ k, q k = 1) (d : ℝ)
    (hd : ∀ k l, Real.log (w k / q k) - Real.log (w l / q l) ≤ d) :
    ∑ k in Finset.univ.filter (fun k => q k ≤ w k), (w k - q k) ≤ d / 4 := by
  have hne : Nonempty ι := by
    by_contra h
    rw [not_nonempty_iff] at h
    simp at hw1
  obtain ⟨kM, -, hkM⟩ := Finset.exists_max_image Finset.univ (fun k => Real.log (w k / q k))
    ⟨Classical.arbitrary ι, Finset.mem_univ _⟩
  obtain ⟨km, -, hkm⟩ := Finset.exists_min_image Finset.univ (fun k => Real.log (w k / q k))
    ⟨Classical.arbitrary ι, Finset.mem_univ _⟩
  set a := Real.log (w kM / q kM) with ha
  set b := - Real.log (w km / q km) with hb
  have hab : a + b ≤ d := by have := hd kM km; rw [ha, hb]; linarith
  have hup : ∀ k, w k ≤ Real.exp a * q k := by
    intro k
    have h := hkM k (Finset.mem_univ _)
    have h' : w k / q k ≤ Real.exp a := (Real.log_le_iff_le_exp (div_pos (hw k) (hq k))).mp h
    rwa [div_le_iff (hq k)] at h'
  have hlo : ∀ k, Real.exp (-b) * q k ≤ w k := by
    intro k
    have h := hkm k (Finset.mem_univ _)
    have h' : Real.exp (-b) ≤ w k / q k := by
      rw [hb, neg_neg, ← Real.exp_log (div_pos (hw k) (hq k))]
      exact Real.exp_le_exp.mpr (by simpa using h)
    rwa [le_div_iff (hq k)] at h'
  have hMge : 1 ≤ Real.exp a := by
    have : ∑ k, w k ≤ ∑ k, Real.exp a * q k := Finset.sum_le_sum (fun k _ => hup k)
    rw [← Finset.mul_sum, hw1, hq1] at this
    linarith
  have hmle : Real.exp (-b) ≤ 1 := by
    have : ∑ k, Real.exp (-b) * q k ≤ ∑ k, w k := Finset.sum_le_sum (fun k _ => hlo k)
    rw [← Finset.mul_sum, hw1, hq1] at this
    linarith
  have ha0 : 0 ≤ a := by
    have := Real.exp_le_exp.mp (show Real.exp 0 ≤ Real.exp a by simpa using hMge)
    linarith
  have hb0 : 0 ≤ b := by
    have := Real.exp_le_exp.mp (show Real.exp (-b) ≤ Real.exp 0 by simpa using hmle)
    linarith
  set P := Finset.univ.filter (fun k => q k ≤ w k) with hP
  set τ := ∑ k in P, (w k - q k) with hτ
  set Q := ∑ k in P, q k with hQ
  have hsplit : ∑ k in P, (w k - q k)
      + ∑ k in Finset.univ.filter (fun k => ¬ q k ≤ w k), (w k - q k) = 0 := by
    rw [hP, Finset.sum_filter_add_sum_filter_not, Finset.sum_sub_distrib, hw1, hq1]; ring
  have hQsplit : Q + ∑ k in Finset.univ.filter (fun k => ¬ q k ≤ w k), q k = 1 := by
    rw [hQ, hP, Finset.sum_filter_add_sum_filter_not, hq1]
  have h1 : τ ≤ (Real.exp a - 1) * Q := by
    rw [hτ, hQ, Finset.mul_sum]
    exact Finset.sum_le_sum (fun k _ => by nlinarith [hup k])
  have h2 : τ ≤ (1 - Real.exp (-b)) * (1 - Q) := by
    have hτ' : τ = ∑ k in Finset.univ.filter (fun k => ¬ q k ≤ w k), (q k - w k) := by
      have e : ∑ k in Finset.univ.filter (fun k => ¬ q k ≤ w k), (q k - w k)
          = - ∑ k in Finset.univ.filter (fun k => ¬ q k ≤ w k), (w k - q k) := by
        rw [← Finset.sum_neg_distrib]; apply Finset.sum_congr rfl; intros; ring
      rw [e]; linarith
    rw [hτ', show 1 - Q = ∑ k in Finset.univ.filter (fun k => ¬ q k ≤ w k), q k by linarith,
      Finset.mul_sum]
    exact Finset.sum_le_sum (fun k _ => by nlinarith [hlo k])
  have hQ0 : 0 ≤ Q := Finset.sum_nonneg (fun k _ => (hq k).le)
  have hτ_nonneg : 0 ≤ τ := Finset.sum_nonneg (fun k hk => by
    have := (Finset.mem_filter.mp hk).2; linarith)
  generalize hMdef : Real.exp a = M at *
  generalize hmdef : Real.exp (-b) = m at *
  have hM1 : 0 ≤ M - 1 := by linarith
  have hm1 : 0 ≤ 1 - m := by linarith
  have h3 : τ * (M - m) ≤ (M - 1) * (1 - m) := by
    nlinarith [mul_le_mul_of_nonneg_left h1 hm1, mul_le_mul_of_nonneg_left h2 hM1]
  have h4 : 4 * ((M - 1) * (1 - m)) ≤ (a + b) * (M - m) := by
    have := exp_ineq a b ha0 hb0
    have hem : Real.exp (-b) * Real.exp b = 1 := by rw [← Real.exp_add]; simp
    have hexp : Real.exp (a + b) = M * Real.exp b := by rw [Real.exp_add, hMdef]
    have e1 : M - m = m * (Real.exp (a + b) - 1) := by
      rw [hexp, ← hmdef]; nlinarith [hem]
    have e2 : (M - 1) * (1 - m) = m * ((M - 1) * (Real.exp b - 1)) := by
      rw [← hmdef]; nlinarith [hem]
    rw [e1, e2]
    have hpos : 0 < m := by rw [← hmdef]; exact Real.exp_pos _
    rw [hMdef] at this
    nlinarith [mul_le_mul_of_nonneg_left this hpos.le]
  have hMm : 0 ≤ M - m := by linarith
  have hd0 : 0 ≤ d := by linarith
  by_cases hMM : M - m = 0
  · have hM_eq : M = 1 := by linarith
    have : τ ≤ 0 := by
      have := h1; rw [hM_eq] at this; simpa using this
    linarith
  · have hpos : 0 < M - m := lt_of_le_of_ne hMm (Ne.symm hMM)
    have : 4 * τ * (M - m) ≤ d * (M - m) := by
      nlinarith [mul_le_mul_of_nonneg_right hab hMm]
    have := le_of_mul_le_mul_right this hpos
    linarith

/-- One-sided bound: `∑ c (w - q) ≤ r · ∑_{w ≥ q} (w - q)` for `c` of range `≤ r`. -/
theorem sum_le_range_tv {ι : Type*} [Fintype ι] [DecidableEq ι] [Nonempty ι] (w q c : ι → ℝ)
    (hw1 : ∑ k, w k = 1) (hq1 : ∑ k, q k = 1) (r : ℝ) (hc : ∀ k l, c k - c l ≤ r) :
    ∑ k, c k * (w k - q k)
      ≤ r * ∑ k in Finset.univ.filter (fun k => q k ≤ w k), (w k - q k) := by
  obtain ⟨k0, -, hk0⟩ := Finset.exists_min_image Finset.univ c
    ⟨Classical.arbitrary ι, Finset.mem_univ _⟩
  have hz : ∑ k, (w k - q k) = 0 := by rw [Finset.sum_sub_distrib, hw1, hq1]; ring
  have e : ∑ k, c k * (w k - q k) = ∑ k, (c k - c k0) * (w k - q k) := by
    have : ∑ k, (c k - c k0) * (w k - q k)
        = ∑ k, c k * (w k - q k) - c k0 * ∑ k, (w k - q k) := by
      rw [Finset.mul_sum, ← Finset.sum_sub_distrib]; apply Finset.sum_congr rfl; intros; ring
    rw [this, hz]; ring
  rw [e, ← Finset.sum_filter_add_sum_filter_not Finset.univ (fun k => q k ≤ w k), Finset.mul_sum]
  have hP : ∑ k in Finset.univ.filter (fun k => q k ≤ w k), (c k - c k0) * (w k - q k)
      ≤ ∑ k in Finset.univ.filter (fun k => q k ≤ w k), r * (w k - q k) :=
    Finset.sum_le_sum (fun k hk => mul_le_mul_of_nonneg_right (hc k k0)
      (by have := (Finset.mem_filter.mp hk).2; linarith))
  have hN : ∑ k in Finset.univ.filter (fun k => ¬ q k ≤ w k), (c k - c k0) * (w k - q k) ≤ 0 :=
    Finset.sum_nonpos (fun k hk => by
      have h1 := (Finset.mem_filter.mp hk).2
      exact mul_nonpos_of_nonneg_of_nonpos (sub_nonneg.mpr (hk0 k (Finset.mem_univ _)))
        (by linarith))
  linarith

theorem hilbert_lipschitz {ι : Type*} [Fintype ι] [DecidableEq ι] (w q c : ι → ℝ)
    (hq : ∀ k, 0 < q k) (hw : ∀ k, 0 < w k) (hw1 : ∑ k, w k = 1) (hq1 : ∑ k, q k = 1)
    (d r : ℝ) (hd : ∀ k l, Real.log (w k / q k) - Real.log (w l / q l) ≤ d)
    (hc : ∀ k l, c k - c l ≤ r) :
    |∑ k, c k * (w k - q k)| ≤ r * (d / 4) := by
  have hne : Nonempty ι := by
    by_contra h
    rw [not_nonempty_iff] at h
    simp at hw1
  have hr : 0 ≤ r := by
    have := hc (Classical.arbitrary ι) (Classical.arbitrary ι); linarith
  have htv := tv_le w q hq hw hw1 hq1 d hd
  have up := sum_le_range_tv w q c hw1 hq1 r hc
  have lo := sum_le_range_tv w q (fun k => - c k) hw1 hq1 r
    (fun k l => by show -c k - -c l ≤ r; have := hc l k; linarith)
  have hlo : - ∑ k, c k * (w k - q k)
      ≤ r * ∑ k in Finset.univ.filter (fun k => q k ≤ w k), (w k - q k) := by
    have : ∑ k, (-c k) * (w k - q k) = - ∑ k, c k * (w k - q k) := by
      rw [← Finset.sum_neg_distrib]; apply Finset.sum_congr rfl; intros; ring
    rwa [this] at lo
  have := mul_le_mul_of_nonneg_left htv hr
  rw [abs_le]
  constructor <;> linarith

/-- Softmax. -/
noncomputable def softmax {ι : Type*} [Fintype ι] (z : ι → ℝ) (i : ι) : ℝ :=
  Real.exp (z i) / ∑ j, Real.exp (z j)

theorem softmax_pos {ι : Type*} [Fintype ι] [Nonempty ι] (y : ι → ℝ) (k : ι) :
    0 < softmax y k :=
  div_pos (Real.exp_pos _) (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty)

theorem softmax_ratio {ι : Type*} [Fintype ι] [Nonempty ι] (z z' : ι → ℝ) (i j : ι) :
    Real.log (softmax z i / softmax z' i) - Real.log (softmax z j / softmax z' j)
      = (z i - z' i) - (z j - z' j) := by
  have hl : ∀ (y : ι → ℝ) (k : ι), Real.log (softmax y k)
      = y k - Real.log (∑ j, Real.exp (y j)) := by
    intro y k
    have hy : 0 < ∑ j, Real.exp (y j) :=
      Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
    unfold softmax
    rw [Real.log_div (Real.exp_pos _).ne' hy.ne', Real.log_exp]
  rw [Real.log_div (softmax_pos z i).ne' (softmax_pos z' i).ne',
    Real.log_div (softmax_pos z j).ne' (softmax_pos z' j).ne',
    hl z i, hl z' i, hl z j, hl z' j]
  ring

/-- **L22H.** In the oscillation (Hilbert) norm on log-coordinates, `w ↦ softmax((β/T) A w)` is
`(κ_H/T)`-Lipschitz with `κ_H = β D / 4`, where `D` bounds every second difference
`A_ik - A_jk - A_il + A_jl`. -/
theorem softmax_hilbert_lipschitz {ι : Type*} [Fintype ι] [DecidableEq ι] [Nonempty ι]
    (A : ι → ι → ℝ) (β T D : ℝ) (hT : 0 < T) (hβ : 0 ≤ β)
    (hD : ∀ i j k l, A i k - A j k - A i l + A j l ≤ D)
    (w w' : ι → ℝ) (hw : ∀ k, 0 < w k) (hw' : ∀ k, 0 < w' k)
    (hw1 : ∑ k, w k = 1) (hw1' : ∑ k, w' k = 1) (d : ℝ)
    (hd : ∀ k l, Real.log (w k / w' k) - Real.log (w l / w' l) ≤ d) (i j : ι) :
    Real.log (softmax (fun i => β / T * ∑ k, A i k * w k) i
        / softmax (fun i => β / T * ∑ k, A i k * w' k) i)
      - Real.log (softmax (fun i => β / T * ∑ k, A i k * w k) j
        / softmax (fun i => β / T * ∑ k, A i k * w' k) j)
      ≤ (β * D / 4 / T) * d := by
  rw [softmax_ratio]
  have key := hilbert_lipschitz w w' (fun k => A i k - A j k) hw' hw hw1 hw1' d D hd
    (fun k l => by show A i k - A j k - (A i l - A j l) ≤ D; have := hD i j k l; linarith)
  have hsum : ((β / T * ∑ k, A i k * w k) - (β / T * ∑ k, A i k * w' k))
      - ((β / T * ∑ k, A j k * w k) - (β / T * ∑ k, A j k * w' k))
      = β / T * ∑ k, (A i k - A j k) * (w k - w' k) := by
    have e : ∑ k, (A i k - A j k) * (w k - w' k)
        = (∑ k, A i k * w k - ∑ k, A i k * w' k) - (∑ k, A j k * w k - ∑ k, A j k * w' k) := by
      simp only [sub_mul, mul_sub, Finset.sum_sub_distrib]
      ring
    rw [e]
    ring
  rw [hsum]
  have hbT : 0 ≤ β / T := div_nonneg hβ hT.le
  calc β / T * ∑ k, (A i k - A j k) * (w k - w' k)
      ≤ β / T * |∑ k, (A i k - A j k) * (w k - w' k)| :=
        mul_le_mul_of_nonneg_left (le_abs_self _) hbT
    _ ≤ β / T * (D * (d / 4)) := mul_le_mul_of_nonneg_left key hbT
    _ = (β * D / 4 / T) * d := by ring

end FCPA

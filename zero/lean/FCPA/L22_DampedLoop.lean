import Mathlib.Topology.MetricSpace.Contracting
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.Convex.Basic

/-!
L22* Damped loop on a fixed support (spec §3.9 + Amendment A). Abstract form: `C` a complete convex set of diameter ≤ 2 in a normed
space `V` (the simplex `Δ_{|T|+1}` in `ℓ¹`), `G` an `L`-Lipschitz self-map of `C`, `u ∈ C` (the uniform point),
`F w = (1-ε) G w + ε u`, and the contraction hypothesis `c = (1-ε) L < 1` stated EXPLICITLY (the README's "ε > 0 alone" is wrong).
Conclusions: unique fixed point `w`; geometric convergence `dist (F^n x) w ≤ dist x (F x) c^n/(1-c)`;
residual `dist (G w) w ≤ 2ε`; and bias to any undamped fixed point `w_G` of `G`: `dist w w_G ≤ 2ε/(1-c)`.
-/

namespace FCPA

theorem damped_loop {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] (C : Set V)
    (hconv : Convex ℝ C) (hcl : IsComplete C) (hdiam : ∀ x ∈ C, ∀ y ∈ C, dist x y ≤ 2)
    (G : V → V) (hGC : Set.MapsTo G C C) (L : ℝ) (hL : 0 ≤ L)
    (hG : ∀ x ∈ C, ∀ y ∈ C, dist (G x) (G y) ≤ L * dist x y)
    (u : V) (hu : u ∈ C) (ε : ℝ) (hε : 0 ≤ ε ∧ ε ≤ 1) (hc : (1 - ε) * L < 1) :
    ∃ w ∈ C, w = (1 - ε) • G w + ε • u ∧
      (∀ w' ∈ C, w' = (1 - ε) • G w' + ε • u → w' = w) ∧
      (∀ x ∈ C, ∀ n : ℕ, dist (((fun z => (1 - ε) • G z + ε • u)^[n]) x) w ≤
          dist x ((1 - ε) • G x + ε • u) * ((1 - ε) * L) ^ n / (1 - (1 - ε) * L)) ∧
      (∀ x ∈ C, Filter.Tendsto (fun n : ℕ => ((fun z => (1 - ε) • G z + ε • u)^[n]) x)
          Filter.atTop (nhds w)) ∧
      dist (G w) w ≤ 2 * ε ∧
      (∀ wg ∈ C, G wg = wg → dist w wg ≤ 2 * ε / (1 - (1 - ε) * L)) := by
  set F : V → V := fun z => (1 - ε) • G z + ε • u with hF
  have hFC : Set.MapsTo F C C := fun z hz =>
    hconv (hGC hz) hu (by linarith [hε.2]) hε.1 (by ring)
  have hFd : ∀ x ∈ C, ∀ y ∈ C, dist (F x) (F y) ≤ (1 - ε) * L * dist x y := by
    intro x hx y hy
    have : dist (F x) (F y) = (1 - ε) * dist (G x) (G y) := by
      rw [dist_eq_norm, dist_eq_norm]
      have : F x - F y = (1 - ε) • (G x - G y) := by
        simp only [hF]; rw [smul_sub]; abel
      rw [this, norm_smul, Real.norm_of_nonneg (by linarith [hε.2])]
    rw [this, mul_assoc]
    exact mul_le_mul_of_nonneg_left (hG x hx y hy) (by linarith [hε.2])
  have hK0 : 0 ≤ (1 - ε) * L := mul_nonneg (by linarith [hε.2]) hL
  set K : NNReal := ⟨(1 - ε) * L, hK0⟩ with hKdef
  have hKc : (K : ℝ) = (1 - ε) * L := rfl
  haveI : CompleteSpace C := hcl.completeSpace_coe
  haveI : Nonempty C := ⟨⟨u, hu⟩⟩
  set g : C → C := hFC.restrict F C C with hg
  have hgv : ∀ z : C, ((g z : C) : V) = F z := fun z => rfl
  have hcontr : ContractingWith K g := by
    refine ⟨by rw [← NNReal.coe_lt_coe]; simpa [hKc] using hc, ?_⟩
    refine LipschitzWith.of_dist_le_mul (fun x y => ?_)
    rw [Subtype.dist_eq, Subtype.dist_eq, hgv, hgv, hKc]
    exact hFd x x.2 y y.2
  set w : C := ContractingWith.fixedPoint g hcontr with hw
  have hwfix : g w = w := ContractingWith.fixedPoint_isFixedPt (hf := hcontr)
  have hwF : (w : V) = F w := by
    have := congrArg Subtype.val hwfix
    exact this.symm
  have hiter : ∀ (x : C) (n : ℕ), (((g^[n]) x : C) : V) = (F^[n]) (x : V) := by
    intro x n
    induction n with
    | zero => rfl
    | succ n ih =>
      rw [Function.iterate_succ_apply', Function.iterate_succ_apply', hgv, ih]
  have hdistG : dist (G w) (w : V) ≤ 2 * ε := by
    have h1 : (w : V) - G w = ε • (u - G w) := by
      have e : (w : V) = (1 - ε) • G w + ε • u := hwF
      generalize G (w : V) = gw at e ⊢
      rw [e, smul_sub, sub_smul, one_smul]
      abel
    rw [dist_comm, dist_eq_norm, h1, norm_smul, Real.norm_of_nonneg hε.1, ← dist_eq_norm]
    have := hdiam u hu (G w) (hGC w.2)
    nlinarith [hε.1]
  refine ⟨w, w.2, hwF, ?_, ?_, ?_, hdistG, ?_⟩
  · intro w' hw' hfix
    have : (⟨w', hw'⟩ : C) = w := by
      apply ContractingWith.fixedPoint_unique (hf := hcontr)
      exact Subtype.ext hfix.symm
    exact congrArg Subtype.val this
  · intro x hx n
    have := ContractingWith.apriori_dist_iterate_fixedPoint_le (hf := hcontr) ⟨x, hx⟩ n
    rw [Subtype.dist_eq, Subtype.dist_eq, hiter, hKc] at this
    simpa [hgv, hKc] using this
  · intro x hx
    have := ContractingWith.tendsto_iterate_fixedPoint (hf := hcontr) ⟨x, hx⟩
    have h2 := (continuous_subtype_val.tendsto w).comp this
    refine h2.congr (fun n => ?_)
    exact hiter ⟨x, hx⟩ n
  · intro wg hwg hGwg
    have hy := ContractingWith.dist_le_of_fixedPoint (hf := hcontr) (⟨wg, hwg⟩ : C) hwfix
    rw [Subtype.dist_eq, Subtype.dist_eq, hgv, hKc] at hy
    have hres : dist wg (F wg) ≤ 2 * ε := by
      have h1 : wg - F wg = ε • (wg - u) := by
        show wg - ((1 - ε) • G wg + ε • u) = ε • (wg - u)
        rw [hGwg, smul_sub, sub_smul, one_smul]
        abel
      rw [dist_eq_norm, h1, norm_smul, Real.norm_of_nonneg hε.1, ← dist_eq_norm]
      have := hdiam wg hwg u hu
      nlinarith [hε.1]
    have hpos : 0 < 1 - (1 - ε) * L := by linarith
    rw [dist_comm]
    calc dist wg (w : V) ≤ dist wg (F wg) / (1 - (1 - ε) * L) := by
          simpa [hKc] using hy
      _ ≤ 2 * ε / (1 - (1 - ε) * L) := div_le_div_of_nonneg_right hres hpos.le

end FCPA

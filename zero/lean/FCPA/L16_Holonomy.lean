import FCPA.L16a_ExpRemainder
import FCPA.L15_CrossingComposition

/-!
L16 (ERRATA 1 sign): the holonomy of a triangle of crossings. With later crossings composed on the left, a closed loop
`i → j → k → i` with generator steps `A` (i→j), `B` (j→k) and `-(A+B)` (k→i) returns
`exp(-(A+B)) exp(B) exp(A) = 1 - ½[A,B] + O(ε³)`, where `[A,B] = AB - BA`: the loop measures the NON-COMMUTATIVITY of the
generators (not "curvature of law space", which is flat). Proved here as an explicit bound
`‖hol - 1 + ½[A,B]‖ ≤ 432 ε³` for `‖A‖,‖B‖ ≤ ε ≤ ½` in any complete normed algebra, then for the crossing operators of
`L15` (matrices with the `ℓ^∞` operator norm). For commuting generators `[A,B] = 0` and L15 gives holonomy exactly `1`.
-/
open NormedSpace BigOperators

namespace FCPA

section General

variable {𝔸 : Type*} [NormedRing 𝔸] [NormedAlgebra ℝ 𝔸] [CompleteSpace 𝔸]

theorem holonomy_quadratic (A B : 𝔸) :
    (1 / 2 : ℝ) • ((-(A + B)) ^ 2 + B ^ 2 + A ^ 2)
      + (-(A + B) * B + -(A + B) * A + B * A) + (1 / 2 : ℝ) • (A * B - B * A) = 0 := by
  have : (2 : ℝ) • ((1 / 2 : ℝ) • ((-(A + B)) ^ 2 + B ^ 2 + A ^ 2)
      + (-(A + B) * B + -(A + B) * A + B * A) + (1 / 2 : ℝ) • (A * B - B * A)) = 0 := by
    rw [smul_add, smul_add, smul_smul, smul_smul, two_smul ℝ (-(A + B) * B + -(A + B) * A + B * A)]
    norm_num
    noncomm_ring
  exact (smul_eq_zero.mp this).resolve_left two_ne_zero

theorem holonomy_bch2 (A B : 𝔸) {ε : ℝ} (hε0 : 0 ≤ ε) (hε : ε ≤ 1 / 2)
    (hA : ‖A‖ ≤ ε) (hB : ‖B‖ ≤ ε) :
    ‖exp ℝ (-(A + B)) * exp ℝ B * exp ℝ A - 1 + (1 / 2 : ℝ) • (A * B - B * A)‖ ≤ 432 * ε ^ 3 := by
  set δ : ℝ := 2 * ε with hδ
  have hδ0 : 0 ≤ δ := by positivity
  have hδ1 : δ ≤ 1 := by linarith
  set X1 : 𝔸 := -(A + B) with hX1
  have n1 : ‖X1‖ ≤ δ := by
    rw [hX1, norm_neg]; exact (norm_add_le _ _).trans (by linarith)
  have n2 : ‖B‖ ≤ δ := by linarith
  have n3 : ‖A‖ ≤ δ := by linarith
  -- remainders
  have remainder : ∀ X : 𝔸, ‖X‖ ≤ δ →
      ∃ w R : 𝔸, exp ℝ X = 1 + X + w ∧ w = (1 / 2 : ℝ) • X ^ 2 + R ∧ ‖R‖ ≤ δ ^ 3 ∧ ‖w‖ ≤ 2 * δ ^ 2 := by
    intro X hX
    have hX1' : ‖X‖ ≤ 1 := hX.trans hδ1
    refine ⟨(1 / 2 : ℝ) • X ^ 2 + (exp ℝ X - (1 + X + (1 / 2 : ℝ) • X ^ 2)),
      exp ℝ X - (1 + X + (1 / 2 : ℝ) • X ^ 2), by abel, rfl, ?_, ?_⟩
    · exact (exp_remainder_le X hX1').trans (pow_le_pow_left (norm_nonneg X) hX 3)
    · have hR : ‖exp ℝ X - (1 + X + (1 / 2 : ℝ) • X ^ 2)‖ ≤ δ ^ 3 :=
        (exp_remainder_le X hX1').trans (pow_le_pow_left (norm_nonneg X) hX 3)
      have hsq : ‖(1 / 2 : ℝ) • X ^ 2‖ ≤ δ ^ 2 / 2 := by
        rw [norm_smul, Real.norm_eq_abs, abs_of_pos (by norm_num : (0 : ℝ) < 1 / 2)]
        have : ‖X ^ 2‖ ≤ ‖X‖ ^ 2 := norm_pow_le' X (by norm_num)
        have h2 : ‖X‖ ^ 2 ≤ δ ^ 2 := pow_le_pow_left (norm_nonneg X) hX 2
        linarith
      have hd3 : δ ^ 3 ≤ δ ^ 2 := pow_le_pow_of_le_one hδ0 hδ1 (by norm_num)
      calc _ ≤ ‖(1 / 2 : ℝ) • X ^ 2‖ + ‖exp ℝ X - (1 + X + (1 / 2 : ℝ) • X ^ 2)‖ := norm_add_le _ _
        _ ≤ 2 * δ ^ 2 := by nlinarith [sq_nonneg δ]
  obtain ⟨w1, R1, e1, hw1, hR1, g1⟩ := remainder X1 n1
  obtain ⟨w2, R2, e2, hw2, hR2, g2⟩ := remainder B n2
  obtain ⟨w3, R3, e3, hw3, hR3, g3⟩ := remainder A n3
  have key := triple_expansion X1 B A w1 w2 w3 hδ0 hδ1 n1 n2 n3 g1 g2 g3
  rw [e1, e2, e3]
  have hI := holonomy_quadratic A B
  have eq : (1 + X1 + w1) * (1 + B + w2) * (1 + A + w3) - 1 + (1 / 2 : ℝ) • (A * B - B * A)
      = ((1 + X1 + w1) * (1 + B + w2) * (1 + A + w3)
          - (1 + (X1 + B + A) + (w1 + w2 + w3) + (X1 * B + X1 * A + B * A))) + (R1 + R2 + R3) := by
    have hsum : w1 + w2 + w3 = (1 / 2 : ℝ) • (X1 ^ 2 + B ^ 2 + A ^ 2) + (R1 + R2 + R3) := by
      rw [hw1, hw2, hw3, smul_add, smul_add]; abel
    have hX0 : X1 + B + A = 0 := by rw [hX1]; abel
    have hI' : (1 / 2 : ℝ) • (X1 ^ 2 + B ^ 2 + A ^ 2) + (X1 * B + X1 * A + B * A)
        + (1 / 2 : ℝ) • (A * B - B * A) = 0 := by rw [hX1]; exact hI
    rw [hsum, hX0]
    have : (1 / 2 : ℝ) • (A * B - B * A) = -((1 / 2 : ℝ) • (X1 ^ 2 + B ^ 2 + A ^ 2) + (X1 * B + X1 * A + B * A)) := by
      rw [eq_neg_iff_add_eq_zero]; rw [← hI']; abel
    rw [this]; abel
  rw [eq]
  calc _ ≤ ‖(1 + X1 + w1) * (1 + B + w2) * (1 + A + w3)
          - (1 + (X1 + B + A) + (w1 + w2 + w3) + (X1 * B + X1 * A + B * A))‖ + ‖R1 + R2 + R3‖ :=
        norm_add_le _ _
    _ ≤ 51 * δ ^ 3 + (δ ^ 3 + δ ^ 3 + δ ^ 3) := by
        gcongr
        exact ((norm_add_le _ _).trans (add_le_add (norm_add_le _ _) le_rfl)).trans
          (by linarith)
    _ = 432 * ε ^ 3 := by rw [hδ]; ring

end General

section MatrixInst

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n] {d : ℕ}

attribute [local instance] Matrix.linftyOpNormedRing Matrix.linftyOpNormedAlgebra

/-- Generator step `Σ_a x_a G_a`. -/
noncomputable def genStep (G : Fin d → Matrix n n ℂ) (x : Fin d → ℝ) : Matrix n n ℂ := ∑ a, (x a : ℂ) • G a

theorem genStep_add (G : Fin d → Matrix n n ℂ) (x y : Fin d → ℝ) :
    genStep G (x + y) = genStep G x + genStep G y := by
  unfold genStep
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro a _
  rw [← add_smul]; congr 1; rw [Pi.add_apply]; push_cast; ring

theorem genStep_neg (G : Fin d → Matrix n n ℂ) (x : Fin d → ℝ) :
    genStep G (-x) = -genStep G x := by
  unfold genStep
  rw [← Finset.sum_neg_distrib]
  apply Finset.sum_congr rfl
  intro a _
  rw [← neg_smul]; congr 1; rw [Pi.neg_apply]; push_cast; ring

/-- **L16 (second-order BCH, finite dimension).** The triangle holonomy of crossing operators satisfies
`‖S(-(a+b)) S(b) S(a) - 1 + ½[A,B]‖ ≤ 432 ε³` when the generator steps `A = Σ aₖGₖ`, `B = Σ bₖGₖ` have norm `≤ ε ≤ ½`
(norm: the `ℓ^∞`-operator norm of matrices). In particular the holonomy measures the commutator `[A,B]`. -/
theorem crossing_holonomy (G : Fin d → Matrix n n ℂ) (a b : Fin d → ℝ) {ε : ℝ} (hε0 : 0 ≤ ε)
    (hε : ε ≤ 1 / 2) (hA : ‖genStep G a‖ ≤ ε) (hB : ‖genStep G b‖ ≤ ε) :
    ‖crossing G (-(a + b)) * crossing G b * crossing G a - 1
        + (1 / 2 : ℝ) • (genStep G a * genStep G b - genStep G b * genStep G a)‖ ≤ 432 * ε ^ 3 := by
  have hc : ∀ x : Fin d → ℝ, crossing G x = exp ℝ (genStep G x) := by
    intro x
    unfold crossing genStep
    exact congrFun (NormedSpace.exp_eq_exp ℂ ℝ (Matrix n n ℂ)) _
  rw [hc, hc, hc, genStep_neg, genStep_add]
  exact holonomy_bch2 (genStep G a) (genStep G b) hε0 hε hA hB

end MatrixInst

end FCPA

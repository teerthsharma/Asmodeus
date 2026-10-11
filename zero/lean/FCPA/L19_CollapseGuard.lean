import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
L19 collapse guard (spec §3.6). The era loss is prediction loss plus SIGReg, whose scope is the isotropic-covariance
constraint. The collapse test is the rank of the embeddings. Here: a batch of `n ≥ 1` embeddings `z_i ∈ ℝᵈ` with
empirical (second-moment) covariance `C = (1/n) Σ_i z_i z_iᵀ`.
* `quad_cov`: the quadratic form of `C` at `v` is `(1/n) Σ_i ⟨v, z_i⟩²`.
* `collapse_violates_isotropy`: if the embeddings lie in a hyperplane (`∃ v ≠ 0, ∀ i, ⟨v, z_i⟩ = 0`, i.e. dimensional
  collapse, rank `< d`) then `C ≠ I`, so the isotropy constraint cannot hold; contrapositive: `C = I` forces the
  embeddings to span `ℝᵈ` (no collapse).
-/
open BigOperators

namespace FCPA

variable {n d : ℕ}

/-- Empirical second-moment covariance of the embeddings. -/
noncomputable def embCov (z : Fin n → Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun a b => (n : ℝ)⁻¹ * ∑ i, z i a * z i b

theorem quad_cov (z : Fin n → Fin d → ℝ) (v : Fin d → ℝ) :
    ∑ a, ∑ b, v a * v b * embCov z a b = (n : ℝ)⁻¹ * ∑ i, (∑ a, v a * z i a) ^ 2 := by
  have hR : ∀ i, (∑ a, v a * z i a) ^ 2 = ∑ a, ∑ b, (v a * z i a) * (v b * z i b) := fun i => by
    rw [pow_two, Finset.sum_mul_sum]
  simp_rw [hR]
  simp only [embCov, Matrix.of_apply, Finset.mul_sum]
  calc ∑ a, ∑ b, ∑ i, v a * v b * ((n : ℝ)⁻¹ * (z i a * z i b))
      = ∑ a, ∑ i, ∑ b, v a * v b * ((n : ℝ)⁻¹ * (z i a * z i b)) :=
        Finset.sum_congr rfl (fun a _ => Finset.sum_comm)
    _ = ∑ i, ∑ a, ∑ b, v a * v b * ((n : ℝ)⁻¹ * (z i a * z i b)) := Finset.sum_comm
    _ = ∑ i, ∑ a, ∑ b, (n : ℝ)⁻¹ * (v a * z i a * (v b * z i b)) :=
        Finset.sum_congr rfl (fun i _ => Finset.sum_congr rfl (fun a _ =>
          Finset.sum_congr rfl (fun b _ => by ring)))

theorem collapse_violates_isotropy (z : Fin n → Fin d → ℝ) (v : Fin d → ℝ) (hv : v ≠ 0)
    (hflat : ∀ i, ∑ a, v a * z i a = 0) : embCov z ≠ 1 := by
  intro hC
  have h1 := quad_cov z v
  have h2 : ∑ i, (∑ a, v a * z i a) ^ 2 = 0 := Finset.sum_eq_zero (fun i _ => by rw [hflat i]; simp)
  rw [h2, mul_zero] at h1
  have h3 : ∑ a, ∑ b, v a * v b * embCov z a b = ∑ a, v a ^ 2 := by
    rw [hC]
    apply Finset.sum_congr rfl
    intro a _
    rw [Finset.sum_eq_single a]
    · simp [pow_two]
    · intro b _ hb; simp [Matrix.one_apply_ne (Ne.symm hb)]
    · intro h; exact absurd (Finset.mem_univ a) h
  rw [h3] at h1
  obtain ⟨a, ha⟩ : ∃ a, v a ≠ 0 := by
    by_contra hcon
    push_neg at hcon
    exact hv (funext hcon)
  have : 0 < ∑ a, v a ^ 2 :=
    Finset.sum_pos' (fun a _ => sq_nonneg _) ⟨a, Finset.mem_univ a, by positivity⟩
  linarith

/-- Contrapositive: isotropic covariance rules out any flat direction (the collapse guard passes). -/
theorem isotropic_no_flat_direction (z : Fin n → Fin d → ℝ) (hC : embCov z = 1) (v : Fin d → ℝ)
    (hv : v ≠ 0) : ¬ ∀ i, ∑ a, v a * z i a = 0 :=
  fun hflat => collapse_violates_isotropy z v hv hflat hC

end FCPA

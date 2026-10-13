import FCPA.L38_GrowingBankRegret

/-!
L41 (tournament winner, `zero/research/VERDICT_TOURNAMENT.md`): the MOM decider (`zero/research/e0/deciders.py::Momentum`)
is an HMM forward filter over the era bank, and its regret is L38 with switch cost `-log((α/2) Q(j,i))` along the
learned era-to-era kernel.

MOM's move step, as coded: `v = (1-α) w + (α/2)/n · 1 + (α/2) · (w Q̃)`, where `Q̃` is the kernel `Q` on eras
(`Q = (switch counts + 1)` with zero diagonal, row-normalised) and the identity on tail experts (window, BOCPD).

* `mom_is_move`: MOM is `L38.move` with `ρ = α/2`, `K = (α/2) Q̃`; so **every** L38 theorem applies to MOM.
* `mom_transition_stochastic`, `mom_eq_hmm_predict`: for a probability vector `w`, `v = w A` with
  `A = (1-α) I + (α/2)(1/n) J + (α/2) Q̃` row-stochastic: MOM is the forward filter of an HMM over the experts with
  transition `A` (its Bayes step is the posterior by `hmm_filter_is_posterior`, L14′).
* `mom_switch_cost`: a comparator switch from live era `j` to `i` costs at most
  `-log((α/2) Q(j,i)) + births · log(1/(1-α))`, and never more than `log(2N/α)`.
* `momQ_pos`, `momQ_row_sum`: the coded `Q` is a strictly positive off-diagonal stochastic kernel, so every switch has a
  finite kernel cost `log(2/α) + log((S_j + n_e - 1)/(S_{ji} + 1))` (`S` = switch counts).
* `soft_is_move`, `mom_identity_is_soft`, `mom_uniform_is_soft`: SOFT is `ρ = α, K = 0`; MOM with `Q̃ = I` is SOFT at
  rate `α/2`; MOM with the uniform kernel `Q̃ = J/n` is SOFT at rate `α` (for a probability vector `w`).
The kernel may change every step (the bound is pathwise with the kernel in force), so the **learned** `Q` is covered;
what stays open is how fast the learned `Q` approaches the true era-switch law.
-/
open BigOperators Real Finset

namespace FCPA
namespace L41

open L38

variable {ι : Type*} [DecidableEq ι]

/-- MOM's move step, as coded. -/
noncomputable def momMove (α : ℝ) (s : Finset ι) (Qt : ι → ι → ℝ) (w : ι → ℝ) : ι → ℝ :=
  fun i => (1 - α) * w i + (α / 2) / s.card + (α / 2) * ∑ j in s, w j * Qt j i

theorem mom_is_move (α : ℝ) (s : Finset ι) (Qt : ι → ι → ℝ) (w : ι → ℝ) :
    momMove α s Qt w = move α (α / 2) s (fun j i => α / 2 * Qt j i) w := by
  funext i
  simp only [momMove, move, mul_sum]
  congr 1
  apply sum_congr rfl
  intro j _
  ring

/-- SOFT's share step. -/
noncomputable def softShare (α : ℝ) (s : Finset ι) (w : ι → ℝ) : ι → ℝ :=
  fun i => (1 - α) * w i + α / s.card

theorem soft_is_move (α : ℝ) (s : Finset ι) (w : ι → ℝ) : softShare α s w = move α α s (fun _ _ => 0) w := by
  funext i
  simp [softShare, move]

/-- MOM with the identity kernel is SOFT at half the rate (on the live pool). -/
theorem mom_identity_is_soft (α : ℝ) (s : Finset ι) (w : ι → ℝ) (i : ι) (hi : i ∈ s) :
    momMove α s (fun j i => if j = i then 1 else 0) w i = softShare (α / 2) s w i := by
  simp only [momMove, softShare, mul_ite, mul_one, mul_zero, sum_ite_eq', if_pos hi]
  ring

/-- MOM with the uniform kernel `J/n` is SOFT at the full rate, for a probability vector `w` on the pool. -/
theorem mom_uniform_is_soft (α : ℝ) (s : Finset ι) (w : ι → ℝ) (hw : ∑ j in s, w j = 1) (i : ι) :
    momMove α s (fun _ _ => 1 / s.card) w i = softShare α s w i := by
  simp only [momMove, softShare, ← sum_mul, hw, one_mul]
  ring

/-- The HMM transition matrix of MOM. -/
noncomputable def momA (α : ℝ) (s : Finset ι) (Qt : ι → ι → ℝ) (j i : ι) : ℝ :=
  (1 - α) * (if j = i then 1 else 0) + (α / 2) / s.card + (α / 2) * Qt j i

/-- `A` is row-stochastic on the pool when `Q̃` is. -/
theorem mom_transition_stochastic (α : ℝ) (s : Finset ι) (hs : s.Nonempty) (Qt : ι → ι → ℝ) (j : ι) (hj : j ∈ s)
    (hQ : ∑ i in s, Qt j i = 1) : ∑ i in s, momA α s Qt j i = 1 := by
  have hc : (s.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (card_pos.mpr hs).ne'
  simp only [momA, sum_add_distrib, ← mul_sum, sum_ite_eq, if_pos hj, sum_const, nsmul_eq_mul, hQ]
  field_simp
  ring

/-- **MOM is an HMM forward filter**: its move step is the HMM predict step `v_i = Σ_j w_j A_{ji}`. -/
theorem mom_eq_hmm_predict (α : ℝ) (s : Finset ι) (Qt : ι → ι → ℝ) (w : ι → ℝ) (hw : ∑ j in s, w j = 1) (i : ι)
    (hi : i ∈ s) : momMove α s Qt w i = ∑ j in s, w j * momA α s Qt j i := by
  simp only [momMove, momA, mul_add, sum_add_distrib, mul_ite, mul_one, mul_zero, sum_ite_eq', if_pos hi,
    ← sum_mul, hw, one_mul]
  congr 1
  · ring
  · rw [mul_sum]; apply sum_congr rfl; intro j _; ring

/-- **L41, regret.** In a MOM run (`ρ = α/2`, `K_t = (α/2) Q̃_t` with any, possibly learned, `Q̃_t ≥ 0`), a comparator
switch from live era `j` to era `i` with `Q̃(j,i) > 0` costs at most `-log((α/2) Q̃(j,i)) + births · log(1/(1-α))`;
any switch costs at most `log(2N/α)`. With `bank_regret` this bounds MOM's regret against the best piecewise path. -/
theorem mom_switch_cost {R : Run ι} (hR : R.Valid) (Qt : ℕ → ι → ι → ℝ) (hρ : R.ρ = R.α / 2)
    (hK : ∀ t j i, R.K t j i = R.α / 2 * Qt t j i) (N : ℕ) (hN0 : 0 < N) (c : ℕ → ι) (t : ℕ) (hs : ¬ Stay R c t) :
    cost R c N t ≤ log (2 * N / R.α) ∧
      ((relabelAll (R.ops t) (c t) ∈ R.s (t + 1)) → 0 < Qt (t + 1) (relabelAll (R.ops t) (c t)) (c (t + 1)) →
        cost R c N t ≤ -log (R.α / 2 * Qt (t + 1) (relabelAll (R.ops t) (c t)) (c (t + 1))) +
          birthsAll (R.ops t) * log (1 / (1 - R.α))) := by
  constructor
  · have := cost_switch_le hR N hN0 c t hs
    rw [hρ] at this
    have e : (N : ℝ) / (R.α / 2) = 2 * N / R.α := by rw [div_div_eq_mul_div]; ring
    rwa [e] at this
  · intro hj hQ
    have hK' : 0 < R.K (t + 1) (relabelAll (R.ops t) (c t)) (c (t + 1)) := by
      rw [hK]; exact mul_pos (by linarith [hR.α_pos]) hQ
    have := cost_kernel_le hR N c t hs hj hK'
    rwa [hK] at this

/-- **MOM vs a tail expert** (window or BOCPD: `Q̃(e,e) = 1`, never folded): each step costs only
`log(1/(1-α/2))`, half of SOFT's, plus `log(1/(1-α))` per birth. This is why MOM pays half SOFT's price for its era
memory when nothing recurs (NOREC). -/
theorem mom_regret_tail {R : Run ι} (hR : R.Valid) (Qt : ℕ → ι → ι → ℝ) (hρ : R.ρ = R.α / 2)
    (hK : ∀ t j i, R.K t j i = R.α / 2 * Qt t j i) (N : ℕ) (hN : ∀ t, (R.s t).card ≤ N) (e : ι)
    (he : ∀ t, e ∈ R.s t) (hfix : ∀ t, relabelAll (R.ops t) e = e) (hQe : ∀ t, Qt t e e = 1) (T : ℕ) :
    ∑ t in range (T + 1), -log (pmix R t) ≤
      ∑ t in range (T + 1), -log (R.P t e) + log ((R.s 0).card / (1 - R.α / 2)) +
        T * log (1 / (1 - R.α / 2)) + (∑ t in range T, (birthsAll (R.ops t) : ℝ)) * log (1 / (1 - R.α)) := by
  have h := bank_regret_fixed hR N hN e he hfix (R.α / 2) (by linarith [hR.α_pos])
    (fun t => by rw [hK, hQe, mul_one]) T
  have e1 : 1 - R.α + R.ρ = 1 - R.α / 2 := by rw [hρ]; ring
  have e2 : 1 - R.α + R.α / 2 = 1 - R.α / 2 := by ring
  rw [e1, e2] at h
  exact h

/-! ### The coded kernel `Q` -/

/-- `Q(j,i) = (S(j,i) + 1) / Σ_{i' ∈ s, i' ≠ j} (S(j,i') + 1)` off the diagonal, `0` on it (`S` = switch counts). -/
noncomputable def momQ (S : ι → ι → ℕ) (s : Finset ι) (j i : ι) : ℝ :=
  if i = j then 0 else ((S j i : ℝ) + 1) / ∑ i' in s.erase j, ((S j i' : ℝ) + 1)

theorem momQ_nonneg (S : ι → ι → ℕ) (s : Finset ι) (j i : ι) : 0 ≤ momQ S s j i := by
  unfold momQ
  split_ifs
  · exact le_rfl
  · exact div_nonneg (by positivity) (sum_nonneg fun _ _ => by positivity)

/-- Every off-diagonal switch inside the pool has positive kernel mass. -/
theorem momQ_pos (S : ι → ι → ℕ) (s : Finset ι) (j i : ι) (hi : i ∈ s) (hij : i ≠ j) : 0 < momQ S s j i := by
  unfold momQ
  rw [if_neg hij]
  apply div_pos (by positivity)
  exact sum_pos (fun _ _ => by positivity) ⟨i, mem_erase.mpr ⟨hij, hi⟩⟩

theorem momQ_row_sum (S : ι → ι → ℕ) (s : Finset ι) (j : ι) (hj : j ∈ s) (h2 : ∃ i ∈ s, i ≠ j) :
    ∑ i in s, momQ S s j i = 1 := by
  obtain ⟨i0, hi0, hne⟩ := h2
  have hpos : 0 < ∑ i' in s.erase j, ((S j i' : ℝ) + 1) :=
    sum_pos (fun _ _ => by positivity) ⟨i0, mem_erase.mpr ⟨hne, hi0⟩⟩
  rw [← add_sum_erase s _ hj]
  simp only [momQ, eq_self_iff_true, if_true, zero_add]
  rw [sum_congr rfl (fun i hi => if_neg (ne_of_mem_erase hi)), ← sum_div, div_self hpos.ne']

end L41
end FCPA

import FCPA.L39c_MergeConsistency

/-!
L41 (learned kernel). MOM's `Q` is the add-one (Laplace) estimator of era-to-era switches: row `j` predicts the next era
`i ≠ j` as `(S_{ji} + 1)/(S_j + k)`, `k` = number of other eras (`L41.momQ`). This is L39's Dirichlet predictive with
`a = 1` on one context, so its prequential product along the switches out of `j` is `ml 1` (`L39.preq_eq`).

* `rise_one`, `rise_le`: `rise 1 m = m!` and `rise k n ≤ n! (n+1)^{k-1}` (Bernoulli).
* `multinom_le`, `multinom_types`: the multinomial bound `n! Π (c_y/n)^{c_y} ≤ Π c_y!`.
* `laplace_regret`: **for every switch sequence**, the learned kernel's total log loss is at most that of the best fixed
  kernel `Q` in hindsight plus `(k-1) log(n+1)`, `n` = switches out of the row.
* `learned_kernel_regret`: the same, stated for the prequential product the code runs.

Read with L41.mom_switch_cost: along the comparator's switches, learning `Q` costs at most `(k_j - 1) log(S_j + 1)` nats
per era row over knowing the best fixed `Q` in hindsight, i.e. `O(k log S / S)` per switch. This is the convergence rate
in the form the regret needs, pathwise, with no assumption on how switches are generated. It applies when `Q` is
counted on the comparator's own switches with a fixed era set; parameter convergence `Q̂ → Q` in probability needs
concentration (Hoeffding) and stays CITED.
-/
open BigOperators Real Finset

namespace FCPA
namespace L41b

open L39

theorem rise_one : ∀ m : ℕ, rise 1 m = (m.factorial : ℝ)
  | 0 => by simp
  | m + 1 => by
    rw [show rise 1 (m + 1) = rise 1 m * (1 + m) from rfl, rise_one m, Nat.factorial_succ]
    push_cast
    ring

/-- `rise (m+1) n ≤ n! (n+1)^m`. -/
theorem rise_le (m : ℕ) : ∀ n : ℕ, rise ((m : ℝ) + 1) n ≤ (n.factorial : ℝ) * ((n : ℝ) + 1) ^ m
  | 0 => by simp
  | n + 1 => by
    have ih := rise_le m n
    have hn : (0 : ℝ) < (n : ℝ) + 1 := by positivity
    -- Bernoulli: `1 + m/(n+1) ≤ (1 + 1/(n+1))^m`
    have hnn : 0 ≤ 1 / ((n : ℝ) + 1) := by positivity
    have hb := one_add_mul_le_pow (by linarith : (-2 : ℝ) ≤ 1 / ((n : ℝ) + 1)) m
    have e1 : ((n : ℝ) + 1 + 1) ^ m = ((n : ℝ) + 1) ^ m * (1 + 1 / ((n : ℝ) + 1)) ^ m := by
      rw [← mul_pow]; congr 1; field_simp; ring
    have key : ((n : ℝ) + 1) ^ m * ((m : ℝ) + 1 + n) ≤ ((n : ℝ) + 1) * ((n : ℝ) + 1 + 1) ^ m := by
      rw [e1]
      have hp : 0 ≤ ((n : ℝ) + 1) ^ m := by positivity
      have h2 : ((m : ℝ) + 1 + n) = ((n : ℝ) + 1) * (1 + (m : ℝ) * (1 / ((n : ℝ) + 1))) := by field_simp; ring
      rw [h2]
      have := mul_le_mul_of_nonneg_left hb (mul_nonneg hp hn.le)
      nlinarith [this]
    rw [show rise ((m : ℝ) + 1) (n + 1) = rise ((m : ℝ) + 1) n * ((m : ℝ) + 1 + n) from rfl,
      Nat.factorial_succ]
    push_cast
    have hf : (0 : ℝ) ≤ (n.factorial : ℝ) := Nat.cast_nonneg _
    calc rise ((m : ℝ) + 1) n * ((m : ℝ) + 1 + n)
        ≤ (n.factorial : ℝ) * ((n : ℝ) + 1) ^ m * ((m : ℝ) + 1 + n) :=
          mul_le_mul_of_nonneg_right ih (by positivity)
      _ = (n.factorial : ℝ) * (((n : ℝ) + 1) ^ m * ((m : ℝ) + 1 + n)) := by ring
      _ ≤ (n.factorial : ℝ) * (((n : ℝ) + 1) * ((n : ℝ) + 1 + 1) ^ m) := mul_le_mul_of_nonneg_left key hf
      _ = ((n : ℝ) + 1) * (n.factorial : ℝ) * ((n : ℝ) + 1 + 1) ^ m := by ring

variable {Y : Type*} [Fintype Y] [DecidableEq Y]

/-- Multinomial bound: `(Σ c)! Π p^c ≤ Π c! · (Σ p)^(Σ c)` for `p ≥ 0`. -/
theorem multinom_le (c : Y → ℕ) (p : Y → ℝ) (hp : ∀ y, 0 ≤ p y) (s : Finset Y) :
    ((∑ y in s, c y).factorial : ℝ) * ∏ y in s, p y ^ c y ≤
      (∏ y in s, ((c y).factorial : ℝ)) * (∑ y in s, p y) ^ (∑ y in s, c y) := by
  induction s using Finset.induction_on with
  | empty => simp
  | insert ha ih =>
    rename_i a s
    rw [sum_insert ha, sum_insert ha, prod_insert ha, prod_insert ha]
    set m := ∑ y in s, c y
    set S := ∑ y in s, p y
    have hS : 0 ≤ S := sum_nonneg fun y _ => hp y
    have hpa : 0 ≤ p a ^ c a := pow_nonneg (hp a) _
    have hF : (0 : ℝ) ≤ ∏ y in s, ((c y).factorial : ℝ) := prod_nonneg fun y _ => Nat.cast_nonneg _
    have hfa : (0 : ℝ) ≤ ((c a).factorial : ℝ) := Nat.cast_nonneg _
    have hch : (0 : ℝ) ≤ ((c a + m).choose (c a) : ℝ) := Nat.cast_nonneg _
    have hbin : ((c a + m).choose (c a) : ℝ) * p a ^ c a * S ^ m ≤ (p a + S) ^ (c a + m) := by
      rw [add_pow]
      have h := single_le_sum (f := fun k => p a ^ k * S ^ (c a + m - k) * ((c a + m).choose k : ℝ))
        (fun k _ => mul_nonneg (mul_nonneg (pow_nonneg (hp a) _) (pow_nonneg hS _)) (Nat.cast_nonneg _))
        (mem_range.mpr (by omega : c a < c a + m + 1))
      simp only [Nat.add_sub_cancel_left] at h
      linarith
    have hfact : ((c a + m).factorial : ℝ) = ((c a + m).choose (c a) : ℝ) * (c a).factorial * m.factorial := by
      have h := Nat.choose_mul_factorial_mul_factorial (Nat.le_add_right (c a) m)
      rw [Nat.add_sub_cancel_left] at h
      exact_mod_cast h.symm
    rw [hfact]
    calc ((c a + m).choose (c a) : ℝ) * (c a).factorial * m.factorial * (p a ^ c a * ∏ y in s, p y ^ c y)
        = ((c a + m).choose (c a) : ℝ) * (c a).factorial * p a ^ c a * (m.factorial * ∏ y in s, p y ^ c y) := by
          ring
      _ ≤ ((c a + m).choose (c a) : ℝ) * (c a).factorial * p a ^ c a *
            ((∏ y in s, ((c y).factorial : ℝ)) * S ^ m) :=
          mul_le_mul_of_nonneg_left ih (mul_nonneg (mul_nonneg hch hfa) hpa)
      _ = (c a).factorial * (∏ y in s, ((c y).factorial : ℝ)) *
            (((c a + m).choose (c a) : ℝ) * p a ^ c a * S ^ m) := by ring
      _ ≤ (c a).factorial * (∏ y in s, ((c y).factorial : ℝ)) * (p a + S) ^ (c a + m) :=
          mul_le_mul_of_nonneg_left hbin (mul_nonneg hfa hF)

/-- Method of types: `n! Π (c/n)^c ≤ Π c!`. -/
theorem multinom_types (c : Y → ℕ) (hn : 0 < rowSum c) :
    ((rowSum c).factorial : ℝ) * ∏ y, ((c y : ℝ) / rowSum c) ^ c y ≤ ∏ y, ((c y).factorial : ℝ) := by
  have hn' : (0 : ℝ) < rowSum c := by exact_mod_cast hn
  have h := multinom_le c (fun y => (c y : ℝ) / rowSum c) (fun y => by positivity) univ
  have hsum : ∑ y, (c y : ℝ) / rowSum c = 1 := by
    rw [← sum_div, ← rowSum_cast, div_self hn'.ne']
  rw [hsum, one_pow, mul_one] at h
  exact h

theorem pow_ratio_pos (c : Y → ℕ) (y : Y) : 0 < ((c y : ℝ) / rowSum c) ^ c y := by
  rcases Nat.eq_zero_or_pos (c y) with h | h
  · rw [h, pow_zero]; exact one_pos
  · have hc : (0 : ℝ) < c y := by exact_mod_cast h
    have hn : (0 : ℝ) < rowSum c := by
      rw [rowSum_cast]
      exact lt_of_lt_of_le hc (single_le_sum (f := fun z => (c z : ℝ)) (fun z _ => by positivity) (mem_univ y))
    positivity

/-- **Laplace regret.** The add-one estimator's log loss (`-logMLrow 1 c`) is at most the log loss of any fixed
sub-probability `Q` (positive where `c` is) plus `(k-1) log(n+1)`. -/
theorem laplace_regret [Nonempty Y] (c : Y → ℕ) (Q : Y → ℝ) (hQ0 : ∀ y, 0 ≤ Q y)
    (hQpos : ∀ y, 0 < c y → 0 < Q y) (hQ1 : ∑ y, Q y ≤ 1) :
    -logMLrow 1 c ≤ -∑ y, (c y : ℝ) * log (Q y) + ((Fintype.card Y : ℝ) - 1) * log ((rowSum c : ℝ) + 1) := by
  obtain ⟨m, hm⟩ : ∃ m, Fintype.card Y = m + 1 := ⟨Fintype.card Y - 1, by have := Fintype.card_pos (α := Y); omega⟩
  have hV : ((Fintype.card Y : ℝ) * 1) = (m : ℝ) + 1 := by rw [hm]; push_cast; ring
  have hV' : (Fintype.card Y : ℝ) - 1 = m := by rw [hm]; push_cast; ring
  have hg := gibbs_row c Q hQ0 hQpos hQ1
  -- `log rise V n ≤ log n! + m log(n+1)`
  have hrise : log (rise ((Fintype.card Y : ℝ) * 1) (rowSum c)) ≤
      log ((rowSum c).factorial : ℝ) + m * log ((rowSum c : ℝ) + 1) := by
    rw [hV]
    have hfpos : (0 : ℝ) < ((rowSum c).factorial : ℝ) := by exact_mod_cast Nat.factorial_pos _
    have hp1 : (0 : ℝ) < (rowSum c : ℝ) + 1 := by positivity
    have := log_le_log (rise_pos (by positivity) _) (rise_le m (rowSum c))
    rwa [log_mul hfpos.ne' (pow_pos hp1 m).ne', log_pow] at this
  have hlog : logMLrow 1 c = ∑ y, log ((c y).factorial : ℝ) - log (rise ((Fintype.card Y : ℝ) * 1) (rowSum c)) := by
    unfold logMLrow
    simp only [rise_one]
  rcases Nat.eq_zero_or_pos (rowSum c) with h0 | hpos
  · -- no switches: every count is zero
    have hc0 : ∀ y, c y = 0 := by
      intro y
      have : c y ≤ rowSum c := single_le_sum (f := c) (fun _ _ => Nat.zero_le _) (mem_univ y)
      omega
    have hrs : (rowSum c : ℝ) = 0 := by exact_mod_cast h0
    rw [hlog, hV']
    simp [hc0, h0, hrs]
  · have hT := multinom_types c hpos
    have hprod : 0 < ∏ y, ((c y : ℝ) / rowSum c) ^ c y := prod_pos fun y _ => pow_ratio_pos c y
    have hfpos : (0 : ℝ) < ((rowSum c).factorial : ℝ) := by exact_mod_cast Nat.factorial_pos _
    have hT' := log_le_log (mul_pos hfpos hprod) hT
    rw [log_mul hfpos.ne' hprod.ne', log_prod _ _ (fun y _ => (pow_ratio_pos c y).ne'),
      log_prod _ _ (fun y _ => by exact_mod_cast (Nat.factorial_pos _).ne')] at hT'
    simp only [log_pow] at hT'
    have hml := maxll_eq c
    rw [hlog, hV']
    linarith

/-- **Learned-kernel regret, as the code runs it.** For one era row, `preq 1 0 zs` is the product of the add-one
predictions along the switch sequence `zs` (context ignored: `R = Unit`). Its log loss is within `(k-1) log(n+1)` of
the best fixed kernel in hindsight. -/
theorem learned_kernel_regret [Nonempty Y] (zs : List (Unit × Y)) (Q : Y → ℝ) (hQ0 : ∀ y, 0 ≤ Q y)
    (hQpos : ∀ y, 0 < counts zs () y → 0 < Q y) (hQ1 : ∑ y, Q y ≤ 1) :
    -log (preq 1 0 zs) ≤ -∑ y, (counts zs () y : ℝ) * log (Q y) +
      ((Fintype.card Y : ℝ) - 1) * log ((rowSum (counts zs ()) : ℝ) + 1) := by
  have h1 : (0 : ℝ) < 1 := one_pos
  rw [preq_eq h1, ← show ml 1 (counts zs) = mlPost 1 0 (counts zs) from rfl, log_ml_eq h1]
  simp only [Finset.univ_unique, Finset.sum_singleton]
  exact laplace_regret _ Q hQ0 hQpos hQ1

end L41b
end FCPA

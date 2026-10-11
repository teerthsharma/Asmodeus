import FCPA.L07_ACICoverage

/-!
L24′ No runaway growth (spec §3.10 + Amendment A, Chase caveat). Births fire only on `C = ∅`, and in a stretch with no new regime
`C = ∅ ⇒ err = 1`. Then (corollary of L7) births in `T` steps ≤ `a·T + (max(α₀,1-α₀)+γ)/γ`.
CAVEAT (Chase, L24′): this is linear in `T`; it bounds spurious births, it does not make them O(1).
`birth_starvation`: a birth needing `n` points while `C = ∅` persists (`n-1` further err=1 steps) keeps `α ≥ 0` iff
`γ ≤ α_start / ((n-1)(1-a))` (spec: ≈ 0.0022 for n = 50, a = 0.1, α_start = 0.1).
-/
open BigOperators

namespace FCPA

theorem births_le (γ a : ℝ) (hγ : 0 < γ) (ha : 0 ≤ a ∧ a ≤ 1) (α err : ℕ → ℝ)
    (hrec : ∀ t, α (t + 1) = α t + γ * (a - err t)) (he : ∀ t, err t = 0 ∨ err t = 1)
    (hlo : ∀ t, α t < 0 → err t = 0) (hhi : ∀ t, 1 < α t → err t = 1)
    (h0 : -γ ≤ α 0 ∧ α 0 ≤ 1 + γ) (birth : ℕ → Prop) [DecidablePred birth]
    (hb : ∀ t, birth t → err t = 1) (T : ℕ) (hT : 0 < T) :
    (((Finset.range T).filter birth).card : ℝ) ≤ a * T + (max (α 0) (1 - α 0) + γ) / γ := by
  have hac := aci γ a hγ ha α err hrec he hlo hhi h0 T hT
  have hTpos : (0 : ℝ) < T := by exact_mod_cast hT
  have h1 := (abs_le.mp hac).2
  have h2 : (∑ t in Finset.range T, err t) ≤ a * T + (max (α 0) (1 - α 0) + γ) / γ := by
    have h3 : (∑ t in Finset.range T, err t) / T ≤ (max (α 0) (1 - α 0) + γ) / (γ * T) + a := by
      linarith
    rw [div_le_iff hTpos] at h3
    have h4 : ((max (α 0) (1 - α 0) + γ) / (γ * T) + a) * T
        = (max (α 0) (1 - α 0) + γ) / γ + a * T := by
      field_simp; ring
    linarith
  have hcard : (((Finset.range T).filter birth).card : ℝ)
      ≤ ∑ t in (Finset.range T).filter birth, err t := by
    rw [Finset.card_eq_sum_ones]
    push_cast
    apply Finset.sum_le_sum
    intro t ht
    rw [hb t (Finset.mem_filter.mp ht).2]
  have hsub : ∑ t in (Finset.range T).filter birth, err t ≤ ∑ t in Finset.range T, err t :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) (fun t _ _ => by
      rcases he t with h | h <;> rw [h] <;> norm_num)
  linarith

theorem birth_starvation (γ a α₀ : ℝ) (hγ : 0 < γ) (ha : a < 1) (n : ℕ) (hn : 2 ≤ n)
    (α err : ℕ → ℝ) (hrec : ∀ t, α (t + 1) = α t + γ * (a - err t))
    (hs : α 0 = α₀) (herr : ∀ t, t < n - 1 → err t = 1) :
    0 ≤ α (n - 1) ↔ γ ≤ α₀ / (((n : ℝ) - 1) * (1 - a)) := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  have hm : 1 ≤ m := by omega
  simp only [Nat.add_sub_cancel] at herr ⊢
  have hform : ∀ k, k ≤ m → α k = α₀ - k * γ * (1 - a) := by
    intro k
    induction k with
    | zero => intro _; simp [hs]
    | succ k ih =>
      intro hk
      rw [hrec k, ih (by omega), herr k (by omega)]
      push_cast
      ring
  rw [hform m le_rfl]
  have hmpos : (0 : ℝ) < m := by exact_mod_cast hm
  have hpos : 0 < ((↑(m + 1) : ℝ) - 1) * (1 - a) := by
    push_cast
    have : (m : ℝ) + 1 - 1 = m := by ring
    rw [this]
    exact mul_pos hmpos (by linarith)
  rw [le_div_iff hpos]
  push_cast
  constructor <;> intro h <;> nlinarith [h]

end FCPA

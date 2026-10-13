import FCPA.L38_GrowingBankRegret

/-!
L42: the filter as shipped by the engineered era LM (batched, log-space, HARD routing) is the L38/L41 filter.

* `lse_shift`, `lse_shift_max`: log-sum-exp is shift-invariant, and shifting by the max keeps every term in `(0,1]` and
  the sum in `[1, n]` (no overflow, no underflow to zero). The stable computation is exact.
* `lse_log`: the log-space sum of positive terms is the log of their sum. A zero term must be dropped (`-∞` mask in
  torch); writing `log 0` as a finite number is the one way a log-space move can differ from L38's.
* `logBayes_exact`, `logLoss_exact`: the log-space Bayes step `lu = lv + lP - lse(lv + lP)` and the log-space
  loss `-lse(lv + lP)` are exactly L38's `bayes` and `-log mix`. Batching runs this per row, so it changes nothing.
* `hard_cost`: answering with the top expert `e` alone when its weight is `≥ θ` changes the log loss by at least
  `log θ` (it can gain at most `-log θ`) and at most `log(θ + (1-θ)/ε)`, where `ε` is the floor on expert
  probabilities (Dirichlet smoothing gives one).
* `hard_regret`: over a run, HARD's cumulative loss is the mixture's plus at most `log(θ + (1-θ)/ε)` per HARD step,
  so L38's regret bound carries over with that additive term. With no floor (`ε → 0`) the extra cost is unbounded:
  the floor is load-bearing.
-/
open BigOperators Real Finset

namespace FCPA
namespace L42

open L38

variable {ι : Type*} [DecidableEq ι]

/-- log-sum-exp over a finite set. -/
noncomputable def lse (s : Finset ι) (x : ι → ℝ) : ℝ := log (∑ i in s, exp (x i))

theorem lse_shift (s : Finset ι) (hs : s.Nonempty) (x : ι → ℝ) (m : ℝ) :
    lse s x = m + log (∑ i in s, exp (x i - m)) := by
  unfold lse
  have h : ∑ i in s, exp (x i) = exp m * ∑ i in s, exp (x i - m) := by
    rw [mul_sum]
    apply sum_congr rfl
    intro i _
    rw [← exp_add]
    congr 1
    ring
  rw [h, log_mul (exp_pos m).ne' (sum_pos (fun i _ => exp_pos _) hs).ne', log_exp]

/-- Shifting by the max: every term is in `(0,1]` and the sum is in `[1, |s|]`. -/
theorem lse_shift_max (s : Finset ι) (hs : s.Nonempty) (x : ι → ℝ) :
    (∀ i ∈ s, 0 < exp (x i - s.sup' hs x) ∧ exp (x i - s.sup' hs x) ≤ 1) ∧
      1 ≤ ∑ i in s, exp (x i - s.sup' hs x) ∧ ∑ i in s, exp (x i - s.sup' hs x) ≤ s.card := by
  have hle : ∀ i ∈ s, exp (x i - s.sup' hs x) ≤ 1 := fun i hi => by
    rw [exp_le_one_iff]
    linarith [le_sup' x hi]
  refine ⟨fun i hi => ⟨exp_pos _, hle i hi⟩, ?_, ?_⟩
  · obtain ⟨k, hk, hkx⟩ := exists_mem_eq_sup' hs x
    have : exp (x k - s.sup' hs x) = 1 := by rw [hkx, sub_self, exp_zero]
    rw [← this]
    exact single_le_sum (f := fun i => exp (x i - s.sup' hs x)) (fun i _ => (exp_pos _).le) hk
  · calc ∑ i in s, exp (x i - s.sup' hs x) ≤ ∑ _i in s, (1 : ℝ) := sum_le_sum hle
      _ = s.card := by rw [sum_const, nsmul_eq_mul, mul_one]

/-- The log-space sum of positive terms is the log of their sum. -/
theorem lse_log (s : Finset ι) (a : ι → ℝ) (ha : ∀ i ∈ s, 0 < a i) :
    lse s (fun i => log (a i)) = log (∑ i in s, a i) := by
  unfold lse
  rw [sum_congr rfl (fun i hi => exp_log (ha i hi))]

/-- The log-space Bayes step as shipped. -/
noncomputable def logBayes (s : Finset ι) (lP lv : ι → ℝ) : ι → ℝ :=
  fun i => lv i + lP i - lse s (fun j => lv j + lP j)

theorem lse_add_logs (s : Finset ι) (P v : ι → ℝ) (hP : ∀ i ∈ s, 0 < P i)
    (hv : ∀ i ∈ s, 0 < v i) : lse s (fun j => log (v j) + log (P j)) = log (mix s P v) := by
  unfold lse mix
  rw [sum_congr rfl (fun j hj => by rw [exp_add, exp_log (hv j hj), exp_log (hP j hj)])]

/-- **Log-space Bayes is exact.** -/
theorem logBayes_exact (s : Finset ι) (hs : s.Nonempty) (P v : ι → ℝ) (hP : ∀ i ∈ s, 0 < P i)
    (hv : ∀ i ∈ s, 0 < v i) (i : ι) (hi : i ∈ s) :
    exp (logBayes s (fun j => log (P j)) (fun j => log (v j)) i) = bayes s P v i := by
  have hm : 0 < mix s P v := sum_pos (fun j hj => mul_pos (hv j hj) (hP j hj)) hs
  unfold logBayes
  rw [lse_add_logs s P v hP hv, exp_sub, exp_add, exp_log (hv i hi), exp_log (hP i hi), exp_log hm]
  rfl

/-- **Log-space loss is exact:** `-lse(lv + lP) = -log mix`. -/
theorem logLoss_exact (s : Finset ι) (P v : ι → ℝ) (hP : ∀ i ∈ s, 0 < P i)
    (hv : ∀ i ∈ s, 0 < v i) : -lse s (fun j => log (v j) + log (P j)) = -log (mix s P v) := by
  rw [lse_add_logs s P v hP hv]

/-- **HARD routing, one step.** Weights `v ≥ 0` with total `≤ 1`, expert probabilities in `[0,1]`, top expert `e`
with weight `≥ θ` and probability `≥ ε`: `log θ ≤ log mix - log P_e ≤ log(θ + (1-θ)/ε)`. -/
theorem hard_cost (s : Finset ι) (v P : ι → ℝ) (e : ι) (he : e ∈ s) (θ ε : ℝ) (hθ : 0 < θ) (hε : 0 < ε)
    (hε1 : ε ≤ 1) (hv0 : ∀ i ∈ s, 0 ≤ v i) (hv1 : ∑ i in s, v i ≤ 1) (hP0 : ∀ i ∈ s, 0 ≤ P i)
    (hP1 : ∀ i ∈ s, P i ≤ 1) (hPe : ε ≤ P e) (hve : θ ≤ v e) :
    log θ ≤ log (mix s P v) - log (P e) ∧ log (mix s P v) - log (P e) ≤ log (θ + (1 - θ) / ε) := by
  have hPe0 : 0 < P e := lt_of_lt_of_le hε hPe
  have hve0 : 0 < v e := lt_of_lt_of_le hθ hve
  have hsplit : mix s P v = v e * P e + ∑ i in s.erase e, v i * P i := by
    unfold mix; rw [← add_sum_erase s _ he]
  have hrest0 : 0 ≤ ∑ i in s.erase e, v i * P i :=
    sum_nonneg fun i hi => mul_nonneg (hv0 i (mem_of_mem_erase hi)) (hP0 i (mem_of_mem_erase hi))
  have hrest1 : ∑ i in s.erase e, v i * P i ≤ ∑ i in s.erase e, v i :=
    sum_le_sum fun i hi => mul_le_of_le_one_right (hv0 i (mem_of_mem_erase hi)) (hP1 i (mem_of_mem_erase hi))
  have hvsplit : ∑ i in s, v i = v e + ∑ i in s.erase e, v i := (add_sum_erase s _ he).symm
  have hm : 0 < mix s P v := by rw [hsplit]; have := mul_pos hve0 hPe0; linarith
  rw [← log_div hm.ne' hPe0.ne']
  constructor
  · apply log_le_log hθ
    rw [le_div_iff hPe0, hsplit]
    have := mul_le_mul_of_nonneg_right hve hPe0.le
    linarith
  · apply log_le_log (div_pos hm hPe0)
    -- `mix ≤ v_e P_e + (1 - v_e)`
    have hmix : mix s P v ≤ v e * P e + (1 - v e) := by rw [hsplit]; linarith
    have h1ve : 0 ≤ 1 - v e := by
      have : 0 ≤ ∑ i in s.erase e, v i := sum_nonneg fun i hi => hv0 i (mem_of_mem_erase hi)
      linarith
    have hq : mix s P v / P e ≤ v e + (1 - v e) / P e := by
      have hc : (1 - v e) / P e * P e = 1 - v e := by field_simp
      rw [div_le_iff hPe0, add_mul, hc]
      linarith
    have hq2 : (1 - v e) / P e ≤ (1 - v e) / ε := div_le_div_of_nonneg_left h1ve hε hPe
    -- `v + (1-v)/ε` is nonincreasing in `v` because `1/ε ≥ 1`
    have hinv : 1 ≤ 1 / ε := by rw [le_div_iff hε]; linarith
    have hmono : v e + (1 - v e) / ε ≤ θ + (1 - θ) / ε := by
      have e1 : v e + (1 - v e) / ε = v e * (1 - 1 / ε) + 1 / ε := by ring
      have e2 : θ + (1 - θ) / ε = θ * (1 - 1 / ε) + 1 / ε := by ring
      rw [e1, e2]
      have : (v e - θ) * (1 - 1 / ε) ≤ 0 :=
        mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith)
      linarith
    linarith

/-- **HARD routing over a run.** If at every HARD step the per-step conditions of `hard_cost` hold, HARD's
cumulative log loss is the mixture's plus at most `log(θ + (1-θ)/ε)` per HARD step. -/
theorem hard_regret (p q : ℕ → ℝ) (hard : ℕ → Bool) (δ : ℝ) (T : ℕ)
    (hstep : ∀ t, hard t = true → log (p t) - log (q t) ≤ δ) :
    ∑ t in range T, -log (if hard t then q t else p t) ≤
      ∑ t in range T, -log (p t) + ((range T).filter (fun t => hard t = true)).card * δ := by
  have h : ∀ t ∈ range T, -log (if hard t then q t else p t) ≤
      -log (p t) + (if hard t = true then δ else 0) := by
    intro t _
    by_cases ht : hard t = true
    · simp only [ht, if_true]; linarith [hstep t ht]
    · simp only [ht, if_false, add_zero]; exact le_refl _
  calc ∑ t in range T, -log (if hard t then q t else p t)
      ≤ ∑ t in range T, (-log (p t) + (if hard t = true then δ else 0)) := sum_le_sum h
    _ = _ := by rw [sum_add_distrib, sum_ite, sum_const_zero, add_zero, sum_const, nsmul_eq_mul]

end L42
end FCPA

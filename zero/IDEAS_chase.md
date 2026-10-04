# IDEAS_chase: what breaks, and the guard plus RED test for each

Stance: conservative. Each item gives the failure mode, the source, the guard Asmodeous should adopt, and a RED test (one that must fail on the unguarded code).

**Source checking.**
- Three sonnet nurses checked the sources on 2026-10-04 by fetching arXiv abstract pages: 8/8 routing ids and 9/9 Lie/DEQ ids resolved to their stated titles.
- Theorem statements marked [mem] were not text-extracted from the papers.
- Items 1–3 are measured here: house-events.jsonl and PREREG_DRAFT_atlas.md hold the numbers.

---

## Already hit in this build (measured)

**1. N-free SIGReg cannot see expansion.**
- **Failure mode:** a model can diverge to large scale and SIGReg does almost nothing to stop it.
  - On N(0, s²I), the scale gradient d sigreg/ds is −0.658 at s = 0.3 but +0.0076 at s = 15 and ≈0 at s = 30.
  - The atlas diverged on 2 of 3 seeds (g std 15.2 and 11.3).
- **Source:** measured. The SIGReg itself is LeJEPA, arXiv 2511.08544.
- **Guard (done):** a second-moment anchor `0.1·log(E[(x·a)²] + 1e-6)²` on the same slices, in `sigreg.py`. The scale force at s = 15 rose from 0.0076 to +0.152.
- **RED test:** `tests/test_sigreg.py::test_restoring_force_on_expansion`.

**2. A falling loss with shrinkage collapse.**
- **Failure mode:** at λ = 0.05 (N-free), every arm's total loss falls while per-dimension std shrinks to 0.05–0.25.
- **Source:** measured. The general phenomenon is in LeJEPA §3.
- **Guard (done):** a scale contract before any comparison: per-dim std ∈ [0.3, 3], effective rank ≥ n, 3 seeds.
- **RED test:** `test_scale_contract_every_arm_same_lambda[*]`, which was red at 0.05 for single, chart-AE, lazy chart-AE and PEER.

**3. One-seed λ selection.**
- **Failure mode:** λ = 0.9 "passed every arm" on seed 0, then the atlas failed 2 of 3 seeds at D_g = 6.
- **Source:** measured.
- **Guard (done):** every λ or hyperparameter pick in a PREREG table needs at least 3 seeds and one grid step of margin.
- **RED test:** a meta-test that parses `sigreg.py`'s LAM comment and asserts the cited grid has at least 3 seeds, so a single-seed pin fails review. Inspector owns it.

## Routing and modularity

**4. Router logit blow-up.**
- **Failure mode:** large router logits saturate the softmax, so the gradient to the losing charts vanishes and the loss spikes, especially in bf16.
- **Source:** Zoph et al. 2022, ST-MoE, arXiv 2202.08906. Router z-loss L_z = mean_tokens (logsumexp_j x_j)², with coefficient c_z = 1e-3.
- **Guard:** add an optional `lam_z` (1e-3) on the atlas `full_logits`. It is per-token and causal-safe. Keep the router in fp32 under autocast.
- **RED test:** initialise keys ×100. After 50 Adam steps, assert max |logit| < 50 with `lam_z` = 1e-3, and > 50 without it.

**5. Expert death under top-k.**
- **Failure mode:** an unselected chart gets zero gradient forever. We measured 2–3 of 8 charts dead at λ_bal = 0.01.
- **Source:** Fedus et al. 2021, Switch, arXiv 2101.03961: α·N·Σ f_i P_i with α = 1e-2.
- **Guard (done):** LAM_BAL = 0.3. The optional L_Div (`lam_div` = 0.1) also kept 8/8 charts alive on 2 seeds.
- **RED test:** `test_grad_reaches_selected_charts_only_and_all_keys`, plus the alive ≥ K−2 leg of the scale contract.

**6. The auxiliary balance loss fights the objective.**
- **Failure mode:** load-balance gradients interfere with the task. DeepSeek measured perplexity 9.56 against 9.50, and MaxVio 0.72 against 0.04, at 1B parameters.
- **Source:** Wang et al. 2024, arXiv 2408.15664. Use a bias b_i only in the top-k choice, updated after each batch by b_i += u·sign(mean_load − load_i) with u = 1e-3. The gate value stays unbiased.
- **Guard:** offer `balance="bias"` as an ablation of λ_bal. It is causal-safe, because it uses only the previous batch.
- **RED test:**
  - with λ_bal = 0 and bias balancing on, after 200 steps max-violation < 0.2 and no loss term carries a balance gradient (assert `keys.grad` is unchanged when the bias update is toggled);
  - with neither guard, max-violation > 0.5.

**7. Balanced assignment leaks the future.**
- **Failure mode:** BASE, Sinkhorn/S-BASE and Expert Choice balance over the whole batch. A token's chart then depends on other tokens in the batch, including future timesteps, which breaks causal parity.
- **Source:** Lewis et al. 2021, arXiv 2103.16716. Clark et al. 2022, arXiv 2202.01169. Zhou et al. 2022, arXiv 2202.09368.
- **Guard:** never use batch-global assignment in the atlas observer. Any owned arm that does must run with batch = 1 at evaluation.
- **RED test:** extend `test_causality` to every router variant: perturbing obs[:, t:] must leave w[:, :t] bit-identical. A Sinkhorn router fails this.

**8. Representation collapse toward router keys.**
- **Failure mode:** router gradients pull hidden states into span{κ_k}, so c_t loses rank (to about K dimensions).
- **Source:** Chi et al. 2022, X-MoE, arXiv 2204.09179. Cosine routing s = (Wh·e)/(‖Wh‖‖e‖)/τ with τ₀ = 0.3.
- **Guard:** report erank(c_t) beside erank(g). Adopt the cosine router if erank(c_t) ≤ min(K, d_key) − 1.
- **RED test:** with d_h = 64 and K = 4, after 500 steps assert erank(c_t) > 8. Run with the dot-product router to see whether it fails before adopting the cosine router.

**9. A sign error in the diversity loss.**
- **Failure mode:** AETHER_qGPU's L = … + γ·Σ D_KL(ρ_i‖ρ_j), when minimised, pulls modules together. That is the opposite of its stated purpose.
- **Source:** `ref/origin/AETHER_qGPU_event_driven.pdf`, training objective.
- **Guard (done):** `chart_diversity` is subtracted, and ε bounds it by about log(1/ε).
- **RED test:** `test_div_rewards_separation`.

## Lie-group, complex and implicit layers (for PHASE-SPHERE and hybrids)

**10. Cayley singularity.**
- **Failure mode:** W = (I+A)⁻¹(I−A) cannot reach eigenvalue −1, and |A| blows up near it (about 447 in the paper's example).
- **Source:** Helfrich et al. 2018, scoRNN, arXiv 1707.09520. Use W = (I+A)⁻¹(I−A)D with a fixed D = diag(±1).
- **Guard:** use `torch.nn.utils.parametrizations.orthogonal` (matrix_exp with trivialisation) [mem], or a scaled Cayley transform with D.
- **RED test:** fit W to Q·diag(−1, 1, …)·Qᵀ in float32. Plain Cayley must end with max|A| > 100. The guarded version must reach ‖W − W₀‖ < 1e-3 with ‖WᵀW − I‖ < 1e-5.

**11. Drift off the manifold.**
- **Failure mode:** accumulating orthogonal or unitary products in float32 drifts. The full-capacity uRNN measured ‖WᵀW − I‖ ≈ 1e-6 after 70 epochs, against 1e-16 for scoRNN.
- **Source:** Arjovsky et al. 2016, arXiv 1511.06464, with figure 4 of scoRNN. Lezcano-Casado and Martínez-Rubio 2019, expRNN, arXiv 1901.08428.
- **Guard:** rebuild W from its parameters every step; never multiply updates in place.
- **RED test:**
  - after 1000 Adam steps, ‖WᵀW − I‖∞ < 1e-5;
  - | ‖Wᵏx‖/‖x‖ − 1 | < 1e-4 at k = 10⁴;
  - a naive in-place multiplicative update fails.

**12. Householder division by ‖u‖².**
- **Failure mode:** H(u) = I − 2uuᵀ/‖u‖² gives NaN as u → 0. With m < n reflections, det = (−1)ᵐ is fixed.
- **Source:** Mhammedi et al. 2017, arXiv 1612.00188.
- **Guard:** add ε to ‖u‖², and use m = n reflections when full O(n) is needed.
- **RED test:** u = 1e-20·randn in float32 must give finite output and gradients. For odd m, det(W) = −1.

**13. Complex-valued pitfalls.**
- **Failure mode:**
  - modReLU z/|z| has a NaN gradient at 0;
  - angle() is singular at 0;
  - real-style initialisation gives the wrong variance;
  - double conjugation flips the sign of Im(grad). PyTorch stores the conjugate Wirtinger gradient [mem].
- **Source:** Trabelsi et al. 2018, Deep Complex Networks, arXiv 1705.09792. Rayleigh modulus with σ = 1/√(n_in + n_out), uniform phase, and complex BN with a 2×2 whitening.
- **Guard:** safe modulus √(|z|² + ε); complex BN; `gradcheck` on complex128 for every complex layer.
- **RED test:**
  - modReLU gradient finite at z = 0 and at |z| = 1e-30;
  - complex128 `gradcheck` passes;
  - Var(Re) = Var(Im) = σ² within 3% at n = 1e5.

**14. A DEQ fixed point near spectral radius 1.**
- **Failure mode:** the backward pass solves (I − J)ᵀg = ∂L/∂z*. As ρ(J) → 1 this is ill-conditioned and the number of function evaluations (NFE) grows during training.
- **Source:** Bai et al. 2019, arXiv 1909.01377. Bai et al. 2021, arXiv 2106.14342: Jacobian regularisation γ‖εᵀJ‖²/d (Hutchinson, one sample), with CIFAR settings γ = 0.5, p = 0.05.
- **Guard:**
  - a Jacobian penalty;
  - a solver that returns a non-converged flag instead of a silent output;
  - optionally the monotone parameterisation W = (1−m)I − AᵀA + B − Bᵀ (Winston and Kolter 2020, arXiv 2006.08591).
- **RED test:**
  - for f = tanh(Wz + x) at ρ(W) = 0.99, the implicit gradient matches finite differences to relative error 1e-4;
  - the residual is < 1e-5;
  - at ρ ≥ 1 the solver raises.

**15. Phantom gradients.**
- **Failure mode:** a cheap gradient approximation can point the wrong way.
- **Source:** Geng et al. 2021, arXiv 2111.05177. A k-step unroll or a damped Neumann series is a descent direction when the Theorem 1 bound holds.
- **Guard:** use the 1–5-step phantom gradient only when estimated ρ(J) < 0.7. Log the cosine to the exact gradient on a probe batch.
- **RED test:** cosine(phantom, exact) > 0.9 at ρ < 0.7. An undamped Neumann series at ρ ≥ 1 diverges past 1e6.

## Topological readout

**16. Betti numbers at a fixed τ are unstable.**
- **Failure mode:** a bar of length about ε near the chosen τ flips β_k.
- **Source:** Cohen-Steiner, Edelsbrunner and Harer 2007, DCG 37:103, bottleneck stability [mem]. The nerve filtration m(σ) = max_x min_{i∈σ} w_i(x) is 1-Lipschitz in the sup-norm of w.
- **Guard:**
  - read persistence over τ, not one τ: build the nerve filtration once (mirror topoml `_reduce_z2`, see PRIOR_CODE_REUSE.md);
  - count only bars longer than 2ε_w, where ε_w is a bootstrap estimate of the sup-norm noise of w_soft.
- **RED test:**
  - perturbing w_soft by sup-norm ε moves the diagram by at most ε (bottleneck);
  - the most-persistent vector of the octahedral S² cover stays (1,0,1);
  - a read at a τ inside a short bar flips. That flip documents why a fixed-τ read is banned.

**17. One outlier row adds a big simplex.**
- **Failure mode:** a near-uniform row of w_soft adds a (k−1)-simplex at τ ≤ 1/k, which can kill H₂.
- **Source:** Chazal, Cohen-Steiner and Mérigot 2011, FoCM 11:733 (distance to a measure).
- **Guard:** support-robust filtration. Replace the max over x with the n_min-th largest value of min_i w_i(x), with n_min = max(3, 0.01N).
- **RED test:** append 5% uniform-Dirichlet rows to octahedral S² w_soft.
  - The `max` filtration must fail (H₂ is lost).
  - The n_min filtration keeps (1,0,1).

**18. The nerve theorem needs a good cover.**
- **Failure mode:** a disconnected chart intersection gives a nerve Betti vector that no τ fixes.
- **Source:** Bauer et al. 2023, Expositiones Math., arXiv 2203.03571 (condition [mem]).
- **Guard:** for each simplex, check that its sample set is connected at scale ρ (single linkage). Report nerve Betti numbers only when no simplex has a disconnected intersection; otherwise flag.
- **RED test:** two arcs covering S¹ with a two-component overlap must be flagged. The octahedral S² cover must pass.

**19. Torsion: Z₂ against Q.**
- **Failure mode:** RP² has Betti (1,1,1) over Z₂ and (1,0,0) over Q, and the Klein bottle (1,2,1) against (1,1,0).
- **Source:** Hatcher §3.A [mem]. Matches PREREG.md's amendment to report Z₂ and Q.
- **Guard:** compute over Z₂ and over F_p (p = 32749, as a stand-in for Q), and flag disagreement.
- **RED test:** a 6-vertex RP² triangulation gives (1,1,1) over Z₂ and (1,0,0) over F_p. Any single-field assertion must fail.

**20. Too few samples for the reach.**
- **Failure mode:** with a sparse sample, persistence can confidently report the wrong homotopy type.
- **Source:** Niyogi, Smale and Weinberger 2008, DCG 39:419 (ε < √(3/5)·reach) [mem constants].
- **Guard:** bootstrap over 80% subsamples. Certify a bar only if its length > max(2ε_w, q₀.₉₅ of the bootstrap bottleneck distance). Report "uncertified" otherwise.
- **RED test:** S² at n = 30 must be uncertified, and at n = 1000 certified as (1,0,1) in at least 99 of 100 bootstraps.

---

## Adoption
Items 1, 2, 3, 5 and 9 are already in code. The next RED tests, by expected damage:
1. item 16, nerve persistence over τ;
2. item 17, support-robust filtration;
3. item 4, router z-loss;
4. item 7, causal parity for every router.

Items 10–15 apply only once PHASE-SPHERE or a hybrid uses Lie-group, complex or implicit layers.

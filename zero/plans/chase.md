# Chase plan: what breaks in QF-LM numerics, and the guards

Stance: conservative. Sources are Wikipedia pages fetched on 2026-10-04. Derived numbers use PLAN.md sizes (N = 2048 complex, |V| = 4096, null records s ≤ 4).

## Findings

1. **Exploding and vanishing gradients come from the spectral radius.**
   - Source: https://en.wikipedia.org/wiki/Vanishing_gradient_problem
   - Fact: when the recurrent matrix has spectral radius γ < 1, the gradient product is bounded by γ^k and decays exponentially. Gradient clipping rescales the gradient into a ball. Residual connections give a Jacobian of ∇f + I.
   - What it means for QF-LM: the butterfly think steps are unitary (γ = 1 exactly), so they neither vanish nor explode. The Lüders write is the only non-unitary step. Its Jacobian carries a factor of 1/‖K_xψ‖ = 1/√p(x), so gradients explode on rare teacher-forced tokens: p = 1e-8 gives a 1e4 gain per step. Guard: clip the global gradient norm, and add a test that bounds the Lüders Jacobian at a minimum p.

2. **fp16 cannot hold the parts of a Born probability at N = 2048 and |V| = 4096.**
   - Source: https://en.wikipedia.org/wiki/Half-precision_floating-point_format
   - Fact: the smallest normal is 2⁻¹⁴ ≈ 6.10e-5, the smallest subnormal is 2⁻²⁴ ≈ 5.96e-8, and the maximum is 65,504.
   - What it means for QF-LM: each term M[n,v]·|(Qψ)_n|² is about (1/4096)(1/2048) ≈ 1.2e-7. That is about 2 subnormal steps above fp16's floor, so squaring in fp16 leaves roughly one significant bit or flushes to zero. Guard: `log_born` upcasts the amplitude before squaring, and the test `test_fp16_fp32_nll_parity` checks it.

3. **bf16 has the range but not the precision.**
   - Source: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
   - Fact: bf16 has 8 exponent bits (the same range as fp32, about 1e-38) and an 8-bit significand (2–3 decimal digits).
   - What it means for QF-LM: bf16 removes the underflow in finding 2. But Σ_v p(v) = 1 then holds only to about 4e-3, which is useless for exact likelihood. Keep ψ, the norm and log-probs in fp32. Low precision is allowed only inside the butterfly matmuls, if anywhere.

4. **fp32 floors bound the length of a null record.**
   - Source: https://en.wikipedia.org/wiki/Single-precision_floating-point_format
   - Fact: the smallest normal is 2⁻¹²⁶ ≈ 1.18e-38, the smallest subnormal is 2⁻¹⁴⁹ ≈ 1.4e-45, and machine epsilon is 2⁻²³ ≈ 1.19e-7.
   - What it means for QF-LM: a rare token after 4 unlikely nulls has a path probability of about e⁻³⁰ per factor × 5 factors = e⁻¹⁵⁰. That underflows fp32 to 0, so the nll becomes +inf. A Lüders divide by ‖K_xψ‖ with p below about 1e-38 gives a NaN state.
   - Guard: work in the log domain everywhere. Also keep M strictly positive (softmax rows), so that p(x) > 0 is structurally guaranteed.

5. **The log-sum-exp trick handles the sum over null records.**
   - Source: https://en.wikipedia.org/wiki/LogSumExp
   - Fact: LSE(x) = x* + log Σ exp(xᵢ − x*), with x* = max xᵢ. Subtracting the maximum prevents overflow and underflow.
   - What it means for QF-LM: compute log p(x) = LSE over s of (Σ_{i<s} log p_null,i + log p_x,s). Never form a product of probabilities. This is `null_record_logp`, with test `test_null_record_no_underflow`.

6. **The summation order decides whether PLAN item 3's ±1e-6 bar holds in fp32.**
   - Source: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
   - Fact: naive summation has an O(εn) worst-case error and O(ε√n) RMS error. Pairwise summation gives O(ε log n). Kahan summation gives O(ε).
   - Measured here, CPU, |V| = 4096:
     - `torch.sum` in fp32 (pairwise or cascade) gave a maximum error of 1.19e-7 over 20 draws;
     - a sequential fp32 accumulation loop gave errors up to 1.19e-6 over 5 draws, which already breaks ±1e-6.
   - What it means for QF-LM: the PLAN item 3 bar "Σp + p_null = 1 ± 1e-6" holds only when the sum uses a pairwise reduction. A hand-written per-outcome accumulation (a custom CUDA kernel or a Python loop) fails the bar for reasons unrelated to the model.
   - Recommendation: test completeness in float64 at ±1e-6, or in fp32 at ±1e-5 (the guard test uses 1e-5). Logged as a finding for Kutner.

7. **Orthogonality drifts unless it is rebuilt from parameters.**
   - Sources: https://en.wikipedia.org/wiki/Orthogonal_matrix and https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
   - Fact: Orthogonal_matrix lists polar decomposition and Newton averaging as ways to re-orthogonalise a matrix that drifted under floating point, and notes that errors are not magnified by an orthogonal matrix. Gram–Schmidt says that classical Gram–Schmidt is numerically unstable and that Householder transformations are more stable.
   - What it means for QF-LM: never store Q as a matrix that is updated in place. Rebuild the butterfly from its angles on every call. Then orthogonality holds to fp32 rounding per call and never accumulates. Guard: `butterfly_apply` plus `orth_error`, with test `test_butterfly_orthogonality` (the control is a stored fp16 matrix product).

8. **Compute ‖K_xψ‖ with hypot-style scaling.**
   - Source: https://en.wikipedia.org/wiki/Givens_rotation
   - Fact: computing r = √(a² + b²) naively can overflow or underflow. Golub & Van Loan's alternative divides by the larger value first (hypot).
   - What it means for QF-LM: squaring tiny Lüders amplitudes underflows before the square root. Compute the norm as max|a|·‖a/max|a|‖, or as exp(½·log_born). Butterfly angles parameterise (c, s) = (cos θ, sin θ) directly, so they never need a division.

9. **Trajectories store N numbers, not N², and the update renormalises.**
   - Source: https://en.wikipedia.org/wiki/Quantum_jump_method
   - Fact: the Monte Carlo wave-function method evolves a pure state (N components, against N² for a density matrix) and averages over trajectories.
   - What it means for QF-LM: QF-LM is a single conditioned trajectory, so the claim "memory O(N)" holds. Renormalisation after every jump is part of the method. Guard: `renorm_psi` after every Lüders step and after each block of think steps. Test: ‖ψ‖ stays within 1e-5 of 1 over 2000 tokens.

10. **Kraus completeness of the instrument.**
    - Source: https://en.wikipedia.org/wiki/Quantum_operation
    - Fact: a channel needs Σ_k B_k†B_k = 1. With ≤ 1 the map is trace non-increasing.
    - What it means for QF-LM: with a row-stochastic M and a unitary Q, Σ_v K_v†K_v = Q†·diag(Σ_v M[n,v])·Q = I exactly. Any drift of the row sums (an M that is not renormalised, or fp16 storage) makes the instrument leak probability, and the null outcome hides it. Assert that the row sums equal 1 in fp32 at every forward pass in debug mode.

11. **Normalisation layers need ε.**
    - Source: https://en.wikipedia.org/wiki/Normalization_(machine_learning)
    - Fact: LayerNorm divides by √(σ² + ε). RMSNorm sets ε = 0.
    - What it means for QF-LM: `renorm_psi` is RMSNorm on C^N with no ε. Add no ε, because it would break ‖ψ‖ = 1 exactly. Instead guarantee ‖K_xψ‖ > 0 structurally (finding 4), and raise an error, not clamp, if it is 0.

## Plan after this session (8 steps; steps 9-10 added in Wander below)

1. Land `qflm/stability.py` and `tests/test_qflm_stability.py` (the nurse's work this session). Mutations must go RED.
2. Get Kutner's primitives to route their nll through `log_born` and `null_record_logp`, and their Lüders step through `renorm_psi`. Add an import-parity test: Kutner's nll must equal the guarded reference in float64.
3. Add a RED test for the Lüders Jacobian: at p(x) = 1e-8, ‖∂ψ'/∂ψ‖ is bounded after gradient clipping. Add the clip to the bake-off harness, with the same clip for every arm so the comparison stays fair.
4. Change the PLAN item 3 completeness bar to float64 ±1e-6 or fp32 ±1e-5 (finding 6). This needs a pre-result PREREG amendment through Wilson.
5. Precision contract for item 5: ψ, the norms and the log-probs stay in fp32; autocast covers only the butterfly matmuls. Test: the nll under autocast is within 1e-3 of fp32 over 1k tokens.
6. Run a 10k-token CUDA-graph soak with Taub (item 4): no NaN, ‖ψ‖ drift < 1e-5, and orth_error < 1e-5 measured on the GPU and not only on the CPU.
7. Add a min-p telemetry counter (the fraction of steps with log p < −60) to the run logs, so underflow pressure is visible before it causes NaN.
8. Run idle-time PLAN item 8 only (Hopf control), and never ahead of items 5 and 6.

## Wander (no agenda)

This crawl started at **Dynamical system**, followed whatever pulled, read 16 articles, and reached a maximum depth of 5 hops.

**Path, in order:**
- **Main chain:** Dynamical system → Outer billiard → Penrose tiling → Wang tile → Diaspora (novel; "Wang's Carpets") → Schild's ladder.
- **Side branches:** Sharkovskii's theorem and Circle map (Arnold tongues), both off Dynamical system.
- **Second chain:** Dynamical system → Kicked rotator → Anderson localization → Many-body localization → Time crystal → Floquet theory. Many-body localization also led to Quantum scar → Quantum ergodicity → Eigenstate thermalization hypothesis.

### The 5 most surprising things
1. **Kicked rotator:** quantum dynamical localization suppresses diffusion that is unbounded classically. Adding noise destroys the localization and restores diffusion.
2. **Many-body localization:** the system never thermalises. Local observables keep memory of the initial state forever, entanglement grows only as log t, and emergent local integrals of motion ("l-bits") appear.
3. **Quantum scars:** eigenstates of a chaotic system have enhanced probability along *unstable* periodic orbits (Heller 1984). Rydberg-atom chains revive instead of thermalising.
4. **Schild's ladder:** parallel transport can be built from geodesics alone. Its error equals the integral of curvature over the enclosed triangle, it is first order, and it is necessarily torsion-free.
5. **Outer billiard:** whether unbounded orbits exist stayed open for about 50 years (Moser and Neumann posed it in the late 1950s). Schwartz answered it in 2007 using the Penrose kite. Separately, the smallest aperiodic Wang set (Jeandel and Rao, 2015) has 11 tiles and 4 colours.

### Connections to Asmodeous (hunch, unmeasured)
- **Hunch: dynamical localization of ψ.** Lüders writes plus butterfly think steps act as periodic kicks. A trained QF-LM may localise ψ in a few directions of CP^{N−1}. The effective N would then be small, and the bound "at most 2N tokens reach p ≥ ½" would bite at N_eff, not at N = 2048. Data-driven writes play the role of noise, which delocalises ψ, so this could go either way. Cheap check: participation ratio of Qψ, and the rank of the ψ trajectory over 1k tokens.
- **Hunch: MBL against ETH is memory against forgetting.** Context memory needs a non-thermalising write/think map. Chaotic, scrambling think unitaries would push the state toward thermal and erase old tokens. Integrable, l-bit-like think steps would keep them. Expect long-range-token loss to get *worse* as more think steps are added unless the angles stay near-integrable.
- **Hunch: scars as the mechanism behind repetition.** LM degeneration into repetition loops may appear in QF-LM as ψ scarring on a periodic orbit. Detector: the revival fidelity |⟨ψ_t|ψ_{t+k}⟩| during generation.
- **Hunch: Floquet and time crystals, a period-2 null loop.** Repeating the same think unitary is a periodic drive. A subharmonic (period-doubled) response would make ψ alternate and never emit, which the S ≤ 4 cap hides. Halting is undecidable in general (Wang tile, Halting problem), so the cap is honest, but the cap-hit rate must be logged.
- **Hunch: Schild's ladder for the holonomy readout (item 7).** A discrete Bargmann loop is a first-order transport, with error equal to the curvature flux. Paraphrase-loop holonomy should be checked for invariance under refining the loop (inserting midpoints) before its value is trusted.

### Plan changes
The wander changed my mind on one point: the stability work guarded *numerics*, while the bigger risk is *dynamics*, where ψ localises, thermalises or loops. Two steps are added to the plan above, which brings it to 10 steps.

9. Dynamics telemetry in the bake-off harness, with the same telemetry for all QF-LM arms: participation ratio of Qψ, ψ-trajectory rank over 1k tokens, null-loop cap-hit rate, and period-2 detection, |⟨ψ_t|ψ_{t+2}⟩| > 0.99 inside null loops. Write the RED tests on synthetic localised, period-2 and scrambling maps first.
10. Revival-fidelity detector during generation (scar hunch), reported beside the repetition rate for QF-LM and GPT. This is an eval readout only, never a training signal.

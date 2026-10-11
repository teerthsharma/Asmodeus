# fcpa — reference implementation contract (numpy / scipy / scikit-learn only)

The spec is `zero/docs/specs/2026-10-11-fcpa-design.md`. Lemma IDs (L1..L27) match the spec. Every public function's docstring names the lemma it implements, and every lemma has one pytest test in `zero/fcpa/tests/` that was RED before the code existed.
Conventions: complex states are `np.ndarray` of complex128, shape (N,); batches have shape (B, N). Determinism comes from `random_state` (an int or a `np.random.Generator`). No torch.

## observer.py  (L1-L4, L23)
- `parseval_frame(N, K, rng) -> V` of shape (K+1, N), with V^H V = I. The last row is the null row v_∅.
- `born(V, psi) -> p` of shape (K+1,), summing to 1 (L1).
- `residual(p, T) -> r` (L2).
- `tail_certificate(p, T, temp) -> bound`, which raises ValueError if temp > 1 (L3).
- `post_select(p, T, temp) -> w` of shape (len(T)+1,): the selected weights plus the "other" mass (L4).
- `split_null(effects, E_new) -> effects'` (L23). Effects are a list of PSD (N, N) matrices summing to I. E_new must satisfy 0 ⪯ E_new ⪯ E_∅. It returns the effects with E_∅ replaced by E_new and E_∅ − E_new. (At N > K+1 the null effect has rank ≥ 1 and is NOT rank-one in general; keep effects as matrices, not rows.)

## router.py  (L5, decision rule)
- `enclose_scores(D, gamma_n) -> R`: elementwise error radii (separatrix style).
- `certified_topm(D, R, m) -> (T or None, frontier)` (L5).
- `decide(conformal_set, certified, requested=None) -> Decision`, with kind ∈ {ALONE, JUMP, BLEND, REFUSE_BLEND, OOD}.

## conformal.py  (L6, L7, L24)
- `SplitConformal(alpha).fit(scores_cal, labels_cal)` and `.predict_set(score_row) -> set[int]` (L6).
- `ACI(alpha, gamma, alpha1)`: `.update(err) -> alpha_t`, and `.bound(T) -> float` = (max(α1, 1−α1) + γ)/(γT) (L7).
- `birth_budget(T, alpha, gamma, alpha1) -> float` (L24).

## bounds.py  (L8-L12)
- `nu_projection(omega_t, adv, lap, f) -> nu_hat` (L8). `nu_error_bound(e, lap) -> float` (L9).
- `markov_kernel(symbols, K1) -> P` (counts with Laplace smoothing) and `kernel_interval(symbols, K1, delta) -> (P_lo, P_hi)`.
- `certified_change(int_a, int_b) -> {CHANGE, NO_CHANGE_UNCERTIFIED, REFUSE}` (L11). Overlap means REFUSE.
- `segment(signal_windows, ...) -> list[era boundaries]`.

## differentiator.py  (L13-L16)
- `Law(P, nu, f)` and `delta(law_a, law_b) -> DeltaE`, with an additive vector form (L13).
- `fingerprint(law, k) -> np.ndarray`, and `address(fingerprints, query) -> k` (L14).
- `crossing_operator(dE, generators) -> S` (unitary, by expm of skew-Hermitian; L15, L17).
- `holonomy(loop_dEs, generators) -> U`, plus `commutator_estimate` (L16).

## beds/ns2d.py
- `simulate(n, nu_schedule, f_schedule, markov_P, T, dt, n_universes, rng) -> dict(omega[u,t,x,y], eras[t], laws[k])`. Periodic 2-D vorticity, pseudo-spectral, RK4 or ETDRK, dealiased 2/3. The eras are ground truth.

## eras.py, liquid.py, crossing.py, ctc.py, rsl.py
- `EraModel` (the JEPA-in-slab): `.fit(Z_t, Z_t1)` and `.predict(Z)`. Ridge in the chart plus a SIGReg-lite isotropy penalty; a collapse guard on embedding rank (L19 test).
- `liquid_step(x, I, F, tau, A, h) -> x'`, which asserts h ≤ τ/(1+τM) (L20).
- `damped_loop(G, w0, eps, L, tol) -> (w*, n_iters, converged)`, with an honest converged flag (faraday style) and the depth bound (L22).
- `RSL.consider_birth(...)`, `.retire(...)` (L23-L26). Each era's training data comes only from its slab (L25, asserted by the dataflow).

## estimator.py  (L27)
- `FCPARegressor(BaseEstimator, RegressorMixin)` with `__init__(n_eras_max=64, d_chart=8, alpha=0.1, gamma=0.005, temp=1.0, eps=0.1, window=32, random_state=None)`, `fit(X, y=None, t=None)`, `partial_fit(X, y=None, t=None)` (exact, L27), `predict(X)` and `score`. It passes `check_estimator`.

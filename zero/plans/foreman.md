# Foreman — plan after the 3-hour session (2026-10-04)

Stance: what is QF-LM *underneath*? Two crawls: a targeted one (first order, kept because its findings
were checked against `qflm/__init__.py`), then the no-agenda wander (Seal's correction).

## Targeted findings (URL — fact — implication)

1. https://en.wikipedia.org/wiki/Belavkin_equation — the quantum filter has a linear (unnormalised, Zakai-type)
   and a nonlinear (normalised, Kushner–Stratonovich-type) form; φ = ψ/‖ψ‖. Jump (counting) and diffusive records.
   → QF-LM is a jump-record Belavkin filter. In the linear form ψ̃_T = K_{x_T}…K_{x_1}ψ₀ and
   p(x_{1:T}) = ‖ψ̃_T‖²: an *associative product*, so training without ψ-dependent think is a parallel prefix scan
   over T. The ψ-dependent think angles are the only thing that forces sequential training.
2. https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation — the normaliser p(z_k|z_{1:k−1}) is the
   per-step evidence; their product is the sequence likelihood. → nll = −Σ log‖K_xψ‖² is exactly the
   prediction-error decomposition; per-step renormalisation is HMM-style scaling, never a change of model.
   Test: Σ_t log p_t == log‖ψ̃_T‖² (telescoping) on a think-free instrument.
3. https://en.wikipedia.org/wiki/Forward_algorithm — α_t = diag(p(y_t|x))·A·α_{t−1}, p(y_{1:t}) = Σα_t, Θ(T·m²).
   → K_v = diag(√M_{·v})·Q is the amplitude version of the HMM forward step. Owned baseline falls out:
   replace amplitudes by probabilities (transition |Q_ij|², unistochastic) = an N-state HMM.
4. https://en.wikipedia.org/wiki/POVM + `qflm/__init__.py:35` — post-measurement state is not fixed by the POVM
   (Kraus freedom). In QF-LM every effect F_v = Q†diag(M_v)Q is diagonal in the same basis, so all effects
   commute: p(v|ψ) = Σ_i |(Qψ)_i|² M_iv. → The head is a classical mixture of N unigram rows read from a
   Born measurement in one fixed basis. Within one step there is no interference; quantumness lives only in
   phase coherence carried *across* steps through Q and think. The decisive control is a **dephased arm**
   (ψ → diag after each step = the HMM of finding 3). If dephased ties QF-LM, the "quantum" claim is dead.
5. https://en.wikipedia.org/wiki/Quantum_nondemolition_measurement — QND iff the observable commutes with the
   dynamics; least-disturbing measurement. → With commuting effects (finding 4) and Q as the only
   non-commuting step, QF-LM sits one rotation away from QND; a no-think arm with Q = I is fully QND,
   i.e. a classical Bayes filter with no hidden dynamics. Use it as the floor.
6. https://en.wikipedia.org/wiki/Quantum_Trajectory_Theory — a pure-state trajectory conditioned on a record
   costs N; the averaged (unconditioned) state costs N². → Summing over unobserved null records makes the
   post-token state **mixed** (rank ≤ s_max+1, compounding). `qflm/__init__.py:83-84` instead takes the coherent
   sum Σ√w_s q_s: a valid normalised LM, but *not* the filter — cross terms between distinguishable null
   histories are kept. Name it ("coherent record erasure") and test it against the exact ρ path at small N.
7. https://en.wikipedia.org/wiki/Lindbladian — no-jump evolution follows H_eff = H − (i/2)ΣL†L; jumps renormalise.
   → The null outcome *is* the no-jump branch: repeated nulls are non-unitary damping toward the
   null-favoured subspace (Zeno-like; `qflm/zeno_probe.py` exists). Think steps then fight a drift, not idle.
8. https://en.wikipedia.org/wiki/Kalman_filter — in the linear-Gaussian case the Riccati covariance and gain are
   data-independent. → Adaptive compute only buys anything because the filter is nonlinear; criterion (b)
   should correlate think steps with the *pre-observation* entropy as well as surprisal, or it rewards
   a post-hoc signal the model never had.
9. https://en.wikipedia.org/wiki/Wave_function_collapse — von Neumann process 1 (non-unitary, irreversible) vs
   process 2 (unitary). → QF-LM alternates them; the only irreversible step is the token. Fine as stated.

## Wander (no agenda)

Path (16 articles, deepest chain 7 hops):
Mathematics → Sphere packing → Leech lattice → Monstrous moonshine → Mathieu (umbral) moonshine →
Mock modular form; Leech lattice → Binary Golay code → Steiner system → Fano plane → Octonion →
Cayley–Dickson construction → Hurwitz's theorem (composition algebras) → Sedenion; Steiner system →
Kirkman's schoolgirl problem; Mathematics → Plimpton 322 → Regular number.

Five most surprising:
1. M24 shows up in the K3 elliptic genus yet has no faithful symplectic action on any K3 surface: a symmetry
   the numbers obey with no object that carries it.
2. The Leech lattice has no roots (no vectors of norm < 4) and still touches 196,560 neighbours, rigid enough
   that "no room to move any of the balls".
3. Knuth credited Inaqibit-Anu (Babylon) with the first computation longer than one second of modern machine
   time: 136 regular-number reciprocals.
4. Sedenions: (e₃ + e₁₀)(e₆ − e₁₅) = 0. One more Cayley–Dickson doubling after octonions and nonzero × nonzero = 0.
5. Kirkman's 15 schoolgirls: 7 non-isomorphic solutions, 404,756,352,000 labelled ones; all live in PG(3,2).

Instinct connections (all **hunch**, none tested):
- **hunch** Kirkman / 1-factorisations → think schedule. A butterfly layer is a perfect matching that only pairs
  coordinates differing in one bit. A resolvable design (round-robin 1-factorisation of K_N) pairs every
  coordinate with every other exactly once over N−1 rounds at the same per-layer cost. Swap the pairing
  schedule of the think unitaries, keep everything else: "butterfly effect bonded via topology" with the
  bonds chosen by a design, not by binary carries.
- **hunch** Octonions / non-associativity → syntax. Matrix products are associative, so QF-LM context is a
  holonomy blind to bracketing; a non-associative state update would depend on the parse tree. It also kills
  the parallel scan of finding 1. Not for this phase; recorded as the one idea that changes the algebra.
- **hunch** Cayley–Dickson ladder → ablation ladder. Each doubling trades one property. Run QF-LM over ℝ
  (orthogonal Q, real ψ) next to ℂ: if real ties complex, phase carries nothing and finding 4's dephased
  arm will tie too. Cheap and decisive.
- **hunch** Sedenion zero divisors → exact-zero tokens. K_vψ = 0 is a nonzero state times a nonzero operator
  giving zero: p = 0 and infinite val NLL. Softmax-parameterised M keeps M_iv > 0, but a guard test
  (min p over val ≥ 1e-12) costs one line.
- **hunch** Mock modular "shadow" → the normaliser. The non-holomorphic completion restores the symmetry and the
  shadow carries the information; in QF-LM the normaliser ‖ψ̃‖² that breaks linearity *is* the likelihood.
  Readout only: log ‖ψ̃‖ per token next to surprisal.
- **hunch** Moonshine (computation first, understanding later) → log the spectra. Dump eigenphases of Q and the
  singular values of M per checkpoint; look for coincidences before theorising.

## Built this session (track B)

- `beds/lang.py` + `tests/test_lang.py` 6/6. The leak-control test was itself broken (|V| = 300 leaves 42 merges,
  so the all-splits control could not learn the leaked word); fixed to 400, control bites.
- `run_lm.py` + `tests/test_run_lm.py` 7/7 (nurse; RED = ModuleNotFoundError; mutation: dropping the stream
  sidecar restore turns both resume tests RED). Equal-FLOPs step budgets, 3 seeds seed-major, skip-done
  rerun, sha256 pin, criteria (a)(b)(c) + talks computed into `criteria.json`, periodic checkpoint
  every 600 s via `run.save_state/load_state` plus a stream sidecar.
- Integration gap: `qflm.model.QFLM.forward` returns a scalar mean NLL; the harness contract
  (`lm/cores.py`) wants `{"logits"|"nll" [B,T], "steps", "flops"}` and `step(state, token)` for (c).

## Small plan (after the session)

1. Adapter `qflm/lm_adapter.py`: QFLM → cores contract (per-token nll, steps = expected think rounds,
   flops, `step()` over `observe/think`). RED: `test_run_lm` end-to-end with arms qf/qf_embed/qf_fixed.
2. Telescoping test (finding 2) and exact-ρ vs coherent-sum test at N = 8 (finding 6); record the gap in nats.
3. Add arms `qf_dephased` (HMM, finding 3/4) and `qf_real` (ℝ ablation); criteria gain "quantum" row:
   QF-LM beats both by > 2% ppl on ≥ 2/3 seeds.
4. Criterion (b): add Spearman(steps, pre-observation entropy) as a second reported column (finding 8).
5. Guard test: min val p ≥ 1e-12 (zero-divisor hunch).
6. Kirkman think schedule vs butterfly at equal FLOPs, one seed on fixture corpus, then TinyStories if it helps.
7. Amend PREREG (pre-result) with rows 2–4 before any TinyStories run.
8. M0 on the fixture corpus through `run_lm.py`, then TinyStories once Seal's download yes lands.

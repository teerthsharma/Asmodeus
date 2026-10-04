# PRIOR_WORK.md — Teerth Sharma (GitHub `teerthsharma`) prior work

**Second pass, 2026-10-04 (Wilson + 16 sonnet nurses).** Every one of the 89 non-fork repos was read: full trees, READMEs, docs, core source and Pages sites, via read-only `gh api` / `ghx.sh`.
- Visibility: 16 public, 73 private; 9 private repos are empty.
- The three product lines (Epsilon, Aether, Lambda) are written up repo by repo in **`PRIOR_LINES.md`**. This file is the index plus the cross-repo sections Seal asked for.
- Numbers are labelled by backing: test/CI/artifact, nurse run, or README prose only. Private repos are summarised by technique and path.

## 1. Quantum

Real quantum computation exists in **one** repo. Everything else named "quantum" is classical.

| Repo (visibility) | What "quantum" is | Paths | Backing |
|---|---|---|---|
| **topobridge-q** (private) — Taub's lane, not re-read | qiskit (`QuantumCircuit`, `Statevector`, `SparsePauliOp`, `partial_trace`) plus IBM QPU runs. "Wormhole teleportation": a message encoded as a homology cycle, sent through a "topological bridge Hamiltonian"; 6 qubits (3+3). README: "This is not a physical wormhole." | `topobridge_q.py:836` `wormhole_teleportation_experiment`; `docs/logs/`, `docs/circuits/` | README: bridge-ON recovery 0.179 vs OFF 0.120 (ibm_marrakesh), 0.169 vs 0.126 (ibm_fez); job ids in README/logs. **Not verified by Wilson** (Taub's lane). |
| aether-link (private); copies in Epsilon-Hollow `kernel/aether/aether-link`, apeiron-runtime `APEIRON/network` | "Quantum-inspired": features → Bloch angles θ = 2·atan(f); "POVM" observables cos(θ+φ), sin(θ/2 − φ); basis rotation φ ← φ + λE₂. README: "No qubits, no quantum hardware." | `src/lib.rs` `prepare_quantum_state`, `simulate_qpu_eval`; `docs/QUANTUM_THEORY.md` | Criterion: `process_io_cycle` 244.77 ns (committed run). Doc's "~1.4 ns" feature extraction conflicts with the README's 90.48 ns. |
| charlie (private) | `POVMMeasurement`: 4×4 diagonal projectors on the Betti vector; p_i ∝ squared Betti component; classical sampling plus an integer "ratchet". README and code disagree (3×3 vs 4×4). | `src/omni_topos/epsilon.py` | Nurse 29 ran it: barcodes from a 1-D cloud always give (15,0,0); seeds 1 and 42 end in `h0_only`, contradicting README §7.1. |
| hamliton (private) | "Tensor-product Hilbert space" = `np.kron` on real float64. "SU(2)/SU(3)" = nearest real orthogonal 16×16 block via SVD; the group argument is ignored. | `src/hamliton/hilbert.py`, `gauge.py` | README Hamilton scores (0.87–0.93) have **no backing** in code or tests. |
| vault-quantum-geo (private) | One GET to the ANU QRNG API, then Shamir secret sharing and a mock-chain Solidity contract (`VaultCollapsed`). | `backend/core/quantum_seed.py`, `geo_sharding.py`, `contracts/QuantumVault.sol` | None. |
| epsilon-cli (private) | Stochastic-resonance gain scheduler. "POVM / Bloch" appears only in commit message `0c4db82`. | `src/epsilon_cli/resonance.py` | README numbers do not reproduce: 0.1183 measured vs 0.0012 claimed (nurse 21 run). |
| resolvent (public) | `docs/NORTH_STAR.md`: Lippmann–Schwinger / Born-series analogy for the resolvent Neumann series (γ·ρ(P) < 1). "Reconciliation, not a result". | `docs/NORTH_STAR.md` | One 12×12 complex128 check. |
| Epsilon-Hollow (public) | `FUTURE_PLAN.md:265-290` Phase 6 vision ("the kernel does not decide, it collapses"); no code. `apps/laamba-governor/docs/V5-QUANTUM-BRIDGE.md`: Qiskit roadmap, unimplemented. | as listed | None. |
| Aether-Lang (public) | `docs/HARDWARE_SPEC.md:50-79,163-210`: speculative "Quantum-Classical Hybrid Kernels"; spec text only. | as listed | None. |
| apeiron-runtime (private) | Copy of aether-link; `QUANTUM_THEORY.md:105-106` "❌ Actual qubits ... ❌ Superposition or entanglement". | `APEIRON/network/src/lib.rs` | README prose. |

**No repo implements Lüders, Born-rule state update, Zeno, unitary evolution of complex amplitudes, CTCs or QQ-equality tests.** The only complex state vectors are in topobridge-q (qiskit).

## 2. Teleportation

| Mechanism | Repo / path | What is teleported | Status |
|---|---|---|---|
| **Context teleportation** `sys_teleport_context` | epsilon (private) `crates/epsilon/src/teleport.rs:183`; Epsilon-Hollow (public) `kernel/epsilon/epsilon/crates/epsilon/src/teleport.rs:182-233` | ≤ 64 JL-projected, L2-normalised token-embedding points on S², plus a Betti signature. Injected into a receiver "hollow manifold", gated by certified β0 = 1. | LocalVoid only; RemoteVoid returns `RemoteUnimplemented`. Tests: 4 unit tests plus `teleport_leaves_governor_unchanged.rs`. **No test shows any transferred context helps a model.** β2 = 1 is assumed, not computed (`betti_2()` returns 0). |
| **ManifoldFS file teleport** (syscall 101) | Epsilon-Hollow `kernel/seal-os/src/fs/manifold_fs.rs:436` | A same-filesystem directory-entry move (rename semantics). | Gate: ≤ 7 metadata ops, 0 bytes written, mock block store. "50 µs / 20,000×" is a modelled comparison, not a measurement (`docs/MANIFOLDFS-O1-DESIGN.md`). |
| **Wormhole teleportation** (quantum) | topobridge-q (private) `topobridge_q.py:836` | A homology-cycle-encoded message across two 3-qubit registers (qiskit + IBM QPU). | Taub's lane (§1). |
| **TeleportPipe** (data paging) | -CE-BB (private) `crates/lambda/src/teleport.rs` | MoE expert weights: NVMe → pinned host ring → VRAM. | No numbers. |
| **Absorbing "teleport"** (PageRank-style restart mass) | resolvent (public) `ceqjepa/operator.py:150` `absorbing_teleport` | A fraction of transient-row mass is moved to absorbing states, bounding ‖Q‖∞ ≤ 1 − teleport. | Ship setting teleport = 0.0. |

Other hits are illustration or naming only: sigmoid `tests/test_agent.py:263` (a test tool named "teleport"); teerthfolio `components/world/monuments/Collapse.jsx` (sculpture); teerthsharma.github.io `work.html:192` (figure caption).

## 3. Topology (code reusable here)

**No prior repo computes Betti numbers over Q, or H3.** Asmodeous ० keeps its own `ref/betti_check.py` (Z2 and Q, any dimension), `metrics.py` and `tests/test_facts.py`.

| Code | Path | What it does | Backing / limits |
|---|---|---|---|
| Z2 Vietoris–Rips PH, H0–H2 (Python, Rust) | topological-ml-toolkit (public) `python/topoml/core.py::persistent_homology`; `crates/topoml-core/src/lib.rs` | Column reduction over Z2; max_dim ≤ 2 enforced. | Rust default 64 points, O(n⁴) enumeration. ripser/GUDHI parity on tiny fixtures (`python/tests/test_tda_baseline_parity.py`). No sphere/torus H2 test. |
| Cover / nerve / Mapper | same repo, `python/topoml/topology.py` (`metric_cover`, `nerve_graph`, `mapper_graph`, `sheaf_consistency_residual`) | Nerve and Mapper are **1-skeletons only** (edges); 1-D filter. | Labelled "prototype". |
| F2 PH ≤ H2, diagram distances | Aether-Lang (public) `crates/aether-core/src/persistence.rs`, `diagram.rs` | Rips + lazy witness; exact bottleneck/Wasserstein; landscapes; images. | 12 invariant tests; 52/52 mutants caught. Slow: H2 at n = 70 takes 15.3 s. |
| Certified β0, certified top-k, S² Voronoi certify-or-fallback, Rips-β1 upper bound | Epsilon-Hollow `kernel/epsilon/epsilon/crates/aether-core/src/certified_betti.rs:80`, `attention.rs:633`, `tss.rs:389,624`, `trajectory_shape.rs` | Refuse rather than guess near merge heights or decision boundaries. | `tests/house_tss_grid_locate.rs`: 0 wrong over 25,000. 500 clouds agree with union-find. |
| Cubical H0 persistence (3-D) | cleave (private) `cleave/persist.py`, `euler3d.py` | Superlevel union-find, elder rule; χ = V−E+F−C; certified b0 = 2 band. | 72 property tests: exact Hungarian bottleneck in `tests/topo_helpers.py`, scipy parity, noise negative control. |
| Two-point collision / injectivity | monodromy (public) `monodromy/injectivity.py::collision_certificate`, `certify_injective_on_box` | λ = min |F(p)−F(q)|/|p−q|; `free_ratio` = min/median of the two-point ratios (< 8.8e-3 counts as a collision, valid n ≥ 400). Interval box proof of injectivity. | 116 tests; 8/8 closed-form maps; levels calibrated on < 30 maps. |
| n − m collision floor | caustic (public) `caustic/theorems.py` (Thm 1); branchcut (private); Aether-Lang `orbit.rs` | For an injective relation, an answer map with m distinct values on n items has ≥ n − m errors. | 60 checks, 0 violations; needs injectivity at the compared encoding. |
| Linking / writhe / Alexander (exact) | nerve (public) `nerve-topo`, `nerve-melt::alexander_det_minus_one` | Closed-form Gauss linking; i128 Bareiss determinant. | 224 tests. **No simplicial nerve** (the repo name is a pun). |
| S²-Rips entity corpus; sheaf gate; CEM-MPC with refusal | sigmoid (public) `sigmoid/mujoco/corpus.py`, `examples/s2_rips_corpus.py`, `sigmoid/sheaf.py`, `sigmoid/control.py` | Corpus: 21 nodes, 900 frames, islands = H0 of the geodesic Rips graph. Gate: ridge restriction-map residual + Mahalanobis. MPC: 96 sequences × horizon 8, rejects gate ≥ 1. | Island partition 0.855 vs 0.469 linear. Gate OOD AUROC 0.846 < Mahalanobis 0.899 (author's own `BENCHMARKS.md`). |
| PH topology claimed without code | topo-asm, hollow-asm (private) | README only, no source. | Unbacked. |
| No topology at all | pytorch-topology (private): SVD, spectral clustering, k-means; lambda-topo "Hilbert coefficients" = sorted bar lengths; charlie / topoflow PH on degenerate or default settings | — | See PRIOR_LINES.md §3 and nurse notes. |

## 4. Directly relevant to Asmodeous ० (relevance notes, Wilson labels)

- **`fold_rate` ↔ monodromy `free_ratio`.**
  - Same family: two-point non-injectivity statistics. PREREG amendment (b) scores C1 on fold_rate = fraction of points with a latent neighbour within fold_delta·median while lying more than `far` apart on M (`metrics.fold_rate`). monodromy uses the minimum/median of |F(p)−F(q)|/|p−q| over all pairs.
  - Facts: monodromy's cutoff (8.8e-3) is calibrated on 8 two-dimensional maps and is stated as not transferable below n = 400 or across dimension (`injectivity.py`, nurse 28). monodromy rejected comparing PH diagrams of X and F(X) for collisions ("measures distortion, not collision").
  - monodromy's box certificate `certify_injective_on_box` proves injectivity on a box from an interval mean-value matrix.
- **Chart selection as a many-to-one map ↔ caustic n − m.** Applies to any discrete readout whose ground truth is injective, for example state → argmax-chart over K charts with K ≥ number of true cells. The floor counts collisions with no labels (`caustic/theorems.py` Thm 1; precondition `regime.verify_injective`).
- **Teleportation ↔ T6 contextuality / OGBench teleport maze.**
  - No Seal repo implements an order-effect or QQ test. T6's statistic is validated only against `ref/identities.py` (`tests/test_t6.py`).
  - topobridge-q's wormhole protocol is a different quantity (recovery fidelity bridge-ON vs OFF) and runs on qiskit/IBM hardware, not this lane.
  - The OGBench `pointmaze-teleport-navigate` "teleport" is environment dynamics (stochastic destinations; Thirteen's Appendix D §D.7: out-cell (1,7) is a trap). No Seal repo touches it.
- **Quantum code ↔ `ref/identities.py` / Arm Q.**
  - identities.py §2/§6 (Lüders, QQ, Born/Gibbs IIA) has no counterpart in any prior repo.
  - The only prior complex-amplitude code is topobridge-q (qiskit `Statevector`).
  - The only prior "Born-like" sampling is charlie's p ∝ squared Betti counts (classical).
  - Arm Q (`atlas/coherent_observer.py`, PREREG Appendix H) is therefore the first Born-decider-with-phases code in Seal's repos found by this pass.
- **Topology-derived selectors ↔ T3/T7.**
  - Aether-Lang's topological block selector recovered ≤ random mass at equal budget, and inverting it beat random (PAPER §6.5).
  - sigmoid's sparse path is 11.6–15× faster in schedule build but loses end-to-end on distilgpt2.
  - Aether-Lang's routing helped only with H0 cluster structure in keys (gap ratio ≥ 2.70).
  - Their evaluation scaffold (random floor, oracle ceiling, placement pl) fits scoring the observer's top-m choice (PRIOR_LINES R-A2, R-A3).
- **Softmax decider over geometries ↔ LAAMBA governor.** It collapsed to one arm within 11 updates in its own demo (nurse 26 run; REINFORCE with no baseline). This is the dead-chart failure PREREG guards with λ_bal = 0.3 and the scale contract (Appendix G).
- **Certify-or-refuse ↔ Betti readout.** Epsilon-Hollow's certified β0 (merge-height band) and grid-hash fallback are the existing pattern for refusing an ambiguous nerve read (PRIOR_LINES R-E1).
- **S² data.**
  - sigmoid's S²-Rips corpus is an entity-cloud corpus with H0 ground truth, not a spherical-pendulum bed (Foreman's s2 bed is first-order on S²).
  - Epsilon-Hollow `tss.rs` gives certified great-circle Voronoi on S².
- **Contraction-certified rollout ↔ T5.** Aether-Lang `coupling.rs` and sigmoid `operator.py` carry the Banach bound ε(1−ρⁿ)/(1−ρ); the author states it is vacuous where ρ ≥ 1 (sigmoid README).
- **Systems ↔ T7.** See PRIOR_LINES.md §5. epsilon-ide has no GPU runtime; the transferable idea is cost-tiered routing with lazy load/unload.

## 5. Per-repo notes, first pass (descriptions + READMEs; §1–4 above and PRIOR_LINES.md supersede where they differ)

Corrections from the second pass:
- topo-asm, hollow-asm and pytorch-topology are **private**.
- topo-asm and hollow-asm have **no source** behind their README numbers.
- pytorch-topology computes no homology (its own README says so).
- faraday's 1.755e-16 ledger value is verified in `runs/transcript.csv`, but it is a normalised power-iteration noise floor, and its "Betti-k err" columns are signature statistics, not Betti numbers (nurse 29). The committed ledger has no `hash` column, so the README's hash-chain audit cannot run.


**resolvent** — https://github.com/teerthsharma/resolvent · https://teerthsharma.github.io/resolvent/
- JEPA planner candidate ranking: resolvent `(I − A)⁻¹ v`, ‖A‖<1, over the set of K candidate futures; builds on D-JEPA (arXiv 2609.24749, title verified: "D-JEPA: A Decision-Aligned Latent World Model").
- K=63, learned predictor, full shared error: widened D-JEPA-spec operator (ε=4) NS 0.732/0.703/0.787 vs D-JEPA as specified (ε=0.2) 0.352/0.354/0.358; 3 seeds × 20,000 eval starts.
- Bayes quality within 0.01 NS at 4.52×–7.89× fewer predictor calls; verifier rejects 22/22 planted cheats, 7/7 planted ranker cheats.
- README "What we got wrong": "D-JEPA owns the combination this repository claimed"; edge law `edge = −0.0683 + 0.6618 × closable` labelled as fitted after reading the data.
- Code: `resolvent/djepa.py` (D-JEPA reimplemented from the paper's equations), `dial.py`, `heads.py`, `shared_error.py`, `reproduce.py`. Env: Windows 11, Python 3.11.9, torch 2.14.0 (CPU).

**sigmoid** — https://github.com/teerthsharma/sigmoid (no Pages)
- Topological world-model engine: activation windows → PH barcode → Hilbert-series embedding → linear coupling operator T (ridge) → Banach fixed point; sheaf consistency gate; CEM-MPC planner (`sigmoid/control.py`).
- One imagined step 97 µs vs 86 ms forward pass (~880× per step).
- S²–Vietoris–Rips entity corpus from `google-deepmind/mujoco#3396` (`sigmoid/mujoco/`, `examples/s2_rips_corpus.py`): island partition 0.855 vs linear 0.469 vs baseline 0.487; β₀ vs true component count 1.0000; partition forecast at lag 0/1/4/16 = 0.920/0.817/0.626/0.494 vs linear 0.311/0.308/0.305/0.201.
- Granular contact components 0.778 vs 0.104; H₀ reproduces MuJoCo `mj_island` partition exactly; Triton `kernels#22` salience bit-identical at 13×.
- README abstract: whether the sparse-attention path is faster "on models this GPU can run, it is not".

**topological-ml-toolkit** — https://github.com/teerthsharma/topological-ml-toolkit · https://teerthsharma.github.io/topological-ml-toolkit/
- Rust crate `topoml-core` + Python `topoml`: `persistent_homology(points, max_dim, max_radius)` = exact Vietoris–Rips over Z/2 (`python/topoml/core.py`: `_vietoris_rips_simplices`, `_reduce_z2`); raises unless `max_dim ∈ {0,1,2}`.
- `metric_cover`, `nerve_graph(cover)`, `mapper_graph(...)`, `sheaf_consistency_residual` in `python/topoml/topology.py`. `nerve_graph` returns nodes + pairwise-intersection edges only (1-skeleton; no 2-simplices).
- `BettiCurve`, sklearn-style `PHFeaturizer`, fixtures `make_noisy_circle`, `make_two_circles`, `make_cluster_bridge`.
- Backend contracts: Safe Rust active; Python reference, C++, ASM AVX-512, Triton, PyTorch, TensorFlow listed as gated.

**Aether-Lang** — https://github.com/teerthsharma/Aether-Lang · https://teerthsharma.github.io/Aether-Lang/ (11★)
- DSL/runtime with PH as a language primitive; loops terminate on Betti invariants (`until topology.betti(...)`); description claims a Lean 4 verified kernel.
- Local clone `C:\Users\seal\Documents\GitHub\Aether-Lang\crates\aether-core\src\`: `persistence.rs` (`persistent_homology`, `persistent_homology_from_distances`, `time_delay_persistence`, `BettiNumbers3` = β₀,β₁,β₂), `attention.rs` (CPU reference sparse attention with topology-derived key selection + three baselines, `attention_mass_recovered`; header says "No backward pass" while `tests/attention_backward.rs` exists), `manifold.rs`, `topology.rs`. Tests incl. `persistence_invariants.rs`, `persistence_scale.rs`, `attention_contracts.rs`, `ablation_baselines.rs`.
- README API rows: `witness_topk(learned, topk, L, segments)`, `ph_dimension`, `orbit_partition`; `TooManySimplices` refusal.

**pytorch-topology** — https://github.com/teerthsharma/pytorch-topology (local clone exists)
- Gated low-rank linear, token co-occurrence prototype embeddings, clustered non-causal attention, relative-drift guard; follows PyTorch PR #187790 / RFC #187781.
- 18/18 CPU and 18/18 CUDA trials pass the 0.5% quality-and-rollback gate; "three of six CUDA cell medians were slower".
- README: modules "do not compute homology, persistent homology, Betti numbers"; causal attention uses an exact dense fallback.
- Historical: 39.24% memory reduction, 1.88% slower, 2.07% worse loss (`PR_THRESHOLD_NOT_MET`).

**topo-asm** — https://github.com/teerthsharma/topo-asm
- README describes AVX-512 Vietoris–Rips PH (dim_max 0–2, d ≤ 16) with a table (N=1k 0.018 s, N=10k 1.1 s vs ripser 0.4 s / 48 s).
- GitHub tree contains **only README.md**; local clone also README only. No source to back the table.

**Epsilon-Hollow** — https://github.com/teerthsharma/Epsilon-Hollow · https://teerthsharma.github.io/Epsilon-Hollow/
- Rust microkernel ("Seal OS"): frames/files/tasks placed on the unit sphere S² with Voronoi-cell ownership; each frame carries 16 points on S² (32 quantized angles, 64 bytes).
- `SYS_FIT_OBSERVE`/`SYS_FIT_REGIME` training-regime syscalls: 7/7 synthetic runs classified; KV prefix-tree eviction reaches the Belady optimum on a constructed trace; "Validation on real models is the open work."

**hollow-asm** — https://github.com/teerthsharma/hollow-asm
- Bare-metal Rust+ASM microkernel; scheduling priority from persistence-landscape distance; claims < 3 ns I/O decision cycles.

**caustic** — https://github.com/teerthsharma/caustic · https://teerthsharma.github.io/caustic/
- LM hallucination detector from the orbit partition of a relation; AUROC 0.995 on collapse-type errors, 0.67–0.71 where errors disperse; Theorem 1: n entities → m distinct answers ⇒ ≥ n − m wrong.

**branchcut / shadow** — https://github.com/teerthsharma/branchcut · https://github.com/teerthsharma/shadow
- branchcut: five results from caustic packaged for upstream targets. shadow supersedes it: 19 upstream targets, each with numpy old/new semantics, failing input, pinned diff (union-find H₀, certified top-k/top-p, JL, randomized SVD, Gauss linking).

**separatrix** — https://github.com/teerthsharma/separatrix
- Certifies top-k/argmin/threshold sets against floating-point rounding; "0 of 1,116 certified sets moved across nine engines; all 8 that moved were refused".

**monodromy** — https://github.com/teerthsharma/monodromy · https://teerthsharma.github.io/monodromy/
- Injectivity/symmetry/dimension decided from topology, set theory, fractal scaling instead of the Jacobian. README line "Jacobian conjecture · refuted July 2026" is an author claim, unchecked.

**nerve** — https://github.com/teerthsharma/nerve
- Rust polymer-chain topology (closed-form Gauss linking, Alexander determinant at t=−1, writhe); Kremer–Grest melts (Zenodo 7319837): knotted fraction 15.2% at N=823 vs published 23.6% at N=1024; 3 of 4 blindness hypotheses withdrawn; 224 tests. Not a nerve-of-a-cover implementation.

**planimeter / tangle / cleave**
- planimeter (https://github.com/teerthsharma/planimeter): pieces/faces/Euler characteristic; 528 drawings → 495 exact, 33 refused, 0 wrong vs shapely 336 wrong.
- tangle (https://github.com/teerthsharma/tangle): linking-number certificate from photos; 0 wrong certificates over 2,000 diagrams, 80 scenes, 247 photos (0 certificates issued on photos).
- cleave (https://github.com/teerthsharma/cleave): topological cell tracking for the Biohub benchmark; targets the division-Jaccard term.

**lambda-topo / topoml / charlie / topoflow / agentic-dream-topoflow / laamba-silence**
- lambda-topo: PH (ripser) → Hilbert coefficients + FAISS retrieval; README "Live Stats" all "—".
- topoml: meta-package over faraday, hamliton, topoflow, lambda_topo, phi_mem, epsilon-cli, sealvault.
- charlie (OmniTopos): Betti numbers as cosmological state variables; no measured results in abstract.
- laamba-silence: "14 engines, 40 use cases" governor; no measured results in README head.

**faraday / hamliton / Electromagnetic-Field-Data-Simulator** (EM-Sim Pages: https://teerthsharma.github.io/Electromagnetic-Field-Data-Simulator/)
- faraday: Banach fixed-point coupling tensor on FDFD EM fingerprints; 50,000-epoch burn (May 5, 2026). hamliton: N-body version over PH-barcode Hilbert space, mean Hamilton score 0.87 held-out, convergence within 500 iterations (exponent of the printed residual garbled in fetch). sigmoid reuses this pattern.

**teerthsharma.github.io / teerthsharma profile** — https://teerthsharma.github.io/
- Merged upstream PRs listed: MuJoCo disjoint-set forest ("1,281× less constraint-solver memory"; union-find public API), MuJoCo convex-hull load speedup, TensorFlow collective ordering, NVIDIA NeMo-Relay, NVIDIA topograph RBAC, Pyrefly reproducer.

**Asmodeus** — https://github.com/teerthsharma/Asmodeus (this repo; no Pages)
- Swarm runtime: Gemini "mother agent" routes work to a Qwen 3B/1.5B/0.5B swarm with telemetry. LM-specialist routing; no JEPA/topology in README.


## 6. All non-fork repos (89; visibility from `gh repo list --json visibility`)

| repo | visibility | updated | ★ | one line (GitHub description) | Pages |
|---|---|---|---|---|---|
| [Asmodeus](https://github.com/teerthsharma/Asmodeus) | private | 2026-10-02 | 0 | (no description) | — |
| [resolvent](https://github.com/teerthsharma/resolvent) | public | 2026-09-30 | 3 | One causal attention head, three switches: softmax attention, unnormalized-kernel attention and the exact Markov path product as settings of one operator. Proved in Lean 4; every failure kept on the record. | https://teerthsharma.github.io/resolvent/ |
| [shadow](https://github.com/teerthsharma/shadow) | private | 2026-09-28 | 0 | Nineteen places in working open-source code that need Teerth Sharma's mathematics, and the code for each: mirror, failing input, pinned diff, PR body. | — |
| [teerthsharma](https://github.com/teerthsharma/teerthsharma) | public | 2026-09-28 | 1 | Invented by Teerth Sharma — performance engineering, systems, research in topology | — |
| [Epsilon-Hollow](https://github.com/teerthsharma/Epsilon-Hollow) | public | 2026-09-27 | 2 | Microkernel OS research in Rust — by Teerth Sharma. | https://teerthsharma.github.io/Epsilon-Hollow/ |
| [Aether-Lang](https://github.com/teerthsharma/Aether-Lang) | public | 2026-09-27 | 11 | Aether-Lang: Rust runtime for topological ML — manifold embeddings, persistent homology, Lean 4 verified kernel. | https://teerthsharma.github.io/Aether-Lang/ |
| [teerthfolio](https://github.com/teerthsharma/teerthfolio) | public | 2026-09-26 | 1 | (no description) | — |
| [teerthsharma.github.io](https://github.com/teerthsharma/teerthsharma.github.io) | public | 2026-09-23 | 0 | Personal site. Eight pull requests merged into Google DeepMind, Google, TensorFlow, NVIDIA and Triton, with the numbers. | https://teerthsharma.github.io/ |
| [separatrix](https://github.com/teerthsharma/separatrix) | public | 2026-09-07 | 1 | Certify that your top-k, argmin or threshold was determined by your data and not by the rounding of the kernel it ran on — or refuse, and name the boundary. | — |
| [planimeter](https://github.com/teerthsharma/planimeter) | public | 2026-09-07 | 1 | Three integers for any geometry file an agent writes: pieces, enclosed faces, Euler characteristic, from the arrangement rather than a render, shipped with the snap radius that decided them or a typed refusal. | — |
| [tangle](https://github.com/teerthsharma/tangle) | public | 2026-09-05 | 1 | Photograph a pile of cables, get an integer with a proof: /lk/>=1 certifies LINKED. Zero training, refuses to guess. | — |
| [cleave](https://github.com/teerthsharma/cleave) | private | 2026-09-05 | 0 | Topological cell tracking for 3D+time microscopy. Divisions split, tracks join. | — |
| [monodromy](https://github.com/teerthsharma/monodromy) | public | 2026-08-23 | 1 | Data analysis and machine learning without the Jacobian. Injectivity, symmetry, dimension and sensitivity decided from topology, set theory, fractal scaling and chaos - not from det DF. | https://teerthsharma.github.io/monodromy/ |
| [topological-ml-toolkit](https://github.com/teerthsharma/topological-ml-toolkit) | public | 2026-08-18 | 9 | Rust + Python Topological ML Toolkit for persistent homology, Betti-curve features, manifold embeddings, topology docs, backend contracts, and E2E benchmarks for sklearn/PyTorch/TensorFlow-style pipelines. | https://teerthsharma.github.io/topological-ml-toolkit/ |
| [sigmoid](https://github.com/teerthsharma/sigmoid) | public | 2026-08-18 | 1 | A world-model inference engine built from topological coupling operators. Converts any model into a world model without touching its weights. | — |
| [Electromagnetic-Field-Data-Simulator](https://github.com/teerthsharma/Electromagnetic-Field-Data-Simulator) | public | 2026-08-18 | 1 | Topological electromagnetic field data simulator with Faraday tensor computations, Rust/Python core, and React GitHub Pages visualization | https://teerthsharma.github.io/Electromagnetic-Field-Data-Simulator/ |
| [caustic](https://github.com/teerthsharma/caustic) | public | 2026-08-18 | 3 | Hallucination detection and repair for language models, from the orbit partition of a relation. 0.995 AUROC with no ground truth, five proved bounds, and a noise intervention whose level is selected blind. | https://teerthsharma.github.io/caustic/ |
| [faraday](https://github.com/teerthsharma/faraday) | public | 2026-08-18 | 5 | Computational Faraday Tensor — discover E×H electromagnetic coupling via topology-fixed-point projection. | — |
| [nerve](https://github.com/teerthsharma/nerve) | public | 2026-08-18 | 0 | Topological witnesses for polymer chains in Rust — Gauss linking in closed form, Alexander determinant at t=-1, and the falsification harness that withdrew three of its own four hypotheses. | — |
| [branchcut](https://github.com/teerthsharma/branchcut) | private | 2026-08-16 | 0 | Five elementary results about many-to-one maps, extracted from caustic, and ten screened upstream targets for them. | — |
| [epsilon-ide](https://github.com/teerthsharma/epsilon-ide) | private | 2026-08-14 | 0 | (no description) | — |
| [dedevanfan](https://github.com/teerthsharma/dedevanfan) | private | 2026-08-14 | 0 | (no description) | — |
| [upgraded-app](https://github.com/teerthsharma/upgraded-app) | private | 2026-08-14 | 0 | (no description) | https://teerthsharma.github.io/upgraded-app/ |
| [fruit-debug](https://github.com/teerthsharma/fruit-debug) | private | 2026-08-14 | 0 | (no description) | https://teerthsharma.github.io/fruit-debug/ |
| [epsilon](https://github.com/teerthsharma/epsilon) | private | 2026-08-14 | 0 | Epsilon: Rust compiler toolkit — by Teerth Sharma | — |
| [charlie](https://github.com/teerthsharma/charlie) | private | 2026-08-14 | 0 | Universal Topology Engine — cosmological emergence via persistent homology phase space. | — |
| [agentic-dream-topoflow](https://github.com/teerthsharma/agentic-dream-topoflow) | private | 2026-08-14 | 0 | Render persistent homology as animated 3D topological landscapes. Visualize Betti bars, persistence landscapes, and VR complex collapses. The topology visualizer that makes TDA actually understandable. | — |
| [hamliton](https://github.com/teerthsharma/hamliton) | private | 2026-08-14 | 0 | N-Body Electromagnetic Coupling — SU(2)/SU(3) gauge theory with topological fixed-point projection. | — |
| [redvelvetcake](https://github.com/teerthsharma/redvelvetcake) | private | 2026-08-14 | 0 | (no description) | https://teerthsharma.github.io/redvelvetcake/ |
| [EPSILON-PHASE](https://github.com/teerthsharma/EPSILON-PHASE) | private | 2026-08-14 | 0 | (no description) | — |
| [trex-phase3](https://github.com/teerthsharma/trex-phase3) | private | 2026-08-14 | 0 | (no description) | https://teerthsharma.github.io/trex-phase3/ |
| [ghost-runner](https://github.com/teerthsharma/ghost-runner) | private | 2026-08-14 | 0 | (no description) | https://teerthsharma.github.io/ghost-runner/ |
| [treasure-runner](https://github.com/teerthsharma/treasure-runner) | private | 2026-08-14 | 0 | (no description) | https://teerthsharma.github.io/treasure-runner/ |
| [matter-engine](https://github.com/teerthsharma/matter-engine) | private | 2026-08-14 | 0 | (no description) | https://teerthsharma.github.io/matter-engine/ |
| [treasurerunner](https://github.com/teerthsharma/treasurerunner) | private | 2026-08-14 | 0 | (no description) | https://teerthsharma.github.io/treasurerunner/ |
| [BALLOn](https://github.com/teerthsharma/BALLOn) | private | 2026-08-14 | 0 | (no description) | https://teerthsharma.github.io/BALLOn/ |
| [lambda-topo](https://github.com/teerthsharma/lambda-topo) | private | 2026-08-14 | 0 | Topological Memory System — TDA-based storage and retrieval with ripser + FAISS; manifold intelligence for physics. | — |
| [seal-demon-tts](https://github.com/teerthsharma/seal-demon-tts) | private | 2026-08-14 | 0 | cause why not | — |
| [pytorch-topology](https://github.com/teerthsharma/pytorch-topology) | private | 2026-08-14 | 0 | Out-of-tree experimental topology-aware PyTorch modules with honest tests and benchmarks | — |
| [aether-link](https://github.com/teerthsharma/aether-link) | private | 2026-08-14 | 1 | High-Performance I/O Prefetch Kernel for DirectStorage, WSL2 and HFT workloads. | — |
| [neuralwhisper](https://github.com/teerthsharma/neuralwhisper) | private | 2026-08-10 | 0 | AI-powered whisper synthesis with 82M-parameter neural TTS running 100% in your browser | — |
| [aether-wave](https://github.com/teerthsharma/aether-wave) | private | 2026-08-10 | 0 | (no description) | — |
| [epsilon-cli](https://github.com/teerthsharma/epsilon-cli) | private | 2026-08-10 | 0 | Epsilon CLI — minimal error propagation and tolerance analysis tool. | — |
| [topoflow](https://github.com/teerthsharma/topoflow) | private | 2026-08-10 | 0 | Topological data flow analysis and visualization. | — |
| [vault-quantum-geo](https://github.com/teerthsharma/vault-quantum-geo) | private | 2026-08-10 | 0 | (no description) | — |
| [Seal-Gif-Maker](https://github.com/teerthsharma/Seal-Gif-Maker) | private | 2026-08-10 | 0 | Create custom animated GIFs of cute seals for any situation. Describe a scenario, and let AI bring a charming seal animation to life for you | — |
| [Radioactivity-shiedling](https://github.com/teerthsharma/Radioactivity-shiedling) | private | 2026-08-10 | 0 | (no description) | — |
| [Reading-Buddy](https://github.com/teerthsharma/Reading-Buddy) | private | 2026-08-10 | 0 | Upload a PDF to start a reading session. Your AI pal will chime in with interesting facts and questions to keep you engaged. | — |
| [BrandGen_AI](https://github.com/teerthsharma/BrandGen_AI) | private | 2026-08-10 | 0 | ai based persona maker for brands | — |
| [my-own-game](https://github.com/teerthsharma/my-own-game) | private | 2026-08-10 | 0 | (no description) | — |
| [sealvault](https://github.com/teerthsharma/sealvault) | private | 2026-08-10 | 0 | (no description) | — |
| [BrandGen-AI](https://github.com/teerthsharma/BrandGen-AI) | private | 2026-08-10 | 0 | (no description) | — |
| [tower1](https://github.com/teerthsharma/tower1) | private | 2026-08-10 | 0 | (no description) | — |
| [topobridge-q](https://github.com/teerthsharma/topobridge-q) | private | 2026-08-10 | 0 | (no description) | — |
| [topoml](https://github.com/teerthsharma/topoml) | private | 2026-08-10 | 0 | Topological machine learning — persistent homology for ML pipelines. | — |
| [phi-mem](https://github.com/teerthsharma/phi-mem) | private | 2026-08-10 | 0 | Memory hierarchy profiling and analysis tools. | — |
| [lambda-azure-engine](https://github.com/teerthsharma/lambda-azure-engine) | private | 2026-08-10 | 0 | (no description) | — |
| [-CE-BB](https://github.com/teerthsharma/-CE-BB) | private | 2026-08-10 | 0 | (no description) | — |
| [sealpasswordgen](https://github.com/teerthsharma/sealpasswordgen) | private | 2026-08-10 | 0 | (no description) | — |
| [netwon](https://github.com/teerthsharma/netwon) | private | 2026-08-10 | 0 | (no description) | — |
| [NEW](https://github.com/teerthsharma/NEW) | private | 2026-08-10 | 0 | (no description) | — |
| [samplenextjs](https://github.com/teerthsharma/samplenextjs) | private | 2026-08-10 | 0 | Public sample Next.js Three.js polar portfolio world | — |
| [geo](https://github.com/teerthsharma/geo) | private | 2026-08-10 | 0 | (no description) | — |
| [tom-jerry](https://github.com/teerthsharma/tom-jerry) | private | 2026-08-10 | 0 | (no description) | — |
| [topo-asm](https://github.com/teerthsharma/topo-asm) | private | 2026-08-10 | 0 | Topological manifolds in x86_64 SIMD assembly — AVX-512 persistent homology. | — |
| [hollow-asm](https://github.com/teerthsharma/hollow-asm) | private | 2026-08-10 | 0 | Bare-metal Rust+ASM microkernel with topological scheduling and sub-3ns I/O latency. | — |
| [pgtable-asm](https://github.com/teerthsharma/pgtable-asm) | private | 2026-08-10 | 0 | x86_64 long-mode paging in pure NASM assembly — KPTI, PCID, huge pages, recursive mapping. | — |
| [vec-simd](https://github.com/teerthsharma/vec-simd) | private | 2026-08-10 | 0 | Hand-written AVX-512 SIMD math library — Mandelbrot, matmul, FFT, dot product in pure x86_64 assembly. | — |
| [websiteforme](https://github.com/teerthsharma/websiteforme) | private | 2026-07-13 | 0 | (no description) | — |
| [mehul](https://github.com/teerthsharma/mehul) | private | 2026-07-08 | 0 | (no description) | — |
| [laamba-silence](https://github.com/teerthsharma/laamba-silence) | private | 2026-05-27 | 0 | A way a fixed formualted way for topological governess | — |
| [mailing-list](https://github.com/teerthsharma/mailing-list) | private | 2026-05-20 | 0 | (no description) | — |
| [seal](https://github.com/teerthsharma/seal) | private | 2026-05-18 | 0 | seal | — |
| [seal-admin](https://github.com/teerthsharma/seal-admin) | private | 2026-05-18 | 0 | (no description) | — |
| [sealseal](https://github.com/teerthsharma/sealseal) | private | 2026-05-18 | 0 | (no description) | — |
| [admin](https://github.com/teerthsharma/admin) | private | 2026-05-18 | 0 | (no description) | — |
| [sealsealseal](https://github.com/teerthsharma/sealsealseal) | private | 2026-05-18 | 0 | (no description) | — |
| [aadioactivity-shiedling](https://github.com/teerthsharma/aadioactivity-shiedling) | private | 2026-05-18 | 0 | (no description) | — |
| [seal-di-website-](https://github.com/teerthsharma/seal-di-website-) | private | 2026-05-18 | 0 | seal seal seal seal | — |
| [seal-di-website-2.3](https://github.com/teerthsharma/seal-di-website-2.3) | private | 2026-05-18 | 0 | seal seal seal seal seal seal seal  | — |
| [mealbuddy](https://github.com/teerthsharma/mealbuddy) | private | 2026-05-18 | 0 | MealBuddy — Find meal companions near you | — |
| [apeiron-runtime](https://github.com/teerthsharma/apeiron-runtime) | private | 2026-05-18 | 0 | (no description) | — |
| [apiclassproject101](https://github.com/teerthsharma/apiclassproject101) | private | 2021-04-13 | 0 | (no description) | — |
| [projectC32--](https://github.com/teerthsharma/projectC32--) | private | 2021-04-13 | 0 | (no description) | — |
| [sunset87](https://github.com/teerthsharma/sunset87) | private | 2021-04-13 | 0 | (no description) | — |
| [sunsetproject--](https://github.com/teerthsharma/sunsetproject--) | private | 2021-04-13 | 0 | (no description) | — |
| [sunsetime-](https://github.com/teerthsharma/sunsetime-) | private | 2021-04-13 | 0 | (no description) | — |
| [sunset](https://github.com/teerthsharma/sunset) | private | 2021-04-13 | 0 | (no description) | — |
| [NEWNEWTONISHERE](https://github.com/teerthsharma/NEWNEWTONISHERE) | private | 2021-04-10 | 0 | (no description) | — |

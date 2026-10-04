# PRIOR_LINES.md — Seal's three product lines (Epsilon, Aether, Lambda), read in detail

Wilson, 2026-10-04. Sources: read-only nurse reads (sonnet) of full trees, READMEs, docs, core source and Pages sites, via `gh api` / `ghx.sh GET`. Wilson read aether-link directly. Every fact carries a repo path. Private repos are summarised by technique and path only.

"Relevance notes" (end of each line) are Wilson's labels linking a fact to an Asmodeous ० item. They are not claims about the prior work.

Cross-reference: **topobridge-q** (private, IBM-quantum "wormhole teleportation", qiskit + IBM QPU runs) belongs to Taub's lane and is not re-read here.

---

## 1. Epsilon line

### 1.0 Verdict on "teleportation"
The Epsilon line contains **two classical teleport mechanisms**. There is **no quantum teleportation** in it: zero hits for qubit, entangle, unitary, Hilbert, Lüders, Zeno, CTC or sheaf in the source of epsilon, EPSILON-PHASE or Epsilon-Hollow.

1. **Context teleportation** — `sys_teleport_context`. A ≤ 64-point cloud on S² (JL-projected, L2-normalised token embeddings) is moved into a receiver "hollow manifold", gated by a Betti-0 certificate.
   - Locations: private `epsilon/crates/epsilon/src/teleport.rs:183`; public copy `Epsilon-Hollow/kernel/epsilon/epsilon/crates/epsilon/src/teleport.rs:182-233`.
   - The remote variant always returns `RemoteUnimplemented`.
2. **ManifoldFS file teleport** — an O(1) metadata-only directory move.
   - Location: `Epsilon-Hollow/kernel/seal-os/src/fs/manifold_fs.rs:436`.
   - Interfaces: syscall `SYS_TELEPORT=101` (`syscall/table.rs:593`); boot bench gate: ≤ 7 metadata ops and 0 file bytes written per move, on a mock block store.

### 1.1 epsilon-ide (private, 89 files) — systems lens
General pass (nurse 21). Systems deep read (nurse 32) is summarised in §5. Also mirrored at `Epsilon-Hollow/kernel/epsilon/epsilon-ide`.

**Purpose** (`README.md`): a local multi-tier LLM coding assistant ("70B + 7B + 1.1B") on 16 GB RAM + 8 GB VRAM, "part of my larger goal to create SEAL".

**Three generations**
- `pentesting/` (despite the name, the "SEAL mega" IDE):
  - FastAPI backend.
  - `backend/core/vram_guard.cpp`: pybind11 atomic VRAM-tier fence with cudaEvent sync.
  - `backend/core/extreme_hardware.cpp`: Windows large pages and a named-pipe token stream.
  - `backend/core/nvme_cuda.cpp`: IoCompletionPort NVMe→RAM layer streaming. Its "Fused FlashAttention" is a stub: "Simulated launch, as actual compilation requires NVCC integration".
  - `backend/core/perplexity_rollback.py`: recompute a layer unpruned when the entropy ratio exceeds 1.8.
- `engine/v1/` ("Potato PC", BitNet 1.58-bit):
  - 4-agent serial pipeline; ROUTER is a keyword dict.
  - `inference/tinygrad_kv.py`: int8 ring-buffer KV with top-64 `sparse_read` by q·k/√d averaged over heads.
- `engine/v2/`:
  - 1.5B / 7B / 33B tiers.
  - `agents/router.py`: additive keyword score clamped to 0–10; fast ≤ 2, balanced 3–6, deep ≥ 7.
  - `tiers/model_manager.py`: llama-server process swap with an idle watchdog.
  - `clara/oracle.py`: TF-IDF in sqlite-vec.

**Results.** No tests, no benchmark artifacts. The README/PDF numbers are budget arithmetic.

**Keyword hits.** None for teleport, quantum, observer, chart or atlas.

**Provenance (nurse 32).**
- README support contact and `/home/lovekesh/...` paths point to collaborator Lovekesh Anand; PR #3 is from LovekeshAnand.
- The Epsilon-Hollow copy (`kernel/epsilon/epsilon-ide`, 57 files) is a cleaned version: 10 files byte-identical, 39 differ only by a license header, 3 differ in substance.
  - `pentesting/backend/main.py` gains a workspace jail and a bearer token.
  - `shared_memory_ipc.py` gains a CRC32 header and per-session segment names.

### 1.2 Epsilon-Hollow (public, 1,122 files; Pages https://teerthsharma.github.io/Epsilon-Hollow/; DOI 10.5281/zenodo.20264206)
**Purpose.** README: "A kernel that reads the shape of what it runs, and would rather say 'cannot decide' than guess." Seal OS is an x86_64 no_std Rust UEFI research OS whose state is geometry on S². README states 194,779 lines Rust, 509 files.

**Architecture** (workspace `Cargo.toml`; `kernel/seal-os` is excluded from the workspace and runs under QEMU only)
- `kernel/seal-os/`: 223 .rs files, 96,110 lines.
  - `memory/topo_ram.rs` (TopoRAM)
  - `process/scheduler.rs`
  - `ml_engine/stratum.rs` (training-regime classifier)
  - `ml_engine/foliation.rs` (KV prefix tree)
  - `fs/manifold_fs.rs`
  - `atlas/` (signed loadable ELF modules called "charts")
  - `net/topological.rs` (Voronoi route table)
  - `syscall/table.rs` (69 calls)
- `kernel/epsilon/epsilon/crates/aether-core`: the maths.
  - `tss.rs` (S² Voronoi)
  - `trajectory_shape.rs`
  - `certified_betti.rs`
  - `persistence.rs` (bounded Rips/Witness PH, Z2, through tetrahedra: H0–H2)
  - `diagram.rs` (bottleneck, Wasserstein, landscapes)
  - `nettree.rs` (Sheehy net-tree)
  - `attention.rs`
  - `scheduled.rs` (topology-derived sparse attention, a port of triton-lang/kernels#22)
  - `world_model_horizon.rs`
- `kernel/aether/`: `aether-verified` (Rust theorem kernels T1–T10 + Lean 4), `aether-link`, Aether-Lang crates, `aether-nexus` (simulator).

**Algorithms** (paths under `kernel/epsilon/epsilon/crates/aether-core/src/` unless stated)
- **S² placement** (`tss.rs`):
  - `SphericalVoronoiIndex::locate` (:624) is a linear scan over K centroids by great-circle distance.
  - `SphericalGridHashIndex::locate` (:389) is certify-or-fallback: it accepts the 3×3-cell answer only if the best distance ≤ distance to the region boundary, otherwise it does a full scan.
  - `CUBE_CENTROIDS` (:282): 8 cube vertices.
  - `verify_separation` (:58).
  - Packing bound p_max = 4/sin²(θ_min/2).
- **Training-regime classifier** (`trajectory_shape.rs`; kernel side in `seal-os/src/ml_engine/stratum.rs`):
  - delay embedding (v_t, v_{t−1}, v_{t−2}), then arc-length resampling;
  - MST edge stats give ε*;
  - `is_monotone` (:241) certifies loop = 0 before any complex is built;
  - `cycle_rank` (:196) = E − V + β0 on the Rips graph at 1.68·ε* (an upper bound on Rips β1);
  - `fold_score` (:268) = min(1, cycle_rank/m);
  - `participation_ratio` (:309) returns NaN (refusal) below its rounding bound;
  - `classify` (:699) outputs Underfit, WellFit, Overfit or Collapsing.
  - The kernel observes 2 scalars per step: syscalls 120–124, with `SYS_FIT_OBSERVE`=121.
- **Certified β0** (`certified_betti.rs:80`): MST merge heights. The answer is certified iff no height lies in [s/√r, s√r]; otherwise `Refused{i,j,height}`.
- **Certified top-k** (`attention.rs:633`): Higham γ_n interval separation.
- **KV prefix tree** (`seal-os/src/ml_engine/foliation.rs`):
  - BLOCK_TOKENS 8, MAX_CHILDREN 32. Eviction only of free faces, so the resident set stays a connected rooted subtree.
  - Policies: Foliation, LRU, Random, Belady, Locality, Adaptive (duels, PSEL_MAX 8). The system default is LRU.
- **Context teleport**: see §1.3. In the Epsilon-Hollow copy, β0 is certified (`manifold.rs:549`, BETA0_RATIO 10) and `assimilate` (:607) is all-or-nothing.
- **"Atlas"** (`seal-os/src/atlas/mod.rs`) is loadable-module vocabulary, not differential geometry:
  - charts = Ed25519-signed ELF64 ET_REL modules; "overlap" edges = a nerve;
  - `bind_overlap` refuses cycles (`AtlasError::NerveCycle`, ~:275);
  - 6 negative controls (docs/RECORD.md:729).

**Measured results.** Source: `docs/RESULTS.md`, CI run 36165748105 @9ebbe2e, QEMU TCG.
- stratum: 7/7 synthetic runs correct; 4,792 bytes/stream. Gate: `seal-mkimage --check-mlfit-proof`.
- Foliation boot trace (30 requests, 1,680 tokens, 24-block pool): 952 bp, equal to Belady 952. LRU 0; random 619 (Foliation beats random on 32/32 seeds).
- Chat trace (16 conversations, 4 live): Foliation 5,284 vs LRU 8,068, Belady 8,143, so Foliation **loses to LRU**.
- Certify-or-refuse nearest centroid: 1,215/5,000 wrong → 0 wrong. `tests/house_tss_grid_locate.rs` asserts 0 wrong over 25,000 queries.
- Host tests: 861 passed / 0 failed / 2 ignored across 94 targets.
- In-kernel tests: 514/514 (CI run 31487106604).
- Microbench (QEMU TCG, emulated cycles):
  - scheduler-select-next p50 304,184
  - toporam-alloc 6,126 (64/64 target-cell hits)
- **Traps (nurse 20):**
  - `seal-mkimage/src/main.rs:4699-4700` `MLFIT_PROOF_LOG`/`KV_POLICY_LOG` are parser test fixtures with made-up values.
  - "50 µs teleport, 20,000×" (`docs/research/epsilon_hollow_hft_ml.tex:208,409-410`; `docs/MANIFOLDFS-O1-DESIGN.md:24-25`) compares against a modelled 2 GB copy at 0.5 ns/byte. It is not a measurement.
  - `kernel/aether/aether-nexus/epsilon-ide-telemetry.json`: 50 events, 0 successful teleports.

**Author-stated limits** (RESULTS.md Limits 1–12):
- no task is ever scheduled; no user space;
- T4 governor refused ("alpha+beta/dt=5.01 >= 1");
- CI red; ML services never saw a real model;
- Foliation loses to LRU on the chat trace;
- Lean statements are not connected to kernel code;
- `manifold.rs` `betti_2()` returns 0 for every input (`tests/house_betti2_not_imposed.rs`).

**Code-reading findings (nurse 20, not run):**
- `process/task.rs:219` `compute_embedding` puts every default task in the positive octant, so it lands in Voronoi cell 0.
- `memory/topo_ram.rs:169,178` uses squared Euclidean distance on u16 embeddings, not great-circle distance.

### 1.3 epsilon (private, 38 files; homepage teerthsharma.vercel.app)
**Purpose.** README:3 "Geometric State Transfer via Topological Surgery on Hollow Manifolds". README:15: "O(P) geometric state transfer between autonomous agents ... (P ≤ 64)". The repo description ("Rust compiler toolkit") does not match the content.

**Code** (`crates/epsilon/src/`)
- `bridge.rs`: seeded JL projection, M ∈ R^{D×E} with N(0,1)/√D entries (Xoshiro256**), then L2-normalise. The ε-graph uses DEFAULT_SPHERE_EPSILON 1.0. Retry: ε ×= 1.2 up to 2.0 while β0 > 1.
- `manifold.rs`:
  - β0 = DFS components;
  - β1 = E − V + β0 (the graph cyclomatic number);
  - β2 = 2 − β0 + β1, clamped at 0. This is an Euler identity that assumes S², not a homology computation.
  - Only β0 gates injection.
- `governor.rs`: PD controller ε ← ε + α·e + β·de/dt, with e = 1000 − Δ/ε. `SurgeryPermit` zeroes β and restores it.
- `teleport.rs:183` `sys_teleport_context`: LocalVoid only.

**Results.** No benchmarks. 37 `#[test]` in the epsilon crate.

**Nurse reading (unrun).** Adjacency is `[u64;256]` with `if i < 64`, so graphs with more than 64 points likely get wrong β0.

**Limits** (SPECIFICATION.md §8): D=3 limits β2 observability; semantic fidelity is bounded by JL distortion; remote transfer is unimplemented.

### 1.4 EPSILON-PHASE (private, 31 files; 1 commit 2026-04-15)
**Purpose** (README): "an interactive wind-simulation physics engine with a built-in frontend, powered by a numeric robustness layer for quantization and noise handling."

**Code**
- `src/epsilon_phase/architecture/stochastic_resonance.py`: gain ×1.15 (cap 0.30) when |Δmetric| < 1e-3, else ×0.96.
- `subtractive_dither.py`: Δ = 1/4096, y = Δ·round((x+d)/Δ) − d.
- `physics/wind_engine.py`: Stam stable fluids.
- `rust/epsilon_prefetch/src/main.rs`: "Betti-0 trajectory prefetch validator" in its help text only. No topology is computed; it predicts the identity on a 97%-stay synthetic trajectory.

**Results.** No recorded numbers.

**Keyword hits.** None for teleport, quantum, observer or measurement.

### 1.5 epsilon-cli (private, 18 files)
**Purpose** (README): "Context window optimization via stochastic resonance — adaptive noise injection for LLM training." The GitHub description ("error propagation and tolerance analysis") matches no file.

**Code** (`src/epsilon_cli/resonance.py`, 149 lines): gain g starts at 0.02, bounded to [0.005, 0.30].
- After 2 consecutive stagnant epochs (Δ < 1e-3): g × 1.15.
- Otherwise: g × 0.96.
- Noise injection: x + g·N(0,1).

**POVM / Bloch.** These appear only in commit message `0c4db82`, never in any file at any commit. The charlie README's "POVM quantum measurement" attribution to epsilon-cli is not backed by code.

**Measured** (nurse 21 ran HEAD locally)
- `run_optimization_benchmark(128, 200, 42)`: final loss 0.118297, against the README's 0.001204. **README numbers do not reproduce.**
- Noise-deleted control: 0.117871. The noise term is inert (≈ 0.4%).
- 6 tests pass.

### 1.6 Relevance notes — Epsilon line → Asmodeous ० (Wilson labels)
- **R-E1, certify-or-refuse.** The `certified_beta0` merge-height band and `SphericalGridHashIndex` certify-or-fallback (aether-core `certified_betti.rs:80`, `tss.rs:389`) are a ready pattern for **refusing an ambiguous Betti read** of the observer nerve at a τ inside a merge band. Related: PREREG §0.4, the τ-sweep.
- **R-E2, nerve vocabulary.** `atlas/mod.rs` already uses chart / overlap / nerve-cycle words for module loading. This is a naming collision with OBSERVER-ATLAS, not prior maths.
- **R-E3, β2 caution.** epsilon's β2 = 2 − β0 + β1 is an Euler identity assumed on S². Epsilon-Hollow's own tests (`house_betti2_not_imposed.rs`, `house_betti1_is_not_a_rank.rs`) retract it. Asmodeous ० binds Betti numbers by homology instead (`ref/betti_check.py`, `tests/test_facts.py`).
- **R-E4, S² geometry.** The great-circle Voronoi on S² with certified separation (`tss.rs`) applies to the S² bed: geodesic distances, cell assignment of ground-truth states, and a ground-truth atlas reference for P-COUNT.
- **R-E5, regime classifier.** The stratum classifier (`trajectory_shape.rs`: fold_score, participation_ratio, Collapsing) is a 2-scalar per-step collapse detector. Relevant to T4a collapse-time detection as a cross-check on `g_erank < 1.5`.
- **R-E6, no quantum.** Nothing in the Epsilon line implements superposition or measurement. The PREREG Arm Q framing (linear glue + nonlinear top-m) has no code precedent here.

---

## 2. Aether line

### 2.1 Aether-Lang (public, 255 files on `master`; Pages https://teerthsharma.github.io/Aether-Lang/; DOI 10.5281/zenodo.21997728)
**Purpose** (README): "A programming language that computes with shape. Its loops end when the shape of the data stops changing, and its answers come with proof, or not at all." The core premise (Betti-stability stopping beats a tuned scalar tolerance) is stated as **unmeasured** (PAPER.md §9). `ConvergenceCond::BettiStable` is never constructed.

**Code size.** Rust 53,228 lines (`crates/`); Lean 11,637 lines (`Aether/`).

**PH engine** (`crates/aether-core/src/persistence.rs`)
- **F2 only, max homology dim 2.** `validate_common` rejects > 2.
- Filtrations: Rips and lazy witness. Plain column reduction (`reduce_z2`).
- Presets: h2_default 48 points / 1e6 simplices; interpreter default `low_load` (refuses ≥ 19 points).
- Never truncates; refuses with `TooManySimplices`.
- No ripser/GUDHI parity (PAPER §9: "largest correctness debt").
- Measured: H0 n=4000 335.0 s; H1 n=300 131.3 s; H2 n=70 15.3 s (single run, one machine).

**PAPER.md 29 results** (§3.1–3.29, table at PAPER.md:230-260), including:
- Rips/flag complex; ∂∂ = 0; β_k rank formula; diagram sizes C(n,2)+1 etc.; CSEH stability.
- Exact bottleneck/Wasserstein; landscapes; persistence images.
- Circle H1 death 2r·sin(π⌈n/3⌉/n), tested to 1e-12.
- Takens delay embedding; streaming-graph β1 = E − V + β0 (≤ 64 points).
- Online softmax; attention backward; block salience (admitted ≠ H0 barcode).
- Placement statistic pl = (m(S) − m(R))/(m(O) − m(R)).
- Routing gap ratio γ (structured ≥ 2.70, chained ≤ 1.04, 6 trials).
- Admissible pruning bound; drift second difference; Chebyshev guard.
- Governor law; ring gossip; learning primitives; convergence predicates.

**Governor** (`governor.rs`)
- ln ε_{t+1} = ln ε_t − α·e_t − β(e_t − e_{t−1}), with e = clamp(1 − R/R*, −1, 1), R = Δ/ε, R* = 1000, α = 0.25, β = 0.05, ε ∈ [0.001, 10].
- Stable iff 0 < α < 2 − 2β (derived).
- Settling test only.

**Sparse attention** (`attention.rs`, `scheduled.rs`): selectors Dense, Local, Random (floor), OracleTopK (ceiling), Topological, TopologicalRouted, Adaptive.
- **Negative result** (PAPER §6.5; `crates/aether-gpu/FEATURES.md`; RTX 4060):
  - The scheduled topological block selector recovers less mass than random at top-k 32, seq 512 (−109%).
  - The inverted ranking beats random.
  - End-to-end associative recall: topological 60.5%, random 61.0%, dense 92.4% (McNemar: indistinguishable).
- Routed selector on clustered keys: 0.449–0.733× dense cost, placement +0.94 to +0.995. Uniform keys cost 0.999×.

**GPU and evidence**
- `crates/aether-gpu`: wgpu, 20 WGSL kernels, 107 tests (80 hardware-gated). "nothing calls it".
- Mutation: 52/52 mutants caught.
- Lean: 48 theorems, 0 sorry; 45 closed by `native_decide`; not in CI; not linked to Rust.

**Keyword hits.** None for quantum (except a prompt string), teleport, observer (JS only), atlas, sheaf, partition of unity or world model.
- "nerve" hits are the sibling repo's name; "monodromy" is the sampled-injectivity module (`monodromy.rs`).
- `coupling.rs`: the only state-transition model, z_{t+1} = W·φ(z,a), ridge fit, with a Banach fixed point when σ_max(T0) < 1 (port of sigmoid).

### 2.2 aether-link (private, 265 files; 19 KB `src/lib.rs`, the rest mostly committed Criterion output under `docs/`)
**Purpose** (`README.md`): "An allocation-free I/O **prefetch decision kernel** in Rust ... a decision function, not an I/O path." Input is a slice of LBAs; output is a bool. It does no I/O itself.

**Quantum status** (README; `docs/QUANTUM_THEORY.md`): "'Quantum-inspired' means the mathematical formalism only — Bloch sphere, POVM observables, basis rotation. No qubits, no quantum hardware."

**Algorithm** (`docs/QUANTUM_THEORY.md` §1–4; `src/lib.rs` `prepare_quantum_state` ~:230, `simulate_qpu_eval` ~:289)
1. Six live features: Δ, V = Δ/2, Welford variance, Chebyshev RMS, decay history (0.8), log-density entropy.
2. θ_i = 2·atan(f_i/scale_i); φ = weighted azimuthal mean.
3. Bloch vector (sin(θ/2)cos φ, sin(θ/2)sin φ, cos(θ/2)), normalised by `fast_inv_sqrt`.
4. Three "POVM" observables: E₁ = cos(θ+φ), E₂ = sin(θ/2 − φ), E₃ = cos(θ·φ_az).
5. Feedback: φ ← φ + λ₂E₂ (mod 2π); ε ← clamp(ε + λ₁E₁, 0.1, 0.9); p_fetch = sigmoid(−(λ₃E₃ + bias)). No trained parameters.

**Measured** (README; committed Criterion run in `docs/`, 95% CI)
| Benchmark | Result |
|---|---|
| `process_io_cycle` | 244.77 ns [244.59, 244.95] |
| `extract_telemetry` | 90.48 ns |
| `prepare_quantum_state` | 61.14 ns |
| fast_atan | 4.69 ns |
| fast_exp | 2.97 ns |
| fast_sigmoid | 3.35 ns |
| 1,000,000 cycles | 239.3 ms (≈ 4.18 M decisions/s, single thread) |

- Stream-length scaling is non-monotonic (+13.8% at length 100); README calls this "not yet explained".
- README withdraws tail-latency percentiles: "any previously published figures for them have been withdrawn".

**Inconsistencies**
- `docs/QUANTUM_THEORY.md` §1 says features extract "at ~1.4 ns", against the README's 90.48 ns `extract_telemetry`.
- Nurse 20 reports that the Epsilon-Hollow copy (`kernel/aether/aether-link/docs/BENCHMARKS.md`) says 14.6 ns mean, P50 17.2 ns, on an unspecified machine.

**Keyword hits.** None for teleport, observer, manifold, Betti or topology in README, docs or lib.rs (Wilson grep).

### 2.3 aether-wave (private, 2,043 blobs; 1,751 of them under `.agent/` — agent skills/notes, not project code)
**Purpose** (README): "AETHER-WAVE: The Kinetic Engine ... a real-time physics simulation that happens to produce sound."

**Real code** (~600 lines)
- `aether-rivermax/src/kinetic/physics.rs`: 1D mass-spring chain, symplectic Euler.
- `filters.rs`: 5-oscillator "glass" bank; one-pole "liquid" lowpass.
- `rtp.rs` / `constants.rs`: RTP and ST 2110-30 sizing (8000 packets/s; 2,304,000 bps stereo 48k), with unit tests.
- `aether-engine`: a wgpu passthrough WGSL kernel (bit-exact test).
- `rivermax-sys/src/stubs.c`: every `rmax_*` returns OK.
- `aether-mother/aether_mother.py`: gRPC placeholder ("Placeholder for actual AirLLM generation").

**Spec only, no code.** 3D FDTD, β-VAE timbre manifold, harmonic-masked block-sparse attention (`.agent/knowledge/aether_mathematical_foundations.md`, `.agent/workflows/p5*`).

**Numbers.** Every number in `paper/aether_wave_paper.tex` (12 ± 3 µs RDMA latency, "40×", etc.) has **no code or benchmark** in the repo.

**Keyword hits.** None for quantum or teleport.

### 2.4 Relevance notes — Aether line
- **R-A1, prior "quantum-inspired" code.** aether-link is Seal's only "quantum-inspired" code located so far:
  - a Bloch-sphere angle encoding plus cosine/sine "POVM" observables and a basis-rotation feedback loop;
  - fully classical, untrained, O(1).
  - It is a precedent for the Arm Q framing (classical GPU, quantum formalism), not an implementation of superposition-as-glue.
  - Its README states the classical/quantum boundary explicitly, in the style PREREG §2 Arm Q needs.
- **R-A2, a topology-derived selector lost to random.** Aether-Lang's own measurement (PAPER §6.5, FEATURES.md, RTX 4060) shows its topological block selector at or below random at equal budget. Its evaluation scaffold is a direct template for scoring the observer's top-m choice against controls:
  - random floor, oracle ceiling, placement pl = (m(S) − m(R))/(m(O) − m(R)), McNemar on end-to-end recall;
  - files: `crates/aether-core/src/attention.rs` `attention_mass_recovered`, `scheduled.rs` `block_mass_recovered`.
- **R-A3, routing pays only with H0 structure.** Paths: PAPER §3.22/§8.4, `attention.rs::routing_plan`.
  - Euclidean routing equals dot-product ranking only for equal key norms, hence routing on unit-normalised keys.
  - The routing gain appeared only when keys had H0 cluster structure: gap ratio γ ≥ 2.70 structured vs ≤ 1.04 chained.
  - This bears on T3 (whether keys for K ≫ 1 charts carry cluster structure) and on the T4b centred-embedding edge rule.
- **R-A4, PH limits.** Aether-Lang PH is F2-only with max dim 2 and is slow (H2 n=70: 15.3 s). It cannot give RP³'s β3 or Q Betti numbers. Asmodeous ० keeps its own Z2/Q code (`metrics.py`, `ref/betti_check.py`).
- **R-A5, contraction-certified world model.** `coupling.rs` (from sigmoid) is a linear world model with a Banach contraction certificate (σ_max < 1; rollout error ε(1−ρⁿ)/(1−ρ)). It is a candidate certified-rollout check for T5 h-step predictions.
- **R-A6, aether-wave is not relevant.** No topology, quantum or routing code; its paper numbers have no backing.

---

## 3. Lambda line

### 3.1 lambda-topo (private, 33 files; homepage lambda-topo.dev did not resolve, `ENOTFOUND`)
**Purpose** (`README.md`): "a topological data analysis system that transforms raw data into compact topological signatures for efficient storage and retrieval."

**Code** (`lambda_shappire/`)
- `topology.py::compute_h0_barcode`: O(n²) union-find on edges ≤ 2.0, giving H0 bars only. **ripser is not imported in `lambda_shappire/`**, contrary to the README.
- `hilbert_coeffs_from_barcode`: finite lifetimes, sorted descending, divided by the max, truncated or zero-padded to 20. The name is a label only; no Hilbert-space object exists.
- `memory.py::TopologicalMemoryStore`: L2-normalised signature, then **one FAISS L2 search**. There is no coarse-to-fine stage, and `IVF*` silently falls back to Flat.
- `manifold_trainer/`: sklearn t-SNE / Isomap / LLE / MDS / PCA, best by composite score.
- `hollow_manifold_sim.py`: 2D FDFD cavity eigenmodes, ripser H0/H1.

**Results**
- README performance numbers (10k points ~0.8 s via ripser, etc.) have no benchmark, test or artifact.
- `manifold_training_results.json`: LLE "wins" at 0.636 because its reconstruction error is ~1e-7.
- `manifold_results.json`: betti_0 = 0, betti_1 = 0, topological_score 0.
- Tests: `tests/test_placeholder.py` (`assert True`).

**Keyword hits.** None for teleport, observer, chart, atlas, nerve or governor.

### 3.2 lambda-azure-engine (private, 32 files)
**Purpose** (`README.md`): "an experimental sandbox for exploring ternary quantization, toy Mixture-of-Experts routing, and bit-packed kernels ... does not run a 14B/200B model." "Many numbers and terms in earlier docs were aspirational."

**Real code**
- `rust-kernel/src/lattice_compute.rs::dot_product_32`: a ternary dot product over 32 trits per u64 via XOR + popcount.
- `python-bridge/triton_ternary_gemm.py`: a 16-trit uint32 Triton GEMM with a pack/unpack round-trip check.
- `quantize_to_ternary.py`: TWN quantisation (0.7·std).

**Toy code**
- `kv_cache_homology.py`: single-scale union-find KV clustering.
- `lambda_engine.py`: LSH router.
- `train_ternary_moe.py`: linear top-1 router.
- `PQRouter`: `% 32768`.
- `padic_quantize`: divisibility by 3. Its docstring claim "exact with zero precision loss" is false.

**Numbers.** `benchmark_suite.py` values are hard-coded in `mock_run_model`; they are not measurements.

**Keyword hits.** None for quantum, teleport, observer, measurement or chart.

### 3.3 laamba-silence (private, 193 files; 6 commits, last 2026-05-27; no tests, no CI)
**Purpose** (`README.md:3`): "FL Studio for manifolds. 14 engines. 40 use cases. One brain." It is a Tauri + React + Python CLI desktop shell around a 7-way manifold chooser.

**Governor** (`topological_governor_ml.py`)
- 13 unnormalised vitals: Levina–Bickel intrinsic dimension, distance stats, kNN spectral gap λ₂/λ_max, KMeans elbow, local-PCA eigenratio std, small-world, sparsity.
- `PolicyNet`: 64-ReLU → 7-softmax MLP.
- Update (`:330-360`) is REINFORCE with R = −loss clipped to [−1, 1] and **no baseline**, plus an entropy nudge below 0.5 nats.
- In the app (`cli/governor_cli.py:163,533`) the governor is **never trained**: a fresh seed-42 policy, no checkpoint.
- Nurse 26 ran the demo (`PYTHONHASHSEED=0`): it collapsed to `hyperbolic_poincare` for every dataset within 11 updates, then crashed with `MemoryError`.

**Battle Royale** (`governor_orchestrator.py`)
- Five preprocessing maps, all scored by Euclidean KMeans silhouette.
- Ranking 0.7·score + 0.3·speed; multiplicative-weights update.
- `TopologyFusion`: random, untrained attention.

**14 engines**
- 10 run the same `builtin_wrapper.py`: an untrained governor plus one preprocessing map + KMeans silhouette. "TopoBridge-Q" here is L2-normalise + KMeans: no Qiskit.
- Faraday, Hamilton, Aether-Lang and Phi-Mem wrappers import external repos and error if those are absent.
- Phi-Mem "barcodes" are SHA-256 truncated to 16 hex characters.

**Use cases.** The 20 coded use cases score manifolds with **hard-coded mock numbers**; the other 20 are pseudo-code.

**UI.** The persistence diagram and Betti curves are **synthetic** functions of the vitals (`src/panels/TopologyScope.tsx:85,220-236`).

**Numbers.** README headline numbers (Faraday 1.755e-16, Hamilton 0.91, topo-asm 11.8×, TopoBridge-Q +50%) are copied from `TEERTH_GITHUB_CONTEXT.md`, a 25-repo scrape, and are not reproduced here.

**Keyword hits.** "Teleport" appears once, describing Epsilon-Hollow. There is no qiskit import in any `.py` file.

### 3.4 Epsilon-Hollow `apps/laamba-governor` (public, 106 files; nurse 19 read it in full)
**Stack.** Tauri 2 + React 18 + TypeScript + Python CLI, plus a "native" Seal OS edition.

**Real code**
- `governor_orchestrator.py` (529 lines): `TopologyRunner` scores 5 embeddings with sklearn metrics.
  - The embeddings are euclidean PCA + k-means; spherical L2; "hyperbolic_poincare" (tanh-radial, then a squared-Euclidean metric, not hyperbolic); Grassmannian QR; product sin/cos.
  - Ranking 0.7·score + 0.3·(1 − latency).
  - Multiplicative weights w ← w·exp(lr·reward).
  - `TopologyFusion` is self-attention with random, untrained Wq/Wk/Wv.
- `topological_governor_ml.py` (511 lines): 13 "vitals" (Levina–Bickel intrinsic dimension, kNN-graph spectral gap λ₂/λ_max, local-PCA curvature, ...). `PolicyNet` is a 2-layer MLP over 7 manifolds, trained by a REINFORCE-like rule ("Hack: scale lr by reward").

**What the numbers are worth**
- `mock_downstream_task` (`topological_governor_ml.py:418-441`) returns **hard-coded scores** (0.92 spherical if cyclic, 0.90 hyperbolic if tree-like, ...), so the "governor learns" demo is circular.
- `native.rs` vitals are closed-form fakes.
- `native/governor.aether` picks a battle winner by `math.random()`.
- `HANDOFF.md`: no real PH (ripser TODO).
- No tests under `apps/`.
- `docs/V5-QUANTUM-BRIDGE.md` is a roadmap only (Qiskit/Aer, QPE-Betti, SWAP test), and is unimplemented.

### 3.5 -CE-BB ("λ lambda-engine", private, 18 files; nurse 31)
**Purpose** (`README.md:1`): "Zero-Copy Bit-Packed Omnicloud MoE Orchestrator".

**Code**
- `crates/lambda/src/teleport.rs` `TeleportPipe`: NVMe → pinned host ring → VRAM streaming of MoE expert weights, with `ExpertManifest` mapping expert ids to byte slices. This is data paging, not quantum.
- Cauchy–Schwarz block gate G_j = max|Q·K|/(‖Q‖‖K‖), skip if G_j < γ, with γ adapted by Chebyshev on perplexity variance.
- ε-graph union-find Betti-0 KV clustering, O(N²D).
- 16-trit/uint32 Triton ternary GEMM.

**Results.** No measured numbers in the files read.

**Author limits** (`docs/RESEARCH_PAPER.md` §6): the gate is "necessary but not sufficient"; "prefetch accuracy collapses" under adversarial routing.

---

### 3.6 Relevance notes — Lambda line
- **R-L1, governor as a precedent.** The LAAMBA governor (softmax policy over manifold labels from 13 cheap vitals, REINFORCE without a baseline) is Seal's closest prior to "a softmax decider over geometries". In its own demo it collapsed to one arm within 11 updates (nurse 26 run). Asmodeous ० already mandates load balance (PREREG Appendix G: λ_bal = 0.3) and the scale contract, because of exactly this failure mode (dead charts).
- **R-L2, vitals as diagnostics.** The vitals (Levina–Bickel intrinsic dimension, kNN spectral gap, local-PCA eigenratio) are cheap label-free diagnostics. They are usable as reported diagnostics of the learned glued space g (for example, intrinsic-dimension estimate vs n), never as inputs to the router (L-GIVEN).
- **R-L3, lambda-topo signature.** The signature (sorted, max-normalised H0 bar lengths) is translation-, rotation- and scale-invariant by construction. Its own demo shows it cannot separate translated copies. It is not usable as a topology readout for Asmodeous ०.
- **R-L4, data paging, not quantum.** -CE-BB `TeleportPipe` (NVMe→pinned→VRAM expert paging) and lambda-azure-engine's XOR-popcount ternary dot product (`lattice_compute.rs`) are the only GPU-memory-path code in the Lambda line. Both are relevant to T7 only if chart weights stop fitting in VRAM at K ≥ 10⁴. They are not relevant to the quantum framing.

## 4. Other repos read in this pass (quantum/teleport lane, nurse 27)
- **vault-quantum-geo** (private, 10 files). "Quantum" means:
  - one HTTP GET to the ANU Quantum Random Numbers API (`backend/core/quantum_seed.py`);
  - Shamir secret sharing 3-of-4 → 3-of-5 → 3-of-5 (`backend/core/geo_sharding.py`, `secretsharing` library);
  - a mock-chain Solidity contract (`backend/contracts/QuantumVault.sol`, event `VaultCollapsed`);
  - a scripted frontend animation.
  - No quantum computing, no tests, no numbers. The README says MIT; LICENSE is an attribution license.
- **projectC32--**, **NEWNEWTONISHERE** (private): empty since 2021 (HTTP 409, "Git Repository is empty").
- **teerthsharma** profile and **teerthsharma.github.io**: no quantum code. The only "teleport" is Epsilon-Hollow's "context teleport" figure alt text (`work.html:192`).
- **topobridge-q**: Taub's lane (see top).

## 5. Systems lessons for T7
**epsilon-ide verdict (nurse 32).** It is a Python orchestration layer plus three small C++ stubs. It has no GPU kernel runtime, no compiler/IR, no CUDA graphs, persistent kernels, sort-by-expert grouping or gather avoidance, and no benchmark artifacts.
- `nvme_cuda.cpp` `launch_fused_flash_attention` is commented out; no `.cu` file exists.
- `vram_guard.cpp` is a global atomic/mutex that serialises model tiers, the opposite of launch amortisation.

Transferable patterns, each with its ceiling:
- **Cost-tiered routing.** `engine/v2/backend/agents/router.py`: additive keyword score clamped to 0–10, tier ≤ 2 / ≤ 6 / else. `tiers/model_manager.py` lazily cold-starts tiers and unloads them after 300 s idle.
  - Hand-written rule, no learning.
  - Process-level (kill/restart a server), so it does not apply at per-expert granularity.
- **CPU top-k pattern.** `inference/kv_cache.py::sparse_read`: `np.argpartition` (O(n)) plus a sort of the k subset, then a fancy-index gather copy. Head-averaged, not per-head. Clip-only int8 with no scale factor.
- **Pinned/large-page buffers.** `extreme_hardware.cpp`: `VirtualAlloc(MEM_LARGE_PAGES)` with fallback; unbuffered overlapped NVMe reads (`nvme_cuda.cpp`, effectively synchronous).
- **Tutorial only (PDF `BitNet_DRAM_CUDA_Complete_Guide`, not repo code):** tiled shared-memory ternary matmul, bump-pointer GPU pool, two-stream copy/compute overlap.

**Facts from other lines that bear on T7:**
- **Sparse attention CPU reference.** aether-core `scheduled.rs` is a topology-derived sparse-attention CPU reference (port of triton-lang/kernels#22). `attention.rs` header: "no GPU path in this workspace" (nurse 20).
- **Allocation-free O(1) kernel.** aether-link is allocation-free, no heap, O(1) per decision, 244.77 ns/decision on CPU. It is a CPU-side precedent for a fixed-cost per-query router (§2.2).
- **Measured RTX 4060 facts from Seal's own repos (most relevant to T7):**
  - sigmoid: Triton block-salience schedule 11.6–15.0× faster and bit-identical to triton-lang/kernels#22; incremental salience 8.6–21.7× per block.
  - The same sparse-attention path **loses end-to-end** on distilgpt2: 51/70/78 vs dense 102/97/95 tok/s (sigmoid README, nurse 28).
  - Aether-Lang `aether-gpu`: 20 WGSL kernels. Matmul GPU-vs-CPU crossover at n = 128. The "10×–63× at n = 512" magnitude is unstable (GPU term moves 5.2× between runs), so only the crossover is claimed (`crates/aether-gpu/FEATURES.md`, nurse 22).
  - Read across: on the same GPU class, small-kernel launch overhead and gather/scatter dominated Seal's earlier sparse paths. This matches Taub's PREREG Appendix C §C.4: the atlas ties dense within the ~60–75 µs graph-launch floor at B = 1, and is 3–4× slower at B = 1024 until gather avoidance and fused or grouped kernels are used.

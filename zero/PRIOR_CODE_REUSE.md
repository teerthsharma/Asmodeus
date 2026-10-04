# Prior code (Seal's repos) that could be reused in the OBSERVER-ATLAS

Chase read these directly: the nurse dispatch was refused because the 20-subagent limit was reached.

- **Method:** read-only, via `ghx.sh GET repos/teerthsharma/<R>/git/trees/HEAD?recursive=1` and `/contents/<PATH>`, on 2026-10-04 at HEAD.
- **Tests:** every file listed below is marked "tests present" from the repo's tree listing. No repo's tests were run here.
- **Labels:** **Fact** = read in the code. **Lesson** = Chase's engineering judgement.

## monodromy
**Fact:**
- `monodromy/injectivity.py::free_ratio(F, X)` computes the two-point ratio |F(p) − F(q)| / |p − q| over all sampled pairs. It returns the minimum divided by the median.
  - No Jacobian is involved. It uses numpy only and costs O(n²).
  - It refuses n > 4000 (`MAX_N`) and is only readable for n ≥ 400 (`FREE_MIN_N`).
  - Its docstring reports, on 8 maps: injective 1.5e-2 to 7.1e-1, non-injective 1.0e-3 to 5.1e-3 (gap factor 3.02).
  - `FREE_LEVEL = 8.8e-3`.
- `lower_ratio_witness(F, X)` returns the colliding pair.
- `phdim.py::ph_dimension` estimates PH dimension from how H0 total persistence scales with sample size. It depends on ripser.
- Tests are present: `tests/test_injectivity.py`, `tests/test_beats_jacobian.py`.

**Lesson:**
- Mirror `free_ratio` (about 10 lines) as a per-chart **chart-validity / fold check**. Apply it at eval only to state → z_k on the inputs routed to chart k.
- It complements `metrics.fold_rate`. A chart with ratio below FREE_LEVEL folds its own domain, which is the failure the atlas exists to avoid.
- Do not import it. The package pulls in ripser, persim and mpmath at import, and its calibration is on 2-D/3-D maps, so the level must be recalibrated on the beds.

## topological-ml-toolkit
**Fact:**
- `python/topoml/core.py::_reduce_z2(simplices, max_dim)` does the standard Z2 column reduction on any list of (vertex tuple, filtration value). It is pure Python.
  - The public `persistent_homology` builds only Vietoris–Rips and caps max_dim at 2. `_reduce_z2` itself has no cap.
- `python/topoml/topology.py::nerve_graph(cover)` returns edges only (a 1-skeleton).
- `topology.py::metric_cover` builds radius balls, which is a given cover.
- Tests are present: `python/tests/test_topology_families.py`, plus contract tests.

**Lesson:**
- Mirror `_reduce_z2` (about 35 lines) to compute **true nerve persistence over τ** up to H3.
  - The filtration value of a simplex σ is −max over inputs x of min over k ∈ σ of w_soft[x, k].
  - This replaces `metrics.most_persistent`'s run-length rule with real birth and death in τ, for Foreman to consider in metrics.
- Do not use `nerve_graph`: it has no b2.
- Do not use `metric_cover` as an atlas component, because a hand-given cover fails L-GIVEN. It is acceptable only as the Mapper baseline.

## pytorch-topology (`src/torch_topology/`)
**Fact:**
- `TopologicalSpectralAttention`, `SpectralPrototypeEmbedding` and `TopologyGatedLowRankLinear` all compress after training, via spectral embedding plus deterministic k-means (`_spectral.py`) or an SVD rank cut.
- `guard.py::LyapunovSpectralGuard` snapshots training state and rolls it back when loss or spectral-energy spikes.
- Tests are present.

**Lesson:**
- Nothing to import. Post-hoc clustering of keys is given structure, not structure learned by SGD.
- The rollback guard would change the training procedure for one arm only, which breaks L-BR fairness unless every arm gets it.

## Aether-Lang
**Fact:**
- `crates/aether-core/src/persistence.rs` (1044 lines, Rust) computes bounded Rips/Čech persistent homology over Z2, up to tetrahedra (H ≤ 2 exact, simplices to dim 3).
- `attention.rs` and `scheduled.rs` implement topology-derived sparse attention, a CPU reference with a port of the Triton kernel.
- `bindings/python/aether_lang/__init__.py` exposes no persistence API.
- Tests are present: `tests/persistence_invariants.rs`, `persistence_scale.rs`.

**Lesson:** Not callable from Python, so there is nothing to import. Its invariant tests are a checklist that a nerve-persistence mirror should satisfy.

## T1 owner verdict
**Fact:** None of the four repos implements a learned router, a learned cover or charts. Every cover or cluster in them is given or post-hoc.

**Lesson:** This does not change the T1 owner verdict.

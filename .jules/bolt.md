# Performance Learnings and Codebase Insights

## Environment & Testing
- Unit tests in `zero/tests/` should be executed from within the `zero/` directory (`cd zero && pytest tests/...`).
- Fast pre-commit gate tests can be run via `cd zero && sh queue/pre-commit`.
- Core test suite (`test_laws.py`, `test_facts.py`, `test_asmodeous.py`, `test_asmodeous_ops.py`, `test_observer_atlas.py`, `test_phase_sphere.py`, `test_qflm.py`, `test_run_lm.py`, `test_sigreg.py`) passes smoothly.
- `mujoco` is installed if MuJoCo swingup tests are executed.

## Codebase Architecture
- **Asmodeus**: A Python/PyTorch-based experimental swarm runtime for hierarchical language-model orchestration and predictive worlds.
- **QF-LM (Quantum-Filter Language Model)**: "Measure to remember" architecture where tokens are measurement outcomes over CP^(N-1) Hilbert states.

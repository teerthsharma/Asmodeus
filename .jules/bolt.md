# Codebase Insights & Performance Learnings

## Overview
Asmodeous is a Python/PyTorch-based swarm runtime for predictive world modeling using quantum-inspired observer mechanics, lazy top-$m$ chart evaluation, and CPTP closed-loop observer dynamics.

## Test Suite & Environment Setup
- Execution command: `PYTHONPATH=zero python3 -m pytest zero/tests/`
- Fast pre-commit tests: `cd zero && python3 -m pytest -q tests/test_laws.py tests/test_facts.py tests/test_queue.py`
- Required runtime dependencies: `torch`, `numpy`, `scipy`, `tokenizers`, `pytest`.
- Note: Optional beds like `swingup.py` require `mujoco`. When running full test suites without optional beds, filter out or ignore `test_swingup.py` and `test_t5_plan.py`.

## Pre-Commit Regulations
The repository uses a pre-commit check (`zero/queue/pre-commit`) enforcing:
1. No AI attribution headers/comments (e.g. `Co-Authored-By`).
2. Immutability of `zero/PREREG.md` once result files exist in `zero/runs/`.
3. Fast test verification passing cleanly.

## Key Architectural Components (`zero/atlas/`)
- `asmodeous.py`: Primary architecture combining Stiefel-retracted Parseval frame POVMs, lazy top-$m$ chart predictors, Cayley unitary local dynamics, learned Čech gauge links, and damped CPTP time-traveller fixed-point loops.
- `observer_atlas.py`: OBSERVER-ATLAS parent model (real-valued JEPA charts with softmax observer and partition-of-unity gluing).
- `phase_sphere.py`: PHASE-SPHERE parent model (amplitudes on $S^2$ under a Born decider, gauge-fixed charts).
- `time_traveller.py`: Closed-loop observer reading candidate futures before committing.

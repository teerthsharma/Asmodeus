# Asmodeus

Asmodeus is a specialist-swarm orchestration runtime focused on scientific rigor, measurable outcomes, and bounded active compute.

## Scientific Objective

The project investigates whether a large pool of small specialists can approach dense-model behavior while activating only a budgeted subset per query.

Working objective:

$$
Q_{virtual} \approx Q_{dense}(P_{ref}) \quad \text{with} \quad P_{active}(t) \ll P_{total}
$$

where:

- $P_{total}$ is total registered specialist capacity.
- $P_{active}(t)$ is active capacity at runtime step $t$.

## Runtime Invariants

1. Budgeted sparse routing

$$
P_{active}(t) = \sum_{i \in S_t} P_i \le P_{budget}
$$

2. Admission-gated activation

$$
S_t \subseteq M_{ready}(t)
$$

3. Verified model supply chain
- Downloaded specialist artifacts must pass integrity and compatibility checks before routing.

## Canonical Documentation

Use this as the primary source:

- docs/MASTER_SCIENTIFIC_SPECIFICATION.md

Supporting documents:

- docs/OVERVIEW.md
- docs/SYSTEM_ARCHITECTURE.md
- docs/GOALS.md
- docs/TOPOLOGY_100M_TO_10B.md
- docs/RUNTIME_INTEGRATION_DOCTRINE.md
- docs/EVALUATION_PLAN.md
- docs/IMPLEMENTATION_PLAN.md
- docs/TASKS.md

## Repository Map

Core orchestration:

- asmodeus/scout.py
- asmodeus/task.py
- asmodeus/worker.py
- asmodeus/task_manager.py

Routing and registry:

- asmodeus/registry.py
- asmodeus/router.py
- asmodeus/runtime.py

Acquisition and verification:

- asmodeus/downloader.py

Topology and fusion:

- asmodeus/cluster_topology.py
- asmodeus/world_model.py
- asmodeus/swarm_convolution.py

Inference and runtime bridge:

- asmodeus/true_inference.py
- asmodeus/hybrid_adapter.py

User interfaces:

- cli.py
- talk_to_asmodeous.cmd

## Setup

Python:

- Python 3.10+

Install dependencies:

```powershell
python -m pip install -e .
```

If editable install is not desired:

```powershell
python -m pip install numpy torch transformers scipy tqdm fastapi uvicorn websockets accelerate
```

## Run Tests

```powershell
python -m unittest discover -s tests -v
```

## Run CLI

Direct:

```powershell
python cli.py --response-mode llm --persona scientist --expert-dir .\expert_checkpoints
```

Windows launcher:

```powershell
.\talk_to_asmodeous.cmd
```

## Runtime Notes

1. GPU is recommended for larger inference tiers.
2. The launcher installs missing non-torch dependencies automatically.
3. Torch is not auto-reinstalled by launcher logic to prevent accidental CUDA wheel replacement.
4. Optional WSL fallback model fetch path can be enabled with:

```powershell
$env:ASMODEUS_ENABLE_OPENCLAW_WSL="1"
```

## Current Maturity

Validated:

1. Core scout-manager-worker orchestration.
2. Skill-based routing and recovery behavior.
3. Registry readiness accounting and sparse routing.
4. Download and verification fallback behavior.

In progress:

1. Benchmark parity studies versus dense references.
2. Cost-latency-quality tradeoff characterization.
3. Extended reproducibility and ablation reporting.

## Citation

When reporting results, cite repository URL and exact commit hash used in experiments.

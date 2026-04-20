# System Architecture

This document maps runtime layers to concrete repository modules.

The authoritative math and invariants are in:

- docs/MASTER_SCIENTIFIC_SPECIFICATION.md

## Layered View

1. Query and task decomposition
- Modules: asmodeus/scout.py, asmodeus/task.py
- Role: infer objectives, derive required skills, produce executable task units.

2. Assignment and execution control
- Modules: asmodeus/task_manager.py, asmodeus/worker.py
- Role: worker assignment, retries, transient-failure recovery.

3. Specialist registry and readiness
- Modules: asmodeus/registry.py, asmodeus/downloader.py
- Role: register specialists, enforce admission gates, track ready set.

4. Sparse routing and topology orchestration
- Modules: asmodeus/router.py, asmodeus/cluster_topology.py, asmodeus/runtime.py
- Role: top-k specialist selection, cluster-level coordination, runtime policy enforcement.

5. State fusion and world consistency
- Modules: asmodeus/swarm_convolution.py, asmodeus/world_model.py
- Role: aggregate active specialist states, maintain shared state with confidence and contradiction handling.

6. Inference and hardware bridge
- Modules: asmodeus/true_inference.py, asmodeus/hybrid_adapter.py
- Role: model-tier generation backend and optional hardware integration path.

7. User interfaces
- Modules: cli.py, talk_to_asmodeous.cmd
- Role: operator-facing control and startup workflows.

## Deterministic Data Flow

$$
q_t \rightarrow \text{skill extraction} \rightarrow \text{router ranking} \rightarrow S_t \rightarrow \text{specialist execution} \rightarrow z_t \rightarrow y_t
$$

with hard constraint:

$$
S_t \subseteq M_{ready}(t)
$$
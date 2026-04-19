# Asmodeus

A scientifically grounded framework for swarm-native agent orchestration, with a central hypothesis that a sparse, coordinated topology of small specialist models can approximate the capability profile of a much larger monolithic model.

## Abstract

Asmodeus studies multi-agent intelligence as a dynamic graph system in which specialist agents communicate, route work adaptively, and aggregate evidence through structured pooling. The primary design is model-centric: many narrowly specialized micro-masters in the 1M to 10M parameter range are composed into a topological swarm. This separates static capacity from active compute and creates a falsifiable research program around quality, latency, cost, and robustness.

All production architecture paths are constrained to a hybrid runtime doctrine:

1. Lambda Azure style sparse expert activation, streaming, and memory-aware execution.
2. Epsilon-Hollow style orchestrator control loops, tiered decision routing, and resilient bootstrap/recovery.
3. Internet-downloaded SLM acquisition as the default model supply mechanism.

The repository currently contains a minimal executable core for scout-driven task discovery, worker execution with retry-based recovery, and task-manager orchestration. Formal topology and benchmarking plans are documented for iterative expansion.

## Scientific Objective

Determine whether sparse coordination among many small specialists can achieve near-dense quality while reducing active compute.

Primary hypothesis:

$$
Q_{virtual}(\{P_i \in [1M,10M]\}) \approx Q_{dense}(P_{ref})
$$

where $P_{ref}$ is a dense large-model reference (for example 10B class), under bounded coordination overhead and controlled world-model drift.

## Mandatory Runtime Doctrine (Non-Negotiable)

### 1) Lambda Azure Execution Logic

1. Sparse active-set inference instead of dense full-set activation.
2. Streaming model materialization and quantization-aware execution.
3. Demand-paged model residency and memory-budgeted scheduling.
4. Cost-aware routing over available specialists.

### 2) Epsilon-Hollow Orchestration Logic

1. Orchestrator-centered control flow for agent, IDE, or full workflows.
2. Tiered control roles for planning, logic-gating, and architectural arbitration.
3. Bootstrap-first operations (dependency setup, model prep, health checks).
4. Retry/fallback semantics for transient failures and degraded-mode continuity.

### 3) Internet SLM Supply Logic

All specialist models are treated as internet-sourced SLM assets unless explicitly pinned as internal artifacts.

Readiness set:

$$
\mathcal{M}_{ready}(t)=\{m \in \mathcal{M}_{internet}\mid D_m(t) \land V_m \land C_m\}
$$

where:

- $D_m(t)$: download complete and local cache materialized
- $V_m$: verification passed (checksum/provenance/license policy)
- $C_m$: runtime compatibility passed (format/device/quantization constraints)

Routing is constrained to verified-ready specialists only.

## Research Questions

1. Capacity composition: Can distributed specialist capacity compose into strong global reasoning?
2. Routing economics: What top-k policy minimizes cost for a target quality threshold?
3. Coordination efficiency: How much communication is necessary before returns diminish?
4. Reliability: Can retry, reroute, and verification loops keep failure rates bounded under perturbation?

## Formal System Definition

Let:

- $E = \{e_1, \dots, e_N\}$ be specialists, with per-specialist parameters $P_i$.
- $h_i^t \in \mathbb{R}^d$ be specialist state at time $t$.
- $w_t \in \mathbb{R}^m$ be global world state.
- $A_t \in \mathbb{R}^{N \times N}$ be weighted swarm adjacency.
- $E_t \subseteq \mathcal{M}_{ready}(t)$ be specialists currently verified and routable.

Static parameter budget:

$$
P_{total} = \sum_{i=1}^{N} P_i
$$

Micro-master topology bands:

$$
P_i \in \{1M, 10M\}
$$

This makes topology design a model-allocation problem over specialist nodes, not a data-sharding problem.

Sparse activation budget:

$$
P_{active}(t) = \sum_{i \in S_t} P_i, \quad |S_t| = k_t \ll N
$$

### Local Message Passing

$$
m_i^t = \sum_{j \in \mathcal{N}(i)} A_{ij}^t W_m h_j^t
$$

### State Update

$$
h_i^{t+1} = \sigma(W_s h_i^t + m_i^t + B w_t)
$$

### Swarm Convolution and Pooling

$$
c_\ell^t = \sum_{i=1}^{N} K_\ell(i) h_i^t, \quad
z_t = \mathrm{concat}(c_1^t, \dots, c_L^t)
$$

### Routing Rule

$$
r_i^t = g(h_i^t, x_t, w_t), \quad S_t = \mathrm{topk}(r^t, k_t), \; S_t \subseteq E_t
$$

### Multi-objective Optimization

$$
J = \lambda_q Q - \lambda_c C - \lambda_l L + \lambda_r R
$$

where $Q$ is quality, $C$ is compute cost, $L$ is latency, and $R$ is robustness.

## Computational Complexity (Conceptual)

Let $d$ be latent width, $|E_t|$ active communication edges at time $t$, and $k_t$ active specialists.

Dense monolith-like compute scales with total parameters:

$$
\mathcal{O}(P_{total})
$$

Sparse swarm step cost scales approximately as:

$$
\mathcal{O}(\sum_{i \in S_t} P_i + |E_t| d^2)
$$

This decomposition clarifies the design goal: maximize quality gain per marginal increase in $k_t$ and communication edges.

## Assumptions and Threats to Validity

1. Specialist separability: assumes tasks can be decomposed into reusable capability slices.
2. Router generalization: assumes routing policy remains calibrated under distribution shift.
3. World-model coherence: assumes stale or conflicting state can be bounded by TTL and arbitration.
4. Benchmark representativeness: assumes benchmark mix captures real deployment workloads.

All four assumptions are explicitly tested through ablations and stress scenarios.

## Topology Thesis: 1M/10M Micro-Masters -> Elastic Virtual Macro-Model

The topology claim is separated into three testable components:

1. Coverage: specialists jointly span the task manifold.
2. Coordination: graph communication preserves global coherence.
3. Arbitration: verifier/coordinator loops resolve local conflicts.

Detailed equations and build targets: [docs/TOPOLOGY_100M_TO_10B.md](docs/TOPOLOGY_100M_TO_10B.md).
Hybrid runtime policy and supply-chain rules: [docs/RUNTIME_INTEGRATION_DOCTRINE.md](docs/RUNTIME_INTEGRATION_DOCTRINE.md).

## Experimental Methodology

### Benchmarks

Evaluate on mixed workloads:

1. Deterministic orchestration tasks.
2. Multi-step coding and reasoning workflows.
3. Adversarial or noisy routing conditions.

### Baselines

1. Single-specialist baselines (1M and 10M classes).
2. Non-convolution multi-agent ablation.
3. Dense large-model reference (10B class or closest practical proxy).

### Core Metrics

1. Quality: success rate, correctness, completeness.
2. Efficiency: cost per successful task.
3. Latency: p50/p95 orchestration latency.
4. Coordination health: hop count, conflict rate, utilization entropy.
5. Reliability: recovery rate after first-pass failure.

### Statistical Rigor

1. Report mean and confidence intervals over repeated runs.
2. Use paired comparisons where tasks are identical across systems.
3. Separate exploratory findings from confirmatory results.
4. Publish ablations for routing, world-model consistency, and convolution depth.

## Falsifiability Criteria

The core thesis is considered unsupported if any of the following hold after controlled evaluation:

1. Quality parity remains materially below target despite tuning.
2. Coordination overhead dominates latency budget.
3. Specialist collapse persists (routing concentrates in few experts).
4. Gains disappear outside synthetic benchmarks.

## Current Implementation Status

Implemented core (minimal executable scaffold):

1. Scout task discovery and simple routing: [asmodeus/scout.py](asmodeus/scout.py)
2. Worker execution with bounded retry recovery: [asmodeus/worker.py](asmodeus/worker.py)
3. Task abstraction with transient-failure simulation: [asmodeus/task.py](asmodeus/task.py)
4. Assignment and orchestration manager: [asmodeus/task_manager.py](asmodeus/task_manager.py)
5. Unit tests for end-to-end execution and failure handling: [tests/test_core.py](tests/test_core.py)
6. Skill-aware routing and hierarchical subagent execution for specialized tasks

## Skill-Based Subagents (How To Assign and Complete Tasks)

Asmodeus now supports explicit task skills and parent-child worker hierarchies.

```python
from asmodeus.scout import ScoutAgent
from asmodeus.task import Task
from asmodeus.task_manager import TaskManager
from asmodeus.worker import WorkerAgent

scout = ScoutAgent(name="Scout-Alpha")
manager = TaskManager()

# 1) Register a parent worker
worker = WorkerAgent(name="Worker-Node-1", skills=["general"])
manager.register_worker(worker)

# 2) Create a specialized subagent under that worker
manager.create_subagent(
	parent_worker_name="Worker-Node-1",
	subagent_name="Worker-Node-1-Network",
	skills=["networking", "recon"],
)

# 3) Define a task that requires a skill
task = Task(name="Network Recon", required_skills=["networking"])

# 4) Route and execute
assigned = scout.route_task_to_agents(task, manager.get_available_workers())
results = manager.execute_all({task: assigned})

print(results["Network Recon"])  # True
```

Behavior summary:

1. Tasks may declare `required_skills`.
2. Workers and subagents may declare `skills`.
3. The scout routes skill-required tasks to matching agents.
4. The manager executes assigned agents in order, allowing fallback attempts.

Planned scientific docs:

1. Goals and measurable targets: [docs/GOALS.md](docs/GOALS.md)
2. Topology spec: [docs/TOPOLOGY_100M_TO_10B.md](docs/TOPOLOGY_100M_TO_10B.md)
3. Runtime integration doctrine: [docs/RUNTIME_INTEGRATION_DOCTRINE.md](docs/RUNTIME_INTEGRATION_DOCTRINE.md)
4. Additional architecture/theory planning artifacts in [docs](docs)

## Reproducibility Quickstart

### Run tests

```powershell
python -m unittest discover -s tests -v
```

Expected current result:

```
Ran 3 tests in ~0.001s
OK
```

## Repository Map

```
Asmodeus/
	asmodeus/
		scout.py
		worker.py
		task.py
		task_manager.py
	tests/
		test_core.py
	docs/
		GOALS.md
		TOPOLOGY_100M_TO_10B.md
		OVERVIEW.md
		THEORY.md
		SYSTEM_ARCHITECTURE.md
		EVALUATION_PLAN.md
		IMPLEMENTATION_PLAN.md
		TASKS.md
	README.md
```

## Roadmap Alignment

Milestones and measurable targets are tracked in [docs/GOALS.md](docs/GOALS.md). The immediate focus is:

1. Formalize heterogeneous 1M/10M specialist topology and capability registry.
2. Implement Lambda Azure + Epsilon-Hollow hybrid runtime adapters.
3. Implement internet SLM registry, downloader, and verifier.
4. Sparse top-k routing with model-cost-aware gating and utilization telemetry.
5. Swarm-convolution integration and ablation validation.
6. Baseline comparison against OpenClaw-like pipelines.

## Safety and Governance Notes

Asmodeus is a research system. Any deployment-facing use should include:

1. Policy-constrained action boundaries.
2. Audit logging for routing and arbitration decisions.
3. Rollback-safe execution paths for irreversible actions.
4. Human override for high-impact operations.

## Citation

If you use this project in reports or experiments, cite the repository and include commit hashes for exact reproducibility.
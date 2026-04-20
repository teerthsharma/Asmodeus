# Asmodeus Master Scientific Specification

## 1. Scope and Intent

This document is the canonical technical specification for Asmodeus.

The goal is to define a falsifiable, reproducible, and mathematically explicit agent-swarm runtime where many small specialists are coordinated under strict runtime and model-admission constraints.

Primary design target:

- Achieve strong task quality at low active compute by routing to a sparse specialist subset.
- Keep the system measurable and auditable under production constraints.

This specification is normative for architecture, math, metrics, and validation.

## 2. System Model

Let:

- $M = \{m_1, \dots, m_N\}$ be the registered specialist set.
- $P_i$ be parameter count of specialist $m_i$.
- $c_i \in \mathbb{R}^d$ be a specialist capability vector.
- $x_t$ be task/query context at step $t$.
- $w_t$ be world-model state at step $t$.

Static budget:

$$
P_{total} = \sum_{i=1}^{N} P_i
$$

Active subset and active budget:

$$
S_t \subseteq M, \quad |S_t| = k_t, \quad P_{active}(t)=\sum_{i\in S_t} P_i
$$

Constraint:

$$
P_{active}(t) \le P_{budget}
$$

where the current default budget in code is $P_{budget}=120\,\text{M}$ parameters.

## 3. Model Admission and Supply Integrity

Specialists are routable only after download, integrity verification, and compatibility checks.

Define the admission set:

$$
M_{ready}(t) = \{m \in M_{internet} \mid D_m(t) \land V_m \land C_m\}
$$

where:

- $D_m(t)$: model artifact materialized.
- $V_m$: integrity/provenance verification succeeded.
- $C_m$: runtime compatibility check succeeded.

Activation rule:

$$
S_t \subseteq M_{ready}(t)
$$

Operational implication:

- No unverified model may be activated in production mode.
- Downgrade paths must remain inside $M_{ready}(t)$.

## 4. Routing Formulation

Baseline routing score:

$$
r_i^t = \beta_1\,\mathrm{sim}(q_t, c_i) + \beta_2\,rel_i - \beta_3\,cost_i + \beta_4\,fresh_i
$$

Readiness-aware penalty:

$$
\tilde{r}_i^t = r_i^t - \beta_5\,\mathbf{1}[m_i \notin M_{ready}(t)]
$$

Selection:

$$
S_t = \mathrm{topk}(\tilde{r}^t, k_t)
$$

with budget-aware truncation if needed.

Router objective (conceptual):

$$
\mathcal{L}_{route} = \mathcal{L}_{task} + \lambda_{cost}\,\frac{P_{active}(t)}{P_{budget}} + \lambda_{bal}\,\mathrm{KL}(U\|U^*)
$$

where $U$ is empirical specialist utilization and $U^*$ is target utilization.

## 5. Cluster Topology and Communication

Asmodeus uses clustered specialists with sparse cross-cluster links.

Adjacency update (pheromone-style reinforcement):

$$
A_{ij}^{t+1} = (1-\rho)A_{ij}^{t} + \rho\,reward_{ij}^{t}
$$

with $\rho \in (0,1)$ controlling update rate.

Message passing for specialist state $h_i^t$:

$$
m_i^t = \sum_{j \in \mathcal{N}(i)} A_{ij}^t W_m h_j^t
$$

$$
h_i^{t+1} = \sigma\left(W_s h_i^t + m_i^t + B w_t\right)
$$

## 6. Swarm Convolution Aggregation

For kernels $K_l$, aggregated channels are:

$$
c_l^t = \sum_{i=1}^{N} K_l(i)\,h_i^t
$$

Global representation:

$$
z_t = \mathrm{concat}(c_1^t, \dots, c_L^t)
$$

$z_t$ is the runtime fused state used by downstream decision and generation logic.

## 7. World Model and Arbitration

World model requirements:

- Task graph and dependency state.
- Specialist outputs plus confidence/provenance.
- Conflict markers and arbitration outcomes.
- TTL and staleness metadata.

Consistency metric:

$$
W_{consistency} = 1 - contradiction\_rate
$$

Target behavior:

- High $W_{consistency}$ under asynchronous updates.
- Bounded contradiction growth with confidence-weighted merge rules.

## 8. Runtime Algorithm per Query

Given query $q_t$:

1. Infer required skills/tags from $q_t$.
2. Route to ranked specialists with readiness and budget constraints.
3. Execute active specialists and collect outputs/states.
4. If failures occur, run bounded recovery over backup specialists.
5. Apply swarm convolution to active states.
6. Call inference engine with telemetry and aggregated state.
7. Record telemetry and update world model.

Pseudo-notation:

$$
(q_t, w_t) \xrightarrow[]{route} S_t \xrightarrow[]{execute} H_t \xrightarrow[]{aggregate} z_t \xrightarrow[]{infer} y_t
$$

## 9. Complexity and Scaling

Let:

- $N$: registered specialists.
- $k$: active specialists ($k \ll N$).
- $d$: latent/state dimension.
- $\bar{d}_{graph}$: average active graph degree.

Approximate per-step costs:

- Routing scan: $\mathcal{O}(N d)$.
- Message passing over active graph: $\mathcal{O}(k\bar{d}_{graph} d)$.
- Swarm convolution: $\mathcal{O}(kLd)$.
- Language generation: dominant model-forward cost, bounded by selected inference tier.

Operational objective:

- Increase quality by larger effective capacity ($P_{total}$) while controlling active cost via $k$ and $P_{active}(t)$.

## 10. Reliability and Degraded Mode

Required runtime behavior under specialist unavailability:

1. Detect missing/unavailable target specialists.
2. Reroute to verified backup specialists.
3. Preserve continuity with explicit downgrade annotation.
4. Emit telemetry with root-cause and fallback path.

Continuity SLO target from doctrine:

- Degraded reroute latency less than or equal to 2.0 seconds.

## 11. Evaluation Hypotheses and Tests

Primary hypotheses:

- H1: Sparse routed swarm improves cost-adjusted quality over baseline orchestration.
- H2: Swarm convolution improves output quality over non-convolution ablations.
- H3: Admission-gated model supply improves reliability versus ungated loads.

Minimum benchmark bundle:

1. Quality tasks: coding, analysis, mathematical reasoning, multi-step planning.
2. Cost metrics: latency, active parameters, GPU memory profile.
3. Reliability metrics: first-pass success, recovery success, contradiction rate.

Core reported metrics:

- $Q_{virtual}/Q_{dense}$ parity proxy.
- Cost per successful task.
- p95 latency.
- Utilization entropy.
- Recovery rate after induced failures.

## 12. Reproducibility Protocol

Required protocol for scientific claims:

1. Fix code revision and dependency lock.
2. Fix benchmark set and scoring scripts.
3. Fix seeds and decode settings.
4. Record hardware profile and runtime mode.
5. Publish raw run logs and aggregate metrics.

Result integrity checklist:

- No hidden prompt/model substitutions.
- No mixed-setup comparisons.
- No unverified specialist activations in production-mode runs.

## 13. Mapping to Repository Modules

Core orchestration:

- asmodeus/scout.py
- asmodeus/task.py
- asmodeus/worker.py
- asmodeus/task_manager.py

Registry and routing:

- asmodeus/registry.py
- asmodeus/router.py
- asmodeus/runtime.py

Acquisition and integrity:

- asmodeus/downloader.py

Topology and state:

- asmodeus/cluster_topology.py
- asmodeus/swarm_convolution.py
- asmodeus/world_model.py

Runtime bridge:

- asmodeus/hybrid_adapter.py
- asmodeus/true_inference.py

CLI and launcher:

- cli.py
- talk_to_asmodeous.cmd

## 14. Known Gaps and Near-Term Work

Near-term high-value work (technically feasible now):

1. Add deterministic benchmark harness for coding and math tasks with fixed seeds.
2. Introduce structured telemetry schema for routing and fallback causality.
3. Add reproducible ablation scripts (router-only vs router+convolution).
4. Add CI checks for admission-gate invariants.
5. Add artifact manifest for specialist provenance and compatibility metadata.

## 15. Definition of Scientific Completion

A claim of "macro-model-equivalent swarm behavior" is accepted only if all conditions hold:

1. Reproducible benchmark package is published.
2. Cost-adjusted quality superiority over baseline is demonstrated.
3. Failure recovery and degraded-mode SLOs are met.
4. Admission and provenance policy compliance is 100 percent in production-mode runs.
5. Independent reruns on controlled hardware reproduce the same directional results.

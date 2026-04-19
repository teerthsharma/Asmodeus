# Topology: 1M/10M Micro-Masters -> Elastic Virtual Macro-Model

This document expands the core Asmodeus thesis:

Many highly specialized small models (masters) can compose into a swarm with macro-model behavior, while keeping runtime compute sparse and controllable.

This is explicitly a model-centric topology: the primitive unit is a specialist model node, not a data shard.

## 1) Parameter Budget and Core Claim

Let:

- N = number of specialist models
- P_i = parameters of specialist i

Then:

P_total = sum_{i=1..N} P_i

For micro-master topology:

- P_i is typically 1M or 10M
- N is selected by capability coverage and budget envelope

This is the static topology budget. Runtime cost is determined by sparse activation.

## 2) Micro-Master Tiers

Tier A: Nano masters (1M)
- Extremely narrow skills
- Fast, low-cost atomic transforms

Tier B: Micro masters (10M)
- Wider context integration
- Cross-atomic synthesis and arbitration support

Tier C: Optional coordinator masters
- May also remain in 10M class or slightly above, depending on deployment constraints

The design intent is many narrow experts first, then sparse composition.

## 2.5) Runtime and Acquisition Constraints

This topology is valid only when all specialists operate under two runtime logics and one acquisition logic:

1. Lambda Azure logic
- sparse active-set execution
- streaming model materialization
- quantization-aware preparation
- memory-budgeted paging and scheduling

2. Epsilon-Hollow logic
- orchestrator-led control modes
- tiered planner/gate/arbitrator behavior
- bootstrap, health-check, and fallback discipline

3. Internet SLM acquisition logic
- specialist models are sourced from online registries
- each model is pinned, downloaded, verified, and then admitted

Admission set:

M_ready(t) = {m in M_internet | downloaded(m,t) and verified(m) and compatible(m)}

Only specialists in M_ready(t) are routable.

## 3) Sparse Activation Economics

Each decision step activates only top-k specialists.

P_active = sum_{i in S_t} P_i

Typical operating points:

- Fast path: mostly 1M masters (for example, 8 to 20 active)
- Standard path: mixed 1M + 10M active set
- Hard path: uncertainty-triggered escalation adds more 10M masters

This gives a high-capacity system with far lower active compute than a dense macro-model forward path.

## 4) Swarm Graph Topology

Use a two-level graph inspired by bees (local hive work) and ants (long-range trail routing):

1. Intra-cluster graph (dense)
- Clusters are organized by capability domain.
- Strong local connectivity for fast role collaboration.

2. Inter-cluster graph (sparse)
- Limited cross-cluster links for global coordination.
- Links are weighted by pheromone-style success signals.

Adjacency matrix A_t evolves with reinforcement:

A_{ij}^{t+1} = (1 - rho) * A_{ij}^t + rho * reward_{ij}^t

where rho controls trail update rate.

## 5) Specialist State and Convolution

Specialist i has latent state h_i^t.

Local message:

m_i^t = sum_{j in N(i)} A_{ij}^t * W_m * h_j^t

State update:

h_i^{t+1} = sigma(W_s * h_i^t + m_i^t + B * w_t)

Swarm-convolution kernel l:

c_l^t = sum_{i=1..N} K_l(i) * h_i^t

Global swarm representation:

z_t = concat(c_1^t, ..., c_L^t)

## 6) Router and Activation Rule

Router scores specialists with context x_t and world state w_t:

r_i^t = g(h_i^t, x_t, w_t)

Active set:

S_t = topk(r^t, k)

Constraint:

S_t subseteq M_ready(t)

Routing policy minimizes quality loss and compute cost:

L_route = L_task + lambda_cost * k + lambda_balance * KL(U || U_target)

where U is specialist utilization distribution.

Model-centric routing score can incorporate specialist mastery fit:

r_i^t = beta_1 * sim(q_t, c_i) + beta_2 * rel_i - beta_3 * cost_i + beta_4 * fresh_i

where:

- c_i is specialist capability vector
- rel_i is historical reliability
- cost_i is per-call compute/token cost
- fresh_i is world-state freshness alignment

Runtime-policy-aware scoring can additionally penalize unavailable or unverified models:

r_i^t <- r_i^t - beta_5 * unavailable_i - beta_6 * unverified_i

## 7) Effective Capacity Condition

The swarm approaches dense-macro behavior when three conditions hold:

1. Coverage: specialists span required task manifolds.
2. Coordination: message passing and world model keep plans coherent.
3. Arbitration: conflicts are resolved with verifier or coordinator loops.

A practical proxy:

Q_virtual ~= Q_dense(P_ref)

subject to:

- H_route <= H_max
- Drift_world <= D_max
- Conflict_rate <= R_max

## 8) World Model Requirements

The shared world model must store:

- Task graph and dependencies
- Specialist outputs with confidence and provenance
- Conflict markers and arbitration outcomes
- Time-to-live and staleness metadata

Consistency objective:

W_consistency = 1 - contradiction_rate

Target is high consistency under asynchronous updates.

## 9) Training and Optimization Path

Phase A: Specialist shaping
- Train or fine-tune 1M and 10M masters for narrow capability slices.

Phase B: Router training
- Train top-k router with cost-aware objective and load balancing.

Phase C: Swarm-conv training
- Train aggregation and coordination heads on multi-step tasks.

Phase D: Distillation (optional)
- Distill virtual swarm behavior into a single dense checkpoint when monolithic inference is required.

## 10) Metrics That Validate the Thesis

1. Quality parity
- Q_virtual / Q_dense(P_ref)

2. Efficiency
- Cost_per_success_virtual / Cost_per_success_dense

3. Routing health
- Utilization entropy and specialist collapse rate

4. Coordination health
- Mean hop count, conflict rate, and recovery rate

5. Stability
- Output variance under repeated runs with controlled randomness

## 11) Failure Modes and Defenses

1. Specialist collapse
- Symptom: too many decisions routed to a few experts.
- Defense: entropy regularization and per-expert caps.

2. Over-coordination
- Symptom: communication dominates latency.
- Defense: sparse inter-cluster edges and bounded hops.

3. World drift
- Symptom: contradictory state causes planner loops.
- Defense: TTL, confidence decay, and verifier arbitration.

4. Router brittleness
- Symptom: wrong experts selected under distribution shift.
- Defense: uncertainty-aware escalation from k=4 to k=8.

## 12) Build Targets for the First Implementation

1. Registry of heterogeneous specialist endpoints (1M and 10M) with capability tags.
2. Dynamic top-k router with telemetry.
3. Two-level graph (cluster + cross-cluster trails).
4. Swarm-convolution aggregator for final decision state.
5. World model with versioned state and contradiction checks.
6. Benchmark harness against OpenClaw-style baseline.
7. Hybrid runtime adapter that enforces Lambda Azure + Epsilon-Hollow execution logic.
8. Internet SLM downloader and verifier with cache-aware fallback behavior.

When these eight targets are met with the success metrics above, the micro-master topology claim is testable and falsifiable.

# Runtime Integration Doctrine

This document defines non-optional architecture rules for Asmodeus runtime behavior.

All specialist execution must follow:

1. Lambda Azure style execution logic.
2. Epsilon-Hollow style orchestration logic.
3. Internet-downloaded SLM supply chain policy.

## 1) Lambda Azure Style Execution Logic

Required behaviors:

1. Sparse active-set inference over specialist pools.
2. Streaming model ingestion/materialization for large artifacts.
3. Quantization-aware preparation paths when needed for constrained hardware.
4. Demand-paged or memory-budgeted model residency policy.
5. Cost-aware routing objective that penalizes expensive activations.

Operational interpretation:

- Keep active compute small relative to registered capacity.
- Prefer staged activation escalation over unconditional full activation.

## 2) Epsilon-Hollow Style Orchestration Logic

Required behaviors:

1. Orchestrator-first control (agent mode, IDE mode, full mode semantics).
2. Tiered decision hierarchy (planner/gate/arbitrator style separation).
3. Bootstrap-first startup (dependencies, environment, model prep).
4. Health checks and explicit recovery/fallback pathways.
5. Retry discipline for transient failures with bounded attempts.

Operational interpretation:

- Routing decisions are not raw model calls; they are orchestrated actions with policy context.

## 3) Internet SLM Supply Chain Policy

All specialist models are internet-sourced SLMs unless an explicit local override policy is declared.

Each registered specialist must include:

1. Source registry identifier.
2. Repository or artifact ID.
3. Revision pin or immutable digest.
4. Integrity metadata (checksum/hash).
5. License/provenance metadata.
6. Runtime compatibility metadata (format, quantization, device constraints).

Admission rule:

M_ready(t) = {m in M_internet | downloaded(m,t) and verified(m) and compatible(m)}

Routing rule:

S_t subseteq M_ready(t)

No model may be activated before admission.

## 4) Download and Verification Pipeline

1. Discover candidate model artifacts from configured registries.
2. Download with retry/backoff and parallel worker controls.
3. Validate integrity and provenance.
4. Validate compatibility against runtime policy.
5. Register into specialist catalog and expose to router.
6. Emit telemetry (source, revision, hash, latency, cache-hit).

## 5) Degraded Mode and Continuity

If target specialists are unavailable:

1. Degrade to verified lower-capability specialists.
2. Maintain response continuity with explicit downgrade annotation.
3. Trigger asynchronous recovery or re-fetch.
4. Preserve audit trail of fallback cause and route.

## 6) Compliance Metrics

Minimum acceptance metrics:

1. 100 percent of model activations satisfy admission rule.
2. 0 unverified model activations in production mode.
3. >= 95 percent bootstrap success under controlled clean-machine runs.
4. >= 80 percent warm-start cache hit rate for recurring specialists.
5. <= 2.0 seconds degraded-mode reroute on model unavailability.

## 7) Relationship to Other Documents

1. Topology math and routing: [docs/TOPOLOGY_100M_TO_10B.md](docs/TOPOLOGY_100M_TO_10B.md)
2. Program-level targets and milestones: [docs/GOALS.md](docs/GOALS.md)
3. Project-level scientific specification: [README.md](../README.md)

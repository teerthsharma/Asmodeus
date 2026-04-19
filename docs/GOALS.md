# Asmodeus Goals

This document defines measurable goals for building a swarm-native, CNN-style agentic system that surpasses OpenClaw-like baselines in quality, speed, and efficiency.

## North Star

Create a production-credible "CNN of agents" where specialized agents coordinate like bee and ant colonies, using a shared world model and elastic runtime, with verified gains over baseline systems.

## Objective Stack

1. Theory Completeness
- Formalize the swarm computation graph and training objectives.
- Publish architecture assumptions and failure modes.

2. Runtime Power
- Integrate custom runtime capabilities and Lambda Azure scale patterns.
- Support dynamic agent population sizing during execution.

3. Swarm Intelligence
- Implement role-specialized agents with pheromone-style routing signals.
- Enable cross-agent memory with conflict-aware merging.

4. Benchmark Superiority
- Exceed OpenClaw baseline performance on selected tasks.
- Maintain lower or equal quality-adjusted compute cost.

5. Runtime Convergence and Supply Integrity
- Standardize all execution paths on Lambda Azure + Epsilon-Hollow hybrid logic.
- Ensure specialist models are internet-downloaded SLMs with verification gates.

## Quantitative Targets (Phase 1 to Phase 3)

1. Quality
- +20 percent improvement on task success rate versus baseline in the initial benchmark suite.

2. Latency
- p95 orchestration latency under 1.5 seconds for medium-complexity tasks.

3. Cost Efficiency
- At least 30 percent lower cost per successful task than baseline at equal quality threshold.

4. Reliability
- 99 percent successful completion for non-adversarial task workflows.

5. Recovery
- Automatic retry and reroute recovers at least 85 percent of first-pass failures.

## Topology-Specific Targets (1M/10M Micro-Masters)

1. Parameter Topology
- Build and register heterogeneous specialist models in 1M and 10M bands.
- Maintain a tracked swarm budget via P_total = sum(P_i), with explicit per-band accounting.

2. Sparse Activation Efficiency
- Default top-k routing set to k = 8 to 24 depending on task complexity and uncertainty.
- Keep average active parameter budget <= 120M per decision across benchmark runs.

3. Specialist Utilization
- At least 80 percent of registered specialists must receive non-trivial routing traffic (>0.2 percent) over benchmark suites.
- No single specialist should exceed 10 percent long-horizon routing share unless explicitly pinned by policy.

4. Coordination Overhead
- Inter-agent communication overhead must remain <= 20 percent of end-to-end latency budget.
- Mean coordination hop count <= 2.0 for medium-complexity workflows.

5. Virtual-to-Dense Parity
- Virtual swarm quality reaches >= 95 percent of the target dense reference model on core benchmark tasks.
- Optional distilled single-model checkpoint reaches >= 90 percent of virtual swarm quality.

## Integration Targets (Lambda Azure + Epsilon-Hollow + Internet SLM)

1. Runtime Path Compliance
- 100 percent of inference/orchestration paths run through the hybrid runtime adapter.
- No direct bypass path to unmanaged execution in production mode.

2. Internet SLM Acquisition Compliance
- 100 percent of registered specialists include source registry, revision pin, and integrity metadata.
- 0 unverified model loads are allowed in normal operation.

3. Bootstrap Reliability
- Clean-machine bootstrap succeeds in >= 95 percent of controlled runs with retry/backoff enabled.
- Parallel download worker tuning is exposed as runtime configuration.

4. Cache and Redownload Efficiency
- Warm-start runs reuse local cached artifacts in >= 80 percent of model loads.
- Forced redownload remains available for corruption recovery.

5. Degraded-Mode Continuity
- If a model fetch fails, router degrades gracefully to available specialists within <= 2.0 seconds.
- Recovery path records provenance and fallback reason in telemetry.

## Milestones

### M0: Theory and Design Freeze
Timeline: Week 1-2

Deliverables:
- Final math spec for swarm convolution and routing dynamics
- Topology spec for 1M/10M heterogeneous specialist layout
- Role taxonomy (scout, worker, verifier, coordinator)
- Evaluation protocol and baseline selection

Exit Criteria:
- All equations and assumptions reviewed
- Benchmark definitions approved

### M1: Orchestration MVP
Timeline: Week 3-5

Deliverables:
- Multi-agent task graph executor
- Basic world model state store
- Heuristic routing with role-based dispatch
- Top-k sparse routing with model-cost-aware gating and expert utilization telemetry
- Hybrid runtime adapter implementing Lambda Azure + Epsilon-Hollow control logic
- Internet SLM registry with download and verification hooks

Exit Criteria:
- End-to-end run across 3 representative workflows
- Deterministic replay for debugging

### M2: Swarm-CNN Layer Integration
Timeline: Week 6-8

Deliverables:
- Graph-convolution aggregation across active agents
- Confidence-weighted merge and conflict resolution
- Runtime autoscaling hooks for burst workloads
- Hierarchical cluster topology (intra-cluster dense, inter-cluster sparse)
- Streaming/quantization-aware model preparation and demand-paged loading path

Exit Criteria:
- Demonstrated improvement over non-convolution ablation
- Stable behavior under load test

### M3: Benchmark and Hardening
Timeline: Week 9-12

Deliverables:
- OpenClaw comparison report
- Cost and latency optimization pass
- Safety and fallback policy implementation
- Supply-chain integrity and provenance audit report for internet-downloaded SLMs

Exit Criteria:
- Target gains met on quality and cost metrics
- Reproducible benchmark package released

## First Task Backlog

1. Define benchmark tasks and scoring rubric.
2. Build baseline runner for OpenClaw-equivalent pipelines.
3. Define heterogeneous 1M/10M specialist registry and capability map.
4. Implement swarm role schema and task packet format.
5. Implement world model state contract and versioning.
6. Implement pheromone-weighted top-k routing heuristic.
7. Add convolution aggregation prototype for agent states.
8. Add evaluator agent for automatic output grading.
9. Add observability: traces, routing entropy, token usage, and latency histograms.
10. Implement internet SLM downloader with retries, parallel workers, and cache reuse.
11. Implement model verification gates (checksum, provenance, policy).
12. Implement runtime degraded-mode fallback when selected specialists are unavailable.

## Risk Register (Initial)

1. Over-coordination overhead
- Risk: too many inter-agent messages increase latency.
- Mitigation: cap neighborhood size and use sparse routing.

2. World model drift
- Risk: stale shared state causes contradictory plans.
- Mitigation: TTL policy, confidence decay, and merge arbitration.

3. Cost blow-up under scaling
- Risk: dynamic scaling exceeds budget.
- Mitigation: budget-aware scheduler and early-stop criteria.

4. Benchmark mismatch
- Risk: gains do not transfer to real user workflows.
- Mitigation: include both synthetic and real workflow tasks.

## Definition of Done

The initial program is complete when:

1. Theory, architecture, and goals are documented and versioned.
2. The swarm orchestration MVP runs end to end.
3. Swarm-CNN aggregation shows measurable lift in ablation tests.
4. OpenClaw baseline comparison is reproducible and favorable.
5. Reliability and latency targets are met in repeated runs.

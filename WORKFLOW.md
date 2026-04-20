# Workflow

## Objective

Develop and validate a specialist-swarm runtime that delivers strong quality under bounded active compute with reproducible evidence.

Core target:

$$
Q_{virtual} \approx Q_{dense}(P_{ref}) \quad \text{while} \quad P_{active}(t) \ll P_{total}
$$

## Current Engineering State

1. Orchestration core is implemented.
2. Registry and sparse routing are implemented.
3. Cluster topology, world model, and swarm convolution are implemented.
4. Runtime adapter and CLI execution path are available.
5. Scientific benchmark package and reproducibility reporting remain incomplete.

## Execution Workflow per Iteration

1. Define hypothesis and metric impact.
2. Implement minimal code change.
3. Add or update tests.
4. Run deterministic benchmarks.
5. Compare against baseline and ablations.
6. Keep or revert change based on evidence.

## Near-Term Milestones

1. Benchmark harness completion
- deterministic datasets, seeds, and score aggregation.

2. Admission and provenance hardening
- complete artifact manifest and activation gate checks.

3. Degraded-mode observability
- structured fallback telemetry and reroute SLO validation.

4. Release reproducibility bundle
- environment matrix, scripts, and archived results.

## Practical Next Steps

1. Add deterministic coding/math benchmark runner.
2. Add ablation scripts for routing and fusion behavior.
3. Add CI checks for docs and runtime invariant compliance.

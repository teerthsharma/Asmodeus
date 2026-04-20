# Tasks

This backlog contains concrete, near-term engineering tasks with measurable acceptance checks.

## Priority A: Integrity and Runtime Safety

1. Add specialist artifact manifest schema
- Scope: include source, revision pin, checksum, license, and compatibility fields.
- Acceptance: manifest validation fails if any required field is missing.

2. Enforce admission gates in all activation paths
- Scope: block specialist activation unless status is ready.
- Acceptance: tests confirm zero bypass paths.

3. Add structured fallback telemetry
- Scope: capture fallback trigger, failed specialist, chosen backup, and reroute latency.
- Acceptance: every degraded response includes a complete fallback event record.

## Priority B: Scientific Benchmarking

1. Build deterministic benchmark suite
- Scope: coding, analysis, and math task packs.
- Acceptance: reruns with same seed produce equivalent aggregate scores.

2. Add ablation runner
- Scope: no-fusion vs fusion, fixed-k vs adaptive-k.
- Acceptance: one command emits a reproducible comparison report.

3. Add confidence and contradiction metrics for world model
- Scope: world consistency and conflict rate over time.
- Acceptance: metrics exported for every benchmark run.

## Priority C: Performance and Lean Operation

1. Add active-budget guardrails in router telemetry
- Scope: per-query budget usage and over-budget prevention signals.
- Acceptance: no successful query exceeds configured active budget.

2. Reduce startup and dependency fragility
- Scope: launcher checks separated by risk class; avoid unsafe automatic torch replacement.
- Acceptance: missing non-torch dependency installs do not alter torch build.

3. Introduce docs quality gate
- Scope: markdown lint and broken-link checks on docs changes.
- Acceptance: CI fails on broken docs links or malformed markdown equations.
# Evaluation Plan

Objective:

- Quantify whether the specialist-swarm runtime delivers better quality-cost behavior than baseline orchestration.

Reference:

- docs/MASTER_SCIENTIFIC_SPECIFICATION.md

## 1. Experimental Design

1. Use fixed benchmark suites across coding, analysis, and mathematical reasoning.
2. Keep prompts, seeds, and decoding settings identical across variants.
3. Report full hardware and runtime configuration for every run.

## 2. Primary Metrics

1. Quality score ($Q$)
- Task-level rubric score normalized to $[0, 1]$.

2. Cost per success ($C_s$)

$$
C_s = \frac{\text{total compute cost}}{\text{successful tasks}}
$$

3. Active parameter pressure ($P_{active}$)

$$
P_{active}(t) = \sum_{i \in S_t} P_i
$$

4. Reliability and recovery
- First-pass success rate.
- Recovery success rate after induced transient failures.

5. Latency profile
- p50, p95, p99 end-to-end response latency.

## 3. Required Ablations

1. Router-only vs router+swarm-convolution.
2. Fixed-k vs adaptive-k routing.
3. Admission-gated routing vs unconstrained test mode (non-production).

## 4. Statistical Reporting

For each metric, report:

1. Mean and standard deviation.
2. Median and robust quantiles.
3. Relative lift against baseline.

Relative lift:

$$
\Delta = \frac{\text{metric}_{model} - \text{metric}_{baseline}}{\text{metric}_{baseline}}
$$

## 5. Pass Criteria

Minimum acceptance criteria for a release candidate:

1. Quality non-inferiority against baseline on all core task groups.
2. Lower or equal cost per success at matched quality threshold.
3. Degraded-mode reroute within continuity SLO.
4. Zero unverified model activations in production-mode runs.

## 6. Reproducibility Checklist

1. Fixed code revision.
2. Fixed dependency lock.
3. Fixed benchmark version.
4. Fixed hardware profile and runtime flags.
5. Published raw run logs and aggregation scripts.
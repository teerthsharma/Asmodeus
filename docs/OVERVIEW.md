# Overview

Asmodeus is a specialist-swarm orchestration runtime designed to maximize quality under bounded active compute.

The core operating idea is:

1. Keep a large registered specialist pool.
2. Activate only a small top-k subset per query.
3. Enforce strict model admission gates before routing.
4. Fuse specialist states through a convolution-style aggregator.

Primary runtime objective:

$$
	ext{maximize task quality} \quad \text{subject to} \quad P_{active}(t) \le P_{budget}
$$

## Canonical Spec

The normative technical document is:

- docs/MASTER_SCIENTIFIC_SPECIFICATION.md

## Supporting Docs

- docs/GOALS.md
- docs/TOPOLOGY_100M_TO_10B.md
- docs/RUNTIME_INTEGRATION_DOCTRINE.md

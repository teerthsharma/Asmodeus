# Theory

This file is a compact index of the formal claims used by Asmodeus.

For complete equations and constraints, see:

- docs/MASTER_SCIENTIFIC_SPECIFICATION.md

## Core Claims

1. Sparse activation economics
- A large specialist pool can be maintained while only a small subset is active per query.

$$
P_{active}(t) \ll P_{total}
$$

2. Admission-gated safety and integrity
- Route selection is constrained to verified and compatible specialists.

$$
S_t \subseteq M_{ready}(t)
$$

3. Message-passing coordination
- Specialist state updates are graph-coupled and reinforced over time.

$$
A_{ij}^{t+1} = (1-\rho)A_{ij}^{t} + \rho\,reward_{ij}^{t}
$$

4. Aggregation benefit hypothesis
- Swarm-convolution fusion of specialist states increases quality versus non-fused ablations.

## Falsifiability

The theory is only accepted if benchmarked claims are reproducible under fixed datasets, seeds, and hardware profiles.
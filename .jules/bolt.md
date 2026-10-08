## 2026-03-08 - Upper-triangular chunking for pairwise manifold metrics

**Learning:** Chunked distance matrix computation in `_pairs` evaluated chunk blocks against the entire array `s[None]`, recalculating `i >= j` pairs redundantly for every chunk and wasting ~50% of metric calculations. Slice chunking against `s[a:]` evaluates only upper-triangular sub-matrices, reducing computation time in `discontinuity_rate` and `fold_rate` by ~40-50%.

**Action:** When computing pairwise distance matrices over symmetric metric distances in chunked loops, slice target array from index `a` onwards (`sub_s = s[a:]`) to skip redundant lower-triangular pairs.

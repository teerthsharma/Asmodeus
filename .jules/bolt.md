## 2026-03-31 - Redundant Pairwise Distance Evaluation in Metric Chunking
**Learning:** In pairwise distance evaluation where only upper triangular pairs ($i < j$) are needed, chunking over rows without restricting the target columns ($s[a:]$) computes $N^2 / 2$ redundant lower-triangular distances that are immediately discarded by boolean masking.
**Action:** Always slice the target array to `s[a:]` when computing upper-triangular chunked distance matrices, shifting local column indices by `a`.

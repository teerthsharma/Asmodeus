## 2026-03-06 - Chunked Pairwise Metric Distance Filtering
**Learning:** In chunked pairwise distance operations filtering symmetric or ordered pairs (`i < j`), evaluating chunk rows against the full array computes redundant lower-triangle distances that are unconditionally discarded by `i < j`. Restricting columns to `s[a:]` cuts distance computations in half and provides ~1.89x speedup without extra memory allocation.
**Action:** Always slice column candidates to `s[a:]` when iterating row chunks `s[a:a+chunk]` where row-column ordering `i < j` is enforced.

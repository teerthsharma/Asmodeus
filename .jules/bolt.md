## 2026-03-06 - N-gram Language Model Probability Computation Optimization
**Learning:** Dense tensor allocations (`torch.zeros(V)`) and redundant counter summations inside inner loops during n-gram probability calculations (`NGramLM._p`) create massive overhead for sparse vocabulary updates.
**Action:** Lazy-cache base unigram distributions and context total sums, and directly mutate sparse counts in cloned distributions rather than instantiating zero tensors per n-gram order.

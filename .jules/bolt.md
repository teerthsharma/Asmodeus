## 2025-05-18 - Pre-convert task tags set in Top-K Router
**Learning:** In `SparseTopKRouter.route`, `_compute_similarity` was creating `set(task_tags)` twice per model in `M_ready(t)`. Pre-converting `task_tags` to a `set` once before the model evaluation loop and calculating Jaccard union length mathematically (`len(A) + len(B) - len(A & B)`) avoids redundant allocations and yields a ~2.2x speedup in top-k model routing.
**Action:** When evaluating set similarity across a list of items, always hoist query set construction out of the loop and compute union cardinality without allocating union sets.

## 2025-05-19 - Avoid negative fast paths in skill normalization
**Learning:** Attempting a negative short-circuit fast-path in `Task.matches_agent_skills` (e.g. checking if set elements are clean/lowercase strings and returning `False` without coercing) assumes normalization won't map or transform canonical skill names (such as aliases or punctuation cleanup). Positive set subset fast-paths (`if issubset(agent_skills): return True`) are safe, whereas negative fast-paths risk false negatives.
**Action:** Use positive short-circuiting for set membership checks, and fall back to full coercion on misses rather than attempting negative short-circuits.

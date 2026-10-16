## 2025-02-18 - Analytical & Batched Linear Algebra for Quantum Instruments

**Learning:** Computing singular values or norms across $V$ outcome operators $K_v$ via individual Python loops (`[np.linalg.norm(k, 2)**2 for k in K]`) incurs massive overhead ($O(V)$ individual SVD calls in Python). For structured instruments such as $K_v = \text{diag}(\sqrt{M_{:,v}}) Q$, the maximum singular value squared is analytically $\max_i M_{i,v}$, reducing computation from $O(V \cdot N^3)$ to $O(N \cdot V)$ elementwise max (~330x speedup). For general array sequences, batched LAPACK operations (`np.linalg.svd(K_arr)`) avoid Python loop dispatch overhead.

**Action:** When evaluating sequence-based operator classes or quantum instrument outcomes, check for structural unitary invariants or utilize batched LAPACK matrix decompositions over 3D tensors rather than iterating in Python loops.

# Bolt Journal

## 2026-03-29 - Batched Matmuls vs einsum in Atlas Gauge Transformations
**Learning:** PyTorch `torch.einsum` on multi-dimensional tensor contractions (like chart gauge transformations `[n, c, N, d_c] x [n, N]`) incurs dispatch and kernel overhead compared to explicit batched matrix multiplication `(x.unsqueeze(1).unsqueeze(-2) @ Vc).squeeze(-2)`.
**Action:** Prefer explicit `@` matmul with `unsqueeze`/`squeeze` over `einsum` for tensor contractions in performance-critical PyTorch forward paths.

# PREREG DRAFT: T3 (lazy scaling) and T7 (systems)

Draft by Taub, 2026-10-04. Not committed. Merges into `PREREG.md` only after Inspector review; until that
commit lands, every number below is a cost-model or timing measurement, never a T3/T7 result (L-PREREG).

## 1. Bars, verbatim from PROMPT.md §8

- **T3 lazy scaling (C3):** m=2; K ∈ {16,64,256,1024}, product keys from K=256. Arms: dense JEPA equal
  active FLOPs; single JEPA with PEER experts. Bars: loss monotone in K, beats dense from K=256; lazy top-m
  within 1% rel of soft mixture over all K. Kill: flat in K, or PEER-in-one-JEPA matches.
- **T7 systems:** lazy routing + tiny JEPAs on 4060, µs/step vs dense JEPA of equal quality. Bar ≥1.5× faster.

The bars stay as written. Sections 2 and 3 only pin the words the bars leave open.

## 2. T3 operational pins

| Word in the bar | Pinned meaning |
|---|---|
| bed | `so3` (n=3, atlas=4, D=256 random lift, `beds.make_bed("so3", obs="lift")`): the largest n and chart count in T2. L-WALD neutral bed beside it: `r2`. |
| loss | held-out h=1 `npersist` from `metrics.pred_error_from` (MSE / MSE of copy-last), computed in each arm's own g. It is comparable across arms because it is normalised by the arm's own copy-last error. A constant latent scores ∞, and so does a latent that encodes only the trajectory id (`tests/test_metrics.py`). Secondary: `metrics.linear_probe` R² of state from g (state is used at evaluation only). |
| seeds | train 0, 1, 2. Every arm, the atlas included, is tuned on seed 3 (L-BR). Steps, optimizer, batch and T come from Foreman's `run.py` and are identical for every arm. |
| atlas config | `ObserverAtlas(D=256, A=a, K, m=2, d_c=3, D_g=8, d_h=64, d_key=32, hidden=32)`, `product_keys = K>=256`. |
| dense JEPA equal active FLOPs | `SingleJEPA(d=D_g)` with `hidden` bisected by `bench_t7.match_flops` to within ±5% of `atlas.active_flops()` at that K. That gives hidden 144–146 at m=2 (table §4). |
| PEER-JEPA | `PEERJEPA(d=D_g, heads=4, topk=min(8,√K), d_key=32)`: `hidden` is bisected to equal active FLOPs and **n_experts is the smallest square whose total params ≥ atlas total params** (L-BR: equal active FLOPs AND equal total params). With n_experts = K, PEER has 4–21× fewer params than the atlas (§4), which is not a fair arm. |
| monotone in K | the 3-seed mean `npersist` drops at every step 16→64→256→1024, and each drop is larger than the pooled seed SD of the two Ks it connects. |
| flat in K (kill) | relative change of the mean `npersist` from K=16 to K=1024 is under 2%. |
| beats dense from K=256 | at K=256 and at K=1024, the atlas has lower `npersist` than dense on 3/3 seed-paired runs. |
| lazy within 1% of soft | the same config is trained with `observe(mode="soft")` (full softmax over all K; costs K charts per step, see §4). The bar holds when \|L_lazy − L_soft\| / L_soft ≤ 0.01 on the 3-seed mean at every K. |
| PEER matches (kill) | PEER-JEPA `npersist` ≤ 1.01 × atlas `npersist` (3-seed means) at K=1024. |

Open flag for Inspector: at this config the GRU backbone (3·d_h·(D+d_h) = 61,440 MACs) is 75% of atlas
active FLOPs, and K·d_key routing adds up to 131k more at K=4096 with flat keys. "Equal active FLOPs" is
therefore dominated by the backbone, not the charts. The draft keeps the contract's definition, which
counts the backbone, and reports the chart-only MACs (m · 9,088) beside it.

## 3. T7 operational pins

**Dense JEPA of equal quality.**
- Bed and metric are T3's: `so3` lift, held-out h=1 `npersist`, 3 seeds, the same training budget.
- Width sweep: `SingleJEPA(d=D_g, hidden=H)` for H ∈ {64, 128, 256, 512, 1024, 2048, 4096}.
- H* is the smallest H whose 3-seed mean `npersist` ≤ the atlas reference (K=1024, m=2, product keys)
  AND whose `linear_probe` R² ≥ the atlas R² − 0.01.
- If no H ≤ 4096 qualifies, H* = 4096 and the ratio is reported as a lower bound.
- The `r2` neutral bed is run beside it and reported, but it does not decide the verdict.

**µs/step.**
- Measured with `bench_t7.py` at the PREREG commit, using the back-to-back CUDA-event protocol: 25 warmup
  steps, 200 timed, median + IQR.
- Each arm gets its best mode (min over eager and CUDA-graph replay). The atlas may use the parity-tested
  Triton chart kernel (`lazy_tri`).
- fp32, forward only. The step is one `observe` on obs [B, 1, 256].

**Batch sizes.**
- **Primary: B = 1.** One agent stepping is the regime that "computed only when observed" claims.
- B = 64 and B = 1024 are reported but do not decide the verdict.

**Bar.**
- The bar is met when median_dense(H*) / median_atlas ≥ 1.5 AND q1_dense > q3_atlas, at B = 1.
- The bench runs 3 times, non-concurrently with any other GPU lane, and the median of the 3 ratios
  decides.
- **Kill:** the ratio is < 1.5.

**Clocks.** The GPU is a laptop part and clocks cannot be locked without admin. The bench prints SM clock,
utilisation and temperature per batch size. A run where another process holds the GPU (`nvidia-smi
--query-compute-apps`) is void and rerun.

## 4. Cost model vs measurement

**Provenance.**
- Hardware and software: `bench_t7.py` (uncommitted, branch atlas-zero), RTX 4060 Laptop 8 GB, torch
  2.14.0+cu126, triton-windows 3.7.1.
- Run conditions: one sweep on 2026-10-04 with CPU at 100% from other lanes. SM clock was 1455–2700 MHz.
- Recorded config: `D=256, A=2, d_c=2, D_g=8, d_h=64, d_key=32, hidden=32`.
- All cells below are graph-mode medians in µs (200 steps).
- These are cost measurements, not T7 results.

### 4.1 Routing: O(K·d) flat keys vs O(√K·d) product keys

The model, in MACs per query (d_key = 32), not counting the shared query projection d_h·d_key = 2,048:
- flat: K·d_key MACs, plus top-m over K, reading K·d_key·4 B of keys
- product keys: √K·d_key MACs, plus 2 top-m over √K, plus top-m over m², reading √K·d_key·4 B of keys

| K | flat MACs | pk MACs | flat B=1 | pk B=1 | flat B=1024 | pk B=1024 |
|---|---|---|---|---|---|---|
| 16 | 512 | 128 | 61.6 | 73.7 | 61.2 | 96.3 |
| 64 | 2,048 | 256 | 68.1 | 68.6 | 60.4 | 89.1 |
| 256 | 8,192 | 512 | 58.6 | 70.8 | 90.1 | 69.6 |
| 1024 | 32,768 | 1,024 | 107.5 | 71.5 | 180.2 | 94.7 |
| 4096 | 131,072 | 2,048 | 58.7 | 73.7 | 589.3 | 106.5 |

**Reading.**
- Up to K=256 at B=1024, both paths sit on the ~60–75 µs graph-launch floor, so
  the O(√K) advantage cannot be seen there.
- Above it, flat keys grow with B·K (589 µs at K=4096, B=1024), while product keys stay at ≤ 107 µs:
  5.5× at K=4096.
- The PROMPT switch point (product keys from K=256) is consistent with this. Product keys cost about
  +10 µs of extra kernels at small K.
- The 107.5 µs for flat at K=1024, B=1 is a single-run outlier; the 3-run rule absorbs it.

### 4.2 Chart compute: lazy top-m vs all-K vs dense at equal active FLOPs (m = 2)

| K | B | lazy (gather) | lazy_tri | soft (all K) | dense eq-FLOPs (H) | PEER eq-FLOPs |
|---|---|---|---|---|---|---|
| 16 | 1 | 89.1 | 60.5 | 74.1 | 59.4 (144) | 132.1 |
| 1024 | 1 | 84.0 | 81.9 | 448.5 | 59.4 (145) | 126.0 |
| 4096 | 1 | 90.1 | 79.9 | 1,843 | 59.4 (146) | 125.3 |
| 16 | 1024 | 916.5 | 335.9 | 477.2 | 109.6 | 597.0 |
| 1024 | 1024 | 1,156.6 | 428.0 | 24,090 | 118.8 | 873.5 |
| 4096 | 1024 | 1,144.3 | 518.1 | 107,924 | 130.0 | 1,276.9 |

Dense SingleJEPA at fixed width (graph mode, µs):

| B | H=128 | H=512 | H=1024 | H=2048 | H=4096 |
|---|---|---|---|---|---|
| 1 | 57.6 | 74.3 | 105.5 | 173.1 | 561.2 |
| 1024 | 104.4 | 361.5 | 969.7 | 3,231.7 | 11,840.5 |

**What this means for the T7 bar.**
- At equal active FLOPs the atlas never wins:
  - At B = 1 it ties dense within the launch floor, or is up to 1.38× slower (K=1024).
  - At B = 1024 it is 3.1–4.0× slower.
- So T7 can pass only through quality matching, that is, if the quality-matched dense width H* is large.
- At the atlas reference (K=1024, m=2, lazy_tri):
  - At B = 1 the bar of 1.5× needs H* between 1024 and 2048: H=1024 gives 105.5 / 81.9 = 1.29×, which is NO, and
    H=2048 gives 2.11×.
  - At B = 1024 it needs H* between 512 and 1024: H=512 gives 0.84×, H=1024 gives 969.7 / 428.0 = 2.27×.
- The same B=1 thresholds hold at K=4096, where lazy_tri is 79.9 µs.

### 4.3 Gather cost and the faster path

- **Reference `StackedMLP.forward` is slow because of the gather.** It materialises the gathered weights
  `W[idx]` as an [N, m, i, h] tensor, which for E at B=1024, m=2 is 64 MB fp32. That tensor is written,
  then read again by an einsum.
- **The fused Triton kernel (`bench_t7._chart_mlp`) avoids that tensor.** It uses one program per
  (row, slot) and computes the whole chart MLP in one kernel: silu(x W1[e] + b1[e]) W2[e] + b2[e].
- **Measured gain:**
  - 2.2–2.7× at B = 1024 (916→336, 1,157→428, 1,144→518 µs).
  - 1.0–1.5× at B = 1.
  - Eager mode drops from ~6 ms to ~3 ms because the launch count falls.
- **Correctness:** parity with the gather path is within 1e-4 at all 5 tested shapes, and with the full
  atlas `observe` at K ∈ {16, 256 product-keys}. Removing the silu breaks 7/7 parity tests.
- **Remaining gap at B = 1024 against equal-FLOPs dense (~4×):** the kernel is a per-row GEMV with no
  tensor cores. Dense is a single SGEMM.
- **Next faster path:** sort (row, slot) by expert, then run a segment-grouped `tl.dot` over rows that
  share an expert, which is useful when B·m ≫ K. Reading bf16 weights would halve the
  ~34 KB-per-slot weight traffic.
- These are inference-only paths; training keeps the reference autograd path.

### 4.4 Eager numbers (not used for any bar)

- Eager steps take 0.4–7 ms on this Windows box under the contended CPU load.
- In eager mode, lazy (6.0 ms) is ~5.5× slower than dense (1.1 ms) at every batch size, because the atlas
  launches more kernels per step and launch overhead dominates.
- CUDA-graph replay removes this. The bar therefore uses each arm's best mode, as pinned in §3.

### 4.5 Design note (not a result): grouped bf16 kernel and backbone

- Measured with graph-mode medians, run conditions as in §4.1, m = 2.
- **Sort-by-expert grouped kernel (`bench_t7._grouped`, bf16 weights, fp32 accumulate, BM = 16).**
  - A single E call at K=4096, N=1024 runs 89 µs, against 211 µs for the per-row kernel and 782 µs for
    the gather path.
  - Inside a full `observe` step it gains only 1.0–1.13×. On narrow-input calls (P, φ, i ≤ 8) it loses to
    the per-row kernel, so `kind="auto"` uses grouped only for i ≥ 64.
  - Bound on bf16 error: 2e-2·max|ref|. Observed: 3.4–3.9e-3.
- **Backward pass.** `_Fused` recomputes the gather reference in fp32. Grads match the reference to 1e-5;
  for bf16 they are straight-through.
- **The backbone is the B = 1024 bottleneck: the cuDNN `nn.GRU` kernel, not the backbone's size.**
  - `nn.GRU` at T=1 takes 222 µs for d_h = 64, while `nn.GRUCell` with the same weights takes 64.5 µs.
  - Swapping in GRUCell (`lazy_cell`) brings the atlas to 127 / 194 / 221 µs at K = 16 / 1024 / 4096,
    from 337 / 381 / 466.
  - The gap to equal-FLOPs dense shrinks from 3.3–4.3× to 1.25–2.0×.
  - Shrinking d_h to 16 or 32 then changes nothing (185 / 187 / 194 µs at d_h = 16 / 32 / 64,
    K = 1024).
- **At B = 1 and B = 64 nothing changes.** Every arm stays on the 60–120 µs graph-replay floor.
- **Effect on the T7 threshold at B = 1024 (if Chase adopts GRUCell for streaming).** Against the earlier
  width sweep, H* ≥ 512 would suffice: 361.5 / 193.5 = 1.87×. Today it needs H* between 512 and 1024. B = 1
  still needs H* between 1024 and 2048.
- **TF32 asymmetry:** cuDNN runs the GRU in TF32 by default (7e-4 off from fp32), while the bench's
  matmuls are fp32. Pin `torch.backends.cudnn.allow_tf32` in PREREG.

### 4.6 Design note (not a result): one-launch fused step, with fair baselines

- **The kernel.** `bench_t7.fused_observe` runs routing, all m charts and the mix in one Triton launch
  per T=1 step: query, gate, exact top-m (flat keys for K ≤ 256, otherwise product keys over the
  √K×√K sum), softmax, E, P, φ(z), φ(ẑ), and the gated mix. The GRU stays outside the kernel.
- **Parity.** It matches `observe(mode="lazy")` on 10/10 configs (idx exact, w within 1e-5, g/gh/gate
  within 1e-4, cuDNN TF32 off). Two deliberate breaks were each caught: swapping the weights failed 6/10,
  inverting the router failed 10/10.
- **Fairness.** Dense, PEER and the width sweep also get `torch.compile` (the `*_c` arms). It changes
  nothing for dense (B=1024: 106→105 µs). It helps PEER only at B=1 (123–136 → 100–125 µs). Dense has no
  backbone, so the GRUCell swap has no counterpart there.
- **Graph-mode medians, fused + GRUCell (`lazy_fused_cell`) vs dense at equal active FLOPs:**

| B | K=16 | K=1024 | K=4096 | dense eq-FLOPs (best of plain/compiled) |
|---|---|---|---|---|
| 1 | 60.5 | 60.1 | 62.5 | 59.4–61.4 |
| 64 | 67.6 | 63.5 | 57.9 | 71.7–77.8 |
| 1024 | 119.8 | 131.1 | 169.0 | 105.5–108.5 |

- **Reading.**
  - B=1: the atlas reaches the ~60 µs graph floor (ties dense).
  - B=64: the atlas beats dense by 1.1–1.3×.
  - B=1024: the atlas is 1.1–1.6× slower than dense. Before the fused kernel, GRUCell and bf16 it was
    3.3–4.3× slower.
  - Eager: the fused step takes 0.68–0.84 ms vs 1.0–1.2 ms for dense (fewer launches).
- **T7 consequence.**
  - The bar still rests on H*. In this run the width sweep at B=1 gives 104 µs at H=1024 and 164 µs at
    H=2048, so the atlas is 1.7× faster at H*=1024 and 2.6× faster at H*=2048.
  - At B=1024 it is 2.2× faster at H*=512 (289 vs 131 µs).
  - Run-to-run drift is about 30% at B=1024 (densew_1024: 970 → 696 µs across sweeps), so this is not
    decisive until the 3-run rule is applied.

### 4.7 Runner pins (`run_t3.py`; decided by the T3 owner, for Inspector)

- **The soft all-K reference runs only at K ∈ {16, 64, 256}.**
  - At K = 1024 it peaks at ≈ 9 GiB on the 8 GB card, about 56 s/step with driver spill (so3, batch
    64×32, 2026-10-04).
  - The "lazy within 1% of soft" bar is therefore checked at K ≤ 256, and K = 1024 is reported as
    "not run (resource)".
  - Per-chart SIGReg in soft mode uses `sigreg_chunked`. Its value and gradient equal `zero/sigreg.py` to
    1e-12 (float64 test), with peak memory O(chunk), chunk = 16,384.
  - Soft K = 256 measured 2.26 GiB peak and ≤ 1.43 s/step. At 20,000 steps that is about 8 h per job,
    and the 12 soft K = 256 jobs (2 beds × 6) total about 95 GPU-hours, about 32 h at 3 concurrent.
  - Inspector to decide: fewer steps for soft, or accept the cost.
- **The scheduler launches only when all hold:**
  - own workers + other GPU compute processes < 3;
  - free RAM > 3 GB;
  - torch threads ≤ 4 per worker;
  - git tree clean (L-REPRO).

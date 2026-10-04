# OWNERS.md — T1 owner hunt (Wilson, 2026-10-04)

**Full combination owned? NO** — no paper/repo located that combines all four OBSERVER-ATLAS deltas (separate encoder+chart per module; lazy eval of selected modules only; partition-of-unity gluing into a higher-dim space; topology/nerve readout from the router). Basis: 16 §5 owners read + 8 §8 queries + 16 own variants, each also run with an " arxiv" suffix (all listed in §C). Absence of a hit is not proof of absence.

**Partial owners per delta** (detail in §A/§C):

| Delta | Owned by (verified) | Not owned |
|---|---|---|
| (1) separate encoder + chart per module | Chart AE 1912.10094; CAE-semi 2208.10570; Floryan & Graham 2108.05928 (per-chart AE **and per-chart dynamics model**); Mixture of VAEs 2303.15244; multi-chart flows 2106.03500 / 2505.24665; autoencoder atlases 2602.22873; RIMs 1909.10893 (separate hidden state per module) | none of these is a JEPA; none routes charts by a learned context-only softmax trained end-to-end with a world model |
| (2) lazy eval: only selected modules compute | MoE 1701.06538; PEER 2407.04153; RIMs 1909.10893 (only top-k_A RIMs update); Mixture-of-Depths 2404.02258 | MoP-JEPA computes all K heads every step (emits full set {(u_k, π_k)}_{k=1..K}); chart AEs evaluate all charts |
| (3) PoU gluing into higher-dim space | POUnets 2101.11256 / 2107.03066 (softmax PoU, polynomial experts, regression); CAE-semi 2208.10570 (softmax-of-RBF PoU, Σρ_α=1); Chart AE 1912.10094 (softmax chart prediction; per-chart decoders back to ambient space) | none glues JEPA latents; none trains gluing maps φ_k with overlap consistency inside a world model |
| (4) topology/nerve readout from the router | partial only: 2602.22873 reads w1 (orientability) from Jacobian-sign Čech cocycle over the chart-cover nerve; cover learning → nerve 2503.09767; G-Mapper 2309.06634; Mapper-nerve representation 2608.22044 | no located work reads Betti numbers from a router's co-selection nerve (nurse query "nerve of mixture-of-experts routing" and "router nerve simplicial complex experts Betti numbers": no such hit) |

Closest single owners: **MoP-JEPA (2607.05238v1)** for the router-over-JEPA-predictors + mode-collapse arm; **Floryan & Graham 2108.05928** for atlas + per-chart latent dynamics; **2602.22873** for topology read from a learned chart atlas.

---

## A. §5 owners — bodies read

Tags: [V] fetched this session; quote = verbatim.

### MoP-JEPA — arXiv 2607.05238 [V] — CITATION HAZARD
- **v1 (6 Jul 2026) and v2 title:** "MoP-JEPA: Hard-Assigned Predictor Mixtures for Stochastic JEPA World Models". Authors v1: Zhi Song, Ximing Xing, Zhenchao Tang, Hanbo Huang, Tianxu Lv, Minghao Yang, Zhongzheng Niu, He Bing, Lusheng Wang, Jianhua Yao (City Univ. Hong Kong; Tencent). https://arxiv.org/abs/2607.05238v1
- **v3 (3 Aug 2026) retitled:** "Branch-JEPA: Finite-Support Predictive Distributions for JEPA World Models"; 12 authors (adds Jiehui Huang, Weilong Yan); abstract now reports Argoverse 2 (trajectory Energy Score 5.8–6.5%, prob.-weighted distance 9.3–10.4%). The bare ID now resolves to Branch-JEPA. MoP-JEPA content must be cited as **2607.05238v1** (v2 title verified; v2 body not read).
- **Numbers verified (v1 abstract):** "planning over single-predictor rollouts performs poorly (0.02–0.09 success) while planning over our predicted modes reaches up to 0.85". **Metric is planAll**, not closed-loop execution: v1 Table 2 caption "planAll (searchability of the predicted graph; 5 seeds)". v3 supplement S10: "planAll is the success rate on up to 300 sampled, truly reachable start–goal queries under predicted-graph BFS" and "none of these quantities is closed-loop task success".
- **v1 Table 2 (planAll, 5 seeds, mean±std):**

  | arm | med-stitch | teleport | large-stitch |
  |---|---|---|---|
  | dense (JEPA) | .084±.03 | .037±.01 | .016±.01 |
  | M3-JEPA (gated MoE) | .075±.02 | .036±.01 | .031±.01 |
  | Var-JEPA | .085±.01 | .035±.01 | .033±.00 |
  | MDN | .566±.20 | .512±.12 | .717±.12 |
  | **MoP-JEPA** | **.851±.08** | .748±.09 | .386±.45 |
  | codebook (control) | .267±.02 | .167±.02 | .149±.04 |

  0.85 = MoP-JEPA on pointmaze-medium-stitch; "0.02–0.09" = dense/M3/Var range (.016–.085).
- **v1 Table 3 realroute (5 seeds):** dense .083/.035/.016; MDN .134/.039/.057; MoP-JEPA .247/.192/.139 (med/teleport/large). Matched K=16, 60k steps on large-stitch: MoP .208 vs MDN .095.
- **v1 Table 4 execution (real env, official info['success'], replanning executor), med/large:** dense .27/.13; M3, Var "----"; MDN .87/.27; MoP .67/.33. Large-stitch .33 placed "second among the seven published OGBench offline-GCRL baselines" (QRL 84, ours 33, GCIQL 31, HIQL 13, GCIVL 12, GCBC 7, CRL 0) — quoted, not re-run by authors. v3 S10: "A separate three-seed execution control is inconclusive: medium-stitch replanning is 86.7±11.5/66.7±11.5% for MDN/Branch-JEPA".
- **Architecture (v1):** shared encoder f(s)→z∈R^d with EMA target, momentum τ=0.996; K predictor heads g_k on c=[z,a] (z L2-normalised); context-only router r: R^{d+p}→R^K, π_k = softmax(r(c))_k, trained by cross-entropy on the hard winning index; loss = best-of-K (WTA) regression + router CE + load-balance KL(usage‖Unif(K)). Router gating counts heads with π_k > 0.5/K. Prop. 2: gated weighted-sum MoE collapses like a single head. K-sweep: teleport COV-R 0.231 (K=1) → 0.712 (K=3) → 0.945 (K=8).
- **OGBench protocol (for reproduction):** datasets "pointmaze-medium-stitch, pointmaze-large-stitch, pointmaze-teleport-navigate, and antmaze-teleport (29-D)"; "coordinate features (no cell-identity memorization) and 20% of unique transitions held out of training"; "Five seeds for planning, three for coverage and execution". From v3 S10 (graph campaigns): "K=8, 30k updates, 512-dimensional latents, coordinate features"; "A candidate creates an edge when its cosine similarity to a cell latent exceeds .5; all heads are retained"; planAll campaign "fits action-marginalized point-maze predictors to current coordinates". Planner: BFS on predicted graph (planAll/realroute); closed-loop replanning executor for execution. Not found in v1 (9-page PDF, appendix cited but absent): encoder architecture, predictor width/depth, total params, batch size, LR, default steps. → **unverified**.
- **Code public?** No code/GitHub URL in v1 PDF, v1 HTML, or abs pages v1–v3; nurse web search found none. → **not located (unverified absence)**.
- Owns: router over K JEPA predictors; mode-collapse fix (C4/T4b). Does NOT: separate encoder/chart per head (encoder shared); lazy eval (all K heads computed); gluing; topology readout (no mention found).

### Chart Auto-Encoders — 1912.10094 [V]
Schonsheck, Chen, Lai; 20 Dec 2019. "Chart Auto-Encoders for Manifold Structured Data". Own encoder–decoder per chart ("encoders E_α map z to several chart spaces"); chart prediction module P outputs p_α summing to 1, CE against softmax(−e_α); transition φ_αβ = E_β∘E∘D∘D_α; "requires L > d charts"; all charts evaluated (min_α e_α). No dynamics/JEPA, no router-driven lazy compute, no nerve/Betti readout.

### CAE semi-supervised — 2208.10570 [V]
Schonsheck, Mahan, Klock, Cloninger, Lai; 22 Aug 2022. "Semi-Supervised Manifold Learning with Complexity Decoupled Chart Autoencoders". N chart latent spaces, each with decoder; PoU P(x)=S(y(x)ᵀw), y_i RBF, S softmax, Σρ_α=1. Inference "can use p(x) to determine which chart(s) to use ... but the network computes all chart predictions". No dynamics, no topology readout.

### POUnets — 2101.11256 [V]
Lee, Trask, Patel, Gulian, Cyr; 27 Jan 2021. "Partition of unity networks: deep hp-approximation". PoU = NN with softmax (RBF-Net or ResNet+softmax); experts = polynomials π_m; regression. No lazy eval, no charts-as-encoders, no nerve.

### Probabilistic POUnets — 2107.03066 [V]
Trask, Gulian, Huang, Lee; 7 Jul 2021. Gaussian-mixture PoU, ML loss, polynomial experts, regression; PCA bisection gives "simply-connected" partitions. Same non-ownership as above.

### PEER — 2407.04153 [V]
Xu Owen He; 4 Jul 2024. "Mixture of A Million Experts". Product-key retrieval over 1024² single-neuron experts; complexity O((√N+k²)d); h heads × k experts summed. Owns lazy top-k at scale + product keys. No charts, gluing, topology, world model.

### MoE — 1701.06538 [V]
Shazeer et al.; 23 Jan 2017. Noisy top-k gating; up to 131,072 experts; 99.994% layer sparsity at 65,536 experts, k=4. Owns sparse routing; nothing else.

### RIMs — 1909.10893 [V]
Goyal, Lamb, Hoffmann, Sodhani, Levine, Bengio, Schölkopf; 24 Sep 2019. Separate recurrent state per module, attention competition (vs null input) selects top-k_A of k_T; inactive RIMs keep state. Owns (1)-like module separation + (2) lazy update. No charts, PoU gluing, topology.

### V-JEPA 2 — 2506.09985 [V]
Assran, Bardes, Fan, Garrido, Howes + 25 others; 11 Jun 2025. V-JEPA 2-AC: latent action-conditioned world model post-trained on <62 h Droid video; planning with image goals. Single encoder/predictor. MoP-JEPA v1 states the released V-JEPA 2-AC predictor is 305M params and "emits a single next-latent per (state, action)". Planner details (CEM horizon) not extracted → unverified.

### I-JEPA — 2301.08243 [V]
Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas; 19 Jan 2023. Single context block → target-block representations. No mixture.

### LLM-JEPA — 2509.14252 [V]
Huang, LeCun, Balestriero; 11 Sep 2025. JEPA objective for LLMs. No mixture/charts.

### LeJEPA / SIGReg — 2511.08544 [V]
Balestriero, LeCun; 11 Nov 2025. SIGReg constrains embeddings toward isotropic Gaussian; "no stop-gradient, no teacher-student". Owns anti-collapse term (T4a).

### Mapper — Singh, Mémoli, Carlsson 2007 [V-citation]
"Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition", Eurographics SPBG 2007, https://diglib.eg.org/handle/10.2312/SPBG.SPBG07.091-100 (title/venue only; body not read).

### Nerve theorem [V-statement]
Wikipedia "Nerve complex": Leray's nerve theorem — if every intersection is contractible or empty, N(C) is homotopy-equivalent to ⋃C. https://en.wikipedia.org/wiki/Nerve_complex

### Rudyak–Schlenk — math/0605350 [V] — see §B.
### Zhou et al. — 1812.07035 [V] — see §B.

### Gated attention — 2505.06708 [V]
Qiu, Wang, Zheng, Huang, Wen, Yang et al.; 10 May 2025. Head-specific sigmoid gate after SDPA; "mitigates 'attention sink'". Supports §3 "gated observer".

### Mixture-of-Depths — 2404.02258 [V]
Raposo, Ritter, Richards, Lillicrap, Humphreys, Santoro; 2 Apr 2024. Top-k token routing with capped compute.

---

## B. §3 maths — verification

| Claim | Status | Source |
|---|---|---|
| No single chart covers a closed manifold | **verified** | Invariance of domain (Wikipedia): image of a chart is open in Rⁿ; a compact image open in Rⁿ is impossible for n≥1. Floryan & Graham 2108.05928 Fig. 2 demonstrates the circle gap. |
| Bound "cat(M)+1 ≤ atlas(M) ≤ n+1" | **verified with a normalisation caveat** | Rudyak–Schlenk math/0605350 eq. (2): "cl(M)+1 ≤ cat M ≤ B(M) ≤ m+1 for any closed m-dimensional manifold", where cat X := min #open sets contractible in X (**unnormalised**, cat(pt)=1) and B(M) := min # open sets diffeomorphic to the open ball covering M. PROMPT's form is correct only with normalised cat (cat(pt)=0). RS Remark 4.2.2: closed manifolds with cat M < B(M) exist, so atlas(M) is not always cat-determined. |
| Sⁿ: 2 charts | **verified** | Lower bound: single-chart impossibility (above). Upper: two stereographic charts. Consistent with RS (2), cl(Sⁿ)=1. |
| Closed surfaces ≠ S²: 3 | **verified (derivation)** | RS (2) with Z/2 cup length 2 for every closed surface with H¹(Σ;Z/2)≠0 (mod-2 Poincaré duality) ⇒ 3 ≤ B ≤ 3. Mod-2 Poincaré duality step = [K]. |
| SO(3) ≅ RP³: 4 charts | **verified (derivation)** | Wikipedia "3D rotation group": SO(3) "diffeomorphic to the real 3-dimensional projective space". H*(RP³;Z/2)=Z/2[w₁]/(w₁⁴) (Wikipedia "Real projective space") ⇒ cl=3 ⇒ RS (2): 4 ≤ B(RP³) ≤ 4. |
| Zhou et al. 1812.07035 | **verified** | Abstract: "for 3D rotations, all representations are discontinuous in the real Euclidean spaces of four or fewer dimensions"; "3D rotations have continuous representations in 5D and 6D". |
| Z2 Betti S² = (1,0,1) | **verified** | Mod-2 Poincaré duality b₂=b₀=1; χ(S²)=2 ⇒ b₁=0. |
| Z2 Betti T² = (1,2,1) | **verified** | Wikipedia "Betti number": torus b₀=1, b₁=2, b₂=1 (torsion-free, same over Z/2). |
| Z2 Betti Klein = (1,2,1) | **verified (derivation)** | Wikipedia "Klein bottle": χ = 0, H₁(K;Z)=Z⊕Z/2. Mod-2 duality b₂=b₀=1, χ=0 ⇒ b₁=2. (Over Q it is (1,1,0); PROMPT's Z2 coefficient choice matters.) |
| Z2 Betti RP³ = (1,1,1,1) | **verified** | H^k(RP³;Z/2)=Z/2 for k=0..3 (Wikipedia "Real projective space", mod-2 ring). |

**Additional verified fact bearing on P-COUNT vs P-BETTI (no recommendation):**
- The nerve theorem needs a good cover (contractible intersections). The minimum size of a good cover is the covering type ct(M) (Karoubi & Weibel, "On the covering type of a space", arXiv 1612.00532): ct(Sᵐ)=m+2 (Ex. 3.2; ct(S²)=4), ct(T²)=7 (Thm 5.3), ct(RP²)=6 (Thm 5.4), ct(Klein) ∈ {7,8} (Cor. 5.6, open). ct(RP³): **unverified**. 2602.22873 abstract: "the minimum number of autoencoder charts is determined by the good cover structure of the manifold".
- Consequence (arithmetic): with K = atlas(M) charts the chart cover is not good for these manifolds (e.g. 2 charts on S² intersect in an annulus; a 2-vertex nerve has Z2 Betti (1,0,0) or (2,0,0), never (1,0,1)), so the nerve theorem does not deliver Betti(M) at K = atlas(M). Nerve Betti recovery via the nerve theorem requires K ≥ ct(M).

Nurse-reported but not verified here: Wikipedia LS-category normalisation/values ("n-spheres: LS-category = 2"); "Minimal atlases on 3-manifolds" (Cambridge) chart counts; Chekanov–van Koert–Schlenk 0807.3047 counts are for **contact** Darboux charts (different invariant), not smooth atlases.

---

## C. Query log (every query; relevant hits)

Hits with arXiv IDs marked ✓ were re-verified via arXiv API (title/authors/date). Others are nurse-reported, ID not re-verified.

**§8 T1 queries**
1. "mixture of JEPA encoders" — no relevant web hits; +arxiv: SiamJEPA 2607.04044 ✓ (Siamese student encoders, no routing); Var-JEPA 2603.20111 ✓ (Gögl & Yau; variational JEPA). Not owners.
2. "atlas world model" — only commercial "Atlas" news; +arxiv: "We Should Chart an Atlas of All the World's Models" 2503.10633 ✓ (model-zoo graph; unrelated).
3. "multi-chart latent world model" — multi-chart flows 2106.03500 ✓ (Kalatzis et al.; glued local flows, density estimation, no dynamics); Chart AE 1912.10094 ✓.
4. "chart-based JEPA" — Bar-JEPA 2608.06062 (bar-chart value extraction; unrelated); T-JEPA 2406.12913; RiJEPA 2603.13265. No owner.
5. "partition of unity mixture of experts gluing" — PPOU-Net high-dim 2210.02694 ✓ (MoE on low-dim manifold, polynomial experts, regression); PUNN classification 2602.00511 ✓; MoE expressivity 2505.24205 ✓ (piecewise functions on manifolds); GLUE 2512.22467.
6. "lazy mixture of experts world model" — PRISM-WM 2512.08411 ✓ (MoE gating identifies physical mode, experts predict transitions, latent orthogonalisation vs mode collapse; no charts/nerve); TMoW 2601.22647 ✓ (test-time routing over world models); PoE-World 2505.10819 ✓ (programmatic experts).
7. "nerve of mixture-of-experts routing" — **no hit combining nerve with MoE routing**; general routing: 2606.14398 ✓, ProbMoE 2606.01509, VMoER 2603.09453.
8. "Mapper learned cover neural" — G-Mapper 2309.06634 ✓; Cover Learning 2503.09767 ✓ (Scoccola, Lim, Harrington; learned covers → nerve); Structured Learning on Mapper 2608.22044 ✓; MAPLE 2607.06797 ✓. None uses a router.

**Own variants**
9. "mixture of world models chart" — MuSix 2607.00457 ✓ (two-stage routing over world models; no charts/topology); PRISM-WM 2512.08411 ✓.
10. "manifold atlas neural network mixture of experts" — stratified-manifold MoE 2502.13577 ✓ (MoE analysis of LLM embedding strata); 2505.24205 ✓; deep MoE 1312.4314.
11. "topological analysis of MoE routing" — tropical-geometry MoE 2602.03204 ✓; MoE-as-soft-clustering 2601.11616 ✓ ("interpreting routing as soft partitioning into overlapping expert-local charts"; Jacobian/PCA probes, no homology); "Equifinality ... Routing Topology" 2604.14419 ✓ ("topology" = routing variant, not homology); TA-MoE 2302.09915, MoX 2607.20220 (network topology; unrelated).
12. "learned atlas latent dynamics" — ATLAS 1404.0667 ✓ (Crosskey & Maggioni; local affine charts + learned SDEs, transition maps); Floryan & Graham 2108.05928 ✓; ScaleAware-JEPA 2606.29723 ✓ (multiscale, no charts).
13. "multiple chart latent dynamics SO(3)" — 2108.05928 ✓; rigid-body latent SO(3) 2209.11355 ✓ (single latent homeomorphic to SO(3)); SO(3) NCDE 2506.06780 ✓; LLPNN 2607.28939 ✓.
14. "neural atlas manifold learning transition maps" — differentiable atlas 2510.17772 ✓ (Klein-bottle classification; no dynamics); **autoencoder atlases 2602.22873 ✓** (Paluzo-Hidalgo & Ike: cocycle transition maps, w1 from Jacobian signs over the cover's nerve, chart count ↔ good cover; no router, no dynamics); Minimal Neural Atlas 2207.14782 ✓ (3-chart surface parameterisation; charts' domains have arbitrary topology).
15. "atlas of charts variational autoencoder dynamics" — Mixture of VAEs 2303.15244 ✓ ("every encoder-decoder pair represents one chart"; inverse problems, no dynamics); 2604.16282 ✓ (single chart + latent SDE); 1803.00156.
16. "mixture of predictors JEPA multimodal future" — M3-JEPA 2409.05929 ✓ (MMoE predictor fused by weighted sum; MoP-JEPA Prop. 2 says it collapses); MoP-JEPA 2607.05238v1 ✓.
17. "persistent homology of expert co-activation mixture of experts" — **no hit combining PH with expert co-activation**.
18. "router nerve simplicial complex experts Betti numbers" — **no relevant hit**.
19. "sparse mixture of tiny world models top-k routing latent prediction" — SMoE discontinuities 2606.19036 ✓; shared routing geometry 2609.02404 ✓. No world-model atlas.
20. "multi-chart embedding neural network sphere torus discontinuity" — multi-chart flows 2505.24665 ✓ (Yu, Hauberg et al.: "if the manifold has a non-trivial topology, it can never be correctly learned using a single flow"); Topological Obstructions to Autoencoding 2102.08380 ✓; 2602.22873 ✓.
21. "mixture of experts JEPA" — JEPA generalization theory 2606.27014 ✓; Orthogonal JEPA 2608.20065 ✓ (factorised predictive branches); MoE latent clusters 2506.01656 ✓.
22. "atlas of JEPA world models charts router topology nerve" (Wilson) — no relevant hit.
23. "multiple local encoders sparse router latent dynamics chart transition partition of unity world model arxiv 2026" (Wilson) — LpWM 2608.22764 ✓ (Kuang, …, Balestriero, LeCun: sparse/one-hot JEPA latents, "mode-factored" support; single encoder, no charts).
24. Nurse 13 extra: "atlas of latent spaces world model 2026", "multi-chart JEPA 2026", "topological world model charts 2026" — ATLAS (Aligned Transport of Latent Structure) 2609.36333 ✓ (relational-geometry regulariser for world-model planning; name collision only); 2607.04500 (JEPA driving data diversity).

### Per-hit "does NOT do" for the closest non-§5 owners
- **Floryan & Graham 2108.05928** ✓ (Nature Mach. Intell. 2022 per nurse): k-means partition expanded along a graph gives overlapping chart domains; one autoencoder + one dynamics NN per chart; chart switching by nearest centroid / interior-point rule. Not SGD-learned routing (k-means), no JEPA/anti-collapse, no PoU gluing (switching), no topology readout.
- **2602.22873** ✓: topology (w1) from transition-map Jacobians of a trained AE atlas; "obtain topological information posthoc from the transition cocycle". No router, no dynamics, no lazy compute; Betti numbers not the readout (orientability).
- **PRISM-WM 2512.08411** ✓: MoE world model with gating per physical mode; per-expert chart/encoder: unverified (abstract only).

---

## D. Limits
Nurse reports (haiku) were used for search coverage; every ID cited as ✓ was re-checked against the arXiv API, and every MoP-JEPA number above was read directly from the v1 PDF text (pdftotext) and v3 PDF supplement. MoP-JEPA hyperparameters beyond those quoted, code availability, PRISM-WM internals, ct(RP³), and the Wikipedia LS-category normalisation are unverified. Bodies of Mapper 2007 and V-JEPA 2-AC planning section were not read.

---

## E. T-EV owner hunt (event-driven laziness in time; Wilson 2026-10-04)

Claim checked:
- (a) trigger Δ_t = ‖o_t − o_last‖/√D ≥ ε against the last computed frame;
- (b) ε learned (softplus, straight-through, compute-cost penalty);
- (c) sample-and-hold between triggers;
- (d) wrapper around a learned world model / latent predictor;
- (e) scored by quality loss vs always-compute plus executed FLOPs.

**Full combination owned? NO.** Not located across 8 queries (nurse 33, sonnet; arXiv IDs re-verified by Wilson via the arXiv API where marked ✓).

| Owner | ID | Owns | Does NOT do |
|---|---|---|---|
| Skip RNN (Campos et al.) | 1708.06834 ✓ | (b)-like learned binary gate, straight-through, budget penalty; (c) state copy on skip | (a): the gate is learned from the hidden state, not an observation-delta threshold; (d) |
| Delta Networks (Neil et al.) | 1612.05571 ✓ | (a) change vs last transmitted value ≥ threshold; (c); (e) op counts | (b): fixed per-neuron threshold; (d) |
| CATS (Zhu et al., WACV 2024) | no arXiv id (CVF page) | (b) gradient-learned temporal thresholds plus event penalty | (a) as a single observation gate (thresholds are per layer in CNNs); (d) |
| Delta Activation Layer (Yousefzadeh, Sifalakis) | 2107.07305 | gradient-optimised quantisation step plus sparsity penalty | (a), (c), (d) |
| Eventful Transformers (Dutson et al.) | 2308.13494 | (a), (c), (e) per token in ViTs | (b): threshold / top-r hand-set; (d) |
| DeltaCNN | 2203.03996 | (a) truncated frame deltas, (e) | (b), (d) |
| Skip-Convolutions | 2104.11487 | learned residual gates or magnitude threshold | single observation ε; (d) |
| DISK (dynamic inference skipping for world models) | 2602.00440 | (c), (d), (e): training-free skip and cache reuse in autoregressive world models | (a): trigger on latent curvature between steps; (b): θ hand-set |
| WorldCache | 2603.06331 | (c), (d), (e); normalised drift trigger | (b): training-free; drift on denoising features, not observations |
| ACT (Graves) | 1603.08983 ✓ | learned compute amount per input | time-skipping on observation deltas |
| PonderNet | 2107.05407 ✓ | learned halting | same as ACT |
| Mixture-of-Depths | 2404.02258 ✓ | top-k token compute routing | time triggers |
| Event cameras survey (Gallego et al.) | 1904.08405 ✓ | per-pixel change events in hardware | learning, (d) |
| Send-on-delta (Miśkowicz 2006, Sensors 6(1):49-63) | doi:10.3390/s6010049 (Crossref) | canonical (a)+(c) | learning, (d) |

Limits: WebSearch snippets plus the arXiv API only; IDs without ✓ were verified by the nurse, not re-checked by Wilson. CATS has no arXiv record.

---

## F. QF-LM owner hunt (PLAN.md; Wilson 2026-10-04)

**Full combination owned? NO.** But the core mechanism (2)+(3), "measure to remember" on natural language, IS owned in formalism by Pestun–Terilla–Vlassopoulos 2017 (row below; body read by Wilson). The unowned parts are (4) input-free butterfly unitaries, (5) the null / think outcome with likelihood summed over null records, and (6) training at LM scale with reported results.

No located work combines all six components:
1. state ψ ∈ C^N;
2. token = outcome of a complete instrument, p(v) = ‖K_vψ‖², no softmax;
3. Lüders back-action as the only input path (no embedding);
4. input-free butterfly unitaries between tokens;
5. null outcome = thinking, with likelihood summed over null records;
6. a trained LM on natural text.

✓ = arXiv API title/author/date verified by Wilson. Mechanism notes come from abstracts (✓) or a nurse reading (n36).

| Owner | ID | Owns | Does NOT do |
|---|---|---|---|
| **Language as a matrix product state (Pestun, Terilla, Vlassopoulos 2017)** | 1711.01416 ✓ (PDF body read) | **(2)+(3) in formalism, on natural language.** Each word w_i maps to a d×d complex matrix M_i ("dictionary"). A phrase maps to the product M_{i1}⋯M_{ik}, with q(x) = tr(P_L M_x P_R M_x†) (§2.2). Under Σ M_i†M_i = I the M_i "can be interpreted as a collection of n measurement operators" (§4). The word matrix both scores the word (Born-type trace density) and transforms the state for the next word: the same operator writes and reads, with no separate input embedding. | Mixed-state trace density, not an explicitly normalised pure-state Lüders filter. **No null outcome, halting or adaptive think steps** (body grep: none). No butterfly or input-free unitary between words. **No experiments:** fitting by maximum likelihood is described only as possible (§5, ref. [18]); no corpus results, tables or figures in this paper. |
| Hidden Quantum Markov Models (Monras, Beige, Wiesner 2010) | 1002.2337 ✓ | (2)+(3) in substance: a sequence generator from "stochastic quantum operations" (Kraus instruments) on a quantum state; outputs are measurement outcomes, and each outcome updates the state | (4), (5), (6); no complex-N LM training at scale |
| Learning HQMMs (Srinivasan, Gordon, Boots 2018) | 1710.09016 ✓ | the learning algorithm for HQMMs (Kraus-operator sequence models) | evaluated "using several synthetic datasets"; (4), (5), (6) |
| uRNN (Arjovsky, Shah, Bengio 2016) | 1511.06464 ✓ | complex hidden state with unitary recurrence composed of structured factors | input embedding; softmax/linear readout; no measurement back-action; (5) |
| MPS Born machine (Bradley, Stoudenmire, Terilla 2019) | 1910.07425 ✓ | sequence distributions as a pure entangled quantum state (Born probabilities) | DMRG-trained tensor network, not a recurrent filter with back-action; (4), (5) |
| u-MPS (Miller, Rabusseau, Terilla 2020/21) | 2003.01039 ✓ | uniform MPS for probabilistic sequence modelling, incl. text experiments; autoregressive sampling | no instrument back-action state; (4), (5) |
| The Quantum Shortcut (Nebli et al. 2026) | 2608.14691 ✓ | ψ ∈ C^N heads, unit-modulus Cayley phase recurrence, Born-style scoring (n36) | "The input embedding is tied to the output head" (keeps an embedding); no complete instrument, back-action or null; input-dependent recurrence |
| Weight tying (Press, Wolf 2017) | 1608.05859 ✓ | input embedding = output matrix | embedding plus softmax |
| Butterfly factorisation (Dao et al. 2019) | 1903.05895 ✓ | learned O(N log N) butterfly factors | not used as an LM state evolution |
| Pause tokens (Goyal et al. 2023) | 2310.02226 ✓ | inserted "thinking" steps before output | "we do not apply a loss ... to predict these tokens"; no marginalisation over null records |
| PonderNet / ACT | 2107.05407 ✓ / 1603.08983 ✓ | loss summed over a halting-step distribution / learned step count | not a sum over null measurement records; not an instrument |

**Closest owners of the core mechanism ("measure to remember").**
- Pestun–Terilla–Vlassopoulos 2017 (1711.01416): word = measurement operator; the phrase is the operator product, i.e. write = read, on natural language, in formalism, with no experiments.
- Hidden Quantum Markov Models (1002.2337, 1710.09016): Kraus-instrument outputs update the state; synthetic data only.
- QF-LM's unclaimed parts are (4) input-free butterfly thinking, (5) the null outcome / think-likelihood sum, and (6) a trained LM-scale result. Criterion (a), the +embed ablation, tests whether write = read pays; it does not establish that write = read is new.

**Not run** (nurses stopped by the user): searches for "measurement-based language model", "Lüders update RNN", "instrument language model" and quantum filtering LMs, beyond nurse 36's three searches. Nurse 36 found no paper combining a no-embedding model, a Lüders back-action update and a Kraus-instrument token readout. Coverage is partial.

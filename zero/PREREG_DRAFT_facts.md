# PREREG_DRAFT_facts.md — verified facts for PREREG (Wilson, 2026-10-04)

Facts only. **[V]** = read from the cited source this session. **[D]** = derived here, with the proof or check given. **[U]** = unverified. Exhaustive checks were run with a Z2/Q Gaussian-elimination script (scratchpad `betti.py`, `enum.py`). Results are reproduced in `ref/triangulations.json`.

## 1. §3 corrected statements

### 1.1 LS category vs atlas count
- **[V]** Rudyak–Schlenk, *Minimal atlases of closed symplectic manifolds*, arXiv math/0605350 (13 May 2006), p.3:
  - cat X := min{k | X = A₁ ∪ … ∪ A_k}, each A_i open and contractible in X. This is **unnormalised**: cat(point) = 1.
  - B(M) := min{k | M = B₁ ∪ … ∪ B_k}, each B_i diffeomorphic to the standard open ball.
  - Eq. (2): **cl(M) + 1 ≤ cat M ≤ B(M) ≤ m + 1** for any closed m-dimensional manifold, where cl is the cup-length over any coefficient ring.
- Correct PROMPT form: **cat(M) ≤ atlas(M) ≤ n+1** with unnormalised cat. With normalised cat (cat(point)=0) it is cat(M)+1 ≤ atlas(M) ≤ n+1. PROMPT §3 as written mixes the two forms.
- **[V]** RS Remark 4.2.2: closed manifolds with cat M < B(M) exist.

### 1.2 Ball-chart atlas counts B(M) **[D from RS eq. (2) + Z2 cup-length]**
| M | Z2 cup-length cl | cl+1 ≤ B(M) ≤ n+1 | B(M) |
|---|---|---|---|
| Sⁿ (n≥1) | 1 | 2 ≤ B ≤ n+1; two stereographic charts | **2** |
| T², Klein, RP², every closed surface ≠ S² | 2 (H¹(Σ;Z2) ≠ 0, nondegenerate mod-2 cup pairing [K: mod-2 Poincaré duality]) | 3 ≤ B ≤ 3 | **3** |
| RP³ ≅ SO(3) | 3 (H*(RP³;Z2)=Z2[w₁]/(w₁⁴), [V] Wikipedia "Real projective space") | 4 ≤ B ≤ 4 | **4** |

### 1.3 Ball charts vs general charts — PROMPT §3 / P-COUNT caveat **[D]**
PROMPT's counts (T² = 3, SO(3) = 4) are **ball-chart** counts B(M). A JEPA chart E_k only needs to be a homeomorphism from its domain onto **some open subset of Rⁿ**. Under that definition the counts drop:
- **T²: 2 charts.** U = S¹ × (S¹∖{p}) and V = S¹ × (S¹∖{q}), with p ≠ q. Each maps onto R²∖{0} by (θ, t) ↦ eᵗ(cos θ, sin θ), and U ∪ V = T².
- **Klein bottle: 2 charts.** Write K = [0,1]×S¹/((0,y)∼(1,−y)). U = (0,1)×S¹ is an annulus. V = a product neighbourhood of the 2-sided circle {0}×S¹ is also an annulus. Both are planar.
  - Explicit form: V := {(t,y) : t ∈ [0,ε) ∪ (1−ε,1]}, with φ: (−ε,ε)×S¹ → V, φ(t,y) = (t,y) for t ≥ 0 and (1+t, −y) for t < 0. At t = 0 this is consistent with the gluing (0,y) ∼ (1,−y). φ is a homeomorphism onto V, so V ≅ (−ε,ε)×S¹, an annulus.
- **RP³ ≅ SO(3): 2 charts.**
  - [V] Wikipedia "Lens space": "a lens space can be visualized as the result of gluing two solid tori together by a homeomorphism of their boundaries", and "L(2;1), three-dimensional real projective space".
  - Write RP³ = V₁ ∪ V₂, solid tori with V₁ ∩ V₂ = ∂V₁ = ∂V₂. Take U_i = V_i ∪ (open collar of ∂V_i in the other torus). Each U_i ≅ S¹ × open disk, which is homeomorphic to the open solid torus of revolution in R³, an open subset of R³. U₁ ∪ U₂ = RP³.
  - One chart cannot be RP³∖{pt}: that set contains RP², and RP² does not embed in R³ [K: closed non-orientable surfaces do not embed in R³].
- **Sⁿ: 2** (two stereographic charts).
- **RP²: 3 general charts. [D, proof]**
  - Lower bound. Suppose RP² = U ∪ V with U, V open and each homeomorphic to an open subset of R². Then U and V are orientable.
  - By naturality ([V] Wikipedia "Stiefel–Whitney class": "w(f*E) = f*w(E)") w₁(RP²)|_U = w₁(TU) = 0, since "The first Stiefel–Whitney class is zero if and only if the bundle is orientable" [V same page]. Likewise for V.
  - By the long exact sequence of the pair, w₁ lifts to classes a ∈ H¹(RP², U; Z2) and b ∈ H¹(RP², V; Z2).
  - [V] Hatcher, *Algebraic Topology* §3.2, p.209–210, gives the relative cup product "H^k(X, A; R) × H^ℓ(X, B; R) → H^{k+ℓ}(X, A ∪ B; R) when A and B are open subsets of X". So w₁² = a ∪ b lies in the image of H²(RP², RP²) = 0, i.e. w₁² = 0.
  - But RP² is non-orientable, so w₁ ≠ 0. Since H¹(RP²;Z2) = Z2·α, w₁ = α, and α² ≠ 0 in H*(RP²;Z2) = Z2[α]/(α³) (Hatcher Thm 3.12; the same page states "X is RP² and α² = … in H²(RP²; Z2)"). Contradiction, so ≥ 3 charts are needed.
  - Upper bound: B(RP²) = 3 ball charts. Hence **G(RP²) = 3**.
  - Same argument for any closed manifold: G(M) ≥ 3 whenever w₁(M)² ≠ 0.
- **Lower bound for every closed manifold:** 2, by invariance of domain [V] (Wikipedia "Invariance of domain"). A compact n-manifold is not homeomorphic to an open subset of Rⁿ.
- **Summary G(M), general-chart minimum:** S² 2 · T² 2 · Klein 2 · RP³ ≅ SO(3) 2 · RP² 3.
- Literature: see §1.5.

### 1.5 General-chart literature
- **[V] Gómez-Larrañaga, González-Acuña, Hoste, "Minimal atlases on 3-manifolds"**, *Math. Proc. Camb. Phil. Soc.* 109(1) (1991) 105–115, doi:10.1017/S0305004100069590.
  - Abstract (Cambridge page, read via fetch and via nurse): "we investigate the minimal number of charts (spaces homeomorphic to open subsets of ℝ3) needed to cover a closed 3-manifold M3. We prove that this number is two if the Bockstein βω1(M3)∈H2(M3;ℤ) of the first Stiefel-Whitney class of M3 is zero and three if it is not."
  - Listed equivalent condition: "the condition of being covered by two orientable subspaces".
  - Full text not read.
  - **Consequence for RP³ ≅ SO(3):** RP³ is orientable ([V] Wikipedia "Real projective space": "RP^n is orientable if and only if n is odd"), so w₁ = 0, βw₁ = 0, and the chart number is **2**. This is a citation for G(RP³) = 2, independent of the solid-torus construction in §1.3.
- **Surfaces (T², Klein, RP²):** no theorem on minimal covers by open subsets of R² located (nurse search across MathOverflow, Math.SE, Clapp–Puppe A-category papers, Cavicchioli–Grasselli 1985 — ball atlases only).
  - G(T²) = 2 and G(Klein) = 2 rest on the explicit constructions in §1.3.
  - G(RP²) = 3 rests on the w₁² proof in §1.3, which is the 2-dimensional analogue of GGH's "covered by two orientable subspaces" criterion.
- **Related, not about open subsets of Rⁿ:**
  - Clapp–Puppe A-category, via Gómez-Larrañaga–González-Acuña–Heil, *Math. Z.* 259 (2008): "The A-category catA (M) of M is the smallest number of sets, open and A-contractible needed to cover M."
  - Chekanov–van Koert–Schlenk 0807.3047 (contact charts).
  - Cavicchioli–Grasselli, Cahiers 26 (1985) (PL ball atlases).
- **Zhou et al. 1812.07035 [V]** is about a *single* global map: "for 3D rotations, all representations are discontinuous in the real Euclidean spaces of four or fewer dimensions", and "the 3D rotations have continuous representations in 5D and 6D". It is consistent with 2 local charts.

### 1.4 Z2 Betti numbers **[V/D]**
Recomputed from the triangulations in `ref/triangulations.json`:
- S² (1,0,1)
- T² (1,2,1)
- Klein (1,2,1) Z2, and **(1,1,0) over Q**
- RP² (1,1,1) Z2, and (1,0,0) over Q
- RP³ (1,1,1,1) Z2, and (1,0,0,1) over Q

**Z2 Betti numbers do not distinguish T² from the Klein bottle. Q Betti numbers do.**

## 2. Minimal good covers (covering type ct) — exact citations

ct(X) is the minimum size of a good cover (Karoubi & Weibel). **[V]** Borghini–Minian 1712.02833 abstract/§1: "if X has the homotopy type of a finite CW-complex, ct(X) coincides with the minimum possible number of vertices of a simplicial complex homotopy equivalent to X (see Lemma 2.1 below)."

| M | ct(M) | Source **[V]** |
|---|---|---|
| S² | **4** | Karoubi & Weibel, *On the covering type of a space*, arXiv 1612.00532, Example 3.2: "The sphere S^m has ct(S^m) = m+2" |
| T² | **7** | KW Theorem 5.3: "The covering type of the torus T is exactly 7." |
| RP² | **6** | KW Theorem 5.4: "The covering type of RP2 is 6." |
| Klein (N₂) | **8** | Borghini & Minian, *The covering type of closed surfaces and minimal triangulations*, arXiv 1712.02833, Prop. 3.2 / text: "ct(N2) = δ(N2) = 8". KW Cor. 5.6 had left it as 7 or 8. |
| RP³ | **10 ≤ ct ≤ 11**, exact value **[U]** | Lower: Govc, Marzantowicz, Pavešić, *Estimates of covering type and the number of vertices of minimal triangulations*, arXiv 1710.03333, Cor. 3.6, ct(RPⁿ) ≥ ½(n+1)(n+2) = 10 for n=3. Upper: Walkup's 11-vertex triangulation (§4), plus KW §2: "the open stars of the vertices form a good cover". KW Example 7.3's upper bound ct(RPᵐ) ≤ 2m+3 (= 9) is stated by GMP to need correction: "even to correct the upper estimate of ct(RP m) in [9, Example 7.3.]". |
| R² (neutral bed) | 1 | contractible ⇒ ct = 1 (KW: "ct(X) = 1 exactly when X is contractible") |

## 3. Consequence table — smallest K at which a nerve CAN have correct Betti numbers

A nerve on K charts is a simplicial complex on ≤ K vertices. Three thresholds follow:
- **(a) Betti-only:** the fewest vertices of any simplicial complex with exactly the target Z2 Betti numbers. **[D]** Exhaustive enumeration of every simplicial complex on exactly 3, 4 and 5 labelled vertices (9, 114 and 6,894 complexes, matching OEIS A307249 "Number of simplicial complexes with n nodes") gives the minima below. RP³ is not achievable with ≤ 5 vertices; the explicit 6-vertex complex is ∂Δ⁴ on {1..5} + triangles {612, 613, 623} + edge {64}, Z2 Betti (1,1,1,1) computed.
- **(b) Homotopy-correct:** a nerve homotopy-equivalent to M needs K ≥ ct(M) [V, Borghini–Minian Lemma 2.1 statement above]. **For K below ct(M), any nerve with correct Betti numbers is not homotopy-equivalent to M.**
- **(c) Dimension:** β_n ≠ 0 needs n-simplices (see §5).

| M | target Z2 β | ball atlas B(M) | general-chart atlas | (a) Betti-only min K | (b) ct(M) = homotopy-correct min K | (c) min simplex size (top-m) |
|---|---|---|---|---|---|---|
| S² | (1,0,1) | 2 | 2 | **4** (∂Δ³) | **4** | m ≥ 3 |
| T² | (1,2,1) | 3 | 2 | **5** (∂Δ³ + vertex 5 joined to 1,2,3) | **7** | m ≥ 3 |
| Klein | (1,2,1) Z2 | 3 | 2 | **5** (same complex; Z2 cannot separate it from T²) | **8** | m ≥ 3 |
| RP² | (1,1,1) | 3 | [U] 2 or 3 | **5** (∂Δ³ + vertex 5 joined to 1,2) | **6** | m ≥ 3 |
| RP³ ≅ SO(3) | (1,1,1,1) | 4 | 2 | **6** | **10 or 11** | m ≥ 4 |
| R² | (1,0,0) | 1 | 1 | 1 | 1 | m ≥ 1 |

**Arithmetic consequence:** in every row except R², both (a) and (b) are strictly larger than the ball-atlas and general-atlas counts:
- S²: 2 < 4
- T² and Klein: 3 < 5
- RP²: 3 < 5
- RP³: 4 < 6

So no nerve on K = atlas(M) charts has the target Z2 Betti numbers. P-COUNT (K = atlas) and P-BETTI (correct Betti) cannot both hold at the same K.

## 4. Vertex-minimal triangulations
All in `ref/triangulations.json`: facet lists, f-vectors, and Z2 and Q Betti numbers recomputed. Each is a closed pseudomanifold whose vertex links have sphere homology (checked).
- S²: ∂Δ³, 4 vertices; octahedron, 6 vertices (Lutz `manifold_lex_d2_n6_o1_g0_#2`, all degrees 4).
- T²: 7 vertices, Möbius–Császár (Lutz `manifold_lex_d2_n7_o1_g1_#1`, unique).
- RP²: 6 vertices (Lutz `…_n6_o0_g1_#1`, unique).
- Klein: 8 vertices (Lutz `…_n8_o0_g2_#1` of 6). Lutz's table has none with 7 vertices.
- RP³: Walkup, 11 vertices, f = (11,51,80,40) (Lutz file `RP3`).
  - **[V]** Lutz, arXiv math/0604018: "By Walkup's theorem, vertex-minimal triangulations of S2×S1, S2×~S1, and RP3 have 9, 10, and 11 vertices, respectively."
  - Walkup, *Acta Math.* 125 (1970) 75–107.
  - Lutz's 3-manifold table lists RP³ "--" for n ≤ 10 and 30 triangulations at n = 11.
- Source: F. H. Lutz, The Manifold Page, https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/ (surface counts table: S² first at n=4; RP² at n=6 (1); T² at n=7 (1); Klein at n=8 (6)).

## 5. Nerve from top-m co-selection — dimension bound

**Claim [D, immediate from the nerve definition].** Let the observer nerve N have vertex set {1..K}. A set σ is a simplex iff σ ⊆ S(x) for some input x, where S(x) = {k : k ∈ top-m(x), w_k(x) ≥ τ}. Then dim N ≤ m − 1, so H_k(N; F) = 0 for all k ≥ m and every field F.

**Proof.** Every simplex is contained in some S(x), and |S(x)| ≤ m, so every simplex has ≤ m vertices, i.e. dimension ≤ m−1. The chain groups C_k(N) vanish for k ≥ m, hence so does H_k. ∎

Cover-theory form [V]: Hatcher §3.3 defines the nerve as "the simplicial complex with n simplices the nonempty (n + 1) fold intersections of sets in the cover". A point lying in at most m sets bounds the dimension. Pairwise overlaps that share no common point create no higher simplices.

Consequences:
- **m = 2 ⇒ N is a graph ⇒ β₂ = β₃ = 0 for every input distribution.** S², T², Klein and RP² need m ≥ 3; RP³ needs m ≥ 4.
- **Threshold form [D]:** if the weights are a partition of unity (Σ_k w_k = 1, w ≥ 0, e.g. softmax or renormalised top-m), then |{k : w_k(x) ≥ τ}| ≤ ⌊1/τ⌋. So dim N ≤ ⌊1/τ⌋ − 1:
  - β₂ ≠ 0 requires τ ≤ 1/3
  - β₃ ≠ 0 requires τ ≤ 1/4
- **Softmax limit [D]:** softmax weights are strictly positive. Over all K with τ → 0⁺, every subset is co-selected on every input, so N = full simplex Δ^{K−1}, which is contractible (Betti (1,0,…,0)). Topology in a τ-filtration can therefore only appear at intermediate τ.

### 5.1 Soft partition-of-unity nerves — literature
- **[V]** Hatcher Cor. 4G.3: "If U is an open cover of a paracompact space X such that every nonempty intersection of finitely many sets in U is contractible, then X is homotopy equivalent to the nerve NU." The proof (Prop. 4G.2) uses a partition of unity subordinate to the cover to build the map. Here the PoU is a tool, and the nerve is of the open supports.
- **[V]** Hatcher Lemma 2C.2: "For vertices v1, …, vn of a simplicial complex X, the intersection st v1 ∩ ··· ∩ st vn is empty unless v1, …, vn are the vertices of a simplex σ of X, in which case st v1 ∩ ··· ∩ st vn = st σ." Hence every simplicial complex is the nerve of its open-vertex-star cover. KW §2 adds that "the open stars of the vertices form a good cover".
- **Closed sets {w_k ≥ τ}.** [V via nurse; BKRR quotes read from PDF] Bauer, Kerber, Roll, Rolle, *A unified view on the functorial nerve theorem and its variations*, arXiv 2203.03571:
  - Closed covers need extra hypotheses: locally finite, plus the homotopy-extension property for latching pairs (Thm D 1(b)), or closed convex sets in R^d (Thm 3.9), or subcomplexes (Thm 4.8).
  - Counterexamples are given: double comb; Warsaw-circle cover of S² with nerve H₂ = 0 while H₂(S²)=Z.
- **Filtered covers:**
  - Chazal–Oudot arXiv 0712.2638 needs good (convex-ball) covers at every scale plus a commuting-homotopy argument.
  - Dey–Mémoli–Wang *Multiscale Mapper* arXiv 1504.03763 takes the nerve of a tower of covers. It does not require good covers; stability is via interleaving with (c,s)-good towers (Thm 7.7).
- **Soft covers:**
  - Oulhaj–Carrière–Michel *Differentiable Mapper* arXiv 2402.12854 ("Soft Mapper") treats soft memberships as Bernoulli parameters; a Mapper graph (dim ≤ 1) is the nerve of a *sampled* cover. Simplices are not thresholded from weights.
  - UMAP arXiv 1802.03426 uses fuzzy simplicial sets with superlevel sets "the set of all elements with membership strength at least a". Only the 1-skeleton is used.
  - F-mapper (Bui et al., Knowl.-Based Syst. 189:105107, 2020) uses fuzzy clustering [U, not read].
- **Not found:**
  - a nerve theorem for superlevel sets {w_k ≥ τ} of a learned partition of unity;
  - any paper reading Betti numbers from an MoE/attention router's weights;
  - Björner Handbook Thm 10.6/10.7 and Borsuk 1948 text (not read).

## 6. MoP-JEPA citable numbers (arXiv 2607.05238)

Version facts **[V]**:
- v1 (6 Jul 2026) and v2: "MoP-JEPA: Hard-Assigned Predictor Mixtures for Stochastic JEPA World Models", 9-page PDF.
- v3 (3 Aug 2026): "Branch-JEPA: Finite-Support Predictive Distributions for JEPA World Models", 21 pp. Primary results are on Argoverse 2; OGBench moved to Table 3 and supplement S10.

### 6.1 planAll — predicted-graph BFS searchability, **not closed-loop**
v3 S10 definition: "planAll is the success rate on up to 300 sampled, truly reachable start–goal queries under predicted-graph BFS". Footnote: "Graph/readout and raw-coverage diagnostics only--not closed-loop performance."

| arm | v1 Table 2 (5 seeds, mean±std) med / tele / large | v3 Table S10(a) (%, mean ± sample SD) med / tele / large |
|---|---|---|
| dense / Standard JEPA | .084±.03 / .037±.01 / .016±.01 | 8.4±3.0 / 3.7±1.0 / 1.6±0.7 |
| M3-JEPA / fused MoE | .075±.02 / .036±.01 / .031±.01 | 7.5±1.7 / 3.6±0.9 / 3.1±1.2 |
| Var-JEPA / variational | .085±.01 / .035±.01 / .033±.00 | 8.5±1.6 / 3.5±1.6 / 3.3±0.4 |
| MDN | .566±.20 / .512±.12 / .717±.12 | 56.6±22.6 / 51.2±13.4 / 71.7±13.7 |
| **MoP-JEPA / Branch-JEPA** | **.851±.08** / .748±.09 / .386±.45 | **85.1±8.7** / 74.8±9.6 / 38.6±50.0 |
| codebook control | .267±.02 / .167±.02 / .149±.04 | 26.7±2.7 / 16.7±1.9 / 14.9±4.0 |

- The "0.85" is MoP on pointmaze-medium-stitch planAll.
- The "0.02–0.09" is the dense / M3 / Var range, .016–.085.
- The means are the same in v1 and v3; the dispersion figures differ (v3 says sample SD; v1 unlabelled).

### 6.2 Verified route (realroute) — still model-graph diagnostic
| arm | v1 Table 3 (5 seeds) med / tele / large | v3 Table 3 / S10(b) "valid2" (%) med / tele / large |
|---|---|---|
| dense / Standard | .083 / .035 / .016 | 8.3 / 3.5 / 1.6 |
| MDN | .134 / .039 / .057 | 13.4 / 3.9 / 5.7 |
| MoP / Branch | .247 / .192 / .139 | 24.7 / 19.2 / 13.9 |

- Transition precision: v1 "0.14–0.21 (ours 0.42–0.56)"; v3 Standard 62–90%, MDN 14–21%, Branch 49–56%.
- v1 matched K=16, 60k steps, large-stitch: MoP .208 vs MDN .095.

### 6.3 Closed-loop execution (real environment)
- **v1 Table 4**, "official info['success'], our replanning executor", med / large:
  - dense .27 / .13
  - MDN .87 / .27
  - MoP .67 / .33
  - M3, Var "----"
  - External anchor (quoted, not re-run by the authors), large-stitch: QRL 84, MoP 33, GCIQL 31, HIQL 13, GCIVL 12, GCBC 7, CRL 0.
- **v3 S10:** "A separate three-seed execution control is inconclusive: medium-stitch replanning is 86.7±11.5/66.7±11.5% for MDN/Branch-JEPA (mean±sample SD); large-stitch and a one-seed DINO-WM pilot are also inconclusive."
- v3 does not restate the v1 Table 4 / QRL anchor (grep for "QRL", "GCIQL" in v3 text: no match).

### 6.4 Protocol
- **[V] v1:** pointmaze-medium-stitch, pointmaze-large-stitch, pointmaze-teleport-navigate, antmaze-teleport (29-D); coordinate features; 20% of unique transitions held out; 5 seeds planning, 3 coverage and execution; EMA τ=0.996; router gate π_k > 0.5/K.
- **[V] v3 S10:** K=8, 30k updates, 512-d latents, edge if cosine > .5, all heads retained.
- **[U]:** encoder architecture, parameter count, batch size, learning rate, and public code (none located).

## 7. Limits
- General-chart atlas counts (§1.3) and Betti-only minima (§3a) are derivations, not citations. §3a is exhaustive only up to 5 vertices; the 6-vertex RP³ value rests on the explicit complex plus the 5-vertex exhaustion.
- ct(RP³) is known here only to lie in {10, 11}.
- The Klein bottle value 8 rests on Borghini–Minian (text read via pdftotext). KW alone gives 7 or 8.
- Soft-cover literature in §5.1 came partly via a sonnet nurse. Hatcher quotes were re-checked against the PDF text; BKRR, Chazal–Oudot, Dey–Mémoli–Wang and Oulhaj et al. quotes were not re-checked by Wilson.

## 8. Berry phase and Hopf bundle (bound by `tests/test_facts.py`, `ref/berry_check.py`)
- **Spin-½ Berry phase.** γ = −Ω/2, under:
  - γ = i∮⟨ψ|dψ⟩, discretised as −arg ∏⟨ψ_j|ψ_{j+1}⟩;
  - loop counter-clockwise seen from outside;
  - ψ = (cos θ/2, e^{iφ} sin θ/2).
  - The triangle Bargmann phase equals −Ω_tri/2 exactly; Ω_tri is from Van Oosterom–Strackee.
- **Lattice Chern number** (Fukui–Hatsugai–Suzuki, arXiv cond-mat/0503172), on outward icospheres at levels 2–4:
  - spin-½: −1, i.e. |c₁| = 1;
  - spin-1: −2;
  - constant state: 0.
- **Sources.**
  - [V] Wikipedia "Berry connection and curvature": half the solid angle; Chern number unity (magnitude).
  - [V] Wikipedia "Tautological bundle": "Its Chern class is −H".
  - [V] Wikipedia "Hopf fibration": equivalent to the Dirac monopole bundle.
  - [K] Berry 1984, γ_m = −mΩ (not fetched).

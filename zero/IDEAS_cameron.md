# IDEAS: positioning angles and missing comparisons (Cameron, 2026-10-04)

These items cover positioning and missing comparisons for the paper and the prereg; they are not results.

Source tags follow PROMPT.md L-OWNER:
- **[V]**: located this session, by search hit or by fetching the page.
- **[K]**: known, not fetched; fetch before citing.

Each item gives what the source owns, then what Asmodeous should do about it.

## A. Owners the paper must answer (new since T1)

1. **Autoencoder atlases already read characteristic classes.**
   - Source: Paluzo-Hidalgo & Ike, *Learning Tangent Bundles and Characteristic Classes with Autoencoder Atlases*, arXiv [2602.22873](https://arxiv.org/abs/2602.22873) (Feb 2026) [V]. It computes the first Stiefel–Whitney class (orientability) from the signs of the Jacobians of learned transition maps, on orientable and non-orientable manifolds.
   - Action: this owns "topology read from a learned atlas" for w₁, the class that separates torus from Klein bottle.
     - Asmodeous's holonomy-sign readout (TO7, `sign_holonomy`: Klein −1, torus +1) has to be compared against this owned readout on the same learned charts. Their test is the sign of det J of φ_j⁻¹∘φ_k on overlaps, and it costs about 20 lines on top of `chart_ae.py`.
     - The delta that survives: the readout comes from a *predictive*, *lazily evaluated* atlas whose cover is chosen online by the observer. They read it post hoc from a reconstruction atlas.
     - Add the paper to T1 and to the related-work table.

2. **Mixture-of-VAE charts for manifolds of arbitrary topology.**
   - Source: Alberti et al., *Manifold Learning by Mixture Models of VAEs for Inverse Problems*, arXiv [2303.15244](https://arxiv.org/abs/2303.15244) [V]. Each encoder–decoder pair is one chart, trained by maximum likelihood.
   - Action: a second owner of separate charts with membership. Cite it beside Chart-AE; no new arm is needed, because `chart_ae.py` covers the class.
   - Also cite *Mixture of Local Normalizing Flows as Chart Maps*, where a VQ-AE learns the atlas [V, search hit; fetch the id before citing].

3. **Learned connections on a cover: sheaf neural networks.**
   - Source: Bodnar et al., *Neural Sheaf Diffusion*, NeurIPS 2022, arXiv [2202.04579](https://arxiv.org/abs/2202.04579) [V]. It learns restriction maps (a discrete connection) on a graph by SGD. Barbero et al., *Sheaf NNs with Connection Laplacians*, [2206.08702](https://arxiv.org/abs/2206.08702) [V], aligns tangent spaces of neighbours.
   - Action: the nerve plus the transition phases χ_jk of PHASE-SPHERE *is* a learned U(1) cellular sheaf on the observer's nerve. Saying so:
     - gives reviewers a familiar frame;
     - makes holonomy and Chern "sheaf cohomology of a learned cover", which is the GDL vocabulary;
     - marks the novelty precisely: the cover itself is learned and lazily evaluated, where a sheaf NN is handed its graph.
   - Missing comparison: a sheaf-diffusion readout on the *given* kNN graph of a single JEPA's latent. This is the L-GIVEN control for TO1/TO7, the analogue of Mapper for C2.

## B. Stronger baselines nobody has costed yet

4. **Soft MoE is the strongest fixed-compute router.**
   - Source: Puigcerver et al., *From Sparse to Soft Mixtures of Experts*, arXiv [2308.00951](https://arxiv.org/abs/2308.00951) [V]. It reports that soft MoE beats sparse MoE at fixed compute.
   - Action: T3's "lazy within 1% of soft" bar compares against the atlas's own soft mixture. Reviewers will ask for Soft-MoE slots instead: each slot is a learned convex combination of inputs, so compute is fixed and every chart gets gradient.
   - It is about one Heads call over S slots, which makes it cheap to add as a T3 arm at equal active FLOPs. If it beats lazy top-m, C3's laziness argument shrinks to "same quality, sparser compute".

5. **Expert-choice routing as the load-balance control.**
   - Source: *Routers in Vision MoE: An Empirical Study*, arXiv [2401.15969](https://arxiv.org/abs/2401.15969) [V]. Expert-choice beats token-choice there.
   - Action: router death is the atlas's known failure mode; LAM_BAL exists for it.
     - An expert-choice variant (each chart picks its top tokens) removes load balancing by construction, and is a one-line `topk` transpose in `route()`.
     - Run it as an ablation, so the T2 verdict cannot be a load-balancing artefact.

6. **The owned geometric baseline for S²: a hyperspherical latent.**
   - Source: Davidson et al., *Hyperspherical VAE*, arXiv [1804.00891](https://arxiv.org/abs/1804.00891) [V]. A von Mises–Fisher latent on S^{d−1}.
   - Action: the C1 kill uses a single JEPA at d = 2n+2. A reviewer will ask why not a single chart that *is* the sphere.
     - MoP-JEPA already normalises to the unit sphere, so it is half of this baseline.
     - Add a single JEPA with unit-norm z, d = n+1 (S² ⊂ R³), as the "right topology by hand" ceiling on the S² bed. Score it "representable only" per L-GIVEN, since the sphere is handed in.
     - It shows how much of the atlas's win is topology versus capacity.

7. **A discrete-codebook world model as the mode-collapse owner.**
   - Source: *Discrete Codebook World Models for Continuous Control*, arXiv [2503.00653](https://arxiv.org/abs/2503.00653) [V]. DreamerV3 ([2301.04104](https://arxiv.org/abs/2301.04104) [V]) uses one-hot categorical latents for multimodal transitions.
   - Action: T4b's stochastic-branching story (MoP versus conditional-mean collapse) has a large owner outside JEPA. Categorical latents handle branching by construction.
     - A DreamerV3-style categorical-latent predictor at matched FLOPs belongs beside MoP-JEPA on pointmaze-stitch.
     - Without it, "beats MoP-JEPA" reads as "beats one JEPA variant".

## C. Framing that lands with ML reviewers

8. **The CTC observer is a deep equilibrium layer.**
   - Source: Bai, Kolter, Koltun, *Deep Equilibrium Models*, arXiv [1909.01377](https://arxiv.org/abs/1909.01377) [V]: implicit differentiation through z* = f(z*).
   - Action: w* = D(P_h(z, w*)) is a DEQ over the router simplex.
     - Training through the fixed point with implicit gradients (constant memory) is the owned way. Use it, or say why the amortised head is better.
     - The amortised "halt before start" head (TO3) is then a learned DEQ initialiser, which has an owner line too [K: Bai et al. 2021, "Neural Deep Equilibrium Solvers"].
     - Positioning: "a Deutsch-consistent observer, trained as a DEQ" makes the time-traveller defensible instead of exotic.

9. **Unitary dynamics have an owner, and Cayley is a choice.**
   - Sources:
     - Lezcano-Casado & Martínez-Rubio, *Cheap Orthogonal Constraints* (expRNN), arXiv [1901.08428](https://arxiv.org/abs/1901.08428) [V].
     - Helfrich et al., scaled-Cayley unitary RNN, [1811.04142](https://arxiv.org/abs/1811.04142) [V].
     - Arjovsky et al. 2016 uRNN [K].
   - Action: PHASE-SPHERE's Cayley-unitary step is in this family; cite it. Also note that Cayley cannot reach eigenvalue −1, while exp can.
     - That matters for Berry/holonomy readouts near antipodal states.
     - Ablation: Cayley versus matrix-exp parametrisation on the Berry bed.

10. **The Born rule is already a generative model.**
    - Source: Born machines (Cheng et al., tree tensor networks, PRB 99, 155131 (2019), arXiv [1901.02217](https://arxiv.org/abs/1901.02217) [V]).
    - Action: "p = |ψ|²" as a learning primitive is owned for density modelling.
      - The delta is the Born rule as a *router/decider over charts*, with the decoherence limit equal to softmax. That is test-bound: `test_decoherence_limit_is_softmax`.
      - The paper should say "Born decider", never "Born machine", and cite Cheng to show the distinction.

11. **Gauge equivariance is the right word for "charts own their coordinates".**
    - Sources:
      - Cohen, Weiler, Kicanaoglu, Welling, *Gauge Equivariant CNNs and the Icosahedral CNN*, arXiv [1902.04615](https://arxiv.org/abs/1902.04615) [V], on local frames on the sphere.
      - Bronstein et al., GDL protobook [2104.13478](https://arxiv.org/abs/2104.13478) [K].
    - Action: PHASE-SPHERE's gauge fixing z_k = V_k†ψ e^{−i arg α_k} makes g phase-invariant, and `test_global_phase_of_psi_does_not_change_g` passes.
      - Calling this a gauge-invariant readout places Asmodeous inside geometric deep learning, the venue-friendly frame.
      - Difference to state: gauge-equivariant CNNs are handed the manifold and its frames. Asmodeous learns the charts (L-GIVEN). This one sentence is the paper's best positioning line.

## D. Story and venue

12. **The Turing paradox has a laboratory anchor.**
    - Sources: Misra & Sudarshan 1977 (J. Math. Phys. 18, 756) [K]; Itano, Heinzen, Bollinger & Wineland 1990, Phys. Rev. A 41, 2295, the trapped-ion Zeno experiment [K].
    - Action: open the paper on the Itano experiment, not the thought experiment. A measured inhibition of a transition by repeated observation is a stronger first sentence.
      - The `identities.py` §1 column, cos^{2N}(π/2N) → 1, then connects Turing to the bench.
      - Fetch both before citing (L-OWNER).

13. **Event-driven compute: cite the field, not only A.E.T.H.E.R.**
    - Sources:
      - Gallego et al., event-based vision survey [K: arXiv 1904.08405].
      - Graves, Adaptive Computation Time [K: 1603.08983].
      - Mixture-of-Depths [2404.02258] (already in refs).
      - Bengio et al. 2013 conditional computation [K].
    - Action: time-laziness ("fire iff Δ ≥ ε") is event-camera logic. ACT and MoD are the learned-halting owners.
      - Missing experiment: on the swing-up bed, skip steps when ‖o_t − o_{t−1}‖ < ε and report the compute and success trade-off against a fixed-rate planner. It needs no new model, only a wrapper around `mpc_policy`.
      - This turns the A.E.T.H.E.R. section from analogy into a measured axis.

14. **What makes a "new architecture" paper land.**
    - Sources: NeurIPS reviewer guidelines ([2026](https://neurips.cc/Conferences/2026/ReviewerGuidelines) [V]). The most common rejections there are weak or unfair baselines, missing ablations, and missing reproducibility details. The guidelines also say novel insight from evaluation counts as originality.
    - Action, already mostly built: L-BR flop- and param-matched pairs, the 3-seed rule, PREREG before results. Three gaps remain:
      - (i) one non-toy bed per claim. OGBench and swing-up exist; T2 is all synthetic lifts, so add the 64×64 render variant to at least one manifold.
      - (ii) a wall-clock column beside FLOPs in every table. T7 has this; T2/T3 do not.
      - (iii) an ablation per atlas part: backbone, gate, keys, gluing, laziness. Lazy chart-AE already ablates backbone, gate and keys together; split them.
    - Also: if a candidate loses a claim, report the loss in the paper. A per-claim winner table with honest losses is the kind of evaluation insight the guidelines credit.

## Priority (what to build next, cheapest kill first)

| # | Item | Cost | What it can change |
|---|---|---|---|
| 1 | w₁ from transition-map Jacobians on chart-AE (item 1) | ~20 lines | TO7 holonomy claim |
| 2 | unit-norm single JEPA, d = n+1, on S² (item 6) | flag on `SingleJEPA` | C1 on S² |
| 3 | Soft-MoE slots arm in T3 (item 4) | one arm | C3 |
| 4 | DreamerV3-style categorical predictor in T4b (item 7) | one arm | C4/T4b |
| 5 | event-skip wrapper on swing-up (item 13) | wrapper | a new measured axis |

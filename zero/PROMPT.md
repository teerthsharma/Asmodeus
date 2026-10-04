# Asmodeous ० — AN ATLAS OF TINY JEPAs, MADE REAL BY AN OBSERVER

Author: Seal (Teerth Sharma). Date: 2026-10-04 (rev. 2). New project, new grounds. One lesson carries over: L-GIVEN.
Files: `PROMPT.md` (this file); `ref/identities.py`, `ref/identities.out` (pinned reference; §3 sinks and §6 IIA load-bearing).
Quantum mechanics is not the build; it survives only as abstraction (§2) and optional arm T6.

Closest owner: MoP-JEPA (Song et al., arXiv 2607.05238, July 2026) [V]: context-only softmax router over K JEPA predictor heads, shared encoder, hard winner-take-all; fixes mode-averaging collapse; OGBench up to 0.85 success vs 0.02–0.09 single predictor.
Delta must live in four things MoP-JEPA lacks: (1) separate chart (encoder + embedding space) per JEPA; (2) lazy evaluation (embedding exists only when observed); (3) gluing charts into a higher-dim space; (4) topology readout from the observer.
Author's prior that the combination beats owned arms on a neutral bed: ~15–20% (guess).

## 0. Idea
Many tiny JEPAs, each with its own small chart, in a fragmented geometry. A scalable softmax observer chooses which to read. Only chosen ones compute ("observed → exists"). Observer weights glue chosen charts into one higher-dim space.
Chain: Turing paradox (observer) → softmax attention is a classical observer → body example (intent picks goal, body geometry computes path; equilibrium-point hypothesis Feldman 1966; Adams, Shipp & Friston 2013) → observer = query, body = tiny JEPAs, motion = prediction in chosen chart → softmax over charts = partition of unity → co-observed charts form the nerve; its homology = learned topology → no single chart covers S², T², SO(3) without tearing.

## 1. Laws
- L-GIVEN: any structure a bed supplies (charts, routing, gluing, manifolds, graphs) must be learned by SGD. Hand-given structure scores "representable only", never "works".
- L-PREREG: beds, bars, kills go into `PREREG.md`, committed before any result file exists. Later = post-hoc label.
- L-BR: baselines = best owned methods at equal active FLOPs and equal total params, SGD-trained, tuned on a separate seed.
- L-WALD: a designed bed counts only if a neutral bed runs beside it.
- L-LEARN: an arm must clear its in-distribution bar before being compared.
- L-OWNER: fetch before cite. T1 owner hunt runs before any bar. Tags [V] located, [K] known not fetched, [U] uncertain.
- Gap rule: a literature gap is not a new primitive; novelty earned only by a decisive test owned parts fail.
- L-REPRO/L-SHA: run `ref/identities.py` first; numbers must match `ref/identities.out` (numpy 2.4.4). SHA-256 of identities.py: `467c2ae25f92121093a60435c3397bd5f1e9fe246a2f602b4fc03bbace40a3fe`. No untracked code produces a score.
- L-AUDIT: committed records read-only.
- Security: no AGI claims; HF/Zenodo uploads only on Seal's explicit word; no AI co-author/attribution lines anywhere; no lane kills a GPU process by name; deciding numbers from the local RTX 4060, Kaggle reproduces; Kaggle token stays local; private slugs gitignored + test asserts it before every commit; autopilot never edits code, tables, contracts or deciding-cell design (Tier 3 HALT).

## 2. Words decoded
- "observer observes → thing computes": demand-driven lazy computation; contextuality kept as abstraction. Owners: lazy eval, Mixture-of-Depths 2024 [K], neural fields [K].
- "hand moves": intent sets goal, local geometry computes path. Owners: Feldman; Adams/Shipp/Friston 2013 [K]; Riemannian motion policies (Ratliff 2018) [K]. Enters as chart predictor.
- "small JEPA with its own xy dims": tiny JEPA, own encoder, own d_c-dim chart. Owner: Chart Auto-Encoders 1912.10094 [V] (AEs, not JEPAs). Atlas of JEPAs not found → T1.
- "higher-dim fragmentation": atlas glued into R^{D_g}. Owner: POUnets 2101.11256 [V] (regression, not world models).
- "JEPA collapses": (a) representation collapse — LeJEPA/SIGReg 2511.08544 [V]; (b) conditional-mean collapse — MoP-JEPA [V]. "Observer prevents collapse" = hypothesis (T4).
- "scalable softmax attention choosing its JEPA": router. Owners: MoE (Shazeer 2017) [K]; PEER 2407.04153 [V]; RIMs 1909.10893 [V]; MoP-JEPA [V].
- "time traveller": observer reads charts' predicted futures before committing (CTC start/halt loop). Owners: V-JEPA 2-AC 2506.09985 [V]; MoP-JEPA. Optional T5.
- "computed when observed … many tiny JEPA scale": lazy top-m, compute ∝ m, params ∝ K. Owners: PEER, MoE. Test T3.
- "topologically higher dimension": glued space + nerve; nerve Betti numbers = learned topology. Owners: nerve theorem [K], Mapper (Singh/Mémoli/Carlsson 2007) [K]. T2.
- quantum: removed from build; contextuality (order effects, QQ equality PNAS 2014 [V]) kept as optional T6.

## 3. Maths before code
- No single chart covers a closed manifold (invariance of domain). cat(M)+1 ≤ atlas(M) ≤ n+1 (LS category) [K]; Rudyak–Schlenk math/0605350 [V]. Sⁿ: 2 charts; T² and every closed surface except S²: 3; SO(3) ≅ RP³: 4. P-COUNT: atlas stops tearing at K = atlas(M), not before.
- SO(3): every Euclidean rep in ≤4 dims is discontinuous; 5-D/6-D continuous (Zhou et al. CVPR 2019, 1812.07035) [V].
- Whitney caveat: single JEPA with d ≥ 2n embeds any n-manifold. Atlas must win on (i) compute/query, (ii) sample efficiency, (iii) branch/mode handling, or (iv) topology readout single chart + post-hoc Mapper can't match.
- Collapse: every SSL term here minimized by a constant. Anti-collapse (SIGReg or EMA+variance) mandatory in every arm. "Observer alone prevents collapse" tested only by ablation (T4).
- Softmax observer = learned partition of unity (POUnets own it for regression).
- From identities.out: observer must be able to observe nothing (causal softmax stacks collapse onto token 1: mean mass 0.2543 → 1.0000 at L=1→16) → gated observer (gated attention, NeurIPS 2025) [V]. Weights obey IIA (§6): near-duplicate charts split weight; T2 measures whether duplicates emerge.

## 4. Architecture: OBSERVER-ATLAS (every parameter learned by SGD)
1. Backbone: small causal model (tiny transformer or GRU) → context c_t.
2. K tiny JEPAs. JEPA k: encoder E_k: o_t → z_k ∈ R^{d_c} (d_c = 2–4); predictor P_k: (z_k, a_t) → ẑ_k next step, same chart; target encoder Ē_k (stop-grad or EMA); gluing map φ_k: R^{d_c} → R^{D_g}, D_g > d_c.
3. Observer: q_t = W_q c_t; one learned key κ_k per JEPA (product keys once K ≥ 10³); w = softmax(q·κ/√d), keep top-m (m ≪ K), renormalize; output gate so observer can observe nothing.
4. Lazy rule: only top-m JEPAs run E_k, P_k on o_t. Cost/step = routing (O(K·d) or O(√K·d)) + m tiny JEPAs.
5. Gluing: g_t = Σ_{k∈top-m} w_k φ_k(z_k); ĝ_{t+1} = Σ w_k φ_k(ẑ_k).
6. Losses: (1) per-chart JEPA ‖P_k(z_k,a_t) − sg(Ē_k(o_{t+1}))‖²; (2) overlap consistency ‖φ_j(z_j) − φ_k(z_k)‖² for co-selected j,k; (3) SIGReg per chart and on g; (4) router load balance; (5) task loss through g.
7. Time-traveller (optional): preselect top-M by static keys; roll predictors h steps; re-score by predicted future + value; keep top-m.
8. Topology readout: nerve N_τ, vertex per JEPA, simplex where charts co-selected with weight ≥ τ on some input; persistent homology over τ → Betti numbers; learned online, no extra pass.

## 5. Owners (T1 reads each)
MoP-JEPA 2607.05238; Chart AE 1912.10094; semi-supervised chart AE 2208.10570; POUnets 2101.11256; probabilistic POUnets 2107.03066; PEER 2407.04153; MoE [K]; RIMs 1909.10893; V-JEPA 2 2506.09985; I-JEPA 2023 [K]; LLM-JEPA 2509.14252; LeJEPA/SIGReg 2511.08544; Mapper 2007 [K]; nerve theorem [K]; Rudyak–Schlenk math/0605350; Zhou et al. 1812.07035.
Not found yet: the full combination.

## 6. Claims
- C1 geometry: on multi-chart manifolds, atlas continuous at K = atlas(M), d_c = n; single JEPA at equal active FLOPs needs d ≥ 2n.
- C2 topology: observer nerve recovers true Betti numbers online, ≥ Mapper post-hoc on best single-JEPA latent.
- C3 lazy scale: fixed active compute, loss falls as K grows, below dense JEPA and below single JEPA with PEER experts.
- C4 collapse: observer competition reduces representation collapse when anti-collapse weakened; atlas matches MoP-JEPA on mode collapse.
- C5 time traveller: reading predicted futures before choosing beats owned latent MPC.

## 7. Quantum (optional T6)
Kept: contextuality, order effects. Reference: Lüders max|QQ| 4.44e−16; classical+disturbance mean|QQ| 0.0751. Dropped: Hilbert spaces, unitaries, Born/Lüders decider.

## 8. Tests
- T0 reproduce: run identities.py, check SHA pin.
- T1 owner hunt (Wilson): read bodies of all §5 owners. Queries: "mixture of JEPA encoders"; "atlas world model"; "multi-chart latent world model"; "chart-based JEPA"; "partition of unity mixture of experts gluing"; "lazy mixture of experts world model"; "nerve of mixture-of-experts routing"; "Mapper learned cover neural". Record every query and hit. Full combination owned → measurement only. Part owned → cite, continue.
- T2 geometry (decisive; C1, C2). Manifolds: S² (spherical pendulum); T² (double-pendulum angles, MuJoCo); Klein bottle (synthetic); SO(3) (free rigid body, MuJoCo); neutral R² (one chart suffices). Observations: 64×64 renders or random nonlinear lift to D=256; manifold, charts, state never given. Arms at equal active FLOPs and total params: single JEPA d ∈ {n, 2n, 2n+2}; MoP-JEPA (K heads, shared encoder); chart-AE-style multi-chart encoder; OBSERVER-ATLAS K ∈ {1,2,3,4,8,16,64}, m ∈ {1,2}, d_c = n. Metrics: latent pred error h=1,10; linear-probe state recovery from g; discontinuity rate (fraction of state pairs within geodesic ε whose glued latents differ by > δ; ε, δ preregistered); nerve Betti over Z2; effective rank per chart; FLOPs/step.
  Predictions: P-COUNT discontinuity ≈ 0 first at K = 2 (S²), 3 (T²), 3 (Klein), 4 (SO(3)). P-BETTI Z2: S² (1,0,1); T² (1,2,1); Klein (1,2,1); SO(3)≅RP³ (1,1,1,1). Neutral: K=1 suffices, no gain.
  Bars: discontinuity ≤ 0.01 at K = atlas(M) on ≥3/4 manifolds over 3 seeds; Betti correct on ≥3/4; pred error ≤ best owned at equal active FLOPs; neutral error within +2% of single JEPA.
  Kills: single JEPA d=2n+2 matches error and discontinuity AND Mapper on its latent recovers same Betti; or MoP-JEPA ties every metric; or Betti wrong on ≥2 manifolds.
- T3 lazy scaling (C3): m=2; K ∈ {16,64,256,1024}, product keys from K=256. Arms: dense JEPA equal active FLOPs; single JEPA with PEER experts. Bars: loss monotone in K, beats dense from K=256; lazy top-m within 1% rel of soft mixture over all K. Kill: flat in K, or PEER-in-one-JEPA matches.
- T4 collapse: (a) SIGReg 0.1× and 0×, single vs atlas, track effective rank of z_k and g; kill: atlas collapses as fast. (b) OGBench pointmaze-stitch with MoP-JEPA protocol; bar success ≥ MoP-JEPA; kill below.
- T5 time traveller: MuJoCo double-pendulum swing-up (T²×R²) and OGBench pointmaze; CEM in glued space with §4.7 re-scoring; arms V-JEPA-2-AC-style single-JEPA MPC, MoP-JEPA, equal planning compute. Bar ≥ best owned + 0.05; kill at or below.
- T6 optional: query pairs both orders, order effect + QQ with bootstrap CIs. Measurement only.
- T7 systems: lazy routing + tiny JEPAs on 4060, µs/step vs dense JEPA of equal quality. Bar ≥1.5× faster.

## 9. Kills and priors
T1 finds full combo → measurement only. T2 killed → geometry claim dead; T3–T5 run only if their claims stand without C1, C2. Hand-given structure → "representable only".
Priors: T2 ~30%; T3 ~25%; T4a ~20%; T4b ~35%; T5 ~20%.

## 10. Deliverables, room, order
Deliverables: `PREREG.md` (first commit); `RESULTS.md`; `VERDICT.md` (first line YES/NO/VOID per T0–T7).
Room: Chase — OBSERVER-ATLAS + beds; Cameron — owned arms (single JEPA, MoP-JEPA, chart AE, PEER-JEPA, V-JEPA-2-AC-style MPC); Foreman — MuJoCo beds, runs, SHA pins, T7; Wilson — T1 + scoring; Inspector — prereg order, L-GIVEN audit, shortcut checks (no state leak into router), fair Mapper baseline; Dr House — one adversarial attempt on every claim in §3 and §6, before T5.
Order: T0 → T1 → T2 (decides) → T3 ∥ T4 → T5 → T7. T6 optional anytime.

## 11. Sources
Seal's notes "AT I ०" (2 pages), Gödel-universe CTC figure (start/halt, tipping light cones, closed curve), and clarifications of 2026-10-04 (see original chat).

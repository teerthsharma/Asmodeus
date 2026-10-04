# PREREG DRAFT — T4b (OGBench pointmaze, MoP-JEPA protocol)

Status: DRAFT, not preregistered. Nothing under `runs/` exists for T4b. Bed: `beds/ogbench_bed.py`. Controls: `arms/controls_t4b.py`. Adapters: `arms/t4b_adapters.py`. Runner: `run_t4b.py` (`--dry-run`). Tests in `tests/` (`test_ogbench_bed`, `test_controls_t4b`, `test_t4b_adapters`, `test_run_t4b`) run green in venv `~/.venvs/ogbench`.

## 1. Source and pins
- Paper: **arXiv 2607.05238v1** (MoP-JEPA, 6 Jul 2026). Cite v1 only: v3 (3 Aug 2026) is retitled Branch-JEPA, reports different campaigns, and calls its own execution check "inconclusive". v2 Table 4 shows large-stitch realroute .139 (K=8, 30k); v1 Table 4 shows .208 for the same cell because it substitutes the matched K=16/60k-step run. Quote .139 for the K=8 protocol.
- No public code (v1–v3 PDFs, abs pages, web search). The hyperparameters below are reconstructed from text.
- ogbench **1.2.1** (PyPI 2026-01-14; repo `seohongpark/ogbench` master `1d4140997f60c52c6fb0702ec100dc988b18c548`). The changelog shows no pointmaze env, dataset or eval change after 1.0.1. The 1.2.0 change affects singletask envs only.
- venv `~/.venvs/ogbench`: mujoco 3.14.0, gymnasium 1.2.0, numpy 2.4.6, torch 2.14.0+cu126. Installing ogbench system-wide would raise the system mujoco from 3.11 to 3.14 and move the T2 beds.
- Dataset SHA-256 values are pinned in `ogbench_bed.SHA256`, and the loader refuses a mismatch. The upstream downloader crashes on Windows (WinError 32, rename of an open file), so the bed fetches the files itself.

## 2. Beds
`pointmaze-medium-stitch-v0`, `pointmaze-large-stitch-v0`, `pointmaze-teleport-navigate-v0` (state, 2-D). The paper's 4th env, antmaze-teleport (29-D), is out of scope for the first pass.
- Stitch: 5000 episodes × 201 stored states (200 transitions). Navigate: 1000 × 1001. Val: 10% of that.
- obs = xy = state (`meta['obs_is_state']`). Here L-GIVEN reduces to "no cell id, no graph, no maze map reach any arm".
- Free space is a planar domain, so atlas = 1. betti_z2: medium (1,1), large (1,3), teleport (2,2), where the 2nd component is the isolated out-cell (1,7). T4b therefore tests branching and mode collapse only. The C1 geometry argument gives the atlas no edge here.

## 3. Protocol G — graph (where 0.851 comes from)
- Nodes are free maze cells (unit 4). A transition is a pair of consecutive distinct cells in a dataset episode. Measured successors per cell: 2.08 / 2.09 / 2.20, against the paper's "2.1–2.2", so the reconstruction matches.
- Holdout: 20% of unique directed edges, `round(0.2|E|)`, never a source cell's last edge. The split seed is the run seed.
- Arms see coordinates only and are action-marginalised (v3 S10). The training pair is (source-cell centre xy → successor-cell centre xy) with held-out edges excluded (`cell_transitions(context='cell')`).
- Hyperparameters: K=8, 30k updates, latent 512.
- Edge proposal, PRIMARY (coordinator decision): centre first. Subtract the arm's mean latent over the training points (`embed(ctx ∪ nxt)`) from embeddings and heads, then propose c→c′ when cos > .5 (`ogbench_bed.decode_edges(rule='centred')`). The same rule applies to every arm and control. A test checks that a constant latent offset leaves the edges unchanged. SECONDARY, reported beside it: v1's raw rule, c→c′ when cos(head_k(c), z(c′)) > .5 (`rule='paper'`). Primary readout uses all heads (as for planAll). Secondary readout gates heads on π_k > 0.5/K.
- Metrics (`graph_metrics`), over 300 truly-reachable sampled queries (`sample_queries(seed)`) plus the 5 official task queries:
  - **planAll**: BFS path in the proposal.
  - **realroute**: path after keeping only real edges.
  - **precision**: macro per-source fraction of real edges.
- Seeds: 5.

## 4. Protocol E — closed-loop execution
- Follows the official OGBench loop (`evaluate`). `reset(options=task_id)`, goal = `info['goal']`, actions clipped to [-1,1], 1000-step limit, terminate at goal, score = final-step `info['success']`.
- Volume: 5 tasks × 50 episodes per seed (the OGBench paper rule; the repo default is 20). Seeds: 5 (the paper used 3).
- Executor: `ogbench_bed.executor(name, edges, replan=True, patience=40)`. The paper's executor is unpublished, so this is a reconstruction. Each step it runs BFS over the proposed edges minus deleted ones and steers at full speed to the next cell centre. A teleport hop steers to its own centre. A hop not reached within 40 steps is deleted and the route is replanned. Mutation tests cover the replanning and the reset between episodes.

## 5. L-LEARN gate (our MoP-JEPA must reproduce before any comparison)
On medium-stitch, 5 seeds, Protocol G, all four must hold:
- (a) planAll ≥ 0.75 (paper .851±.08, within 0.10).
- (b) realroute in [0.15, 0.35] (paper .247±.08).
- (c) precision in [0.32, 0.66] (paper .42–.56).
- (d) our dense K=1 arm planAll ≤ 0.15 (paper .084). If dense does not collapse, the bed is not reproducing the phenomenon and T4b is VOID, not a pass.

Teleport and large are reported, not gated: large is seed-bimodal (.386±.45). Execution is reported, not gated: paper .67/.33 med/large, likely 5 episodes per seed. 66.7±11.5 over 3 seeds equals {0.6, 0.6, 0.8}, which is an inference.

## 6. T4b bar (proposal, both metrics as Wilson asked)
- **Primary: realroute, with precision ≥ MoP-JEPA precision − 0.05**, paired over the same 5 split seeds.
  - Bar: atlas realroute ≥ MoP-JEPA on ≥2/3 mazes.
  - Kill: below on ≥2/3 mazes.
  - Paired bootstrap 95% CI over seeds.
- Secondary: planAll, which is weak because hallucinated edges raise it (MDN case).
- Tertiary: execution.
- Mandatory controls, every run:
  - **train-edge memoriser** (`arms/controls_t4b.Memoriser`): proposes exactly the training edges.
  - **propose-all** (`ProposeAll`): realroute = planAll = 1.0, precision .086/.046/.050.
  - context-free codebook, latent space as in v1 (`t4b_adapters.CodebookGraph`, `codebook_latent`): MoP-JEPA with zeros fed to heads and router. A coordinate-space k-means version (`codebook_xy`) runs as a labelled extra.
  - shuffled-context test.
- An arm "works" only if realroute beats the memoriser and precision beats propose-all by ≥ 0.3.

## 7. Hidden variables found
1. **0.851 is planAll, not success.** It is BFS over the model's own proposal and is inflated by fake edges. MoP precision is .42–.56, so about half of its edges do not exist.
2. **The memoriser beats the paper's verified numbers** (our reconstruction, 5 split seeds):
   - A lookup of training edges, with no generalisation, scores planAll = realroute = **0.321±0.075 / 0.237±0.028 / 0.419±0.131** (med/large/teleport).
   - Paper MoP realroute is .247 / .139 / .192. All three are below the memoriser.
   - Either our query/holdout reconstruction differs from theirs, or the paper's realroute gain is over a baseline weaker than a table lookup. This must be settled before any bar is trusted.
3. **realroute is freeloadable.** Proposing every edge scores 1.0, so precision must be reported beside it. The paper never reports realroute for the codebook.
4. **Context granularity decides whether branching exists.** With the cell-centre context there are about 2.1 successors per context. With the exact exit coordinate the successor is nearly determined and dense would not collapse. The paper says only "coordinate features".
5. **Unclipped actions tunnel through walls.** `PointEnv.step` adds 0.2·a to qpos with no clip, and a mutation test confirms it. The official loop clips, so any CEM or planner path must clip as well.
6. **Global numpy RNG.** Init/goal noise (±1 unit) and the teleport destination come from `np.random`, not `env.np_random`. OGBench impls seed it once, so per-episode noise depends on how many episodes came before. `evaluate` seeds `np.random`, `env.reset(seed)` and `action_space.seed`, and a mutation test confirms this matters.
7. **Teleport trap.** Out-cell (1,7) is isolated, so 1/3 of teleports end the episode as a failure. The replanning executor on the true graph scores 0.493±0.002 on teleport against 1.000 on medium and large (3 seeds × 250). Teleport execution has a low ceiling that no world model can lift: it only just exceeds the published GCIVL score of 45, so it cannot separate arms.
8. **Version, seed and campaign mixing.**
   - v1 Table 4 mixes K=8/30k and K=16/60k cells.
   - v3 says planAll and realroute come from different checkpoint campaigns.
   - Planning uses 5 seeds and execution 3.
   - The large-stitch execution of .33 places "second of seven" only on large. On medium, GCIVL 70, QRL 80 and HIQL 74 all beat MoP's .67.
9. **Raw files store L+1 states per episode.** The last action has no successor. The bed zeros it, so T = 201, not 200.
10. **The cosine edge rule depends on latent geometry.** cos > .5 does not change with scale, but it does change with the latent's offset. Latents that are not normalised and not centred (single, atlas, chart-AE) put most points in one cone. After 2 training steps on medium, they proposed 246 / 287 / 580 of 650 possible edges, which raises planAll, while MoP and the latent codebook proposed 0. Decided: the centred rule is PRIMARY and v1's raw rule is SECONDARY. The precision floor stays.
11. **Unpublished.** Encoder and head widths, LR, batch size, λ_route, λ_bal, the edge-decoding latent (online or EMA encoder), the query sampling RNG, and the replanning executor.

## 8. Smoke (not results; PREREG not committed)
- Random uniform policy, 5×50 episodes, seed 0: medium **0.000**, large **0.000**, teleport **0.016**.
- Executor on the true graph, 3 seeds × 5 tasks × 50 episodes: medium 1.000, large 1.000, teleport 0.493±0.002 (per task .47/.53/.50/.47/.50).
- Executor on the true graph plus 10% fake edges (3 seeds × 250): replanning gives 1.000 / 1.000 (medium/large); without replanning, 0.133 / 0.200. The paper reports 0.40 → 1.00.

# PLAN — the plan of record (Dr House's revision of ACHIEVABLE.md, 2026-10-04)

Status: hypotheses until bound by RED tests. This file supersedes the row order of `ACHIEVABLE.md`.
Hard requirement (Seal): Asmodeous is categorised as AI and can TALK after training.

## Why row 9 of ACHIEVABLE.md was replaced
Every complete instrument on C^N obeys Σ_v max_ψ p(v|ψ) ≤ Σ_v tr(K_v†K_v) = N.
With a Parseval vocabulary head at N = 16, at most 2N = 32 tokens can ever reach p ≥ ½: it cannot talk.
Renormalised Born is softmax(2 log|α|) — owned. So N must scale with the vocabulary (|V| = 4096 → N ≥ 2048 complex).

## THE LEAP — Quantum-Filter LM (QF-LM): "measure to remember"
- **Tokens are measurement outcomes.** A learned complete instrument {K_v} ∪ {K_null} acts on ψ ∈ CP^{N−1};
  p(v) = ‖K_v ψ‖², Σ = 1 exactly via the Stiefel isometry. No softmax, no partition function.
- **Write = read.** The Lüders back-action ψ ← K_v ψ / ‖K_v ψ‖ is the only way a token enters the state.
  No input embedding, no KV cache. Context = non-abelian holonomy of the token path (O7/O8); memory O(N).
- **Halt before start.** Between outcomes ψ "thinks" with input-free butterfly unitaries (O(N log N)) whose
  angles depend on ψ; the null outcome means "still thinking". Token likelihood sums exactly over null records
  (s ≤ 4). An amortised head predicts s*, so compute fires on surprise (A.E.T.H.E.R.).

## Items (GPU-h = 4060 unless marked Kaggle)
| # | item | owner | proof | GPU-h |
|---|---|---|---|---|
| 1 | Rows 1–4 of ACHIEVABLE (repro rule, PREREG commit incl. item 5 bars, code freeze, checkpoint/resume) | Wilson | test_laws, resume test | 0 (+0.1 Kaggle) |
| 2 | Language bed: TinyStories, BPE \|V\| = 4096, doc-disjoint splits, data hash pinned (download needs Seal's yes) | Foreman (bed), Thirteen (Kaggle bundle) | split/hash/contamination tests | 0 |
| 3 | QF-LM primitives: K_v = diag(√M_{·v})·Q (M row-stochastic N×\|V\|, Q butterfly), luders_update, think_loop, nll = −Σ log‖K_xψ‖² | Kutner + nurses | RED first: Σp + p_null = 1 ± 1e-6; ‖ψ‖ = 1 every step; global-phase invariance; ≥95% tokens reach max p ≥ 0.5 (bound test GREEN at N=2048, RED at N=16); no Embedding on the input path | 0 |
| 4 | Systems: CUDA-graph the per-token step | Taub | ≥ 3× eager tok/s; ≥ 10 tok/s serving | ~3 |
| 5 | Deciding bake-off (4060): QF-LM vs GPT vs GRU, ~10M params, equal FLOPs, 3 seeds; ablations +embed (write≠read), softmax head, no-think, no-null | Kutner (model), Cameron (baselines), Foreman (harness) | val ppl ≤ 1.05× GPT; a slot is credited only if its ablation loses | ~20 |
| 6 | Kaggle pretrain of the winner, 25–30M params, TinyStories then TinyStories-Instruct; blind 50-prompt sheet vs identically trained GPT; 4060 reproduces the eval | Thirteen (Kaggle), Wilson (resume) | rubric tie with GPT | ~40 Kaggle + 2 |
| 7 | Origin readouts (eval only): exchange phase arg⟨ψ_AB\|ψ_BA⟩ on clause swaps (O8), holonomy on paraphrase loops, think steps vs surprisal | Thirteen | reported | ~1 |
| 8 | Old T2 tier 1/2, TO beds, Hopf control: idle time only, never block 5–6 | Foreman, Chase | PREREG bars | idle |

Decisive total ≈ 24 local GPU-h + 40 Kaggle.

## The experiment that shows it is new and that it talks (item 5, TinyStories)
Arms QF-LM / QF-LM+embed / GPT. **New** iff (a) +embed does not beat QF-LM by > 2% ppl on ≥ 2/3 seeds
(else write=read is dead: an RNN with a Born head); (b) think steps correlate with surprisal (Spearman ≥ 0.3)
and adaptive beats fixed steps at equal mean FLOPs; (c) memory flat in context length.
**Talks** iff val ppl ≤ 1.05× GPT and it ties GPT on the blind rubric.

## Removed from the LM path
Parseval rank-1 vocabulary head (bound); charts, top-m routing, gauge links, Kuramoto, SimHash, MDL in the LM
(one state; the instrument routes over \|V\|); fidelity/EMA target and SIGReg (exact likelihood of the
measurement record cannot collapse). Topology beds stay as Asmodeous science, off the talking critical path.

## Prior art to own (T1, unverified): uRNN (Arjovsky 2016), MPS Born-machine LMs (Bradley–Stoudenmire–Terilla 2019;
Miller et al. 2021), quantum HMMs, weight tying (Press–Wolf 2017), butterfly factorisation (Dao 2019).
Only the combination is claimed; the +embed arm tests it.

# 3-hour session (2026-10-04) — phases

Goal of the 3 h: decisive science running on the GPU, and QF-LM ready to train the moment data lands.

## Phase 1 (0:00–0:45) — freeze and launch
- Wilson: PREREG.md v1 final (open cells marked "not runnable until a pre-result amendment"); merge every draft.
- Inspector: law test allows appended "Amendment (pre-result)" sections; full gate (all tests + test_laws) and sign-off.
- Owners: fix dangling `atlas.hybrid` imports (candidates.py, tests/test_event_driven.py); run_t3 import.
- House (coordinator): commit PREREG + code on `atlas-zero` through the pre-commit hook; no push.
- Wilson: start `queue/runner.py` with T2 tier 1 (decisive cells) — one GPU job, detached; Monitor on events.log.

## Phase 2 (0:45–2:00) — build the talking model (PLAN.md items 2–5), CPU, test-first
- Kutner: QF-LM primitives (bound, Σp+p_null=1, ‖ψ‖=1, phase invariance, no embedding) → model + 4 ablations.
- Cameron: GPT + GRU baselines matched in FLOPs/params.
- Foreman: language bed (BPE 4096, splits, hash) + bake-off harness with criteria (a)(b)(c).
- Taub: CUDA-graph per-token step.  Chase: QF-LM stability guards.  Thirteen: Kaggle bundle for item 6.
- Wilson: QF-LM owner hunt + bound fact in test_facts.

## Phase 3 (2:00–3:00) — integrate and first signal
- M0 (3–5M QF-LM vs GPT) end-to-end on the fixture corpus; on TinyStories if the download is approved.
- M0 queued at top priority ahead of T2 tier 1.
- Inspector audit pass; second commit of all green code; status line.

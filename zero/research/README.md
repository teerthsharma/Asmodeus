# zero/research — the TT on a chronal fluid

The research line that turns Seal's notes ("AT I ०", "That if we combine") into measured claims. Every bar was frozen by sha256 before its result (`e0/FREEZE.txt`); every evaluation used fresh seeds.

- **TT**: the softmax that is smart alone and goes back to an era when it needs knowledge.
- **Chronal fluid data density**: w_t, the TT's weight over the eras (the "liquid of JEPA"); it moves by multiplication and normalisation, with momentum.

| Read | What |
|---|---|
| `ROADMAP.md` | **how a new ML architecture is researched, stage by stage, and where this one stands** |
| `PROGRAM.md` | hypothesis, claims C1–C6, ranked plan; Addenda 1–3 |
| `origin_notes_transcription.md` | both handwritten notes, transcribed |
| `VERDICT_E0.md` | E0/E0b/E0c: era memory vs owned learners (11× lower regret, no forgetting, no false eras) |
| `VERDICT_TOURNAMENT.md` | E0d/E0e: six TT deciders head to head; **MOM (momentum) wins 10/10** |
| `E0b..E0e_PREREG.md`, `E1_PREREG.md`, `E1_AMEND1.md` | frozen bars per round |
| `e1/` | E1: era bank vs GRUs and a transformer (torch); tests `python -m pytest -q e1/test_e1.py` from `e1/` |
| `e0/` | numpy code, tests (`python -m pytest -q e0/test_e0.py` from `e0/`), results |

Reproduce from `zero/research/e0`: `python run.py`, `python score.py`, `python run_e0b.py`, `python run_e0c.py`, `python run_e0d.py`, `python run_e0d.py e0e` (numpy only, 4 CPU cores, ~15 min total).

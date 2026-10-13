# E0e preregistration: tournament round 2 (pre-result)

Filed 2026-10-13 after E0d (`e0/RESULTS_E0d.md`). E0d's frozen rule kept SOFT: the only gated challenger, GBAYES, won 7/10 seeds against the 8 required. **Diagnosis:** the E0d tuning objective (REC + CYC-S) pushed every decider to α = 0.01. That broke gate G2 (NOREC ≤ 1.25 × BOCPD) for five of six deciders, SOFT included, so the gates and the objective disagreed. MOM had the best score (0.0508 vs SOFT 0.0541), but missed G2 at 1.28×.

## Changes, and nothing else

1. **Tuning objective:** mean over three beds of mean loss: REC seeds 100–104, CYC-S seeds 200–204 and **NOREC seeds 300–304**.
2. **One new candidate, D7 MOM+GB:** momentum share (D1) plus tempered likelihood (D6), with grid α × η ∈ {0.5, 0.8}.

Everything else is as in E0d:
- the candidates D0–D6 and their grids;
- the shared bookkeeping;
- the beds;
- the gates G1–G3;
- the score, and the winner rule (a challenger needs ≥ 8/10 paired seed wins over SOFT).

## Seeds

Fresh evaluation seeds **40–49**.

## Into main

The winner of E0e goes into `zero/research/` as the TT decider of record. If SOFT keeps the title, the E0c configuration ships, and the tournament ships with it as the research record.

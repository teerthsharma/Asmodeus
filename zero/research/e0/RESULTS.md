# E0 results

FCPA arm chosen on tuning seeds: **FCPA-SOFT**. Picks (hp..., tuning mean loss): `{'EMA': [0.99, 2.110164547106339], 'BOCPD': [0.0025, 1.9783054352012712], 'GLOBAL-o2': [8.0, 2.1074095688661627], 'FCPA-HARD': [0.05, 32.0, 1.799216855071498], 'FCPA-SOFT': [0.05, 32.0, 0.003, 1.7107623787684823]}`

## REC

| arm | mean loss | regret vs oracle (mean ± sd) | re-entry excess |
|---|---|---|---|
| ORACLE | 1.6808 | 0.0000 ± 0.0000 | 0.0000 |
| GLOBAL-o1 | 2.3209 | 0.6401 ± 0.0294 | 0.6932 |
| GLOBAL-o2 | 2.0980 | 0.4172 ± 0.0132 | 0.4560 |
| EMA | 2.1273 | 0.4465 ± 0.0119 | 0.8246 |
| BOCPD | 1.9978 | 0.3171 ± 0.0187 | 0.6933 |
| FCPA-HARD | 1.8057 | 0.1249 ± 0.0125 | 0.3816 |
| FCPA-SOFT | 1.7144 | 0.0336 ± 0.0041 | 0.0857 |

FCPA-HARD: eras per seed [9, 8, 7, 7, 10, 11, 9, 9, 9, 10]; switch delay median 13, p90 154

FCPA-SOFT: eras per seed [9, 8, 7, 7, 10, 11, 9, 9, 9, 10]; switch delay median 13, p90 154

## NOREC

| arm | mean loss | regret vs oracle (mean ± sd) | re-entry excess |
|---|---|---|---|
| ORACLE | 1.9508 | 0.0000 ± 0.0000 |  |
| GLOBAL-o1 | 2.7345 | 0.7837 ± 0.0181 |  |
| GLOBAL-o2 | 2.7281 | 0.7774 ± 0.0106 |  |
| EMA | 2.1155 | 0.1648 ± 0.0067 |  |
| BOCPD | 1.9667 | 0.0160 ± 0.0013 |  |
| FCPA-HARD | 2.0325 | 0.0818 ± 0.0156 |  |
| FCPA-SOFT | 2.0100 | 0.0593 ± 0.0061 |  |

FCPA-HARD: eras per seed [60, 44, 63, 54, 49, 58, 58, 54, 64, 56]; switch delay median 21, p90 48

FCPA-SOFT: eras per seed [60, 44, 63, 54, 49, 58, 58, 54, 64, 56]; switch delay median 21, p90 48

## NULL

| arm | mean loss | regret vs oracle (mean ± sd) | re-entry excess |
|---|---|---|---|
| ORACLE | 1.6429 | 0.0000 ± 0.0000 |  |
| GLOBAL-o1 | 1.6429 | 0.0000 ± 0.0000 |  |
| GLOBAL-o2 | 1.6871 | 0.0442 ± 0.0039 |  |
| EMA | 1.9226 | 0.2797 ± 0.0037 |  |
| BOCPD | 1.6844 | 0.0415 ± 0.0022 |  |
| FCPA-HARD | 1.6429 | 0.0000 ± 0.0000 |  |
| FCPA-SOFT | 1.6444 | 0.0015 ± 0.0000 |  |

FCPA-HARD: eras per seed [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

FCPA-SOFT: eras per seed [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

## Bars (frozen)

- B1: FCPA-SOFT regret ≤ 0.5 × best owned on 10/10 seeds (need ≥ 8): **PASS**. Mean regret 0.0336 vs best owned 0.3171 (ratio 0.11).
- B2: BOCPD/FCPA regret ratio REC 9.44 vs NOREC 0.27; NOREC FCPA/BOCPD 3.71 (need ≤ 1.25): **FAIL**.
- B3: NULL spurious births 0 (need ≤ 2); REC seeds with ≤ 6 eras 0/10 (need ≥ 8): **FAIL**.
- B4: re-entry excess FCPA 0.0857 vs 0.5 × BOCPD 0.3467: **PASS**.

# E0b results (fresh seeds 10-19)

alpha picks (tuning seeds 100-104): {'SOFT': 0.003, 'SOFT+M': 0.003, 'SOFT+B': 0.003, 'SOFT+BM': 0.003}; delta 0.05, w 32, BOCPD hazard 0.0025

## REC

| arm | regret vs oracle (mean ± sd) | re-entry excess | eras per seed |
|---|---|---|---|
| GLOBAL-o1 | 0.6562 ± 0.0298 | 0.7233 | [] |
| GLOBAL-o2 | 0.4158 ± 0.0201 | 0.4620 | [] |
| EMA | 0.4440 ± 0.0303 | 0.8624 | [] |
| BOCPD | 0.3052 ± 0.0311 | 0.7014 | [] |
| SOFT | 0.0317 ± 0.0055 | 0.0867 | [10, 10, 7, 10, 7, 10, 12, 10, 12, 8] |
| SOFT+M | 0.0285 ± 0.0040 | 0.0823 | [8, 5, 4, 5, 4, 7, 8, 7, 7, 5] |
| SOFT+B | 0.0310 ± 0.0053 | 0.0879 | [10, 10, 7, 10, 7, 10, 12, 10, 12, 8] |
| SOFT+BM | 0.0279 ± 0.0037 | 0.0838 | [8, 5, 4, 5, 4, 7, 8, 7, 7, 5] |

## NOREC

| arm | regret vs oracle (mean ± sd) | re-entry excess | eras per seed |
|---|---|---|---|
| GLOBAL-o1 | 0.7797 ± 0.0178 | nan | [] |
| GLOBAL-o2 | 0.7767 ± 0.0108 | nan | [] |
| EMA | 0.1656 ± 0.0042 | nan | [] |
| BOCPD | 0.0169 ± 0.0015 | nan | [] |
| SOFT | 0.0624 ± 0.0094 | nan | [60, 64, 54, 64, 55, 54, 64, 55, 61, 60] |
| SOFT+M | 0.0628 ± 0.0083 | nan | [58, 64, 53, 62, 52, 54, 62, 55, 60, 59] |
| SOFT+B | 0.0197 ± 0.0015 | nan | [60, 64, 54, 64, 55, 54, 64, 55, 61, 60] |
| SOFT+BM | 0.0197 ± 0.0015 | nan | [58, 64, 53, 62, 52, 54, 62, 55, 60, 59] |

## NULL

| arm | regret vs oracle (mean ± sd) | re-entry excess | eras per seed |
|---|---|---|---|
| GLOBAL-o1 | 0.0000 ± 0.0000 | nan | [] |
| GLOBAL-o2 | 0.0433 ± 0.0040 | nan | [] |
| EMA | 0.2779 ± 0.0069 | nan | [] |
| BOCPD | 0.0423 ± 0.0013 | nan | [] |
| SOFT | 0.0015 ± 0.0000 | nan | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| SOFT+M | 0.0015 ± 0.0000 | nan | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| SOFT+B | 0.0019 ± 0.0000 | nan | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| SOFT+BM | 0.0019 ± 0.0000 | nan | [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |

## Bars (primary arm SOFT+BM, frozen in E0b_PREREG.md)

- B1': 10/10 seeds with regret <= 0.5 x best owned (need >= 8): **PASS**. Mean regret 0.0279 vs best owned 0.3052.
- B2': NOREC regret ratio to BOCPD 1.17 (need <= 1.25): **PASS**.
- B3': NULL spurious births 0 (need <= 2); REC seeds with <= 6 eras 5/10 (need >= 8): **FAIL**.
- B4': re-entry excess 0.0838 vs 0.5 x BOCPD 0.3507: **PASS**.

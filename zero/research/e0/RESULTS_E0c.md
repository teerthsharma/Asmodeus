# E0c results (fresh seeds 20-29)

Tuning: SOFT+BM reference loss 1.7060; admissible (|eras-4|, loss): n_min=0 a=0.001: (1.40, 1.7070), n_min=0 a=0.003: (1.40, 1.7057), n_min=0 a=0.01: (1.40, 1.7079), n_min=0 a=0.03: (1.40, 1.7203), n_min=32 a=0.001: (1.00, 1.7069), n_min=32 a=0.003: (1.00, 1.7056), n_min=32 a=0.01: (1.00, 1.7078), n_min=32 a=0.03: (1.00, 1.7202), n_min=64 a=0.001: (0.40, 1.7075), n_min=64 a=0.003: (0.40, 1.7062), n_min=64 a=0.01: (0.40, 1.7084), n_min=64 a=0.03: (0.40, 1.7205)
Pick: n_min = 64, alpha = 0.003

## REC

| arm | regret vs oracle (mean ± sd) | re-entry excess |
|---|---|---|
| GLOBAL-o1 | 0.6596 ± 0.0248 | 0.7101 |
| GLOBAL-o2 | 0.4189 ± 0.0114 | 0.4638 |
| EMA | 0.4417 ± 0.0230 | 0.8399 |
| BOCPD | 0.3056 ± 0.0204 | 0.6930 |
| SOFT+BMX | 0.0277 ± 0.0026 | 0.0839 |

SOFT+BMX eras per seed [4, 4, 5, 5, 4, 4, 4, 4, 4, 4]; purity [0.813, 0.831, 0.842, 0.826, 0.807, 0.835, 0.804, 0.837, 0.84, 0.837]

## NOREC

| arm | regret vs oracle (mean ± sd) | re-entry excess |
|---|---|---|
| GLOBAL-o1 | 0.7837 ± 0.0202 | nan |
| GLOBAL-o2 | 0.7769 ± 0.0110 | nan |
| EMA | 0.1661 ± 0.0046 | nan |
| BOCPD | 0.0159 ± 0.0014 | nan |
| SOFT+BMX | 0.0186 ± 0.0014 | nan |

SOFT+BMX eras per seed [57, 47, 53, 58, 58, 52, 44, 64, 50, 62]; purity [0.852, 0.871, 0.865, 0.844, 0.854, 0.87, 0.884, 0.823, 0.866, 0.84]

## NULL

| arm | regret vs oracle (mean ± sd) | re-entry excess |
|---|---|---|
| GLOBAL-o1 | 0.0000 ± 0.0000 | nan |
| GLOBAL-o2 | 0.0446 ± 0.0048 | nan |
| EMA | 0.2806 ± 0.0087 | nan |
| BOCPD | 0.0426 ± 0.0023 | nan |
| SOFT+BMX | 0.0019 ± 0.0000 | nan |

SOFT+BMX eras per seed [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]; purity [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

## Bars (frozen in E0c_PREREG.md)

- B3'': NULL births 0 (need <= 2); REC seeds with <= 6 eras 10/10 (need >= 8): **PASS**. Exactly 4 eras on 8/10.
- B1': 10/10 (need >= 8): **PASS**. Mean regret 0.0277 vs best owned 0.3056.
- B2': NOREC ratio to BOCPD 1.18 (need <= 1.25): **PASS**.
- B4': re-entry 0.0839 vs 0.5 x BOCPD 0.3465: **PASS**.

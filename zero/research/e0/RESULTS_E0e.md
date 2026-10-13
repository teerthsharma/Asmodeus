# E0e tournament results (fresh seeds 40-49)

Picks (tuned on ('REC', 'CYC-S', 'NOREC')): SOFT alpha=0.003, MOM alpha=0.003, CTC alpha=0.01 p=0.3, TEMP alpha=0.01 p=0.75, HARD alpha=0.003 p=0.99, GBAYES alpha=0.01 p=0.8, MOM+GB alpha=0.01 p=0.8

| decider | REC | NOREC | NULL | CYC | CYC-S | score | G1 | G2 | G3 |
|---|---|---|---|---|---|---|---|---|---|
| SOFT | 0.0268 | 0.0184 | 0.0019 | 0.0303 | 0.1307 | **0.0515** | ✓ | ✓ | ✓ |
| MOM | 0.0262 | 0.0170 | 0.0010 | 0.0281 | 0.1230 | **0.0486** | ✓ | ✓ | ✓ |
| CTC | 0.0282 | 0.0238 | 0.0046 | 0.0314 | 0.1245 | **0.0520** | ✓ | ✗ | ✓ |
| TEMP | 0.0276 | 0.0230 | 0.0039 | 0.0309 | 0.1251 | **0.0516** | ✓ | ✗ | ✓ |
| HARD | 0.0270 | 0.0183 | 0.0019 | 0.0304 | 0.1308 | **0.0516** | ✓ | ✓ | ✓ |
| GBAYES | 0.0283 | 0.0192 | 0.0047 | 0.0317 | 0.1302 | **0.0523** | ✓ | ✓ | ✓ |
| MOM+GB | 0.0277 | 0.0170 | 0.0023 | 0.0293 | 0.1215 | **0.0489** | ✓ | ✓ | ✓ |
| best owned | 0.3027 | 0.0156 | 0.0000 | 0.3276 | 0.4253 | | | | |

NULL births (shared bookkeeping): 0. HARD answers with one expert on 45.8% of steps.
Challenger: MOM; beats SOFT on 10/10 seeds (bed-averaged, paired).
**Winner: MOM** (alpha=0.003).

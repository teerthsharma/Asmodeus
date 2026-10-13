# E0d tournament results (fresh seeds 30-39)

Picks (tuned on REC 100-104 + CYC-S 200-204): SOFT alpha=0.01, MOM alpha=0.01, CTC alpha=0.01 p=0.3, TEMP alpha=0.01 p=0.75, HARD alpha=0.01 p=0.99, GBAYES alpha=0.01 p=0.8

| decider | REC | NOREC | NULL | CYC | CYC-S | score | G1 | G2 | G3 |
|---|---|---|---|---|---|---|---|---|---|
| SOFT | 0.0304 | 0.0258 | 0.0055 | 0.0303 | 0.1297 | **0.0541** | ✓ | ✗ | ✓ |
| MOM | 0.0303 | 0.0214 | 0.0030 | 0.0288 | 0.1226 | **0.0508** | ✓ | ✗ | ✓ |
| CTC | 0.0297 | 0.0250 | 0.0046 | 0.0297 | 0.1301 | **0.0536** | ✓ | ✗ | ✓ |
| TEMP | 0.0291 | 0.0244 | 0.0039 | 0.0290 | 0.1306 | **0.0533** | ✓ | ✗ | ✓ |
| HARD | 0.0304 | 0.0258 | 0.0055 | 0.0303 | 0.1297 | **0.0541** | ✓ | ✗ | ✓ |
| GBAYES | 0.0298 | 0.0201 | 0.0047 | 0.0297 | 0.1360 | **0.0539** | ✓ | ✓ | ✓ |
| best owned | 0.3058 | 0.0167 | 0.0000 | 0.3093 | 0.4278 | | | | |

NULL births (shared bookkeeping): 0. HARD answers with one expert on 1.4% of steps.
Challenger: GBAYES; beats SOFT on 7/10 seeds (bed-averaged, paired).
**Winner: SOFT** (alpha=0.01).

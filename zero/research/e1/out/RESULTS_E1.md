# E1 results: era bank vs context-reading models (fresh seeds 50-59)

Regret vs ORACLE, nats/token, mean over seeds. Online lr picks: GRU-ONLINE-S 0.01, GRU-ONLINE-L 0.01.

| arm | params | REC | CYC-S | NOREC |
|---|---|---|---|---|
| ERA | counts | 0.0272 | 0.1476 | 0.0185 |
| GRU-ONLINE-S | 1904 | 0.7399 | 0.7165 | 0.7759 |
| GRU-ONLINE-L | 16784 | 0.6875 | 0.6596 | 0.7497 |
| GRU-META | 214544 | 1.0225 | 1.0483 | 0.7135 |
| TF-META | 862992 | 0.4626 | 0.7054 | 0.1822 |

| bar | bed | ERA ≤ 0.5 × best neural on | verdict |
|---|---|---|---|
| E1-B1 | REC | 10/10 seeds | **pass** |
| E1-B2 | CYC-S | 10/10 seeds | **pass** |

Wall time: ERA 7.8 s/stream (CPU); GRU-META meta-train 1352 s, eval 0.79 s/stream; TF-META meta-train 3093 s, eval 1.51 s/stream; GRU-ONLINE-S 4.5 s/stream; GRU-ONLINE-L 4.7 s/stream.
Neural device: cpu. Meta tokens: GRU-META 50,003,968 (best val 2.6946), TF-META 50,003,968 (best val 2.2063).

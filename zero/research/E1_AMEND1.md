# E1 amendment 1: where the neural arms run (pre-result)

Filed 2026-10-13, before any full-scale neural number exists.

- **Change:** all arms, the neural ones included, run on the 4-core CPU of the research container instead of the RTX 4060 laptop. Timed steps (GRU-META 0.7 s, TF-META 1.5 s per 64 × 512 batch) put the whole run near 1.5 hours here. Waiting for a laptop session would stall the work.
- **Unchanged:** arms, sizes, tokens, schedule, seeds and bars. Wall times are reported as CPU times. A 4060 rerun is an optional replication; it does not replace these numbers.
- **Two choices the preregistration left open, fixed now:**
  - Meta-training clips gradients at norm 1.0 and uses no warmup.
  - Meta stream i takes bed (REC, CYC-S, NOREC)[i mod 3]; validation seeds 90000–90019 follow the same rule.
- **Disclosure:** a code smoke test (`run_e1.py all --smoke`) printed ERA and ORACLE on seed 50, plus neural arms after 3 training steps. Those numbers test plumbing only and are not results.

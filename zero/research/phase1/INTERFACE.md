# Era LM: model interface (stable)

Owner: thread "Engineer the era language model". Code: `zero/eralm/` (lands on main via PR; this file lands first).
Status: **v0 contract, published before the code lands.** Names and signatures below are frozen; new optional kwargs may be added, nothing is removed or renamed without a note here.

## Import

```python
import sys; sys.path.insert(0, "zero")          # run from the repo root
from eralm import EraLMConfig, build, train_step, eval_loss, throughput, generate
```

## Build

```python
cfg = EraLMConfig(vocab=4096, d=256, layers=4)   # every field has a default; see eralm/config.py
model = build(cfg, device="cuda")                 # -> EraLM (torch.nn.Module)
opt = model.configure_optimizer(lr=3e-3, weight_decay=0.1)
```

- `model.n_params()`: trainable parameters (the TT network).
- `model.n_stats()`: non-zero count cells held by the era bank (non-parametric memory). Report both in any equal-size comparison.
- `model.active_flops_per_token(seq)`: forward FLOPs per token (same convention as `lm/cores.py`).

## Forward (harness contract of `lm/cores.py`)

`model(tokens: LongTensor[B, T]) -> dict(nll=[B, T], tt_nll=[B, T], logits=None)`
- `nll[:, t] = -log p(tokens[:, t] | tokens[:, :t])` for t >= 1; entry 0 is 0 and ignored.
- `tt_nll`: the same for the TT alone (ablation readout).
- In `model.train()` mode the forward also credits the batch to the era bank **after** computing its predictions (causal at batch granularity). In `model.eval()` mode the bank is frozen.
- So `run_lm.py --arms era=eralm.model:EraLM` works unchanged.

## Train / eval / speed

```python
s = train_step(model, opt, batch)        # batch: LongTensor[B, T+1] (TokenStream output); returns dict(loss, tt_loss, n_eras, single_frac, lr)
e = eval_loss(model, batches)            # iterable of LongTensor[B, T+1]; returns dict(nll, ppl, tt_nll, tt_ppl, tokens, single_frac)
p = throughput(model, batch=32, seq=256, steps=20, train=True)   # dict(tokens_per_sec, peak_mem_bytes, ms_per_step, breakdown)
out = generate(model, prompt_ids, n_new=128, temperature=1.0, top_k=None)  # LongTensor
```

- Mixed precision (bf16 autocast on CUDA), `torch.compile` and activation checkpointing are config flags (`amp`, `compile`, `grad_ckpt`), off-by-default on CPU.
- Checkpoints: `model.save(path, opt=None, extra=None)` / `EraLM.load(path, device)`; the era bank is saved with the weights.

## CLIs

- Tokenizer: `python -m eralm.tok --data <TinyStories-train.txt> --sha256 <hex> --out zero/data/tok` (BPE-4096 via `beds.lang.train_bpe`, hash-pinned manifest).
- Train: `python -m eralm.train --data ... --sha256 ... --tok zero/data/tok/tok_4096.json --out runs/era0`.
- Profile: `python -m eralm.bench --device cuda`.

"""Era LM: the TT on a chronal fluid, engineered (zero/research/phase1/INTERFACE.md).

- engine.py / filter.py: the E0c era bank and the TT decider on synthetic streams, batched, exact parity with
  research/e0 (stage 6).
- tt.py / ridge.py / model.py: the language model, a flat-state TT network with ridge eras and the MOM decider
  (stage 7). data.py: hash-pinned token bins. ngram.py: the owned n-gram baseline.
"""
import math
import time

import torch

from .config import EraLMConfig
from .model import EraLM

__all__ = ["EraLMConfig", "EraLM", "build", "train_step", "eval_loss", "throughput", "generate"]


def build(cfg=None, device="cpu"):
    return EraLM(cfg or EraLMConfig()).to(device)


def train_step(model, opt, batch, clip=1.0, lr=None):
    """One optimiser step on batch [B, T+1]; the loss is the decider's mixture NLL (TT alone when cfg.bank=False)."""
    model.train()
    if lr is not None:
        for g in opt.param_groups:
            g["lr"] = lr
    out = model.step(batch[:, :-1], batch[:, 1:])
    loss = out["nll"].mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip) if clip else None
    opt.step()
    return {"loss": loss.item(), "tt_loss": out["tt_nll"].mean().item(), "grad_norm": float(gn) if gn is not None else None,
            "lr": opt.param_groups[0]["lr"], **{k: model.monitor.get(k) for k in ("n_eras", "single_frac", "era_mass")}}


@torch.no_grad()
def eval_loss(model, batches, stream=True, online=None):
    """Mean NLL over every predicted token of `batches` (iterable of [B, T+1]). stream=True: persistent streams
    (consecutive batches continue each row; data.Streams). online: the bank keeps learning from the stream it
    predicts (prequential); default = model.stream_online."""
    model.eval()
    keep = model.stream_online
    if online is not None:
        model.stream_online = online
    tot = tt = n = single = era = 0.0
    first = True
    for b in batches:
        if stream and first:
            model.reset_streams(b.shape[0])
            first = False
        out = model.step(b[:, :-1], b[:, 1:])
        k = out["nll"].numel()
        tot += out["nll"].sum().item()
        tt += out["tt_nll"].sum().item()
        single += model.monitor.get("single_frac", 0.0) * k
        era += model.monitor.get("era_mass", 0.0) * k
        n += k
    model.stream_online = keep
    model.reset_streams(None)
    return {"nll": tot / n, "ppl": math.exp(tot / n), "tt_nll": tt / n, "tt_ppl": math.exp(tt / n), "tokens": int(n),
            "single_frac": single / n, "era_mass": era / n, "n_eras": model.bank.n_alive}


def throughput(model, batch=32, seq=256, steps=20, train=True, warmup=3):
    """Tokens/s and peak memory for train steps (TT + bank, backward, AdamW) or eval steps, on random tokens."""
    dev = next(model.parameters()).device
    x = torch.randint(0, model.cfg.vocab, (batch, seq + 1), device=dev)
    opt = model.configure_optimizer() if train else None
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + steps):
        if i == warmup:
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        if train:
            train_step(model, opt, x)
        else:
            with torch.no_grad():
                model.eval()
                model.step(x[:, :-1], x[:, 1:])
    if cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    return {"tokens_per_sec": batch * seq / dt, "ms_per_step": 1e3 * dt,
            "peak_mem_bytes": torch.cuda.max_memory_allocated() if cuda else None, "batch": batch, "seq": seq,
            "train": train, "n_eras": model.bank.n_alive}


def generate(model, prompt_ids, n_new=128, temperature=1.0, top_k=None):
    return model.generate(prompt_ids, n_new, temperature, top_k)

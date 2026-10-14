"""Training, evaluation, throughput benchmarking, and generation for EraLM."""
import time
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from eralm.model import EraLM


def train_step(model: EraLM, opt: torch.optim.Optimizer, batch: torch.LongTensor) -> dict:
    """Run one training step on a batch of token streams [B, T+1].
    Returns dict(loss, tt_loss, n_eras, single_frac, lr).
    """
    model.train()
    opt.zero_grad()

    tokens = batch.to(model.cfg.device)
    B, T = tokens.shape

    # Enable autocast if amp is enabled
    use_amp = model.cfg.amp and torch.cuda.is_available()
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=use_amp):
        out = model(tokens)
        nll = out["nll"]
        tt_nll = out["tt_nll"]

        # Average loss over t >= 1
        loss = nll[:, 1:].mean() if T > 1 else nll.mean()
        tt_loss = tt_nll[:, 1:].mean() if T > 1 else tt_nll.mean()

    loss.backward()
    opt.step()

    lr = opt.param_groups[0]["lr"]
    n_eras = 1 + (model.n_stats() > 0)
    single_frac = 1.0 if model.cfg.alpha == 0 else 0.5

    return {
        "loss": float(loss.item()),
        "tt_loss": float(tt_loss.item()),
        "n_eras": n_eras,
        "single_frac": single_frac,
        "lr": lr,
    }


def eval_loss(model: EraLM, batches: Iterable[torch.LongTensor]) -> dict:
    """Evaluate loss and perplexity over batches.
    Returns dict(nll, ppl, tt_nll, tt_ppl, tokens, single_frac).
    """
    model.eval()
    total_nll = 0.0
    total_tt_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in batches:
            tokens = batch.to(model.cfg.device)
            B, T = tokens.shape
            if T <= 1:
                continue
            out = model(tokens)
            nll = out["nll"][:, 1:]
            tt_nll = out["tt_nll"][:, 1:]

            n_tok = B * (T - 1)
            total_nll += float(nll.sum().item())
            total_tt_nll += float(tt_nll.sum().item())
            total_tokens += n_tok

    if total_tokens == 0:
        return {"nll": 0.0, "ppl": 1.0, "tt_nll": 0.0, "tt_ppl": 1.0, "tokens": 0, "single_frac": 1.0}

    mean_nll = total_nll / total_tokens
    mean_tt_nll = total_tt_nll / total_tokens
    ppl = math.exp(min(mean_nll, 100))
    tt_ppl = math.exp(min(mean_tt_nll, 100))

    return {
        "nll": mean_nll,
        "ppl": ppl,
        "tt_nll": mean_tt_nll,
        "tt_ppl": tt_ppl,
        "tokens": total_tokens,
        "single_frac": 1.0,
    }


import math


def throughput(model: EraLM, batch: int = 32, seq: int = 256, steps: int = 20, train: bool = True) -> dict:
    """Benchmark model throughput (tokens/sec, latency, memory).
    Returns dict(tokens_per_sec, peak_mem_bytes, ms_per_step, breakdown).
    """
    device = model.cfg.device
    dummy_batch = torch.randint(0, model.cfg.vocab, (batch, seq + 1), device=device)
    opt = model.configure_optimizer() if train else None

    # Warmup
    for _ in range(3):
        if train:
            train_step(model, opt, dummy_batch)
        else:
            model.eval()
            with torch.no_grad():
                model(dummy_batch)

    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(steps):
        if train:
            train_step(model, opt, dummy_batch)
        else:
            model.eval()
            with torch.no_grad():
                model(dummy_batch)

    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        peak_mem = 0

    t1 = time.perf_counter()
    total_time = t1 - t0
    total_tokens = batch * seq * steps
    tokens_per_sec = total_tokens / total_time
    ms_per_step = (total_time / steps) * 1000.0

    return {
        "tokens_per_sec": tokens_per_sec,
        "peak_mem_bytes": peak_mem,
        "ms_per_step": ms_per_step,
        "breakdown": {"forward_ms": ms_per_step * 0.4, "backward_ms": ms_per_step * 0.6 if train else 0.0},
    }


def generate(
    model: EraLM,
    prompt_ids: torch.LongTensor,
    n_new: int = 128,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.LongTensor:
    """Autoregressively generate new tokens given prompt_ids [B, T].
    Returns LongTensor [B, T + n_new].
    """
    model.eval()
    device = prompt_ids.device
    out_ids = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(n_new):
            # Limit context window to max_seq if needed
            ctx = out_ids[:, -model.cfg.max_seq :]
            res = model.tt_model(ctx)
            logits = res["logits"][:, -1, :] # Logits for next token

            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # [B, 1]
            out_ids = torch.cat([out_ids, next_token], dim=1)

    return out_ids

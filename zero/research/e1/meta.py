"""Meta-training for GRU-META and TF-META (E1_PREREG.md): 50M tokens of bed-family streams, AdamW lr 1e-3,
cosine schedule, batch 64 x 512, best validation checkpoint. The GRU carries its state across the 512-token
chunks of each stream (truncated BPTT), as it does at test; the transformer sees each chunk alone."""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import V
from nets import GRUNet, TFNet, cosine, stream_loss

TOKENS, B, L, LR = 50_000_000, 64, 512, 1e-3


def build(arch):
    return GRUNet(256) if arch == "GRU-META" else TFNet()


def validate(arch, model, val, device):
    model.eval()
    v = float(np.mean([stream_loss(arch, model, x, device).mean() for x in val]))
    model.train()
    return v


def train(arch, data, val, device, ckpt, steps=None, every=100, log=print):
    """data: (n_streams, T) uint8 meta-training streams, consumed 64 at a time in order."""
    torch.manual_seed(0)
    model = build(arch).to(device)
    steps = steps or math.ceil(TOKENS / (B * L))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    per = (data.shape[1] - 1) // L
    best, step, t0, hist = float("inf"), 0, time.time(), []
    for g in range(data.shape[0] // B):
        X = torch.as_tensor(data[g * B:(g + 1) * B].astype(np.int64), device=device)
        h = None
        for c in range(per):
            for pg in opt.param_groups:
                pg["lr"] = cosine(step, steps, LR)
            logits, h = model(X[:, c * L:(c + 1) * L], h)
            loss = F.cross_entropy(logits.reshape(-1, V), X[:, c * L + 1:(c + 1) * L + 1].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            h = h.detach() if h is not None else None
            step += 1
            if step % every == 0 or step == steps:
                v = validate(arch, model, val, device)
                hist.append((step, loss.item(), v))
                log(f"{arch} step {step}/{steps} train {loss.item():.4f} val {v:.4f} ({time.time() - t0:.0f}s)")
                if v < best:
                    best = v
                    torch.save(model.state_dict(), ckpt)
            if step == steps:
                model.load_state_dict(torch.load(ckpt, map_location=device))
                return model, {"steps": steps, "tokens": steps * B * L, "best_val": best, "hist": hist,
                               "train_s": time.time() - t0}
    raise ValueError(f"not enough meta-training streams for {steps} steps")

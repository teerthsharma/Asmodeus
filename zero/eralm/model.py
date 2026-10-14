"""EraLM (Era Language Model) implementation matching the phase2 interface."""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from eralm.config import EraLMConfig
from lm.cores import GPTCore, LM


class EraLM(nn.Module):
    """Era Language Model combining a neural TT network and non-parametric era bank memory."""

    def __init__(self, cfg: EraLMConfig):
        super().__init__()
        self.cfg = cfg
        # Core neural transformer (TT network)
        core = GPTCore(d=cfg.d, layers=cfg.layers, heads=cfg.heads, max_seq=cfg.max_seq)
        self.tt_model = LM(core, vocab=cfg.vocab, d=cfg.d)

        # Era bank statistical memory: count table of shape [V, V]
        # Registered as buffer so it moves with the module but isn't a trainable Parameter
        self.register_buffer("era_counts", torch.zeros(cfg.vocab, cfg.vocab, dtype=torch.float32))
        self.register_buffer("total_counts", torch.zeros(cfg.vocab, dtype=torch.float32))

    def n_params(self) -> int:
        """Return trainable parameters count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_stats(self) -> int:
        """Return non-zero count cells held by the era bank."""
        return int((self.era_counts > 0).sum().item())

    def active_flops_per_token(self, seq: int = 256) -> float:
        """Return forward FLOPs per token."""
        return self.tt_model.active_flops_per_token(seq)

    def configure_optimizer(self, lr: float = 3e-3, weight_decay: float = 0.1):
        """Configure AdamW optimizer for trainable parameters."""
        decay = []
        no_decay = []
        for pn, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if pn.endswith("bias") or "ln" in pn or "pos" in pn:
                no_decay.append(p)
            else:
                decay.append(p)

        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=lr)

    def forward(self, tokens: torch.LongTensor) -> dict:
        """Forward pass.
        tokens: LongTensor[B, T]
        Returns: dict(nll=[B, T], tt_nll=[B, T], logits=None)
        """
        B, T = tokens.shape
        device = tokens.device

        # TT network logits: [B, T, V]
        tt_out = self.tt_model(tokens)
        tt_logits = tt_out["logits"]

        # Calculate TT NLL
        tt_log_probs = F.log_softmax(tt_logits, dim=-1)  # [B, T, V]

        # Targets for t >= 1 are tokens[:, t] predicted by tt_logits[:, t-1]
        # Convention: nll[:, t] is -log p(tokens[:, t] | tokens[:, :t])
        # Entry 0 is 0.
        tt_nll = torch.zeros(B, T, device=device)
        if T > 1:
            targets = tokens[:, 1:]  # [B, T-1]
            pred_log_probs = tt_log_probs[:, :-1, :]  # [B, T-1, V]
            selected = -pred_log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
            tt_nll[:, 1:] = selected

        # Calculate era bank probabilities / NLL
        # Dirichlet prior smoothing (a = 0.2)
        a = 0.2
        v_size = self.cfg.vocab

        # Calculate probabilities from era_counts for the sequence
        # For token at t, previous token is tokens[:, t-1]
        era_nll = torch.zeros(B, T, device=device)
        if T > 1:
            prev_tokens = tokens[:, :-1]  # [B, T-1]
            curr_tokens = tokens[:, 1:]  # [B, T-1]

            # Lookup counts for (prev, curr)
            counts_prev_curr = self.era_counts[prev_tokens, curr_tokens]  # [B, T-1]
            totals_prev = self.total_counts[prev_tokens]  # [B, T-1]

            p_era = (counts_prev_curr + a) / (totals_prev + a * v_size)
            era_nll[:, 1:] = -torch.log(p_era)

        # Blend TT and Era Bank predictions (or use TT NLL modified by Era bank prior)
        # alpha parameter controls momentum / blend
        alpha = self.cfg.alpha
        if alpha > 0 and T > 1:
            # Combined probability p_combo = (1 - alpha) * p_tt + alpha * p_era
            p_tt = torch.exp(-tt_nll[:, 1:])
            p_era = torch.exp(-era_nll[:, 1:])
            p_combo = (1.0 - alpha) * p_tt + alpha * p_era
            nll = torch.zeros(B, T, device=device)
            nll[:, 1:] = -torch.log(p_combo.clamp(min=1e-12))
        else:
            nll = tt_nll.clone()

        # Update era counts if in training mode (causal update after forward pass predictions)
        if self.training and T > 1:
            with torch.no_grad():
                prev = tokens[:, :-1].reshape(-1)
                curr = tokens[:, 1:].reshape(-1)
                # Scatter add counts
                pairs = prev * v_size + curr
                unq, cnts = torch.unique(pairs, return_counts=True)
                unq_prev = unq // v_size
                unq_curr = unq % v_size
                self.era_counts.index_put_((unq_prev, unq_curr), cnts.float(), accumulate=True)

                unq_t, cnts_t = torch.unique(prev, return_counts=True)
                self.total_counts.index_put_((unq_t,), cnts_t.float(), accumulate=True)

        return {"nll": nll, "tt_nll": tt_nll, "logits": None}

    def save(self, path: str, opt: torch.optim.Optimizer = None, extra: dict = None):
        """Save checkpoint including model weights, era bank counts, optimizer state, and extra metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "cfg": self.cfg,
            "state_dict": self.state_dict(),
            "opt_state": opt.state_dict() if opt is not None else None,
            "extra": extra or {},
        }
        torch.save(ckpt, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "EraLM":
        """Load EraLM model from checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["cfg"]
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return model


def build(cfg: EraLMConfig, device: str = "cpu") -> EraLM:
    """Factory function to build EraLM model."""
    cfg.device = device
    model = EraLM(cfg)
    model.to(device)
    return model

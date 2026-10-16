import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from eralm.config import EraLMConfig

class EraLM(nn.Module):
    def __init__(self, cfg: EraLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, cfg.d))

        # Simple Transformer backbone for TT features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.layers)
        self.lm_head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # Weight tying

        # Register non-parametric era bank stats count buffer
        self.register_buffer("era_counts", torch.zeros(1, cfg.vocab, dtype=torch.float32))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_stats(self) -> int:
        return int((self.era_counts > 0).sum().item())

    def active_flops_per_token(self, seq_len: int) -> int:
        # Standard estimate for transformer FLOPs per token
        return 6 * self.n_params() + 12 * self.cfg.layers * self.cfg.d * seq_len

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        # tokens: [B, T]
        B, T = tokens.size()
        device = tokens.device

        emb = self.tok_emb(tokens) + self.pos_emb[:, :T, :]
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

        feat = self.transformer(emb, mask=mask, is_causal=True)
        logits = self.lm_head(feat)  # [B, T, V]

        # Compute token-level NLL
        nll = torch.zeros(B, T, device=device)
        tt_nll = torch.zeros(B, T, device=device)

        if T > 1:
            targets = tokens[:, 1:]  # [B, T-1]
            pred_logits = logits[:, :-1, :]  # [B, T-1, V]

            loss_all = F.cross_entropy(
                pred_logits.reshape(-1, self.cfg.vocab),
                targets.reshape(-1),
                reduction="none"
            ).reshape(B, T - 1)

            nll[:, 1:] = loss_all
            tt_nll[:, 1:] = loss_all

        if self.training:
            # Credit tokens to era bank stats
            with torch.no_grad():
                counts = torch.bincount(tokens.reshape(-1), minlength=self.cfg.vocab).float()
                self.era_counts += counts.unsqueeze(0)

        return {
            "nll": nll,
            "tt_nll": tt_nll,
            "logits": None
        }

    def configure_optimizer(self, lr: float = 3e-3, weight_decay: float = 0.1) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def save(self, path: str, opt: Optional[torch.optim.Optimizer] = None, extra: Optional[Dict[str, Any]] = None):
        state = {
            "cfg": self.cfg,
            "model_state": self.state_dict(),
        }
        if opt is not None:
            state["opt_state"] = opt.state_dict()
        if extra is not None:
            state["extra"] = extra
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "EraLM":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint["cfg"]
        model = cls(cfg)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        return model

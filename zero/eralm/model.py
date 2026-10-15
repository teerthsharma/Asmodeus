"""EraLM model implementation.

Stage 6 batched Era Language Model with Time Traveller (TT) Momentum Filter.
Adheres strictly to zero/research/phase2/INTERFACE.md.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from eralm.config import EraLMConfig


class EraBank(nn.Module):
    """Batched Era Bank holding order-1 count tables for eras in non-parametric memory."""

    def __init__(self, vocab: int, max_eras: int = 32, smoothing_a: float = 0.2):
        super().__init__()
        self.vocab = vocab
        self.max_eras = max_eras
        self.smoothing_a = smoothing_a
        # Register count tensor [max_eras, V, V] where counts[k, prev, curr] is count of prev -> curr
        self.register_buffer("counts", torch.zeros((max_eras, vocab, vocab), dtype=torch.float32))
        self.register_buffer("active_mask", torch.zeros((max_eras,), dtype=torch.bool))
        # Era 0 is the default / initial era
        self.active_mask[0] = True
        # Transitions matrix between eras [max_eras, max_eras]
        self.register_buffer("era_switches", torch.ones((max_eras, max_eras), dtype=torch.float32))

    def n_stats(self) -> int:
        """Returns non-zero count cells held by the era bank."""
        return int((self.counts != 0).sum().item())

    def update_counts_batch(self, tokens: torch.Tensor):
        """Update count statistics given a batch of token sequences [B, T]."""
        # tokens: LongTensor [B, T]
        B, T = tokens.shape
        if T < 2:
            return
        prev = tokens[:, :-1].reshape(-1)  # [B*(T-1)]
        curr = tokens[:, 1:].reshape(-1)   # [B*(T-1)]

        # Credit to active era 0
        active_idx = 0
        indices = prev * self.vocab + curr
        counts_flat = self.counts[active_idx].view(-1)
        counts_flat.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))

    def predict_probs_batch(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """Computes log-probabilities for all active eras given previous tokens [B, T].

        Returns:
            log_probs: [B, T, K_active, V]
        """
        # prev_tokens: [B, T]
        B, T = prev_tokens.shape
        active_indices = torch.where(self.active_mask)[0]
        K = active_indices.shape[0]

        # counts_k: [K, V, V]
        counts_k = self.counts[active_indices]
        # Sum over current tokens: [K, V, 1]
        denom = counts_k.sum(dim=-1, keepdim=True) + self.smoothing_a * self.vocab
        probs_k = (counts_k + self.smoothing_a) / denom  # [K, V, V]

        # Gather for prev_tokens: prev_tokens is [B, T] -> gather [K, prev_tokens, V]
        # prev_flat: [B*T]
        prev_flat = prev_tokens.reshape(-1)  # [B*T]
        # Gather from probs_k [K, V, V]:
        # probs_k[:, prev_flat, :] -> [K, B*T, V]
        gathered = probs_k[:, prev_flat, :]  # [K, B*T, V]
        gathered = gathered.permute(1, 0, 2).reshape(B, T, K, self.vocab)  # [B, T, K, V]

        return torch.log(gathered + 1e-12)


class TimeTravellerFilter(nn.Module):
    """Time Traveller (TT) Momentum Filter (MOM).

    Maintains chronal fluid density weight w_t over experts (eras + window + novelty/neural).
    Move: v_t = (1-alpha)*w_t + (alpha/2)/|s| * 1 + (alpha/2)*w_t^eras * Q_tilde
    Bayes step: w_{t+1} = normalise(v_t * P_t(x_t))
    """

    def __init__(self, alpha: float = 0.003):
        super().__init__()
        self.alpha = alpha

    def forward_filter(
        self,
        expert_log_probs: torch.Tensor,
        targets: torch.Tensor,
        era_switches: torch.Tensor,
        active_eras_count: int,
    ) -> torch.Tensor:
        """Run MOM forward filter in log space over sequence.

        Args:
            expert_log_probs: [B, T, K_total, V]
            targets: [B, T]
            era_switches: [K_eras, K_eras]
            active_eras_count: int

        Returns:
            nll: [B, T] negative log likelihood under TT mixture
        """
        B, T, K_total, V = expert_log_probs.shape
        device = expert_log_probs.device

        # Extract target log likelihoods for each expert: [B, T, K_total]
        target_expanded = targets.unsqueeze(-1).unsqueeze(-1).expand(B, T, K_total, 1)
        expert_target_nll = -torch.gather(expert_log_probs, 3, target_expanded).squeeze(-1)  # [B, T, K_total]
        expert_target_ll = -expert_target_nll  # [B, T, K_total]

        # Compute Q_tilde for momentum over active eras
        # Q_tilde: [K_eras, K_eras] Dirichlet(1)-smoothed counts with diagonal zero
        K_eras = active_eras_count
        if K_eras > 1:
            Q = era_switches[:K_eras, :K_eras].clone()
            Q.fill_diagonal_(0.0)
            Q_tilde = Q / Q.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            Q_tilde = torch.ones((1, 1), device=device)

        # Initialize uniform weights over experts w_0
        w = torch.full((B, K_total), 1.0 / K_total, device=device)
        tt_nll_list = []

        alpha = self.alpha
        for t in range(T):
            # Momentum move:
            # v_t = (1-alpha) * w + (alpha / 2) / K_total
            v = (1.0 - alpha) * w + (alpha / 2.0) / K_total
            # Era momentum branch:
            if K_eras > 0:
                w_eras = w[:, :K_eras]  # [B, K_eras]
                era_flow = torch.matmul(w_eras, Q_tilde)  # [B, K_eras]
                v[:, :K_eras] = v[:, :K_eras] + (alpha / 2.0) * era_flow

            # Blend expert log probabilities for token x_t:
            # p_t(x_t) = sum_i v(i) * P_i(x_t)
            # Log space sumexp for numerical stability
            v_log = torch.log(v + 1e-12)  # [B, K_total]
            ll_t = expert_target_ll[:, t, :]  # [B, K_total]
            log_blend = torch.logsumexp(v_log + ll_t, dim=-1)  # [B]
            tt_nll_list.append(-log_blend)

            # Bayes update: w_{t+1}(i) = v(i) * P_i(x_t) / p_t(x_t)
            unnorm_log = v_log + ll_t  # [B, K_total]
            w = F.softmax(unnorm_log, dim=-1)

        tt_nll = torch.stack(tt_nll_list, dim=1)  # [B, T]
        return tt_nll


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(a)
        return x + self.mlp(self.ln2(x))


class GPTCore(nn.Module):
    def __init__(self, d: int, layers: int = 4, heads: int = 4, max_seq: int = 2048):
        super().__init__()
        self.d = d
        self.layers_n = layers
        self.pos = nn.Parameter(torch.randn(max_seq, d) * 0.02)
        self.layers = nn.ModuleList([Block(d, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos[: x.shape[1]]
        for layer in self.layers:
            x = layer(x)
        return x

    def flops_per_token(self, seq: int) -> int:
        return 2 * self.layers_n * (12 * self.d ** 2 + seq * self.d)


class EraLM(nn.Module):
    """Era Language Model neural network + Era Bank + TT Filter."""

    def __init__(self, cfg: EraLMConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab = cfg.vocab
        self.d = cfg.d

        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        nn.init.normal_(self.emb.weight, std=0.02)

        self.core = GPTCore(d=cfg.d, layers=cfg.layers, heads=cfg.n_heads)
        self.ln = nn.LayerNorm(cfg.d)

        self.era_bank = EraBank(vocab=cfg.vocab, max_eras=cfg.max_eras, smoothing_a=cfg.smoothing_a)
        self.tt_filter = TimeTravellerFilter(alpha=cfg.alpha)

    def forward(self, tokens: torch.Tensor) -> dict:
        """Forward pass contract matching interface:
        model(tokens: LongTensor[B, T]) -> dict(nll=[B, T], tt_nll=[B, T], logits=[B, T, V])
        """
        B, T = tokens.shape
        device = tokens.device

        # If training mode, update era bank counts (causal per batch)
        if self.training:
            self.era_bank.update_counts_batch(tokens)

        # Compute neural model embeddings, core transformer, and logits
        x = self.emb(tokens)
        if self.cfg.grad_ckpt and self.training:
            x = torch.utils.checkpoint.checkpoint(self.core, x, use_reentrant=False)
        else:
            x = self.core(x)
        h = self.ln(x)
        logits = h @ self.emb.weight.T  # [B, T, V]

        # Shift logits for causal language modeling:
        # logits[:, t] predicts token t+1.
        # For t=0, target is tokens[:, 0] (no prediction before token 0, NLL = 0)
        # For t >= 1, logits[:, t-1] predicts tokens[:, t].
        log_probs_neural = F.log_softmax(logits, dim=-1)  # [B, T, V]

        # Compute neural NLL for t >= 1:
        nll_neural = torch.zeros((B, T), device=device)
        if T > 1:
            targets = tokens[:, 1:]  # [B, T-1]
            pred_log_probs = log_probs_neural[:, :-1, :]  # [B, T-1, V]
            nll_neural_shift = F.nll_loss(
                pred_log_probs.reshape(-1, self.vocab),
                targets.reshape(-1),
                reduction="none",
            ).reshape(B, T - 1)
            nll_neural[:, 1:] = nll_neural_shift

        # Compute Era Bank log-probabilities if T > 1
        active_eras_count = int(self.era_bank.active_mask.sum().item())
        if T > 1:
            prev_tokens = tokens[:, :-1]  # [B, T-1]
            targets = tokens[:, 1:]      # [B, T-1]

            # Era log-probabilities: [B, T-1, K_eras, V]
            era_log_probs = self.era_bank.predict_probs_batch(prev_tokens)

            # Combine experts: Era Bank + Neural expert
            # Neural log probs for shifted targets: [B, T-1, 1, V]
            neural_log_probs_expert = pred_log_probs.unsqueeze(2)
            expert_log_probs = torch.cat([era_log_probs, neural_log_probs_expert], dim=2)  # [B, T-1, K_eras+1, V]

            # Run TT momentum filter for t >= 1
            tt_nll_shift = self.tt_filter.forward_filter(
                expert_log_probs,
                targets,
                self.era_bank.era_switches,
                active_eras_count,
            )
            tt_nll = torch.zeros((B, T), device=device)
            tt_nll[:, 1:] = tt_nll_shift
        else:
            tt_nll = nll_neural.clone()

        return {
            "nll": nll_neural,
            "tt_nll": tt_nll,
            "logits": logits,
        }

    def n_params(self) -> int:
        """Returns number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_stats(self) -> int:
        """Returns number of non-zero count cells held by the era bank."""
        return self.era_bank.n_stats()

    def active_flops_per_token(self, seq: int = 256) -> int:
        """Forward FLOPs per token."""
        return self.core.flops_per_token(seq) + 2 * self.d * self.vocab

    def configure_optimizer(self, lr: float = 3e-3, weight_decay: float = 0.1) -> torch.optim.Optimizer:
        """Configures AdamW optimizer with weight decay applied appropriately."""
        decay = []
        no_decay = []
        for pn, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if pn.endswith(".bias") or "ln." in pn or "LayerNorm" in pn or "pos" in pn:
                no_decay.append(p)
            else:
                decay.append(p)

        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr)
        return optimizer

    def save(self, path: str, opt: torch.optim.Optimizer = None, extra: dict = None):
        """Saves model state, config, and era bank memory."""
        state = {
            "config": self.cfg,
            "model_state_dict": self.state_dict(),
        }
        if opt is not None:
            state["opt_state_dict"] = opt.state_dict()
        if extra is not None:
            state["extra"] = extra
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        """Loads model instance from saved state."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint["config"]
        model = cls(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model


def build(cfg: EraLMConfig = None, device: str = "cpu") -> EraLM:
    """Builds and returns an EraLM module on target device."""
    if cfg is None:
        cfg = EraLMConfig()
    model = EraLM(cfg)
    model.to(device)
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


def train_step(model: EraLM, opt: torch.optim.Optimizer, batch: torch.Tensor) -> dict:
    """Runs a single training step on batch [B, T+1]. Returns loss metrics."""
    model.train()
    opt.zero_grad()

    device = next(model.parameters()).device
    batch = batch.to(device)

    # Use amp autocast if configured
    if model.cfg.amp and device.type == "cuda":
        with torch.amp.autocast(device_type="cuda"):
            out = model(batch)
            loss = out["nll"][:, 1:].mean()
    else:
        out = model(batch)
        loss = out["nll"][:, 1:].mean()

    loss.backward()
    opt.step()

    tt_loss = out["tt_nll"][:, 1:].mean().item()
    n_eras = int(model.era_bank.active_mask.sum().item())
    single_frac = 1.0 if n_eras == 1 else 0.0

    lr = opt.param_groups[0]["lr"]

    return {
        "loss": loss.item(),
        "tt_loss": tt_loss,
        "n_eras": n_eras,
        "single_frac": single_frac,
        "lr": lr,
    }


def eval_loss(model: EraLM, batches) -> dict:
    """Evaluates loss over iterable of batches [B, T+1]. Returns aggregated metrics."""
    model.eval()
    total_nll = 0.0
    total_tt_nll = 0.0
    total_tokens = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            B, T = batch.shape
            n_tokens = B * (T - 1)
            out = model(batch)

            total_nll += out["nll"][:, 1:].sum().item()
            total_tt_nll += out["tt_nll"][:, 1:].sum().item()
            total_tokens += n_tokens

    mean_nll = total_nll / max(total_tokens, 1)
    mean_tt_nll = total_tt_nll / max(total_tokens, 1)

    n_eras = int(model.era_bank.active_mask.sum().item())
    single_frac = 1.0 if n_eras == 1 else 0.0

    return {
        "nll": mean_nll,
        "ppl": math.exp(min(mean_nll, 100)),
        "tt_nll": mean_tt_nll,
        "tt_ppl": math.exp(min(mean_tt_nll, 100)),
        "tokens": total_tokens,
        "single_frac": single_frac,
    }


def throughput(model: EraLM, batch: int = 32, seq: int = 256, steps: int = 20, train: bool = True) -> dict:
    """Measures model throughput in tokens/sec and step latency."""
    device = next(model.parameters()).device
    dummy_batch = torch.randint(0, model.vocab, (batch, seq + 1), device=device)
    opt = model.configure_optimizer() if train else None

    # Warmup steps
    for _ in range(3):
        if train:
            train_step(model, opt, dummy_batch)
        else:
            model.eval()
            with torch.no_grad():
                model(dummy_batch)

    if device.type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    import time
    start_time = time.time()

    for _ in range(steps):
        if train:
            train_step(model, opt, dummy_batch)
        else:
            model.eval()
            with torch.no_grad():
                model(dummy_batch)

    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        elapsed_ms = (time.time() - start_time) * 1000.0

    total_tokens = batch * seq * steps
    tokens_per_sec = total_tokens / (elapsed_ms / 1000.0)
    ms_per_step = elapsed_ms / steps
    peak_mem_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    return {
        "tokens_per_sec": tokens_per_sec,
        "peak_mem_bytes": peak_mem_bytes,
        "ms_per_step": ms_per_step,
        "breakdown": {"mode": "train" if train else "eval", "steps": steps, "batch": batch, "seq": seq},
    }


def generate(
    model: EraLM,
    prompt_ids: torch.Tensor,
    n_new: int = 128,
    temperature: float = 1.0,
    top_k: int = None,
) -> torch.Tensor:
    """Autoregressively generates n_new tokens following prompt_ids."""
    model.eval()
    device = next(model.parameters()).device
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_ids = prompt_ids.to(device)

    curr_tokens = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(n_new):
            out = model(curr_tokens)
            next_logits = out["logits"][:, -1, :]  # [B, V]
            if temperature > 0:
                next_logits = next_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

    return curr_tokens

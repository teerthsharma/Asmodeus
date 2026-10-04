"""Owned LM baselines for the QF-LM bake-off: pre-LN GPT decoder and GRU LM, both with tied embedding/head.

active_flops = multiply-accumulates per token of a forward pass (Linear in*out; embeddings, norms,
activations, softmax, GRU gate nonlinearities ignored), same convention as arms/__init__.py.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class _LM(nn.Module):
    """Shared contract. Subclasses set `window` (max context) and implement _logits(x [B, T]) -> [B, T, V]."""
    window = 1 << 30

    def _run(self, tokens):
        logits = self._logits(tokens[:, :-1])
        nll = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten(), reduction="none")
        return logits, nll.view(tokens.shape[0], -1)

    def forward(self, tokens):
        logits, nll = self._run(tokens)
        return dict(loss=nll.mean(), logits=logits)

    def nll(self, tokens):
        return self._run(tokens)[1]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # parameters() yields a tied weight once

    @torch.no_grad()
    def generate(self, prompt, n, temperature=1.0, top_k=None, generator=None):
        was = self.training
        self.eval()
        x = prompt
        for _ in range(n):
            lg = self._logits(x[:, -self.window:])[:, -1]
            if temperature == 0:
                nxt = lg.argmax(-1, keepdim=True)
            else:
                lg = lg / temperature
                if top_k is not None:
                    lg = lg.masked_fill(lg < lg.topk(top_k).values[:, -1:], -math.inf)
                nxt = torch.multinomial(lg.softmax(-1), 1, generator=generator)
            x = torch.cat([x, nxt], 1)
        self.train(was)
        return x


class _Block(nn.Module):
    def __init__(self, d, n_head, dropout):
        super().__init__()
        self.h, self.p = n_head, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0, is_causal=True)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)))))


class GPTLM(_LM):
    """GPT-2 style: learned positions, pre-LN, GELU MLP 4x, init std 0.02 (residual projections / sqrt(2L)).

    active_flops(T) = n_layer * (12 d^2 + 2 d (T+1)/2) + d V
      12 d^2 = qkv 3d^2 + proj d^2 + mlp 8d^2; 2 d (T+1)/2 = QK^T and AV averaged over a causal context of T.
    """

    def __init__(self, vocab, d_model, n_layer, n_head, seq_len, tie=True, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d, self.L, self.V, self.window = d_model, n_layer, vocab, seq_len
        self.tok, self.pos = nn.Embedding(vocab, d_model), nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(_Block(d_model, n_head, dropout) for _ in range(n_layer))
        self.ln_f, self.drop = nn.LayerNorm(d_model), nn.Dropout(dropout)
        self.head = nn.Linear(d_model, vocab, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        for b in self.blocks:
            for lin in (b.proj, b.out):
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * n_layer))
        if tie:
            self.head.weight = self.tok.weight

    def _logits(self, x):
        h = self.drop(self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device)))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln_f(h))

    def active_flops(self, T=None) -> int:
        T = T or self.window
        return self.L * (12 * self.d ** 2 + self.d * (T + 1)) + self.d * self.V


class GRULM(_LM):
    """Embedding -> GRU(n_layer) -> LayerNorm -> (Linear hidden->d_emb if hidden != d_emb) -> tied head.

    active_flops = 3 (d_emb h + h^2) + (n_layer-1) 3 (2 h^2) + [h d_emb if h != d_emb] + d_emb V
      (3 gates, each an input matmul and a recurrent matmul).
    """

    def __init__(self, vocab, d_emb, hidden, n_layer, tie=True):
        super().__init__()
        self.e, self.h, self.L, self.V = d_emb, hidden, n_layer, vocab
        self.tok = nn.Embedding(vocab, d_emb)
        self.rnn = nn.GRU(d_emb, hidden, n_layer, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.down = nn.Linear(hidden, d_emb, bias=False) if hidden != d_emb else nn.Identity()
        self.head = nn.Linear(d_emb, vocab, bias=False)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        if tie:
            self.head.weight = self.tok.weight

    def _logits(self, x):
        return self.head(self.down(self.ln(self.rnn(self.tok(x))[0])))

    def active_flops(self, T=None) -> int:
        e, h = self.e, self.h
        return (3 * (e * h + h * h) + (self.L - 1) * 6 * h * h
                + (h * e if h != e else 0) + e * self.V)

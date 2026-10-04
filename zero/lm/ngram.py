"""Jelinek-Mercer interpolated n-gram LM; also a qflm.generate stepper (calibration control arm)."""
from collections import Counter, defaultdict

import torch

from qflm.generate import NULL


class NGramLM:
    def __init__(self, vocab, order=4, lambdas=None):
        self.V, self.order = vocab, order
        self.lam = tuple(lambdas) if lambdas is not None else (0.1, 0.2, 0.3, 0.4)[:order]
        assert len(self.lam) == order
        self.c = defaultdict(Counter)  # context tuple (len 0..order-1) -> Counter(next id)
        self.toks = []

    def fit(self, seqs):
        for s in seqs:
            for t in range(len(s)):
                for k in range(self.order):
                    if t - k >= 0:
                        self.c[tuple(s[t - k:t])][s[t]] += 1

    def _p(self, hist):
        uni = self.c[()]
        tot = sum(uni.values())
        p0 = torch.full((self.V,), 1.0)
        for v, n in uni.items():
            p0[v] += n
        p, w = self.lam[0] * p0 / (tot + self.V), self.lam[0]
        for k in range(1, self.order):
            ctx = tuple(hist[len(hist) - k:]) if k <= len(hist) else None
            cc = self.c.get(ctx) if ctx is not None else None
            if not cc:
                continue
            n = sum(cc.values())
            pk = torch.zeros(self.V)
            for v, x in cc.items():
                pk[v] = x / n
            p, w = p + self.lam[k] * pk, w + self.lam[k]
        return p / w

    def nll(self, seq):
        return -sum(torch.log(self._p(seq[:t])[seq[t]]).item() for t in range(1, len(seq))) / (len(seq) - 1)

    def reset(self, prompt_ids):
        self.toks = list(prompt_ids)

    def dist(self):
        return self._p(self.toks), 0.0

    def observe(self, o):
        if o != NULL:
            self.toks.append(o)

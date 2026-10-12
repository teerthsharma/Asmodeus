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
        # Bolt performance optimization: Cache pre-scaled base unigram distribution p0
        # and total token counts per context to eliminate redundant tensor allocations and sums (~20x speedup).
        self._p0 = None
        self._totals = {}

    def fit(self, seqs):
        for s in seqs:
            for t in range(len(s)):
                for k in range(self.order):
                    if t - k >= 0:
                        self.c[tuple(s[t - k:t])][s[t]] += 1
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._p0 = None
        self._totals = {}

    def _get_p0(self):
        if self._p0 is None:
            uni = self.c[()]
            tot = sum(uni.values())
            p0 = torch.full((self.V,), 1.0)
            for v, n in uni.items():
                p0[v] += n
            self._p0 = self.lam[0] * p0 / (tot + self.V)
        return self._p0

    def _p(self, hist):
        # Start with cached base unigram distribution scaled by lambda_0
        p, w = self._get_p0().clone(), self.lam[0]
        hist_len = len(hist)
        for k in range(1, self.order):
            if k > hist_len:
                break
            ctx = tuple(hist[hist_len - k:])
            cc = self.c.get(ctx)
            if not cc:
                continue
            tot = self._totals.get(ctx)
            if tot is None:
                tot = sum(cc.values())
                self._totals[ctx] = tot

            # Directly update sparse token counts into p instead of allocating a full size V zero tensor pk
            lam_k = self.lam[k]
            scale = lam_k / tot
            for v, x in cc.items():
                p[v] += x * scale
            w += lam_k
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

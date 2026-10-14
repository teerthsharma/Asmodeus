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
        self._unigram_p = None
        self._ctx_cache = {}

    def fit(self, seqs):
        for s in seqs:
            for t in range(len(s)):
                for k in range(self.order):
                    if t - k >= 0:
                        self.c[tuple(s[t - k:t])][s[t]] += 1
        # Invalidate cached distributions on re-fit
        self._unigram_p = None
        self._ctx_cache = {}

    def _get_unigram_p(self):
        # Bolt performance optimization: Cache unigram probability vector
        # Avoid re-instantiating torch.full and looping over unigram counts per step
        if self._unigram_p is None:
            uni = self.c[()]
            tot = sum(uni.values())
            p0 = torch.ones(self.V, dtype=torch.float32)
            if uni:
                ks = list(uni.keys())
                vs = list(uni.values())
                p0[ks] += torch.tensor(vs, dtype=torch.float32)
            self._unigram_p = self.lam[0] * p0 / (tot + self.V)
        return self._unigram_p

    def _p(self, hist):
        # Bolt performance optimization: Start from cached unigram distribution
        # and vectorise higher-order n-gram probability insertion using slice/index assignment
        p = self._get_unigram_p().clone()
        w = self.lam[0]
        hist_len = len(hist)
        for k in range(1, self.order):
            if k > hist_len:
                break
            ctx = tuple(hist[hist_len - k:])
            if ctx in self._ctx_cache:
                pk = self._ctx_cache[ctx]
            else:
                cc = self.c.get(ctx)
                if not cc:
                    continue
                n = sum(cc.values())
                ks = list(cc.keys())
                vs = list(cc.values())
                pk = torch.zeros(self.V, dtype=torch.float32)
                pk[ks] = torch.tensor(vs, dtype=torch.float32) / n
                self._ctx_cache[ctx] = pk
            p += self.lam[k] * pk
            w += self.lam[k]
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

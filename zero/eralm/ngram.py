"""Owned n-gram baseline at corpus scale: the Jelinek-Mercer interpolation of lm/ngram.py (same smoothing: add-one
unigram, maximum-likelihood higher orders, an order's weight dropped when its context is unseen), vectorised with
numpy over exact int64 keys (12 bits per token, so order <= 5 needs no hashing).

The token stream is treated as one sequence (EOT is a token), so contexts may cross document boundaries.
"""
import numpy as np

BITS = 12


def _keys(x, k):
    """int64 key of the k tokens ending at each position i >= k-1 (x[i-k+1..i]); -1 before."""
    key = np.zeros(x.size, dtype=np.int64)
    for j in range(k):
        sh = np.full(x.size, -1, dtype=np.int64)
        sh[k - 1:] = x[k - 1 - j: x.size - j] if j else x[k - 1:]
        key = (key << BITS) | np.where(sh >= 0, sh, 0)
    key[: k - 1] = -1
    return key


class Counts:
    def __init__(self, keys):
        keys = keys[keys >= 0]
        self.u, self.c = np.unique(keys, return_counts=True)

    def get(self, q):
        i = np.searchsorted(self.u, q).clip(0, self.u.size - 1)
        return np.where(self.u[i] == q, self.c[i], 0)


class JMNGram:
    def __init__(self, vocab, order=4, lambdas=(0.1, 0.2, 0.3, 0.4)):
        assert order <= 5 and vocab <= 1 << BITS and len(lambdas) == order
        self.V, self.order, self.lam = vocab, order, np.asarray(lambdas, dtype=np.float64)

    def fit(self, x):
        x = np.asarray(x, dtype=np.int64)
        self.uni = np.bincount(x, minlength=self.V).astype(np.float64)
        self.tot = float(x.size)
        self.ctx, self.full = {}, {}
        for k in range(1, self.order):
            self.ctx[k] = Counts(_keys(x[:-1], k))      # context of length k followed by some token
            self.full[k] = Counts(_keys(x, k + 1))      # context + next token
        return self

    def logprob(self, x, lambdas=None):
        """log p(x_t | x_<t) for t = 1..len(x)-1 (contexts truncated at the start of x)."""
        lam = self.lam if lambdas is None else np.asarray(lambdas, dtype=np.float64)
        x = np.asarray(x, dtype=np.int64)
        y = x[1:]
        p = lam[0] * (self.uni[y] + 1) / (self.tot + self.V)
        w = np.full(y.size, lam[0])
        for k in range(1, self.order):
            ck = _keys(x[:-1], k)                       # context ending at t-1
            kk = np.where(ck >= 0, (y << (BITS * k)) | ck, -1)   # _keys puts the newest token highest
            n_ctx = np.where(ck >= 0, self.ctx[k].get(ck), 0)
            n_full = np.where(ck >= 0, self.full[k].get(kk), 0)
            seen = n_ctx > 0
            p = p + np.where(seen, lam[k] * n_full / np.maximum(n_ctx, 1), 0.0)
            w = w + np.where(seen, lam[k], 0.0)
        return np.log(p / w)

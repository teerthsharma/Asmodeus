"""Regular-language bed (PREREG Amendment 3): five stochastic sources over {a,b} with exact true log-probs.
Tokens a=0 b=1 EOS=2 BOS=3. A sequence is its symbols then EOS; after max_len symbols EOS is forced (prob 1)."""
import hashlib

import numpy as np
import torch

LANGS = ("parity", "last_a", "astar_bstar", "kgap3", "dyck4")
A, B, EOS, BOS = 0, 1, 2, 3


def _dist(lang, s, hist):
    """(p_a, p_b, p_eos) given scalar state s and the symbols so far."""
    if lang == "parity":
        return (.45, .45, .1) if s == 0 else (.5, .5, 0.)
    if lang == "last_a":
        return (.45, .45, .1) if s == A else (.5, .5, 0.)
    if lang == "astar_bstar":
        return (0., .8, .2) if s else (.45, .45, .1)
    if lang == "kgap3":
        if len(hist) < 3:
            return (.5, .5, 0.)
        c, f = .9 * 63 / 64, .1 * 63 / 64
        return (c, f, 1 / 64) if hist[-3] == A else (f, c, 1 / 64)
    if s == 0:  # dyck4: renormalise over allowed symbols
        return (2 / 3, 0., 1 / 3)
    return (0., 1., 0.) if s == 4 else (.5, .5, 0.)


def _step(lang, s, x):
    if lang == "parity":
        return s ^ (x == A)
    if lang == "last_a":
        return x
    if lang == "astar_bstar":
        return s | (x == B)
    return s + (1 if x == A else -1) if lang == "dyck4" else 0


def _s0(lang):
    return -1 if lang == "last_a" else 0


def sample(lang, n, seed, max_len=64):
    rng, out = np.random.default_rng(seed), []
    for _ in range(n):
        s, h = _s0(lang), []
        while True:
            p = (0., 0., 1.) if len(h) >= max_len else _dist(lang, s, h)
            x = int(rng.choice(3, p=p))
            if x == EOS:
                break
            h.append(x)
            s = _step(lang, s, x)
        out.append(h + [EOS])
    return out


def true_logprob(lang, seq, max_len=64):
    s, lp = _s0(lang), np.zeros(len(seq))
    for t, x in enumerate(seq):
        p = 1.0 if t >= max_len else _dist(lang, s, seq[:t])[x]
        with np.errstate(divide="ignore"):
            lp[t] = np.log(p)
        if x != EOS:
            s = _step(lang, s, x)
    return lp


def entropy_rate_estimate(lang, seqs, max_len=64):
    return -sum(true_logprob(lang, q, max_len).sum() for q in seqs) / sum(map(len, seqs))


def data_hash(seqs):
    return hashlib.sha256("\n".join(",".join(map(str, q)) for q in seqs).encode()).hexdigest()


def batch(seqs, bos=True, pad=EOS):
    """(tokens [B,T(+1)], mask [B,T]); mask[b,i] = i-th real symbol (incl. EOS) exists, aligned to scored positions."""
    T = max(map(len, seqs))
    x = torch.full((len(seqs), T + bos), pad, dtype=torch.long)
    m = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, q in enumerate(seqs):
        x[i, bos:bos + len(q)] = torch.tensor(q)
        m[i, :len(q)] = 1
    if bos:
        x[:, 0] = BOS
    return x, m

"""Era heads for language: closed-form ridge adapters on the TT's features, with Gaussian evidence for address, birth
and merge (the language analogue of the E0c count eras; ROADMAP stage 7).

Era k holds additive statistics of (phi_t, rho_t) pairs: A = sum phi phi^T, B = sum phi rho^T, c = sum |rho|^2, n.
- phi_t = [h_t, 1], the TT's final features.
- rho_t = 2 (e_{y_t} - p_t E) G^{-1}, G = E^T E: one Bohning-bound Newton step of the multinomial-logistic correction
  of the TT logits in hidden space (the softmax Hessian is <= 1/2 I, so the step is a majorise-minimise step).
- Readout W_k = (A_k + lam I)^{-1} B_k; era k predicts softmax((h + phi W_k) E^T): the TT, tilted by what era k knows.
Statistics only add: exact partial_fit = fit, non-destructive births, settled tokens never change (L40).

Evidence: Bayesian linear regression rho = phi W + N(0, sigma2), W ~ N(0, sigma2/lam), shared across d_out columns:
  log ML = -(n d_out/2) log(2 pi sigma2) + (d_out/2)(d_in log lam - log|A + lam I|) - (c - tr(B^T (A+lam I)^{-1} B))/(2 sigma2).
The address of a segment is argmax_k log ML(era_k + seg) - log ML(era_k), the fresh era scoring log ML(seg): the
prequential (chain-rule) likelihood, exactly the E0 address rule with Gaussian eras. Merge: global best pair by
Bayes factor, under the L39d threshold BF_j >= (j+1)(j+2)/delta (or BF > 1).
"""
import math

import torch

from .filter import Decider, log_share


def gauss_logml(A, B, c, n, lam, sigma2):
    """Batched log marginal likelihood. A [..., di, di], B [..., di, do], c [...], n [...] -> [...]."""
    di, do = B.shape[-2], B.shape[-1]
    I = torch.eye(di, dtype=A.dtype, device=A.device)
    L = torch.linalg.cholesky(A + lam * I)
    logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    Z = torch.linalg.solve_triangular(L, B, upper=False)           # L^{-1} B
    quad = (Z * Z).sum((-1, -2))                                    # tr(B^T (A + lam I)^{-1} B)
    return (-0.5 * n * do * math.log(2 * math.pi * sigma2) + 0.5 * do * (di * math.log(lam) - logdet)
            - (c - quad) / (2 * sigma2))


class RidgeBank(torch.nn.Module):
    """Shared, persistent bank of ridge eras + switch counts (MOM's kernel). Everything lives in buffers, so the bank
    is saved and moved with the model."""

    def __init__(self, d_in, d_out, n_max=16, lam=1.0, sigma2=0.0, merge_rule="anytime", merge_delta=0.05,
                 dtype=torch.float64):
        super().__init__()
        assert merge_rule in ("bf1", "anytime")
        self.d_in, self.d_out, self.n_max, self.lam = d_in, d_out, n_max, lam
        self.merge_rule, self.merge_delta = merge_rule, merge_delta
        K = n_max
        self.register_buffer("A", torch.zeros(K, d_in, d_in, dtype=dtype))
        self.register_buffer("B", torch.zeros(K, d_in, d_out, dtype=dtype))
        self.register_buffer("c", torch.zeros(K, dtype=dtype))
        self.register_buffer("n", torch.zeros(K, dtype=dtype))
        self.register_buffer("alive", torch.zeros(K, dtype=torch.bool))
        self.register_buffer("rank", torch.full((K,), 1 << 40, dtype=torch.long))
        self.register_buffer("S", torch.zeros(K, K, dtype=dtype))             # era -> era switch counts
        self.register_buffer("lm", torch.zeros(K, dtype=dtype))
        self.register_buffer("pj", torch.full((K, K), -1, dtype=torch.long))  # L39d pair indices
        self.register_buffer("W", torch.zeros(K, d_in, d_out))                 # readouts (float32)
        self.register_buffer("sigma2", torch.tensor(float(sigma2), dtype=dtype))
        self.register_buffer("counters", torch.zeros(4, dtype=torch.long))     # next_rank, n_pairs, births, folds

    @property
    def n_alive(self):
        return int(self.alive.sum())

    def n_stats(self):
        """Statistics held (floats): the non-parametric memory."""
        return self.n_alive * (self.d_in * self.d_in + self.d_in * self.d_out + 2)

    def _logml(self, A, B, c, n):
        return gauss_logml(A, B, c, n, self.lam, float(self.sigma2))

    def _refresh(self, ks):
        if not ks:
            return
        k = torch.tensor(sorted(ks), device=self.A.device)
        k = k[self.alive[k]]
        if k.numel() == 0:
            return
        I = torch.eye(self.d_in, dtype=self.A.dtype, device=self.A.device)
        self.lm[k] = self._logml(self.A[k], self.B[k], self.c[k], self.n[k])
        self.W[k] = torch.linalg.solve(self.A[k] + self.lam * I, self.B[k]).float()

    def order(self):
        idx = self.alive.nonzero().flatten()
        return idx[torch.argsort(self.rank[idx])]

    def log_share(self, d, n_tail):
        return log_share(self.alive, self.S, d, n_tail)

    @torch.no_grad()
    def credit(self, A, B, c, n, prev):
        """Address one segment per stream, credit it, then merge. A [b, di, di], B [b, di, do], c [b], n [b] (float64),
        prev [b] long (era each stream was last addressed to, -1 none). -> new prev [b]."""
        if float(self.sigma2) <= 0:                                  # fixed once, from the first segments
            self.sigma2.fill_(float((c.sum() / (n.sum() * self.d_out)).clamp_min(1e-8)))
        order = self.order()
        nb = A.shape[0]
        fresh = self._logml(A, B, c, n)                             # log ML(seg) - log ML(empty) (= 0)
        if len(order):
            joint = self._logml(self.A[order][None] + A[:, None], self.B[order][None] + B[:, None],
                                self.c[order][None] + c[:, None], self.n[order][None] + n[:, None])
            score = torch.cat([joint - self.lm[order][None], fresh[:, None]], 1)
        else:
            score = fresh[:, None]
        best = score.argmax(1).tolist()
        no = len(order)
        new_prev = prev.clone()
        dirty = set()
        for b in range(nb):
            j = best[b]
            full = self.n_alive >= self.n_max
            if j == no and full and no:                             # bank full: best existing era
                j = int(score[b, :-1].argmax())
            if j == no and full:                                    # filled within this window: newest era
                k = int(self.order()[-1])
            elif j == no:                                             # fresh era wins: birth
                k = int((~self.alive).nonzero().flatten()[0])
                self.alive[k] = True
                self.rank[k] = self.counters[0]
                self.counters[0] += 1
                self.counters[2] += 1
                self.A[k] = 0
                self.B[k] = 0
                self.c[k] = 0
                self.n[k] = 0
                self.S[k] = 0
                self.S[:, k] = 0
                self.pj[k] = -1
                self.pj[:, k] = -1
            else:
                k = int(order[j])
            self.A[k] += A[b]
            self.B[k] += B[b]
            self.c[k] += c[b]
            self.n[k] += n[b]
            dirty.add(k)
            p = int(prev[b])
            if p >= 0 and p != k and bool(self.alive[p]):
                self.S[p, k] += 1
            new_prev[b] = k
        self._refresh(dirty)
        return self._merge(new_prev)

    def _merge(self, prev):
        while True:
            order = self.order()
            m = len(order)
            if m <= 1:
                return prev
            iu, ju = torch.triu_indices(m, m, 1, device=self.A.device)
            oi, oj = order[iu], order[ju]
            bf = (self._logml(self.A[oi] + self.A[oj], self.B[oi] + self.B[oj], self.c[oi] + self.c[oj],
                              self.n[oi] + self.n[oj]) - self.lm[oi] - self.lm[oj])
            if self.merge_rule == "anytime":
                pj = self.pj[oi, oj]
                new = pj < 0
                if bool(new.any()):
                    nn = int(new.sum())
                    pj[new] = torch.arange(int(self.counters[1]), int(self.counters[1]) + nn, device=pj.device)
                    self.counters[1] += nn
                    self.pj[oi, oj] = pj
                    self.pj[oj, oi] = pj
                jf = pj.to(bf.dtype)
                bf = bf - torch.log((jf + 1) * (jf + 2) / self.merge_delta)
            top = bf.max()
            if (top <= 0) if self.merge_rule == "bf1" else (top < 0):
                return prev
            q = int((bf == top).nonzero().flatten()[-1])
            i, j = int(oi[q]), int(oj[q])
            self.A[i] += self.A[j]
            self.B[i] += self.B[j]
            self.c[i] += self.c[j]
            self.n[i] += self.n[j]
            self.S[i] += self.S[j]
            self.S[:, i] += self.S[:, j]
            self.S[i, i] = 0
            self.S[j] = 0
            self.S[:, j] = 0
            self.alive[j] = False
            self.rank[j] = 1 << 40
            self.pj[i] = -1
            self.pj[:, i] = -1
            self.counters[3] += 1
            prev = torch.where(prev == j, torch.full_like(prev, i), prev)
            self._refresh({i})

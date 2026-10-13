"""Era engine, pass 1: the E0c era bank (research/e0/arms.py `fcpa`, merge="global") and the BOCPD novelty expert
(`arms.bocpd`) for S streams at once, in torch.

The bookkeeping (detector, address, birth, global Bayes-factor merge) never reads the decider's weights, so it runs
first and emits (a) every expert's probability of the realised token and (b) an event log of births, switches and
folds. The decider (filter.py) then replays those as a forward filter. Hot path: one batched step per token over all
streams; cold path: a stream whose detector fires is re-addressed alone (alarms are rare: tens per 20k tokens).

Eras live in fixed slots (no index shifting); `rank` keeps the reference's era order so candidate and merge-pair order,
and therefore every tie-break, match the numpy code.
"""
from dataclasses import dataclass, field

import torch

A_PRIOR = 0.2
HIST = 4000          # arms.fcpa: hist = deque(maxlen=4000)
_BIG = 1 << 40


def bocpd(x, V, hazard, rmax=1500, a=A_PRIOR, block=None):
    """x [S, T] long -> [S, T-1] float64: BOCPD probability of x_t (t = 1..T-1), i.e. exp(-arms.bocpd(...)).

    Run length r uses the last r transitions; pi has the fixed length rmax+1 (zero padding is exact, see the
    reference's capped-bin merge). The run-length predictives of a block of steps do not depend on pi, so they are
    computed for the whole block at once; only the pi recursion runs token by token."""
    S, T = x.shape
    dev, f = x.device, torch.float64
    L = rmax + 1
    block = block or (64 if x.is_cuda else 8)
    pid = torch.full((S, T), -1, dtype=torch.long, device=dev)
    rid = torch.full((S, T), -1, dtype=torch.long, device=dev)
    pid[:, 1:] = x[:, :-1] * V + x[:, 1:]
    rid[:, 1:] = x[:, :-1]
    pad = torch.full((S, L), -1, dtype=torch.long, device=dev)
    pidf = torch.cat([pid.flip(1), pad], 1)       # pidf[:, T-1-s] = pid[:, s]
    ridf = torch.cat([rid.flip(1), pad], 1)
    pi = torch.zeros(S, L, dtype=f, device=dev)
    pi[:, 0] = 1.0
    out = torch.empty(S, T - 1, dtype=f, device=dev)
    for t0 in range(1, T, block):
        t1 = min(T, t0 + block)
        nb = t1 - t0
        # step t reads transitions s = t-1, ..., t-L+1, i.e. pidf[:, T-t : T-t+L-1]
        wp = pidf[:, T - t1 + 1: T - t0 + L - 1].unfold(1, L - 1, 1).flip(1)    # [S, nb, L-1], row i <-> t0+i
        wr = ridf[:, T - t1 + 1: T - t0 + L - 1].unfold(1, L - 1, 1).flip(1)
        cp = (wp == pid[:, t0:t1, None]).to(f).cumsum(2)
        cr = (wr == rid[:, t0:t1, None]).to(f).cumsum(2)
        pr = torch.cat([torch.full((S, nb, 1), 1.0 / V, dtype=f, device=dev),          # r = 0: empty run
                        (cp + a) / (cr + a * V)], 2)
        for i in range(nb):
            joint = pi * pr[:, i]
            p = joint.sum(1)
            out[:, t0 + i - 1] = p
            grow = joint * (1 - hazard)
            new = torch.empty_like(pi)
            new[:, 0] = hazard * p
            new[:, 1:] = grow[:, :-1]
            new[:, -1] += grow[:, -1]
            pi = new / p[:, None]
    return out


def logml(C, a=A_PRIOR):
    """Dirichlet-multinomial log marginal likelihood of count rows; C [..., V, V] -> [...] (arms._logml)."""
    V = C.shape[-1]
    a_ = torch.tensor(a, dtype=C.dtype, device=C.device)
    cell = (torch.lgamma(a + C) - torch.lgamma(a_)).sum((-1, -2))
    row = (torch.lgamma(V * a_) - torch.lgamma(V * a + C.sum(-1))).sum(-1)
    return cell + row


def window_probs(x, V, w, a=A_PRIOR):
    """[S, T-1]: window expert's probability of x_t; the window holds the last w transitions before t."""
    S, T = x.shape
    pid = torch.full((S, T), -1, dtype=torch.long, device=x.device)
    rid = torch.full((S, T), -1, dtype=torch.long, device=x.device)
    pid[:, 1:] = x[:, :-1] * V + x[:, 1:]
    rid[:, 1:] = x[:, :-1]
    out = torch.empty(S, T - 1, dtype=torch.float64, device=x.device)
    pad = torch.full((S, w), -1, dtype=torch.long, device=x.device)
    pp, rp = torch.cat([pad, pid], 1), torch.cat([pad, rid], 1)
    for c0 in range(1, T, 4096):                     # chunked: [S, chunk, w] compares
        c1 = min(T, c0 + 4096)
        wp = pp[:, c0:c1 + w - 1].unfold(1, w, 1)[:, : c1 - c0]      # transitions t-w .. t-1 for t in [c0, c1)
        wr = rp[:, c0:c1 + w - 1].unfold(1, w, 1)[:, : c1 - c0]
        cp = (wp == pid[:, c0:c1, None]).sum(-1).double()
        cr = (wr == rid[:, c0:c1, None]).sum(-1).double()
        out[:, c0 - 1:c1 - 1] = (cp + a) / (cr + a * V)
    return out


@dataclass
class BankOut:
    """Pass-1 output. Probabilities are of the realised token x_t, index t-1."""
    p_era: torch.Tensor      # [S, T-1, K] per era slot (dead slots: arbitrary)
    p_win: torch.Tensor      # [S, T-1] window expert
    p_ext: torch.Tensor      # [S, T-1] BOCPD expert
    hard: torch.Tensor       # [S, T-1] -log p of the active era (TT ALONE on the active era)
    events: list             # per stream: list of (t, kind, a, b); kind in birth/switch/fold
    n_eras: list             # per stream, at the end
    births: list             # per stream: birth times
    K: int                   # slot capacity


class EraBank:
    """E0c era bank for S streams. Defaults are the E0d/E0e configuration (DELTA, W, NMIN; lam 0.5; n_max 64)."""

    def __init__(self, V, delta=0.05, w=32, lam=0.5, n_max=64, n_min=64, a=A_PRIOR, K=16, merge_rule="anytime",
                 merge_delta=0.05):
        """merge_rule: "bf1" = merge while BF > 1 (E0c reference, no finite-sample guarantee); "anytime" = L39d,
        pair j (indexed as pairs are first tested) merges iff BF_j >= (j+1)(j+2)/merge_delta, so P(any false merge)
        <= merge_delta."""
        assert merge_rule in ("bf1", "anytime")
        self.V, self.delta, self.w, self.lam, self.n_max, self.n_min, self.a, self.K0 = V, delta, w, lam, n_max, n_min, a, K
        self.merge_rule, self.merge_delta = merge_rule, merge_delta

    def run(self, x, p_ext=None, block=256):
        """x [S, T] long; p_ext [S, T-1] BOCPD probabilities (or None: a constant expert) -> BankOut.

        Event-driven: between two alarms every quantity is a prefix sum, so each stream advances a whole block (or up to
        its next alarm) per iteration. Iterations ~ alarms + T/block, not T."""
        x = x.long()
        S, T = x.shape
        V, a, K = self.V, self.a, self.K0
        dev, f = x.device, torch.float64
        ar = torch.arange(S, device=dev)
        self.C = torch.zeros(S, K, V, V, dtype=f, device=dev)
        self.N = torch.zeros(S, K, V, dtype=f, device=dev)
        self.alive = torch.zeros(S, K, dtype=torch.bool, device=dev)
        self.alive[:, 0] = True
        self.rank = torch.full((S, K), _BIG, dtype=torch.long, device=dev)
        self.rank[:, 0] = 0
        self.next_rank = [1] * S
        self.lm = torch.full((S, K), float("nan"), dtype=f, device=dev)
        self.bf = torch.full((S, K, K), float("nan"), dtype=f, device=dev)
        self.pj = torch.full((S, K, K), -1, dtype=torch.long, device=dev)
        self.n_pairs = [0] * S
        self.era_of = torch.zeros(S, T, dtype=torch.long, device=dev)
        self.events = [[] for _ in range(S)]
        self.births = [[] for _ in range(S)]
        p_win = window_probs(x, V, self.w, a)
        p_era = torch.ones(S, T - 1, K, dtype=f, device=dev)
        hard = torch.empty(S, T - 1, dtype=f, device=dev)
        k = torch.zeros(S, dtype=torch.long, device=dev)
        ptr = torch.ones(S, dtype=torch.long, device=dev)
        Sd, n, cum, cmin = (torch.zeros(S, dtype=f, device=dev) for _ in range(4))
        tmin = torch.zeros(S, dtype=torch.long, device=dev)
        pending = torch.zeros(S, dtype=torch.bool, device=dev)
        thr = 1.0 / self.delta
        J = torch.arange(block, device=dev)
        tril = torch.ones(block, block, dtype=torch.bool, device=dev).tril(-1)
        while bool((ptr < T).any()):
            K = self.C.shape[1]
            if p_era.shape[2] < K:
                p_era = torch.cat([p_era, p_era.new_ones(S, T - 1, K - p_era.shape[2])], 2)
            t = ptr[:, None] + J                                   # [S, L]
            valid = t < T
            tc = t.clamp(max=T - 1)
            prev, cur = x.gather(1, tc - 1), x.gather(1, tc)
            pid = prev * V + cur
            Cg = self.C.view(S, K, V * V).gather(2, pid[:, None].expand(S, K, block)).transpose(1, 2).clone()
            Ng = self.N.gather(2, prev[:, None].expand(S, K, block)).transpose(1, 2).clone()
            c_in = ((pid[:, :, None] == pid[:, None, :]) & tril).sum(-1).to(f)   # earlier same pair in the block
            r_in = ((prev[:, :, None] == prev[:, None, :]) & tril).sum(-1).to(f)
            Cg.scatter_add_(2, k[:, None, None].expand(S, block, 1), c_in[..., None])
            Ng.scatter_add_(2, k[:, None, None].expand(S, block, 1), r_in[..., None])
            le = (Cg + a) / (Ng + a * V)                           # [S, L, K]
            A = le.gather(2, k[:, None, None].expand(S, block, 1))[..., 0]
            Wp = p_win.gather(1, tc - 1)
            loge = torch.log((self.lam * A + (1 - self.lam) * Wp) / A)
            loge = torch.where(valid, loge, torch.zeros_like(loge))
            cumj = torch.cumsum(torch.cat([cum[:, None], loge], 1), 1)[:, 1:]   # same rounding as the token loop
            nj = n[:, None] + (J + 1).to(f)
            logP = torch.cumsum(loge, 1)
            logPm1 = torch.cat([torch.zeros_like(logP[:, :1]), logP[:, :-1]], 1)
            Sj = torch.exp(logP) * (Sd[:, None] + torch.cumsum(torch.exp(-logPm1) / (nj * (nj + 1.0)), 1))
            pend = pending[:, None] | torch.cummax((Sj + 1.0 / (nj + 1.0) >= thr).to(torch.int8), 1)[0].bool()
            prevmin = torch.cummin(torch.cat([cmin[:, None], cumj], 1), 1)[0][:, :-1]
            newmin = cumj < prevmin
            tminj = torch.cummax(torch.where(newmin, t, torch.full_like(t, -1)), 1)[0]
            tminj = torch.where(tminj < 0, tmin[:, None], tminj)
            fire = pend & (t - tminj >= self.n_min) & valid
            fired = fire.any(1)
            m = torch.where(fired, fire.to(torch.int8).argmax(1) + 1, valid.sum(1))     # committed steps per stream
            cm = J[None] < m[:, None]
            si, ji = cm.nonzero(as_tuple=True)
            ti = t[si, ji]
            p_era[si, ti - 1, :K] = le[si, ji]
            hard[si, ti - 1] = -torch.log(A[si, ji])
            ks = k[si]
            one = torch.ones(si.numel(), dtype=f, device=dev)
            self.C.view(S, K, V * V).index_put_((si, ks, pid[si, ji]), one, accumulate=True)
            self.N.index_put_((si, ks, prev[si, ji]), one, accumulate=True)
            self.era_of[si, ti] = ks
            last = (m - 1).clamp_min(0)[:, None]
            adv = m > 0
            Sd = torch.where(adv, Sj.gather(1, last)[:, 0], Sd)
            n = torch.where(adv, nj.gather(1, last)[:, 0], n)
            cum = torch.where(adv, cumj.gather(1, last)[:, 0], cum)
            cmin = torch.where(adv, torch.minimum(prevmin, cumj).gather(1, last)[:, 0], cmin)
            tmin = torch.where(adv, tminj.gather(1, last)[:, 0], tmin)
            pending = torch.where(adv, pend.gather(1, last)[:, 0], pending)
            tf = ptr + m - 1
            ptr = ptr + m
            for s in fired.nonzero().flatten().tolist():
                k[s] = self._alarm(x[s], s, int(tf[s]), int(tmin[s]), int(k[s]))
                pending[s] = False
                Sd[s] = n[s] = cum[s] = cmin[s] = 0.0
                tmin[s] = tf[s]
        if p_ext is None:
            p_ext = torch.ones(S, T - 1, dtype=f, device=dev)
        K = self.C.shape[1]
        if p_era.shape[2] < K:
            p_era = torch.cat([p_era, p_era.new_ones(S, T - 1, K - p_era.shape[2])], 2)
        return BankOut(p_era, p_win, p_ext.to(f), hard, self.events, self.alive.sum(1).tolist(), self.births, K)

    def _order(self, s):
        idx = self.alive[s].nonzero().flatten()
        return idx[torch.argsort(self.rank[s, idx])]

    def _grow(self):
        S, K, V = self.C.shape[0], self.C.shape[1], self.V
        self.C = torch.cat([self.C, self.C.new_zeros(S, K, V, V)], 1)
        self.N = torch.cat([self.N, self.N.new_zeros(S, K, V)], 1)
        self.alive = torch.cat([self.alive, self.alive.new_zeros(S, K)], 1)
        self.rank = torch.cat([self.rank, torch.full_like(self.rank, _BIG)], 1)
        self.lm = torch.cat([self.lm, torch.full_like(self.lm, float("nan"))], 1)
        bf = torch.full((S, 2 * K, 2 * K), float("nan"), dtype=self.bf.dtype, device=self.bf.device)
        bf[:, :K, :K] = self.bf
        self.bf = bf
        pj = torch.full((S, 2 * K, 2 * K), -1, dtype=torch.long, device=self.pj.device)
        pj[:, :K, :K] = self.pj
        self.pj = pj

    def _refresh(self, s, dirty):
        """Recompute cached log ML of dirty slots and their pairwise log Bayes factors with every alive slot."""
        if not dirty:
            return
        a = self.a
        d = torch.tensor(sorted(dirty), device=self.C.device)
        d = d[self.alive[s, d]]
        if d.numel() == 0:
            return
        C = self.C[s]
        self.lm[s, d] = logml(C[d], a)
        al = self.alive[s].nonzero().flatten()
        both = logml(C[d][:, None] + C[al][None], a)                # [|d|, |al|]
        b = both - self.lm[s, d][:, None] - self.lm[s, al][None]
        self.bf[s, d[:, None], al[None]] = b
        self.bf[s, al[None], d[:, None]] = b

    def _new_identity(self, s, slot):
        """Slot holds a new era (birth or fold result): its merge-test pair indices start afresh (L39d)."""
        self.pj[s, slot] = -1
        self.pj[s, :, slot] = -1

    def _alarm(self, xs, s, t, tmin, k):
        """Re-address the segment since the CUSUM change estimate; birth or jump; then global merges. -> new k."""
        V, a = self.V, self.a
        lo = max(tmin, t - HIST)
        tt = torch.arange(lo + 1, t + 1, device=xs.device)
        if tt.numel() == 0:
            tt = torch.tensor([t], device=xs.device)
        pr, cu, eo = xs[tt - 1], xs[tt], self.era_of[s, tt]
        C, N = self.C[s], self.N[s]                                # views
        ones = torch.ones(tt.numel(), dtype=C.dtype, device=C.device)
        C.index_put_((eo, pr, cu), -ones, accumulate=True)
        N.index_put_((eo, pr), -ones, accumulate=True)
        G = torch.zeros(V, V, dtype=C.dtype, device=C.device).index_put_((pr, cu), ones, accumulate=True)
        dirty = set(eo.unique().tolist()) | {k}                   # the hot path only ever adds to the active era
        self._refresh(s, dirty)
        order = self._order(s)
        ll = logml(C[order] + G, a) - self.lm[s, order]
        if len(order) < self.n_max:
            ll = torch.cat([ll, (logml(G, a) - logml(torch.zeros_like(G), a))[None]])
        j = int(torch.argmax(ll))                                  # first maximum, as the reference's strict '>'
        if j == len(order):                                        # fresh era wins: birth
            free = (~self.alive[s]).nonzero().flatten()
            if free.numel() == 0:
                self._grow()
                free = (~self.alive[s]).nonzero().flatten()
                C, N = self.C[s], self.N[s]
            best = int(free[0])
            self.alive[s, best] = True
            self.rank[s, best] = self.next_rank[s]
            self.next_rank[s] += 1
            self._new_identity(s, best)
            self.births[s].append(t)
            self.events[s].append((t, "birth", best, -1))
        else:
            best = int(order[j])
        C[best] += G
        N[best] += G.sum(1)
        self.era_of[s, tt] = best
        self.events[s].append((t, "switch", k, best))
        k = best
        dirty = {best}
        while True:                                                # global merge: best pair while it clears the bar
            self._refresh(s, dirty)
            order = self._order(s)
            m = len(order)
            if m <= 1:
                break
            iu, ju = torch.triu_indices(m, m, 1, device=C.device)
            oi, oj = order[iu], order[ju]
            score = self.bf[s, oi, oj]
            if self.merge_rule == "anytime":                       # L39d: BF_j >= (j+1)(j+2)/delta
                pj = self.pj[s, oi, oj]
                new = pj < 0
                if bool(new.any()):
                    nn = int(new.sum())
                    pj[new] = torch.arange(self.n_pairs[s], self.n_pairs[s] + nn, device=C.device)
                    self.n_pairs[s] += nn
                    self.pj[s, oi, oj] = pj
                    self.pj[s, oj, oi] = pj
                jf = pj.to(score.dtype)
                score = score - torch.log((jf + 1) * (jf + 2) / self.merge_delta)
            best_v = score.max()
            if best_v <= 0 if self.merge_rule == "bf1" else best_v < 0:
                break
            q = int((score == best_v).nonzero().flatten()[-1])     # ties: largest (i, j), as max() over tuples
            si, sj = int(oi[q]), int(oj[q])
            C[si] += C[sj]
            N[si] += N[sj]
            C[sj] = 0
            N[sj] = 0
            self.alive[s, sj] = False
            self.rank[s, sj] = _BIG
            self.lm[s, sj] = float("nan")
            self._new_identity(s, si)
            row = self.era_of[s, : t + 1]
            row[row == sj] = si
            if k == sj:
                k = si
            self.events[s].append((t, "fold", si, sj))
            dirty = {si}
        return k

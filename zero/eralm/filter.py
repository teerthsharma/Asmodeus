"""Era engine, pass 2: the TT decider as a forward filter over the experts [era slots..., window, BOCPD].

w_t (the chronal fluid density) moves by w <- E_t((w M_t) * l_t^eta) up to scale: share (M_t, row-stochastic), Bayes
step (multiply by the likelihood l_t, normalise), then the bank's events at t (birth, fold) as linear maps E_t. M_t
changes only at events, so the whole filter is a product of per-step matrices: an HMM forward pass (L41). Only the
direction of w matters, so products are rescaled by their max and no normaliser is tracked.

`scan` runs it as a chunked parallel scan: (1) every chunk's transfer matrix in parallel, (2) a short sequential
pass over chunk boundaries, (3) every chunk's interior in parallel. Python iterations: 2*chunk + T/chunk, not T.
`sequential` is the plain token loop, kept as the test oracle.

Deciders (research/e0/deciders.py): SOFT, MOM, HARD (thr), TEMP (T), GBAYES (eta), MOM+GB (eta). CTC needs every
expert's full predictive vector and lost the tournament; it is not ported.
"""
from dataclasses import dataclass

import torch

N_TAIL = 2


@dataclass(frozen=True)
class Decider:
    name: str              # SOFT | MOM | HARD | TEMP | GBAYES | MOM+GB
    alpha: float
    p: float = None        # HARD thr, TEMP T, GBAYES/MOM+GB eta

    @property
    def momentum(self):
        return self.name in ("MOM", "MOM+GB")

    @property
    def eta(self):
        return self.p if self.name in ("GBAYES", "MOM+GB") else 1.0


def _share(alive, Smat, d):
    """Share matrix over n = K + N_TAIL slots for one epoch. alive [K] bool, Smat [K, K] switch counts."""
    K = alive.numel()
    n = K + N_TAIL
    al = torch.cat([alive, alive.new_ones(N_TAIL)]).to(Smat.dtype)
    m = al.sum()
    a = d.alpha
    eye = torch.diag(al)
    out = (1 - a) * eye
    if not d.momentum:
        return out + (a / m) * torch.outer(al, al)
    out = out + (0.5 * a / m) * torch.outer(al, al)
    mv = eye.clone()
    ne = int(alive.sum())
    if ne > 1:
        Q = (Smat + 1.0) * torch.outer(alive, alive).to(Smat.dtype)
        Q.fill_diagonal_(0.0)
        Q = Q / Q.sum(1, keepdim=True).clamp_min(1e-300)
        mv[:K, :K] = Q
    return out + 0.5 * a * mv


def _event_map(kind, a_, b_, K, alpha, dtype, dev):
    n = K + N_TAIL
    E = torch.eye(n, dtype=dtype, device=dev)
    if kind == "birth":                       # Soft.birth: old mass * (1-alpha), new slot gets alpha * total
        E = (1 - alpha) * E
        E[:, a_] = alpha
    elif kind == "fold":                      # mass of slot b moves to slot a
        E[b_, b_] = 0.0
        E[b_, a_] = 1.0
    return E


def epochs(out, d):
    """Per stream: M per epoch [S, nE, n, n], epoch id per step [S, T-1], event maps {(s, i): E}."""
    S, Tm1, K = out.p_era.shape
    dev, f = out.p_era.device, torch.float64
    Ms, eid, ev = [], torch.zeros(S, Tm1, dtype=torch.long, device=dev), {}
    for s in range(S):
        alive = torch.zeros(K, dtype=torch.bool, device=dev)
        alive[0] = True
        Sm = torch.zeros(K, K, dtype=f, device=dev)
        mats = [_share(alive, Sm, d)]
        by_t = {}
        for (t, kind, a_, b_) in out.events[s]:
            by_t.setdefault(t, []).append((kind, a_, b_))
        for t in sorted(by_t):
            i = t - 1
            E = torch.eye(K + N_TAIL, dtype=f, device=dev)
            for kind, a_, b_ in by_t[t]:
                if kind == "birth":
                    alive[a_] = True
                    Sm[a_] = 0
                    Sm[:, a_] = 0
                elif kind == "switch":
                    if a_ != b_:
                        Sm[a_, b_] += 1
                elif kind == "fold":
                    Sm[a_] += Sm[b_]
                    Sm[:, a_] += Sm[:, b_]
                    Sm[a_, a_] = 0
                    Sm[b_] = 0
                    Sm[:, b_] = 0
                    alive[b_] = False
                E = E @ _event_map(kind, a_, b_, K, d.alpha, f, dev)
            if not torch.equal(E, torch.eye(K + N_TAIL, dtype=f, device=dev)):
                ev[(s, i)] = E
            mats.append(_share(alive, Sm, d))
            eid[s, i + 1:] = len(mats) - 1
        Ms.append(torch.stack(mats))
    nE = max(m.shape[0] for m in Ms)
    M = torch.zeros(S, nE, K + N_TAIL, K + N_TAIL, dtype=f, device=dev)
    for s, m in enumerate(Ms):
        M[s, : m.shape[0]] = m
    return M, eid, ev


def likelihoods(out):
    """[S, T-1, n]: realised-token probability per expert slot."""
    return torch.cat([out.p_era, out.p_win[..., None], out.p_ext[..., None]], -1)


def _answer(v, l, d):
    """v [..., n] normalised predictive weights, l [..., n] -> (p [...], single [...] bool)."""
    if d.name == "HARD":
        vm, km = v.max(-1)
        single = vm >= d.p
        return torch.where(single, l.gather(-1, km[..., None])[..., 0], (v * l).sum(-1)), single
    if d.name == "TEMP":
        u = v ** (1.0 / d.p)
        return (u / u.sum(-1, keepdim=True) * l).sum(-1), torch.zeros_like(v[..., 0], dtype=torch.bool)
    return (v * l).sum(-1), torch.zeros_like(v[..., 0], dtype=torch.bool)


def _w0(S, K, dev):
    w = torch.zeros(S, K + N_TAIL, dtype=torch.float64, device=dev)
    w[:, 0] = 1.0
    w[:, K:] = 1.0
    return w / w.sum(1, keepdim=True)


def sequential(out, d):
    """Token loop: O(n^2) work per token, T Python steps. -> (loss [S, T-1] = -log p, single fraction [S])."""
    S, Tm1, K = out.p_era.shape
    M, eid, ev = epochs(out, d)
    L = likelihoods(out)
    ar = torch.arange(S, device=L.device)
    w = _w0(S, K, L.device)
    loss = torch.empty(S, Tm1, dtype=torch.float64, device=L.device)
    single = torch.zeros(S, dtype=torch.float64, device=L.device)
    by_i = {}
    for (s, i), E in ev.items():
        by_i.setdefault(i, []).append((s, E))
    by_i = {i: (torch.tensor([e[0] for e in v], device=L.device), torch.stack([e[1] for e in v])) for i, v in by_i.items()}
    eta = d.eta
    for i in range(Tm1):
        v = torch.bmm(w[:, None], M[ar, eid[:, i]])[:, 0]
        l = L[:, i]
        p, sg = _answer(v, l, d)
        loss[:, i] = -torch.log(p)
        single += sg
        post = v * l if eta == 1.0 else v * l ** eta
        w = post / post.sum(1, keepdim=True)
        if i in by_i:
            si, E = by_i[i]
            w[si] = torch.bmm(w[si, None], E)[:, 0]
    return loss, single / Tm1


def run(out, d, method="auto", chunk=128):
    """Decider losses. 'auto': the chunked scan on CUDA (few launches), the token loop on CPU (less work: the scan
    multiplies n x n matrices, the loop only vectors)."""
    if method == "auto":
        method = "scan" if out.p_era.is_cuda else "sequential"
    return scan(out, d, chunk) if method == "scan" else sequential(out, d)


def scan(out, d, chunk=128):
    """Chunked parallel scan; same result as `sequential` up to float rounding."""
    S, Tm1, K = out.p_era.shape
    n = K + N_TAIL
    dev, f = out.p_era.device, torch.float64
    M, eid, ev = epochs(out, d)
    L = likelihoods(out)
    nC = -(-Tm1 // chunk)
    pad = nC * chunk - Tm1
    if pad:                                                    # padded steps: identity transfer, ignored outputs
        L = torch.cat([L, L.new_ones(S, pad, n)], 1)
        eid = torch.cat([eid, eid[:, -1:].expand(S, pad)], 1)
    Lc = L.view(S, nC, chunk, n)
    ec = eid.view(S, nC, chunk)
    valid = (torch.arange(nC * chunk, device=dev) < Tm1).view(nC, chunk)
    ar = torch.arange(S, device=dev)[:, None]
    evs = {}                                                   # local position -> (s idx, chunk idx, E stack)
    for (s, i), E in ev.items():
        evs.setdefault(i % chunk, []).append((s, i // chunk, E))
    evs = {j: (torch.tensor([e[0] for e in v], device=dev), torch.tensor([e[1] for e in v], device=dev),
               torch.stack([e[2] for e in v])) for j, v in evs.items()}
    eta = d.eta

    def step_mat(j):
        A = M[ar, ec[:, :, j]] * (Lc[:, :, j, None, :] ** eta)  # M diag(l^eta)
        A = torch.where(valid[None, :, j, None, None], A, torch.eye(n, dtype=f, device=dev))
        if j in evs:
            si, ci, E = evs[j]
            A[si, ci] = A[si, ci] @ E
        return A

    P = torch.eye(n, dtype=f, device=dev).expand(S, nC, n, n).clone()
    for j in range(chunk):                                     # (1) chunk transfer matrices, all chunks at once
        P = P @ step_mat(j)
        P = P / P.amax((-1, -2), keepdim=True)
    W = torch.empty(S, nC, n, dtype=f, device=dev)            # (2) w at each chunk start
    w = _w0(S, K, dev)
    for c in range(nC):
        W[:, c] = w
        w = torch.einsum("sn,snm->sm", w, P[:, c])
        w = w / w.sum(1, keepdim=True)
    loss = torch.empty(S, nC, chunk, dtype=f, device=dev)     # (3) interiors, all chunks at once
    single = torch.zeros(S, nC, chunk, dtype=torch.bool, device=dev)
    w = W
    for j in range(chunk):
        v = torch.einsum("scn,scnm->scm", w, M[ar, ec[:, :, j]])
        l = Lc[:, :, j]
        p, sg = _answer(v, l, d)
        loss[:, :, j] = -torch.log(p)
        single[:, :, j] = sg
        post = v * l ** eta
        if j in evs:
            si, ci, E = evs[j]
            post[si, ci] = torch.einsum("en,enm->em", post[si, ci], E)
        w = post / post.sum(-1, keepdim=True)
    loss = loss.view(S, -1)[:, :Tm1]
    single = single.view(S, -1)[:, :Tm1]
    return loss, single.to(f).mean(1)

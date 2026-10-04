"""Quantum-filter LM primitives: instrument K_v = diag(sqrt(M[:,v])) Q, Q a butterfly unitary."""
import math
import torch

EPS = 1e-12


def _cdt(rd):
    return torch.complex128 if rd == torch.float64 else torch.complex64


def butterfly(psi, theta, phi):
    """Apply Q in O(N log N). theta, phi: (L, N/2) or batched (B, L, N/2)."""
    B, N = psi.shape
    for k in range(theta.shape[-2]):
        s = 1 << k
        t = theta[..., k, :].reshape(*theta.shape[:-2], N // (2 * s), s)
        e = torch.polar(torch.ones_like(t), phi[..., k, :].reshape(t.shape))
        x = psi.reshape(B, N // (2 * s), 2, s)
        a, b = x[:, :, 0], x[:, :, 1]
        c, sn = torch.cos(t), torch.sin(t)
        psi = torch.stack([c * a - e.conj() * sn * b, e * sn * a + c * b], 2).reshape(B, N)
    return psi


def butterfly_matrix(theta, phi, N):
    """Dense Q (tests only)."""
    return butterfly(torch.eye(N, dtype=_cdt(theta.dtype)), theta, phi).T


def _w(z):
    return z.real ** 2 + z.imag ** 2


def probs(psi, M, theta, phi):
    return _w(butterfly(psi, theta, phi)) @ M


def luders_update(psi, v, M, theta, phi):
    out = M.gather(1, v[None].expand(M.shape[0], -1)).T.sqrt() * butterfly(psi, theta, phi)  # (B,N)
    n = _w(out).sum(1, keepdim=True).clamp_min(EPS ** 2).sqrt()
    return out / n


def think_step(psi, think_params):
    th0, ph0, At, Ap = think_params
    w = _w(psi)  # |psi|^2 only: global-phase invariant
    return butterfly(psi, th0 + (w @ At.T).reshape(-1, *th0.shape), ph0 + (w @ Ap.T).reshape(-1, *ph0.shape))


def nll(tokens, M, theta, phi, think_params, s_max=4, psi0=None):
    """Mean NLL (nats/token). Round 0 measures directly; each later round = think_step then measure.
    Round s emits x w.p. P(null^s) p_s(x); at s_max, p_s(x)/(1-p_null) (forced emission).
    Post-emission state: normalised sum_s sqrt(weight_s) * normalised(K_x psi_s)."""
    B, T = tokens.shape
    N, V1 = M.shape
    V = V1 - 1
    if psi0 is None:
        psi0 = torch.zeros(N, dtype=_cdt(theta.dtype)); psi0[0] = 1
    psi = psi0.expand(B, N)
    nulls = torch.full((B,), V, dtype=torch.long)
    total = 0.0
    for t in range(T):
        x = tokens[:, t]
        logw = torch.zeros(B, dtype=M.dtype)
        terms, posts, ws = [], [], []
        for s in range(s_max + 1):
            if s:
                psi = think_step(psi, think_params)
            p = probs(psi, M, theta, phi)
            px = p.gather(1, x[:, None])[:, 0]
            if s == s_max:
                px = px / (1 - p[:, V]).clamp_min(EPS)
            lt = logw + px.clamp_min(1e-30).log()
            terms.append(lt)
            ws.append(lt.exp())
            posts.append(luders_update(psi, x, M, theta, phi))
            if s < s_max:
                logw = logw + p[:, V].clamp_min(1e-30).log()
                psi = luders_update(psi, nulls, M, theta, phi)
        total = total - torch.logsumexp(torch.stack(terms), 0).sum()
        mix = sum(w.clamp_min(1e-30).sqrt()[:, None] * q for w, q in zip(ws, posts))
        psi = mix / _w(mix).sum(1, keepdim=True).clamp_min(EPS ** 2).sqrt()
    return total / (B * T)


def max_reachable_fraction(M, thresh=0.5):
    return (M[:, :-1].max(0).values >= thresh).float().mean().item()


def capacity_M(N, V):
    """Row i puts 0.5/0.5 on tokens (2i)%V, (2i+1)%V (1.0 if equal); null column 0."""
    M = torch.zeros(N, V + 1)
    i = torch.arange(N)
    M[i, (2 * i) % V] += 0.5
    M[i, (2 * i + 1) % V] += 0.5
    return M

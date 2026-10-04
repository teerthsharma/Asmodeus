"""Eval-only readouts on a QF-LM state psi in CP^{N-1} (complex unit vector modulo global phase).
`state_fn(token_ids) -> complex vector`. Pure numpy; nothing here is trained or differentiated.

Bargmann sign convention: bargmann([a, b, c]) = arg(<a|b><b|c><c|a>) = +Omega/2 (mod 2pi), Omega = signed solid
angle of the geodesic triangle a->b->c on the Bloch sphere (counter-clockwise seen from outside = +). The Berry
phase is minus this. Check: z -> x -> y octant gives +pi/4.
"""
import numpy as np
from scipy.stats import spearmanr

_MIN_OVERLAP = 1e-3


def _unit(v):
    v = np.asarray(v, complex).ravel()
    return v / np.linalg.norm(v)


def _states(state_fn, seqs):
    return [_unit(state_fn(tuple(s))) for s in seqs]


def bargmann(states):
    """arg prod_i <psi_i|psi_{i+1}> over the closed loop (last wraps to first). Gauge invariant."""
    s = [_unit(v) for v in states]
    z = 1.0 + 0j
    for a, b in zip(s, s[1:] + s[:1]):
        z *= np.vdot(a, b)
    return float(np.angle(z))


def _min_overlap(states):
    return min(abs(np.vdot(a, b)) for a, b in zip(states, states[1:] + states[:1]))


def exchange_phase(state_fn, A, B, prefix=(), suffix=()):
    """-> (theta = arg<psi_AB|psi_BA>, |overlap|). Gauge-DEPENDENT: state_fn may return psi up to a global phase
    per call, which shifts theta arbitrarily. Use exchange_loop for the invariant."""
    ab, ba = _states(state_fn, [(*prefix, *A, *B, *suffix), (*prefix, *B, *A, *suffix)])
    o = np.vdot(ab, ba)
    return float(np.angle(o)), float(abs(o))


def exchange_loop(state_fn, A, B, prefix=(), suffix=(), ref=None):
    """Gauge-invariant exchange phase: bargmann([psi_AB, psi_BA, psi_ref]), psi_ref = psi(prefix+suffix) by default.
    -> dict(phase, min_overlap); phase is nan if any overlap < 1e-3."""
    ref = (*prefix, *suffix) if ref is None else tuple(ref)
    st = _states(state_fn, [(*prefix, *A, *B, *suffix), (*prefix, *B, *A, *suffix), ref])
    mn = _min_overlap(st)
    return dict(phase=bargmann(st) if mn >= _MIN_OVERLAP else float("nan"), min_overlap=float(mn))


def holonomy_loop(state_fn, paraphrases):
    """-> (bargmann phase over the closed loop of paraphrase states, min |overlap|); phase nan if min < 1e-3."""
    st = _states(state_fn, paraphrases)
    mn = float(_min_overlap(st))
    return (bargmann(st) if mn >= _MIN_OVERLAP else float("nan")), mn


def think_vs_surprisal(think_steps, surprisal, n_boot=2000, seed=0):
    """-> (Spearman rho, lo, hi): 95% percentile bootstrap over items."""
    x, y = np.asarray(think_steps, float), np.asarray(surprisal, float)
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        with np.errstate(all="ignore"):
            bs.append(spearmanr(x[i], y[i])[0])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return rho, float(lo), float(hi)


def _resultant(p):
    z = np.exp(1j * p).mean()
    return float(np.angle(z)), float(abs(z))


def circular_stats(phases, n_boot=2000, seed=0):
    """-> (circular mean, resultant length R, (lo, hi) bootstrap 95% CI on R); nan entries dropped."""
    p = np.asarray(phases, float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    mean, R = _resultant(p)
    rng = np.random.default_rng(seed)
    bs = [_resultant(p[rng.integers(0, p.size, p.size)])[1] for _ in range(n_boot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return mean, R, (float(lo), float(hi))


circular_summary = circular_stats


def summary_table(state_fn, exchange_items, loops, seed=0):
    """Per-readout circular summary: exchange_phase (gauge-dependent, reference only), exchange_loop, holonomy."""
    ex = [exchange_phase(state_fn, it["A"], it["B"], it.get("prefix", ()), it.get("suffix", ()))[0] for it in exchange_items]
    el = [exchange_loop(state_fn, it["A"], it["B"], it.get("prefix", ()), it.get("suffix", ()))["phase"] for it in exchange_items]
    ho = [holonomy_loop(state_fn, lp)[0] for lp in loops]
    out = {}
    for k, v in (("exchange_phase", ex), ("exchange_loop", el), ("holonomy", ho)):
        m, R, ci = circular_stats(v, seed=seed)
        out[k] = dict(n=int(np.isfinite(v).sum()), mean=m, R=R, ci=ci)
    return out

"""The first-class candidates behind one constructor (PREREG 'Candidates and scoring matrix').

Runners read z/zidx from forward() when an arm emits them; there is no hook. All three are built from the same
shared knobs: D, A, manifold dim n (d_c = n, D_g = 2n + 2, PREREG pin), K, m.
"""
CANDIDATES = ("atlas", "phase_sphere", "asmodeous")  # HYBRID withdrawn (attic/)

_TINY = dict(
    atlas=dict(hidden=8, d_h=8, d_key=8),
    phase_sphere=dict(N=4, hidden=8),
    asmodeous=dict(N=4, hidden=8),
)


def build(name, D, A, n, K=None, m=2, tiny=False, **over):
    """-> arm for `name`; `over` goes last (runners override lam, ema, ...). Imports are lazy per candidate.
    A scalar `lam` is the SIGReg weight for every candidate: PhaseSphere keeps a per-term dict, so it is
    expanded there (sig=lam, prediction terms (1 - lam)), matching its own default table."""
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}; one of {CANDIDATES}")
    kw = dict(K=K or 8, m=m, d_c=n, D_g=2 * n + 2, **(_TINY[name] if tiny else {}))
    if name in _WEIGHTS and not isinstance(over.get("lam", {}), dict):
        lam = over["lam"]
        over = {**over, "lam": {k: (lam if k == "sig" else (1 - lam) * w) for k, w in _WEIGHTS[name]().items()}}
    kw.update(over)
    if name == "atlas":
        from atlas.observer_atlas import ObserverAtlas as cls
    elif name == "asmodeous":
        from atlas.asmodeous import Asmodeous as cls
    else:
        from atlas.phase_sphere import PhaseSphere as cls
    return cls(D, A, **kw)


def _phase_weights():
    from sigreg import LAM, LAM_BAL
    return dict(jepa=1.0, overlap=0.1, task=1.0, bal=LAM_BAL, kl=1.0, sig=LAM)


def _asmodeous_weights():
    from sigreg import LAM, LAM_BAL
    return dict(fid=1.0, chart=1.0, cocycle=0.1, task=1.0, sig=LAM, haar=0.1, bal=LAM_BAL, distill=1.0)


_WEIGHTS = dict(phase_sphere=_phase_weights, asmodeous=_asmodeous_weights)


def shared_knobs(name):
    """Attribute/kwarg names for the SIGReg scale and the online-target EMA momentum (identical on all candidates)."""
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}")
    return {"lam": "lam", "momentum": "ema"}

"""The first-class candidates behind one constructor (PREREG 'Candidates and scoring matrix').

Runners read z/zidx from forward() when an arm emits them; there is no hook. All three are built from the same
shared knobs: D, A, manifold dim n (d_c = n, D_g = 2n + 2, PREREG pin), K, m.
"""
CANDIDATES = ("atlas", "phase_sphere")  # HYBRID withdrawn (attic/); ASMODEOUS added when atlas/asmodeous.py lands

_TINY = dict(
    atlas=dict(hidden=8, d_h=8, d_key=8),
    phase_sphere=dict(N=4, hidden=8),
)


def build(name, D, A, n, K=None, m=2, tiny=False, **over):
    """-> arm for `name`; `over` goes last (runners override lam, ema, ...). Imports are lazy per candidate.
    A scalar `lam` is the SIGReg weight for every candidate: PhaseSphere keeps a per-term dict, so it is
    expanded there (sig=lam, prediction terms (1 - lam)), matching its own default table."""
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}; one of {CANDIDATES}")
    kw = dict(K=K or 8, m=m, d_c=n, D_g=2 * n + 2, **(_TINY[name] if tiny else {}))
    if name == "phase_sphere" and not isinstance(over.get("lam", {}), dict):
        lam = over["lam"]
        over = {**over, "lam": {k: (lam if k == "sig" else (1 - lam) * w) for k, w in _phase_weights().items()}}
    kw.update(over)
    if name == "atlas":
        from atlas.observer_atlas import ObserverAtlas as cls
    else:
        from atlas.phase_sphere import PhaseSphere as cls
    return cls(D, A, **kw)


def _phase_weights():
    from sigreg import LAM, LAM_BAL
    return dict(jepa=1.0, overlap=0.1, task=1.0, bal=LAM_BAL, kl=1.0, sig=LAM)


def shared_knobs(name):
    """Attribute/kwarg names for the SIGReg scale and the online-target EMA momentum (identical on all three)."""
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}")
    return {"lam": "lam", "momentum": "ema"}

"""L-BR sizing: pick each arm's widths so active_flops AND n_params land within tol of the targets.

Two knobs (one mostly drives active compute, one drives params) can hit both targets only if the arm's
params/FLOPs ratio is free. Dense arms (single JEPA, chart-AE, non-lazy MoP) have params ~= active MACs,
so against a lazy atlas (params >> FLOPs) they cannot match both; then match() returns two baselines:
flop_matched (equal compute, fewer params) and param_matched (equal params, more compute). Both are run;
the atlas must beat each on the axis it is matched on. PEER-JEPA and lazy MoP-JEPA can match both.
Models are built on the meta device: sizing costs no memory.
"""
import torch

from arms.chart_ae import ChartAEJEPA
from arms.lm_baselines import GPTLM, GRULM
from arms.lazy_chart_ae import LazyChartAEJEPA
from arms.mop_jepa import MoPJEPA
from arms.peer_jepa import PEERJEPA
from arms.single_jepa import SingleJEPA

# arm -> (class, fixed kwargs, [(knob, value -> kwargs)])
SPECS = {
    "single": (SingleJEPA, {}, [("hidden", lambda v: {"hidden": v})]),
    "mop": (MoPJEPA, {}, [("hidden", lambda v: {"hidden": v}), ("head_hidden", lambda v: {"head_hidden": v})]),
    "mop_lazy": (MoPJEPA, {"lazy": True},
                 [("hidden", lambda v: {"hidden": v}), ("head_hidden", lambda v: {"head_hidden": v})]),
    "chart_ae": (ChartAEJEPA, {},
                 [("hidden", lambda v: {"hidden": v}), ("chart_hidden", lambda v: {"chart_hidden": v})]),
    "lazy_chart_ae": (LazyChartAEJEPA, {},
                      [("hidden", lambda v: {"hidden": v}), ("chart_hidden", lambda v: {"chart_hidden": v})]),
    "peer": (PEERJEPA, {}, [("hidden", lambda v: {"hidden": v}), ("n_experts", lambda v: {"n_experts": v * v})]),
}
GRID = sorted({int(round(4 * 1.08 ** i)) for i in range(90)})  # 4 .. ~4000, 8% steps


def measure(arm: str, **kw):
    cls, fixed, _ = SPECS[arm]
    with torch.device("meta"):
        m = cls(**{**fixed, **kw})
    return m.active_flops(), m.n_params()


def _bisect(f, target, lo=1, hi=4096):
    """smallest v in [lo, hi] with f(v) >= target (f nondecreasing), or the nearer neighbour."""
    if f(hi) < target:
        return hi
    a = lo
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (mid + 1, hi) if f(mid) < target else (lo, mid)
    return lo if lo == a or abs(f(lo) - target) <= abs(f(lo - 1) - target) else lo - 1


def match(arm: str, flops: int, params: int, tol: float = 0.05, min_width: int = 32, **base):
    """base: fixed arm kwargs (D, A, d, K, ...). Returns dict(both | None, flop_matched, param_matched, note)."""
    knobs = SPECS[arm][2]
    rec = lambda kw: dict(kw=kw, **dict(zip(("flops", "params"), measure(arm, **kw))))
    err = lambda r: max(abs(r["flops"] / flops - 1), abs(r["params"] / params - 1))
    lo = base.get("topk", 8) if arm == "peer" else 1  # PEER needs sqrt(N) >= topk
    width = lambda r: min(r["kw"].get(n, 10 ** 9) for n in ("hidden", "head_hidden", "chart_hidden"))
    ok = []
    if len(knobs) == 2:
        (_, k1), (_, k2) = knobs
        for v1 in GRID:
            v2 = _bisect(lambda v: measure(arm, **base, **k1(v1), **k2(v))[1], params, lo=lo)
            r = rec({**base, **k1(v1), **k2(v2)})
            ok += [r] if err(r) <= tol else []
    # single-axis baselines: one width s for every width knob (PEER: n_experts stays at its default)
    tied = lambda v: {k: v for k, _ in knobs if k != "n_experts"}
    fm = rec({**base, **tied(_bisect(lambda v: measure(arm, **base, **tied(v))[0], flops))})
    pm = rec({**base, **tied(_bisect(lambda v: measure(arm, **base, **tied(v))[1], params))})
    ok += [r for r in (fm, pm) if err(r) <= tol]
    # several knob pairs can hit both targets; take the most balanced one (no 7-wide encoder feeding
    # 575-wide heads): largest smallest width; below min_width the match is degenerate -> split instead
    ok = [r for r in ok if width(r) >= min_width]
    out = dict(both=max(ok, key=width) if ok else None, flop_matched=fm, param_matched=pm)
    if out["both"] is None:
        out["note"] = (f"{arm}: params/FLOPs ratio out of reach (dense: ~1; lazy top-1 of K: < K); run flop_matched "
                       f"(P={fm['params']}) and param_matched (F={pm['flops']}) as two baselines")
    return out


def match_lm(arm: str, flops: int, params: int, tol: float = 0.05, max_layers: int = 24, max_width: int = 4096, **base):
    """arm in {"gpt", "gru"}; base: vocab and n_head+seq_len (gpt) or optional d_emb (gru). Knobs: width
    (gpt d_model in steps of n_head, gru hidden) and n_layer. Same return format as match()."""
    cls = GPTLM if arm == "gpt" else GRULM

    def kw(w, L):
        if arm == "gpt":
            return {**base, "d_model": w * base["n_head"], "n_layer": L}
        return {"d_emb": w, **base, "hidden": w, "n_layer": L}

    def rec(w, L):
        with torch.device("meta"):
            m = cls(**kw(w, L))
        return dict(kw=kw(w, L), flops=m.active_flops(), params=m.n_params())

    fit = lambda L, i, t: rec(_bisect(lambda w: rec(w, L)[("flops", "params")[i]], t, hi=max_width), L)
    err = lambda r, i, t: abs(r[("flops", "params")[i]] / t - 1)
    fm = min((fit(L, 0, flops) for L in range(1, max_layers + 1)), key=lambda r: err(r, 0, flops))
    cands = [fit(L, 1, params) for L in range(1, max_layers + 1)]
    pm = min(cands, key=lambda r: err(r, 1, params))
    ok = [r for r in cands if max(err(r, 0, flops), err(r, 1, params)) <= tol]
    out = dict(both=min(ok, key=lambda r: max(err(r, 0, flops), err(r, 1, params))) if ok else None,
               flop_matched=fm, param_matched=pm)
    if out["both"] is None:
        out["note"] = f"{arm}: no (width, n_layer) hits both within {tol}; run flop_matched and param_matched"
    return out


if __name__ == "__main__":
    for arm, base in [("single", dict(d=4)), ("mop", dict(d=4, K=8)), ("mop_lazy", dict(d=4, K=8)),
                      ("chart_ae", dict(K=8, d_c=2, D_g=8)), ("peer", dict(d=4))]:
        r = match(arm, flops=200_000, params=2_000_000, D=256, A=2, **base)
        print(arm, {k: (v if not isinstance(v, dict) else (v["flops"], v["params"], v["kw"])) for k, v in r.items()})

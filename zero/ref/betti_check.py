"""Independent Betti numbers of a finite simplicial complex, over Z2 (GF(2) elimination) or Q (exact Fractions).

Written separately from metrics.betti_z2 so the two can cross-check. Input: facets as iterables of vertex labels
(any hashable, sortable); the complex is their downward closure. Output: (b_0, ..., b_top).
"""
from fractions import Fraction
from itertools import combinations


def faces_of(facets):
    out = set()
    for f in facets:
        f = tuple(sorted(set(f)))
        for k in range(1, len(f) + 1):
            out.update(combinations(f, k))
    return out


def _rank(rows, field):
    """Rank of a sparse matrix given as list of {col: int} rows."""
    piv = {}
    r = 0
    for row in rows:
        if field == "Z2":
            row = {c for c, v in row.items() if v % 2}
            while row:
                c = min(row)
                if c not in piv:
                    piv[c] = row
                    r += 1
                    break
                row = row ^ piv[c]
        else:
            row = {c: Fraction(v) for c, v in row.items() if v}
            while row:
                c = min(row)
                if c not in piv:
                    piv[c] = row
                    r += 1
                    break
                p = piv[c]
                f = row[c] / p[c]
                for cc, vv in p.items():
                    nv = row.get(cc, 0) - f * vv
                    if nv:
                        row[cc] = nv
                    else:
                        row.pop(cc, None)
    return r


def betti(facets, field="Z2"):
    if field not in ("Z2", "Q"):
        raise ValueError(field)
    S = faces_of(facets)
    if not S:
        return ()
    top = max(len(s) for s in S) - 1
    by = [sorted(s for s in S if len(s) == k + 1) for k in range(top + 1)]
    idx = [{s: i for i, s in enumerate(b)} for b in by]
    rk = [0] * (top + 2)
    for k in range(1, top + 1):
        rows = [{idx[k - 1][s[:i] + s[i + 1:]]: (-1) ** i for i in range(len(s))} for s in by[k]]
        rk[k] = _rank(rows, field)
    return tuple(len(by[k]) - rk[k] - rk[k + 1] for k in range(top + 1))

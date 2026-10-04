"""Z2 homology on known triangulations (tda-tdd: known-manifold ground truth, d∘d = 0, permutation invariance)."""
import itertools

import numpy as np
import pytest

import json
from pathlib import Path

from metrics import betti_z2, betti_q, boundary_z2, closure, nerve_betti_z2, most_persistent

TRI = json.loads((Path(__file__).resolve().parents[1] / "ref" / "triangulations.json").read_text())["triangulations"]


def faces(tris):
    return closure([tuple(t) for t in tris])


def is_closed_2manifold(tris):
    count = {}
    for t in tris:
        for e in itertools.combinations(sorted(t), 2):
            count[e] = count.get(e, 0) + 1
    return set(count.values()) == {2}


OCTAHEDRON = [(a, b, c) for a in (0, 1) for b in (2, 3) for c in (4, 5)]
TORUS7 = [tuple(sorted(((i) % 7, (i + 1) % 7, (i + 3) % 7))) for i in range(7)] + \
         [tuple(sorted(((i) % 7, (i + 2) % 7, (i + 3) % 7))) for i in range(7)]
RP2_6 = [(1, 2, 3), (1, 3, 4), (1, 4, 5), (1, 5, 6), (1, 6, 2),
         (2, 3, 5), (3, 4, 6), (4, 5, 2), (5, 6, 3), (6, 2, 4)]


def grid_surface(a, b, klein):
    """a x b grid on the square; (i+a, j) ~ (i, -j) if klein else (i, j)."""
    def v(i, j):
        if i >= a:
            i, j = i - a, (-j if klein else j)
        return i * b + (j % b)
    tris = []
    for i in range(a):
        for j in range(b):
            tris += [tuple(sorted((v(i, j), v(i + 1, j), v(i + 1, j + 1)))),
                     tuple(sorted((v(i, j), v(i, j + 1), v(i + 1, j + 1))))]
    return tris


def rp3_from_16cell():
    """RP^3 = sd(boundary of 4-D cross-polytope) / antipodal map. Antipodal vertices of the
    barycentric subdivision are >= 3 apart in the 1-skeleton, so the quotient is simplicial."""
    verts = [(i, s) for i in range(4) for s in (1, -1)]
    facets = [tuple((i, s[i]) for i in range(4)) for s in itertools.product((1, -1), repeat=4)]
    simp = set()
    for f in facets:
        for k in range(1, 5):
            simp.update(frozenset(c) for c in itertools.combinations(f, k))
    neg = lambda sig: frozenset((i, -s) for i, s in sig)
    orbit = {}
    for sig in simp:
        key = min(tuple(sorted(sig)), tuple(sorted(neg(sig))))
        orbit.setdefault(key, len(orbit))
    oid = lambda sig: orbit[min(tuple(sorted(sig)), tuple(sorted(neg(sig))))]
    tets = set()
    for f in facets:
        for perm in itertools.permutations(f):
            chain = [frozenset(perm[:k]) for k in range(1, 5)]
            tets.add(tuple(sorted(oid(c) for c in chain)))
    return list(tets), len(verts)


def test_boundary_squared_is_zero():
    cx = closure(TORUS7)
    d2, d1 = boundary_z2(cx, 2), boundary_z2(cx, 1)
    assert d1.shape[1] == d2.shape[0] and not ((d1.astype(int) @ d2.astype(int)) % 2).any()


@pytest.mark.parametrize("name,tris,betti", [
    ("S2 octahedron", OCTAHEDRON, (1, 0, 1)),
    ("T2 Mobius-Csaszar 7", TORUS7, (1, 2, 1)),
    ("RP2 6-vertex", RP2_6, (1, 1, 1)),
    ("T2 grid 4x4", grid_surface(4, 4, False), (1, 2, 1)),
    ("Klein grid 4x4", grid_surface(4, 4, True), (1, 2, 1)),
    ("Klein grid 3x5", grid_surface(3, 5, True), (1, 2, 1)),
])
def test_surfaces(name, tris, betti):
    assert is_closed_2manifold(tris), name
    assert betti_z2(faces(tris)) == betti


def test_three_dim_spaces():
    s3 = list(itertools.combinations(range(5), 4))
    assert betti_z2(faces(s3)) == (1, 0, 0, 1)
    sus = [t + (c,) for t in RP2_6 for c in (7, 8)]                 # suspension of RP^2
    assert betti_z2(faces(sus)) == (1, 0, 1, 1)
    tets, _ = rp3_from_16cell()
    cx = faces(tets)
    chi = sum((-1) ** (len(s) - 1) for s in cx)
    assert chi == 0 and betti_z2(cx) == (1, 1, 1, 1)


def test_permutation_invariance():
    rng = np.random.default_rng(1)
    p = rng.permutation(16)
    tris = [tuple(sorted(p[v] for v in t)) for t in grid_surface(4, 4, True)]
    assert betti_z2(faces(tris)) == (1, 2, 1)


def test_nerve_from_weights_octahedron():
    # one sample per triangle: weights 1/3 on its three charts -> nerve = octahedron boundary
    w = np.zeros((8, 6))
    for r, t in enumerate(OCTAHEDRON):
        w[r, list(t)] = 1 / 3
    assert nerve_betti_z2(w, tau=0.2) == (1, 0, 1)
    assert nerve_betti_z2(w, tau=0.5) == (0, 0, 0)                   # nothing survives -> empty
    # row permutations and duplicated rows change nothing
    assert nerve_betti_z2(np.concatenate([w[::-1], w]), tau=0.2) == (1, 0, 1)


def test_top2_weights_cannot_see_h2_without_flag():
    # only pairs co-selected (m=2): the true nerve is a graph, the flag complex fills it
    w = []
    for t in OCTAHEDRON:
        for a, b in itertools.combinations(t, 2):
            row = np.zeros(6); row[[a, b]] = 0.5; w.append(row)
    w = np.array(w)
    assert nerve_betti_z2(w, tau=0.3) == (1, 7, 0)                  # octahedron 1-skeleton: 12-6+1
    assert nerve_betti_z2(w, tau=0.3, flag=True, max_dim=2) == (1, 0, 1)


def test_most_persistent():
    seq = [(1, 0, 0), (1, 2, 1), (1, 2, 1), (1, 2, 1), (3, 0, 0)]
    assert most_persistent(seq) == ((1, 2, 1), 3)


@pytest.mark.parametrize("key", sorted(TRI))
def test_lutz_triangulations_z2_and_q(key):
    """Wilson-verified vertex-minimal triangulations (Lutz Manifold Page). Q separates T2 from Klein."""
    t = TRI[key]
    cx = faces(t["facets"])
    assert betti_z2(cx) == tuple(t["betti_Z2"]), key
    assert betti_q(cx) == tuple(t["betti_Q"]), key


def test_q_separates_torus_from_klein():
    assert betti_z2(faces(TORUS7)) == betti_z2(faces(grid_surface(4, 4, True))) == (1, 2, 1)
    assert betti_q(faces(TORUS7)) == (1, 2, 1) and betti_q(faces(grid_surface(4, 4, True))) == (1, 1, 0)


def test_soft_weight_nerve_threshold_bound():
    """Partition of unity: |{k: w_k >= tau}| <= 1/tau, so b3 needs tau <= 1/4 (Wilson facts 5)."""
    tets = [tuple(f) for f in TRI["RP3_Walkup_11"]["facets"]]
    w = np.zeros((len(tets), 12))
    for r, t in enumerate(tets):
        w[r, list(t)] = 0.25
    assert nerve_betti_z2(w, tau=0.25, max_dim=3) == (1, 1, 1, 1)
    assert nerve_betti_z2(w, tau=0.26, max_dim=3) == (0, 0, 0, 0)


def test_large_complex_bounded_memory():
    """Smoke of sweep_t2 dry-run OOMed: dense bool d_3 of 60722 x 200167 (11.3 GiB) on a K=64 atlas.
    Ranks must be computed from sparse columns: flag complex of K_30 (27,405 tetrahedra) under 150 MB."""
    import tracemalloc
    w = []
    for a, b in itertools.combinations(range(30), 2):
        row = np.zeros(30); row[[a, b]] = 0.5; w.append(row)
    tracemalloc.start()
    b = nerve_betti_z2(np.array(w), tau=0.3, flag=True, max_dim=2)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert b == (1, 0, 0) and peak < 150e6, (b, peak / 1e6)


def test_oversized_complex_returns_none():
    from metrics import MAX_SIMPLICES
    cx = closure([tuple(range(12))], 13)                       # 4095 faces
    assert betti_z2(cx) == (1,) + (0,) * 11
    import metrics
    old, metrics.MAX_SIMPLICES = metrics.MAX_SIMPLICES, 1000
    try:
        assert betti_z2(cx) is None and nerve_betti_z2(np.ones((1, 12)) / 12, 0.05, max_dim=2) is not None
    finally:
        metrics.MAX_SIMPLICES = old

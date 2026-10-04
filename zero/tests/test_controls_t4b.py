"""T4b controls (arms/controls_t4b.py): memoriser, propose-all, codebook. Interface shared with learned arms:
fit(ctx [M,2], nxt [M,2]); embed(xy [C,2]) -> [C,d]; predict(ctx [C,2]) -> (heads [C,K,d], pi [C,K]).
Edges come from ogbench_bed.propose_edges(cells, embed(centres), heads, pi?)."""
import numpy as np
import pytest

pytest.importorskip("ogbench")
from arms.controls_t4b import Codebook, Memoriser, ProposeAll  # noqa: E402
from beds import ogbench_bed as ob  # noqa: E402

MED = "pointmaze-medium-stitch-v0"


@pytest.fixture(scope="module")
def setup():
    bed = ob.make_bed(MED)
    E = ob.data_edges(bed)
    tr, ho = ob.holdout_edges(E, 0.2, 0)
    ctx, nxt, _, _ = ob.cell_transitions(bed, exclude=frozenset(ho))
    cells = sorted({a for a, _ in ob.true_edges(MED)})
    return tr, ho, ctx, nxt, cells, ob.ij_to_xy(MED, np.array(cells))


def edges(arm, cells, xy, gate=False):
    h, pi = arm.predict(xy)
    assert h.ndim == 3 and pi.shape == h.shape[:2] and np.allclose(pi.sum(1), 1)
    return ob.propose_edges(cells, arm.embed(xy), h, pi if gate else None)


def test_memoriser_is_exactly_training_edges(setup):
    tr, ho, ctx, nxt, cells, xy = setup
    m = Memoriser().fit(ctx, nxt)
    assert edges(m, cells, xy) == tr
    assert edges(m, cells, xy, gate=True) <= tr


def test_propose_all_freeloads(setup):
    tr, ho, ctx, nxt, cells, xy = setup
    p = ProposeAll().fit(ctx, nxt)
    E = edges(p, cells, xy)
    r = ob.graph_metrics(E, MED, ob.sample_queries(MED, 300, 0))
    assert r["realroute"] == 1.0 and r["precision"] < 0.1 and tr <= E


def test_codebook_is_input_agnostic(setup):
    tr, ho, ctx, nxt, cells, xy = setup
    c = Codebook(K=8, seed=0).fit(ctx, nxt)
    h, pi = c.predict(xy)
    assert h.shape[1] == 8 and np.allclose(h, h[:1]) and np.allclose(pi, pi[:1])
    assert Codebook(K=8, seed=0).fit(ctx, nxt).predict(xy)[0].tolist() == h.tolist()     # seed-deterministic
    E = edges(c, cells, xy)
    targets = {b for _, b in E}
    assert 1 <= len(targets) <= 8 and all(len([e for e in E if e[0] == a]) <= 8 for a in cells)


def test_controls_never_see_holdout(setup):
    tr, ho, ctx, nxt, cells, xy = setup
    seen = {(tuple(a), tuple(b)) for a, b in zip(ob.xy_to_ij(MED, ctx), ob.xy_to_ij(MED, nxt))}
    assert not seen & ho

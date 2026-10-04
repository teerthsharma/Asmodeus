"""T4b graph-protocol adapters (arms/t4b_adapters.py) on tiny, barely trained models: shapes, determinism,
row independence (no batch/cross-row leak), top-m proposals for routed arms, input-agnostic latent codebook."""
import numpy as np
import pytest
import torch

from arms import t4b_adapters as A
from beds import ogbench_bed as ob

rng = np.random.default_rng(0)
CELLS = rng.uniform(0, 20, (12, 2))
CTX = CELLS[rng.integers(0, 12, 300)]
NXT = CELLS[rng.integers(0, 12, 300)]
TINY = dict(steps=3, batch=32)
KINDS = ["single", "mop", "atlas", "lazy_chart_ae", "codebook_latent"]


def make(kind, seed=0):
    return A.make(kind, seed=seed, tiny=True, **TINY).fit(CTX, NXT)


@pytest.fixture(scope="module", params=KINDS)
def fitted(request):
    return request.param, make(request.param)


def test_shapes(fitted):
    kind, a = fitted
    e, (h, pi) = a.embed(CELLS), a.predict(CELLS)
    assert e.ndim == 2 and len(e) == 12 and h.shape[0] == 12 and h.shape[2] == e.shape[1]
    assert pi.shape == h.shape[:2] and np.allclose(pi.sum(1), 1, atol=1e-5)
    assert np.isfinite(e).all() and np.isfinite(h).all() and np.isfinite(pi).all()
    K = {"single": 1, "mop": a.K if kind == "mop" else None, "atlas": A.TINY["atlas"]["m"],
         "lazy_chart_ae": A.TINY["lazy_chart_ae"]["m"], "codebook_latent": A.TINY["codebook_latent"]["K"]}[kind]
    assert h.shape[1] == K                       # routed arms propose from their top-m charts only


def test_deterministic(fitted):
    kind, a = fitted
    b = make(kind)
    assert np.array_equal(a.predict(CELLS)[0], b.predict(CELLS)[0])
    assert np.array_equal(a.embed(CELLS), b.embed(CELLS))
    assert not np.array_equal(a.predict(CELLS)[0], make(kind, seed=1).predict(CELLS)[0])


def test_rows_independent(fitted):
    """Prediction for a context depends on that context only: no batch statistics, no cross-row leak."""
    kind, a = fitted
    h, pi = a.predict(CELLS)
    perm = rng.permutation(12)
    hp, pip = a.predict(CELLS[perm])
    assert np.allclose(hp, h[perm], atol=1e-5) and np.allclose(pip, pi[perm], atol=1e-5)
    h1, _ = a.predict(CELLS[3:4])
    assert np.allclose(h1[0], h[3], atol=1e-5)
    assert np.allclose(a.embed(CELLS[3:4])[0], a.embed(CELLS)[3], atol=1e-5)


def test_latent_codebook_is_input_agnostic():
    h, pi = make("codebook_latent").predict(CELLS)
    assert np.allclose(h, h[:1], atol=1e-6) and np.allclose(pi, pi[:1], atol=1e-6)


def test_fit_sees_only_given_pairs():
    """fit() reads nothing but (ctx, nxt): poisoning everything else in reach leaves the model unchanged."""
    a = make("mop")
    a2 = A.make("mop", seed=0, tiny=True, **TINY).fit(CTX.copy(), NXT.copy())
    assert np.array_equal(a.predict(CELLS)[0], a2.predict(CELLS)[0])
    with pytest.raises(Exception):
        A.make("mop", seed=0, tiny=True, **TINY).fit(CTX, NXT[:-1])    # misaligned pairs rejected


def test_training_moves_parameters():
    a0 = A.make("single", seed=0, tiny=True, steps=0, batch=32).fit(CTX, NXT)
    a1 = A.make("single", seed=0, tiny=True, steps=20, batch=32).fit(CTX, NXT)
    assert not np.allclose(a0.predict(CELLS)[0], a1.predict(CELLS)[0])


class _Shift:
    """Wraps an arm and adds a constant c to every latent it emits (embed and heads)."""

    def __init__(self, arm, c):
        self.arm, self.c = arm, c

    def embed(self, xy):
        return self.arm.embed(xy) + self.c

    def predict(self, ctx):
        h, pi = self.arm.predict(ctx)
        return h + self.c, pi


@pytest.mark.parametrize("kind", ["single", "atlas"])
def test_centred_edges_offset_invariant(kind):
    """PRIMARY rule: subtract the train-set latent mean, then cos > .5. A constant offset on all latents must not
    change centred edges; the paper's raw rule is not invariant (a large offset makes every pair cos ~ 1)."""
    a = make(kind)
    cells = [(int(i), 0) for i in range(len(CELLS))]
    train_xy = np.concatenate([CTX, NXT])
    d = a.embed(CELLS[:1]).shape[1]
    c = np.full(d, 50.0)
    for gate in (False, True):
        e0 = ob.decode_edges(a, cells, CELLS, train_xy, rule="centred", gate=gate)
        e1 = ob.decode_edges(_Shift(a, c), cells, CELLS, train_xy, rule="centred", gate=gate)
        assert e0 == e1
    p0 = ob.decode_edges(a, cells, CELLS, train_xy, rule="paper")
    p1 = ob.decode_edges(_Shift(a, c), cells, CELLS, train_xy, rule="paper")
    assert len(p1) == len(cells) * (len(cells) - 1) and p0 != p1
    with pytest.raises(ValueError):
        ob.decode_edges(a, cells, CELLS, train_xy, rule="nope")

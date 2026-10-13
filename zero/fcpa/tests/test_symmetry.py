"""L37 era model: the group average of mode-feature statistics, the symmetric ridge, and its wiring into FCPA.

(1) symmetry.average equals the statistics of the data augmented by every element of Gamma_17 (L37 Q2);
(2) the quadratic block is an orthonormal basis of Sym^2 (L37 R1: invariant penalty);
(3) SymEraModel: chunked partial_fit equals fit (L27') and the fitted map commutes with O(2), not only Gamma_17;
(4) FCPARegressor(era=...): the era answers through the symmetric ridge and partial_fit stays exact.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa import run_primary as rp  # noqa: E402
from fcpa.eras import SymEraModel  # noqa: E402
from fcpa.estimator import FCPARegressor  # noqa: E402
from fcpa.laws import NS2DLaw  # noqa: E402
from fcpa.symmetry import ModeFeatures, average, average_vec  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")
LAW = NS2DLaw(n=32, kmax=4, dt=0.1)


def _states(n, seed=0):
    """Stored coefficients of random real fields: they satisfy the conjugate symmetry the bed's states do."""
    rng = np.random.default_rng(seed)
    return LAW._coords(rng.standard_normal((n, 32, 32)))


def _act(g, X, n_lags):
    D = LAW.D
    return np.concatenate([X[:, i * D:(i + 1) * D] @ g.T for i in range(n_lags)], axis=1)


@pytest.mark.parametrize("quad", [False, True])
def test_average_is_group_augmentation(quad):
    X, Y = np.c_[_states(40, 0), _states(40, 1)], _states(40, 2)
    F = ModeFeatures(LAW, 2, quad, ModeFeatures.rms(LAW, X[:, :LAW.D]))
    Z = F(X)
    Gs, Cs, ms = [], [], []
    for g in LAW.group():
        Zg = F(_act(g, X, 2))
        Gs.append(Zg.T @ Zg), Cs.append(Zg.T @ (Y @ g.T)), ms.append(Zg.mean(0))
    assert np.allclose(average(Z.T @ Z, F.desc, F.desc), np.mean(Gs, 0), atol=1e-10 * np.abs(Z.T @ Z).max())
    assert np.allclose(average(Z.T @ Y, F.desc, F.ydesc), np.mean(Cs, 0), atol=1e-10 * np.abs(Z.T @ Y).max())
    assert np.allclose(average_vec(Z.mean(0), F.desc), np.mean(ms, 0), atol=1e-12)


def test_quadratic_block_is_orthonormal_sym2():
    X = _states(30, 3)
    F = ModeFeatures(LAW, 1, True, 1.0)
    q = F(X)[:, 80:]
    _, _, ind = LAW.modes()
    r = np.c_[X[:, ind], X[:, ind + LAW.D // 2]]
    assert q.shape[1] == 80 * 81 // 2
    assert np.allclose((q ** 2).sum(1), (r ** 2).sum(1) ** 2)  # |sym2(x)| = |x|^2 exactly in an orthonormal basis


def _era_data(n=600, seed=4):
    X = np.c_[_states(n, seed), _states(n, seed + 1)]
    Y = 0.1 * _states(n, seed + 2) + 0.05 * X[:, :LAW.D]
    return X, Y


def test_sym_era_partial_fit_equals_fit():
    X, Y = _era_data()
    F = ModeFeatures(LAW, 2, True, ModeFeatures.rms(LAW, X[:, :LAW.D]))
    a = SymEraModel(0.1, F.desc, F.ydesc).fit(F(X), Y)
    b = SymEraModel(0.1, F.desc, F.ydesc)
    for c in np.array_split(np.arange(len(X)), 7):
        b.partial_fit(F(X[c]), Y[c], solve=False)
    b.set_chart(None)
    Xq = np.c_[_states(20, 9), _states(20, 10)]
    assert np.allclose(a.predict(F(Xq)), b.predict(F(Xq)), atol=1e-10)


@pytest.mark.parametrize("m,j,e", [(17, 5, 0), (17, 3, 1), (13, 5, 1), (29, 11, 0)])
def test_sym_era_commutes_with_o2(m, j, e):
    """pred(g x) = g pred(x) for g in Gamma_17 and for shifts outside it (m = 13, 29): the averaged ridge is
    O(2)-equivariant (L37 Q2), so it does not depend on the grid of shifts it was averaged over."""
    X, Y = _era_data()
    F = ModeFeatures(LAW, 2, True, ModeFeatures.rms(LAW, X[:, :LAW.D]))
    em = SymEraModel(0.1, F.desc, F.ydesc).fit(F(X), Y)
    g = LAW.group(m)[2 * j + e]
    Xq = np.c_[_states(20, 11), _states(20, 12)]
    lhs, rhs = em.predict(F(_act(g, Xq, 2))), em.predict(F(Xq)) @ g.T
    assert np.allclose(lhs, rhs, atol=1e-9 * np.abs(rhs).max())


def test_group_matrices_are_orthogonal():
    for g in LAW.group():
        assert np.allclose(g @ g.T, np.eye(LAW.D))


def _small_world():
    P = dict(rp.SMOKE, n_traj=12, L=110)
    X, _ = rp.gen_bed(7002, P)
    return P, rp.lag_stack(X, P["n_lags"])


@pytest.mark.parametrize("era", ["lin_vq", "quad_vq"])
def test_estimator_era_is_symmetric_and_exact(era):
    P, XL = _small_world()
    law = NS2DLaw(n=P["grid"], kmax=P["kmax"], dt=P["dt"] * P["steps_per_snapshot"])
    D = XL.shape[-1] // 2
    kw = dict(window=P["w"], random_state=0, n_lags=2, law=law, era=era, era_lam=0.1,
              era_scale=ModeFeatures.rms(law, XL[..., :D].reshape(-1, D)),
              y_edges=FCPARegressor.make_step_edges(XL, n_lags=2))
    full = FCPARegressor(**kw).fit(XL[0])
    part = FCPARegressor(**kw)
    for c in np.array_split(XL[0], 4):
        part.partial_fit(c)
    assert full.n_eras_ >= 1 and all(isinstance(e, SymEraModel) for e in full._model["eras"])
    q = XL[1, :30]
    assert np.allclose(full.era_predict(0, q), part.era_predict(0, q), atol=1e-8)
    with pytest.raises(ValueError):
        FCPARegressor(window=20, era="quad_vq").fit(XL[0, :, :D])  # no law: no symmetry action


def test_fit_fcpa_wires_the_era_pin():
    P, XL = _small_world()
    est = rp.fit_fcpa(XL[:8], P, 7002, pins=dict(rp.FCPA_PINS, era="quad_vq", era_lam=0.1), cal=XL[8:])
    assert est.era == "quad_vq" and est._feat.quad
    assert all(isinstance(e, SymEraModel) for e in est._model["eras"])
    B = rp.fcpa_train_macs(est, 8 * (P["L"] - 1), 8, P["w"])
    base = rp.fit_fcpa(XL[:8], P, 7002, pins=dict(rp.FCPA_PINS, era="phi"), cal=XL[8:])
    assert B > rp.fcpa_train_macs(base, 8 * (P["L"] - 1), 8, P["w"])

"""L19 (CITED, SIGReg scope narrower than billing) + L27 for era statistics. EraModel = JEPA-in-slab."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa.eras import (EraModel, effective_rank, era_chart, readout_lipschitz,  # noqa: E402
                       sigreg_penalty, zoom_floor)


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 6)) * np.array([1, .5, .25, .12, .06, .03])
    return Z, Z.copy()  # identity dynamics: target spectrum == input spectrum


def test_L19_sigreg_isotropic_vs_collapsed():
    rng = np.random.default_rng(1)
    iso = rng.standard_normal((5000, 6))
    rank1 = rng.standard_normal((5000, 1)) @ rng.standard_normal((1, 6))
    assert sigreg_penalty(iso) < 0.05
    assert sigreg_penalty(rank1) > 4.0  # d - 1 = 5 at exact rank one
    assert effective_rank(iso) > 5.5
    assert effective_rank(rank1) < 1.05
    assert effective_rank(np.zeros((10, 6))) == 0.0


def test_L19_collapse_guard_on_embedding_rank():
    Z, Z1 = _data()
    kw = dict(lam=1e3, beta=0.0, min_rank_frac=0.8)
    bad = EraModel(guard=False, **kw).fit(Z, Z1)
    good = EraModel(guard=True, **kw).fit(Z, Z1)
    tgt = effective_rank(Z1)
    assert effective_rank(bad.predict(Z)) < 0.8 * tgt  # unguarded ridge collapses the embedding
    assert effective_rank(good.predict(Z)) >= 0.8 * tgt  # guard restores it
    assert not good.collapsed_ and good.lam_eff_ < 1e3


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((500, 4))
    Wt = rng.standard_normal((4, 4))
    m = EraModel(lam=1e-6, beta=0.0).fit(Z, Z @ Wt + 0.5)
    assert np.allclose(m.predict(Z), Z @ Wt + 0.5, atol=1e-3)


def test_L27_era_stats_additive():
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((300, 5))
    Z1 = np.tanh(Z @ rng.standard_normal((5, 3)))
    a = EraModel().fit(Z, Z1)
    b = EraModel()
    for s in np.array_split(np.arange(300), 7):
        b.partial_fit(Z[s], Z1[s])
    assert np.allclose(a.W_, b.W_, atol=1e-10)
    assert a.n_ == b.n_ == 300


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        EraModel().predict(np.zeros((2, 3)))


def test_chart_solve_equals_manual_projection():
    rng = np.random.default_rng(4)
    Z = rng.standard_normal((200, 6))
    Z1 = rng.standard_normal((200, 2))
    P = np.linalg.qr(rng.standard_normal((6, 3)))[0].T  # (3, 6) isometry
    a = EraModel(chart=P).fit(Z, Z1)
    b = EraModel().fit(Z @ P.T, Z1)
    assert np.allclose(a.predict(Z), b.predict(Z @ P.T), atol=1e-10)
    # L27': stats live in the ambient space, so the chart can change after the data arrived, exactly
    c = EraModel()
    for s in np.array_split(np.arange(200), 5):
        c.partial_fit(Z[s], Z1[s])
    c.set_chart(P)
    assert np.allclose(c.predict(Z), a.predict(Z), atol=1e-10)


def test_L18star_zoom_floor_and_ky_fan():
    rng = np.random.default_rng(5)
    N, dc, n = 10, 3, 4000
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    psi = (rng.standard_normal((n, N)) + 1j * rng.standard_normal((n, N))) @ (A * np.linspace(1, .1, N)).T
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    rho = psi.T @ psi.conj() / n
    V = era_chart(rho, dc)
    zeta = np.sum(np.abs(psi @ V.conj()) ** 2, axis=1)
    lam = np.linalg.eigvalsh(rho)[::-1]
    assert abs(zeta.mean() - lam[:dc].sum()) < 1e-10  # the top-d_c chart attains the Ky Fan value
    Q = np.linalg.qr(rng.standard_normal((N, dc)) + 1j * rng.standard_normal((N, dc)))[0]
    assert np.sum(np.abs(psi @ Q.conj()) ** 2, axis=1).mean() <= lam[:dc].sum() + 1e-12  # L18 ceiling
    for z0 in (0.1, 0.3, 0.5, 0.7):  # the floor, an empirical Markov bound
        assert np.mean(zeta <= z0) <= zoom_floor(lam[:dc].sum(), z0, V, rho) + 1e-12
    with pytest.raises(ValueError):  # hypothesis of the floor is asserted, not assumed
        zoom_floor(lam[:dc].sum(), 0.3, Q, rho)


def test_L18star_readout_lipschitz():
    rng = np.random.default_rng(6)
    N, dc, zmin = 8, 3, 0.2
    V = np.linalg.qr(rng.standard_normal((N, dc)) + 1j * rng.standard_normal((N, dc)))[0]
    J = rng.standard_normal((2, 2 * dc))
    LJ = np.linalg.norm(J, 2)

    def R(psi):
        z = V.conj().T @ psi
        z = z / np.linalg.norm(z)
        return J @ np.concatenate([z.real, z.imag])

    worst, tried = 0.0, 0
    while tried < 3000:
        a = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        a /= np.linalg.norm(a)
        b = a + 0.05 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        b /= np.linalg.norm(b)
        if min(np.linalg.norm(V.conj().T @ a) ** 2, np.linalg.norm(V.conj().T @ b) ** 2) < zmin:
            continue
        tried += 1
        worst = max(worst, np.linalg.norm(R(a) - R(b)) / np.linalg.norm(a - b))
    assert worst <= readout_lipschitz(LJ, zmin)


def test_default_era_model_is_unbiased_on_clean_linear_data():
    """e7: the isotropy shrink (beta) belongs to the embedding guard, not to the predictor's coefficients."""
    rng = np.random.default_rng(10)
    Z = rng.standard_normal((2000, 4)) * np.array([3.0, 1.0, 0.3, 0.1]) + 5.0  # anisotropic and offset
    Wt = rng.standard_normal((4, 2))
    m = EraModel().fit(Z, Z @ Wt + 1.0)
    Zt = rng.standard_normal((500, 4)) * np.array([3.0, 1.0, 0.3, 0.1]) + 5.0
    assert np.sqrt(np.mean((m.predict(Zt) - (Zt @ Wt + 1.0)) ** 2)) < 2e-3


def test_era_ridge_is_scale_free_under_a_dominant_offset():
    """e7: a constant offset in the features must not set the ridge scale (centred covariance does)."""
    rng = np.random.default_rng(11)
    for s in (1.0, 1e-3):
        Z = np.c_[rng.standard_normal((400, 3)) * s, np.ones(400) * 1e3]  # signal at scale s, huge offset column
        w = np.array([1.0, -2.0, 0.5, 0.0])
        m = EraModel().fit(Z, Z @ w)
        Zt = np.c_[rng.standard_normal((200, 3)) * s, np.ones(200) * 1e3]
        assert np.sqrt(np.mean((m.predict(Zt).ravel() - Zt @ w) ** 2)) < 1e-2 * s * 3

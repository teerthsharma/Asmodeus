"""NS-2D multiverse bed: solver physics, ground-truth eras, Markov switching, E1/E2, bed segmentation."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa.beds import ns2d  # noqa: E402

N = 32
X = 2 * np.pi * np.arange(N) / N
XX, YY = np.meshgrid(X, X, indexing="ij")


def _spec(w):
    return np.fft.fft2(w, axes=(-2, -1))


def test_taylor_green_exact_decay():
    """Solver is exact on the Taylor-Green vortex: advection vanishes, w = w0 exp(-2 nu t)."""
    nu, dt, T = 0.05, 0.01, 101
    w0 = 2 * np.cos(XX) * np.cos(YY)
    out = ns2d.simulate(N, [(0, nu)], None, None, T, dt, 1, 0, omega0=w0)
    exact = w0 * np.exp(-2 * nu * dt * (T - 1))
    assert out["omega"].shape == (1, T, N, N)
    assert np.max(np.abs(out["omega"][0, -1] - exact)) < 1e-8


def test_dealiased_two_thirds():
    out = ns2d.simulate(N, [(0, 0.01)], [(0, (0, 4, 1.0))], None, 80, 0.01, 2, 1)
    k = np.fft.fftfreq(N, 1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    outside = (np.abs(KX) > N / 3) | (np.abs(KY) > N / 3)
    wh = _spec(out["omega"][:, -1])
    assert np.max(np.abs(wh[:, outside])) < 1e-10 * np.max(np.abs(wh))


def test_viscous_decay_and_inviscid_enstrophy():
    out = ns2d.simulate(N, [(0, 0.02)], None, None, 120, 0.01, 2, 2)
    Z = 0.5 * (out["omega"] ** 2).mean(axis=(-2, -1))  # (U,T) enstrophy
    assert np.all(np.diff(Z, axis=1) < 0)  # strictly decaying without forcing
    inv = ns2d.simulate(N, [(0, 0.0)], None, None, 120, 0.01, 2, 2)
    Zi = 0.5 * (inv["omega"] ** 2).mean(axis=(-2, -1))
    assert np.max(np.abs(Zi / Zi[:, :1] - 1)) < 1e-3  # RK4 + 2/3 rule: advection conserves enstrophy


def test_exact_tendency_identity_and_consistency():
    """derivs are the solver's own spectral terms: dwdt + adv = nu*lap + f to rounding, and match the trajectory."""
    nu = 0.02
    out = ns2d.simulate(N, [(0, nu)], [(0, (0, 4, 1.0))], None, 60, 0.005, 3, 3, return_derivs=True)
    res = out["dwdt"] + out["adv"] - nu * out["lap"] - out["f"][None]
    assert np.max(np.abs(res)) < 1e-10
    w, dt = out["omega"], out["dt"]
    cd = (w[:, 2:] - w[:, :-2]) / (2 * dt)
    rel = np.linalg.norm(cd - out["dwdt"][:, 1:-1]) / np.linalg.norm(cd)
    assert rel < 1e-3


def test_ground_truth_eras_and_laws():
    nu_s = [(0, 0.02), (50, 0.01), (50, 0.01), (90, 0.01)]  # last two entries are redundant
    f_s = [(0, (0, 4, 1.0)), (50, (0, 4, 1.0)), (70, (3, 0, 1.0))]
    out = ns2d.simulate(N, nu_s, f_s, None, 120, 0.01, 2, 4)
    assert out["boundaries"] == [50, 70]  # redundant entries collapse; (50,nu) and (70,f) are real
    e = out["eras"]
    assert e.shape == (120,) and e[0] == 0 and e[49] == 0 and e[50] == 1 and e[69] == 1 and e[70] == 2 and e[-1] == 2
    L = out["laws"]
    assert [l["nu"] for l in L] == [0.02, 0.01, 0.01]
    assert [l["start"] for l in L] == [0, 50, 70] and L[-1]["stop"] == 120
    assert L[1]["f"] == (0, 4, 1.0) and L[2]["f"] == (3, 0, 1.0)


def test_determinism_and_distinct_universes():
    a = ns2d.simulate(N, [(0, 0.02)], None, None, 10, 0.01, 3, 7)["omega"]
    b = ns2d.simulate(N, [(0, 0.02)], None, None, 10, 0.01, 3, np.random.default_rng(7))["omega"]
    c = ns2d.simulate(N, [(0, 0.02)], None, None, 10, 0.01, 3, 8)["omega"]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    assert np.linalg.norm(a[0] - a[1]) > 0.1 * np.linalg.norm(a[0])  # different random ICs


def test_markov_switching_forcing():
    """Chain statistics match the kernel per era; the applied forcing follows the chain state."""
    P1 = np.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]])
    P2 = P1.T.copy()
    ch = ns2d.markov_chain([(0, P1), (3000, P2)], 6000, np.random.default_rng(0))
    for a, b, P in [(0, 3000, P1), (3000, 6000, P2)]:
        C = np.zeros((3, 3))
        np.add.at(C, (ch[a:b - 1], ch[a + 1:b]), 1)
        np.testing.assert_allclose(C / C.sum(1, keepdims=True), P, atol=0.04)
    pats = [(0, 4, 1.0), (3, 0, 1.0), (2, 2, 1.0)]
    out = ns2d.simulate(N, [(0, 0.02)], pats, [(0, P1), (30, P2)], 60, 0.01, 2, 5, return_derivs=True)
    assert out["boundaries"] == [30] and out["states"].shape == (60,)
    for t in (0, 17, 45):
        k = out["states"][t]
        want = pats[k][2] * np.cos(pats[k][0] * XX + pats[k][1] * YY)
        np.testing.assert_allclose(out["f"][t], want, atol=1e-12)
    assert np.array_equal(out["laws"][1]["P"], P2)


def test_regime_schedule_enforces_switch_rate_bar():
    """Amendment A bar sizing: lambda_sw * w <= 0.1."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ns2d.regime_schedule(400, 0.02, rng, w=16)  # 0.32 > 0.1
    r = ns2d.regime_schedule(2000, 0.1 / 16, rng, w=16, min_dwell=32)
    assert r["starts"][0] == 0 and len(r["classes"]) == len(r["starts"]) >= 3
    assert all(a != b for a, b in zip(r["classes"], r["classes"][1:]))  # every switch is a real law change
    assert min(np.diff(r["starts"])) >= 32
    assert len(set(r["classes"])) < len(r["classes"])  # regimes recur (L14')


def test_E1_amplitude_channel_and_scaling_confound():
    """E1: unit-norm psi loses amplitude; NS scaling w_c(x,t)=c w(x,ct) has nu'=c nu, f'=c^2 f(ct)."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((N, N))
    z1, a1 = ns2d.encode(w)
    z2, a2 = ns2d.encode(3.0 * w)
    np.testing.assert_allclose(z1, z2, atol=1e-12)
    assert abs(np.linalg.norm(z1) - 1) < 1e-12 and abs(a2 / a1 - 3.0) < 1e-12
    nu, c, T, dt = 0.02, 2.0, 40, 0.01
    base = ns2d.simulate(N, [(0, nu)], [(0, (0, 4, 1.0))], None, T, dt, 1, 3)["omega"]
    w0 = base[0, 0]
    sc = ns2d.simulate(N, [(0, c * nu)], [(0, (0, 4, c * c))], None, T, dt / c, 1, 3, omega0=c * w0)["omega"]
    np.testing.assert_allclose(sc, c * base, atol=1e-9)
    zb, ab = ns2d.encode(base[0, -1])
    zs, as_ = ns2d.encode(sc[0, -1])
    np.testing.assert_allclose(zb, zs, atol=1e-9)  # directions agree, amplitudes (and nu) do not
    assert abs(as_ / ab - c) < 1e-9


@pytest.fixture(scope="module")
def bed():
    """Regime-switching bed sized to lambda_sw*w <= 0.1 (w=16), eras >= 2w apart."""
    w = 16
    reg = ns2d.regime_schedule(480, 0.1 / w, np.random.default_rng(3), w=w, min_dwell=2 * w)
    t0 = time.time()
    out = ns2d.simulate(N, reg["nu_schedule"], reg["f_schedule"], None, 480, 0.01, 3, 11, return_derivs=True)
    out["runtime"] = time.time() - t0
    out["w"], out["reg"] = w, reg
    return out


def _f1(pred, true, tol):
    pred, true = list(pred), list(true)
    if not pred or not true:
        return float(pred == true)
    hit, used = 0, set()
    for b in true:
        for i, p in enumerate(pred):
            if i not in used and abs(p - b) <= tol:
                used.add(i)
                hit += 1
                break
    P, R = hit / len(pred), hit / len(true)
    return 0.0 if hit == 0 else 2 * P * R / (P + R)


def test_bed_runtime_and_nontrivial(bed):
    assert len(bed["boundaries"]) >= 2
    assert bed["runtime"] < 30, bed["runtime"]


def test_bed_segment_exact_noise_free(bed):
    """Generator boundaries recovered EXACTLY by segment() on the exact-derivative signal."""
    from fcpa import bounds
    x = bounds.era_signal(bed["dwdt"], bed["adv"], bed["lap"])
    assert bounds.segment(x, bed["w"]) == bed["boundaries"]


def test_bed_segment_small_noise_f1(bed):
    from fcpa import bounds
    rng = np.random.default_rng(0)
    sd = 0.02 * bed["dwdt"].std()
    noisy = bed["dwdt"] + sd * rng.standard_normal(bed["dwdt"].shape)
    x = bounds.era_signal(noisy, bed["adv"], bed["lap"])
    got = bounds.segment(x, bed["w"], sigma="auto")
    assert _f1(got, bed["boundaries"], tol=2) >= 0.9, (got, bed["boundaries"])


def test_E2_nonidentifiable_when_forcing_tracks_laplacian():
    """E2: (nu', f + (nu-nu') lap w) reproduces the same trajectory; only time-constant f pins nu."""
    from fcpa import bounds
    nu, nu2 = 0.02, 0.005
    out = ns2d.simulate(N, [(0, nu)], [(0, (0, 4, 1.0))], None, 40, 0.01, 1, 6, return_derivs=True)
    lap, dw, adv, f0 = out["lap"][0], out["dwdt"][0], out["adv"][0], out["f"]
    f_alt = f0 + (nu - nu2) * lap  # time-varying forcing aligned with lap w
    assert np.max(np.abs(dw + adv - nu2 * lap - f_alt)) < 1e-10  # same omega solves the alternate law exactly
    assert np.ptp(f_alt, axis=0).max() > 1e-3  # ...but alt forcing is not time-constant, so E2 excludes it
    est = bounds.within_slab_estimate(dw, adv, lap)  # f constant in time: identifiable
    assert est.verdict != bounds.REFUSE
    assert abs(est.nu - nu) < 1e-9 * nu
    assert np.max(np.abs(est.f - f0.mean(0))) < 1e-8
    steady = [np.repeat(a[:1], 5, 0) for a in (dw, adv, lap)]  # lap constant in time: nothing leaves f's subspace
    assert bounds.within_slab_estimate(*steady).verdict == bounds.REFUSE

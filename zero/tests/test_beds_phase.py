"""TO1 Berry bed (+ gamma = 0 control) and TO7 held-out closed-loop sets."""
import numpy as np
import pytest

from beds import make_bed, closed_loops, lift, BEDS


def hopf(s):                                        # state [.., 4] = (Re psi1, Im psi1, Re psi2, Im psi2)
    p1, p2 = s[..., 0] + 1j * s[..., 1], s[..., 2] + 1j * s[..., 3]
    c = np.conj(p1) * p2
    return np.stack([2 * c.real, 2 * c.imag, abs(p1) ** 2 - abs(p2) ** 2], -1)


def test_berry_contract_and_horizontal_transport():
    b = make_bed("berry", n_traj=16, T=32, obs="lift", seed=0)
    m = b["meta"]
    assert (m["n"], m["atlas"], m["betti_z2"]) == (3, 2, (1, 0, 0, 1))      # total space S^3
    s = b["state"]
    assert s.shape == (16, 32, 4) and b["obs"].shape == (16, 32, 256) and b["act"].shape == (16, 32, 3)
    assert np.allclose((s ** 2).sum(-1), 1)
    p = s[..., 0::2] + 1j * s[..., 1::2]
    inner = (np.conj(p[:, :-1]) * p[:, 1:]).sum(-1)
    assert np.abs(inner.imag).max() < 1e-12 and (inner.real > 0).all()         # horizontal: <psi_t|psi_t+1> real
    assert np.allclose(b["geo"](s, s), 0, atol=1e-6)


def test_berry_control_shares_base_dynamics():
    b, c = make_bed("berry", 8, 24, "lift", 3), make_bed("berry0", 8, 24, "lift", 3)
    assert np.allclose(hopf(b["state"]), c["state"], atol=1e-9)              # same n_t, same actions
    assert np.allclose(b["act"], c["act"])
    assert c["meta"]["n"] == 2 and c["meta"]["betti_z2"] == (1, 0, 1)
    assert c["obs"].shape[-1] == 256 and (c["obs"] - lift("berry0", c["state"])).std() < 0.05


def test_berry_phase_accumulates():
    # horizontal lift around a closed base loop returns rotated by e^{-i Omega/2}: the fibre phase is not
    # a function of n, so a latent of n alone cannot predict obs (that is what TO1 tests)
    b = make_bed("berry", 64, 64, "lift", 1)
    p = b["state"][..., 0::2] + 1j * b["state"][..., 1::2]
    n = hopf(b["state"])
    ph = np.angle(p[..., 0] * np.exp(-1j * 0))                                  # phase of psi1
    close = np.linalg.norm(n[:, 1:, None] - n[:, None, :-1], axis=-1) < 0.05   # revisits of the same n
    i, t1, t0 = np.nonzero(close)
    keep = t1 > t0 + 5
    dphi = np.abs(np.angle(np.exp(1j * (ph[i[keep], t1[keep] + 1] - ph[i[keep], t0[keep]]))))
    assert keep.sum() > 50 and np.median(dphi) > 0.3


@pytest.mark.parametrize("name,classes", [("t2", {"a": 1, "b": 1, "ab": 1, "0": 1}),
                                          ("klein", {"a": -1, "b": 1, "0": 1})])
def test_closed_loops(name, classes):
    L = closed_loops(name, n_loops=40, L=128, seed=5)
    S = BEDS[name]["geo"]
    st = L["state"]
    assert st.shape[:2] == (40, 128) and L["obs"].shape == (40, 128, 256) and L["obs"].dtype == np.float32
    step = S(st[:, 1:], st[:, :-1])
    assert (S(st[:, -1], st[:, 0]) < 3 * np.median(step)).all()                 # closed
    assert set(L["cls"]) == set(classes) and all(L["sign"][i] == classes[c] for i, c in enumerate(L["cls"]))
    assert (np.abs(L["obs"] - lift(name, st)).std() < 0.05)                     # same lift as make_bed
    L2 = closed_loops(name, n_loops=40, L=128, seed=6)
    assert not np.allclose(L2["state"], st)


@pytest.mark.parametrize("name", ["berry", "berry0"])
def test_berry_base_walk_uniform_and_small_steps(name):
    """Calibration targets (coordinator): no south lean, median base step 0.1-0.2 rad (s2 is 0.18)."""
    from beds import sample_uniform
    b = make_bed(name, 128, 64, "lift", 1000)
    n = hopf(b["state"]) if name == "berry" else b["state"]
    step = np.arccos(np.clip((n[:, 1:] * n[:, :-1]).sum(-1), -1, 1))
    assert 0.1 <= np.median(step) <= 0.2, np.median(step)
    assert np.abs(n.reshape(-1, 3).mean(0)).max() < 0.05, n.reshape(-1, 3).mean(0)
    ref = sample_uniform("s2", 400, np.random.default_rng(1))
    sub = n.reshape(-1, 3)[np.random.default_rng(2).choice(n.shape[0] * n.shape[1], 3000, replace=False)]
    cr = max(np.arccos(np.clip(sub @ r, -1, 1)).min() for r in ref)
    assert cr < 0.06 * np.pi, cr


def test_berry_torn_and_fold_charts_exist():
    from beds import torn_chart
    import calibrate
    for name in ("berry", "berry0"):
        b = make_bed(name, 4, 8, "lift", 0)
        assert torn_chart(name, b["state"]).shape[:2] == (4, 8)
        assert name in calibrate.DIAM and name in calibrate.FOLD

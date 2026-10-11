"""L5 (certified top-m, REFUSE = blend) and the router decision rule."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa.router import (  # noqa: E402
    ALONE,
    BLEND,
    JUMP,
    OOD,
    REFUSE_BLEND,
    certified_topm,
    decide,
    enclose_scores,
    gamma,
    state_radius,
)
from fcpa.observer import born, parseval_frame  # noqa: E402


def _topm(s, m):
    return set(np.argsort(-s, kind="stable")[:m].tolist())


def test_L5_certified_topm_sound_on_every_box_corner_and_samples():
    rng = np.random.default_rng(0)
    n_cert = n_ref = 0
    for _ in range(400):
        n = int(rng.integers(3, 10))
        m = int(rng.integers(1, n))
        D = rng.standard_normal(n)
        R = rng.uniform(0, 0.8, size=n) * (rng.random(n) < 0.7)
        T, fr = certified_topm(D, R, m)
        if T is None:
            n_ref += 1
            assert fr is not None
            continue
        n_cert += 1
        assert fr is None and len(T) == m
        # worst corner: members pushed down, non-members pushed up; T must survive it
        s = D.copy()
        idx = np.array(sorted(T))
        s[idx] -= R[idx]
        out = np.setdiff1d(np.arange(n), idx)
        s[out] += R[out]
        assert _topm(s, m) == set(T)
        for _ in range(20):  # and random points of the box
            u = D + R * rng.uniform(-1, 1, n)
            assert _topm(u, m) == set(T)
    assert n_cert > 30 and n_ref > 30  # both verdicts exercised


def test_L5_separatrix_counterexample_is_refused():
    # separatrix's pinned false-theorem case, mirrored to "largest": the naive rank-m / rank-(m+1)
    # pair would certify {0,1}, but the box contains (10, 9, 8, 12) whose top-2 is {3, 0}.
    D = np.array([10.0, 9.0, 8.0, 0.0])
    R = np.array([0.0, 0.0, 0.0, 12.0])
    T, fr = certified_topm(D, R, 2)
    assert T is None
    assert fr.outside == 3 and fr.inside == 1  # blocking pair: member 1 (lo=9) vs outsider 3 (hi=12)
    assert fr.gap < 0 or fr.inside_lo <= fr.outside_hi


def test_L5_strict_and_edge_cases():
    D = np.array([3.0, 2.0, 1.0])
    assert certified_topm(D, 0.0, 1)[0] == {0}
    assert certified_topm(D, 0.5, 1)[0] is None  # 2.5 > 2.5 is false: touching boxes are refused
    assert certified_topm(D, 0.49, 1)[0] == {0}
    assert certified_topm(D, 0.5, 3)[0] == {0, 1, 2}  # m = n: nothing outside, trivially certified
    with pytest.raises(ValueError):
        certified_topm(D, 0.1, 0)
    with pytest.raises(ValueError):
        certified_topm(D, np.ones(2), 1)  # R not broadcastable to D
    with pytest.raises(ValueError):
        certified_topm(D, -0.1, 1)  # a negative radius inverts the rule


def test_L5_touching_boxes_refuse():
    # D=[3,2], R=0.5: member lo = 2.5, outsider hi = 2.5 -> NOT strictly greater -> REFUSE
    T, fr = certified_topm(np.array([3.0, 2.0]), 0.5, 1)
    assert T is None and fr is not None


def test_enclose_scores_covers_float32_rounding_error():
    rng = np.random.default_rng(1)
    n = 64
    X = rng.uniform(0, 1, size=(200, n))
    y = rng.uniform(0, 1, size=n)
    exact = X @ y  # float64 (positive terms: the relative form is valid)
    D32 = (X.astype(np.float32) @ y.astype(np.float32)).astype(np.float64)
    # inputs are rounded to f32 as well: n+1 roundings => use gamma_{n+2}
    R = enclose_scores(D32, gamma(n + 2, np.float32))
    assert (R >= 0).all()
    assert (np.abs(D32 - exact) <= R).all()
    assert np.all(R > 0)


def test_enclose_scores_guards():
    assert gamma(1, np.float64) == pytest.approx(2.0**-53 / (1 - 2.0**-53))
    with pytest.raises(ValueError):
        gamma(10**8, np.float32)  # n*u >= 1/2: bound vacuous
    with pytest.raises(ValueError):
        enclose_scores(np.ones(3), 1.0)  # gamma >= 1 would give a negative relative radius
    with pytest.raises(ValueError):
        enclose_scores(np.ones(3), -0.1)
    assert np.allclose(enclose_scores(np.array([-2.0, 2.0]), 0.1), 2 * 0.1 / 0.9)


def test_decide_rule_table():
    # explicit request wins
    d = decide({1, 2}, None, requested=5)
    assert d.kind == JUMP and d.eras == (5,) and not d.ood
    # C = empty -> answer alone, flag OOD (RSL birth candidate)
    d = decide(set(), None)
    assert d.kind == OOD and d.eras == () and d.ood
    # C = {k} -> jump
    d = decide({3}, {3})
    assert d.kind == JUMP and d.eras == (3,) and not d.ood
    # |C| >= 2 with a certified top-m -> blend over the certified eras (within C)
    d = decide({1, 2, 4}, {2, 4})
    assert d.kind == BLEND and d.eras == (2, 4)
    # REFUSE (certified is None) -> blend over all of C
    d = decide({1, 2, 4}, None)
    assert d.kind == REFUSE_BLEND and d.eras == (1, 2, 4)
    # certified set disjoint from C is no evidence about C -> refuse-blend
    d = decide({1, 2}, {7})
    assert d.kind == REFUSE_BLEND and d.eras == (1, 2)
    # C = {null} only: the observer says "no era needed" -> ALONE, not OOD
    d = decide({9}, None, null=9)
    assert d.kind == ALONE and d.eras == () and not d.ood
    # null inside a larger set is an ordinary member of the blend
    d = decide({1, 9}, None, null=9)
    assert d.kind == REFUSE_BLEND and d.eras == (1, 9)
    # kinds are the five documented strings
    assert {ALONE, JUMP, BLEND, REFUSE_BLEND, OOD} == {"ALONE", "JUMP", "BLEND", "REFUSE_BLEND", "OOD"}


def test_E3_radius_from_state_error_is_sound_and_certifies():
    """Amendment A, L5 radius: ||p(psi)-p(psi')||_1 <= 2||psi-psi'|| (E3), so |dp_k| <= ||psi-psi'||."""
    rng = np.random.default_rng(2)
    n_cert = 0
    for _ in range(300):
        N = int(rng.integers(2, 6))
        K = int(rng.integers(N, 10))
        V = parseval_frame(N, K, rng)
        psi = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        psi /= np.linalg.norm(psi)
        eps = 10 ** rng.uniform(-3, -0.5)
        psi2 = psi + eps * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        psi2 /= np.linalg.norm(psi2)
        d = np.linalg.norm(psi - psi2)
        p, p2 = born(V, psi), born(V, psi2)
        assert np.abs(p - p2).sum() <= 2 * d + 1e-12  # E3
        R = state_radius(d)
        assert np.all(np.abs(p - p2) <= R + 1e-15)  # per-coordinate radius for the box rule
        m = int(rng.integers(1, K + 1))
        T, _ = certified_topm(p, R, m)
        if T is not None:
            n_cert += 1
            assert _topm(p2, m) == set(T)  # certified set survives the perturbation
    assert n_cert > 50
    assert state_radius(0.0) == 0.0
    with pytest.raises(ValueError):
        state_radius(-1.0)

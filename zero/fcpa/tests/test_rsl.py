"""RSL: L23/L23'' (birth via RSL), L24* (anytime-valid trigger), L25' (grounding), L26 (Shapley retire)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa.observer import effects_from_frame, parseval_frame  # noqa: E402
from fcpa.rsl import (  # noqa: E402
    RSL,
    GroundingError,
    concat,
    ground,
    leave_one_out,
    shapley,
    tt_output,
)

N, K = 4, 3
FIRE, QUIET = 1e-12, 1.0  # novelty p-values: overwhelming evidence (factor 5e5 clears j(j+1)/delta for j<=70) / none


def _rsl(rng, **kw):
    V = parseval_frame(N, K + 3, rng)  # 7 rows; era effects = 3 rank-one, null = sum of the other 4 (full rank)
    rows = effects_from_frame(V)
    return RSL(rows[:K] + [sum(rows[K:])], **kw)


def _psi(rng, b):
    x = rng.standard_normal((b, N)) + 1j * rng.standard_normal((b, N))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _probs(effects, psi):
    return np.array([np.vdot(psi, e @ psi).real for e in effects])


def _slab(rng, n=30, d=3, t0=100, t1=200, shift=0.0):
    t = rng.integers(t0 - 20, t1 + 20, n)
    X = rng.standard_normal((n, d)) + shift
    return X, t, t0, t1


# ---- L23 / L23'' through RSL --------------------------------------------------
def test_L23_birth_fires_only_on_evidence_and_leaves_old_probabilities():
    rng = np.random.default_rng(0)
    r = _rsl(rng)
    old = [e.copy() for e in r.effects]
    X, t, t0, t1 = _slab(rng)
    g = ground(X, t, t0, t1)
    for _ in range(5):
        assert r.consider_birth(QUIET, g) is None
    assert len(r.effects) == K + 1 and r.births == 0 and r.steps == 5
    k = r.consider_birth(FIRE, g)
    assert k == K and r.births == 1 and len(r.effects) == K + 2 and len(r.eras) == 1
    assert r.eras[0].index == K
    assert np.allclose(sum(r.effects), np.eye(N), atol=1e-12)
    assert all(np.linalg.eigvalsh(e).min() >= -1e-12 for e in r.effects)
    for psi in _psi(rng, 100):
        for j in range(K):
            assert abs(_probs(r.effects, psi)[j] - _probs(old, psi)[j]) <= 1e-12
    assert np.allclose(r.effects[K] + r.effects[K + 1], old[-1], atol=1e-12)  # E_new + E_null' = E_null


def test_L23_birth_cap_and_starved_birth():
    rng = np.random.default_rng(1)
    r = _rsl(rng, n_max=2, n_min=10)
    X, t, t0, t1 = _slab(rng, n=30)
    few = ground(X[:5], np.full(5, 150), t0, t1)
    assert r.consider_birth(FIRE, few) is None and r.starved == 1 and r.births == 0  # n_min = 10 > 5
    ok = ground(X, np.full(30, 150), t0, t1)
    assert r.consider_birth(FIRE, ok) == K
    assert r.consider_birth(FIRE, ok) == K + 1
    assert r.consider_birth(FIRE, ok) is None and r.refused_cap == 1  # n_max = 2 eras born


def test_L23_double_prime_birth_direction_captures_top_r_mass():
    rng = np.random.default_rng(2)
    r = _rsl(rng)
    psis = _psi(rng, 25)
    C = sum(np.outer(p, p.conj()) for p in psis)
    Enull = r.effects[-1].copy()
    w, U = np.linalg.eigh(Enull)
    S = (U * np.sqrt(np.clip(w, 0, None))) @ U.conj().T
    lam = np.linalg.eigvalsh(S @ C @ S)[::-1]
    X, t, t0, t1 = _slab(rng)
    k = r.consider_birth(FIRE, ground(X, t, t0, t1), psis=psis, r=2)
    assert np.trace(r.effects[k] @ C).real == pytest.approx(lam[:2].sum(), rel=1e-9)


# ---- L24* through RSL ----------------------------------------------------------
def test_L24_star_rsl_null_stream_rarely_births():
    rng = np.random.default_rng(3)
    X, t, t0, t1 = _slab(rng)
    g = ground(X, t, t0, t1)
    births = 0
    for run in range(100):
        r = _rsl(np.random.default_rng(run), delta=0.01)
        for p in rng.uniform(0, 1, 1000):  # exact-null p-values (uniform)
            r.consider_birth(float(p), g)
        births += r.births
    assert births <= 4  # P(any birth per run) <= delta/2 = 0.005 -> about 0.5 expected
    r = _rsl(rng)
    for p in [FIRE] * 6:  # a real new regime: evidence accumulates and births do fire
        r.consider_birth(p, g)
    assert r.births >= 2


# ---- L25' --------------------------------------------------------------------
def test_L25_prime_provenance_tags_gate_births_and_updates():
    rng = np.random.default_rng(4)
    r = _rsl(rng)
    X, t, t0, t1 = _slab(rng)
    g = ground(X, t, t0, t1)
    assert len(g) == int(((t >= t0) & (t < t1)).sum())  # slab selection by time
    assert set(g.src.tolist()) == {"slab"}
    bad = tt_output(rng.standard_normal((5, 3)))
    assert set(bad.src.tolist()) == {"tt"}
    with pytest.raises(GroundingError):
        r.consider_birth(FIRE, bad)
    mixed = concat(g, bad)  # one generated row taints the whole batch: never partially admitted
    assert len(mixed) == len(g) + 5
    with pytest.raises(GroundingError):
        r.consider_birth(FIRE, mixed)
    with pytest.raises(GroundingError):
        r.consider_birth(QUIET, mixed)  # gated on every call, not only when a birth fires
    assert r.births == 0 and len(r.eras) == 0
    k = r.consider_birth(FIRE, g)
    n_before = r.eras[0].n
    with pytest.raises(GroundingError):
        r.update(k, bad)
    with pytest.raises(GroundingError):
        r.update(k, mixed)
    assert r.eras[0].n == n_before
    r.update(k, g)
    assert r.eras[0].n == 2 * n_before
    r.assert_grounded()
    assert all(src.startswith("slab") for src, dst in r.dataflow if dst.startswith("era"))


def test_L25_prime_no_generated_content_in_era_training():
    """Scope (Amendment C): era statistics are computed from slab rows only. This is NOT an invariance
    claim about the TT: births still fire on the router's evidence, so the TT can change *whether* an era exists."""
    X, t, t0, t1 = _slab(np.random.default_rng(5), n=80)
    inside = (t >= t0) & (t < t1)
    for depth in (0, 1, 7, 40):
        r = _rsl(np.random.default_rng(6))
        k = r.consider_birth(FIRE, ground(X, t, t0, t1))
        y = np.zeros(3)
        for _ in range(depth):  # TT recursion: reads era estimates, logs its own tokens
            m = r.eras[0].mean  # a copy: the TT cannot mutate the era through it
            y = np.tanh(y + m + r.eras[0].cov.diagonal())
            m += 1.0
            r.log_tt_output(tt_output(y[None, :] + 100.0))
        r.update(k, ground(X[:10], t[:10], t0, t1))
        r.assert_grounded()
        Xs = np.vstack([X[inside], X[:10][inside[:10]]])
        e = r.eras[0]
        assert e.n == len(Xs)
        assert np.allclose(e.mean, Xs.mean(0), atol=1e-12)  # no trace of the +100 TT tokens
        assert np.allclose(e.cov, np.cov(Xs.T, bias=True), atol=1e-12)
        assert len(r.tt_log) == depth


# ---- L26 ---------------------------------------------------------------------
def _additive(a):
    return lambda S: float(sum(a[i] for i in S))


def test_L26_shapley_efficiency_exact_and_sampled():
    rng = np.random.default_rng(7)
    for n in (1, 3, 6):
        w = rng.standard_normal((n, n))  # a non-additive game with pairwise synergy
        v = lambda S, w=w: float(sum(w[i, j] for i in S for j in S)) + 0.3  # v(empty) != 0
        phi = shapley(v, n)
        assert phi.shape == (n,)
        assert phi.sum() == pytest.approx(v(frozenset(range(n))) - v(frozenset()), abs=1e-12)
    n = 14  # sampled: efficiency holds exactly for every permutation sample
    w = rng.standard_normal((n, n))
    v = lambda S: float(sum(w[i, j] for i in S for j in S))
    phi = shapley(v, n, n_perm=200, rng=np.random.default_rng(8))
    assert phi.sum() == pytest.approx(v(frozenset(range(n))) - v(frozenset()), abs=1e-10)
    with pytest.raises(ValueError):
        shapley(v, n)  # exact enumeration refused for large n without n_perm


def test_L26_shapley_known_games_and_loo():
    a = np.array([0.5, -0.2, 0.0, 1.3])
    assert np.allclose(shapley(_additive(a), 4), a, atol=1e-12)  # additive game: phi = a
    assert np.allclose(leave_one_out(_additive(a), 4), a, atol=1e-12)  # and LOO agrees
    # v(S) = |S|^2: symmetric players, phi_i = n; null player gets 0
    assert np.allclose(shapley(lambda S: len(S) ** 2, 5), 5.0)
    phi = shapley(lambda S: float(len(S - {2})), 4)
    assert phi[2] == pytest.approx(0.0, abs=1e-12) and np.allclose(phi[[0, 1, 3]], 1.0)
    # brute-force reference over all permutations
    v = lambda S: float(len(S) ** 3 + (1 in S) * 2.0)
    ref = np.zeros(4)
    for perm in itertools.permutations(range(4)):
        S = frozenset()
        for i in perm:
            ref[i] += v(S | {i}) - v(S)
            S = S | {i}
    assert np.allclose(shapley(v, 4), ref / 24, atol=1e-12)


def test_L26_duplicate_eras_loo_retires_both_shapley_keeps_both():
    # eras 0 and 1 are exact duplicates: v(S) = 1[0 or 1 in S] + 0.5 [2 in S]
    v = lambda S: float((0 in S or 1 in S) + 0.5 * (2 in S))
    assert np.allclose(leave_one_out(v, 3), [0.0, 0.0, 0.5])  # either duplicate looks worthless alone
    assert np.allclose(shapley(v, 3), [0.5, 0.5, 0.5], atol=1e-12)  # Shapley shares the credit
    rng = np.random.default_rng(10)
    for shift in (0.0, 5.0):  # decision uses v(empty) = 0: a constant offset changes nothing
        r = _rsl(rng)
        assert r.retire(lambda S: v(S) + shift, threshold=0.1, method="loo") == [0, 1]
        r = _rsl(rng)
        assert r.retire(lambda S: v(S) + shift, threshold=0.1) == []
        assert len(r.effects) == K + 1


def test_L26_retire_merges_low_value_eras_into_null():
    rng = np.random.default_rng(9)
    r = _rsl(rng)
    X, t, t0, t1 = _slab(rng)
    g = ground(X, t, t0, t1)
    r.consider_birth(FIRE, g)  # now 4 eras
    r.consider_birth(FIRE, g)  # 5 eras
    gains = np.array([0.4, 0.01, 0.3, -0.05, 0.2])
    before = [e.copy() for e in r.effects]
    retired = r.retire(_additive(gains), threshold=0.05)
    assert retired == [1, 3]
    assert len(r.effects) == 5 - 2 + 1 and np.allclose(sum(r.effects), np.eye(N), atol=1e-12)
    assert all(np.linalg.eigvalsh(e).min() >= -1e-12 for e in r.effects)
    keep = [0, 2, 4]
    for new, old in enumerate(keep):  # survivors untouched
        assert np.array_equal(r.effects[new], before[old])
    assert np.allclose(r.effects[-1], before[-1] + before[1] + before[3], atol=1e-12)  # mass lands in null
    assert r.retired == 2
    assert r.retire(_additive(gains[[0, 2, 4]]), threshold=-1.0) == []


def test_L26_retire_reindexes_survivors_contiguously_and_update_still_works():
    rng = np.random.default_rng(11)
    r = _rsl(rng)
    X, t, t0, t1 = _slab(rng)
    g = ground(X, t, t0, t1)
    assert r.consider_birth(FIRE, g) == 3 and r.consider_birth(FIRE, g) == 4  # effects: 3 initial, 2 born, null
    gains = np.array([0.0, 0.0, 1.0, 1.0, 1.0])  # retire initial eras 0 and 1
    assert r.retire(_additive(gains), threshold=0.5) == [0, 1]
    assert [e.index for e in r.eras] == [1, 2]  # 3, 4 shifted down by the two removed eras
    assert len(r.effects) == 4 and sorted(e.index for e in r.eras) == list(range(1, 3))
    n0 = r.eras[0].n
    r.update(1, g)  # the surviving era is addressed by its NEW index
    assert r.eras[0].n == 2 * n0
    with pytest.raises(IndexError):
        r.update(4, g)  # the old index no longer exists
    r.assert_grounded()


@pytest.mark.parametrize("method", ["shapley", "loo"])
def test_L26_retire_hands_the_decision_a_value_function_with_v_empty_zero(method, monkeypatch):
    import fcpa.rsl as rsl_mod

    seen = []

    def spy(real):
        def f(vf, n, *a, **kw):
            seen.append(vf(frozenset()))
            return real(vf, n, *a, **kw)
        return f

    monkeypatch.setattr(rsl_mod, "shapley", spy(rsl_mod.shapley))
    monkeypatch.setattr(rsl_mod, "leave_one_out", spy(rsl_mod.leave_one_out))
    r = _rsl(np.random.default_rng(12))
    r.retire(lambda S: 7.0 + len(S), threshold=0.0, method=method)  # v(empty) = 7 on the raw function
    assert seen == [0.0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_H3_birth_within_40_steps_of_regime_change_after_1e4_quiet_steps(seed):
    """Mixture e-detector wiring: no floor needed after a long quiet stretch (the product martingale sits near
    exp(-1900) by then and needs >1000 steps).  Novel regime = novelty p-value 0.02 each step."""
    rng = np.random.default_rng(seed)
    g = ground(*_slab(rng))
    r = _rsl(rng, delta=0.01)
    for p in rng.uniform(0, 1, 10_000):  # quiet stretch: exact-null p-values
        r.consider_birth(float(p), g)
    assert r.births == 0
    delay = None
    for i in range(1, 200):
        if r.consider_birth(0.02, g) is not None:
            delay = i
            break
    assert delay is not None and delay <= 40
    assert r.trigger.delta == pytest.approx(0.01 / (2 * 3))  # reset to delta_2 = delta / (j (j + 1)), j = 2

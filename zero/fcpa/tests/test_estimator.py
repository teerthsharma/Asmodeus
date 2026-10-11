"""L27' (exact partial_fit with the w-sample buffer), TT-0/1 (softmax predictor), router wiring, check_estimator."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa.estimator import FCPARegressor  # noqa: E402


def _two_regimes(n=900, seed=0, shift=4.0):
    """First half: x ~ N(0, I), y = 2 x0 - x1.  Second half: x ~ N(shift, I), y = -2 x0 + x2 + 3."""
    rng = np.random.default_rng(seed)
    h = n // 2
    A = rng.standard_normal((h, 3))
    B = rng.standard_normal((n - h, 3)) + shift
    y = np.r_[2 * A[:, 0] - A[:, 1], -2 * B[:, 0] + B[:, 2] + 3.0]
    return np.vstack([A, B]), y + 0.05 * rng.standard_normal(n), np.arange(n, dtype=float)


KW = dict(window=60, d_chart=4, random_state=0)


def _chunks(n, sizes):
    cuts = np.cumsum(sizes)[:-1]
    return np.split(np.arange(n), cuts)


def test_L27prime_chunked_partial_fit_equals_fit_with_buffer():
    X, y, t = _two_regimes()
    edges = FCPARegressor.make_edges(y, 8)
    a = FCPARegressor(y_edges=edges, **KW).fit(X, y, t=t)
    b = FCPARegressor(y_edges=edges, **KW)
    for idx in _chunks(len(X), [37, 111, 5, 250, 197, 300]):  # none is a multiple of the window
        b.partial_fit(X[idx], y[idx], t=t[idx])
    assert a.n_eras_ == b.n_eras_ >= 2  # the regime switch was found, in both
    for ca, cb in zip(a.law_counts_, b.law_counts_):
        assert np.array_equal(ca, cb)
    Xq = X[::7]
    assert np.allclose(a.predict(Xq), b.predict(Xq), atol=1e-8)
    assert np.allclose(a.tt_proba(Xq), b.tt_proba(Xq), atol=1e-10)


def test_L27prime_exactness_fails_without_the_buffer():
    X, y, t = _two_regimes()
    edges = FCPARegressor.make_edges(y, 8)
    a = FCPARegressor(y_edges=edges, **KW).fit(X, y, t=t)
    b = FCPARegressor(y_edges=edges, **KW)
    for idx in _chunks(len(X), [37, 111, 5, 250, 197, 300]):
        b.partial_fit(X[idx], y[idx], t=t[idx])
        b._drop_buffer()  # emulate an implementation that forgets the open window between calls
    assert sum(c.sum() for c in a.law_counts_) != sum(c.sum() for c in b.law_counts_) or not np.allclose(
        a.predict(X[::7]), b.predict(X[::7]), atol=1e-8
    )


def test_L27prime_chunk_boundary_transition_is_counted():
    X, y, t = _two_regimes(n=300, shift=0.0)
    kw = dict(window=40, d_chart=4, random_state=1)
    a = FCPARegressor(**kw).fit(X, y)
    edges = a.y_edges_
    b = FCPARegressor(y_edges=edges, **kw).partial_fit(X[:100], y[:100])
    b.partial_fit(X[100:], y[100:])  # symbol a = last of chunk 1, b = first of chunk 2: a->b must be counted
    n_trans = lambda m: int(sum(c.sum() for c in m.law_counts_))
    assert n_trans(b) == len(X) - 1
    assert n_trans(FCPARegressor(y_edges=edges, **kw).fit(X, y)) == len(X) - 1


def test_timestamps_order_the_stream():
    X, y, t = _two_regimes(n=400)
    perm = np.random.default_rng(3).permutation(len(X))
    edges = FCPARegressor.make_edges(y, 6)
    a = FCPARegressor(y_edges=edges, **KW).fit(X, y)
    b = FCPARegressor(y_edges=edges, **KW).fit(X[perm], y[perm], t=t[perm])
    assert np.allclose(a.predict(X[::5]), b.predict(X[::5]), atol=1e-9)


def test_TT_is_a_softmax_over_next_step_bins():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 2))
    y = 3 * X[:, 0] + 0.1 * rng.standard_normal(600)
    m = FCPARegressor(n_bins=6, window=100, random_state=0).fit(X, y)
    P = m.tt_proba(X)
    assert P.shape == (600, 6) and np.allclose(P.sum(1), 1.0) and (P >= 0).all()
    yh = m.predict_tt(X)
    assert 1 - np.mean((yh - y) ** 2) / y.var() > 0.8  # the TT alone answers


def test_router_is_called_and_jumps_to_the_matching_era():
    X, y, t = _two_regimes(seed=2)
    m = FCPARegressor(**KW).fit(X, y, t=t)
    dec = m.route(X)
    kinds = {d.kind for d in dec}
    assert kinds <= {"ALONE", "JUMP", "BLEND", "REFUSE_BLEND", "OOD"}
    first = [d for d, i in zip(dec, range(len(X))) if i < 300 and d.kind == "JUMP"]
    last = [d for d, i in zip(dec, range(len(X))) if i >= 600 and d.kind == "JUMP"]
    assert len(first) > 150 and len(last) > 150
    assert np.mean([d.eras[0] == first[0].eras[0] for d in first]) > 0.9
    assert np.mean([d.eras[0] == last[0].eras[0] for d in last]) > 0.9
    assert first[0].eras[0] != last[0].eras[0]
    # JUMP rows are answered by the era model through the crossing operator, not by the TT
    jump_rows = np.array([i for i, d in enumerate(dec) if d.kind == "JUMP"])[:20]
    assert np.allclose(m.predict(X[jump_rows]), [m.era_predict(dec[i].eras[0], X[i:i + 1])[0] for i in jump_rows])
    # a point no era has seen is flagged OOD and answered by the TT alone
    far = np.array([[60.0, -60.0, 60.0]])
    d = m.route(far)[0]
    assert d.kind in ("OOD", "REFUSE_BLEND", "BLEND", "JUMP")
    if d.kind == "OOD":
        assert np.allclose(m.predict(far), m.predict_tt(far))


def test_eras_beat_a_global_linear_model_on_regime_switching_data():
    from sklearn.linear_model import Ridge

    X, y, t = _two_regimes(n=1200, seed=4)
    Xte, yte, _ = _two_regimes(n=600, seed=5)
    fc = FCPARegressor(**KW).fit(X, y, t=t)
    g = Ridge(alpha=1e-3).fit(X, y)
    err = lambda p: np.mean((p - yte) ** 2)
    assert err(fc.predict(Xte)) < 0.5 * err(g.predict(Xte))


def test_y_none_is_next_step_prediction():
    rng = np.random.default_rng(0)
    X = np.zeros((500, 2))
    for i in range(1, 500):
        X[i] = 0.9 * X[i - 1] + rng.standard_normal(2) * 0.3
    m = FCPARegressor(window=100, random_state=0).fit(X)
    P = m.predict(X[:-1])
    assert P.shape == (499, 2)
    assert np.mean((P - X[1:]) ** 2) < np.mean((X[1:] - X[1:].mean(0)) ** 2)


def test_chunked_y_none_counts_the_cross_chunk_pair():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 2)).cumsum(0) * 0.1
    kw = dict(window=40, random_state=0)
    ref = FCPARegressor(**kw).fit(X)
    b = FCPARegressor(y_edges=ref.y_edges_, **kw).partial_fit(X[:77])
    b.partial_fit(X[77:])
    assert b.n_samples_seen_ == ref.n_samples_seen_ == 199


@parametrize_with_checks([FCPARegressor(window=16, random_state=0)])
def test_sklearn_check_estimator(estimator, check):
    check(estimator)


def _stationary_linear(n_train=5000, n_test=2000, seed=0, d=3):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    Xtr, Xte = rng.standard_normal((n_train, d)), rng.standard_normal((n_test, d))
    return Xtr, Xtr @ w + 0.5 + 0.1 * rng.standard_normal(n_train), Xte, Xte @ w + 0.5


def test_stationary_linear_data_matches_linear_regression_within_2x():
    from sklearn.linear_model import LinearRegression

    fc, lr = [], []
    for seed in range(5):  # one noise realisation can favour either side by 2x; compare the seed means
        Xtr, ytr, Xte, yte = _stationary_linear(seed=seed)
        rmse = lambda p: float(np.sqrt(np.mean((p - yte) ** 2)))
        lr.append(rmse(LinearRegression().fit(Xtr, ytr).predict(Xte)))
        fc.append(rmse(FCPARegressor(window=60, d_chart=4, random_state=0).fit(Xtr, ytr).predict(Xte)))
    assert np.mean(fc) <= 2 * np.mean(lr), (np.mean(fc), np.mean(lr))


def test_in_distribution_ood_fallback_rate_is_at_most_alpha_and_warns_when_it_happens():
    Xtr, ytr, Xte, _ = _stationary_linear(n_train=6000, n_test=3000, seed=1)
    m = FCPARegressor(window=60, d_chart=4, alpha=0.1, random_state=0).fit(Xtr, ytr)
    rate = np.mean([d.kind == "OOD" for d in m.route(Xte)])
    assert rate <= m.alpha, rate
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # in-distribution prediction is silent
        m.predict(Xte[:200])
    far = np.full((5, Xtr.shape[1]), 80.0)
    with pytest.warns(UserWarning, match="OOD"):
        m.predict(far)


def _ar(n=1500, d=20, seed=0, theta=0.1, sig=0.09):
    rng = np.random.default_rng(seed)
    mu = 3 + rng.standard_normal(d)
    X = np.zeros((n, d))
    X[0] = mu
    for t in range(1, n):
        X[t] = X[t - 1] + theta * (mu - X[t - 1]) + sig * rng.standard_normal(d)
    return X


def _random_walk(n=1500, d=20, seed=0, step=0.05, offset=3.0):
    rng = np.random.default_rng(seed)
    return offset + np.cumsum(step * rng.standard_normal((n, d)), axis=0)


def test_y_none_predicts_the_increment_with_an_exact_skip_path():
    """Persistence is the zero model: on a stationary AR(1) the next-step error cannot exceed persistence's, even
    with D = 20 > 2 * d_chart, i.e. through a chart that cannot hold the state."""
    X = _ar()
    tr, te = X[:1000], X[1000:]
    m = FCPARegressor(window=60, d_chart=4, random_state=0).fit(tr)
    err = lambda p: float(np.linalg.norm(p - te[1:]))
    assert err(m.predict(te[:-1])) <= err(te[:-1])
    # the skip path is exact: predict(x) - x is the increment model alone, so a zero increment is x itself
    Xc = np.tile(np.arange(20.0), (300, 1)) + 0.0
    mc = FCPARegressor(window=60, d_chart=4, random_state=0).fit(Xc)
    assert np.allclose(mc.predict(Xc[:5]), Xc[:5], atol=1e-9)


def test_encoder_E1_is_injective_when_the_state_is_wider_than_the_chart():
    """The linear encoder phi = R [x; 1] must keep the state recoverable (2N >= D + 1), whatever d_chart is."""
    X = _random_walk(n=200, d=60)
    m = FCPARegressor(window=60, d_chart=4, random_state=0).fit(X)
    _, phi, _ = m._encode(X)
    xt = np.c_[X, np.ones(len(X))]
    rec = np.linalg.lstsq(phi, xt, rcond=None)[0]
    assert np.linalg.norm(phi @ rec - xt) / np.linalg.norm(xt) < 1e-9


def _stack2(X):
    """rows [x_t, x_{t-1}] for t = 1 .. T-1 of a time-ordered (T, D) array"""
    return np.hstack([X[1:], X[:-1]])


def test_lag_stack_lets_the_increment_model_learn_velocity():
    """Each coordinate is a lightly damped oscillator, x_{t+1} = 2 r cos(w) x_t - r^2 x_{t-1} + noise: persistence
    is poor, and the velocity is only in the lag features."""
    rng = np.random.default_rng(0)
    r, w = 0.97, 0.3
    X = np.zeros((1500, 6))
    X[0], X[1] = rng.standard_normal(6), rng.standard_normal(6)
    for t in range(1, 1499):
        X[t + 1] = 2 * r * np.cos(w) * X[t] - r * r * X[t - 1] + 0.003 * rng.standard_normal(6)
    S = _stack2(X)
    tr, te = S[:1000], S[1000:]
    m = FCPARegressor(window=60, d_chart=2, n_lags=2, random_state=0).fit(tr)
    P = m.predict(te[:-1])
    assert P.shape == te[:-1].shape  # the next stacked row [x_{t+1}, x_t]
    assert np.allclose(P[:, 6:], te[:-1, :6])  # the shift is exact: x_t becomes the lag
    err = lambda p: float(np.linalg.norm(p[:, :6] - te[1:, :6]))
    persist = err(te[:-1])
    assert err(P) <= 0.5 * persist, (err(P), persist)
    one = FCPARegressor(window=60, d_chart=2, random_state=0).fit(X[:1000])  # n_lags = 1 cannot see the velocity
    err1 = np.linalg.norm(one.predict(X[1001:-1]) - X[1002:])
    assert err1 > 0.8 * persist, (err1, persist)


def test_n_lags_must_divide_the_feature_count():
    with pytest.raises(ValueError, match="n_lags"):
        FCPARegressor(n_lags=2).fit(np.zeros((50, 5)))


# ---- law-based era identification (L14', L8/E2') --------------------------------------------------------
class _ToyLaw:
    """R = (x_t - x_{t-1}) / dt and L = -k^2 x_{t-1} for rows [x_t, x_{t-1}]: the model R = nu L + f."""

    def __init__(self, D, dt):
        self.D, self.dt, self.k2 = D, dt, np.arange(1, D + 1, dtype=float)

    def __call__(self, S):
        return (S[:, :self.D] - S[:, self.D:2 * self.D]) / self.dt, -self.k2 * S[:, self.D:2 * self.D]


def _toy_regimes(n_traj=36, L=300, D=12, dt=0.02, seed=0):
    """Two laws with the SAME stationary law of the state (same fixed point, same variance): only the dynamics differ."""
    rng = np.random.default_rng(seed)
    k2 = np.arange(1, D + 1, dtype=float)
    xs = rng.standard_normal(D)
    nus = (0.4, 1.6)
    X, R = np.zeros((n_traj, L, D)), np.zeros((n_traj, L), int)
    for i in range(n_traj):
        r, x = rng.integers(2), xs.copy()
        for t in range(L):
            if rng.random() < 1 / 120:
                r = 1 - r
            nu = nus[r]
            x = x + dt * (-nu * k2 * (x - xs)) + 0.12 * np.sqrt(nu * dt) * rng.standard_normal(D)
            X[i, t], R[i, t] = x, r
    return X, R, _ToyLaw(D, dt)


def _id_score(est, X, R, tr, ca, te, w=20, filt=False):
    """P3 as run_primary computes it: era map frozen on cal, then correct / total on rows > w from a switch."""
    from fcpa import bars

    D2 = X.shape[-1]
    S = np.concatenate([X[:, 1:], X[:, :-1]], -1)
    Rg = R[:, 1:]
    scores = lambda Z: est._nc_scores(est._model["segs"], est._encode(Z)[1])
    K = est.n_eras_
    cal_lab = np.concatenate([scores(S[i]).argmin(1) for i in ca])
    emap = bars.era_map(cal_lab, Rg[ca].reshape(-1), K)
    corr = tot = 0
    for i in te:
        nc = scores(S[i])
        if filt:
            nc = est.hmm_filter(nc)
        pred = np.array([emap[k] for k in nc.argmin(1)])
        c, n_ = bars.id_counts(pred, Rg[i], w)
        corr, tot = corr + c, tot + n_
    return corr / tot


def test_law_based_eras_identify_regimes_that_the_state_cannot():
    X, R, law = _toy_regimes()
    S = np.concatenate([X[:, 1:], X[:, :-1]], -1)
    tr, ca, te = np.arange(0, 20), np.arange(20, 28), np.arange(28, 36)
    kw = dict(window=20, d_chart=2, n_lags=2, n_eras_max=4, random_state=0)
    st = FCPARegressor(**kw)  # state-only eras: the two laws share the stationary state distribution
    law_est = FCPARegressor(law=law, **kw)
    for tj in S[tr]:
        st.partial_fit(tj), law_est.partial_fit(tj)
        st._st.last_x = law_est._st.last_x = None
    assert law_est.n_eras_ == 2, law_est.n_eras_
    nus = sorted(sg.law.model()[0] for sg in law_est._model["segs"])  # the eras ARE the two laws (not a blend)
    assert nus == pytest.approx([0.4, 1.6], rel=0.1), nus
    assert _id_score(law_est, X, R, tr, ca, te) >= 0.9
    assert _id_score(st, X, R, tr, ca, te) < 0.7  # the contrast: state evidence cannot do it


def test_hmm_filter_is_a_forward_filter_with_a_stay_prior():
    nc = np.array([[0.0, 9.0], [0.0, 9.0], [9.0, 0.0], [9.0, 0.0], [9.0, 0.0]])
    m = FCPARegressor()
    f = m.hmm_filter(nc, stay=0.9)
    assert list(f.argmin(1)) == [0, 0, 1, 1, 1] and f.shape == nc.shape
    one = m.hmm_filter(np.array([[0.0, 1.0], [0.3, 0.0], [0.3, 0.0]]), stay=0.99)  # weak evidence: the prior holds
    assert one[1].argmin() == 0


def test_damping_scales_the_increment_exactly_and_only_for_y_none():
    rng = np.random.default_rng(0)
    X = np.zeros((1500, 6))
    X[0], X[1] = rng.standard_normal(6), rng.standard_normal(6)
    for t in range(1, 1499):  # a lightly damped oscillator: the lag stack predicts the increment well
        X[t + 1] = 2 * 0.97 * np.cos(0.3) * X[t] - 0.97**2 * X[t - 1] + 0.003 * rng.standard_normal(6)
    S = _stack2(X)
    m = FCPARegressor(window=60, d_chart=2, n_lags=2, random_state=0).fit(S[:1000])
    q = S[1000:1100]
    base = m.predict(q)[:, :6] - q[:, :6]
    half = m.set_params(damping=0.5).predict(q)[:, :6] - q[:, :6]
    assert np.abs(base).max() > 0 and np.allclose(half, 0.5 * base, atol=1e-12)
    Xr, yr, Xt, _ = _stationary_linear(n_train=1500, n_test=100)
    r = FCPARegressor(window=60, d_chart=4, random_state=0).fit(Xr, yr)
    assert np.allclose(r.set_params(damping=0.3).predict(Xt), r.set_params(damping=1.0).predict(Xt))  # regression


def _oscillator(n=1500, d=6, seed=0, noise=0.02):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, d))
    X[0], X[1] = rng.standard_normal(d), rng.standard_normal(d)
    for t in range(1, n - 1):
        X[t + 1] = 2 * 0.97 * np.cos(0.3) * X[t] - 0.97**2 * X[t - 1] + noise * rng.standard_normal(d)
    return X


def test_energy_projection_fixes_the_norm_ratio_to_the_fitted_era_value():
    """energy=True: after each step the newest block has norm g_k |x_t|, g_k the mean norm ratio of era k's slab
    pairs (additive statistics). energy=False leaves the model's own norm."""
    S = _stack2(_oscillator())
    tr, q = S[:1000], S[1000:1100]
    m = FCPARegressor(window=60, d_chart=2, n_lags=2, random_state=0, energy=True).fit(tr)
    g = m._model["g"]
    assert len(g) == m.n_eras_ == 1 and 0.5 < g[0] < 1.5
    P = m.predict(q)
    assert np.allclose(np.linalg.norm(P[:, :6], axis=1), g[0] * np.linalg.norm(q[:, :6], axis=1), rtol=1e-9)
    free = m.set_params(energy=False).predict(q)
    assert not np.allclose(np.linalg.norm(free[:, :6], axis=1), g[0] * np.linalg.norm(q[:, :6], axis=1), rtol=1e-6)


def test_energy_ratio_statistics_are_exact_under_partial_fit():
    S = _stack2(_oscillator(seed=1))
    edges = FCPARegressor.make_step_edges(S, n_lags=2)
    kw = dict(window=60, d_chart=2, n_lags=2, random_state=0, energy=True, y_edges=edges)
    a = FCPARegressor(**kw).fit(S)
    b = FCPARegressor(**kw)
    for idx in _chunks(len(S), [37, 111, 5, 250, 300, 796]):
        b.partial_fit(S[idx])
    assert np.allclose(a._model["g"], b._model["g"], rtol=1e-12)
    assert np.allclose(a.predict(S[::9]), b.predict(S[::9]), atol=1e-8)

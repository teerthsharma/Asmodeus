"""FCPARegressor on the NS-2D bed, TUNING seed 3 only (bed seed 2003). Seeds 0, 1, 2, 4, 5 are never touched here.

The bed is generated in the fixture (32 trajectories, L = 300, the pinned laws), so the tests need no stored data.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa import bars  # noqa: E402
from fcpa import run_primary as rp  # noqa: E402
from fcpa.estimator import FCPARegressor  # noqa: E402
from fcpa.laws import NS2DLaw  # noqa: E402

P = dict(rp.PINS, n_traj=32, L=300, origins=(20, 40, 60, 80, 100), horizon=50)


@pytest.fixture(scope="module")
def bed():
    X, reg = rp.gen_bed(rp.TUNE_BED_SEED, P, workers=4)
    return X, reg


def _fit(Xtr, **kw):
    est = FCPARegressor(n_eras_max=P["n_eras_max"], d_chart=P["d_chart"], alpha=bars.ALPHA, gamma=bars.GAMMA,
                        window=P["w"], random_state=bars.TUNE_SEED, y_edges=FCPARegressor.make_step_edges(Xtr), **kw)
    for tr in Xtr:  # one chunk per trajectory, as run_primary.fit_fcpa does
        est.partial_fit(tr)
        est._st.last_x = None
    return est


def test_ns3_one_step_error_is_not_worse_than_persistence(bed):
    X, _ = bed
    est = _fit(X[:24])
    Xt = X[24:]
    D = Xt.shape[-1]
    a, b = Xt[:, :-1].reshape(-1, D), Xt[:, 1:].reshape(-1, D)
    ours, persist = bars.rel_err(est.predict(a), b).mean(), bars.rel_err(a, b).mean()
    assert ours <= persist + 1e-12, (ours, persist)


def test_ns3_closed_loop_E50_is_not_worse_than_persistence(bed):
    X, _ = bed
    est = _fit(X[:24])
    ours = rp.rollout_errors(est.predict, X[24:], P).mean()
    persist = rp.rollout_errors(lambda x: x, X[24:], P).mean()
    assert ours <= persist + 1e-12, (ours, persist)


def test_ns3_encoder_E1_path_reconstructs_the_state(bed):
    """The linear encoder phi = R [x; 1] is injective on the 90-dim state: reconstruction error ~ 1e-13 (before the
    fix N = 2 d_chart = 16 left 32 reals for 91 unknowns)."""
    X, _ = bed
    est = _fit(X[:2])
    Z = X[0]
    _, phi, _ = est._encode(Z)
    rec = np.linalg.lstsq(phi, np.c_[Z, np.ones(len(Z))], rcond=None)[0]
    rel = np.linalg.norm(phi @ rec - np.c_[Z, np.ones(len(Z))]) / np.linalg.norm(Z)
    assert rel < 1e-9


def _stack2(X):
    return np.concatenate([X[:, 1:], X[:, :-1]], axis=-1)  # rows [x_t, x_{t-1}], t = 1 .. L-1


def _fit_stacked(S, law=True):
    return FCPARegressor(n_eras_max=P["n_eras_max"], d_chart=P["d_chart"], alpha=bars.ALPHA, gamma=bars.GAMMA,
                         window=P["w"], random_state=bars.TUNE_SEED, n_lags=2, law=NS2DLaw() if law else None,
                         y_edges=FCPARegressor.make_step_edges(S, n_lags=2))


def _train(est, S):
    for tr in S:  # one chunk per trajectory, as run_primary.fit_fcpa does
        est.partial_fit(tr)
        est._st.last_x = None
    return est


def _one_step(est, S):
    D = S.shape[-1] // 2
    a, b = S[:, :-1].reshape(-1, 2 * D), S[:, 1:].reshape(-1, 2 * D)
    return bars.rel_err(est.predict(a)[:, :D], b[:, :D]).mean(), bars.rel_err(a[:, :D], b[:, :D]).mean()


def test_ns3_lag_stack_beats_persistence_on_the_dev_bed(bed):
    """24 training trajectories are too few for the 0.5x bar (see the gated full-bed test); the velocity in the lag
    features must still win over persistence."""
    X, _ = bed
    S = _stack2(X)
    ours, persist = _one_step(_train(_fit_stacked(S[:24]), S[:24]), S[24:])
    assert ours < persist, (ours, persist)


@pytest.mark.skipif(not os.environ.get("FCPA_NS3_CACHE"), reason="needs the full seed-3 bed: set FCPA_NS3_CACHE")
def test_ns3_lag_stack_one_step_error_is_at_most_half_of_persistence():
    """Full tuning bed (128 / 64 / 64): the lag stack lets the era models learn velocity, error <= 0.5x persistence."""
    X, reg = rp.gen_bed(rp.TUNE_BED_SEED, rp.PINS, workers=1, cache=os.environ["FCPA_NS3_CACHE"])
    tr, ca, te = rp.three_way_split(len(X), bars.TUNE_SEED)
    S = _stack2(X)
    ours, persist = _one_step(_train(_fit_stacked(S[tr]), S[tr]), S[te])
    assert ours <= 0.5 * persist, (ours, persist)


@pytest.mark.skipif(not os.environ.get("FCPA_NS3_CACHE"),
                    reason="needs the full seed-3 bed (256 x 400): set FCPA_NS3_CACHE to a directory for gen_bed's cache")
def test_ns3_P3_identification_at_least_0_9_on_the_full_tuning_bed():
    """P3 >= 0.9 on seed 3 (128 / 64 / 64 split) with the lag stack and the NS2D law plug-in, scored by
    run_primary.fcpa_stream on the stacked rows."""
    X, reg = rp.gen_bed(rp.TUNE_BED_SEED, rp.PINS, workers=1, cache=os.environ["FCPA_NS3_CACHE"])
    tr, ca, te = rp.three_way_split(len(X), bars.TUNE_SEED)
    S, Rg = _stack2(X), reg[:, 1:]
    est = FCPARegressor(n_eras_max=rp.PINS["n_eras_max"], d_chart=rp.PINS["d_chart"], alpha=bars.ALPHA,
                        gamma=bars.GAMMA, window=rp.PINS["w"], random_state=bars.TUNE_SEED, n_lags=2, law=NS2DLaw(),
                        y_edges=FCPARegressor.make_step_edges(S[tr], n_lags=2))
    for t in S[tr]:
        est.partial_fit(t)
        est._st.last_x = None
    st = rp.fcpa_stream(est, S[ca], Rg[ca], S[te], Rg[te], rp.PINS)
    assert sum(st["correct"]) / sum(st["total"]) >= 0.9

"""H3: mixture-over-change-time e-detector birth trigger (S_t = (S_{t-1} + w_t) e_t, M_t = S_t + 1/(t+1))."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa.edetector import MixtureBirthTrigger, first_birth, step  # noqa: E402

DELTA, KAPPA = 0.05, 0.5


def _old_delays(P, L, delta=DELTA, kappa=KAPPA):
    """Old product martingale (no floor, no restart), threshold 2/delta as in conformal.BirthTrigger j=1."""
    with np.errstate(divide="ignore"):
        lf = math.log(kappa) + (kappa - 1.0) * np.log(P)
    hit = np.cumsum(lf, axis=1) >= math.log(2.0 / delta)
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    return first


def _shift_stream(rng, n, L, extra):
    P = rng.random((n, L + extra))
    P[:, L:] = P[:, L:] ** 3
    return P


def _delays(first, L, extra):
    ok = (first >= L) | (first < 0)  # not fired before the change
    d = np.where(first < 0, extra, first - L)  # never fired: censored at the horizon
    return d[ok], int((~ok).sum())


def test_H3_M0_is_one_and_martingale_property_monte_carlo():
    assert MixtureBirthTrigger(delta=DELTA, kappas=(0.5,)).M == 1.0
    assert MixtureBirthTrigger(delta=DELTA, kappas=(0.3, 0.5, 0.8)).M == 1.0
    rng = np.random.default_rng(0)
    for kappa in (0.8, 0.9):  # finite-variance kappas so the Monte-Carlo SE is honest
        for n_past in (0, 2, 5):  # short pasts: the w_t terms dominate there, so a wrong w is visible
            S, t = 0.0, 0
            for p in rng.random(n_past):  # an arbitrary past
                t += 1
                S = step(S, t, p, kappa)
            M_prev = S + 1.0 / (t + 1)
            p = rng.random(4_000_000)
            M_next = step(S, t + 1, p, kappa) + 1.0 / (t + 2)
            se = M_next.std() / math.sqrt(len(p))
            assert abs(M_next.mean() - M_prev) < 4 * se, (kappa, n_past, M_next.mean(), M_prev, se)


def test_H3_delay_after_shift_beats_product_martingale_at_long_null_prefix():
    rng = np.random.default_rng(1)
    extra, n = 4000, 500  # horizon long enough for the old arm to recover from its drift
    med = {}
    for L in (100, 10_000):
        P = _shift_stream(rng, n, L, extra)
        d, early = _delays(first_birth(P, DELTA, (KAPPA,)), L, extra)
        old, _ = _delays(_old_delays(P, L), L, extra)
        med[L] = (float(np.median(d)), float(np.median(old)), early)
        print(f"L={L}: e-detector median delay {med[L][0]}, old product {med[L][1]}, early fires {early}/{n}")
    assert med[10_000][0] <= 40
    assert med[100][0] <= 40
    assert med[10_000][1] >= 1000
    assert med[10_000][1] > 10 * med[10_000][0]


def test_H3_false_birth_rate_under_null_within_ville():
    rng = np.random.default_rng(2)
    n, T, fires = 2000, 10_000, 0
    for _ in range(4):
        fires += int((first_birth(rng.random((n // 4, T)), DELTA, (KAPPA,)) >= 0).sum())
    rate = fires / n
    se = math.sqrt(DELTA * (1 - DELTA) / n)
    print(f"false-birth rate {fires}/{n} = {rate:.4f} (bar {DELTA + 2 * se:.4f})")
    assert rate <= DELTA + 2 * se


def test_H3_kappa_grid_matches_scalar_and_streaming_matches_vectorised():
    rng = np.random.default_rng(3)
    P = _shift_stream(rng, 20, 300, 100)
    grid = (0.3, 0.5, 0.8)
    f = first_birth(P, DELTA, grid)
    for row, fi in zip(P, f):
        trig = MixtureBirthTrigger(delta=DELTA, kappas=grid)
        got = next((i for i, p in enumerate(row) if trig.update(p)), -1)
        assert got == fi

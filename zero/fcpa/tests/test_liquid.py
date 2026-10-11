"""L20 discrete liquid invariant interval: 0<=F<=M, h <= tau/(1+tau*M), x0 in [min(0,Amin), max(0,Amax)] => x_n stays."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa.liquid import box_distance, invariant_box, liquid_step, max_step  # noqa: E402


def test_L20_box_holds_1e5_random_steps():
    rng = np.random.default_rng(0)
    tau, M = 0.7, 2.5
    h = max_step(tau, M)  # the boundary itself
    A = rng.uniform(-1.5, 3.0, size=16)
    lo, hi = invariant_box(A)
    x = rng.uniform(lo, hi, size=16)
    for _ in range(100_000):
        I = rng.standard_normal(16)
        x = liquid_step(x, I, lambda x_, I_: M * rng.uniform(size=16), tau, A, h, M=M)
        assert np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12)


def test_L20_counterexample_when_h_too_large():
    # tau=1, M=1 -> h* = 0.5. h=1, F=M, A=[1,0], x0=[0,1] in B=[0,1]: coordinate 2 goes to 1 - 1 - 1 = -1.
    tau, M, h = 1.0, 1.0, 1.0
    A, x0 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    F = lambda x, I: np.full(2, M)
    lo, hi = invariant_box(A)
    with pytest.raises(ValueError):  # the guard refuses the unsafe step
        liquid_step(x0, None, F, tau, A, h, M=M)
    x1 = liquid_step(x0, None, F, tau, A, h, M=M, check=False)  # the raw map really escapes
    assert x1[1] < lo and box_distance(x1, lo, hi) == pytest.approx(1.0)


def test_L20_rejects_F_out_of_range():
    with pytest.raises(ValueError):
        liquid_step(np.zeros(2), None, lambda x, I: np.array([0.0, 2.0]), 1.0, np.zeros(2), 0.1, M=1.0)


def test_L20prime_dist_to_box_across_events():
    # kicks of size <= dmax at random events: dist(x_n, B) <= dmax * tau / h
    rng = np.random.default_rng(1)
    tau, M, dmax = 0.8, 1.5, 0.3
    h = max_step(tau, M)
    A = rng.uniform(-1, 2, size=8)
    lo, hi = invariant_box(A)
    x = rng.uniform(lo, hi, size=8)
    worst = 0.0
    for n in range(20_000):
        x = liquid_step(x, None, lambda x_, I_: M * rng.uniform(size=8), tau, A, h, M=M)
        if rng.uniform() < 0.3:  # an event: adversarial push outward by dmax
            x = x + dmax * np.where(x > (lo + hi) / 2, 1.0, -1.0)
        worst = max(worst, box_distance(x, lo, hi))
    assert worst <= dmax * tau / h + 1e-12

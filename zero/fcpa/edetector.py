"""H3: mixture-over-change-time e-detector birth trigger (Shin-Ramdas-Rinaldo e-detectors).

S_t = (S_{t-1} + w_t) e_t,  w_t = 1/(t(t+1)),  e_t = kappa p_t^(kappa-1),  M_t = S_t + 1/(t+1).
E[e_t | past] = 1 under a super-uniform p, so M is a test martingale with M_0 = 1: Ville gives
P(sup_t M_t >= 1/delta) <= delta with no floor and no restart.  A kappa-grid averages the martingales
(still a test martingale).
"""
import numpy as np


def step(S, t, p, kappa):
    """One recursion step (t >= 1); works elementwise on arrays.  p = 0 gives e = inf (fires)."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return (S + 1.0 / (t * (t + 1.0))) * (kappa * np.power(p, kappa - 1.0))


def _check(delta, kappas):
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if not kappas or any(not 0 < k < 1 for k in kappas):
        raise ValueError("every kappa must be in (0, 1)")


class MixtureBirthTrigger:
    """Streaming trigger.  update(p) is True on the step M first reaches 1/delta; latched until reset()."""

    def __init__(self, delta=0.05, kappas=(0.5,)):
        _check(delta, kappas)
        self.delta, self.kappas = delta, tuple(kappas)
        self.reset()

    def reset(self, delta=None):
        if delta is not None:
            _check(delta, self.kappas)
            self.delta = delta
        self.t, self.fired, self.S = 0, False, [0.0] * len(self.kappas)

    @property
    def M(self):
        return sum(self.S) / len(self.S) + 1.0 / (self.t + 1.0)

    def update(self, p):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.t += 1
        self.S = [float(step(s, self.t, p, k)) for s, k in zip(self.S, self.kappas)]
        if self.fired or not self.M >= 1.0 / self.delta:
            return False
        self.fired = True
        return True


def first_birth(P, delta=0.05, kappas=(0.5,)):
    """First firing index of `MixtureBirthTrigger(delta, kappas)` for each row of P (runs, T); -1 if none."""
    P = np.asarray(P, dtype=float)
    kappas = tuple(kappas)
    _check(delta, kappas)
    n, T = P.shape
    S = np.zeros((len(kappas), n))
    first = np.full(n, -1)
    k = np.array(kappas)[:, None]
    for i in range(T):
        S = step(S, i + 1, P[:, i][None, :], k)
        hit = (first < 0) & (S.mean(0) + 1.0 / (i + 2.0) >= 1.0 / delta)
        first[hit] = i
    return first

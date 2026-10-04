import math

import torch

from qflm.generate import NULL
from qflm.stability import butterfly_apply, renorm_psi
from qflm.zeno_probe import repeat_escape, summary

N, V = 16, 8


def rand_angles(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(int(math.log2(N)), N // 2, generator=g) * 2 * math.pi


class RefQF:
    """Reference QF stepper: M is N x (V+1), last column = null; think = rotation by theta_scale*think_angles."""

    def __init__(self, M, q_angles, think_angles, theta_scale):
        self.sM, self.Q, self.T, self.th = M.sqrt(), q_angles, think_angles, theta_scale

    def reset(self, prompt_ids):
        self.psi = torch.ones(N, dtype=torch.complex64) / math.sqrt(N)

    def dist(self):
        q = butterfly_apply(self.psi, self.Q)
        p = (self.sM ** 2 * (q.abs() ** 2)[:, None]).sum(0)
        return p[:V], float(p[V])

    def observe(self, o):
        v = V if o == NULL else o
        self.psi = renorm_psi(self.sM[:, v] * butterfly_apply(self.psi, self.Q))
        if o == NULL:
            self.psi = butterfly_apply(self.psi, self.th * self.T)


def sharp_M(eps=0.005):
    M = torch.full((N, V + 1), eps / (V + 1))
    M[torch.arange(N), torch.arange(N) % V] += 1 - eps
    return M


def stepper(theta):
    return RefQF(sharp_M(), torch.zeros(4, N // 2), rand_angles(1), theta)


def test_zeno_trap():
    r = repeat_escape(stepper(0.0), [0], max_think=4)
    assert len(r["p_repeat"]) == 5 and len(r["p_null"]) == 5
    assert min(r["p_repeat"]) > 0.99 and r["escape"] < 0.01


def test_think_rotation_escapes():
    assert repeat_escape(stepper(math.pi / 4), [0])["escape"] > 0.3


def test_escape_grows_with_angle():
    lo, hi = repeat_escape(stepper(0.005), [0])["escape"], repeat_escape(stepper(0.5), [0])["escape"]
    assert lo < 0.05 and lo < hi


def test_probabilities_valid_and_summary():
    r = repeat_escape(stepper(0.5), [0])
    assert all(0 <= x <= 1 for x in r["p_repeat"]) and all(0 <= x <= 1 for x in r["p_null"])
    assert all(a + b <= 1 + 1e-6 for a, b in zip(r["p_repeat"], r["p_null"]))
    s = summary(stepper(0.5), [[0], [1]])
    assert len(s) == 5 and abs(s[0] - r["p_repeat"][0]) < 1e-6

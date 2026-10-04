"""T6: the order-effect / QQ statistic must reproduce ref/identities.py §2 on its own generators
(same rng stream), and a context-free observer must score exactly 0."""
import numpy as np
import torch

import t6_contextuality as t6


# ---- identities §2 generators, returning joint tables P[a, b] (a = answer to A, b = answer to B) ----
def _rand_proj(rng, n):
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, _ = np.linalg.qr(z)
    v = q[:, : rng.integers(1, n)]
    return v @ v.conj().T


def _quantum(rng, n, mixed):
    if mixed:
        g = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); rho = g @ g.conj().T
    else:
        v = rng.normal(size=n) + 1j * rng.normal(size=n); rho = np.outer(v, v.conj())
    rho /= np.trace(rho).real
    A, B = _rand_proj(rng, n), _rand_proj(rng, n)
    Ao, Bo = (np.eye(n) - A, A), (np.eye(n) - B, B)
    p = lambda X, Y: np.trace(Y @ X @ rho @ X @ Y).real
    ab = np.array([[p(Ao[a], Bo[b]) for b in (0, 1)] for a in (0, 1)])
    ba = np.array([[p(Bo[b], Ao[a]) for b in (0, 1)] for a in (0, 1)])
    return ab, ba


def _classical(rng, m, disturb):
    pi = rng.dirichlet(np.ones(m)); R = {q: rng.uniform(size=m) for q in "AB"}
    T = {(q, a): rng.dirichlet(np.ones(m), size=m) if disturb else np.eye(m) for q in "AB" for a in (0, 1)}

    def p(q1, a1, q2, a2):
        r1 = R[q1] if a1 else 1 - R[q1]; r2 = R[q2] if a2 else 1 - R[q2]
        return float(((pi * r1) @ T[(q1, a1)]) @ r2)
    ab = np.array([[p("A", a, "B", b) for b in (0, 1)] for a in (0, 1)])
    ba = np.array([[p("B", b, "A", a) for b in (0, 1)] for a in (0, 1)])
    return ab, ba


def test_reproduces_identities_section2():
    rng = np.random.default_rng(0)  # §1 draws nothing from rng, so §2 starts at the seed
    want = [(0.0000, 0.0702), (0.0000, 0.0397), (0.0751, 0.0747), (0.0000, 0.0000)]
    gens = [lambda: _quantum(rng, 4, False), lambda: _quantum(rng, 4, True),
            lambda: _classical(rng, 4, True), lambda: _classical(rng, 4, False)]
    for i, (f, (qq_w, oe_w)) in enumerate(zip(gens, want)):
        r = np.array([t6.qq_order(*f()) for _ in range(1000)])
        assert round(np.abs(r[:, 0]).mean(), 4) == qq_w, i
        assert round(np.abs(r[:, 1]).mean(), 4) == oe_w, i
        if i in (0, 1, 3):
            assert np.abs(r[:, 0]).max() < 1e-14, i  # Luders / undisturbed: QQ = 0 to rounding


def test_bootstrap_ci_covers_truth_from_samples():
    rng = np.random.default_rng(1)
    ab, ba = _classical(rng, 4, True)
    true_qq, true_oe = t6.qq_order(ab, ba)
    n = 4000
    draw = lambda P: np.array(np.unravel_index(rng.choice(4, n, p=P.ravel()), (2, 2))).T
    s = t6.stats(draw(ab), draw(ba), n_boot=500, seed=0, paired=False)
    assert s["qq_ci"][0] <= true_qq <= s["qq_ci"][1]
    assert s["order_ci"][0] <= true_oe <= s["order_ci"][1]


class ContextFree(torch.nn.Module):
    """Observer whose weights depend only on the current frame: no order effect is possible."""
    def __init__(self, D, K=4):
        super().__init__(); self.lin = torch.nn.Linear(D, K)

    def forward(self, obs, act):
        return dict(w=self.lin(obs).softmax(-1))


class Sticky(torch.nn.Module):
    """Observer that keeps the chart chosen at the first frame of the sequence: maximal disturbance."""
    def __init__(self, D, K=4):
        super().__init__(); self.lin = torch.nn.Linear(D, K)

    def forward(self, obs, act):
        return dict(w=self.lin(obs[:, :1]).softmax(-1).expand(-1, obs.shape[1], -1))


def _segments(N=200, L=3, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda: torch.randn(N, L, D, generator=g)
    return mk(), mk(), mk() + 1.0  # B drawn off-centre so the two questions differ


def test_context_free_observer_scores_zero():
    torch.manual_seed(0)
    pre, A, B = _segments()
    arm = ContextFree(8)
    ab, ba = t6.answers(arm, pre, A, B)
    s = t6.stats(ab, ba, n_boot=200, seed=0)
    assert s["qq"] == 0 and s["order"] == 0


def test_sticky_observer_shows_order_effect():
    torch.manual_seed(0)
    pre, A, B = _segments(L=1)
    arm = Sticky(8)
    ab, ba = t6.answers(arm, pre[:, :0], A, B)  # empty prefix: first question decides the chart
    s = t6.stats(ab, ba, n_boot=200, seed=0)
    excl = lambda k: s[k + "_ci"][0] > 0 or s[k + "_ci"][1] < 0
    assert excl("order") or excl("qq") or excl("dA") or excl("dB")


def test_runs_on_observer_atlas():
    from atlas.observer_atlas import ObserverAtlas
    torch.manual_seed(0)
    arm = ObserverAtlas(D=8, A=2, K=4, m=2)
    pre, A, B = _segments(N=32)
    ab, ba = t6.answers(arm, pre, A, B)
    assert ab.shape == ba.shape == (32, 2) and set(np.unique(ab)) <= {0, 1}

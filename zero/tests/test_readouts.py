"""Arm-agnostic topology readouts: orientation holonomy from any arm's (w, per-chart z, zidx); Chern from
any arm that returns a complex psi. Synthetic Klein / torus atlases are fixtures, never model inputs."""
import numpy as np

from metrics import chart_transitions, orientation_holonomy


def sector_atlas(klein, n=6000, seed=0):
    """3 charts = u-sectors of width 2pi/3 (+ overlap). Chart k uses local (u - c_k, v); on Klein the
    sector that wraps through u = 0 sees v flipped on its far side, so one transition is orientation-reversing."""
    rng = np.random.default_rng(seed)
    u, v = rng.uniform(0, 2 * np.pi, n), rng.uniform(-0.6, 0.6, n)
    cen = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    d = (u[:, None] - cen[None] + np.pi) % (2 * np.pi) - np.pi               # signed offset to each centre
    inside = np.abs(d) < 0.55 * 2 * np.pi / 3 + 0.3
    vv = np.where(klein & (np.abs(u[:, None] - cen[None] - d) > 1), -v[:, None], v[:, None])
    z = np.stack([d, vv], -1)                                                 # [n, 3, 2]
    w = inside / inside.sum(1, keepdims=True)
    order = np.argsort(-w, 1)[:, :2]
    zidx = order
    zz = np.take_along_axis(z, order[..., None], 1)
    wtop = np.take_along_axis(w, order, 1)
    W = np.zeros_like(w); np.put_along_axis(W, order, wtop, 1)
    return u, W, zz, zidx, cen


def loop_weights(cen, u_path):
    d = (u_path[:, None] - cen[None] + np.pi) % (2 * np.pi) - np.pi
    return np.exp(-d ** 2 / 0.3)


def test_orientation_holonomy_klein_vs_torus():
    for klein, want in ((True, -1), (False, 1)):
        u, W, zz, zidx, cen = sector_atlas(klein)
        tr = chart_transitions(W, zz, zidx, min_pairs=20)
        assert len(tr) >= 3                                                    # every co-observed pair fitted
        a_loop = loop_weights(cen, np.linspace(0, 2 * np.pi, 200, endpoint=False))
        assert orientation_holonomy(tr, a_loop) == want
        assert orientation_holonomy(tr, loop_weights(cen, np.full(50, 0.4))) == 1   # stays in one chart


def test_unobserved_transition_gives_none():
    u, W, zz, zidx, cen = sector_atlas(True)
    tr = chart_transitions(W, zz, zidx, min_pairs=20)
    tr = {k: v for k, v in tr.items() if set(k) != {0, 2}}
    a_loop = loop_weights(cen, np.linspace(0, 2 * np.pi, 200, endpoint=False))
    assert orientation_holonomy(tr, a_loop) is None


def test_to7_score_constant_and_abstention_cannot_pass():
    """Smoke on an untrained PHASE-SPHERE: loop_orientation abstained (None) on exactly the Klein 'a' loops
    and reported acc = 1.0 over the rest; all-(+1) loop_berry scored 0.667. Score = balanced accuracy over
    classes with None counted wrong."""
    from readouts import to7_score
    cls = ["a", "b", "0"] * 4
    truth = [-1, 1, 1] * 4
    assert to7_score([1] * 12, truth, cls) == {"a": 0.0, "b": 1.0, "0": 1.0, "balanced": 2 / 3, "min": 0.0}
    abst = [None if c == "a" else 1 for c in cls]
    assert to7_score(abst, truth, cls)["min"] == 0.0
    assert to7_score(truth, truth, cls)["min"] == 1.0


def test_loop_readouts_report_to7_score():
    import inspect
    import readouts
    for fn in (readouts.loop_orientation, readouts.loop_berry):
        assert "to7_score" in inspect.getsource(fn), fn.__name__


def test_chern_readout_vertex_order_and_prefix_free():
    """Kutner flag: _chern_psi fed all icosphere vertices as ONE sequence, so a recurrent arm's psi at a vertex
    depended on earlier vertices. Each vertex must be its own length-1 sequence; permuting vertices is a no-op."""
    import torch
    from readouts import chern_of_arm, icosphere

    class Recurrent(torch.nn.Module):
        A, seen = 3, []

        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.W = torch.nn.Parameter(torch.randn(256, 2, generator=g, dtype=torch.cfloat))

        def forward(self, obs, act):
            Recurrent.seen.append(obs.shape[1])
            x = obs.to(torch.cfloat) @ self.W
            x = x + 0.9 * torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)   # depends on the previous frame
            psi = x / x.abs().pow(2).sum(-1, keepdim=True).sqrt()
            return dict(g=psi.real, psi=psi)

    arm = Recurrent()
    c = chern_of_arm(arm, "s2")["chern"]
    assert max(Recurrent.seen) == 1, Recurrent.seen                            # one vertex per sequence
    assert abs(c - round(c)) < 1e-6

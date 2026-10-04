"""Contracts for zero/sigreg.py (shared anti-collapse term) and the one-lambda scale contract over every arm."""
import time

import pytest
import torch

from sigreg import LAM, LAM_BAL, sigreg


def test_gaussian_beats_constant_and_degenerate_is_zero():
    torch.manual_seed(0)
    assert sigreg(torch.randn(4000, 4)) < 0.01 * sigreg(torch.zeros(4000, 4))
    assert sigreg(torch.zeros(1, 4)) == 0
    assert sigreg(torch.randn(5, 3), torch.tensor([0, 1, 2, 3, 4]), 5) == 0  # every group < 2 rows


def test_grouped_equals_mean_of_groups_with_one_shared_slice_draw():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(600, 3, generator=g, dtype=torch.float64)
    grp = torch.randint(0, 4, (600,), generator=g)
    grp[0] = 4  # a single-row group is skipped
    torch.manual_seed(1)
    s = sigreg(x, grp, 6)
    ref = []
    for k in range(4):
        torch.manual_seed(1)
        ref.append(sigreg(x[grp == k]))
    assert torch.allclose(s, torch.stack(ref).mean(), atol=1e-12)


def test_group_relabelling_invariant():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, 2, generator=g, dtype=torch.float64) * torch.rand(500, 1, generator=g, dtype=torch.float64)
    grp = torch.randint(0, 7, (500,), generator=g)
    perm = torch.randperm(7, generator=g)
    torch.manual_seed(3)
    a = sigreg(x, grp, 7)
    torch.manual_seed(3)
    b = sigreg(x, perm[grp], 7)
    assert torch.allclose(a, b, atol=1e-12)


def test_cost_independent_of_group_count():
    x, t = torch.randn(4000, 2), {}
    for G in (2, 2000):
        grp = torch.arange(4000) % G
        sigreg(x, grp, G)  # warm-up
        s = time.perf_counter()
        for _ in range(3):
            sigreg(x, grp, G)
        t[G] = time.perf_counter() - s
    assert t[2000] < 5 * t[2], t  # a Python loop over groups costs ~1000x here


def test_gradcheck_and_closed_gate_grad_finite():
    x = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    grp = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3])

    def f(x):
        torch.manual_seed(0)
        return sigreg(x, grp, 4, num_slices=4)

    assert torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-5)
    z = torch.zeros(10, 3, requires_grad=True)
    sigreg(z).backward()
    assert torch.isfinite(z.grad).all()


def test_restoring_force_on_expansion():
    torch.manual_seed(0)
    base = torch.randn(2000, 6)
    g = {}
    for s in (0.3, 15.0):
        x = (s * base).requires_grad_()
        torch.manual_seed(1)
        sigreg(x).backward()
        g[s] = (x.grad * base).sum().item()  # d sigreg / d scale
    assert g[0.3] < 0 < g[15.0]  # pushes back toward scale 1 from both sides
    # EP-only was +0.0076 here; 0.0658 = 0.1 * the EP-only contraction force (-0.658) at s=0.3.
    # (Relative to the anchored g[0.3] = -3.87 the bound would be 0.39, unreachable at ANCHOR=0.1: the anchor's
    # own force is 0.4*log(s^2)/s, i.e. 0.144 at s=15 vs -3.2 at s=0.3.)
    assert g[15.0] > 0.0658, g


def test_gaussian_still_minimal():
    def v(scale):
        torch.manual_seed(0)
        x = torch.randn(4000, 4) * scale
        torch.manual_seed(1)
        return sigreg(x)
    assert v(1.0) < v(1.5) and v(1.0) < v(0.67)


def test_closed_gate_zero_input_finite():
    z = torch.zeros(10, 3, requires_grad=True)
    v = sigreg(z)
    v.backward()
    assert torch.isfinite(v) and torch.isfinite(z.grad).all()


def test_mean_drift_penalised():
    def v(shift):
        torch.manual_seed(0)
        x = torch.randn(2000, 4) + shift
        torch.manual_seed(1)
        return sigreg(x)
    assert v(5.0) > 10 * v(0.0)


# ---- one lambda for every arm: 200 smoke steps on the S2 bed, no shrinkage, no blow-up, rank >= n ----------
def _arms(D, A):
    from arms.chart_ae import ChartAEJEPA
    from arms.lazy_chart_ae import LazyChartAEJEPA
    from arms.mop_jepa import MoPJEPA
    from arms.peer_jepa import PEERJEPA
    from arms.single_jepa import SingleJEPA
    from atlas.observer_atlas import ObserverAtlas
    # (constructor, tensors the arm's SIGReg acts on)
    z_of = lambda key: lambda m, obs, act, out: [out["g"], key(m, obs, act)]
    return {
        "single": (lambda: SingleJEPA(D, A, d=6, hidden=64), lambda m, o, a, out: [out["g"]]),
        "mop": (lambda: MoPJEPA(D, A, d=6, K=4, hidden=64, head_hidden=32, router_hidden=32),
                lambda m, o, a, out: [m.enc(o)]),
        "chart_ae": (lambda: ChartAEJEPA(D, A, K=4, d_c=2, D_g=6, hidden=64), z_of(lambda m, o, a: m.E(o))),
        "lazy_chart_ae": (lambda: LazyChartAEJEPA(D, A, K=8, m=2, d_c=2, D_g=6, hidden=64),
                          z_of(lambda m, o, a: m.E(o))),
        "peer": (lambda: PEERJEPA(D, A, d=6, hidden=64, n_experts=64, heads=2, topk=4, d_key=16),
                 lambda m, o, a, out: [out["g"]]),
        "atlas": (lambda: ObserverAtlas(D, A, K=8, m=2, d_c=2, D_g=6), z_of(lambda m, o, a: m.observe(o, a)["z"])),
    }


def _erank(x):
    s = torch.linalg.svdvals((x - x.mean(0)).double())
    p = s[s > 0] / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


CHART_AE_RANK = pytest.mark.xfail(strict=True, reason=(
    "chart-AE erank(g) 1.57-2.0 < n=2 on 3/3 seeds at every lam in {0.75, 0.9, 0.95} (std passes): a "
    "learning failure of the sigmoid-chart design, not collapse; Cameron's arm, flagged. strict: a pass is news"))


@pytest.mark.parametrize("name", ["single", "mop", "chart_ae", "lazy_chart_ae",
                                  "peer", "atlas"])
def test_scale_contract_every_arm_same_lambda(name):
    import beds
    bed = beds.make_bed("s2", 256, 16, seed=0)
    obs, act, n = torch.from_numpy(bed["obs"]), torch.from_numpy(bed["act"]), bed["meta"]["n"]
    build, regularised = _arms(obs.shape[-1], act.shape[-1])[name]
    torch.manual_seed(0)
    model = build().train()
    assert getattr(model, "lam", LAM) == LAM  # one lambda table for every arm (fairness)
    if name in ("lazy_chart_ae", "atlas"):  # same Switch load balance (MoP's lam_bal is the paper's own term)
        assert model.lam_bal == LAM_BAL
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(200):
        i = torch.randint(0, 256, (32,))
        out = model(obs[i], act[i])
        assert torch.isfinite(out["loss"])
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(obs, act)
        xs = regularised(model, obs, act, out)
    for x in xs:
        s = x.reshape(-1, x.shape[-1]).std(0).mean()
        assert 0.3 < s < 3, (name, float(s))
    g = out["g"].reshape(-1, out["g"].shape[-1])
    assert _erank(g) >= n, (name, _erank(g))
    if out.get("w") is not None and name in ("lazy_chart_ae", "atlas"):
        K = out["w"].shape[-1]
        assert (out["w"].reshape(-1, K).gt(0).float().mean(0) > 0.01).sum() >= K - 2, name  # LAM_BAL guards

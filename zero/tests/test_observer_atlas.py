"""Contracts for OBSERVER-ATLAS (PROMPT.md §4). CPU, float64 where exactness matters."""
import inspect
import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import atlas.observer_atlas as oa  # noqa: E402
from atlas.observer_atlas import ObserverAtlas, erank, nerve  # noqa: E402

D, A, B, T = 12, 2, 3, 5


def make(K=8, m=2, seed=0, **kw):
    torch.manual_seed(seed)
    return ObserverAtlas(D=D, A=A, K=K, m=m, d_c=2, D_g=6, d_h=16, d_key=8, hidden=8, **kw).double().eval()


def data(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g, dtype=torch.float64), torch.randn(B, T, A, generator=g, dtype=torch.float64)


def run(model, obs, act, mode="lazy", seed=7):
    torch.manual_seed(seed)  # SIGReg draws random projections
    return model(obs, act, mode=mode)


# ---- contract shapes / L-GIVEN -------------------------------------------------------------
def test_contract_shapes_and_no_state_input():
    model = make()
    obs, act = data()
    out = run(model, obs, act)
    assert out["g"].shape == (B, T, 6) and out["ghat"].shape == (B, T - 1, 6)
    assert out["w"].shape == (B, T, 8) and out["loss"].dim() == 0
    assert torch.allclose(out["w"].sum(-1), torch.ones(B, T, dtype=torch.float64))
    assert ((out["w"] > 0).sum(-1) == 2).all()
    assert set(inspect.signature(ObserverAtlas.forward).parameters) == {"self", "obs", "act", "mode"}
    assert set(inspect.signature(ObserverAtlas.rollout).parameters) == {"self", "obs0", "act"}
    assert model.rollout(obs[:, 0], act[:, :4]).shape == (B, 4, 6)
    assert isinstance(model.active_flops(), int) and model.n_params() > 0


# ---- lazy == soft mixture restricted to top-m ---------------------------------------------
@pytest.mark.parametrize("K,pk", [(8, False), (16, True)])
def test_lazy_matches_dense_topm_exactly(K, pk):
    model = make(K=K, product_keys=pk)
    obs, act = data()
    lazy, dense = run(model, obs, act, "lazy"), run(model, obs, act, "dense")
    for k in ("g", "ghat", "w", "loss"):
        assert torch.allclose(lazy[k], dense[k], atol=1e-10), k


def test_product_keys_topm_equals_exhaustive_topm():
    model = make(K=16, m=3, product_keys=True)
    obs, act = data()
    with torch.no_grad():
        c = model.backbone(obs)[0]
        logits, idx = model.route(c)
        full = model.full_logits(c)
    ref = full.topk(3, -1).indices.sort(-1).values
    assert torch.equal(idx.sort(-1).values, ref)


# ---- unselected charts execute zero FLOPs -------------------------------------------------
def _poison_unselected(model, used):
    with torch.no_grad():
        for mlp in (model.E, model.P, model.phi, model.E_tgt):
            for p in mlp.parameters():
                keep = torch.zeros(p.shape[0], dtype=torch.bool)
                keep[used] = True
                p[~keep] = float("nan")


def test_unselected_charts_never_touched():
    model = make(K=16)
    obs, act = data()
    ref = run(model, obs, act)
    used = ref["w"].reshape(-1, 16).gt(0).any(0)
    assert (~used).any(), "test needs some unselected chart"
    _poison_unselected(model, used)
    out = run(model, obs, act)
    for k in ("g", "ghat", "loss"):
        assert torch.isfinite(out[k]).all() and torch.allclose(out[k], ref[k], atol=1e-12), k
    assert torch.isfinite(model.rollout(obs[:, 0], act)).all()
    # control: the dense reference DOES touch them, so the poison detects K-proportional compute
    assert not torch.isfinite(run(model, obs, act, "dense")["g"]).all()


def test_active_flops_scale_with_m_not_K():
    f = lambda K, m: make(K=K, m=m).active_flops()
    router = lambda K: K * 8  # flat keys: K * d_key
    assert f(64, 2) - f(8, 2) == router(64) - router(8)
    per_chart = f(8, 2) - f(8, 1)
    assert per_chart > 0 and f(8, 3) - f(8, 2) == per_chart
    pk = make(K=256, m=2, product_keys=True).active_flops() - make(K=16, m=2, product_keys=True).active_flops()
    assert pk == (16 - 4) * 8  # 2 * sqrt(K) * d_key/2


def test_active_flops_match_flop_counter():
    from torch.utils.flop_counter import FlopCounterMode
    model = make(K=16)
    obs, act = data()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model.observe(obs, act)
    counted = fc.get_total_flops() / 2 / (B * T)  # FLOPs -> MACs per step
    assert abs(counted - model.active_flops()) / model.active_flops() < 0.05, (counted, model.active_flops())


# ---- gate: observer can observe nothing --------------------------------------------------
def test_gate_closed_outputs_nothing_without_nan():
    model = make()
    with torch.no_grad():
        model.gate.weight.zero_()
        model.gate.bias.fill_(-1e4)
    obs, act = data()
    obs.requires_grad_(True)
    out = run(model, obs, act)
    assert out["g"].abs().max() == 0 and out["ghat"].abs().max() == 0
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert torch.isfinite(obs.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


# ---- permutation of chart labels ---------------------------------------------------------
def test_permutation_equivariance():
    model = make(K=8)  # needs sigreg's single shared slice draw (per-group draws were label-dependent)
    obs, act = data()
    ref = run(model, obs, act)
    perm = torch.randperm(8, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        model.keys.copy_(model.keys[perm])
        for mlp in (model.E, model.P, model.phi, model.E_tgt):
            for p in mlp.parameters():
                p.copy_(p[perm])
    out = run(model, obs, act)
    for k in ("g", "ghat", "loss"):
        assert torch.allclose(out[k], ref[k], atol=1e-10), k
    assert torch.allclose(out["w"], ref["w"][..., perm], atol=1e-12)


def test_determinism_under_seed():
    obs, act = data()
    a, b = run(make(seed=5), obs, act), run(make(seed=5), obs, act)
    assert all(torch.equal(a[k], b[k]) for k in ("g", "ghat", "w", "loss"))


# ---- gradients ---------------------------------------------------------------------------
def test_gradcheck_tiny():
    model = make(K=4, m=2)
    obs, act = data()
    obs = obs[:2, :3].clone().requires_grad_(True)
    act = act[:2, :3].clone().requires_grad_(True)

    # loss itself is not gradcheck-able: its stop-grad targets move under finite differences by design
    def f(o, a):
        out = model(o, a)
        return out["g"], out["ghat"]

    assert torch.autograd.gradcheck(f, (obs, act), eps=1e-6, atol=1e-5)


def test_grad_reaches_selected_charts_only_and_all_keys():
    model = make(K=16).train()
    model.ema = 1.0  # freeze target so train-mode forward is pure
    obs, act = data()
    out = run(model, obs, act)
    out["loss"].backward()
    used = out["w"].reshape(-1, 16).gt(0).any(0)
    g = model.E.W1.grad.abs().flatten(1).sum(1)
    assert (g[used] > 0).all() and (g[~used] == 0).all()
    assert (model.keys.grad.abs().sum(1) > 0).all(), "load-balance must reach unselected keys (router-death guard)"


def test_rollout_first_step_equals_ghat():
    model = make()
    obs, act = data()
    out = run(model, obs, act)
    r = model.rollout(obs[:, 0], act[:, :3])
    assert torch.allclose(r[:, 0], out["ghat"][:, 0], atol=1e-12)


def test_causality():
    model = make()
    obs, act = data()
    ref = run(model, obs, act)
    obs2 = obs.clone()
    obs2[:, 3:] += 5.0
    out = run(model, obs2, act)
    assert torch.allclose(out["g"][:, :3], ref["g"][:, :3]) and torch.allclose(out["w"][:, :3], ref["w"][:, :3])
    assert torch.allclose(out["ghat"][:, :3], ref["ghat"][:, :3])  # ghat_{t+1} uses only o_<=t, a_<=t


def test_time_traveller_is_explicit_stub():
    with pytest.raises(NotImplementedError):
        make(time_travel=True)


# ---- regulariser / helpers ---------------------------------------------------------------
def test_nerve_closed_under_faces():
    w = torch.tensor([[0.6, 0.4, 0.0, 0.0], [0.0, 0.5, 0.3, 0.2], [0.0, 0.0, 0.0, 1.0]])
    N = nerve(w, 0.1)
    assert frozenset({1, 2, 3}) in N and frozenset({0, 1}) in N
    assert all(frozenset(s) in N for s in [{1, 2}, {2, 3}, {1, 3}, {0}, {1}, {2}, {3}])
    assert frozenset({0, 2}) not in N and frozenset() not in N
    assert nerve(w, 0.35) == {frozenset({0, 1}), frozenset({0}), frozenset({1}), frozenset({3})}


def test_w_soft_is_pre_topm_softmax_and_lifts_nerve_dimension():
    model = make(K=8, m=2)
    obs, act = data()
    out = run(model, obs, act)
    ws, w = out["w_soft"], out["w"]
    assert ws.shape == (B, T, 8) and torch.allclose(ws.sum(-1), torch.ones(B, T, dtype=torch.float64))
    restricted = torch.where(w > 0, ws, torch.zeros_like(ws))
    assert torch.allclose(w, restricted / restricted.sum(-1, keepdim=True), atol=1e-12)
    # top-m nerve is a graph (dim <= m-1); the soft nerve is not capped by m
    assert max(len(s) for s in nerve(w, 1e-3)) <= 2
    assert max(len(s) for s in nerve(ws, 1e-3)) >= 3
    big = nerve(torch.full((1, 8), 1 / 8), 0.1, max_dim=2)
    assert max(len(s) for s in big) == 3 and len(big) == 8 + 28 + 56


def test_nerve_integrates_with_metrics_betti():
    metrics = pytest.importorskip("metrics")
    tet = torch.tensor([[1 / 3 if v != k else 0.0 for v in range(4)] for k in range(4)])  # 4 facets of dS^3
    assert metrics.betti_z2({tuple(sorted(s)) for s in nerve(tet, 0.2)}, 2) == (1, 0, 1)
    model = make(K=8, m=2)
    obs, act = data()
    ws = run(model, obs, act)["w_soft"].detach().numpy()
    mine = metrics.betti_z2({tuple(sorted(s)) for s in nerve(torch.from_numpy(ws), 0.1, max_dim=3)}, 2)
    assert mine == metrics.nerve_betti_z2(ws, 0.1, max_dim=2)


def test_erank():
    assert erank(torch.zeros(10, 3)) == 0
    assert math.isclose(erank(torch.randn(5000, 3, generator=torch.Generator().manual_seed(0))), 3, rel_tol=0.05)


# ---- smoke: noisy circle lifted to D=64 --------------------------------------------------
def circle_bed(n, t, seed):
    g = torch.Generator().manual_seed(seed)
    act = 0.3 * torch.randn(n, t, 1, generator=g)
    th = torch.rand(n, 1, generator=g) * 2 * math.pi + torch.cumsum(act, 1)[..., 0] - act[..., 0]
    x = torch.stack([th.cos(), th.sin()], -1) + 0.02 * torch.randn(n, t, 2, generator=g)
    W1, W2 = torch.randn(2, 32, generator=g), torch.randn(32, 64, generator=g) / 32 ** 0.5
    return torch.tanh(torch.tanh(x @ W1) @ W2), act  # state (th) never leaves this function


def test_smoke_circle_trains_without_collapse():
    torch.manual_seed(0)
    model = ObserverAtlas(D=64, A=1, K=8, m=2, d_c=2, D_g=8).train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    obs, act = circle_bed(256, 16, 0)
    losses = []
    for step in range(200):
        i = torch.randint(0, 256, (32,))
        out = model(obs[i], act[i])
        assert torch.isfinite(out["loss"])
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())
    assert sum(losses[-20:]) < sum(losses[:20])
    model.eval()
    with torch.no_grad():
        out = model(obs, act)
    usage = out["w"].reshape(-1, 8).gt(0).float().mean(0)
    assert erank(out["g"].reshape(-1, 8)) > 1.0
    # loss falling + erank > 1 is passable by shrinkage collapse (seen: g std 0.017); demand O(1) scale
    z = model.observe(obs, act)["z"]
    assert 0.3 < out["g"].std() < 3 and 0.3 < z.std() < 3, (out["g"].std(), z.std())
    assert (usage > 0.01).sum() >= 6, usage  # router death guard: >= 6 of 8 charts alive


# ---- streaming path (GRUCell, same weights) and pinned numerics ----------------------------
def test_streaming_step_matches_sequence_and_skips_nn_gru():
    model = make()
    obs, act = data()
    ref = model.observe(obs, act)
    calls = []
    hook = model.backbone.register_forward_hook(lambda *a: calls.append(1))
    h, g, gh, w = None, [], [], []
    for t in range(T):
        out, h = model.step(obs[:, t], act[:, t], h)
        g.append(out["g"]), gh.append(out["ghat"]), w.append(out["w"])
    model.rollout(obs[:, 0], act)
    hook.remove()
    assert not calls, "streaming must use the GRUCell, not nn.GRU"
    rs = lambda x: x.reshape(B, T, -1)
    assert torch.allclose(torch.stack(g, 1), rs(ref["g"]), atol=1e-12)
    assert torch.allclose(torch.stack(gh, 1), rs(ref["gh"]), atol=1e-12)
    assert torch.allclose(torch.stack(w, 1), rs(ref["w"]), atol=1e-12)
    assert model.cell.weight_ih is model.backbone.weight_ih_l0  # one set of weights
    assert model.n_params() == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_tf32_pinned_off():
    assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False


# ---- chart-diversity term lam_div (paper L_Div, diagonal rho; rewarded, default off) ---------
def test_div_off_by_default_changes_nothing():
    obs, act = data()
    base = make()
    out = run(base, obs, act)
    assert base.lam_div == 0 and "div" not in out["terms"]
    assert torch.equal(out["loss"], run(make(lam_div=0.0), obs, act)["loss"])
    on = run(make(lam_div=0.5), obs, act)
    expect = out["loss"] - (1 - base.lam) * 0.5 * on["terms"]["div"]
    assert torch.allclose(on["loss"], expect, atol=1e-10)
    assert abs(on["loss"].item() - out["loss"].item()) > 1e-8  # the term is not vacuous


def test_div_matches_bruteforce_kl():
    torch.manual_seed(3)
    w = torch.randn(50, 5, dtype=torch.float64).softmax(-1)
    rho = [[(w[x, k].item() + 1e-6) for x in range(50)] for k in range(5)]
    rho = [[v / sum(r) for v in r] for r in rho]
    tot, n = 0.0, 0
    for i in range(5):
        for j in range(i + 1, 5):
            tot += sum(rho[i][x] * (math.log(rho[i][x]) - math.log(rho[j][x])) for x in range(50))
            n += 1
    assert abs(oa.chart_diversity(w).item() - tot / n) < 1e-10


def test_div_zero_for_duplicate_charts_and_finite_for_onehot():
    col = torch.rand(20, 1, dtype=torch.float64).softmax(0)
    assert abs(oa.chart_diversity(col.expand(20, 4)).item()) < 1e-12
    logits = (torch.eye(4, dtype=torch.float64)[torch.arange(12) % 4] * 50).requires_grad_()
    d = oa.chart_diversity(logits.softmax(-1))
    assert torch.isfinite(d) and d.item() > 0
    d.backward()
    assert torch.isfinite(logits.grad).all()
    assert oa.chart_diversity(torch.ones(7, 1, dtype=torch.float64)).item() == 0  # K == 1


def test_div_rewards_separation():
    same = torch.full((20, 2), 0.5, dtype=torch.float64)
    disjoint = torch.zeros(20, 2, dtype=torch.float64)
    disjoint[:10, 0] = 1.0
    disjoint[10:, 1] = 1.0
    assert oa.chart_diversity(disjoint).item() > oa.chart_diversity(same).item()


def test_div_off_costs_nothing(monkeypatch):
    from torch.utils.flop_counter import FlopCounterMode
    obs, act = data()

    def boom(*a, **k):
        raise AssertionError("chart_diversity called with lam_div=0")
    with monkeypatch.context() as mp:
        mp.setattr(oa, "chart_diversity", boom)
        run(make(lam_div=0.0), obs, act)

    def flops(model):
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            run(model, obs, act)
        return fc.get_total_flops()
    K = 64
    assert flops(make(K=K, lam_div=0.1)) - flops(make(K=K, lam_div=0.0)) >= 2 * B * T * K * K


# ---- router z-loss (ST-MoE, arXiv 2202.08906), default off ---------------------------------
def test_zloss_off_by_default_changes_nothing():
    obs, act = data()
    base = make()
    out = run(base, obs, act)
    assert base.lam_z == 0 and "z" not in out["terms"]
    assert torch.equal(out["loss"], run(make(lam_z=0.0), obs, act)["loss"])
    on = run(make(lam_z=1e-2), obs, act)
    assert "z" in on["terms"]
    assert torch.allclose(on["loss"], out["loss"] + (1 - base.lam) * 1e-2 * on["terms"]["z"], atol=1e-10)


def _zloss_final_lse(lam_z, steps=50, lr=1e-3):
    obs, act = data()
    model = make(K=8, lam_z=lam_z).train()
    model.ema = 1.0
    with torch.no_grad():
        model.keys.mul_(100)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        run(model, obs, act)["loss"].backward()
        opt.step()
    with torch.no_grad():
        c = model.backbone(obs)[0].reshape(B * T, -1)
        return torch.logsumexp(model.full_logits(c), -1).abs().mean().item()


def test_zloss_bounds_logits():
    with_z, without = _zloss_final_lse(1e-2), _zloss_final_lse(0.0)
    print(f"zloss arms: with={with_z:.6f} without={without:.6f} ratio={with_z / without:.4f}")
    assert with_z < 0.8 * without

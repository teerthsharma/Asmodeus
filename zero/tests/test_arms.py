"""Contracts for the owned arms (Cameron): single JEPA, MoP-JEPA, chart-AE JEPA, PEER-JEPA, CEM planner, matcher."""
import copy
import inspect
import math
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from arms import macs  # noqa: E402
from arms.chart_ae import ChartAEJEPA  # noqa: E402
from arms.lazy_chart_ae import LazyChartAEJEPA  # noqa: E402
from arms.match import match  # noqa: E402
from arms.mop_jepa import MoPJEPA  # noqa: E402
from arms.peer_jepa import PEERJEPA  # noqa: E402
from arms.planner import cem_plan  # noqa: E402
from arms.single_jepa import SingleJEPA  # noqa: E402

D, A, B, T = 32, 1, 3, 5

# hand-sized configs and their hand-counted multiply-adds per step (Linear in*out; PEER per its formula)
CFG = {
    # enc 32*16+16*16+16*2=800, pred 3*16+16*16+16*2=336
    "single": (lambda: SingleJEPA(D, A, d=2, hidden=16), 2, 800 + 336),
    # enc 32*16+256+16*4=832, router 5*8+8*3=64, head 5*8+8*4=72 (x3 heads)
    "mop": (lambda: MoPJEPA(D, A, d=4, K=3, hidden=16, head_hidden=8, router_hidden=8), 4, 832 + 64 + 3 * 72),
    "mop_lazy": (lambda: MoPJEPA(D, A, d=4, K=3, hidden=16, head_hidden=8, router_hidden=8, lazy=True), 4,
                 832 + 64 + 72),
    # E 32*16+256+16*6=864, member 6*8+8*3=72, per chart: Ec 6*8+8*2=64, Pc 3*8+8*2=40, Dc twice 2*(2*8+8*6)=128
    "chart_ae": (lambda: ChartAEJEPA(D, A, K=3, d_c=2, D_g=6, hidden=16, chart_hidden=8), 6, 864 + 72 + 3 * 232),
    # chart_ae counts with only m=2 of K=3 charts computing: 864 + 72 + 2*232
    "lazy_chart_ae": (lambda: LazyChartAEJEPA(D, A, K=3, m=2, d_c=2, D_g=6, hidden=16, chart_hidden=8), 6,
                      864 + 72 + 2 * 232),
    # enc: 32*16 + PEER(q 16*8 + subkeys 2*4*4 + experts 2*2*2*16) + 16*2 = 512+288+32; pred: 3*16+288+32
    "peer": (lambda: PEERJEPA(D, A, d=2, hidden=16, n_experts=16, heads=2, topk=2, d_key=4), 2, 832 + 368),
}
ARMS = list(CFG)


def make(name, seed=0):
    torch.manual_seed(seed)
    return CFG[name][0]()


def data(seed=1, b=B, t=T):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, D, generator=g), torch.randn(b, t, A, generator=g)


def circle_bed(n, t, seed):
    g = torch.Generator().manual_seed(seed)
    act = 0.3 * torch.randn(n, t, A, generator=g)
    th = torch.rand(n, 1, generator=g) * 2 * math.pi + torch.cumsum(act, 1)[..., 0] - act[..., 0]
    x = torch.stack([th.cos(), th.sin()], -1)
    W1, W2 = torch.randn(2, 32, generator=g), torch.randn(32, D, generator=g) / 32 ** 0.5
    return torch.tanh(torch.tanh(x @ W1) @ W2), act  # state (th) never leaves this function


def erank(x):
    s = torch.linalg.svdvals(x - x.mean(0))
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


# ---- contract ------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ARMS)
def test_contract_shapes_and_no_state_input(name):
    m, Dg = make(name).eval(), CFG[name][1]
    obs, act = data()
    out = m(obs, act)
    assert out["loss"].dim() == 0 and torch.isfinite(out["loss"])
    assert out["g"].shape == (B, T, Dg) and out["ghat"].shape == (B, T - 1, Dg)
    if out["w"] is not None:
        assert out["w"].shape[:2] == (B, T) and torch.allclose(out["w"].sum(-1), torch.ones(B, T))
    assert set(inspect.signature(type(m).forward).parameters) == {"self", "obs", "act"}  # no state input
    assert set(inspect.signature(type(m).rollout).parameters) == {"self", "obs0", "act"}
    assert m.rollout(obs[:, 0], act[:, :4]).shape == (B, 4, Dg)
    assert isinstance(m.active_flops(), int) and isinstance(m.n_params(), int) and m.n_params() > 0


@pytest.mark.parametrize("name", ARMS)
def test_rollout_first_step_equals_ghat(name):
    m = make(name).eval()
    obs, act = data()
    with torch.no_grad():
        out, r = m(obs, act), m.rollout(obs[:, 0], act[:, :3])
    assert torch.allclose(r[:, 0], out["ghat"][:, 0], atol=1e-5)


@pytest.mark.parametrize("name", ARMS)
def test_causality(name):
    m = make(name).eval()
    obs, act = data()
    with torch.no_grad():
        ref = m(obs, act)
        obs2 = obs.clone()
        obs2[:, 3:] += 5.0
        out = m(obs2, act)
    for k in ("g", "ghat", "w"):
        if ref[k] is not None:
            assert torch.allclose(out[k][:, :3], ref[k][:, :3], atol=1e-6), k


# ---- EMA target ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", ARMS)
def test_ema_target_gets_no_grad_and_tracks_online(name):
    m = make(name).train()
    tgt = [(n, p) for n, p in m.named_parameters() if n.startswith("tgt_")]
    assert tgt, "arm must have an EMA target"
    assert all(not p.requires_grad for _, p in tgt)
    obs, act = data()
    m(obs, act)["loss"].backward()
    assert all(p.grad is None for _, p in tgt)
    torch.optim.SGD(m.parameters(), lr=0.1).step()
    before = {n: p.detach().clone() for n, p in tgt}
    online = dict(m.named_parameters())
    m(obs, act)  # train-mode forward applies one EMA step before computing targets
    for n, p in tgt:
        o = online[n[4:]]
        assert torch.allclose(p, before[n] + (1 - m.tau) * (o - before[n]), atol=1e-6), n
    assert m.n_params() == sum(p.numel() for n, p in m.named_parameters() if not n.startswith("tgt_"))
    m.eval()  # the target must actually be the regression target: perturbing it moves the loss
    torch.manual_seed(0)
    ref = m(obs, act)["loss"]
    with torch.no_grad():
        for _, p in tgt:
            p.add_(torch.randn_like(p))
    torch.manual_seed(0)
    assert not torch.allclose(m(obs, act)["loss"], ref)


@pytest.mark.parametrize("name", ARMS)
def test_router_arms_expose_soft_weights(name):
    out = make(name).eval()(*data())
    assert "w_soft" in out and (out["w_soft"] is None) == (out["w"] is None)
    if out["w"] is not None:  # w = w_soft restricted to the selected charts, renormalised
        ws, w = out["w_soft"], out["w"]
        assert torch.allclose(ws.sum(-1), torch.ones(B, T), atol=1e-6)
        r = torch.where(w > 0, ws, torch.zeros_like(ws))
        assert torch.allclose(w, r / r.sum(-1, keepdim=True), atol=1e-6)


ROUTER_ARMS = ["mop", "mop_lazy", "chart_ae", "lazy_chart_ae"]


@pytest.mark.parametrize("name", ROUTER_ARMS)
def test_nerve_betti_runs_on_router_weights(name):
    metrics = pytest.importorskip("metrics")
    with torch.no_grad():
        ws = make(name).eval()(*data())["w_soft"]
    b = metrics.nerve_betti_z2(ws.numpy(), 0.2)
    assert len(b) >= 1 and b[0] >= 1 and all(int(x) == x >= 0 for x in b)


# ---- lazy chart-AE: the observer ablation (no backbone, no gate; router = chart-AE membership) ----
def lazy(K=8, m=2, seed=0, dense=False):
    torch.manual_seed(seed)
    mdl = LazyChartAEJEPA(D, A, K=K, m=m, d_c=2, D_g=6, hidden=16, chart_hidden=8).double().eval()
    mdl.dense = dense
    return mdl


def ddata():
    o, a = data()
    return o.double(), a.double()


def test_lazy_chart_ae_topm_weights():
    out = lazy()(*ddata())
    assert ((out["w"] > 0).sum(-1) == 2).all()


def test_lazy_chart_ae_equals_dense_restricted_to_topm():
    obs, act = ddata()
    a, b = lazy(), lazy(dense=True)
    torch.manual_seed(7)
    oa = a(obs, act)
    torch.manual_seed(7)
    ob = b(obs, act)
    for k in ("g", "ghat", "w", "w_soft", "loss"):
        assert torch.allclose(oa[k], ob[k], atol=1e-10), k


def test_lazy_chart_ae_unselected_charts_never_touched():
    obs, act = ddata()
    m = lazy(K=16)
    torch.manual_seed(7)
    ref = m(obs, act)
    used = ref["w"].reshape(-1, 16).gt(0).any(0)
    assert (~used).any()
    with torch.no_grad():
        for h in (m.Ec, m.Pc, m.Dc, m.tgt_Ec):
            for p in h.parameters():
                p[~used] = float("nan")
    torch.manual_seed(7)
    out = m(obs, act)
    for k in ("g", "ghat", "loss"):
        assert torch.isfinite(out[k]).all() and torch.allclose(out[k], ref[k], atol=1e-12), k
    m.dense = True  # control: dense reference touches them, so the poison detects K-proportional compute
    assert not torch.isfinite(m(obs, act)["g"]).all()


def test_lazy_chart_ae_flops_scale_with_m_not_K():
    f = lambda K, m: lazy(K=K, m=m).active_flops()
    assert f(16, 2) - f(8, 2) == 8 * (16 - 8)  # only the membership output layer (chart_hidden * K) grows
    per = f(8, 2) - f(8, 1)
    assert per > 0 and f(8, 3) - f(8, 2) == per


def test_lazy_chart_ae_grad_selected_charts_only_and_balance_reaches_all():
    m = lazy(K=16).train()
    m.tau = 1.0
    out = m(*ddata())
    out["loss"].backward()
    used = out["w"].reshape(-1, 16).gt(0).any(0)
    g = m.Ec.W1.grad.abs().flatten(1).sum(1)
    assert (g[used] > 0).all() and (g[~used] == 0).all()
    assert (~used).any(), "test needs some unselected chart"
    gm = m.member[-1].weight.grad.abs().sum(1)
    assert (gm[~used] > 1e-3 * gm[used].mean()).all(), "load balance must reach unselected charts"


# ---- FLOPs ---------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ARMS)
def test_active_flops_hand_count(name):
    assert make(name).active_flops() == CFG[name][2]


@pytest.mark.parametrize("name", ARMS)
def test_active_flops_match_flop_counter(name):
    from torch.utils.flop_counter import FlopCounterMode
    m = make(name).eval()
    obs, act = data()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m.observe(obs, act)
    counted = fc.get_total_flops() / 2 / (B * T)
    assert abs(counted - m.active_flops()) / m.active_flops() < 0.05, (counted, m.active_flops())


def test_macs_counts_linear_only():
    assert macs(torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))) == 25


# ---- training smoke ------------------------------------------------------------------------
@pytest.mark.parametrize("name", ARMS)
def test_loss_decreases_and_no_collapse(name):
    torch.manual_seed(0)
    m = make(name).train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    obs, act = circle_bed(256, 8, 0)
    losses = []
    for _ in range(200):
        i = torch.randint(0, 256, (32,))
        out = m(obs[i], act[i])
        assert torch.isfinite(out["loss"])
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())
    assert sum(losses[-20:]) < sum(losses[:20]), (losses[:3], losses[-3:])
    m.eval()
    with torch.no_grad():
        g = m(obs, act)["g"]
    assert erank(g.reshape(-1, g.shape[-1])) > 1.0


# ---- MoP-JEPA fidelity (arXiv 2607.05238v1 §method) ----------------------------------------
def mop(**kw):
    torch.manual_seed(0)
    return MoPJEPA(D, A, d=4, K=4, hidden=16, head_hidden=8, router_hidden=8, **kw)


def test_mop_winner_is_argmin_cosine_distance_and_router_targets_it():
    m = mop().train()
    m.tau = 1.0  # freeze target: forward is pure
    obs, act = data()
    out = m(obs, act)
    with torch.no_grad():
        r = m.enc(obs)
        z = F.normalize(r, dim=-1)
        zp = F.normalize(m.tgt_enc(obs[:, 1:]), dim=-1)
        c = torch.cat([z[:, :-1], act[:, :-1]], -1)
        u = F.normalize(m.heads(c.reshape(-1, c.shape[-1])), dim=-1)  # [N, K, d]
        dist = 1 - (u * zp.reshape(-1, 1, 4)).sum(-1)
        k = dist.argmin(-1)
        logits = m.router(c.reshape(-1, c.shape[-1]))
    assert torch.equal(out["kstar"].reshape(-1), k)
    assert torch.allclose(out["pred"], dist.gather(1, k[:, None]).mean(), atol=1e-6)
    assert torch.allclose(out["route"], F.cross_entropy(logits, k), atol=1e-6)


def test_mop_hard_assignment_grad_only_to_winner():
    m = mop().train()
    obs, act = data(b=1, t=2)  # one transition -> exactly one winner
    out = m(obs, act)
    out["loss"].backward()
    k = int(out["kstar"].reshape(-1)[0])
    gnorm = m.heads.W1.grad.abs().flatten(1).sum(1)
    assert gnorm[k] > 0 and (gnorm[torch.arange(4) != k] == 0).all()


def test_mop_router_is_context_only():
    m = mop()
    assert m.router[0].in_features == 4 + A  # [z_t, a_t] only: never sees z_{t+1}
    assert "obs" not in inspect.signature(m.router.forward).parameters


def test_mop_lazy_equals_dense_and_is_cheaper():
    obs, act = data()
    a, b = mop().eval(), mop(lazy=True).eval()
    with torch.no_grad():
        oa, ob = a(obs, act), b(obs, act)
        ra, rb = a.rollout(obs[:, 0], act), b.rollout(obs[:, 0], act)
    for k in ("g", "ghat", "w"):
        assert torch.allclose(oa[k], ob[k], atol=1e-6), k
    assert torch.allclose(ra, rb, atol=1e-6)
    assert b.active_flops() < a.active_flops() and a.n_params() == b.n_params()


def test_mop_inference_reads_router_argmax_head():
    m = mop().eval()
    obs, act = data()
    with torch.no_grad():
        out = m(obs, act)
        z = out["g"]
        c = torch.cat([z[:, :-1], act[:, :-1]], -1).reshape(-1, 4 + A)
        u = F.normalize(m.heads(c), dim=-1)
        k = out["w"][:, :-1].reshape(-1, 4).argmax(-1)
    assert torch.allclose(out["ghat"].reshape(-1, 4), u[torch.arange(len(k)), k], atol=1e-6)


# ---- planner -------------------------------------------------------------------------------
class Integrator(torch.nn.Module):
    """g_{h+1} = g_h + a_h, 2-D: CEM must steer to the goal."""

    def rollout(self, obs0, act):
        return obs0[:, None] + act.cumsum(1)


def test_cem_reaches_goal_on_integrator():
    torch.manual_seed(0)
    obs0, goal = torch.zeros(2, 2), torch.tensor([[1.0, -0.5], [-0.8, 0.3]])
    a0, plan = cem_plan(Integrator(), obs0, goal, horizon=3, act_dim=2, n_samples=256, n_elites=16, iters=8)
    assert a0.shape == (2, 2) and plan.shape == (2, 3, 2)
    assert (plan.sum(1) - goal).abs().sum(-1).max() < 0.1


# ---- FLOP/param matching (L-BR) ------------------------------------------------------------
def test_match_peer_hits_both_targets():
    base = dict(D=256, A=2, d=4)
    r = match("peer", flops=200_000, params=2_000_000, **base)
    assert r["both"] is not None, r
    assert abs(r["both"]["flops"] / 200_000 - 1) <= 0.05 and abs(r["both"]["params"] / 2_000_000 - 1) <= 0.05


def test_match_single_reports_split_when_infeasible():
    r = match("single", flops=200_000, params=2_000_000, D=256, A=2, d=4)
    assert r["both"] is None and "note" in r
    assert abs(r["flop_matched"]["flops"] / 200_000 - 1) <= 0.05
    assert abs(r["param_matched"]["params"] / 2_000_000 - 1) <= 0.05

import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

torch.set_num_threads(2)

from qflm import model as qm
from qflm.model import QFLM


def make(N=8, V=6, r=4, seed=0, rho=None, s_max=50, arm="ctc"):
    m = QFLM(N, V, s_max=s_max, arm=arm, seed=seed, think_rank=r)
    if rho is not None:  # rho_raw is linear in ctc_W (rho_global is capped at RHO_MAX)
        with torch.no_grad():
            m.ctc_W.mul_(rho / m.rho_raw())
    return m


def simplex(n, V, g):
    return torch.distributions.Dirichlet(torch.full((V + 1,), 0.5)).sample((n,))


def toks(B=4, T=10, V=6, seed=1):
    return torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(seed))


def l1(a, b):
    return (a - b).abs().sum(-1)


def psi_of(m, B):
    return m.psi0().detach().expand(B, m.N)


def test_F_maps_simplex_to_simplex():
    m = make(seed=1, rho=2.0)
    p = simplex(64, m.V, None)
    with torch.no_grad():
        out = m.ctc_F(p, psi_of(m, 64))
    assert (out >= 0).all() and (out.sum(1) - 1).abs().max() < 1e-6


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_certificate_sound(seed):
    torch.manual_seed(seed)
    m = make(seed=seed, rho=[0.5, 1.5, 5.0][seed])
    rho = m.rho_global()
    psi = psi_of(m, 2000)
    with torch.no_grad():
        p = simplex(2000, m.V, None)
        q = simplex(2000, m.V, None)
        near = (p + 0.01 * torch.randn_like(p)).clamp_min(1e-6)
        near = near / near.sum(1, keepdim=True)
        for a, b in ((p, q), (p, near)):
            lhs = l1(m.ctc_F(a, psi), m.ctc_F(b, psi))
            assert (lhs <= rho * l1(a, b) + 1e-6).all()
    # adversarial: gradient ascent on the Lipschitz ratio over (p, p')
    za = torch.randn(256, m.V + 1, requires_grad=True)
    zb = (za.detach() + 0.3 * torch.randn(256, m.V + 1)).requires_grad_()
    opt = torch.optim.Adam([za, zb], lr=0.05)
    psi = psi_of(m, 256)
    best = 0.0
    for _ in range(150):
        a, b = torch.softmax(za, 1), torch.softmax(zb, 1)
        ratio = l1(m.ctc_F(a, psi), m.ctc_F(b, psi)) / l1(a, b).clamp_min(1e-4)
        best = max(best, ratio.max().item())
        opt.zero_grad()
        (-ratio.sum()).backward()
        opt.step()
    assert best <= rho + 1e-6, (best, rho)


def _states(m, tokens):
    """Yield (info dict, token) for each step of the stateful API."""
    m.reset(tokens.shape[0])
    for t in range(tokens.shape[1]):
        with torch.no_grad():
            m.dist()
        yield dict(m._info), tokens[:, t]
        m.observe(tokens[:, t])


def test_halting_w_zero():
    m = make(seed=3, rho=0.5)
    with torch.no_grad():
        m.ctc_W.zero_()
    assert m.rho_global() == 0 and m.certified()
    # F is constant: p1 = F(p0) may differ from p0, p2 = p1, so the residual first passes at s=1 (2 applications)
    r = m.steps_per_token(toks())
    assert (r == 2).all() and m.halted_per_token(toks()).all()


def test_halting_certified_bound_and_error():
    m = make(seed=4, rho=0.6, s_max=50)
    rho = m.rho_global()
    assert 0.59 < rho < 0.61 and m.certified()
    for info, _ in _states(m, toks(T=8)):
        assert info["halted"].all()
        # D6 round bound: ceil(log(eps/d0)/log rho) + 1, d0 = ||p1-p0||_1
        bound = torch.ceil(math.log(qm.EPS_CTC) / math.log(rho) - torch.log(info["d0"]) / math.log(rho)) + 1
        assert (info["rounds"].float() <= bound.clamp_min(2) + 1e-6).all()
        assert (info["rounds"] <= 50).all()
        p = info["p0"]
        psi = info["psi"]
        with torch.no_grad():
            for _ in range(300):
                p = m.ctc_F(p, psi)
        assert (l1(info["p_emit"], p) <= info["resid"] / (1 - rho) + 1e-6).all()


def test_uncertified_runs_s_max(monkeypatch):
    m = make(seed=5, rho=3.0, s_max=5, arm="ctc_hinge")
    assert not m.certified()
    monkeypatch.setattr(qm, "EPS_CTC", 10.0)  # every residual is below eps: still no early emit
    t = toks()
    assert (m.steps_per_token(t) == 5).all() and not m.halted_per_token(t).any()


def test_capped_not_halted():
    m = make(seed=6, rho=0.9, s_max=1)
    assert m.certified()
    t = toks()
    assert (m.steps_per_token(t) == 1).all()
    assert not m.halted_per_token(t).any()


def test_arm_forward_backward():
    m = make(N=16, V=12, r=4, seed=7, rho=0.8, s_max=4)
    loss = m(toks(V=12))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for n in ("ctc_W", "A_theta_U", "A_phi_U", "theta0", "phi0", "theta", "phi", "logits"):
        g = getattr(m, n).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, n


def test_nll_and_stepping_parity():
    m = make(N=16, V=12, r=4, seed=8, rho=0.8, s_max=4)
    t = toks(V=12)
    with torch.no_grad():
        nl = m.nll_per_token(t)
        loss, ss = m(t, return_states=True)
        assert abs(nl.mean() - loss) < 1e-6
        m.reset(t.shape[0])
        for i in range(t.shape[1]):
            lp = m.dist()[0].gather(1, t[:, i:i + 1])[:, 0].log()
            assert (lp + nl[:, i]).abs().max() < 1e-5
            m.observe(t[:, i])
            assert (m.psi - ss[:, i]).abs().max() < 1e-5


def test_generate_poststate():
    m = make(N=16, V=12, r=4, seed=9, rho=0.8, s_max=4)
    x = m.generate(7, B=3, generator=torch.Generator().manual_seed(0))
    assert x.shape == (3, 7) and ((x >= 0) & (x < 12)).all()
    post = m.psi.clone()
    m.reset(3)
    for i in range(7):
        m.observe(x[:, i])
    assert (m.psi - post).abs().max() < 1e-5


def test_rounds_vary():
    m = make(N=16, V=12, r=4, seed=10, rho=0.8, s_max=50)
    r = m.steps_per_token(toks(B=4, T=20, V=12))
    print("rounds", r.unique(return_counts=True))
    assert r.unique().numel() > 1


def test_point_jacobian_diagnostic():
    m = make(seed=11, rho=0.7)
    p = simplex(1, m.V, None)[0]
    j = m.point_jacobian_l1(p)
    assert 0 <= j <= m.rho_global() + 1e-6


@pytest.mark.parametrize("seed,raw", [(0, 0.5), (1, 0.5), (2, 0.5), (3, 5.0)])
def test_certificate_tight_minimal_model(seed, raw):
    """N=2, V=1, M ~ identity: the bound is within ~2x of the true sup (ascent over p, p', psi), so dropping
    the factor 2 or the phi term in rho_global makes this fail."""
    torch.manual_seed(seed)
    m = QFLM(2, 1, s_max=3, arm="ctc", seed=seed, think_rank=1)
    with torch.no_grad():
        m.logits.copy_(torch.tensor([[20.0, 0.0], [0.0, 20.0]]))
        m.theta.zero_(), m.phi.zero_()
        m.ctc_W.copy_(torch.tensor([[1.0, -1.0]]))
        m.A_theta_U.fill_(1.0), m.A_phi_U.fill_(1.0)
        m.ctc_W.mul_(raw / m.rho_raw())
    rho = m.rho_global()
    za, zb = torch.randn(512, 2, requires_grad=True), torch.randn(512, 2, requires_grad=True)
    zp = torch.randn(512, 2, 2, requires_grad=True)
    opt = torch.optim.Adam([za, zb, zp], lr=0.05)
    best = 0.0
    for _ in range(300):
        psi = torch.complex(zp[..., 0], zp[..., 1])
        psi = psi / psi.abs().pow(2).sum(1, keepdim=True).sqrt()
        a, b = torch.softmax(za, 1), torch.softmax(zb, 1)
        ratio = l1(m.ctc_F(a, psi), m.ctc_F(b, psi)) / l1(a, b).clamp_min(1e-3)
        best = max(best, ratio.max().item())
        opt.zero_grad()
        (-ratio.sum()).backward()
        opt.step()
    assert best <= rho + 1e-6, (best, rho)


# ---- Amendment 10: rho_global <= RHO_MAX by construction ----

def test_rho_capped_by_construction():
    assert qm.RHO_MAX == 0.9
    for raw in (0.3, 0.89, 5.0):
        m = make(seed=12, rho=raw)
        assert abs(m.rho_raw() - raw) < 1e-5 and abs(m.rho_global() - min(m.rho_raw(), 0.9)) < 1e-6
    with torch.no_grad():
        m.ctc_W.mul_(100)
    assert abs(m.rho_raw() - 500.0) < 1e-3
    assert m.rho_global() <= 0.9 + 1e-6 and abs(m.rho_global() - min(m.rho_raw(), 0.9)) < 1e-6  # float32 params
    assert m.certified()
    with torch.no_grad():
        m.ctc_W.zero_()  # raw = 0 guard
    assert m.rho_raw() == 0 and m.rho_global() == 0


def test_rescale_active_and_differentiable():
    m = make(N=16, V=12, r=4, seed=13, rho=5.0, s_max=4)
    m(toks(V=12)).backward()
    assert m.ctc_W.grad.abs().sum() > 0 and torch.isfinite(m.ctc_W.grad).all()


def test_ctc_training_stays_capped():
    torch.manual_seed(0)
    N, V = 16, 12
    m = QFLM(N, V, s_max=4, arm="ctc", seed=0, think_rank=4)
    opt = torch.optim.Adam(m.parameters(), lr=3e-2)
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        x = torch.randint(0, V, (4, 3), generator=g)
        opt.zero_grad(); m(x).backward(); qm.clip(m); opt.step()
    rho = m.rho_global()
    assert rho <= 0.9 + 1e-6 and m.certified()
    held = torch.randint(0, V, (4, 6), generator=g)
    # D6 round bound with d0 <= 2 (l1 diameter of the simplex), then s_max = that bound
    m.s_max = int(math.ceil(math.log(qm.EPS_CTC / 2) / math.log(rho))) + 1 if rho > 0 else 2
    for info, _ in _states(m, held):
        assert info["halted"].all()
        bound = torch.ceil((math.log(qm.EPS_CTC) - torch.log(info["d0"])) / math.log(rho)) + 1 if rho > 0 else 2
        assert (info["rounds"].float() <= bound.clamp_min(2) + 1e-6).all()
        assert (info["rounds"] <= m.s_max).all()


def _raw_F(m, p, psi):
    a = p @ m.ctc_W.T
    sh = m.theta0.shape
    th = m.theta0 + (a @ m.A_theta_U.T).reshape(-1, *sh)
    ph = m.phi0 + (a @ m.A_phi_U.T).reshape(-1, *sh)
    q = qm.butterfly(qm.butterfly(psi, th, ph), m.theta, m.phi)
    return (q.real ** 2 + q.imag ** 2) @ m.M()


def test_hinge_forward_is_raw_map_and_penalty():
    m = make(seed=14, rho=5.0, arm="ctc_hinge")
    p, psi = simplex(32, m.V, None), psi_of(m, 32)
    with torch.no_grad():
        assert (m.ctc_F(p, psi) - _raw_F(m, p, psi)).abs().max() < 1e-6
    assert abs(m.rho_global() - 5.0) < 1e-6 and not m.certified()
    pen = m.ctc_penalty(lam=2.0)
    assert abs(pen.item() - 2.0 * (5.0 - 0.9)) < 1e-5
    pen.backward()
    assert m.ctc_W.grad.abs().sum() > 0
    lo = make(seed=14, rho=0.5, arm="ctc_hinge")
    assert lo.ctc_penalty().item() == 0
    # capped arm below the cap is the raw map too
    c = make(seed=14, rho=0.5)
    with torch.no_grad():
        assert (c.ctc_F(p, psi) - _raw_F(c, p, psi)).abs().max() < 1e-6


# ---- active_macs (ctc) ----

def _ctc_macs_by_hand(N, V, r, k):
    L = int(math.log2(N))
    bf = 8 * N * L
    r0 = bf + 2 * N + N * (V + 1)
    F = r * (V + 1) + 2 * (L * N // 2) * r + 2 * bf + 2 * N + N * (V + 1)
    return r0 + k * F + N + 2 * N  # Luders post: N scale + 2N renorm


def test_ctc_active_macs():
    m = QFLM(16, 12, s_max=4, arm="ctc", think_rank=4)
    assert m.active_macs(worst_case=True) >= m.active_macs(rounds=1)
    assert m.active_macs(worst_case=True) == _ctc_macs_by_hand(16, 12, 4, 4)
    assert m.active_macs(rounds=2.5) == _ctc_macs_by_hand(16, 12, 4, 2.5)
    assert m.active_macs(rounds=1) == _ctc_macs_by_hand(16, 12, 4, 1)
    c = lambda N, V: QFLM(N, V, arm="ctc", think_rank=4).active_macs(rounds=1)
    assert c(16, 12) < c(32, 12) and c(16, 12) < c(16, 24)
    with pytest.raises(ValueError):
        m.active_macs()


def _ascent_ratio(m, n=256, steps=300, seed=0):
    """Gradient ascent of ||F(p)-F(p')||_1/||p-p'||_1 over simplex p,p' and psi, using the model's own ctc_F."""
    torch.manual_seed(seed)
    V1, N = m.V + 1, m.N
    za, zb = torch.randn(n, V1, requires_grad=True), torch.randn(n, V1, requires_grad=True)
    zp = torch.randn(n, N, 2, requires_grad=True)
    opt = torch.optim.Adam([za, zb, zp], lr=0.05)
    best = 0.0
    for _ in range(steps):
        psi = torch.complex(zp[..., 0], zp[..., 1])
        psi = psi / psi.abs().pow(2).sum(1, keepdim=True).sqrt()
        a, b = torch.softmax(za, 1), torch.softmax(zb, 1)
        ratio = l1(m.ctc_F(a, psi), m.ctc_F(b, psi)) / l1(a, b).clamp_min(1e-3)
        best = max(best, ratio.max().item())
        opt.zero_grad()
        (-ratio.sum()).backward()
        opt.step()
    return best


@pytest.mark.parametrize("N,V", [(2, 1), (8, 6)])
def test_certificate_covers_actual_map(N, V):
    """Capped ctc (rho_raw ~ 5): the map forward actually uses is the raw map with angles scaled by c, and its
    measured Lipschitz ratio stays <= rho_global(). A map using unscaled angles under a min(rho_raw, 0.9) report fails."""
    m = make(N=N, V=V, r=1 if N == 2 else 4, seed=3, rho=5.0)
    c = qm.RHO_MAX / m.rho_raw()
    assert m.rho_raw() > 4.9 and abs(m.rho_global() - 0.9) < 1e-6
    p = simplex(5, V, None)
    psi = m.psi0().detach().expand(5, N)
    with torch.no_grad():
        m.ctc_W.mul_(c)  # scaled raw params -> independent raw map == capped map
        want = _raw_F(m, p, psi)
        m.ctc_W.div_(c)
        got = m.ctc_F(p, psi)
    assert (got - want).abs().max() < 1e-6
    best = _ascent_ratio(m)
    assert best <= m.rho_global() + 1e-6, (best, m.rho_global())

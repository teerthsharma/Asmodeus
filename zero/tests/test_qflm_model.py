import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

torch.set_num_threads(2)

from qflm import nll
from qflm import model as qm
from qflm.model import QFLM, clip

ARMS = ["qflm", "nothink", "nonull", "embed", "softmax"]


def toks(B=4, T=10, V=12, seed=1):
    return torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("arm", ARMS)
def test_forward_backward_finite(arm):
    m = QFLM(16, 12, arm=arm, seed=0)
    if arm == "embed":
        torch.nn.init.normal_(m.write_phase, std=0.3)
    loss = m(toks())
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    unused = {"theta0", "phi0", "A_theta", "A_phi"} if arm in ("nothink", "nonull") else set()
    for n, p in m.named_parameters():
        if n in unused:
            continue  # think params are never read by these arms
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_qflm_equals_primitive_tau0(monkeypatch):
    monkeypatch.setattr(qm, "TAU", 0.0)
    m = QFLM(16, 12, arm="qflm", seed=3)
    t = toks()
    M = torch.softmax(m.logits, 1)
    psi0 = torch.complex(m.psi0_re, m.psi0_im)
    psi0 = psi0 / psi0.abs().pow(2).sum().sqrt()
    ref = nll(t, M, m.theta, m.phi, (m.theta0, m.phi0, m.A_theta, m.A_phi), m.s_max, psi0)
    assert abs(m(t).item() - ref.item()) < 1e-6


def test_embed_init_equals_qflm_then_differs():
    t = toks()
    a, b = QFLM(16, 12, arm="qflm", seed=5), QFLM(16, 12, arm="embed", seed=5)
    assert abs(a(t).item() - b(t).item()) < 1e-6
    with torch.no_grad():
        b.write_phase.copy_(torch.randn_like(b.write_phase))
    assert abs(a(t).item() - b(t).item()) > 1e-4


def test_arms_differ():
    t = toks()
    base = QFLM(16, 12, arm="qflm", seed=7)(t).item()
    for arm in ["nonull", "nothink"]:
        assert abs(QFLM(16, 12, arm=arm, seed=7)(t).item() - base) > 1e-4, arm


_cache = {}


def trained(arm, V=6, N=16, steps=150, B=4, T=12, s_max=2):
    if arm in _cache:
        return _cache[arm]
    torch.manual_seed(0)
    m = QFLM(N, V, s_max=s_max, arm=arm, seed=0)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(steps):
        off = torch.randint(0, V, (B, 1))
        loss = m((off + torch.arange(T)) % V)
        opt.zero_grad(); loss.backward(); clip(m); opt.step()
    off = torch.randint(0, V, (32, 1))
    with torch.no_grad():
        final = m((off + torch.arange(T)) % V).item()
    print(arm, "final nll", final, "log6", math.log(6))
    _cache[arm] = (m, final)
    return _cache[arm]


@pytest.mark.parametrize("arm", ["qflm", "nonull"])
def test_learns_cycle(arm):
    _, final = trained(arm)
    assert final < 0.5 * math.log(6), final


def test_generate_shape_and_cycle():
    m, _ = trained("qflm")
    V = 6
    g = torch.Generator().manual_seed(0)
    x, steps = m.generate(20, B=2, generator=g, return_steps=True)
    assert x.shape == (2, 20) and x.dtype == torch.long
    assert ((x >= 0) & (x < V)).all()
    assert steps.shape == (2, 20) and ((steps >= 0) & (steps <= m.s_max)).all()
    x = m.generate(30, B=8, temperature=0.3, generator=g)
    ok = (x[:, 1:] == (x[:, :-1] + 1) % V).float().mean().item()
    print("cycle acc", ok)
    assert ok >= 0.8, ok


def test_generate_unit_norm():
    m, _ = trained("qflm")
    m.generate(15, B=3, generator=torch.Generator().manual_seed(1))
    assert m.last_norm_dev < 1e-4, m.last_norm_dev


# ---- stateful API ----

def test_stateful_matches_forward_nonull():
    m = QFLM(16, 12, arm="nonull", seed=2)
    t = toks()
    with torch.no_grad():
        m.reset(t.shape[0])
        tot = 0.0
        for i in range(t.shape[1]):
            p, _ = m.dist()
            assert (p.sum(1) - 1).abs().max() < 1e-6
            x = t[:, i]
            tot -= p.gather(1, x[:, None])[:, 0].log().sum().item()
            m.observe(x)
        assert abs(tot / t.numel() - m(t).item()) < 1e-5


def _set_projective(m, V):
    with torch.no_grad():
        M = torch.full((m.N, V + 1), 0.001 / V)
        M[torch.arange(m.N), torch.arange(m.N) % V] += 0.999
        m.logits.copy_(M.log())
        m.psi0_re.fill_(1.0)
        m.psi0_im.zero_()


def _repeat_prob(m, V):
    out = []
    for x in range(V):
        m.reset(1)
        m.observe(torch.tensor([x]))
        out.append(m.dist()[0][0, x].item())
    return sum(out) / V, min(out)


def test_zeno_repeat_loop():
    V = 12
    m = QFLM(16, V, seed=0)
    _set_projective(m, V)
    with torch.no_grad():
        m.theta.zero_()
        m.phi.zero_()
        avg, lo = _repeat_prob(m, V)
    assert lo >= 0.99, lo
    m = QFLM(16, V, seed=0)  # Q at init angles
    _set_projective(m, V)
    with torch.no_grad():
        avg, _ = _repeat_prob(m, V)
    assert avg <= 0.5, avg
    for a in (m.theta, m.theta0):  # wide init, not small
        assert a.abs().max() > 2.5 and a.std() > 1.0


def test_zeno_think_small_angles():
    N = 16

    def move(eps, seed):
        m = QFLM(N, 12, seed=0)
        with torch.no_grad():
            m.A_theta.zero_()
            m.A_phi.zero_()
            m.theta0.fill_(eps)
            m.phi0.fill_(eps)
            g = torch.Generator().manual_seed(seed)
            psi = torch.complex(torch.randn(1, N, generator=g), torch.randn(1, N, generator=g))
            m.psi = old = psi / psi.abs().pow(2).sum().sqrt()
            m.think()
            return 1 - (old.conj() * m.psi).sum().abs().pow(2).item()

    eps = 0.01
    assert max(move(eps, s) for s in range(10)) <= 4 * eps ** 2 * math.log2(N)
    assert max(move(math.pi / 4, s) for s in range(10)) >= 0.2


@pytest.mark.parametrize("t", [0.37, 1.6])
def test_on_manifold_lerp(t):
    a, b, c = QFLM(16, 12, seed=0), QFLM(16, 12, seed=1), QFLM(16, 12, seed=2)
    sa, sb = a.state_dict(), b.state_dict()
    c.load_state_dict({k: torch.lerp(sa[k], sb[k], t) for k in sa})
    with torch.no_grad():
        c.reset(2)
        g = torch.Generator().manual_seed(0)
        for _ in range(20):
            p, _ = c.dist()
            assert p.shape == (2, 12) and (p.sum(1) - 1).abs().max() < 1e-6
            assert (p >= 0).all()
            c.observe(torch.randint(0, 12, (2,), generator=g))
        assert (c.psi.abs().pow(2).sum(1) - 1).abs().max() < 1e-5


# ---- one semantics for training and serving ----

def _scaled(N=16, V=12, seed=4):
    m = QFLM(N, V, seed=seed)
    with torch.no_grad():
        m.A_theta.mul_(100); m.A_phi.mul_(100)
    return m


def test_forward_states_equal_stepping():
    m, t = _scaled(), toks(B=16, T=10)
    with torch.no_grad():
        loss, states = m(t, return_states=True)
        assert (m.steps_per_token(t) > 0).any()
        m.reset(16)
        for i in range(10):
            m.observe(t[:, i])
            assert (m.psi - states[:, i]).abs().max() < 1e-5, i


def test_dist_marginal_and_generate_frequencies():
    V = 5
    m = _scaled(8, V)
    with torch.no_grad():
        m.reset(1)
        p, r = m.dist()
        assert p.shape == (1, V) and abs(p.sum().item() - 1) < 1e-6 and r.shape == (1,)
        x = m.generate(1, B=4000, generator=torch.Generator().manual_seed(0))
    freq = torch.bincount(x[:, 0], minlength=V) / 4000
    assert (freq - p[0]).abs().max() < 0.05, (freq, p)


def test_generate_state_equals_observe():
    m = _scaled()
    tr = []
    x = m.generate(6, B=8, generator=torch.Generator().manual_seed(2), trace=tr)
    with torch.no_grad():
        m.reset(8)
        for i in range(6):
            m.observe(x[:, i])
            assert (m.psi - tr[i]["post"]).abs().max() < 1e-5, i


def test_tau_truncates_rounds():
    m, t = QFLM(16, 12, seed=0), toks()
    with torch.no_grad():
        m.logits[:, 12].fill_(-30.0)  # null mass ~0
    assert (m.steps_per_token(t) == 1).all()  # P(null^1)<TAU: round 1 is the forced round


# ---- teammate API ----

def test_api_counts_and_nll():
    t = toks()
    m = QFLM(16, 12, seed=0)
    assert m.n_params() == sum(p.numel() for p in m.parameters())
    assert QFLM(16, 12, arm="nothink").n_params() < m.n_params()
    w = lambda m: m.active_macs(worst_case=True)
    assert w(m) < w(QFLM(32, 12)) and w(m) < w(QFLM(16, 24))
    assert w(QFLM(16, 12, arm="nonull")) < w(m)
    assert w(m) >= m.active_macs(rounds=1.0)
    with pytest.raises(ValueError):
        m.active_macs()
    n = m.nll_per_token(t)
    assert n.shape == t.shape and m(t).item() == n.mean().item()
    assert m.steps_per_token(t).shape == t.shape
    m.reset(2); m.step(torch.tensor([1, 2]))


def test_low_rank_think():
    m = QFLM(16, 12, think_rank=4)
    m(toks()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    big, low = QFLM(1024, 1024), QFLM(1024, 1024, think_rank=16)
    for k, q in (("full", big), ("rank16", low)):
        print(k, "n_params", q.n_params(), "think share", q.think_params_count() / q.n_params())
    assert low.n_params() < big.n_params()


def _rand_psi(B, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.complex(torch.randn(B, N, generator=g), torch.randn(B, N, generator=g))
    return z / (z.real ** 2 + z.imag ** 2).sum(1, keepdim=True).sqrt()


def test_low_rank_think_equals_materialised_reference():
    from qflm import think_step
    m = QFLM(32, 12, think_rank=4, seed=3)
    with torch.no_grad():
        for p in (m.A_theta_U, m.A_theta_Vt, m.A_phi_U, m.A_phi_Vt):
            p.mul_(5)
    psi = _rand_psi(3, 32)
    ref = think_step(psi, (m.theta0, m.phi0, m.A_theta_U @ m.A_theta_Vt, m.A_phi_U @ m.A_phi_Vt))
    assert (m._think(psi) - ref).abs().max() < 1e-5


def test_low_rank_think_never_materialises_big_map():
    from torch.utils._python_dispatch import TorchDispatchMode
    N, big = 1024, (10 * 1024 // 2) * 1024
    sizes = []

    class Spy(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            sizes.extend(o.numel() for o in torch.utils._pytree.tree_leaves(out) if isinstance(o, torch.Tensor))
            return out

    m = QFLM(N, 1024, think_rank=16)
    psi = _rand_psi(1, N)
    with torch.no_grad(), Spy():
        m._think(psi)
    assert max(sizes) < big


def test_low_rank_think_speed_and_macs():
    import time
    N = 1024
    full, low = QFLM(N, 64), QFLM(N, 64, think_rank=16)
    psi = _rand_psi(1, N)
    ts = {}
    with torch.no_grad():
        for k, q in (("full", full), ("low", low)):
            q._think(psi)
            t0 = time.perf_counter()
            for _ in range(20):
                q._think(psi)
            ts[k] = (time.perf_counter() - t0) / 20
    print("think seconds", ts)
    # think contribution = macs(rounds=2) - macs(rounds=1) is (meas+lud) + (lud+thk) - (meas+lud) = lud + thk
    thk = lambda q: q.active_macs(rounds=2) - q.active_macs(rounds=1)
    assert thk(low) < 0.2 * thk(full)


def test_no_forbidden_names():
    src = open(qm.__file__).read()
    assert "embedding" not in src and "one_hot" not in src


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_cuda_smoke():
    m = QFLM(16, 12).cuda()
    assert torch.isfinite(m(toks().cuda()))
    assert m.generate(5, B=2).device.type == "cuda"


# ---- parity vs frozen reference + kernel-count proxy (cached M / Q angles) ----
import importlib.util
import qflm as qpkg

_spec = importlib.util.spec_from_file_location(
    "qflm_model_ref", os.path.join(os.path.dirname(__file__), "fixtures", "qflm_model_ref.py"))
ref_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref_mod)


def _trace(cls, arm, r):
    torch.manual_seed(0)
    m = cls(16, 12, arm=arm, seed=4, think_rank=r)
    if arm == "embed":
        torch.nn.init.normal_(m.write_phase, std=0.3)
    t = toks(B=3, T=8)
    with torch.no_grad():
        loss, ss = m(t, return_states=True)
        nl, sp = m.nll_per_token(t), m.steps_per_token(t)
        m.reset(3)
        for v in t[:, :3].T:
            m.observe(v)
        d, dr = m.dist()
        g = m.generate(10, B=2, generator=torch.Generator().manual_seed(9))
    return loss, ss, nl, sp, d, dr, g


@pytest.mark.parametrize("r", [None, 4])
@pytest.mark.parametrize("arm", ARMS)
def test_parity_with_frozen_reference(arm, r):
    a, b = _trace(ref_mod.QFLM, arm, r), _trace(QFLM, arm, r)
    for x, y in zip(a[:5], b[:5]):
        assert torch.allclose(x, y, atol=1e-6, rtol=1e-6)
    assert torch.equal(a[5], b[5]) and torch.equal(a[6], b[6])


def test_kernel_count_proxy(monkeypatch):
    calls = {"bf": 0, "sm": 0}
    for mod in (qpkg, qm):
        orig = mod.butterfly
        def bf(*a, _o=orig, **k):
            calls["bf"] += 1
            return _o(*a, **k)
        monkeypatch.setattr(mod, "butterfly", bf)
    osm = torch.softmax
    def sm(*a, **k):
        calls["sm"] += 1
        return osm(*a, **k)
    monkeypatch.setattr(torch, "softmax", sm)
    m = QFLM(16, 12, arm="qflm", seed=0)
    T, rounds = 8, m.s_max + 1
    with torch.no_grad():
        m(toks(B=3, T=T))
    # per round: Q psi once + think once (no think at s=0)
    assert calls["bf"] <= T * (2 * rounds - 1), calls
    assert calls["sm"] == 1, calls

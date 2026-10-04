"""Arm Q: coherent observer. Decoherence limit = softmax; identities §6 Born-Gibbs numbers; IIA broken
only when coupled; simplex; O(K r) cost; learnable from the limit; T6 hook."""
import pytest
import torch

from atlas.coherent_observer import CoherentAtlas, born_gibbs, cayley_apply, coherent_probs
from atlas.observer_atlas import ObserverAtlas

f64 = dict(dtype=torch.float64)


def _pq(K, r, seed, **kw):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(K, r, generator=g, **kw), torch.randn(K, r, generator=g, **kw)


def test_decoherence_limit_is_softmax():
    s = torch.randn(4, 7, **f64)
    _, Q = _pq(7, 3, 0, **f64)
    assert torch.allclose(coherent_probs(s, torch.zeros_like(Q), Q), s.softmax(-1), atol=1e-12)
    torch.manual_seed(0)
    base = ObserverAtlas(D=16, A=2, K=8, m=2)
    coh = CoherentAtlas(D=16, A=2, K=8, m=2)
    coh.load_state_dict(base.state_dict(), strict=False)
    assert (coh.Cp == 0).all()
    base.eval(); coh.eval()
    o, a = torch.randn(3, 5, 16), torch.randn(3, 5, 2)
    rb, rc = base(o, a), coh(o, a)
    assert torch.allclose(rb["w"], rc["w"], atol=1e-5) and torch.allclose(rb["g"], rc["g"], atol=1e-5)


@pytest.mark.parametrize("g,want", [(0.0, 1.3499), (0.5, 1.5165), (1.0, 2.0587)])
def test_identities_section6(g, want):
    s3 = torch.tensor([0.3, 0.0, 0.2], **f64)
    H3 = -torch.diag(s3); H3[0, 2] = H3[2, 0] = -g
    p3, p2 = born_gibbs(H3), born_gibbs(-torch.diag(s3[:2]))
    assert round(float(p3[0] / p3[1]), 4) == want
    assert round(float(p2[0] / p2[1]), 4) == 1.3499


def test_iia_broken_only_when_coupled():
    s = torch.randn(1, 6, **f64)
    s2 = s.clone(); s2[0, 5] += 2
    P, Q = _pq(6, 2, 1, **f64)
    ratio = lambda p: float(p[0, 0] / p[0, 1])
    r1, r2 = ratio(coherent_probs(s, P, Q)), ratio(coherent_probs(s2, P, Q))
    assert abs(r2 / r1 - 1) > 1e-3
    Z = torch.zeros_like(P)
    r1, r2 = ratio(coherent_probs(s, Z, Q)), ratio(coherent_probs(s2, Z, Q))
    assert abs(r2 / r1 - 1) < 1e-9


def test_simplex_and_orthogonal():
    P, Q = _pq(64, 4, 2)
    p = coherent_probs(torch.randn(128, 64), P, Q)
    assert (p >= 0).all() and (p.sum(-1) - 1).abs().max() < 1e-5
    P, Q = _pq(16, 3, 3, **f64)
    x = torch.randn(10, 16, **f64)
    y = cayley_apply(x, P, Q)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-10)
    A = P @ Q.T - Q @ P.T
    I = torch.eye(16, **f64)
    U = torch.linalg.solve(I - A / 2, I + A / 2)
    assert torch.allclose(y, x @ U.T, atol=1e-10)


def test_cost_linear_in_K_and_never_dense():
    d = lambda K: (CoherentAtlas(D=16, A=2, K=K, m=2, rank=4).active_flops()
                   - ObserverAtlas(D=16, A=2, K=K, m=2).active_flops())
    assert d(1024) - d(512) == 2 * (d(512) - d(256)) > 0  # affine in K: O(K r), not O(K^2)
    P, Q = _pq(200_000, 4, 4)
    assert cayley_apply(torch.randn(4, 200_000), P, Q).shape == (4, 200_000)  # dense KxK = 160 GB


def test_learnable_from_decoherence_limit():
    torch.manual_seed(0)
    coh = CoherentAtlas(D=16, A=2, K=8, m=2)
    coh(torch.randn(3, 5, 16), torch.randn(3, 5, 2))["loss"].backward()
    assert coh.Cp.grad is not None and coh.Cp.grad.norm() > 0


def test_rejects_product_keys():
    with pytest.raises(ValueError):
        CoherentAtlas(D=16, A=2, K=16, m=2, product_keys=True)


def test_t6_hook():
    import t6_contextuality as t6
    torch.manual_seed(0)
    coh = CoherentAtlas(D=16, A=2, K=8, m=2)
    ab, ba = t6.answers(coh, torch.randn(32, 3, 16), torch.randn(32, 3, 16), torch.randn(32, 3, 16))
    assert ab.shape == ba.shape == (32, 2)


def test_phases_matter_only_when_coupled():
    """Origin: 'small phases all around the sphere but the decider will be a softmax'."""
    s = torch.randn(2, 6, **f64)
    P, Q = _pq(6, 2, 5, **f64)
    th = torch.randn(2, 6, **f64)
    Z = torch.zeros_like(P)
    assert torch.allclose(coherent_probs(s, Z, Q, th), s.softmax(-1), atol=1e-12)
    assert torch.allclose(coherent_probs(s, P, Q, th * 0), coherent_probs(s, P, Q), atol=1e-12)
    p = coherent_probs(s, P, Q, th)
    assert (p >= 0).all() and (p.sum(-1) - 1).abs().max() < 1e-12
    assert (p - coherent_probs(s, P, Q)).abs().max() > 1e-3
    torch.manual_seed(0)
    coh = CoherentAtlas(D=16, A=2, K=8, m=2)
    assert (coh.theta_keys == 0).all()  # starts at the real (sign-only) model

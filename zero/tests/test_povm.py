import torch
from atlas.phase_sphere import born, cayley, skew_hermitian
from atlas.povm import naimark_povm, povm_probs, projective_povm

C = torch.complex128


def _unitary(K, gen):
    X = torch.randn(K, K, dtype=C, generator=gen)
    return cayley(skew_hermitian(X))


def _eye(N):
    return torch.eye(N, dtype=C)


def _psd(E):
    return torch.linalg.eigvalsh(E).min() >= -1e-12


def test_projective_povm():
    g = torch.Generator().manual_seed(0)
    for N in (2, 4, 8):
        for _ in range(50):
            U = _unitary(N, g)
            E = projective_povm(U)
            assert E.shape == (N, N, N)
            assert torch.allclose(E.sum(0), _eye(N), atol=1e-10)
            assert torch.allclose(E, E.conj().transpose(-1, -2), atol=1e-12)
            assert _psd(E)
            assert torch.allclose(E @ E, E, atol=1e-10)
            # mutation control: dropping one effect breaks completeness
            assert not torch.allclose(E[1:].sum(0), _eye(N))


def test_naimark_povm():
    g = torch.Generator().manual_seed(1)
    for N, K in ((2, 3), (2, 8), (4, 16)):
        for _ in range(50):
            V = _unitary(K, g)[:, :N]
            E = naimark_povm(V)
            assert E.shape == (K, N, N)
            assert torch.allclose(E.sum(0), _eye(N), atol=1e-10)
            assert _psd(E)
            assert (E @ E - E).flatten(1).norm(dim=1).max() > 1e-3
            assert not torch.allclose(E[1:].sum(0), _eye(N))


def test_povm_probs():
    g = torch.Generator().manual_seed(2)
    for N in (2, 4, 8):
        for _ in range(50):
            U = _unitary(N, g)
            psi = torch.randn(N, dtype=C, generator=g)
            psi = psi / psi.norm()
            p = povm_probs(projective_povm(U), psi)
            assert not p.is_complex()
            assert p.min() >= -1e-12
            assert abs(p.sum() - 1) < 1e-10
            assert torch.allclose(p, born(U @ psi), atol=1e-10)

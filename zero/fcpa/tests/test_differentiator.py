"""L13 cocycle, L14' law-class address + HMM forward filter, L15 commuting composition, L16 holonomy
(generator non-commutativity, BCH-2), L17 unitary crossing (real coefficients)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import logm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa import differentiator as D  # noqa: E402


def rand_law(rng, K=2):
    P = rng.dirichlet(np.ones(K), size=K)
    return D.Law(P=P, nu=float(rng.uniform(0.005, 0.03)), f=rng.standard_normal(2))


DIM = 1 + 2 + 4  # nu + f(2) + P(2x2)


@pytest.mark.parametrize("chart", ["log", "id"])
def test_L13_cocycle(chart):
    """dE_{i->k} = dE_{i->j} + dE_{j->k}, exactly additive in the chosen chart."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b, c = (rand_law(rng) for _ in range(3))
        dab, dbc, dac = D.delta(a, b, chart), D.delta(b, c, chart), D.delta(a, c, chart)
        np.testing.assert_allclose((dab + dbc).vec, dac.vec, atol=1e-12)
        np.testing.assert_allclose(D.delta(a, a, chart).vec, 0, atol=0)
        np.testing.assert_allclose(D.delta(b, a, chart).vec, -dab.vec, atol=1e-12)
    assert dab.vec.shape == (DIM,)


def test_L14prime_address_by_law_class_with_recurrence():
    """Regimes recur: eras 0 and 2 share a law => one class; fingerprints keep the chronology index but the
    address is the class. s-separated classes + estimates within < s/2 => the true class."""
    rng = np.random.default_rng(0)
    laws = [rand_law(rng), rand_law(rng), None, rand_law(rng)]
    laws[2] = D.Law(P=laws[0].P.copy(), nu=laws[0].nu, f=laws[0].f.copy())  # recurrence of era 0
    fps = np.stack([D.fingerprint(l, k) for k, l in enumerate(laws)])
    assert fps.shape == (4, DIM + 1) and list(fps[:, -1]) == [0, 1, 2, 3]
    cls, centers = D.law_classes(fps, tol=1e-9)
    assert list(cls) == [0, 1, 0, 2] and len(centers) == 3
    sep = min(np.linalg.norm(a - b) for a, b in itertools.combinations(centers, 2))
    for _ in range(300):
        k = int(rng.integers(3))
        noise = rng.standard_normal(DIM)
        q = centers[k] + noise / np.linalg.norm(noise) * rng.uniform(0, 0.499) * sep  # < s/2
        assert D.address(centers, q) == k


def test_L14prime_exact_tie_is_deterministic_and_belief_breaks_it():
    centers = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    mid = np.array([1.0, 0.0])
    assert D.address(centers, mid) == 0  # equidistant: lowest class index
    assert D.address(centers, mid, belief=np.array([0.2, 0.8, 0.0])) == 1
    assert D.address(centers, mid, belief=np.array([0.8, 0.2, 0.0])) == 0


def _emission(centers, x, sigma):
    return np.exp(-np.sum((centers - x) ** 2, axis=1) / (2 * sigma ** 2))


def test_L14prime_hmm_forward_filter_matches_bruteforce():
    """alpha_t(j) ∝ e_j(x_t) sum_i alpha_{t-1}(i) A_ij equals the brute-force path-sum posterior."""
    rng = np.random.default_rng(1)
    K, T, sigma = 3, 5, 0.8
    centers = rng.standard_normal((K, 2)) * 2
    A = rng.dirichlet(np.ones(K), size=K)
    pi = rng.dirichlet(np.ones(K))
    xs = rng.standard_normal((T, 2)) * 2
    f = D.HMMFilter(centers, A, sigma, prior=pi)
    for x in xs:
        alpha = f.update(x)
    post = np.zeros(K)
    for path in itertools.product(range(K), repeat=T):
        p = pi[path[0]] * _emission(centers, xs[0], sigma)[path[0]]
        for t in range(1, T):
            p *= A[path[t - 1], path[t]] * _emission(centers, xs[t], sigma)[path[t]]
        post[path[-1]] += p
    np.testing.assert_allclose(alpha, post / post.sum(), atol=1e-12)
    assert abs(alpha.sum() - 1) < 1e-12


def test_L14prime_filter_beats_memoryless_nearest_class():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    A = np.full((3, 3), 0.01) + np.eye(3) * 0.97
    A /= A.sum(1, keepdims=True)
    z = [0]
    for _ in range(599):
        z.append(int(rng.choice(3, p=A[z[-1]])))
    z = np.array(z)
    xs = centers[z] + 1.0 * rng.standard_normal((600, 2))  # noise sd 1.0 >> s/2 = 1.0 boundary: memoryless errs
    f = D.HMMFilter(centers, A, 1.0)
    filt = np.array([int(np.argmax(f.update(x))) for x in xs])
    near = np.array([D.address(centers, x) for x in xs])
    acc_f, acc_n = (filt == z).mean(), (near == z).mean()
    assert acc_f >= 0.9 and acc_f > acc_n + 0.1, (acc_f, acc_n)


def _gens(N=4, A=DIM, commuting=False, seed=0):
    return D.make_generators(N, A, np.random.default_rng(seed), commuting=commuting)


def test_L17_crossing_unitary_real_coefficients():
    G = _gens()
    assert all(np.allclose(g, -g.conj().T) for g in G)
    rng = np.random.default_rng(2)
    a, b = rand_law(rng), rand_law(rng)
    S = D.crossing_operator(D.delta(a, b), G)
    np.testing.assert_allclose(S.conj().T @ S, np.eye(4), atol=1e-12)
    psi, phi = (rng.standard_normal(4) + 1j * rng.standard_normal(4) for _ in range(2))
    np.testing.assert_allclose(np.linalg.norm(S @ psi), np.linalg.norm(psi), rtol=1e-12)
    np.testing.assert_allclose(abs(np.vdot(S @ phi, S @ psi)), abs(np.vdot(phi, psi)), rtol=1e-12)  # fidelity
    np.testing.assert_allclose(D.crossing_operator(D.DeltaE(np.zeros(DIM)), G), np.eye(4), atol=1e-14)
    np.testing.assert_allclose(D.crossing_operator(-D.delta(a, b), G), S.conj().T, atol=1e-12)
    with pytest.raises(ValueError):  # real coefficients required: complex ones break skew-Hermiticity
        D.crossing_operator(D.DeltaE(np.full(DIM, 1j)), G)
    with pytest.raises(ValueError):  # non-skew-Hermitian generator
        D.crossing_operator(D.delta(a, b), [np.ones((4, 4))] * DIM)


def test_L15_commuting_crossings_compose_and_loops_close():
    G = _gens(commuting=True)
    rng = np.random.default_rng(3)
    li, lj, lk = (rand_law(rng) for _ in range(3))
    S = lambda a, b: D.crossing_operator(D.delta(a, b), G)  # noqa: E731
    np.testing.assert_allclose(S(lj, lk) @ S(li, lj), S(li, lk), atol=1e-12)
    np.testing.assert_allclose(S(lk, li) @ S(lj, lk) @ S(li, lj), np.eye(4), atol=1e-12)
    loop = [D.delta(li, lj), D.delta(lj, lk), D.delta(lk, li)]
    np.testing.assert_allclose(D.holonomy(loop, G), np.eye(4), atol=1e-12)
    # non-commuting generators break composition
    Gn = _gens(commuting=False)
    assert np.linalg.norm(D.crossing_operator(D.delta(lj, lk), Gn) @ D.crossing_operator(D.delta(li, lj), Gn)
                          - D.crossing_operator(D.delta(li, lk), Gn)) > 1e-3
    with pytest.raises(ValueError):  # holonomy needs a closed loop
        D.holonomy(loop[:2], G)


def test_L16_holonomy_measures_generator_noncommutativity_bch2():
    """Closed loop i->j->k->i returns U = exp(-1/2 [A,B] + O(eps^3)) (order: A first, then B); with
    commuting generators U = I. The estimate is checked against logm(U) with the expected eps^3 error scaling.
    (This measures non-commutativity of the generators G_a, not curvature of the law manifold.)"""
    G = _gens(commuting=False, seed=1)
    rng = np.random.default_rng(4)
    li, lj, lk = (rand_law(rng) for _ in range(3))
    base = [D.delta(li, lj), D.delta(lj, lk), D.delta(lk, li)]
    errs, norms = [], []
    for eps in (0.2, 0.1, 0.05):
        loop = [e * eps for e in base]
        U = D.holonomy(loop, G)
        est = D.commutator_estimate(loop, G)
        L = logm(U)
        errs.append(np.linalg.norm(L - est))
        norms.append(np.linalg.norm(est))
        np.testing.assert_allclose(U.conj().T @ U, np.eye(4), atol=1e-12)
    assert norms[0] > 1e-3
    assert 6.0 < errs[0] / errs[1] < 10.5 and 6.0 < errs[1] / errs[2] < 10.5, errs  # O(eps^3)
    assert errs[2] / norms[2] < 0.1  # leading term dominates at small eps
    assert 3.0 < norms[0] / norms[1] < 4.5  # estimate itself scales as eps^2
    Gc = _gens(commuting=True)
    np.testing.assert_allclose(D.commutator_estimate(base, Gc), 0, atol=1e-12)

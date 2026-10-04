"""Metric contracts. Every metric must fail a constant latent; torn and folded latents must be told apart."""
import numpy as np

from metrics import (discontinuity_rate, fold_rate, effective_rank, linear_probe,
                     pred_error_from, mapper_betti, tune_mapper, chart_effective_rank)


def sphere(n, seed=0):
    x = np.random.default_rng(seed).normal(size=(n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def geo_s2(a, b):
    return np.arccos(np.clip((a * b).sum(-1), -1, 1))


def torn_s2(x):                                   # (longitude, z): tears along the date line
    return np.stack([np.arctan2(x[:, 1], x[:, 0]), x[:, 2]], 1)


X = sphere(3000)


def test_discontinuity_oracle_zero_torn_positive():
    assert discontinuity_rate(X, X, geo_s2, eps=0.15, delta=0.5) == 0.0
    assert discontinuity_rate(torn_s2(X), X, geo_s2, eps=0.15, delta=0.5) > 0.005


def test_discontinuity_constant_fails_and_invariances():
    assert discontinuity_rate(np.zeros((len(X), 4)), X, geo_s2, 0.15, 0.5) == 1.0
    t = torn_s2(X)
    r = discontinuity_rate(t, X, geo_s2, 0.15, 0.5)
    p = np.random.default_rng(3).permutation(len(X))
    assert discontinuity_rate(t[p], X[p], geo_s2, 0.15, 0.5) == r          # permutation invariance
    assert discontinuity_rate(1e3 * t + 7, X, geo_s2, 0.15, 0.5) == r     # scale/shift invariance


def test_fold_catches_continuous_noninjective():
    proj = X[:, :2]                                 # continuous S^2 -> R^2: no tear, but folds
    assert discontinuity_rate(proj, X, geo_s2, 0.15, 0.5) == 0.0
    assert fold_rate(proj, X, geo_s2, far=1.0, delta=0.05) > 0.01
    assert fold_rate(X, X, geo_s2, far=1.0, delta=0.05) == 0.0
    assert fold_rate(np.zeros((len(X), 2)), X, geo_s2, far=1.0, delta=0.05) == 1.0


def test_pred_error_constant_and_static_latents_cannot_win():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(20, 30, 4)).cumsum(1)
    now, true = g[:, :-1], g[:, 1:]
    assert pred_error_from(true, true, now)["npersist"] == 0.0
    assert pred_error_from(now, true, now)["npersist"] == 1.0              # copy-last = 1
    c = np.zeros_like(g)
    e = pred_error_from(c[:, 1:], c[:, 1:], c[:, :-1])
    assert e["nvar"] == np.inf and e["npersist"] == np.inf
    s = np.repeat(rng.normal(size=(20, 1, 4)), 30, 1)                      # encodes trajectory id only
    e = pred_error_from(s[:, 1:], s[:, 1:], s[:, :-1])
    assert e["nvar"] == 0.0                                                # the variance norm is fooled
    assert e["npersist"] == np.inf                                         # the persistence norm is not


def test_effective_rank():
    rng = np.random.default_rng(0)
    assert effective_rank(rng.normal(size=(5000, 5))) > 4.9
    assert abs(effective_rank(np.outer(rng.normal(size=500), rng.normal(size=6))) - 1) < 1e-6
    assert effective_rank(np.zeros((100, 4))) == 0.0


def test_linear_probe_split_by_trajectory():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(50, 40, 3))
    assert linear_probe(s @ rng.normal(size=(3, 8)), s) > 0.999
    assert linear_probe(np.ones((50, 40, 8)), s) < 0.01
    # a latent that memorises trajectory ids cannot transfer to held-out trajectories
    ids = np.repeat(np.eye(50)[:, None, :], 40, 1)
    assert linear_probe(ids, np.repeat(s[:, :1], 40, 1)) < 0.2


def mb(x, filter_dim, max_dim, tune_seed=0):
    """Mapper as preregistered: cover tuned label-free on a TUNING sample, frozen, applied to a fresh one."""
    cover = tune_mapper(x(tune_seed), filter_dim=filter_dim, max_dim=max_dim)
    assert set(cover) == {"n_cover", "overlap"}
    return mapper_betti(x(tune_seed + 1), filter_dim=filter_dim, max_dim=max_dim, **cover)


def test_mapper_ground_truth():
    def circle(seed):
        th = np.random.default_rng(seed).uniform(0, 2 * np.pi, 1500)
        return np.stack([np.cos(th), np.sin(th)], 1)

    def torus(seed):
        th = np.random.default_rng(seed).uniform(-np.pi, np.pi, (4000, 2))
        return np.concatenate([np.cos(th), np.sin(th)], 1)
    assert mb(circle, 2, 1) == (1, 1)
    assert mb(lambda s: sphere(4000, s), 3, 2) == (1, 0, 1)
    assert mb(torus, 4, 2) == (1, 2, 1)
    cover = tune_mapper(torus(0), filter_dim=4, max_dim=2)
    p = np.random.default_rng(5).permutation(4000)                         # permutation invariance
    assert mapper_betti(torus(1)[p], filter_dim=4, max_dim=2, **cover) == (1, 2, 1)


def test_invariance_of_domain_every_continuous_map_folds():
    """No continuous injective S^2 -> R^2 (invariance of domain; Borsuk-Ulam: some x has f(x) = f(-x)).
    So a d = n latent of S^2 MUST fold, whatever the network: fold_rate is bounded away from 0 for random
    continuous maps (random tanh MLPs at several scales, linear projections), and 0 for an embedding."""
    rng = np.random.default_rng(11)
    maps = [lambda x, A=rng.normal(size=(3, 2)): x @ A for _ in range(3)]
    for scale in (0.5, 2.0, 5.0):
        for _ in range(3):
            W1, b1, W2 = rng.normal(0, scale, (3, 32)), rng.normal(0, 0.5, 32), rng.normal(size=(32, 2))
            maps.append(lambda x, W1=W1, b1=b1, W2=W2: np.tanh(x @ W1 + b1) @ W2)
    rates = [fold_rate(f(X), X, geo_s2, far=1.0, delta=0.05) for f in maps]
    assert min(rates) > 0.05, rates
    assert fold_rate(np.tanh(X @ rng.normal(size=(3, 8))) @ rng.normal(size=(8, 6)), X, geo_s2, 1.0, 0.05) < min(rates)


def test_chart_effective_rank_uses_z():
    rng = np.random.default_rng(0)
    B, T, m, dc, K = 8, 50, 2, 3, 4
    zidx = np.stack([rng.permutation(K)[:m] for _ in range(B * T)]).reshape(B, T, m)
    z = rng.normal(size=(B, T, m, dc))
    z[zidx == 1] = z[zidx == 1][:, :1] * np.ones(dc)            # chart 1 collapsed to one direction
    er = chart_effective_rank(z, zidx)
    assert set(er) == set(range(K)) and abs(er[1] - 1) < 1e-6 and er[0] > 2.5
    assert set(chart_effective_rank(rng.normal(size=(B, T, 5)))) == {0}  # single latent: one "chart"

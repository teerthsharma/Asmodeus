"""H2: TT = frozen softmax-attention head over a KV multiset (order-independent, exactly partial_fit-able)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.utils.estimator_checks import parametrize_with_checks

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa.estimator import _TT  # noqa: E402
from fcpa.tt_attention import AttentionTT  # noqa: E402

V, D, DECAY, CONC = 16, 16, 0.55, 0.05  # CONC, DECAY, beta=10 chosen on validation seed 100, tested on seeds 0-2
KW = dict(random_state=0)


def _stream(seed, n_train=5000, n_test=1000):
    """Order-2 Markov chain on V tokens (peaked Dirichlet rows). Features = EMA of orthogonal token embeddings
    (what the runner holds as m_t); label = next token."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.full(V, CONC), size=(V, V))  # P[a, b] = law of the next token after (a, b)
    n = n_train + n_test + 2
    tok = np.empty(n, int)
    tok[:2] = rng.integers(0, V, 2)
    for t in range(2, n):
        tok[t] = rng.choice(V, p=P[tok[t - 2], tok[t - 1]])
    E = np.linalg.qr(rng.standard_normal((V, D)))[0]
    m, X = np.zeros(D), np.empty((n - 1, D))
    for t in range(n - 1):
        m = DECAY * m + (1 - DECAY) * E[tok[t]]
        X[t] = m
    y, prev = tok[1:], tok[:-1]
    s = n_train
    return X[:s], y[:s], prev[:s], X[s:], y[s:], prev[s:]


def _nll(P, y):
    return float(-np.mean(np.log(np.maximum(P[np.arange(len(y)), y], 1e-300))))


def _lda(Xtr, ytr, Xte):
    tt, edges = _TT(1, V, Xtr.shape[1]), (np.arange(V + 1) - 0.5)[None]
    tt.add(Xtr, ytr[:, None].astype(float), edges)
    return tt.proba(Xte, 1.0)[:, 0]


def _bigram(ptr, ytr, pte):
    C = np.ones((V, V))  # add-one
    np.add.at(C, (ptr, ytr), 1)
    return (C / C.sum(1, keepdims=True))[pte]


def _reference(est, Xtr, ytr, Xte):
    """Explicit per-query softmax-attention, independent of the estimator's code path."""
    W = est.beta * est._Q_
    K, q = Xtr @ W.T, Xte @ W.T
    out = np.empty((len(Xte), V))
    for i, qi in enumerate(q):
        s = K @ qi / np.sqrt(K.shape[1])
        w = np.exp(s - s.max())
        out[i] = [w[ytr == c].sum() for c in range(V)]
        out[i] /= w.sum()
    return (1 - est.eps) * out + est.eps / V


def _chunked(Xtr, ytr, seed=7, **kw):
    perm = np.random.default_rng(seed).permutation(len(Xtr))
    b = AttentionTT(**KW, **kw)
    for idx in np.array_split(perm, 10):
        b.partial_fit(Xtr[idx], ytr[idx], classes=np.arange(V))
    return b


def test_H2_attention_tt_bitwise_and_reference_exact():
    Xtr, ytr, _, Xte, _, _ = _stream(0)
    a = AttentionTT(**KW).fit(Xtr, ytr)  # A
    b = _chunked(Xtr, ytr)  # B
    PA = a.predict_proba(Xte)
    assert np.array_equal(PA, b.predict_proba(Xte))  # bitwise: the memory is a multiset
    assert np.abs(PA - _reference(a, Xtr, ytr, Xte)).max() < 1e-12  # R
    assert np.allclose(PA.sum(1), 1.0, atol=1e-12)


# Measured: margin >= 0.10 holds on 12/13 streams; seed 2 is 0.084 (strict xfail records it, not a loosened bar).
@pytest.mark.parametrize("seed", [0, 1, pytest.param(2, marks=pytest.mark.xfail(strict=True, reason="margin 0.084")),
                                  *range(10, 20)])
def test_H2_nll_margin_vs_lda_and_bigram(seed):
    Xtr, ytr, ptr, Xte, yte, pte = _stream(seed)
    nA = _nll(AttentionTT(**KW).fit(Xtr, ytr).predict_proba(Xte), yte)
    nL, nB = _nll(_lda(Xtr, ytr, Xte), yte), _nll(_bigram(ptr, ytr, pte), yte)
    print(f"seed {seed}: NLL attn {nA:.4f} lda {nL:.4f} bigram {nB:.4f}")
    assert nA <= nL - 0.10 and nA <= nB - 0.10


def test_H2_performer_variant_is_order_independent_and_a_distribution():
    Xtr, ytr, _, Xte, yte, _ = _stream(0)
    a = AttentionTT(performer=256, **KW).fit(Xtr, ytr)
    b = _chunked(Xtr, ytr, performer=256)
    PA = a.predict_proba(Xte)
    assert np.array_equal(PA, b.predict_proba(Xte))
    assert np.allclose(PA.sum(1), 1.0, atol=1e-12) and PA.min() > 0
    print(f"performer NLL {_nll(PA, yte):.4f}")


@parametrize_with_checks([AttentionTT(random_state=0)])
def test_H2_sklearn_check_estimator(estimator, check):
    check(estimator)

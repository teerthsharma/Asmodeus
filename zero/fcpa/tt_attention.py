"""H2 TT: a frozen softmax-attention head whose KV cache is the sufficient statistic.

partial_fit appends (x_i, y_i) to a multiset; predict_proba(q) = softmax(q.K^T / sqrt(d)) . onehot(Y), computed on
the memory sorted lexicographically, so the result is a function of the multiset only (chunking/order-free, bitwise).
W_q = W_k = beta * Q with Q a seeded orthogonal matrix (frozen): the head is then a kernel smoother in x.
`performer=m` swaps the softmax for positive random features (S = sum phi(k_i) e_{y_i}^T, linear in the memory).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class AttentionTT(ClassifierMixin, BaseEstimator):
    def __init__(self, beta=10.0, eps=1e-3, performer=0, random_state=None):
        self.beta, self.eps, self.performer, self.random_state = beta, eps, performer, random_state

    def fit(self, X, y):
        for a in ("classes_", "_X", "_y"):
            self.__dict__.pop(a, None)
        return self.partial_fit(X, y)

    def partial_fit(self, X, y, classes=None):
        first = not hasattr(self, "classes_")
        X, y = validate_data(self, X, y, reset=first, dtype=np.float64)
        check_classification_targets(y)
        if first:
            self.classes_ = np.unique(y if classes is None else classes)
            rng = check_random_state(self.random_state)
            d = X.shape[1]
            self._Q_ = np.linalg.qr(rng.standard_normal((d, d)))[0]
            self._omega_ = rng.standard_normal((self.performer, d)) if self.performer else None
            self._X, self._y = np.empty((0, d)), np.empty(0, int)
        idx = np.searchsorted(self.classes_, y)
        if np.any(idx >= len(self.classes_)) or np.any(self.classes_[np.minimum(idx, len(self.classes_) - 1)] != y):
            raise ValueError("partial_fit got a label outside classes_")
        self._X, self._y = np.vstack([self._X, X]), np.r_[self._y, idx]
        return self

    def predict(self, X):
        check_is_fitted(self, "_X")
        return self.classes_[self.predict_proba(X).argmax(1)]  # predict_proba does check_is_fitted

    def _sorted(self):
        # ponytail: re-sorts per call (n log n); cache behind a dirty flag if predict dominates
        o = np.lexsort((self._y, *self._X.T[::-1]))
        return self._X[o], self._y[o]

    def predict_proba(self, X):
        check_is_fitted(self, "_X")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        Xm, ym = self._sorted()
        W = self.beta * self._Q_
        K, q = Xm @ W.T, X @ W.T
        Y = np.eye(len(self.classes_))[ym]
        if self.performer:
            P = self._performer(q, K, Y)
        else:
            s = q @ K.T / np.sqrt(K.shape[1])
            w = np.exp(s - s.max(1, keepdims=True))
            P = (w @ Y) / w.sum(1, keepdims=True)
        return (1 - self.eps) * P + self.eps / len(self.classes_)

    def _performer(self, q, K, Y):
        """Positive random features for exp(q.k / sqrt(d)); per-row / global shifts cancel in the ratio."""
        def logphi(U):
            U = U * U.shape[1] ** -0.25
            return U @ self._omega_.T - 0.5 * (U * U).sum(1, keepdims=True)

        lk, lq = logphi(K), logphi(q)
        fk, fq = np.exp(lk - lk.max()), np.exp(lq - lq.max(1, keepdims=True))
        return (fq @ (fk.T @ Y)) / (fq @ fk.sum(0))[:, None]

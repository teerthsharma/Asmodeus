"""Era model: the JEPA inside a slab (D5, D9, L19, L18*, L27').

Ridge predictor in a chart, with an isotropy penalty and a collapse guard on the embedding rank.
All state is additive sufficient statistics in the AMBIENT space, so partial_fit equals fit (L27')
and the chart may change after the data arrived (the chart solve is exact from the statistics).
"""
import numpy as np


def _eff_rank_cov(C):
    ev = np.clip(np.linalg.eigvalsh((C + C.T) / 2), 0.0, None)
    s = ev.sum()
    if s <= 1e-300:
        return 0.0
    p = ev[ev > 0] / s
    return float(np.exp(-(p * np.log(p)).sum()))


def effective_rank(E):
    """exp(entropy of the normalised covariance spectrum) of embeddings E (n, d). L19 guard quantity."""
    E = np.asarray(E, float)
    return _eff_rank_cov(np.cov(E, rowvar=False).reshape(E.shape[1], E.shape[1]))


def sigreg_penalty(E):
    """SIGReg-lite (L19): |C/(tr C/d) - I|_F^2 / d for the embedding covariance C. 0 isotropic, d-1 rank one."""
    E = np.asarray(E, float)
    d = E.shape[1]
    C = np.cov(E, rowvar=False).reshape(d, d)
    m = np.trace(C) / d
    if m <= 1e-300:
        return float(d - 1)
    return float(np.linalg.norm(C / m - np.eye(d)) ** 2 / d)


def era_chart(rho, d_c):
    """L18: V (N, d_c) = top-d_c eigenvectors of the era density rho (Ky Fan maximiser)."""
    ev, U = np.linalg.eigh((rho + rho.conj().T) / 2)
    return U[:, ::-1][:, :d_c]


def is_top_eigenspace(V, rho, tol=1e-8):
    """True iff span(V) is the top-d_c eigenspace of rho, i.e. tr(V^H rho V) equals the Ky Fan sum."""
    d_c = V.shape[1]
    top = np.sort(np.linalg.eigvalsh((rho + rho.conj().T) / 2))[::-1][:d_c].sum()
    return bool(abs(np.trace(V.conj().T @ rho @ V).real - top) <= tol)


def zoom_floor(lam_sum, zeta0, V=None, rho=None):
    """L18*: P(zeta <= zeta0) <= (1 - sum_{i<=d_c} lam_i) / (1 - zeta0).

    Valid ONLY when E[zeta] >= sum_{i<=d_c} lam_i, i.e. V is the top-d_c eigenspace of rho (Markov on 1 - zeta).
    Pass V and rho to have that hypothesis asserted (ValueError otherwise).
    """
    if V is not None and rho is not None and not is_top_eigenspace(V, rho):
        raise ValueError("L18* floor needs V to be the top-d_c eigenspace of rho")
    return (1.0 - lam_sum) / (1.0 - zeta0)


def readout_lipschitz(L_J, zeta_min):
    """L18*: Lip of psi -> J(V^H psi / |V^H psi|) on {zeta >= zeta_min}: 2 L_J / sqrt(zeta_min)."""
    return 2.0 * L_J / np.sqrt(zeta_min)


class EraModel:
    """Ridge JEPA predictor Z_t -> Z_t1 in a chart (rows of `chart`, shape (d_c, D); None = identity).

    The state is the CENTRED sufficient statistics (count, means, scatter, cross-scatter), merged by the exact
    parallel (Chan) update, so partial_fit equals fit (L27') without the cancellation an uncentred Gram suffers
    under a large offset. lam is a ridge relative to the per-sample variance, lam * tr(Gc)/(d n): its relative
    effect on the coefficients is O(lam / n) and vanishes with data. beta shrinks the Gram toward isotropy
    (SIGReg-lite on the input side); it is OFF by default because it biases the coefficients: the isotropy term
    belongs on the embedding, which is what the guard does (L19): lam is cut by 10x until the predicted
    embeddings keep >= min_rank_frac of the target's effective rank; collapsed_ records failure.
    """

    def __init__(self, lam=1e-2, beta=0.0, min_rank_frac=0.5, guard=True, chart=None):
        self.lam, self.beta, self.min_rank_frac, self.guard, self.chart = lam, beta, min_rank_frac, guard, chart
        self.n_ = 0

    @property
    def s_(self):
        return self.n_ * self.mu_

    @property
    def G_(self):
        return self.M_ + self.n_ * np.outer(self.mu_, self.mu_)

    def partial_fit(self, Z, Z1, solve=True):
        Z = np.asarray(Z, float)
        Z1 = np.asarray(Z1, float).reshape(len(Z), -1)
        nb = len(Z)
        if nb == 0:
            return self
        mub, mub1 = Z.mean(0), Z1.mean(0)
        dZ, dZ1 = Z - mub, Z1 - mub1
        Mb, Cb, Tb = dZ.T @ dZ, dZ.T @ dZ1, dZ1.T @ dZ1
        if self.n_ == 0:
            self.n_, self.mu_, self.mu1_, self.M_, self.C_, self.T_ = nb, mub, mub1, Mb, Cb, Tb
        else:
            na, n = self.n_, self.n_ + nb
            d, d1, f = mub - self.mu_, mub1 - self.mu1_, self.n_ * nb / n
            self.M_ = self.M_ + Mb + f * np.outer(d, d)
            self.C_ = self.C_ + Cb + f * np.outer(d, d1)
            self.T_ = self.T_ + Tb + f * np.outer(d1, d1)
            self.mu_, self.mu1_, self.n_ = self.mu_ + d * nb / n, self.mu1_ + d1 * nb / n, n
        return self._solve() if solve else self

    def fit(self, Z, Z1):
        self.n_ = 0
        return self.partial_fit(Z, Z1)

    def set_chart(self, chart):
        self.chart = chart
        return self._solve()

    def _solve(self):
        n = self.n_
        P = np.eye(len(self.mu_)) if self.chart is None else np.asarray(self.chart, float)
        G, C = P @ self.M_ @ P.T, P @ self.C_
        d = len(G)
        tgt = _eff_rank_cov(self.T_ / n)
        scale = max(np.trace(G) / (d * n), 1e-300)  # per-sample variance, offset-free
        Giso = (1 - self.beta) * G + self.beta * n * scale * np.eye(d)
        lam, W, ok = self.lam, None, False
        for _ in range(9):
            W = np.linalg.lstsq(Giso + lam * scale * np.eye(d), C, rcond=None)[0]
            ok = _eff_rank_cov(W.T @ G @ W / n) >= self.min_rank_frac * tgt
            if ok or not self.guard:
                break
            lam /= 10.0
        self.W_, self.b_, self.lam_eff_, self._P = W, self.mu1_ - W.T @ (P @ self.mu_), lam, P
        self.collapsed_ = bool(self.guard and not ok)
        return self

    def predict(self, Z):
        if self.n_ == 0:
            raise RuntimeError("EraModel is not fitted")
        return (np.asarray(Z, float) @ self._P.T) @ self.W_ + self.b_


class SymEraModel(EraModel):
    """L37 era model: ridge on fixed, pre-scaled features Z of the state (symmetry.ModeFeatures), restricted to
    the maps that commute with the bed's symmetry group (V_Q).

    The statistics are those of the data as seen, Chan-merged, so partial_fit equals fit (L27'). The solve reads
    their group average (symmetry.average), which is exactly the statistics of the data augmented by every group
    image weighted 1/|G|; the ridge on it lands in V_Q (L37 Q2). The penalty is lam * n on every feature but the
    intercept (a fixed relative shrinkage at any era size, the features being pre-scaled). No rank guard and no
    chart: lam is a pin and the features are fixed."""

    def __init__(self, lam, zdesc, ydesc):
        super().__init__(lam=lam, guard=False)
        self.zdesc, self.ydesc = zdesc, ydesc

    def set_chart(self, chart):
        return self._solve()

    def _solve(self):
        from .symmetry import average, average_vec
        n, mu, mu1 = self.n_, self.mu_, self.mu1_
        ms, ms1 = average_vec(mu, self.zdesc), average_vec(mu1, self.ydesc)
        G = average(self.M_ + n * np.outer(mu, mu), self.zdesc, self.zdesc) - n * np.outer(ms, ms)
        C = average(self.C_ + n * np.outer(mu, mu1), self.zdesc, self.ydesc) - n * np.outer(ms, ms1)
        A = G + self.lam * n * np.eye(len(G))
        try:
            W = np.linalg.solve(A, C)
        except np.linalg.LinAlgError:
            W = np.linalg.lstsq(A, C, rcond=None)[0]
        self.W_, self.b_, self.lam_eff_, self._P, self.collapsed_ = W, ms1 - W.T @ ms, self.lam, None, False
        return self

    def predict(self, Z):
        if self.n_ == 0:
            raise RuntimeError("EraModel is not fitted")
        return np.asarray(Z, float) @ self.W_ + self.b_

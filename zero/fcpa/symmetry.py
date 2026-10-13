"""L37: era features with a known action of the bed's symmetry group, and the group average of their statistics.

The NS-2D bed is O(2)-equivariant (L37 S0-S2): an x-shift T_s multiplies stored mode k by e^{-i kx s}, and the
rotation by pi, C, conjugates every mode. Every feature below is the Re or Im part of a complex monomial f of the
modes with a CHARGE q (T_s f = e^{-i q s} f, C f = conj f), or a real invariant (charge 0, Re only):
    linear     a_k of x_t, x_{t-1}, ...                 charge kx
    quadratic  a_i a_j (i <= j), a_i conj a_j (i < j)   charge kx_i + kx_j, kx_i - kx_j
               |a_i|^2                                  charge 0
The quadratic block is an orthonormal basis of Sym^2 of the independent coordinates (L37 R1: the ridge penalty
is then invariant), and the group acts on each (Re f, Im f) pair by a rotation (by q s) and a reflection (C).

`average` is the Reynolds operator: the mean over the group of g S g'^T for a second-moment statistic S, i.e. the
statistic of the data augmented by every group image, each weighted 1/|G|. For 2 x 2 blocks B between a pair of
charge p and one of charge q, writing B = a I + b J + c K + d L (J the rotation generator, K = diag(1, -1),
L = [[0, 1], [1, 0]]): rotations keep a I + b J iff p = q and c K + d L iff p = -q, and C kills J and L. So the
average keeps (B00 + B11) / 2 on the diagonal if p = q != 0, +-(B00 - B11) / 2 if p = -q != 0, the diagonal of B
if p = q = 0, and nothing else. The shift average over Gamma_17 (L37 Q2) equals that over O(2) because every
charge here has |q| <= 8 < 17 / 2; `average` is exact, and costs O(size of S) instead of |G| products.
"""
import numpy as np

_R2 = 1.0 / np.sqrt(2.0)


def average(S, rows, cols):
    """Group average of a second-moment statistic S (rows x cols); rows / cols = (charge, part, partner)."""
    (cr, ar, pr), (cc, ac, pc) = rows, cols
    S = np.asarray(S, float)
    Spp = S[np.ix_(pr, pc)]  # the same entry of the 2 x 2 block with both parts swapped (B11 for B00, ...)
    same = cr[:, None] == cc[None, :]
    opp = cr[:, None] == -cc[None, :]
    out = np.where(same & opp, S, np.where(same, (S + Spp) / 2, np.where(opp, (S - Spp) / 2, 0.0)))
    out[ar[:, None] != ac[None, :]] = 0.0
    return out


def average_vec(v, desc):
    """Group average of a first moment: only the invariant (charge 0, Re) entries survive."""
    c, a, _ = desc
    return np.where((c == 0) & (a == 0), v, 0.0)


def _desc(charge):
    """(charge, part, partner) of [Re block, Im block] of len(charge) complex features."""
    m = len(charge)
    i = np.arange(m)
    return np.r_[charge, charge], np.r_[np.zeros(m, int), np.ones(m, int)], np.r_[i + m, i]


class ModeFeatures:
    """z(x) = [lin(x_t), lin(x_{t-1}), ..., quad(x_t)] / scale on rows of the lag stack; `desc` / `ydesc` give the
    (charge, part, partner) of every feature / stored target coordinate. law: an object with .D and .modes()."""

    def __init__(self, law, n_lags, quad, scale):
        charge, part, ind = law.modes()
        self.D, self.n_lags, self.quad, self.scale = law.D, int(n_lags), bool(quad), float(scale)
        self.re, self.im = ind, ind + law.D // 2
        q = charge[ind]
        blocks = [_desc(q) for _ in range(self.n_lags)]
        if self.quad:
            m = len(ind)
            self._iu, self._ju = np.triu_indices(m)
            self._iv, self._jv = np.triu_indices(m, 1)
            self._w = np.where(self._iu == self._ju, _R2, 1.0)  # a_i^2 has norm sqrt 2 in orthonormal Sym^2
            blocks += [_desc(q[self._iu] + q[self._ju]), _desc(q[self._iv] - q[self._jv]),
                       (np.zeros(m, int), np.zeros(m, int), np.arange(m))]
        off, c, a, p = 0, [], [], []
        for bc, ba, bp in blocks:
            c.append(bc), a.append(ba), p.append(bp + off)
            off += len(bc)
        self.desc = (np.concatenate(c), np.concatenate(a), np.concatenate(p))
        h = law.D // 2
        self.ydesc = (charge, part, np.r_[np.arange(h) + h, np.arange(h)])
        self.n_features = off

    @staticmethod
    def rms(law, X):
        """The pinned scale: root mean square of the independent coordinates of the newest block of the rows X."""
        _, _, ind = law.modes()
        X = np.asarray(X, float)
        return float(np.sqrt(np.mean(np.r_[X[:, ind], X[:, ind + law.D // 2]] ** 2)))

    def __call__(self, X):
        X = np.asarray(X, float)
        cols = []
        a0 = None
        for lag in range(self.n_lags):
            b = X[:, lag * self.D:(lag + 1) * self.D]
            a = (b[:, self.re] + 1j * b[:, self.im]) / self.scale
            a0 = a if a0 is None else a0
            cols += [a.real, a.imag]
        if self.quad:
            f = a0[:, self._iu] * a0[:, self._ju] * self._w
            g = a0[:, self._iv] * np.conj(a0[:, self._jv])
            cols += [f.real, f.imag, g.real, g.imag, np.abs(a0) ** 2 * _R2]
        return np.concatenate(cols, axis=1)

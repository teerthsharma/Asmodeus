"""FCPARegressor: the scikit-learn reference implementation of FCPA (numpy / scipy / scikit-learn only).

Data flow (spec D1-D11, L27'):
  x -> psi = normalise(R [x;1]) in C^N, amplitude a = |R [x;1]| kept (E1), phi = a psi is linear in x.
  Observer (L1): Born outcome of a Parseval frame, coarse symbol s in {0..S-1}.
  Eras (D5): tumbling windows of `window` rows; a window whose Hoeffding interval of 'fraction explained by the
    open era's Gaussian' is disjoint from that of the two windows before it (L11, bounds.certified_change)
    opens a new era. Every era keeps ADDITIVE statistics only: rho = sum psi psi^H, observer-symbol Markov
    transition counts (the law), an EraModel's Gram/cross-moment (L19), and the TT's class-conditional sums. Every 5th window is held out for conformal calibration (L6') and drives ACI (L7).
  TT (D7, TT-1): softmax predictor over discretised outputs. It is the LDA posterior (a softmax of linear
    logits) of the output bins, built from additive statistics only, so it is exact under partial_fit.
  Router (L5): conformal era set C from the Mahalanobis score d2_k/df_k; decide(C, certified_top1).
    JUMP/BLEND/REFUSE_BLEND -> the era model(s) answer through the crossing operator S_k (D8, L15/L17);
    OOD/ALONE -> the TT answers alone. Blends use the damped CTC loop (L22-dagger), uniform if it refuses.
  partial_fit state = (sufficient statistics, w-sample buffer, open-segment statistics) = L27'.
"""
import copy
import warnings

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import functools
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_random_state, validate_data

from . import bounds, conformal, differentiator as diff
from .ctc import damped_loop
from .eras import EraModel, SymEraModel, era_chart
from .symmetry import ModeFeatures
from .observer import born, parseval_frame
from .router import ALONE, OOD, certified_topm, decide

_S = 2  # coarse observer symbols (the window test needs a small alphabet: windows are short)
_CAL_EVERY = 5  # holdout="interleaved": window j is a calibration window iff j % 5 == 4
_CAL_MAX = 400
_DELTA = 1e-3  # failure level of the window comparison (L11)
_CROSS = 0.3  # scale of the crossing generators
_OOD_FRAC = 0.1  # OOD level = alpha * _OOD_FRAC
_TAU = 1.7  # law mode: a window belongs to an era iff its mean normalised law residual is <= _TAU
_N_BIRTH = 3  # law mode: unexplained windows needed (two to fit, one to confirm) to open a new era
_MIN_ERA_ROWS, _MIN_ERA_FRAC = 60, 0.03  # an era answers queries only with this many training rows (or share)
_MIN_GAIN = 0.2  # y=None: an increment model must beat persistence by 20% on held-out windows. Windows are interleaved with training windows of the same sequence, so small gains are partly leakage; a large one survives it


def _single_blas(f):
    """Run `f` with single-threaded BLAS: every matrix here is at most a few hundred wide, where a thread pool is
    200-700x SLOWER than one thread when the machine is busy (measured: 0.24 s vs 0.3 ms per 182x182 solve)."""
    @functools.wraps(f)
    def g(*a, **k):
        with threadpool_limits(limits=1, user_api="blas"):
            return f(*a, **k)
    return g


class _LawStats:
    """Additive (Chan-merged) statistics of a law R = nu L + f over rows: nu = S_RL / S_LL (E2': both centred, so
    f is constant along the rows), f = mean(R) - nu mean(L), sigma2 = residual energy per row."""

    def __init__(self):
        self.n, self.mR, self.mL, self.SRL, self.SLL, self.SRR = 0, None, None, 0.0, 0.0, 0.0

    def add(self, R, L):
        nb, mR, mL = len(R), R.mean(0), L.mean(0)
        dR, dL = R - mR, L - mL
        b = ((dR * dL).sum(), (dL * dL).sum(), (dR * dR).sum())
        if self.n == 0:
            self.n, self.mR, self.mL = nb, mR, mL
            self.SRL, self.SLL, self.SRR = b
            return
        n, fac = self.n + nb, self.n * nb / (self.n + nb)
        a, c = mR - self.mR, mL - self.mL
        self.SRL += b[0] + fac * float(a @ c)
        self.SLL += b[1] + fac * float(c @ c)
        self.SRR += b[2] + fac * float(a @ a)
        self.mR, self.mL, self.n = self.mR + a * nb / n, self.mL + c * nb / n, n

    def model(self, min_n=10):
        """(nu, f, sigma2) or None if the law is not identifiable yet (E2': a steady L gives REFUSE)."""
        if self.n < min_n or self.SLL <= 1e-30 * max(self.SRR, 1e-300):
            return None
        nu = self.SRL / self.SLL
        s2 = max((self.SRR - 2 * nu * self.SRL + nu * nu * self.SLL) / self.n, 1e-300)
        return nu, self.mR - nu * self.mL, s2


class _Seg:
    """One era: additive statistics only."""

    def __init__(self, N):
        self.n = 0
        self.rho = np.zeros((N, N), complex)
        self.counts = np.zeros((_S, _S))
        self.era = EraModel()
        self.xera = None  # era != "phi": the L37 symmetric ridge on mode features (SymEraModel)
        self.law = _LawStats()
        self.gs, self.gn = 0.0, 0  # y=None: sum and count of the norm ratios |x_{t+1}| / |x_t| (additive)

    def gauss(self):
        """(mean, precision, df) of the era's features phi, from the EraModel's additive statistics; None if empty."""
        if not self.n:
            return None
        e = self.era
        mu, C = e.mu_, e.M_ / self.n
        lam = 1e-6 * max(np.trace(C) / len(mu), 1e-12)
        P = np.linalg.inv(C + lam * np.eye(len(mu)))
        return mu, P, max(float(np.trace(P @ C)), 1.0)


class _TT:
    """Per-output class-conditional sums: counts[p,B], sumx[p,B,d], sumy[p,B] and the Gram G[d,d]."""

    def __init__(self, p, B, d):
        self.n = 0
        self.counts, self.sumy = np.zeros((p, B)), np.zeros((p, B))
        self.sumx, self.G = np.zeros((p, B, d)), np.zeros((d, d))

    def add(self, X, Y, edges):
        self.n += len(X)
        self.G += X.T @ X
        for j in range(Y.shape[1]):
            b = np.searchsorted(edges[j, 1:-1], Y[:, j], side="right")
            for k in np.unique(b):
                m = b == k
                self.counts[j, k] += m.sum()
                self.sumy[j, k] += Y[m, j].sum()
                self.sumx[j, k] += X[m].sum(0)

    def proba(self, X, temp, ridge=1e-3):
        """pi(b | x) (n, p, B): the LDA posterior, a softmax of linear logits (TT-1: 1-Lipschitz in l2)."""
        p, B, d = self.sumx.shape
        n_b = self.counts
        mu = np.divide(self.sumx, n_b[..., None], out=np.zeros_like(self.sumx), where=n_b[..., None] > 0)
        out = np.empty((len(X), p, B))
        for j in range(p):
            Sig = (self.G - np.einsum("b,bi,bk->ik", n_b[j], mu[j], mu[j])) / max(self.n, 1)
            Sig = (Sig + Sig.T) / 2 + (ridge * np.trace(Sig) / d + 1e-8) * np.eye(d)
            W = np.linalg.solve(Sig, mu[j].T)  # (d, B)
            logit = X @ W - 0.5 * np.sum(mu[j].T * W, axis=0) + np.log(np.maximum(n_b[j], 1e-300) / self.n)
            logit = np.where(n_b[j] > 0, logit / temp, -np.inf)
            out[:, j] = np.exp(logit - logsumexp(logit, axis=1, keepdims=True))
        return out

    def centers(self):
        return np.divide(self.sumy, self.counts, out=np.zeros_like(self.sumy), where=self.counts > 0)


class _State:
    """The L27' partial_fit state: statistics (segs, tt, cal), the w-sample buffer, the open segment."""

    def __init__(self, d, p, B, N, alpha, gamma):
        self.segs = [_Seg(N)]
        self.tt = _TT(p, B, d)
        self.bufX, self.bufY = np.zeros((0, d)), np.zeros((0, p))
        self.hist, self.last_sym, self.window_idx = [], None, 0
        self.cal_phi, self.cal_lab = np.zeros((0, 2 * N)), np.zeros(0, int)
        self.cal_X, self.cal_Y = np.zeros((0, d)), np.zeros((0, p))
        self.cal_traj, self.n_cal_traj = np.zeros(0, int), 0  # Cal trajectory of each pool row (-1: interleaved)
        self.aci = conformal.ACI(alpha, gamma, alpha)
        self.n_pairs = 0
        self.nov = []  # law mode: windows no era explains yet
        self.last_x = None  # y=None: the previous chunk's last row (its pair crosses the chunk boundary)


class FCPARegressor(RegressorMixin, BaseEstimator):
    """Fluid Chrononal Predictive Architecture regressor.

    fit(X, y=None, t=None): y=None means next-step prediction of X (X[i] -> X[i+1]); t orders the rows.
    partial_fit is exact (L27'): chunked partial_fit equals fit on the concatenation when both use the same
    output discretisation (`y_edges`, e.g. FCPARegressor.make_edges(y, n_bins)); without it the edges come
    from the first chunk. Chunks are assumed to arrive in time order.
    """

    def __init__(self, n_eras_max=64, d_chart=8, alpha=0.1, gamma=0.005, temp=1.0, eps=0.1, window=32,
                 random_state=None, n_bins=8, y_edges=None, ood="nearest_era", n_lags=1, law=None, damping=1.0, energy=False,
                 holdout="interleaved", era="phi", era_lam=0.1, era_pool=None, era_scale=None):
        self.n_eras_max = n_eras_max
        self.d_chart = d_chart
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp
        self.eps = eps
        self.window = window
        self.random_state = random_state
        self.n_bins = n_bins
        self.y_edges = y_edges
        self.ood = ood
        self.n_lags = n_lags
        self.law = law
        self.damping = damping
        self.energy = energy
        self.era = era  # "phi": EraModel on phi in the chart; "lin_vq" / "quad_vq": SymEraModel on mode features (L37)
        self.era_lam = era_lam  # SymEraModel ridge, lam * n on the pre-scaled features
        self.era_pool = era_pool  # None: each era alone; a float: ridge toward the pooled model of all eras (inf: pooled)
        self.era_scale = era_scale  # feature scale; None: rms of the first chunk's newest block (like y_edges)
        self.holdout = holdout  # "interleaved" (every 5th window) or "trajectory" (L36 R1: only calibrate() feeds the pool)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        return tags

    @staticmethod
    def make_edges(y, n_bins=8):
        """Quantile bin edges (p, n_bins + 1) of the output; pass as y_edges for exact chunked fitting."""
        y = np.asarray(y, float)
        y = y.reshape(len(y), -1)
        q = np.linspace(0, 1, n_bins + 1)
        return np.stack([np.quantile(y[:, j], q) for j in range(y.shape[1])])

    @staticmethod
    def make_step_edges(X, n_bins=8, n_lags=1):
        """Bin edges of the increments x_{t+1} - x_t of a time-ordered array (T, D) or of trajectories (n, T, D).
        With y=None the TT and the eras model the increment, so y_edges refers to increments. With n_lags > 1 the
        rows are lag stacks [x_t, x_{t-1}, ...] and the increment is that of the newest block."""
        X = np.asarray(X, float)
        X = X[..., :X.shape[-1] // n_lags]
        inc = np.diff(X, axis=0) if X.ndim == 2 else np.concatenate([np.diff(t, axis=0) for t in X])
        return FCPARegressor.make_edges(inc, n_bins)

    # ------------------------------------------------------------------ frame (data independent)
    def _init_frame(self, d, rng):
        N = max(2 * int(self.d_chart), (d + 2) // 2)  # 2N >= d + 1: the linear encoder keeps the state (E1)
        g = np.random.default_rng(rng.randint(2**31 - 1))
        self._N = N
        # a real 2N x (d+1) matrix with orthonormal columns: phi_real = Q [x; 1] is an isometry (condition number of
        # the state itself, not that of a random Gaussian map, which is ~1/N for a square one), R = Q_re + i Q_im
        Q = np.linalg.qr(g.standard_normal((2 * N, d + 1)))[0]
        self._R = Q[:N] + 1j * Q[N:]
        self._V = parseval_frame(N, N, g)
        self._Rinv = np.linalg.pinv(np.r_[self._R.real, self._R.imag])  # (d + 1, 2N)
        self._gens = diff.make_generators(N, 1 + 2 + _S * _S, g, commuting=True)

    def _encode(self, X):
        phi = np.c_[X, np.ones(len(X))] @ self._R.T
        a = np.linalg.norm(phi, axis=1, keepdims=True)
        psi = phi / a
        sym = np.argmax(born(self._V, psi), axis=1) * _S // (self._N + 1)
        return psi, np.c_[phi.real, phi.imag], sym

    # ------------------------------------------------------------------ streaming state
    def _validate(self, X, y, first):
        kw = dict(reset=first, dtype=[np.float64, np.float32], ensure_min_samples=1)
        if y is None:
            X = validate_data(self, X, **kw)
        else:
            X, y = validate_data(self, X, y, y_numeric=True, multi_output=False, **kw)
        return np.asarray(X, float), (None if y is None else np.asarray(y, float))

    def fit(self, X, y=None, t=None):
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if self.holdout not in ("interleaved", "trajectory"):
            raise ValueError("holdout must be 'interleaved' or 'trajectory'")
        for a in ("_st", "_model"):
            self.__dict__.pop(a, None)
        return self.partial_fit(X, y, t)

    @_single_blas
    def partial_fit(self, X, y=None, t=None):
        first = not hasattr(self, "_st")
        X, y = self._validate(X, y, first)
        if t is not None:
            t = np.asarray(t, float).ravel()
            if len(t) != len(X) or not np.all(np.isfinite(t)):
                raise ValueError("t must be finite with one entry per row of X")
            o = np.argsort(t, kind="stable")
            X, y = X[o], (None if y is None else y[o])
        if y is None and X.shape[1] % self.n_lags:
            raise ValueError(f"n_lags={self.n_lags} must divide the number of features {X.shape[1]}")
        if first:
            self._start(X, y)
        st = self._st
        if y is None:  # next-step pairs; the pair across the chunk boundary uses the stored last row
            full = X if st.last_x is None else np.vstack([st.last_x, X])
            st.last_x = full[-1:].copy()
            Db = full.shape[1] // self.n_lags  # rows are [x_t, x_{t-1}, ...]; the target is the newest block's increment
            X, Y = full[:-1], np.diff(full[:, :Db], axis=0)  # persistence is the zero model
        else:
            Y = y.reshape(len(y), -1)
        if Y.shape[1] != st.tt.sumy.shape[0]:
            raise ValueError("output dimension changed between calls")
        st.n_pairs += len(X)
        st.bufX, st.bufY = np.vstack([st.bufX, X]), np.vstack([st.bufY, Y])
        w = self.window
        while len(st.bufX) >= w:
            self._consume(st, st.bufX[:w], st.bufY[:w], flush=False)
            st.bufX, st.bufY = st.bufX[w:], st.bufY[w:]
        self._finalize()
        return self

    def _start(self, X, y):
        d = X.shape[1]
        self._feat = None
        if self.era not in ("phi", "lin_vq", "quad_vq"):
            raise ValueError("era must be 'phi', 'lin_vq' or 'quad_vq'")
        if self.era != "phi":
            if y is not None or not hasattr(self.law, "modes"):
                raise ValueError(f"era={self.era!r} needs y=None and a law with a symmetry action (law.modes())")
            scale = self.era_scale or ModeFeatures.rms(self.law, X[:, :d // self.n_lags])
            self._feat = ModeFeatures(self.law, self.n_lags, self.era == "quad_vq", scale)
        rng = check_random_state(self.random_state)
        self._init_frame(d, rng)
        if self.y_edges is not None:
            edges = np.atleast_2d(np.asarray(self.y_edges, float))
            p = edges.shape[0]
        else:
            Yfit = np.diff(X[:, :d // self.n_lags], axis=0) if y is None else y.reshape(len(y), -1)
            if len(Yfit) == 0:
                raise ValueError("y=None needs at least 2 rows (n_samples = 1 has no next step)")
            edges = self.make_edges(Yfit, self.n_bins)
            p = Yfit.shape[1]
        self._edges = edges
        self._st = _State(d, p, edges.shape[1] - 1, self._N, self.alpha, self.gamma)
        self._st.y_vec = y is not None

    def _drop_buffer(self):
        """Test hook: forget the open window (what an implementation without the L27' buffer does)."""
        st = self._st
        st.bufX, st.bufY = st.bufX[:0], st.bufY[:0]
        st.hist, st.last_sym = [], None
        self._finalize()

    # ------------------------------------------------------------------ window consumption
    @staticmethod
    def _d2(g, phi):
        mu, P, _ = g
        z = phi - mu
        return np.sum((z @ P) * z, axis=1)

    def _hit(self, g, phi):
        """Fraction of rows the snapshot Gaussian explains: Mahalanobis d2 <= 2 df. In [0, 1]."""
        return float(np.mean(self._d2(g, phi) <= 2.0 * g[2]))

    def _changed(self, st, phi):
        """L11 on a bounded statistic: window j vs windows j-1, j-2, all scored by ONE snapshot taken before
        j-2 (independent of the three windows). Disjoint Hoeffding intervals => CHANGE."""
        if len(st.hist) < 2 or st.hist[0][1] is None:
            return False
        g = st.hist[0][1]
        r = np.sqrt(np.log(4.0 / _DELTA) / (2 * len(phi)))  # delta/2 per comparison, two comparisons
        u = self._hit(g, phi)
        for old, _ in st.hist:
            v = self._hit(g, old)
            if bounds.certified_change((np.array([v - r]), np.array([v + r])),
                                       (np.array([u - r]), np.array([u + r]))) == bounds.CHANGE:
                return True
        return False

    def _held_out(self, j, flush):
        return self.holdout == "interleaved" and not flush and j % _CAL_EVERY == _CAL_EVERY - 1

    def _consume(self, st, X, Y, flush):
        if self.law is not None:
            return self._consume_law(st, X, Y, flush)
        psi, phi, sym = self._encode(X)
        j = st.window_idx
        if not flush and len(st.segs) < self.n_eras_max and self._changed(st, phi):
            st.segs.append(_Seg(self._N))
            st.hist = []
        seg = st.segs[-1]
        snap = seg.gauss()
        seq = sym if st.last_sym is None else np.r_[st.last_sym, sym]
        np.add.at(seg.counts, (seq[:-1], seq[1:]), 1)  # includes the cross-window / cross-chunk pair
        is_cal = self._held_out(j, flush)
        if is_cal:
            self._cal_window(st, phi, len(st.segs) - 1)
            st.cal_X = np.vstack([st.cal_X, X])[-_CAL_MAX:]
            st.cal_Y = np.vstack([st.cal_Y, Y])[-_CAL_MAX:]
        else:
            seg.rho += psi.T @ psi.conj()
            seg.n += len(X)
            self._era_stats(seg, X, Y, phi)
            self._ratios(st, seg, X, Y)
            st.tt.add(X, Y, self._edges)
        if not flush:
            st.hist = (st.hist + [(phi, snap)])[-2:]
            st.last_sym, st.window_idx = int(sym[-1]), j + 1

    # ---- law mode: eras are law classes, found and scored by the residual of R = nu L + f
    @staticmethod
    def _law_nc(model, R, L, nll=False):
        """Per-row normalised residual c = |R - nu L - f|^2 / sigma2 (about 1 under the era's own law), or with
        nll=True the Gaussian negative log-likelihood per coordinate: c/2 + log(2 pi sigma2 / m) / 2. The
        likelihood is what compares laws (a broad law makes every window look consistent); c tests consistency."""
        nu, f, s2 = model
        c = ((R - nu * L - f) ** 2).sum(1) / s2
        return c / 2 + 0.5 * np.log(2 * np.pi * s2 / R.shape[1]) if nll else c

    def _rows_from_phi(self, phi):
        return (phi @ self._Rinv.T)[:, :-1]  # E1 is injective: the stacked state rows come back exactly

    def _ratios(self, st, seg, X, Y):
        if st.y_vec:
            return
        Db = Y.shape[1]
        cur = X[:, :Db]
        r = np.linalg.norm(cur + Y, axis=1) / np.maximum(np.linalg.norm(cur, axis=1), 1e-300)
        seg.gs += float(r.sum())
        seg.gn += len(r)

    def _add_to_seg(self, st, seg, idx, X, Y, psi, phi, sym, R, L, cal):
        seq = sym if st.last_sym is None else np.r_[st.last_sym, sym]
        np.add.at(seg.counts, (seq[:-1], seq[1:]), 1)
        if cal:
            self._cal_window(st, phi, idx)
            st.cal_X = np.vstack([st.cal_X, X])[-_CAL_MAX:]
            st.cal_Y = np.vstack([st.cal_Y, Y])[-_CAL_MAX:]
        else:
            seg.rho += psi.T @ psi.conj()
            seg.n += len(X)
            self._era_stats(seg, X, Y, phi)
            self._ratios(st, seg, X, Y)
            seg.law.add(R, L)
            st.tt.add(X, Y, self._edges)

    def _era_stats(self, seg, X, Y, phi):
        """Add one training window to the era's statistics; _finalize re-solves from them (set_chart)."""
        seg.era.partial_fit(phi, Y, solve=False)
        if self._feat is not None:
            if seg.xera is None:
                seg.xera = SymEraModel(self.era_lam, self._feat.desc, self._feat.ydesc)
            seg.xera.partial_fit(self._feat(X), Y, solve=False)

    def _xera_model(self, xera, pooled):
        """The answering L37 era: the era's own statistics; with era_pool, shrunk toward the pooled model."""
        if pooled is None:
            return xera
        em = copy.copy(xera)  # shares the statistics; the solve (and its cache) lands on the copy
        em.prior, em.pool = pooled, float(self.era_pool)
        return em

    def _era_in(self, X, phi):
        """The era models' input: phi, or the L37 mode features of the rows X."""
        return phi if getattr(self, "_feat", None) is None else self._feat(X)

    def _consume_law(self, st, X, Y, flush):
        psi, phi, sym = self._encode(X)
        R, L = self.law(X)
        j = st.window_idx
        cal = self._held_out(j, flush)
        win = (X, Y, psi, phi, sym, R, L, cal)
        k = self._best_law(st.segs, R, L)
        if k >= 0:  # a known law explains the window
            self._add_to_seg(st, st.segs[k], k, *win)
        elif not flush:
            st.nov.append(win)
            self._try_birth(st)
        if not flush:
            st.last_sym, st.window_idx = int(sym[-1]), j + 1

    def _best_law(self, segs, R, L):
        """Index of the era whose law is consistent with the window (c <= _TAU) and has the best likelihood; -1 if none."""
        best, k = np.inf, -1
        for i, sg in enumerate(segs):
            m = sg.law.model()
            if m is not None and np.mean(self._law_nc(m, R, L)) <= _TAU and (v := np.mean(self._law_nc(m, R, L, True))) < best:
                best, k = v, i
        return k

    def _try_birth(self, st):
        """RSL birth (L23/L24): _N_BIRTH unexplained windows whose first two fit a law that explains the last."""
        if len(st.nov) < _N_BIRTH:
            return
        fit = _LawStats()
        for w in st.nov[:-1]:
            fit.add(w[5], w[6])
        m = fit.model()
        last = st.nov[-1]
        free = len(st.segs) < self.n_eras_max or st.segs[0].n == 0
        if m is not None and np.mean(self._law_nc(m, last[5], last[6])) <= _TAU and free:
            if st.segs[0].n == 0:
                idx = 0
            else:
                st.segs.append(_Seg(self._N))
                idx = len(st.segs) - 1
            for w in st.nov:
                self._add_to_seg(st, st.segs[idx], idx, *w)
            st.nov = []
        else:
            st.nov.pop(0)

    def hmm_filter(self, nc, stay=0.995):
        """L14' tie-break: a forward HMM filter over eras (stay prob `stay`, else uniform) on per-row scores nc
        (n, K) along ONE time-ordered stream. Returns -log filtered posterior, so argmin is the filtered era."""
        nc = np.asarray(nc, float)
        K = nc.shape[1]
        A = np.full((K, K), (1 - stay) / max(K - 1, 1)) + np.eye(K) * (stay - (1 - stay) / max(K - 1, 1))
        a, out = np.full(K, 1.0 / K), np.empty_like(nc)
        for t, row in enumerate(nc):
            a = (a @ A) * np.exp(-0.5 * (row - row.min()))
            a /= a.sum()
            out[t] = -np.log(np.maximum(a, 1e-300))
        return out

    def _nc_scores(self, segs, phi):
        """Nonconformity of each row to each era: law mode: normalised law residual; else d2_k / df_k of the era's
        Gaussian. An empty era scores 1e9."""
        if self.law is not None:
            out = np.full((len(phi), len(segs)), 1e9)
            R, L = self.law(self._rows_from_phi(phi))
            for k, sg in enumerate(segs):
                if (m := sg.law.model()) is not None:
                    out[:, k] = self._law_nc(m, R, L, True)
            return out
        out = np.full((len(phi), len(segs)), 1e9)
        for k, sg in enumerate(segs):
            g = sg.gauss()
            if g is not None:
                out[:, k] = self._d2(g, phi) / g[2]
        return out

    def _cal_window(self, st, phi, label):
        """Online ACI step on a held-out window, then append it to the calibration set."""
        if len(st.cal_lab) and any(sg.n for sg in st.segs):
            cp = conformal.SplitConformal(self.alpha).fit(self._nc_scores(st.segs, st.cal_phi), st.cal_lab)
            for row in self._nc_scores(st.segs, phi):
                err = int(label not in cp.predict_set(row, alpha=st.aci.alpha_t))
                f = st.aci.forced_err()
                st.aci.update(err if f is None else f)
        st.cal_phi = np.vstack([st.cal_phi, phi])[-_CAL_MAX:]
        st.cal_lab = np.r_[st.cal_lab, np.full(len(phi), label)][-_CAL_MAX:]
        st.cal_traj = np.r_[st.cal_traj, np.full(len(phi), -1)][-_CAL_MAX:]

    @_single_blas
    def calibrate(self, X):
        """L36 R1 (Amendment 11c): held-out trajectories, (n, L, d) or one (L, d), never seen by partial_fit,
        calibrate the conformal sets, the ACI state and the skill gate. Each full window a live era explains joins
        the pool (uncapped) with that era's label; a window no era explains and each part-filled tail are dropped.
        No era statistic, TT sum, change test or birth reads these windows."""
        check_is_fitted(self, "_model")
        if self.holdout != "trajectory":
            raise ValueError("calibrate needs holdout='trajectory' (the interleaved rule already holds windows out)")
        st = self._st
        if self.law is None or st.y_vec:
            raise ValueError("calibrate needs law mode and y=None")
        Xs = np.asarray(X, float)
        Xs = Xs[None] if Xs.ndim == 2 else Xs
        w, Db = self.window, self.n_features_in_ // self.n_lags
        # eras are fixed while calibrating: each pool row is scored once
        pool = self._nc_scores(st.segs, st.cal_phi) if len(st.cal_lab) else np.zeros((0, len(st.segs)))
        for traj in Xs:
            traj = np.asarray(validate_data(self, traj, reset=False, dtype=[np.float64, np.float32]), float)
            Xp, Yp = traj[:-1], np.diff(traj[:, :Db], axis=0)
            tid, st.n_cal_traj = st.n_cal_traj, st.n_cal_traj + 1
            for a in range(0, len(Xp) - w + 1, w):
                Xw, Yw = Xp[a:a + w], Yp[a:a + w]
                k = self._best_law(st.segs, *self.law(Xw))
                if k < 0:
                    continue
                phi = self._encode(Xw)[1]
                nc = self._nc_scores(st.segs, phi)
                if len(st.cal_lab) and any(sg.n for sg in st.segs):  # the online ACI step of _cal_window
                    cp = conformal.SplitConformal(self.alpha).fit(pool, st.cal_lab)
                    for row in nc:
                        err = int(k not in cp.predict_set(row, alpha=st.aci.alpha_t))
                        f = st.aci.forced_err()
                        st.aci.update(err if f is None else f)
                pool = np.vstack([pool, nc])
                st.cal_phi, st.cal_lab = np.vstack([st.cal_phi, phi]), np.r_[st.cal_lab, np.full(w, k)]
                st.cal_traj = np.r_[st.cal_traj, np.full(w, tid)]
                st.cal_X, st.cal_Y = np.vstack([st.cal_X, Xw]), np.vstack([st.cal_Y, Yw])
        self._finalize()
        return self

    # ------------------------------------------------------------------ derived models
    @staticmethod
    def _gamma(Y, P):
        """Least-squares shrink of the prediction P toward the zero model, clipped to [0, 1]."""
        den = float((P * P).sum())
        if den <= 0:
            return 1.0
        g = float(np.clip((Y * P).sum() / den, 0.0, 1.0))
        base = float((Y * Y).sum())
        gain = 1.0 - float(((Y - g * P) ** 2).sum()) / base if base > 0 else 0.0
        return g if gain > _MIN_GAIN else 0.0  # a gain inside the validation noise is not evidence

    def _shrink(self, st, act, eras, tt):
        """y=None: the zero increment (persistence) is the baseline. Each era's increment model, and the TT's, is
        shrunk toward it by the factor that is best on the held-out calibration rows (1 if there are none)."""
        gam, gam_tt = np.ones(len(act)), 1.0
        if st.y_vec or not len(st.cal_lab):
            return gam, gam_tt
        pos = {k: i for i, k in enumerate(act)}
        lab = np.array([pos.get(l, -1) for l in st.cal_lab])
        P = np.zeros_like(st.cal_Y)
        Z = self._era_in(st.cal_X, st.cal_phi)
        for i, em in enumerate(eras):
            P[lab == i] = em.predict(Z[lab == i])
        pooled = self._gamma(st.cal_Y[lab >= 0], P[lab >= 0]) if (lab >= 0).any() else 1.0
        for i in range(len(act)):
            gam[i] = self._gamma(st.cal_Y[lab == i], P[lab == i]) if (lab == i).sum() >= 10 else pooled
        T = np.einsum("npb,pb->np", tt.proba(st.cal_X, self.temp), tt.centers())
        return gam, self._gamma(st.cal_Y, T)

    def _crossing(self, law0, lawk):
        dE = diff.delta(law0, lawk)
        return diff.crossing_operator(diff.DeltaE(_CROSS * dE.vec), self._gens)

    def _finalize(self):
        """Models derived from a COPY of the state with the open buffer flushed into the open era."""
        # the flush only mutates the open segment and the TT sums; everything else is shared, read-only
        st = copy.copy(self._st)
        st.segs = self._st.segs[:-1] + [copy.deepcopy(self._st.segs[-1])]
        st.tt = copy.deepcopy(self._st.tt)
        if len(st.bufX):
            self._consume(st, st.bufX, st.bufY, flush=True)
        total = sum(sg.n for sg in st.segs)
        big = max(_MIN_ERA_ROWS, _MIN_ERA_FRAC * total, 3 * 2 * self._N)  # and 3 rows per ambient feature  # a newborn with a handful of rows is in every conformal set (L6')
        act = [k for k, sg in enumerate(st.segs) if sg.n >= big]  # and would drag every blend: it waits for data
        if not act and total:
            act = [int(np.argmax([sg.n for sg in st.segs]))]
        # a lag stack hides the velocity in a tiny difference of two large blocks: the chart must not cut it
        d_c = self._N if self.n_lags > 1 else min(int(self.d_chart), self._N)
        laws = [diff.Law(P=(s.counts + 1) / (s.counts + 1).sum(1, keepdims=True), nu=0.0, f=np.zeros(2))
                for s in st.segs]
        eras = []
        xs = [s.xera for s in st.segs if s.xera is not None and s.xera.n_]
        pooled = SymEraModel.pooled(xs, self.era_lam) if xs and self.era_pool is not None else None
        for k in act:
            s = st.segs[k]
            S = self._crossing(laws[act[0]], laws[k])
            B = era_chart(s.rho / s.n, d_c).conj().T @ S  # z = V^H S psi, a z = B phi
            M = np.block([[B.real, -B.imag], [B.imag, B.real]])
            em = copy.copy(s.era) if s.xera is None else self._xera_model(s.xera, pooled)  # set_chart rebinds chart/W_/b_ and only reads the (never mutated in place) statistics
            em.set_chart(M)
            eras.append(em)
        pos = {k: i for i, k in enumerate(act)}
        keep = np.array([l in pos for l in st.cal_lab], bool)
        cal = cal_ood = None
        if keep.any():
            lab = np.array([pos[l] for l in st.cal_lab[keep]])
            sc = self._nc_scores([st.segs[k] for k in act], st.cal_phi[keep])
            cal = conformal.SplitConformal(self.alpha).fit(sc, lab)
            # OOD is declared only beyond this much stricter level: an empty alpha-set alone is ~alpha of the
            # in-distribution rows by construction and must not discard the era (e7).
            cal_ood = conformal.SplitConformal(self.alpha * _OOD_FRAC).fit(sc, lab)
            if self.holdout == "trajectory":  # 11c item 4: n_min counts Cal trajectories, not rows
                for i, k in enumerate(act):
                    if len(np.unique(st.cal_traj[st.cal_lab == k])) < cal.n_min_:
                        for cp in (cal, cal_ood):  # a newborn: no calibration points, in every set
                            cp.sorted_[i], cp.n_cal_[i], cp.quantiles_[i] = np.zeros(0), 0, np.inf
        tt = st.tt
        gam, gam_tt = self._shrink(st, act, eras, tt)
        g = np.array([st.segs[k].gs / st.segs[k].gn if st.segs[k].gn else 1.0 for k in act])
        g_all = sum(sg.gs for sg in st.segs) / max(sum(sg.gn for sg in st.segs), 1) or 1.0
        self._model = dict(gam=gam, gam_tt=gam_tt, g=g, g_all=g_all, eras=eras, segs=[st.segs[k] for k in act], act=act, n=np.array([st.segs[k].n for k in act], float), cal=cal, cal_ood=cal_ood,
                           tt=tt, centers=tt.centers(), alpha_t=st.aci.alpha_t)
        self.n_eras_ = len(act)
        self.shrink_ = gam
        self.shrink_tt_ = gam_tt
        self.law_counts_ = [st.segs[k].counts.copy() for k in range(len(st.segs))]
        self.y_edges_ = self._edges
        self.n_samples_seen_ = st.n_pairs
        self.alpha_t_ = st.aci.alpha_t

    # ------------------------------------------------------------------ inference
    def _check(self, X):
        check_is_fitted(self, "_model")
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        return np.asarray(X, float)

    @_single_blas
    def tt_proba(self, X):
        """TT softmax distribution over output bins: (n, B) for one output, (n, p, B) otherwise."""
        P = self._model["tt"].proba(self._check(X), self.temp)
        return P[:, 0] if P.shape[1] == 1 else P

    def _tt_pred(self, X):
        m = self._model
        return np.einsum("npb,pb->np", m["tt"].proba(X, self.temp), m["centers"])

    @_single_blas
    def predict_tt(self, X):
        X = self._check(X)
        return self._out(X, self._tt_pred(X) * self._dmp() * self._model["gam_tt"])

    def _y_1d(self):
        return self._st.y_vec

    def _dmp(self):
        """Scale on the validated increments (y=None only): below 1 trades one-step error for a stabler rollout."""
        return 1.0 if self._y_1d() else self.damping

    def _out(self, X, out):
        """Regression: the prediction. y=None: the exact skip path x + increment (persistence is the zero model)."""
        if self._y_1d():
            return out[:, 0]
        Db = out.shape[1]
        if self.n_lags == 1:
            return X + out
        return np.hstack([X[:, :Db] + out, X[:, :-Db]])  # the next lag stack: [x_t + dx, x_t, ..., x_{t-L+2}]

    def _route(self, X, nc=None):
        m = self._model
        if nc is None:
            nc = self._nc_scores(m["segs"], self._encode(X)[1])
        R = 1.0 / np.sqrt(m["n"])
        n, K = nc.shape
        if m["cal"] is None:
            Cs = [{int(k)} for k in np.argmin(nc, axis=1)]
        else:  # SplitConformal.predict_set for every row at once: one quantile vector, one comparison
            q = self._quantiles(m["cal"], m["alpha_t"])
            Cs = [{int(k) for k in np.flatnonzero(r)} for r in (nc <= q[None, :])]
            if m["cal_ood"] is not None:  # empty at alpha but inside the OOD level: nearest admitted era
                q2 = self._quantiles(m["cal_ood"], None)
                for i in np.flatnonzero([not C for C in Cs]):
                    adm = np.flatnonzero(nc[i] <= q2)
                    if len(adm):
                        Cs[i] = {int(adm[np.argmin(nc[i, adm])])}
        # certified_topm(D, R, 1): the top-1 is certified iff lo[top] > max_{j != top} hi[j] (K == 1: always)
        D = np.exp(-nc / 2.0)
        rows = np.arange(n)
        top = np.argmax(D, axis=1)
        lo, hi = D - R, D + R
        hi[rows, top] = -np.inf
        cert = np.ones(n, bool) if K == 1 else lo[rows, top] > hi.max(axis=1)
        return [decide(C, {int(top[i])} if cert[i] else None) for i, C in enumerate(Cs)]

    @staticmethod
    def _quantiles(cp, alpha):
        if alpha is None:
            return cp.quantiles_
        return np.array([conformal._quantile(a, cp.n_min_, alpha) for a in cp.sorted_])

    @_single_blas
    def route(self, X):
        """Router decisions (one Decision per row; eras index the active eras)."""
        return self._route(self._check(X))

    @_single_blas
    def era_predict(self, k, X):
        """Era k's JEPA prediction (through its chart and crossing operator) for the rows of X."""
        X = self._check(X)
        _, phi, _ = self._encode(X)
        return self._out(X, self._dmp() * self._model["gam"][k] * self._model["eras"][k].predict(self._era_in(X, phi)))

    def _blend(self, eras, preds):
        """Weights over eras from the damped CTC loop on the agreement map; uniform if the loop refuses."""
        Y = np.stack(preds)  # (E, p)
        s2 = Y.var() + 1e-12

        def G(w):
            d = ((Y - w @ Y) ** 2).sum(1) / s2
            e = np.exp(-(d - d.min()) / self.temp)
            return e / e.sum()

        def Gb(W):  # G on every row of W (B, E) at once
            d = ((Y[None] - (W @ Y)[:, None, :]) ** 2).sum(2) / s2
            e = np.exp(-(d - d.min(1, keepdims=True)) / self.temp)
            return e / e.sum(1, keepdims=True)

        G.batch = Gb

        w, _, ok = damped_loop(G, np.full(len(eras), 1.0 / len(eras)), self.eps, None, 1e-9, max_iter=500)
        return w if ok else np.full(len(eras), 1.0 / len(eras))

    @_single_blas
    def predict(self, X):
        X = self._check(X)
        m = self._model
        out = self._tt_pred(X) * (self._dmp() * m["gam_tt"])
        _, phi, _ = self._encode(X)
        nc = self._nc_scores(m["segs"], phi)
        Z = self._era_in(X, phi)
        decs = self._route(X, nc)
        n_ood = sum(d.kind == OOD for d in decs)
        if n_ood:
            who = "the nearest era (extrapolating)" if self.ood == "nearest_era" else "the TT alone"
            warnings.warn(f"FCPARegressor: {n_ood} of {len(X)} rows are OOD (no era admits them at level "
                          f"{self.alpha * _OOD_FRAC:g}); answered by {who}", UserWarning, stacklevel=2)
        grow, used = np.full(len(X), m["g_all"]), np.zeros(len(X), bool)
        for i, dec in enumerate(decs):
            eras = dec.eras
            if dec.kind == OOD and self.ood == "nearest_era":
                eras = (int(np.argmin(nc[i])),)
            if dec.kind == ALONE or not eras:
                continue
            if not self._y_1d():  # increments: an era with no validated skill (shrink 0) is persistence, not a member
                eras = tuple(k for k in eras if m["gam"][k] > 0)
                if not eras:
                    out[i] = 0.0
                    continue
            preds = [self._dmp() * m["gam"][k] * m["eras"][k].predict(Z[i:i + 1])[0] for k in eras]
            w = np.ones(1) if len(preds) == 1 else self._blend(eras, preds)
            out[i] = w @ np.stack(preds)
            grow[i], used[i] = w @ m["g"][list(eras)], True
        if self.energy and not self._y_1d() and used.any():
            # physics: the flow dissipates at a rate the era's slab pairs fix, |x_{t+1}| = g_k |x_t|; the linear
            # surrogate does not know it and gains energy in closed loop. Project the newest block onto that sphere.
            Db = out.shape[1]
            cur = X[used, :Db]
            nxt = cur + out[used]
            tgt = grow[used] * np.linalg.norm(cur, axis=1)
            nrm = np.linalg.norm(nxt, axis=1)
            out[used] = nxt * (tgt / np.maximum(nrm, 1e-300))[:, None] - cur
        return self._out(X, out)

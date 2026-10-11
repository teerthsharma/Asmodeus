"""Bounds (spec 3.3, E2', L-GIVEN): NS identifiability, Markov kernel intervals, certified change, segmentation.

Lemma map: L8 nu_projection, L9 nu_error_bound, E2' within_slab_estimate, L10 kernel_interval,
L11 certified_change, segment / segment_markov, L-GIVEN library / learn_operator / certify_sparse.
numpy only.
"""
from typing import NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from fcpa.beds import ns2d

CHANGE, NO_CHANGE_UNCERTIFIED, REFUSE = "CHANGE", "NO_CHANGE_UNCERTIFIED", "REFUSE"
PRESENT, SMALL = "PRESENT", "SMALL"


# --------------------------------------------------------------------------- L8 / L9 / E2'
def nu_projection(omega_t, adv, lap, f, axis=None):
    """L8 (NS identifiability, known f). `omega_t` is d_t w. Orthogonal projection of R = w_t + u.grad w - f
    onto lap w: nu = <R, lap w> / ||lap w||^2, exact when w solves the vorticity equation and lap w != 0.
    `axis` picks the pooled axes (None = everything)."""
    r = omega_t + adv - f
    return np.sum(r * lap, axis=axis) / np.sum(lap * lap, axis=axis)


def nu_error_bound(e, lap):
    """L9: if the residual carries error e then |nu_hat - nu| <= ||e|| / ||lap w|| (Cauchy-Schwarz)."""
    return np.linalg.norm(e) / np.linalg.norm(lap)


class SlabEstimate(NamedTuple):
    verdict: str   # "OK" or REFUSE
    nu: float
    f: np.ndarray  # estimated time-constant forcing (None on REFUSE)
    denom: float   # ||lap~||_F, the demeaned Laplacian norm

    def interval(self, e_norm):
        """E2' certified interval [nu - e/denom, nu + e/denom] for a residual error of Frobenius norm <= e_norm."""
        r = e_norm / self.denom
        return np.array([self.nu - r]), np.array([self.nu + r])


def within_slab_estimate(dwdt, adv, lap, axis=0, rel_tol=1e-8):
    """E2' (explicit within-slab estimator). R = w_t + u.grad w = nu*lap + f with f constant along `axis`
    (time, or the universe axis of a multiverse). Demean along `axis`:
        nu_hat = sum <R~, lap~> / sum ||lap~||^2,   |nu_hat - nu| <= ||e||_F / ||lap~||_F,
        f_hat  = mean(R) - nu_hat * mean(lap).
    Identifiability (E2, an "if"): with f time-constant, nu is identifiable iff lap w(t) is not
    time-constant; if lap~ = 0 (steady slab) return verdict REFUSE. Otherwise (nu', f + (nu-nu') lap w)
    reproduces the same w."""
    R = dwdt + adv
    Rt, Lt = R - R.mean(axis, keepdims=True), lap - lap.mean(axis, keepdims=True)
    denom = float(np.linalg.norm(Lt))
    if denom <= rel_tol * max(float(np.linalg.norm(lap)), 1e-300):
        return SlabEstimate(REFUSE, np.nan, None, denom)
    nu = float(np.sum(Rt * Lt) / denom ** 2)
    return SlabEstimate("OK", nu, R.mean(axis) - nu * lap.mean(axis), denom)


def era_signal(dwdt, adv, lap, kmax=3):
    """Per-step law-coordinate signal (T, d) for segmentation: [nu_hat_t, low-mode Fourier coefs of f_hat_t].
    Pools the multiverse axis (universe 0) where f is common: dwdt/adv/lap have shape (U, T, n, n), U >= 2."""
    T, n = dwdt.shape[1], dwdt.shape[-1]
    kx = np.r_[0:kmax + 1, n - kmax:n]
    ky = np.r_[0:kmax + 1, n - kmax:n]
    rows = []
    for t in range(T):
        est = within_slab_estimate(dwdt[:, t], adv[:, t], lap[:, t])
        if est.verdict == REFUSE:
            raise ValueError(f"step {t}: universes have identical lap w; cannot identify nu")
        fh = np.fft.fft2(est.f)[np.ix_(kx, ky)].ravel() / n
        rows.append(np.r_[est.nu, fh.real, fh.imag])
    return np.array(rows)


# --------------------------------------------------------------------------- L10 Markov kernel
def _counts(symbols, K1):
    s = np.asarray(symbols)
    C = np.zeros((K1, K1))
    np.add.at(C, (s[:-1], s[1:]), 1)
    return C


def markov_kernel(symbols, K1):
    """Laplace-smoothed row-stochastic transition estimate."""
    C = _counts(symbols, K1) + 1.0
    return C / C.sum(1, keepdims=True)


def _interval_from_counts(C, delta, m_max):
    """Per-entry radius r = min(Hoeffding, empirical Bernstein) at failure delta/(2*K1^2*m_max) each, jointly
    covering all K1^2 entries with probability >= 1-delta (union over entries, the two bounds, and the visit
    count m <= m_max). Rows with no visits are [0, 1]."""
    K1 = len(C)
    n = C.sum(1, keepdims=True)
    d1 = delta / (2 * K1 * K1 * m_max)
    ph = np.divide(C, n, out=np.zeros_like(C), where=n > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rh = np.sqrt(np.log(2 / d1) / (2 * n))
        v = ph * (1 - ph) * n / (n - 1)  # unbiased sample variance of a Bernoulli
        rb = np.sqrt(2 * v * np.log(4 / d1) / n) + 7 * np.log(4 / d1) / (3 * (n - 1))
    r = np.where(n > 1, np.minimum(rh, rb), rh)
    r = np.where(n > 0, r, 1.0)
    return np.clip(ph - r, 0, 1), np.clip(ph + r, 0, 1)


def kernel_interval(symbols, K1, delta):
    """L10 (CITED: Paulin 2015, McDiarmid / Bernstein type for Markov chains; Chung, Lam, Liu, Mitzenmacher
    2012, Chernoff-Hoeffding with L1 mixing time). Joint (P_lo, P_hi) box containing the true kernel with
    probability >= 1-delta. By the Markov property the successors of one state form an i.i.d. sample
    from its row, so per-row Hoeffding / empirical-Bernstein radii apply; the mixing time only controls how
    many visits n_i the window collects (L12), which is why the window must exceed it."""
    s = np.asarray(symbols)
    return _interval_from_counts(_counts(s, K1), delta, max(len(s) - 1, 1))


# --------------------------------------------------------------------------- L11 certified change
def certified_change(int_a, int_b):
    """L11. Intervals are (lo, hi) arrays (windows' joint boxes). Disjoint in any coordinate => CHANGE.
    Soundness: if both windows have the SAME true law (H0) and each joint box covers its true law with
    probability >= 1-delta, then P(false CHANGE) <= 2*delta (union bound). Overlap is REFUSE, never "no
    change". NO_CHANGE_UNCERTIFIED is returned only for coincident exact points (zero-width, equal boxes),
    which carry no probabilistic guarantee."""
    (la, ha), (lb, hb) = [(np.asarray(a, float), np.asarray(b, float)) for a, b in (int_a, int_b)]
    if np.any(la > ha) or np.any(lb > hb):
        raise ValueError("interval with lo > hi")
    if np.any((ha < lb) | (hb < la)):
        return CHANGE
    if np.array_equal(la, ha) and np.array_equal(lb, hb) and np.array_equal(la, lb):
        return NO_CHANGE_UNCERTIFIED
    return REFUSE


# --------------------------------------------------------------------------- segmentation
def _peaks(ts, stat, verdict_change, w):
    """Non-maximum suppression with exclusion radius w (mixed windows exist only within w of a switch)."""
    cand = [(stat[i], ts[i]) for i in range(len(ts)) if verdict_change[i]]
    kept = []
    for s, t in sorted(cand, key=lambda c: -c[0]):
        if all(abs(t - k) >= w for k in kept):
            kept.append(t)
    return sorted(int(k) for k in kept)


def segment(x, w, sigma=None, delta=1e-3, floor=1e-6, tol=1e-8):
    """Era boundaries of a per-step signal x (T,) or (T,d): t is a boundary when the certified intervals of
    [t-w, t) and [t, t+w) are disjoint (certified_change == CHANGE) and t is the strongest such t within w.
    Noise-free (sigma=None): window interval = [min, max] +- tol, so mixed windows overlap and a switch is
    recovered EXACTLY, provided eras last >= w. Noisy: interval = window mean +- z*sigma/sqrt(w) with
    z = sqrt(2 ln(2d/delta)); sigma is a per-coordinate noise sd or "auto" (MAD of first differences).
    Coordinates are scaled by their std (floored at floor*max std, so numerically-constant ones do not
    amplify rounding). Boundaries within w of either end of the series are unobservable (a full window on
    both sides is required)."""
    x = np.asarray(x, float)
    x = x[:, None] if x.ndim == 1 else x
    T, d = x.shape
    if T < 2 * w:
        return []
    s = x.std(0)
    if s.max() == 0:
        return []
    s = np.maximum(s, floor * s.max())
    z = x / s
    win = sliding_window_view(z, w, axis=0)  # (T-w+1, d, w)
    mean = win.mean(-1)
    if sigma is None:
        lo, hi, sig = win.min(-1) - tol, win.max(-1) + tol, np.ones(d)
    else:
        if isinstance(sigma, str):
            sig = np.median(np.abs(np.diff(z, axis=0) - np.median(np.diff(z, axis=0), axis=0)), axis=0) / (0.6745 * np.sqrt(2))
        else:
            sig = np.broadcast_to(np.asarray(sigma, float), (d,)) / s
        sig = np.maximum(sig, 1e-12)
        r = np.sqrt(2 * np.log(2 * d / delta)) * sig / np.sqrt(w) + tol
        lo, hi = mean - r, mean + r
    ts = np.arange(w, T - w + 1)
    change = np.zeros(len(ts), bool)
    stat = np.zeros(len(ts))
    for i, t in enumerate(ts):
        a, b = (lo[t - w], hi[t - w]), (lo[t], hi[t])
        change[i] = certified_change(a, b) == CHANGE
        stat[i] = np.max(np.abs(mean[t] - mean[t - w]) / (sig if sigma is not None else 1.0))
    return _peaks(ts, stat, change, w)


def segment_markov(symbols, K1, w, delta=1e-3):
    """Markov bound: boundary at t when kernel_interval boxes of the transitions in [t-w, t) and [t, t+w) are
    disjoint (CHANGE), strongest within w. Multiple-testing caveat: delta is per t, not per sequence."""
    s = np.asarray(symbols)
    T = len(s)
    C = np.zeros((T - 1, K1 * K1))
    C[np.arange(T - 1), s[:-1] * K1 + s[1:]] = 1
    cs = np.vstack([np.zeros((1, K1 * K1)), np.cumsum(C, 0)])
    ts = np.arange(w, T - w)
    change = np.zeros(len(ts), bool)
    stat = np.zeros(len(ts))
    for i, t in enumerate(ts):
        ca, cb = (cs[t] - cs[t - w]).reshape(K1, K1), (cs[t + w] - cs[t]).reshape(K1, K1)
        ia, ib = _interval_from_counts(ca, delta, w), _interval_from_counts(cb, delta, w)
        change[i] = certified_change(ia, ib) == CHANGE
        pa, pb = [np.divide(c, c.sum(1, keepdims=True), out=np.zeros_like(c), where=c.sum(1, keepdims=True) > 0) for c in (ca, cb)]
        stat[i] = np.abs(pa - pb).max()
    return _peaks(ts, stat, change, w)


# --------------------------------------------------------------------------- L-GIVEN learned operator
def library(omega, kf=4):
    """Candidate-term matrix (M, p) and names, no NS structure fixed: derivatives and products of w
    (1, w, lap w, lap2 w, w_x, w_y, u.grad w, w^2, w*lap w) plus a Fourier forcing dictionary
    cos/sin(kx x + ky y), (kx,ky) in the half plane |k_i| <= kf. omega: (..., n, n)."""
    n = omega.shape[-1]
    sp = ns2d._Spec(n)
    wh = sp.fft(omega) * sp.mask
    w = sp.ifft(wh)
    lap = sp.ifft(-sp.k2 * wh)
    terms = {"1": np.ones_like(w), "w": w, "lap w": lap, "lap2 w": sp.ifft(sp.k2 ** 2 * wh),
             "w_x": sp.ifft(1j * sp.kx * wh), "w_y": sp.ifft(1j * sp.ky * wh),
             "u.grad w": sp.ifft(sp.adv_hat(wh)), "w^2": w * w, "w lap w": w * lap}
    lead = w.shape[:-2]
    for kx in range(-kf, kf + 1):
        for ky in range(0, kf + 1):
            if ky == 0 and kx <= 0:
                continue
            ph = kx * sp.X + ky * sp.Y
            terms[f"cos({kx},{ky})"] = np.broadcast_to(np.cos(ph), w.shape)
            terms[f"sin({kx},{ky})"] = np.broadcast_to(np.sin(ph), w.shape)
    names = list(terms)
    return np.stack([terms[k].reshape(-1) for k in names], 1), names


def normalise_columns(A):
    nrm = np.linalg.norm(A, axis=0)
    nrm[nrm == 0] = 1.0
    return A / nrm, nrm


def stridge(A, y, tol=1e-6, lam=1e-12, iters=10):
    """Sequentially thresholded ridge (PDE-FIND / STRidge). A is column-normalised internally; a coefficient
    is dropped when its contribution |x_j| ||A_j|| < tol * ||y||. Returns coefficients in A's own units."""
    An, nrm = normalise_columns(A)
    thr = tol * np.linalg.norm(y)
    keep = np.ones(A.shape[1], bool)
    x = np.zeros(A.shape[1])
    for _ in range(iters):
        B = An[:, keep]
        x[:] = 0
        x[keep] = np.linalg.solve(B.T @ B + lam * np.eye(B.shape[1]), B.T @ y)
        new = keep & (np.abs(x) >= thr)
        if new.sum() == keep.sum():
            break
        keep = new
    x[~keep] = 0
    return x / nrm


def learn_operator(omega, dwdt, kf=4, tol=1e-6):
    """L-GIVEN arm: dwdt ~ library(omega) @ coef with sparse coef. Returns coef (dict over the support),
    support names, nu (coefficient of lap w), and the relative residual."""
    A, names = library(omega, kf)
    y = np.asarray(dwdt).reshape(-1)
    x = stridge(A, y, tol)
    sup = [n for n, c in zip(names, x) if c != 0]
    return dict(coef={n: c for n, c in zip(names, x) if c != 0}, support=sup, names=names, x=x,
                nu=float(x[names.index("lap w")]), rel_residual=float(np.linalg.norm(A @ x - y) / np.linalg.norm(y)))


def certify_sparse(A, y, e_norm, scale=None):
    """L-GIVEN certified sparse law. y = A x* + e with ||e|| <= e_norm, A in the units the caller wants
    (use normalise_columns for a library). x0 = lstsq(A, y), eta = e_norm / sigma_min(A), so
    |x0_j - x*_j| <= eta. Term j is PRESENT if |x0_j| > eta (then x*_j != 0); SMALL if |x0_j| <= eta (then
    |x*_j| <= 2 eta); REFUSE if A is rank-deficient, or if `scale` is given and 2*eta > scale for a
    non-PRESENT term (resolution too coarse to call it small). With the literal rule alone the middle band is
    empty; `scale` creates it. Returns dict(verdicts, eta, x0, certified_fraction)."""
    A = np.asarray(A, float)
    sv = np.linalg.svd(A, compute_uv=False)
    p = A.shape[1]
    if A.shape[0] < p or sv[-1] <= 1e-12 * sv[0]:
        return dict(verdicts=[REFUSE] * p, eta=np.inf, x0=None, certified_fraction=0.0)
    x0 = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)[0]
    eta = float(e_norm / sv[-1])
    v = [PRESENT if abs(c) > eta else REFUSE if scale is not None and 2 * eta > scale else SMALL for c in x0]
    return dict(verdicts=v, eta=eta, x0=x0, certified_fraction=sum(s != REFUSE for s in v) / p)

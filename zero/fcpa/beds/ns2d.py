"""NS-2D multiverse bed (spec section 5): periodic 2-D vorticity, pseudo-spectral, 2/3 dealiased, RK4.

    d_t w + u.grad w = nu * lap w + f,   u = (d_y psi, -d_x psi),   w = -lap psi,   domain [0, 2*pi)^2.

A "multiverse" is `n_universes` trajectories with independent random initial conditions that share
one law schedule (nu(t), f(t)). The law schedule is the ground truth: `eras[t]`, `boundaries`, `laws[k]`.

Forcing modes (E2): `f` is constant in time inside an era (or a finite-state Markov chain over
patterns), and lies in a few low Fourier modes, so lap w(t) leaves its subspace. Time-constant f gives
identifiability of nu iff lap w(t) is not constant in time (see bounds.within_slab_estimate).

Index convention: omega[u, t] is the state at the START of step t; step t -> t+1 uses (nu[t], f[t]).
With return_derivs=True, dwdt/adv/lap[u, t] are the solver's exact spectral terms at omega[u, t], so
dwdt + adv - nu[t]*lap - f[t] == 0 to rounding (L8 oracle).
"""
import numpy as np


class _Spec:
    def __init__(self, n):
        k = np.fft.fftfreq(n, 1.0 / n)
        self.n = n
        self.kx, self.ky = np.meshgrid(k, k, indexing="ij")
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.inv = np.divide(1.0, self.k2, out=np.zeros_like(self.k2), where=self.k2 > 0)
        self.mask = (np.abs(self.kx) <= n / 3) & (np.abs(self.ky) <= n / 3)  # 2/3 rule
        # half-spectrum (rfft2) versions for the stepper: last axis holds ky = 0 .. n/2
        self.nh = n // 2 + 1
        self.kxh, self.kyh = np.meshgrid(k, np.fft.rfftfreq(n, 1.0 / n), indexing="ij")
        self.k2h = self.kxh ** 2 + self.kyh ** 2
        self.invh = np.divide(1.0, self.k2h, out=np.zeros_like(self.k2h), where=self.k2h > 0)
        self.maskh = (np.abs(self.kxh) <= n / 3) & (np.abs(self.kyh) <= n / 3)  # same 2/3 rule
        x = 2 * np.pi * np.arange(n) / n
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")

    def fft(self, a):
        return np.fft.fft2(a, axes=(-2, -1))

    def ifft(self, ah):
        return np.fft.ifft2(ah, axes=(-2, -1)).real

    def adv_hat(self, wh):
        psi = wh * self.inv
        u, v = self.ifft(1j * self.ky * psi), self.ifft(-1j * self.kx * psi)
        wx, wy = self.ifft(1j * self.kx * wh), self.ifft(1j * self.ky * wh)
        return self.fft(u * wx + v * wy) * self.mask

    def rfft(self, a):
        return np.fft.rfft2(a, axes=(-2, -1))

    def irfft(self, ah):
        return np.fft.irfft2(ah, s=(self.n, self.n), axes=(-2, -1))

    def adv_hat_r(self, wh):
        """adv_hat on the rfft2 half-spectrum; equal to adv_hat restricted to ky >= 0 for real fields."""
        psi = wh * self.invh
        u, v = self.irfft(1j * self.kyh * psi), self.irfft(-1j * self.kxh * psi)
        wx, wy = self.irfft(1j * self.kxh * wh), self.irfft(1j * self.kyh * wh)
        return self.rfft(u * wx + v * wy) * self.maskh

    def pattern(self, p):
        """(kx, ky, amp[, phase]) -> amp*cos(kx x + ky y + phase); an (n,n) array passes through; None -> 0."""
        if p is None:
            return np.zeros((self.n, self.n))
        if isinstance(p, tuple):
            kx, ky, amp, *ph = p
            return amp * np.cos(kx * self.X + ky * self.Y + (ph[0] if ph else 0.0))
        return self.ifft(self.fft(np.asarray(p, float)) * self.mask)


def spectral_terms(omega):
    """(adv, lap) of a vorticity snapshot batch (..., n, n), the solver's own dealiased terms."""
    s = _Spec(omega.shape[-1])
    wh = s.fft(omega) * s.mask
    return s.ifft(s.adv_hat(wh)), s.ifft(-s.k2 * wh)


def encode(omega, kmax=4):
    """E1 encoder: truncated rfft (|kx|,ky <= kmax) as a unit complex vector z, plus the amplitude channel a=||z||.

    Normalising psi alone loses the amplitude, which confounds nu under w_c(x,t) = c w(x,ct) (nu' = c nu).
    """
    wh = np.fft.rfft2(omega, axes=(-2, -1))
    n = omega.shape[-1]
    kx = np.r_[0:kmax + 1, n - kmax:n]
    z = wh[..., kx, : kmax + 1].reshape(*omega.shape[:-2], -1)
    a = np.linalg.norm(z, axis=-1)
    return z / a[..., None], a


def _collapse(sched):
    """Sorted [(start, value)] with consecutive equal values dropped; scalar -> [(0, scalar)]."""
    if not isinstance(sched, list):
        sched = [(0, sched)]
    out = []
    for s, v in sorted(sched, key=lambda e: e[0]):
        if not out and s != 0:
            raise ValueError("schedule must start at step 0")
        if out and np.array_equal(np.asarray(out[-1][1], dtype=object), np.asarray(v, dtype=object)) \
                and _same(out[-1][1], v):
            continue
        if out and out[-1][0] == s:
            out[-1] = (s, v)
        else:
            out.append((s, v))
    return out


def _same(a, b):
    if a is None or b is None:
        return a is b
    try:
        return bool(np.array_equal(np.asarray(a, float), np.asarray(b, float)))
    except (TypeError, ValueError):
        return a == b


def _at(sched, t):
    v = sched[0][1]
    for s, x in sched:
        if s <= t:
            v = x
    return v


def markov_chain(P_sched, T, rng, hold=1):
    """Finite-state chain s_0..s_{T-1}; step t -> t+1 uses the kernel active at t; advances every `hold` steps."""
    rng = np.random.default_rng(rng)
    sched = _collapse(P_sched if isinstance(P_sched, list) else [(0, np.asarray(P_sched))])
    K = len(sched[0][1])
    s = np.empty(T, int)
    s[0] = rng.integers(K)
    for t in range(1, T):
        if (t - 1) % hold:
            s[t] = s[t - 1]
        else:
            s[t] = rng.choice(K, p=np.asarray(_at(sched, t - 1))[s[t - 1]])
    return s


CLASS_LAWS = [(0.02, (0, 4, 1.0)), (0.006, (3, 0, 1.0)), (0.012, (2, 2, 1.0))]


def regime_schedule(T, lam_sw, rng, laws=CLASS_LAWS, A=None, w=None, min_dwell=0):
    """Markov-switching regimes over a bank of recurring laws, switching at rate lam_sw per step.

    Amendment A bar: era identification needs lam_sw * w <= 0.1 (expected switches inside a window).
    Returns nu_schedule, f_schedule (simulate inputs), starts, classes (class per era), A.
    """
    if w is not None and lam_sw * w > 0.1 + 1e-12:
        raise ValueError(f"lam_sw*w = {lam_sw * w:.3f} > 0.1: segmentation bar not satisfiable")
    rng = np.random.default_rng(rng)
    R = len(laws)
    A = (np.ones((R, R)) - np.eye(R)) / (R - 1) if A is None else np.asarray(A, float)
    if np.any(np.diag(A) > 0):
        raise ValueError("A must have zero diagonal: every switch is a law change")
    c, t = int(rng.integers(R)), 0
    starts, classes = [], []
    while t < T:
        starts.append(t)
        classes.append(c)
        t += max(min_dwell, int(rng.geometric(lam_sw)))
        c = int(rng.choice(R, p=A[c]))
    return dict(nu_schedule=[(s, laws[k][0]) for s, k in zip(starts, classes)],
                f_schedule=[(s, laws[k][1]) for s, k in zip(starts, classes)],
                starts=starts, classes=classes, A=A, laws=laws)


def simulate(n, nu_schedule, f_schedule, markov_P, T, dt, n_universes, rng, *,
             omega0=None, return_derivs=False, hold=1):
    """Multiverse of `n_universes` trajectories of the forced 2-D vorticity equation. Returns a dict.

    nu_schedule: scalar or [(start_step, nu)]. f_schedule: None, or [(start_step, pattern)] with
    pattern = (kx, ky, amp[, phase]) or an (n,n) array. If markov_P is given (a kernel, or
    [(start_step, P)]), f_schedule is instead the list of K patterns and the forcing at step t is
    patterns[s_t] for a chain s with kernel P (re-drawn every `hold` steps), shared by all universes.
    Era = maximal interval of constant (nu, f schedule entry, kernel).
    Keys: omega[u,t,x,y], eras[t], boundaries, laws[k]={start,stop,nu,f,P}, f[t], nu[t], states[t]|None, dt,
    and (return_derivs) dwdt, adv, lap.
    """
    rng = np.random.default_rng(rng)
    sp = _Spec(n)
    nus = _collapse(nu_schedule)
    if markov_P is not None:
        Ps = _collapse(markov_P if isinstance(markov_P, list) else [(0, np.asarray(markov_P))])
        pats = list(f_schedule)
        fs = [(0, None)]
        states = markov_chain(Ps, T, rng, hold)
    else:
        Ps, pats, states = [(0, None)], None, None
        fs = _collapse(f_schedule if f_schedule is not None else [(0, None)])
    pts = sorted({s for sc in (nus, fs, Ps) for s, _ in sc if s > 0 and s < T})
    bounds_ = [0] + pts + [T]
    eras = np.repeat(np.arange(len(bounds_) - 1), np.diff(bounds_))
    laws = [dict(start=a, stop=b, nu=_at(nus, a), f=_at(fs, a), P=_at(Ps, a)) for a, b in zip(bounds_[:-1], bounds_[1:])]
    nu_t = np.array([_at(nus, t) for t in range(T)], float)
    if markov_P is not None:
        bank = np.stack([sp.pattern(p) for p in pats])
        f_t = bank[states]
    else:
        cache = {}
        f_t = np.empty((T, n, n))
        for t in range(T):
            key = id(_at(fs, t))
            if key not in cache:
                cache[key] = sp.pattern(_at(fs, t))
            f_t[t] = cache[key]
    fh_t = sp.fft(f_t) * sp.mask  # full spectrum: used by the return_derivs path only
    fhh_t = sp.rfft(f_t) * sp.maskh  # half spectrum: used by the stepper

    if omega0 is None:
        k0 = 3.0
        wh = sp.fft(rng.standard_normal((n_universes, n, n))) * np.exp(-sp.k2 / (2 * k0 ** 2)) * sp.mask
        wh[:, 0, 0] = 0
        w0 = sp.ifft(wh)
        w0 /= np.sqrt((w0 ** 2).mean(axis=(-2, -1), keepdims=True))
    else:
        w0 = np.broadcast_to(np.asarray(omega0, float), (n_universes, n, n)).copy()
    wh = sp.rfft(w0) * sp.maskh

    def rhs(wh_, nu, fh):
        return -sp.adv_hat_r(wh_) - nu * sp.k2h * wh_ + fh

    om = np.empty((n_universes, T, n, n))
    if return_derivs:
        dw, ad, lp = (np.empty_like(om) for _ in range(3))
    for t in range(T):
        om[:, t] = sp.irfft(wh)
        nu, fh = nu_t[t], fhh_t[t]
        if return_derivs:  # exact spectral terms at this snapshot, on the full complex path
            wc = sp.fft(om[:, t]) * sp.mask
            dw[:, t] = sp.ifft(-sp.adv_hat(wc) - nu * sp.k2 * wc + fh_t[t])
            ad[:, t], lp[:, t] = sp.ifft(sp.adv_hat(wc)), sp.ifft(-sp.k2 * wc)
        k1 = rhs(wh, nu, fh)
        k2 = rhs(wh + 0.5 * dt * k1, nu, fh)
        k3 = rhs(wh + 0.5 * dt * k2, nu, fh)
        k4 = rhs(wh + dt * k3, nu, fh)
        wh = wh + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    out = dict(omega=om, eras=eras, boundaries=pts, laws=laws, f=f_t, nu=nu_t, states=states, dt=dt, n=n)
    if return_derivs:
        out.update(dwdt=dw, adv=ad, lap=lp)
    return out

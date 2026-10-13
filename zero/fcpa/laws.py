"""Law plug-ins for FCPARegressor(law=...): rows of the lag stack -> (R, L) with the model R = nu L + f.

NS2DLaw decodes the E1 coefficient vector x = a z of two consecutive snapshots [x_t, x_{t-1}] back to vorticity on
the n x n grid, and returns, in the encoder's own coordinates,
    R = (w_t - w_{t-1}) / dt + u . grad w   (tendency plus advection, advection at the midpoint field)
    L = lap w                               (the Laplacian at the midpoint field)
so that R = nu L + f (L8 / E2') holds up to the modes the encoder does not keep. Under-resolved modes make the
residual large (about 40% of |R| at kmax = 4), so the law is identified by likelihood, not by a tight fit.
"""
import numpy as np

from fcpa.beds import ns2d


class NS2DLaw:
    def __init__(self, n=32, kmax=4, dt=0.1, chunk=2048):
        self.n, self.kmax, self.dt, self.chunk = n, kmax, dt, chunk
        self.kx = np.r_[0:kmax + 1, n - kmax:n]
        self.ky = np.arange(kmax + 1)
        self.D = 2 * len(self.kx) * len(self.ky)

    def group(self, m=17):
        """The bed's exact symmetries (L37 S0-S2) as orthogonal matrices on the D stored coordinates: T_s C^e,
        s = 2 pi j / m, e in {0, 1}; T_s: a_k -> e^{-i kx s} a_k (an x-shift), C: a -> conj a (rotation by pi).
        m = 17 > 12 makes the averaged ridge O(2)-equivariant, not only Z_m (L37 Q2)."""
        kx = np.where(self.kx > self.n // 2, self.kx - self.n, self.kx)
        kxf = np.repeat(kx, len(self.ky))
        h, out = self.D // 2, []
        for j in range(m):
            c, s = np.cos(kxf * 2 * np.pi * j / m), np.sin(kxf * 2 * np.pi * j / m)
            for e in (0, 1):
                g = np.zeros((self.D, self.D))
                sg = -1.0 if e else 1.0  # C flips Im before the shift
                i = np.arange(h)
                g[i, i], g[i, h + i] = c, s * sg  # Re' = c Re + s Im
                g[h + i, i], g[h + i, h + i] = -s, c * sg  # Im' = -s Re + c Im
                out.append(g)
        return out

    def modes(self):
        """The x-shift charge of every stored coordinate (signed kx: T_s multiplies mode k by e^{-i kx s}), its part
        (0 = Re, 1 = Im; C flips Im), and the stored indices of the 40 independent complex modes (Re part; Im is
        + D/2): (0, 0) is zero and ky = 0, kx < 0 are the conjugates of ky = 0, kx > 0."""
        kx = np.where(self.kx > self.n // 2, self.kx - self.n, self.kx)
        kxf, kyf = np.repeat(kx, len(self.ky)), np.tile(self.ky, len(self.kx))
        h = self.D // 2
        return np.r_[kxf, kxf], np.r_[np.zeros(h, int), np.ones(h, int)], np.flatnonzero(~((kyf == 0) & (kxf <= 0)))

    def _decode(self, x):
        h = self.D // 2
        z = (x[:, :h] + 1j * x[:, h:]).reshape(len(x), len(self.kx), len(self.ky))
        wh = np.zeros((len(x), self.n, self.n // 2 + 1), complex)
        wh[:, self.kx[:, None], self.ky[None, :]] = z
        return np.fft.irfft2(wh, s=(self.n, self.n), axes=(-2, -1))

    def _coords(self, w):
        wh = np.fft.rfft2(w, axes=(-2, -1))[:, self.kx[:, None], self.ky[None, :]].reshape(len(w), -1)
        return np.concatenate([wh.real, wh.imag], axis=1)

    def __call__(self, S):
        S = np.asarray(S, float)
        D = self.D
        R, L = [], []
        for a in range(0, len(S), self.chunk):
            s = S[a:a + self.chunk]
            w1, w0 = self._decode(s[:, :D]), self._decode(s[:, D:2 * D])
            adv, lap = ns2d.spectral_terms((w1 + w0) / 2)
            R.append(self._coords((w1 - w0) / self.dt + adv))
            L.append(self._coords(lap))
        return np.concatenate(R), np.concatenate(L)

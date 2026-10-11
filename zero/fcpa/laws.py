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

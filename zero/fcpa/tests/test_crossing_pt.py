"""H1: perturbation-theory crossing S_dE = exp(X), X_ab = dE_ab/(l_b - l_a) (a in top-d, b not), vs
S = I, diagonal@0.3, and the oracle U_k U_j^dagger. Gain = captured variance tr(S P_j S^dag rho_k)
minus tr(P_j rho_k), rho_k known only through 2000 sampled psi. Also the sin-Theta lemma."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fcpa import crossing_pt as C  # noqa: E402

N, D, NPSI = 32, 4, 2000
LAM = np.r_[[1.0, 0.8, 0.64, 0.51], 0.1 * 0.85 ** np.arange(N - D)]
LAM = LAM / LAM.sum()


def herm_skew(rng):
    G = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    return (G - G.conj().T) / (2 * np.sqrt(N))  # skew-Hermitian, op norm ~ 1


def top(rho, d=D):
    w, U = np.linalg.eigh(rho)
    return U[:, ::-1][:, :d]


def run(eps, seed):
    rng = np.random.default_rng(seed)
    V = np.linalg.qr(rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))[0]
    rj = (V * LAM) @ V.conj().T
    Q = expm(eps * herm_skew(rng))
    rk = Q @ rj @ Q.conj().T
    A = np.linalg.cholesky(rk + 1e-14 * np.eye(N))
    g = (rng.standard_normal((N, NPSI)) + 1j * rng.standard_normal((N, NPSI))) / np.sqrt(2)
    psi = A @ g
    rk_hat = psi @ psi.conj().T / NPSI
    Uj, Uk = top(rj), top(rk)
    Pj = Uj @ Uj.conj().T
    cap = lambda S: np.real(np.trace(S @ Pj @ S.conj().T @ rk))
    diag_gen = 1j * np.diag(np.real(np.diag(rk_hat - rj)))
    arms = {
        "I": np.eye(N),
        "diag": expm(0.3 * diag_gen / np.abs(np.diag(diag_gen)).max()),
        "dE": C.crossing_pt(rj, rk_hat - rj, D),
        "dE_js": C.crossing_pt(rj, rk_hat - rj, D, n_samples=NPSI),
        "oracle": Uk @ Uj.conj().T,
    }
    c = {k: cap(S) for k, S in arms.items()}
    # sin-Theta lemma on the true P_k, evaluated with the exact rho_j
    lhs, rhs = C.sin_theta_bound(rj, Uk @ Uk.conj().T, D)
    return c, lhs, rhs, np.allclose(arms["dE"] @ arms["dE"].conj().T, np.eye(N), atol=1e-9)


@pytest.mark.parametrize("eps", [
    pytest.param(0.05, marks=pytest.mark.xfail(strict=True, reason="H1 DEAD at its stated bar: 0.781 < 0.90 at 2000 psi (sampling floor; 0.9999 noise-free). Kept failing on purpose; see ERRATA 13.")),
    0.1, 0.2])
def test_H1_crossing_pt(eps, capsys):
    g = {"diag": 0.0, "dE": 0.0, "dE_js": 0.0, "oracle": 0.0}
    ok = 0
    for seed in range(20):
        c, lhs, rhs, unitary = run(eps, seed)
        assert unitary
        ok += lhs >= rhs - 1e-12
        for k in g:
            g[k] += c[k] - c["I"]
    rec_dE, rec_diag = g["dE"] / g["oracle"], g["diag"] / g["oracle"]
    with capsys.disabled():
        print(f"\nH1 eps={eps}: S_dE recovers {rec_dE:.3f}, JS {g['dE_js'] / g['oracle']:.3f}, diag@0.3 {rec_diag:.3f}, sinTheta bound {ok}/20")
    assert ok == 20
    if eps <= 0.1:
        assert rec_dE >= 0.9
        assert rec_diag < 0.1

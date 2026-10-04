"""AT I 0 -- reference identities (numpy only, seed 0).
Identities and theorem checks only. None of these numbers is evidence for any primitive (L-GIVEN)."""
import numpy as np

rng = np.random.default_rng(0)

print("== 1. Turing paradox (quantum Zeno): survival after a pi rotation measured N times ==")
for N in [1, 2, 10, 100, 1000]:
    c, s = np.cos(np.pi / (2 * N)), np.sin(np.pi / (2 * N))
    R = np.array([[c, -1j * s], [-1j * s, c]])            # exp(-i (pi/N) sigma_x / 2)
    surv = 1.0
    for _ in range(N):                                    # evolve, measure, keep the |0> branch
        surv *= abs((R @ np.array([1, 0]))[0]) ** 2
    print(f"N={N:5d}  cos^(2N)(pi/2N)={c ** (2 * N):.6f}  simulated={surv:.6f}  "
          f"exp(-pi^2/4N)={np.exp(-np.pi ** 2 / (4 * N)):.6f}")

print("\n== 2. QQ equality: Luders measurements vs classical models (1000 random draws each) ==")
def rand_proj(n):
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, _ = np.linalg.qr(z)
    v = q[:, : rng.integers(1, n)]
    return v @ v.conj().T

def quantum_draw(n, mixed):
    if mixed:
        g = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); rho = g @ g.conj().T
    else:
        v = rng.normal(size=n) + 1j * rng.normal(size=n); rho = np.outer(v, v.conj())
    rho /= np.trace(rho).real
    A, B = rand_proj(n), rand_proj(n); An, Bn = np.eye(n) - A, np.eye(n) - B
    p = lambda X, Y: np.trace(Y @ X @ rho @ X @ Y).real   # X answered first, then Y
    return (p(A, Bn) + p(An, B)) - (p(B, An) + p(Bn, A)), p(A, B) - p(B, A)

def classical_draw(m, disturb):
    pi = rng.dirichlet(np.ones(m)); R = {q: rng.uniform(size=m) for q in "AB"}
    T = {(q, a): rng.dirichlet(np.ones(m), size=m) if disturb else np.eye(m) for q in "AB" for a in (0, 1)}
    def p(q1, a1, q2, a2):
        r1 = R[q1] if a1 else 1 - R[q1]; r2 = R[q2] if a2 else 1 - R[q2]
        return float(((pi * r1) @ T[(q1, a1)]) @ r2)
    return ((p("A", 1, "B", 0) + p("A", 0, "B", 1)) - (p("B", 1, "A", 0) + p("B", 0, "A", 1)),
            p("A", 1, "B", 1) - p("B", 1, "A", 1))

for name, f in [("Luders, pure states   ", lambda: quantum_draw(4, False)),
                ("Luders, mixed states  ", lambda: quantum_draw(4, True)),
                ("classical + disturb   ", lambda: classical_draw(4, True)),
                ("classical, no disturb ", lambda: classical_draw(4, False))]:
    r = np.array([f() for _ in range(1000)])
    print(f"{name} max|QQ|={np.abs(r[:, 0]).max():.2e}  mean|QQ|={np.abs(r[:, 0]).mean():.4f}  "
          f"mean|order effect|={np.abs(r[:, 1]).mean():.4f}")

print("\n== 3. Stacked causal softmax, no MLP: attention mass on token 1 (BOS), N = 16 ==")
N = 16
mask = np.tril(np.ones((N, N))) > 0
def causal_softmax(sigma):
    sc = np.where(mask, rng.normal(scale=sigma, size=(N, N)), -np.inf)
    a = np.exp(sc - sc.max(1, keepdims=True)); return a / a.sum(1, keepdims=True)
for sigma in (1.0, 3.0):
    for gamma, name in ((1.0, "pure attention"), (0.5, "residual 0.5  ")):
        P, cells = np.eye(N), []
        for L in range(1, 65):
            P = (gamma * causal_softmax(sigma) + (1 - gamma) * np.eye(N)) @ P
            if L in (1, 4, 16, 64):
                cells.append(f"L={L}: mean {P[:, 0].mean():.4f} min {P[:, 0].min():.4f}")
        print(f"sigma={sigma} {name} " + " | ".join(cells))

print("\n== 4. Lossless + causal over infinite time, cut to a window of 16 tokens: leaks vs state dim ==")
def allpass(poles, n=4096):
    h = np.zeros(n); h[0] = 1.0
    for a in poles:                                        # H(z) = (-a + z^-1) / (1 - a z^-1)
        g = np.zeros(n); g[0] = -a; g[1:] = (1 - a * a) * a ** np.arange(n - 1)
        h = np.convolve(h, g)[:n]
    return h
W = 16
for poles in ((0.5,), (0.5, -0.3), (0.5, -0.3, 0.8)):
    h = allpass(poles)
    T = np.array([[h[i - j] if i >= j else 0.0 for j in range(W)] for i in range(W)])
    sv = np.linalg.svd(T, compute_uv=False)
    print(f"poles={str(poles):17s} sum h^2={np.sum(h ** 2):.6f}  singular values < 1: "
          f"{int((sv < 1 - 1e-9).sum())} (state dim {len(poles)})  max sv={sv.max():.6f}  min sv={sv.min():.6f}")

print("\n== 5. T-A: the nearest unitary to a causal mixer is acausal ==")
shares = []
for _ in range(100):
    u, _, vh = np.linalg.svd(np.tril(rng.normal(size=(W, W))))
    shares.append(np.linalg.norm(np.triu(u @ vh, 1)) ** 2 / W)
u, _, vh = np.linalg.svd(np.diag(rng.normal(size=W)))
print(f"random lower-triangular mixer: mean share of |U_polar|_F^2 above the diagonal = {np.mean(shares):.4f}")
print(f"diagonal control:              share = {np.linalg.norm(np.triu(u @ vh, 1)) ** 2 / W:.4f}")

print("\n== 6. softmax = Born rule on a decohered Gibbs state; a coherent coupling breaks Luce IIA ==")
def born_gibbs(H, beta=1.0):
    w, V = np.linalg.eigh(H); rho = (V * np.exp(-beta * w)) @ V.conj().T
    d = np.real(np.diag(rho)); return d / d.sum()
s3 = np.array([0.3, 0.0, 0.2])
for g in (0.0, 0.5, 1.0):
    H3 = -np.diag(s3); H3[0, 2] = H3[2, 0] = -g             # coupling between key 1 and key 3 only
    p3, p2 = born_gibbs(H3), born_gibbs(-np.diag(s3[:2]))
    print(f"g={g}: p1/p2 with key 3 present = {p3[0] / p3[1]:.4f}  key 3 removed = {p2[0] / p2[1]:.4f}  "
          f"softmax, any context = {np.exp(s3[0] - s3[1]):.4f}")

print("\n== 7. Delta-rule transition I - beta k k^T: beta=1 projector (reset), beta=2 reflection (group) ==")
k = rng.normal(size=8); k /= np.linalg.norm(k)
for beta in (1.0, 2.0):
    M = np.eye(8) - beta * np.outer(k, k)
    ev = sorted({float(x) for x in np.round(np.linalg.eigvalsh(M), 6) + 0.0})
    print(f"beta={beta}: distinct eigenvalues {ev}  |M^2-M|={np.linalg.norm(M @ M - M):.1e}  "
          f"|M^2-I|={np.linalg.norm(M @ M - np.eye(8)):.1e}")

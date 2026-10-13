"""EXPLORATORY: which era model can beat persistence at 50 closed-loop steps (L28, L37)? Not the preregistered run.

Runs only on the tuning bed (seed 3, bed seed 2003); refuses eval seeds. Fit on the Train split, scored on the Cal
split as validation (as run_primary.tune scores ASMODEOUS's lr); Test is never read. One model for all rows (one
era): the question here is whether the model class can carry a 50-step rollout at all.

Candidates (ridge on z = [x_t, x_{t-1}] (+ quadratic library of x_t), target y = x_{t+1} - x_t):
  lin        linear, unconstrained
  lin_vq     linear, restricted to V_Q (O(2)-equivariant, L37 Q2: 34-element augmentation)
  quad       linear + orthonormal Sym^2 quadratic of x_t, unconstrained
  quad_vq    the same, restricted to V_Q
each at ridge levels LAMS and increment scales GAMMAS (the rollout steps x + gamma * y_hat).

    python -m fcpa.explore_era_model --out DIR
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcpa import bars, run_primary as rp  # noqa: E402

KMAX, N_GRID = 4, 32
KX = np.r_[0:KMAX + 1, -KMAX:0]  # signed kx of the 9 stored rows (ns2d.encode: r_[0:5, 28:32] mod 32)
KY = np.arange(KMAX + 1)
KXM, KYM = np.meshgrid(KX, KY, indexing="ij")
KXF, KYF = KXM.ravel(), KYM.ravel()  # the 45 stored complex modes, in encoder order
# independent modes: drop (0,0) (zero) and ky = 0, kx < 0 (conjugates of ky = 0, kx > 0): 40 complex = 80 real
KEEP = ~((KYF == 0) & (KXF <= 0))
CONJ = {i: int(np.flatnonzero((KXF == -KXF[i]) & (KYF == 0))[0]) for i in np.flatnonzero((KYF == 0) & (KXF > 0))}
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
GAMMAS = (1.0, 0.9, 0.8)
GROUP = [(m, e) for m in range(17) for e in (0, 1)]  # Gamma_17: T_{2 pi m/17} C^e (L37 Q2)


def to_c(x):
    return x[..., :45] + 1j * x[..., 45:]


def from_c(z):
    return np.concatenate([z.real, z.imag], axis=-1)


def act(x, m, e):
    """g = T_s C^e on stored coordinates (L37 S0, S2): a -> e^{-i kx s} (conj a if e)."""
    z = to_c(x)
    z = np.conj(z) if e else z
    return from_c(z * np.exp(-1j * KXF * (2 * np.pi * m / 17)))


def reduce(x):
    """90 stored coordinates -> the 80 independent ones."""
    z = to_c(x)[..., KEEP]
    return np.concatenate([z.real, z.imag], axis=-1)


def expand(r):
    """80 independent coordinates -> 90 stored (conjugate partners filled, (0,0) = 0)."""
    h = r.shape[-1] // 2
    zk = r[..., :h] + 1j * r[..., h:]
    z = np.zeros(r.shape[:-1] + (45,), complex)
    z[..., KEEP] = zk
    for i, j in CONJ.items():
        z[..., j] = np.conj(z[..., i])
    return from_c(z)


IU = np.triu_indices(80)
QW = np.where(IU[0] == IU[1], 1.0, np.sqrt(2.0))  # orthonormal Sym^2 basis (L37 R1: invariant penalty)


def features(xr, xr1, quad, scale):
    """z = [1, x_t, x_{t-1}, q(x_t)] on scaled independent coordinates; q in the orthonormal Sym^2 basis."""
    a, b = xr / scale, xr1 / scale
    cols = [np.ones((len(a), 1)), a, b]
    if quad:
        cols.append((a[:, IU[0]] * a[:, IU[1]]) * QW)
    return np.concatenate(cols, axis=1)


def pairs(X):
    """Rows (x_t, x_{t-1}, y_t = x_{t+1} - x_t) of every trajectory, t = 0 .. L-2, x_{-1} = x_0 (rp.lag_stack)."""
    XL = rp.lag_stack(X, 2)
    D = X.shape[-1]
    return XL[:, :-1, :D].reshape(-1, D), XL[:, :-1, D:].reshape(-1, D), np.diff(X, axis=1).reshape(-1, D)


def gram(xt, xt1, y, quad, scale, sym, chunk=6000):
    """Sums Z^T Z and Z^T Y over the rows, or over the 34 group images of every row (sym: L37 Q2 augmentation)."""
    gs = GROUP if sym else [(0, 0)]
    G = C = None
    for m, e in gs:
        for a in range(0, len(xt), chunk):
            s = slice(a, a + chunk)
            Z = features(reduce(act(xt[s], m, e)), reduce(act(xt1[s], m, e)), quad, scale)
            Y = reduce(act(y[s], m, e)) / scale
            G = Z.T @ Z if G is None else G + Z.T @ Z
            C = Z.T @ Y if C is None else C + Z.T @ Y
    return G / len(gs), C / len(gs)


def solve(G, C, lam, n):
    """Ridge with the invariant penalty lam * n on every block but the intercept (column 0)."""
    P = np.full(len(G), lam * n)
    P[0] = 0.0
    return np.linalg.solve(G + np.diag(P), C)


def stepper(W, quad, scale, gamma, D=90):
    def step(rows):
        xt, xt1 = rows[:, :D], rows[:, D:2 * D]
        dy = features(reduce(xt), reduce(xt1), quad, scale) @ W * scale
        return xt + gamma * expand(dy)
    return step


def run(out, log=print):
    P = rp.PINS
    X, reg = rp.gen_bed(rp.TUNE_BED_SEED, P, 4, out)
    tr, ca, _ = rp.three_way_split(len(X), bars.TUNE_SEED)  # Test (third split) is never read here
    xt, xt1, y = pairs(X[tr])
    scale = float(np.sqrt(np.mean(reduce(xt) ** 2)))  # one scalar: invariant under the group
    n = len(xt)
    pers = rp.rollout_errors(rp.persistence_step(90), X[ca], P, None, 2)
    vt, vt1, vy = pairs(X[ca])
    p1 = float(bars.rel_err(vt + 0 * vy, vt + vy).mean())
    log(f"seed 3: train rows {n}, val trajectories {len(ca)}; persistence one-step {p1:.4f}, E50 {pers.mean():.4f}")
    res = [dict(model="persistence", lam=None, gamma=None, one_step=p1, e50=float(pers.mean()), nonfinite=0)]
    for quad, sym in itertools.product((False, True), (False, True)):
        name = ("quad" if quad else "lin") + ("_vq" if sym else "")
        t0 = time.time()
        G, C = gram(xt, xt1, y, quad, scale, sym)
        log(f"{name}: {G.shape[0]} features, gram {time.time() - t0:.0f}s")
        for lam in LAMS:
            W = solve(G, C, lam, n)
            one = stepper(W, quad, scale, 1.0)(np.c_[vt, vt1])
            o1 = float(bars.rel_err(one, vt + vy).mean())
            for gamma in GAMMAS:
                bad = [0]
                with np.errstate(all="ignore"):
                    e = rp.rollout_errors(stepper(W, quad, scale, gamma), X[ca], P, bad, 2)
                r = dict(model=name, lam=lam, gamma=gamma, one_step=o1, e50=float(e.mean()), nonfinite=bad[0],
                         e50_vs_pers=float(e.mean() / pers.mean()), wins=float((e.mean(1) < pers.mean(1)).mean()))
                log(json.dumps(r))
                res.append(r)
                if out:
                    with open(Path(out) / "era_model_results.jsonl", "a") as fh:
                        fh.write(json.dumps(r) + "\n")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        run(a.out)


if __name__ == "__main__":
    main()

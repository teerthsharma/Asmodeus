"""Arm-free calibration of eps/delta (allowed before PREREG: no arm is trained or scored).
oracle = embedded state; pca = top 2n+2 PCs of the noisy obs (a continuous linear encoder, noise floor);
torn = atan2-type chart (one seam); fold = continuous non-injective projection; const = collapse."""
import sys

import numpy as np

from beds import make_bed, torn_chart, hopf
import metrics as M

DIAM = {"s2": np.pi, "t2": np.pi * 2 ** .5, "klein": np.pi * 2 ** .5, "so3": np.pi, "r2": 2 * 2 ** .5,
        "berry": np.pi, "berry0": np.pi}   # geo = great circle on S^3 / S^2
FOLD = {"s2": lambda s: s[..., :2], "t2": lambda s: s[..., 0::2], "klein": lambda s: s[..., :2],
        "so3": lambda s: s[..., :3], "r2": lambda s: np.stack([s[..., 0], np.abs(s[..., 1])], -1),
        "berry": hopf,                              # Hopf map: collapses the U(1) fibre
        "berry0": lambda s: s[..., :2]}


def latents(bed):
    s, o, n = bed["state"], bed["obs"].astype(np.float64), bed["meta"]["n"]
    oc = o.reshape(-1, o.shape[-1]); oc = oc - oc.mean(0)
    pca = (oc @ np.linalg.svd(oc, full_matrices=False)[2][:2 * n + 2].T).reshape(s.shape[:2] + (-1,))
    name = bed["meta"]["name"]
    return dict(oracle=s, pca=pca, torn=torn_chart(name, s), fold=FOLD[name](s), const=np.zeros_like(s))


def main(names, seed=1000):
    for name in names:
        bed = make_bed(name, n_traj=128, T=64, obs="lift", seed=seed)
        L, d = latents(bed), DIAM[name]
        print(f"\n== {name}  diam={d:.3f}  (eval bed seed {seed}, 4000 of 8192 points)")
        for f in (0.04, 0.06, 0.08, 0.10):
            for delta in (0.25, 0.5, 1.0):
                row = {k: M.discontinuity_rate(v, bed["state"], bed["geo"], f * d, delta) for k, v in L.items()}
                print(f"disc eps={f:.2f}*diam={f*d:.3f} delta={delta:4.2f}  " + "  ".join(f"{k}={v:.4f}" for k, v in row.items()))
        for far in (0.3, 0.45):
            for fd in (0.02, 0.05):
                row = {k: M.fold_rate(v, bed["state"], bed["geo"], far * d, fd) for k, v in L.items()}
                print(f"fold far={far:.2f}*diam={far*d:.3f} fdelta={fd:.2f}  " + "  ".join(f"{k}={v:.4f}" for k, v in row.items()))
        print("probe_r2 " + "  ".join(f"{k}={M.linear_probe(v, bed['state']):.3f}" for k, v in L.items()))


if __name__ == "__main__":
    main(sys.argv[1:] or ["s2", "t2", "klein", "so3", "r2"])

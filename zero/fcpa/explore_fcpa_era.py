"""EXPLORATORY: the L37 era models inside the full FCPA fit (run_primary.fit_fcpa). Not the preregistered run.

Runs only on the tuning bed (seed 3, bed seed 2003) and non-eval seeds (7001, ...); refuses eval seeds 0, 1, 2, 4, 5.
Each candidate is an FCPA_PINS variant, fitted as the primary run fits it (Train trajectories, then the Cal split
calibrates, Amendment 11c) and scored on the Test split: closed-loop 50-step relative error (E50) against
persistence, the one-step error, and the share of Test trajectories where FCPA beats persistence.

    python -m fcpa.explore_fcpa_era --seeds 3 7001 --out DIR
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcpa import bars, run_primary as rp  # noqa: E402

BASE = dict(damping=1.0, law="ns2d", era="phi", era_lam=0.1)  # the pins before Amendment 11d
CANDIDATES = {
    "phi": dict(BASE),
    "lin_vq": dict(BASE, era="lin_vq", era_lam=1e-3),
    "quad_vq": dict(BASE, era="quad_vq", era_lam=0.1),
    "quad_vq_d0.9": dict(BASE, era="quad_vq", era_lam=0.1, damping=0.9),
    "quad_vq_l0.03": dict(BASE, era="quad_vq", era_lam=0.03),
    "quad_vq_l0.3": dict(BASE, era="quad_vq", era_lam=0.3),
    "quad_vq_pool1": dict(BASE, era="quad_vq", era_lam=0.1, era_pool=1.0),
    "quad_vq_pool0.1": dict(BASE, era="quad_vq", era_lam=0.1, era_pool=0.1),
    "quad_vq_shared": dict(BASE, era="quad_vq", era_lam=0.1, era_pool=float("inf")),
    "quad_vq_oodtt": dict(BASE, era="quad_vq", era_lam=0.1, ood="tt"),
    "quad_vq_shared_oodtt": dict(BASE, era="quad_vq", era_lam=0.1, era_pool=float("inf"), ood="tt"),
    "phi_energy": dict(BASE, energy=True),
    "lin_vq_energy": dict(BASE, era="lin_vq", era_lam=1e-3, energy=True),
    "quad_vq_energy_d0.9": dict(BASE, era="quad_vq", era_lam=0.1, energy=True, damping=0.9),
    "quad_vq_energy_l0.3": dict(BASE, era="quad_vq", era_lam=0.3, energy=True),
    "quad_vq_shared_energy": dict(BASE, era="quad_vq", era_lam=0.1, era_pool=float("inf"), energy=True),
    "quad_vq_pool1_energy": dict(BASE, era="quad_vq", era_lam=0.1, era_pool=1.0, energy=True),
    "quad_vq_energy": dict(BASE, era="quad_vq", era_lam=0.1, energy=True),
    "quad_vq_1era": dict(BASE, era="quad_vq", era_lam=0.1, n_eras_max=1),
    "quad_vq_1era_l0.3": dict(BASE, era="quad_vq", era_lam=0.3, n_eras_max=1),
}


def run_seed(seed, P, names, out, log=print):
    if seed in bars.EVAL_SEEDS:
        raise SystemExit(f"seed {seed} is an eval seed: exploration never reads it")
    bed_seed = rp.TUNE_BED_SEED if seed == bars.TUNE_SEED else seed
    X, _ = rp.gen_bed(bed_seed, P, 4, out)
    tr, ca, te = rp.three_way_split(len(X), seed)
    nl, D = P["n_lags"], X.shape[-1]
    XL = rp.lag_stack(X, nl)
    pers = rp.rollout_errors(rp.persistence_step(D), X[te], P, None, nl)
    xt, y = XL[te][:, :-1].reshape(-1, nl * D), X[te][:, 1:].reshape(-1, D)
    p1 = float(bars.rel_err(xt[:, :D], y).mean())
    log(f"seed {seed}: persistence E50 {pers.mean():.4f}, one-step {p1:.4f}")
    res = []
    for name in names:
        t0 = time.time()
        pins = dict(CANDIDATES[name])
        Pc = dict(P, n_eras_max=pins.pop("n_eras_max", P["n_eras_max"]))
        est = rp.fit_fcpa(XL[tr], Pc, seed, pins=pins, cal=XL[ca])
        bad = [0]
        ref = float(np.linalg.norm(XL[tr].reshape(-1, nl * D), axis=1).max())

        def step(rows):  # a diverged rollout (|row| > 1e3 x the largest Train row) answers non-finite: counted, error 1
            ok = np.isfinite(rows).all(1) & (np.linalg.norm(rows, axis=1) < 1e3 * ref)
            out = np.full((len(rows), D), np.nan)
            if ok.any():
                out[ok] = rp.first_block(est.predict(rows[ok]), D)
            return out

        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            e = rp.rollout_errors(step, X[te], P, bad, nl)
            f1 = float(bars.rel_err(rp.first_block(est.predict(xt), D), y).mean())
        r = dict(seed=seed, model=name, pins=CANDIDATES[name], e50=float(e.mean()), e50_pers=float(pers.mean()),
                 e50_vs_pers=float(e.mean() / pers.mean()), wins=float((e.mean(1) < pers.mean(1)).mean()),
                 one_step=f1, one_step_pers=p1, nonfinite=bad[0], n_eras=int(est.n_eras_),
                 shrink=[round(float(g), 3) for g in est.shrink_], seconds=round(time.time() - t0))
        log(json.dumps(r))
        res.append(r)
        if out:
            with open(Path(out) / "fcpa_era_results.jsonl", "a") as fh:
                fh.write(json.dumps(r) + "\n")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[bars.TUNE_SEED])
    ap.add_argument("--models", nargs="+", default=list(CANDIDATES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
    P = rp.SMOKE if a.smoke else rp.PINS
    for s in a.seeds:
        run_seed(s, P, a.models, a.out)


if __name__ == "__main__":
    main()

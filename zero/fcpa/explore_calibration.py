"""EXPLORATORY: compare conformal-calibration protocols for FCPA (L36). Not the preregistered run.

Runs only on the tuning bed (seed 3, bed seed 2003) and non-eval seeds (7001, ...); refuses eval seeds 0, 1, 2, 4, 5.
Nothing here changes FCPARegressor: `CalRegressor` overrides only which windows are held out for calibration.

    python -m fcpa.explore_calibration --check               # 'interleaved' mode reproduces FCPARegressor exactly
    python -m fcpa.explore_calibration --seeds 3 7001 --out DIR

Modes (Train/Cal/Test is run_primary.three_way_split; Test is never used for fitting or calibration):
  interleaved   current code: window j of the global stream is held out iff j % 5 == 4; FIFO pool of 400 rows
  interleaved*  same split, pool uncapped
  gap1, gap2    L36d P_g: held out iff j % 10 == 4, the g windows on each side dropped (neither train nor cal)
  R2, R2-1w     L36 R2: every 5th Train trajectory held out whole (R2) or one random window of it (R2-1w, L36c(i))
  R1, R1-1w     L36 R1: all Train trajectories train; the Cal split calibrates, whole or one random window each
"""
import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcpa import bars, conformal, estimator as E, run_primary as rp  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
from fcpa.estimator import FCPARegressor  # noqa: E402

UNCAPPED = 10 ** 9
MODES = {  # name: (rule, pool cap)
    "interleaved": ("interleaved", 400), "interleaved*": ("interleaved", UNCAPPED), "gap1": ("gap1", UNCAPPED),
    "gap2": ("gap2", UNCAPPED), "R2": ("R2", UNCAPPED), "R2-1w": ("R2-1w", UNCAPPED), "R1": ("R1", UNCAPPED),
    "R1-1w": ("R1-1w", UNCAPPED),
}
TRAIN, CAL, DROP = 0, 1, 2


class CalRegressor(FCPARegressor):
    """FCPARegressor whose held-out windows follow `rule` instead of j % 5 == 4. `role` ('train' / 'cal') and
    `pick` (the one window kept of a cal trajectory in the -1w modes) are set per trajectory by the driver."""

    rule, role, pick, k_in_traj = "interleaved", "train", None, 0

    def _what(self, j):
        r = self.rule
        if r == "interleaved":
            return CAL if j % E._CAL_EVERY == E._CAL_EVERY - 1 else TRAIN
        if r in ("gap1", "gap2"):
            g, m = (1 if r == "gap1" else 2), j % 10
            return CAL if m == 4 else DROP if abs(m - 4) <= g else TRAIN
        if self.role == "train":
            return TRAIN
        return CAL if self.pick is None or self.k_in_traj == self.pick else DROP

    def _add_to_seg(self, st, seg, idx, X, Y, psi, phi, sym, R, L, cal):
        if not cal:
            self._ver = getattr(self, "_ver", 0) + 1  # the law statistics change: cached pool scores go stale
        return super()._add_to_seg(st, seg, idx, X, Y, psi, phi, sym, R, L, cal)

    def _pool_scores(self, st):
        """_nc_scores(st.segs, st.cal_phi), recomputed only for rows added since the eras last changed."""
        v, n = getattr(self, "_ver", 0), len(st.cal_phi)
        c = getattr(self, "_cache", None)
        if c is None or c[0] != v or c[1] != len(st.segs) or len(c[2]) > n:
            c = (v, len(st.segs), self._nc_scores(st.segs, st.cal_phi))
        elif len(c[2]) < n:
            c = (v, len(st.segs), np.vstack([c[2], self._nc_scores(st.segs, st.cal_phi[len(c[2]):])]))
        self._cache = c
        return c[2]

    def _cal_window(self, st, phi, label):
        """The base _cal_window (online ACI step, then append), with cached pool scores; no FIFO cut when uncapped."""
        if len(st.cal_lab) and any(sg.n for sg in st.segs):
            cp = conformal.SplitConformal(self.alpha).fit(self._pool_scores(st), st.cal_lab)
            for row in self._nc_scores(st.segs, phi):
                err = int(label not in cp.predict_set(row, alpha=st.aci.alpha_t))
                f = st.aci.forced_err()
                st.aci.update(err if f is None else f)
        cut = len(st.cal_phi) + len(phi) > E._CAL_MAX
        st.cal_phi = np.vstack([st.cal_phi, phi])[-E._CAL_MAX:]
        st.cal_lab = np.r_[st.cal_lab, np.full(len(phi), label)][-E._CAL_MAX:]
        if cut:
            self._cache = None

    def _consume_law(self, st, X, Y, flush):
        if flush:  # the part-filled tail: train rows only on a train trajectory (base: never a cal window)
            return super()._consume_law(st, X, Y, flush) if self.role == "train" else None
        what = self._what(st.window_idx)
        psi, phi, sym = self._encode(X)
        if what != DROP:
            R, L = self.law(X)
            win = (X, Y, psi, phi, sym, R, L, what == CAL)
            k = self._best_law(st.segs, R, L)
            if k >= 0:
                self._add_to_seg(st, st.segs[k], k, *win)
            else:  # as the base: an unexplained window waits for a birth with its held-out flag
                st.nov.append(win)
                self._try_birth(st)
        st.last_sym, st.window_idx = int(sym[-1]), st.window_idx + 1
        self.k_in_traj += 1


@contextmanager
def cal_cap(n):
    old, E._CAL_MAX = E._CAL_MAX, n
    try:
        yield
    finally:
        E._CAL_MAX = old


def fit(XL_tr, XL_ca, P, seed, mode):
    """Fit like run_primary.fit_fcpa (same kwargs, one chunk per trajectory), with the mode's held-out rule."""
    rule, cap = MODES[mode]
    from fcpa.laws import NS2DLaw
    edges = FCPARegressor.make_step_edges(XL_tr, n_lags=P["n_lags"])
    est = CalRegressor(n_eras_max=P["n_eras_max"], d_chart=P["d_chart"], alpha=bars.ALPHA, gamma=bars.GAMMA,
                       window=P["w"], random_state=seed, y_edges=edges, damping=rp.FCPA_PINS["damping"], n_lags=P["n_lags"],
                       law=NS2DLaw(n=P["grid"], kmax=P["kmax"], dt=P["dt"] * P["steps_per_snapshot"]))
    est.rule = rule
    rng = np.random.default_rng(seed + 991)
    n_win = (XL_tr.shape[1] - 1) // P["w"]
    if rule == "R2":
        feed = [(x, "cal" if i % 5 == 4 else "train", None) for i, x in enumerate(XL_tr)]
    elif rule == "R2-1w":
        feed = [(x, "cal" if i % 5 == 4 else "train", int(rng.integers(n_win))) for i, x in enumerate(XL_tr)]
    elif rule in ("R1", "R1-1w"):
        # cal trajectories go last: the era set is fixed by then, so the pool is scored by the final eras
        feed = [(x, "train", None) for x in XL_tr]
        feed += [(x, "cal", None if rule == "R1" else int(rng.integers(n_win))) for x in XL_ca]
    else:
        feed = [(x, "train", None) for x in XL_tr]
    with cal_cap(cap):
        for x, role, pick in feed:
            est.role, est.pick, est.k_in_traj = role, pick, 0
            est.partial_fit(x)
            rp.end_trajectory(est)
        est.role = "train"
    return est


def static_coverage(est, XLte, rte, emap):
    """Split-conformal sets at the fixed level alpha (no ACI) on every Test row: coverage of the true regime, per
    row and per trajectory, and mean set size."""
    m, D = est._model, XLte.shape[-1]
    nc = rp.fcpa_scores(est, XLte.reshape(-1, D))
    if m["cal"] is None:
        sets = [{int(k)} for k in nc.argmin(1)]
    else:
        sets = [m["cal"].predict_set(r, alpha=bars.ALPHA) for r in nc]
    y = rte.reshape(-1)
    hit = np.array([any(emap[k] == t for k in C) for C, t in zip(sets, y)], float)
    size = np.array([len(C) for C in sets], float)
    per_traj = hit.reshape(rte.shape).mean(1)
    return dict(coverage=float(hit.mean()), cov_traj_min=float(per_traj.min()), cov_traj_sd=float(per_traj.std()),
                mean_C=float(size.mean()), empty=float((size == 0).mean()))


def one_step(est, X, XL, D):
    """Relative one-step error on Test rows (FCPA vs persistence), raw encoded space."""
    rows, nxt = XL[:, :-1].reshape(-1, XL.shape[-1]), X[:, 1:].reshape(-1, D)
    f = bars.rel_err(rp.first_block(est.predict(rows), D), nxt).mean()
    p = bars.rel_err(rows[:, :D], nxt).mean()
    return float(f), float(p)


def run_seed(seed, P, modes, out, workers, log=print):
    if seed in bars.EVAL_SEEDS:
        raise SystemExit(f"refused: seed {seed} is an eval seed {bars.EVAL_SEEDS}")
    bed = rp.TUNE_BED_SEED if seed == bars.TUNE_SEED else seed
    X, reg = rp.gen_bed(bed, P, workers, out)
    tr, ca, te = rp.three_way_split(len(X), seed)
    nl, D = P["n_lags"], X.shape[-1]
    XL = rp.lag_stack(X, nl)
    pers = rp.rollout_errors(rp.persistence_step(D), X[te], P, None, nl).mean()
    res = []
    for mode in modes:
        t0 = time.time()
        est = fit(XL[tr], XL[ca], P, seed, mode)
        st = est._st
        stream = rp.fcpa_stream(est, XL[ca], reg[ca], XL[te], reg[te], P)
        sc = static_coverage(est, XL[te], reg[te], stream["emap"])
        f1, p1 = one_step(est, X[te], XL[te], D)
        e50 = rp.rollout_errors(lambda rows: rp.first_block(est.predict(rows), D), X[te], P, None, nl).mean()
        n_train = int(sum(s.n for s in st.segs))
        r = dict(seed=seed, mode=mode, eras=int(est.n_eras_), train_rows=n_train, cal_rows=int(len(st.cal_lab)),
                 **sc, aci_dev=float(stream["dev"]), P3=sum(stream["correct"]) / max(sum(stream["total"]), 1),
                 gam=[round(float(g), 3) for g in est.shrink_], gam_tt=round(float(est.shrink_tt_), 3),
                 one_step=f1, one_step_pers=p1, e50=float(e50), e50_pers=float(pers), seconds=round(time.time() - t0))
        log(json.dumps(r))
        res.append(r)
        if out:
            with open(Path(out) / "calib_results.jsonl", "a") as fh:
                fh.write(json.dumps(r) + "\n")
    return res


def check(P=dict(rp.SMOKE, n_traj=40, L=200), seed=7001):
    """'interleaved' with the 400-row cap must equal FCPARegressor exactly (same eras, shrink, calibration pool)."""
    X, _ = rp.gen_bed(seed, P)
    tr, ca, _ = rp.three_way_split(len(X), seed)
    XL = rp.lag_stack(X, P["n_lags"])
    a, b = rp.fit_fcpa(XL[tr], P, seed), fit(XL[tr], XL[ca], P, seed, "interleaved")
    np.testing.assert_array_equal(a._st.cal_phi, b._st.cal_phi)
    np.testing.assert_array_equal(a._st.cal_lab, b._st.cal_lab)
    np.testing.assert_array_equal(a.shrink_, b.shrink_)
    assert a.alpha_t_ == b.alpha_t_ and a.n_eras_ == b.n_eras_
    rows = XL[tr][0]
    np.testing.assert_array_equal(a.predict(rows), b.predict(rows))
    print(f"check: interleaved == FCPARegressor (pool {len(a._st.cal_lab)} rows, labels, shrink, alpha_t, predictions)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[bars.TUNE_SEED])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    if a.check:
        with threadpool_limits(limits=1):
            return check()
    P = rp.SMOKE if a.smoke else rp.PINS
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):  # BLAS threads are 10x slower here (see estimator._single_blas)
        for s in a.seeds:
            run_seed(s, P, a.modes, a.out, a.workers)


if __name__ == "__main__":
    main()

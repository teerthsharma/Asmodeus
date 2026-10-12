"""Primary-claim runner for Amendment 11 (P1-P4, C1-C2 on the NS-2D multiverse bed).

    python -m fcpa.run_primary --smoke            # tiny sizes on non-eval seeds, < 2 minutes, P4 VOID by construction
    python -m fcpa.run_primary                    # the full evaluation (House launches this)

Order (the amendment's "Order"): verify the FREEZE hashes and refuse if they differ -> tune on seed 3 (bed seed
2003): FCPA fit, ASMODEOUS m by the compute rule, lr in {1e-3, 3e-3, 1e-2} by val E50 -> evaluate seeds
0,1,2,4,5 (never 3) -> bars -> results.json + report.md. C2' (Amendment 11a): baselines train at their pinned
budgets and training MACs are reported, not equalised; `--c2 strict` is the draft's C2 as a labelled DIAGNOSTIC. Secondary bars S1-S5 need arms this driver does not
train (learned-operator arm, null streams, certified-change events, Benettin); they are reported NOT RUN.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fcpa import bars, baselines  # noqa: E402
from fcpa.beds import ns2d  # noqa: E402
from fcpa.conformal import trajectory_split  # noqa: E402

SPECS = ROOT / "docs" / "specs"
TUNE_BED_SEED = 2003
LAWS = [(0.010, (0, 4, 0.5)), (0.020, (0, 4, 0.5)), (0.010, (0, 2, 0.5)), (0.020, (0, 2, 1.0))]
PINS = dict(grid=32, kmax=4, dt=0.01, steps_per_snapshot=10, burnin=200, lam_sw=0.005, w=20, n_traj=256, L=400,
            origins=tuple(bars.ORIGINS), horizon=bars.HORIZON, train_steps=5000, tune_steps=1000, batch=64,
            window=8, lrs=(1e-3, 3e-3, 1e-2), hidden=64, d_chart=8, n_eras_max=8, probe_frac=0.2, c2_mode="prime", n_lags=2)
SMOKE = dict(PINS, n_traj=8, L=110, burnin=10, origins=(20, 30, 40, 50), train_steps=20, tune_steps=10, hidden=16)
# Estimator pins, one place: the final values Builder-model freezes land here. `law="ns2d"` builds NS2DLaw from the bed's
# grid, encoder and snapshot step (R = nu L + f from the lag stack); damping scales the increment (1.0 = undamped).
FCPA_PINS = dict(damping=1.0, law="ns2d")
SMOKE_SEEDS = (7001, 7002)  # never an eval seed: development runs stay off the eval beds
FREEZE_FILES = ("FREEZE.md", "FREEZE2.md", "FREEZE3.md")


class FreezeError(RuntimeError):
    pass


def verify_freeze(specs=SPECS):
    """Every `- <file> sha256 <hex>` line of FREEZE.md, FREEZE2.md and FREEZE3.md must match the file on disk; otherwise refuse."""
    for freeze in FREEZE_FILES:
        f = Path(specs) / freeze
        if not f.exists():
            raise FreezeError(f"{freeze} is missing")
        pairs = re.findall(r"^- (\S+) sha256 ([0-9a-f]{64})\s*$", f.read_text(encoding="utf-8"), re.M)
        if not pairs:
            raise FreezeError(f"{freeze} lists no hashes")
        for name, want in pairs:
            # line endings are not content: a pin taken on a CRLF checkout must hold on an LF one and back
            lf = (Path(specs) / name).read_bytes().replace(b"\r\n", b"\n")
            got = hashlib.sha256(lf).hexdigest()
            if want not in (got, hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()):
                raise FreezeError(f"{name} changed after the freeze: sha256 {got} != frozen {want}")
    return True


def check_seeds(seeds, smoke):
    """Development stays off the eval beds: smoke uses non-eval seeds; a full run uses exactly eval seeds."""
    seeds = tuple(seeds)
    if smoke and any(s in bars.EVAL_SEEDS or s == bars.TUNE_SEED for s in seeds):
        raise ValueError(f"--smoke must not use eval seeds {bars.EVAL_SEEDS} or tuning seed 3 (got {seeds})")
    if not smoke and sorted(seeds) != sorted(bars.EVAL_SEEDS):
        raise ValueError(f"a full run uses exactly the eval seeds {bars.EVAL_SEEDS} (got {seeds})")


def tuned_fingerprint(P, specs=SPECS, bed_seed=TUNE_BED_SEED):
    """sha256 over everything tuning depended on: FCPA_PINS, the pins in force (C2 mode excluded: tuning is
    independent of it), the bytes of every freeze file, and the tuning bed seed."""
    frz = [hashlib.sha256((Path(specs) / f).read_bytes()).hexdigest() for f in FREEZE_FILES]
    blob = json.dumps(dict(fcpa_pins=FCPA_PINS, pins={k: v for k, v in P.items() if k != "c2_mode"}, freeze=frz,
                           bed_seed=bed_seed), sort_keys=True, default=list)
    return hashlib.sha256(blob.encode()).hexdigest()


def load_tuned(path, P, specs=SPECS):
    """A frozen tuned.json is reused only if its fingerprint matches this configuration (a smoke-generated file never does)."""
    tuned = json.loads(Path(path).read_text())
    want = tuned_fingerprint(P, specs)
    if tuned.get("fingerprint") != want:
        raise FreezeError(f"{path}: tuned.json fingerprint {tuned.get('fingerprint')} != {want} for this configuration")
    tuned["asmodeous_macs_by_m"] = {int(k): v for k, v in tuned["asmodeous_macs_by_m"].items()}
    return tuned


def global_n_use(mode, B, D, p, width, n_pairs):
    """Train pairs Global fits on: prime = the full Train split; strict = what B_FCPA affords (>= F + 1)."""
    if mode == "prime":
        return n_pairs
    return max(baselines.global_n_afford(B, D, p, width, n_pairs), D + width + 1)


def asmodeous_steps(mode, B, step_macs, probe_cost, cap):
    """Adam steps: prime = the pinned cap; strict = floor((B - probe) / MACs per step), capped, never negative."""
    if mode == "prime":
        return int(cap)
    return int(min(cap, max(bars.steps_for_budget(B - probe_cost, step_macs), 0)))


# ------------------------------------------------------------------ the bed
def regime_path(L, seed, i, lam_sw, w=None):
    """Regime id per step of trajectory i of a bed: the 4-state chain (stay 1 - lam_sw, else uniform over the other 3)."""
    rng = np.random.default_rng([seed, i])
    sch = ns2d.regime_schedule(L, lam_sw, rng, laws=LAWS, w=w)
    reg = np.zeros(L, int)
    for s0, s1, c in zip(sch["starts"], sch["starts"][1:] + [L], sch["classes"]):
        reg[s0:min(s1, L)] = c
    return reg


def _gen(args):
    """One trajectory: its own schedule and initial condition, one simulate() call with n_universes = 1."""
    seed, i, P = args
    rng = np.random.default_rng([seed, i])
    sch = ns2d.regime_schedule(P["L"], P["lam_sw"], rng, laws=LAWS, w=P["w"])
    reg = np.zeros(P["L"], int)
    for s0, s1, c in zip(sch["starts"], sch["starts"][1:] + [P["L"]], sch["classes"]):
        reg[s0:min(s1, P["L"])] = c
    k, b = P["steps_per_snapshot"], P["burnin"] * P["steps_per_snapshot"]
    at = lambda j, s: 0 if j == 0 else b + k * s  # the first regime also runs through the burn-in
    ns = [(at(j, s), LAWS[c][0]) for j, (s, c) in enumerate(zip(sch["starts"], sch["classes"]))]
    fs = [(at(j, s), LAWS[c][1]) for j, (s, c) in enumerate(zip(sch["starts"], sch["classes"]))]
    out = ns2d.simulate(P["grid"], ns, fs, None, b + P["L"] * k, P["dt"], 1, rng)
    z, a = ns2d.encode(out["omega"][0, b::k][:P["L"]], P["kmax"])
    x = a[:, None] * z  # E1: the unnormalised coefficient vector a * psi
    return np.c_[x.real, x.imag], reg


def gen_bed(seed, P, workers=1, cache=None):
    """(X (n_traj, L, D), regime (n_traj, L)) for a bed seed; cached as npz when `cache` is a directory."""
    f = None if cache is None else Path(cache) / "beds" / f"bed_{seed}_{P['n_traj']}x{P['L']}_b{P['burnin']}.npz"
    if f is not None and f.exists():
        d = np.load(f)
        return d["X"], d["reg"]
    jobs = [(seed, i, P) for i in range(P["n_traj"])]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_gen, jobs, chunksize=1))
    else:
        res = [_gen(j) for j in jobs]
    X, reg = np.stack([r[0] for r in res]), np.stack([r[1] for r in res])
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, X=X, reg=reg)
    return X, reg


def three_way_split(n, seed):
    """Whole-trajectory Train / Cal / Test (128 / 64 / 64 at n = 256), conformal.trajectory_split with rng = seed."""
    ids = np.arange(n)
    tr, rest = trajectory_split(ids, 0.5, seed)
    ca, te = trajectory_split(ids[rest], 0.5, seed)
    return ids[tr], ids[rest][ca], ids[rest][te]


# ------------------------------------------------------------------ lag stack and reference steps
def lag_stack(X, n_lags):
    """(n, L, D) -> (n, L, n_lags D) rows [x_t, x_{t-1}, ...]; the first rows pad with x_0 (zero velocity). Every
    arm sees these rows, so the information is equal; time t is still row t."""
    X = np.asarray(X, float)
    L = X.shape[1]
    return np.concatenate([X[:, np.maximum(np.arange(L) - k, 0)] for k in range(n_lags)], axis=-1)


def first_block(out, D):
    """The next x from a prediction of width D (next x) or of width n_lags D (next stacked row: first block)."""
    return np.asarray(out, float)[:, :D]


def persistence_step(D):
    """Reference: x_hat_{t+1} = x_t."""
    return lambda rows: np.asarray(rows, float)[:, :D]


def constant_velocity_step(D):
    """Reference: x_hat_{t+1} = x_t + (x_t - x_{t-1}); fed back on itself it extrapolates the last true velocity."""
    return lambda rows: 2 * np.asarray(rows, float)[:, :D] - np.asarray(rows, float)[:, D:2 * D]


# ------------------------------------------------------------------ FCPA arm
class EraSet:
    """A conformal set of learned eras, scored in regimes: y in C iff some era in C maps to regime y; |C| = #eras."""

    def __init__(self, eras, emap):
        self.eras, self.emap = eras, emap

    def __contains__(self, y):
        return any(self.emap[k] == y for k in self.eras)

    def __len__(self):
        return len(self.eras)


def end_trajectory(est):
    """Close a trajectory: its part-filled tail window is consumed into the open era as a short window (every row
    kept, so partial_fit stays exact, L27'), then the buffer, the change-test history and the last observer symbol are
    cleared so that no window and no transition spans two trajectories."""
    st = est._st
    if len(st.bufX):
        est._consume(st, st.bufX, st.bufY, flush=True)
    st.bufX, st.bufY = st.bufX[:0], st.bufY[:0]
    st.hist, st.last_sym, st.last_x = [], None, None
    est._finalize()


def fit_fcpa(XL, P, seed, pins=None):
    """XL: lag-stacked Train trajectories (n, L, n_lags D). The estimator predicts increments (y = None), so the
    TT bins are step edges of the stacked rows. Pins come from FCPA_PINS (law, damping); one chunk per trajectory."""
    from fcpa.estimator import FCPARegressor
    from fcpa.laws import NS2DLaw

    pins = FCPA_PINS if pins is None else pins
    lags = "n_lags" in FCPARegressor().get_params()
    edges = FCPARegressor.make_step_edges(XL, n_lags=P["n_lags"]) if lags else FCPARegressor.make_step_edges(XL)
    kw = dict(n_eras_max=P["n_eras_max"], d_chart=P["d_chart"], alpha=bars.ALPHA, gamma=bars.GAMMA, window=P["w"],
              random_state=seed, y_edges=edges, damping=pins["damping"])
    if pins["law"] == "ns2d":
        kw["law"] = NS2DLaw(n=P["grid"], kmax=P["kmax"], dt=P["dt"] * P["steps_per_snapshot"])
    elif pins["law"] is not None:
        raise ValueError(f"unknown law pin {pins['law']!r}")
    if lags:
        kw["n_lags"] = P["n_lags"]
    est = FCPARegressor(**kw)
    for tr in XL:
        est.partial_fit(tr)
        end_trajectory(est)
    return est


def fcpa_scores(est, X):
    """Nonconformity of each row to each live era (the estimator's own router scores)."""
    _, phi, _ = est._encode(np.asarray(X, float))
    return est._nc_scores(est._model["segs"], phi)


def fcpa_stream(est, Xca, rca, Xte, rte, P):
    """Era map on Cal (frozen before Test is read), then P3 counts and the ACI run on the Test stream."""
    K, D = est.n_eras_, Xte.shape[-1]
    emap = bars.era_map(fcpa_scores(est, Xca.reshape(-1, D)).argmin(1), rca.reshape(-1), K)
    nc = fcpa_scores(est, Xte.reshape(-1, D))
    pred = np.array([emap[k] for k in nc.argmin(1)]).reshape(rte.shape)
    counts = [bars.id_counts(pred[i], rte[i], P["w"]) for i in range(len(rte))]
    cal = est._model["cal"]
    set_fn = lambda t, a: EraSet(cal.predict_set(nc[t], alpha=a) if cal is not None else {int(nc[t].argmin())}, emap)
    errs, sizes = bars.run_aci(set_fn, rte.reshape(-1), K, bars.ALPHA, bars.GAMMA, bars.ALPHA1)
    return dict(emap=emap, correct=[c for c, _ in counts], total=[t for _, t in counts], errs=errs, sizes=sizes,
                dev=bars.coverage_deviation(errs, bars.ALPHA), T=int(rte.size),
                size_traj=sizes.reshape(rte.shape).mean(1))


def fcpa_params(est):
    """Fitted parameters: TT (LDA W, intercepts, bin centres), observer V, every era model, crossing generators."""
    m, N = est._model, est._N
    tt = m["tt"]
    p, B, d = tt.sumx.shape
    gens = sum(N if np.allclose(g, np.diag(np.diag(g))) else N * N for g in est._gens)
    parts = dict(tt=p * B * (d + 2), observer=2 * (N + 1) * N,
                 eras=sum(e.W_.size + e.b_.size + np.size(e.chart) for e in m["eras"]), generators=gens)
    return dict(parts, total=sum(parts.values()))


def fcpa_inference_macs(est, sizes):
    """Mean MACs/step over the Test stream: encode + Born + TT + router over all live eras + the consulted eras
    (+ the damped CTC blend loop, at its a-priori iteration bound, when |C| >= 2)."""
    m, N, D = est._model, est._N, est.n_features_in_
    tt = m["tt"]
    p, B, _ = tt.sumx.shape
    E = len(m["eras"])
    rows = m["eras"][0].chart.shape[0] if E else 0
    fixed = 2 * N * (D + 1) + bars.macs_complex((N + 1) * N) + p * D * B + 2 * p * B + E * ((2 * N) ** 2 + 2 * N + 1)
    per_era = rows * 2 * N + rows * p + p
    iters = min(500, int(np.ceil(np.log(1e-9) / np.log(1 - est.eps))))
    blend = np.where(sizes >= 2, iters * (2 * sizes * p + sizes), 0)
    return float(fixed + per_era * sizes.mean() + blend.mean())


def fcpa_train_macs(est, n_pairs, n_calls, w):
    """Analytic MAC model of the estimator's fit: per-sample statistics, per-window change test / Gaussian
    inverse / era solve, per-partial_fit finalisation (chart eigh, chart solve, crossing expm, calibration
    scoring over <= 400 held-out rows). Closed-form solves at their cost; no gradient-trained parts."""
    m, N, D = est._model, est._N, est.n_features_in_
    p, B, _ = m["tt"].sumx.shape
    E, n2 = len(m["eras"]), 2 * N
    d_c = N if getattr(est, "n_lags", 1) > 1 else est.d_chart
    sample = 2 * N * (D + 1) + bars.macs_complex((N + 1) * N) + D * D + p * D + n2 * n2 + n2 * p + p * p
    window = 3 * w * n2 * n2 + n2 ** 3 + bars.macs_solve(n2 + 1, p)
    final = E * (bars.macs_solve(2 * d_c + 1, p) + 10 * N ** 3 + 20 * N ** 3) + 400 * E * n2 * n2
    return int(n_pairs * sample + (n_pairs // w) * window + n_calls * final)


# ------------------------------------------------------------------ closed-loop scoring
def rollout_errors(step, X, P, counter=None, n_lags=1):
    """Closed-loop 50-step rollout from the true history at every origin: relative error (n_traj, n_origins) in the
    raw encoded space. `step` maps rows [x_t, x_{t-1}, ...] (n_lags D) to the next x (D); the loop feeds each
    prediction back into the stack. Non-finite predictions are replaced by 0 (relative error 1) and counted."""
    n, _, D = X.shape
    o, H = np.array(P["origins"]), P["horizon"]
    cur = lag_stack(X, n_lags)[:, o].reshape(-1, n_lags * D)
    for _ in range(H):
        nxt = np.asarray(step(cur), float)
        bad = ~np.isfinite(nxt).all(1)
        if bad.any():
            if counter is not None:
                counter[0] += int(bad.sum())
            nxt[bad] = 0.0
        cur = np.concatenate([nxt, cur[:, :(n_lags - 1) * D]], axis=1)
    return bars.rel_err(cur[:, :D], X[:, o + H].reshape(-1, D)).reshape(n, len(o))


def crosses_switch(reg, P):
    """(n_traj, n_origins) bool: a true switch falls inside (t0, t0 + 50]."""
    o, H = np.array(P["origins"]), P["horizon"]
    sw = np.zeros(reg.shape, bool)
    sw[:, 1:] = reg[:, 1:] != reg[:, :-1]
    return np.stack([sw[:, t + 1:t + H + 1].any(1) for t in o], 1)


# ------------------------------------------------------------------ tuning (seed 3, bed seed 2003)
def tune(P, workers, out, device="cpu", log=print):
    X, reg = gen_bed(TUNE_BED_SEED, P, workers, out)
    tr, ca, te = three_way_split(len(X), bars.TUNE_SEED)
    nl, D = P["n_lags"], X.shape[-1]
    XL = lag_stack(X, nl)
    est = fit_fcpa(XL[tr], P, bars.TUNE_SEED)
    st = fcpa_stream(est, XL[ca], reg[ca], XL[te], reg[te], P)
    fcpa_inf, K = fcpa_inference_macs(est, st["sizes"]), est.n_eras_
    by_m = baselines.asmodeous_macs_by_m(nl * D, K, P["hidden"], D_out=D)
    m = bars.choose_m(by_m, fcpa_inf)
    scores = {}
    for lr in P["lrs"]:
        arm = baselines.AsmodeousArm(nl * D, K, m, lr, seed=bars.TUNE_SEED, device=device, batch=P["batch"],
                                     window=P["window"], hidden=P["hidden"], D_out=D).fit(XL[tr], P["tune_steps"])
        scores[lr] = float(rollout_errors(arm.step, X[ca], P, n_lags=nl).mean())
        log(f"  tune lr={lr:g}: val E50 {scores[lr]:.4f}")
    lr = min(scores, key=scores.get)
    return dict(lr=lr, m=m, K=K, fcpa_inf_macs=fcpa_inf, asmodeous_macs_by_m=by_m, val_e50={str(k): v for k, v in scores.items()})


# ------------------------------------------------------------------ one eval seed
def evaluate(seed, P, tuned=None, out=None, workers=1, device="cpu", log=print):
    assert seed != bars.TUNE_SEED, "tuning seed 3 is never evaluated"
    t0 = time.time()
    X, reg = gen_bed(seed, P, workers, out)
    tr, ca, te = three_way_split(len(X), seed)
    n, L, D = X.shape
    nl = P["n_lags"]
    XL, DL = lag_stack(X, nl), nl * D
    log(f"seed {seed}: bed {X.shape} ({time.time() - t0:.0f}s); train/cal/test = {len(tr)}/{len(ca)}/{len(te)}")
    est = fit_fcpa(XL[tr], P, seed)
    st = fcpa_stream(est, XL[ca], reg[ca], XL[te], reg[te], P)
    K, nonfinite = est.n_eras_, {k: [0] for k in ("fcpa", "global", "asmodeous")}
    n_pairs = len(tr) * (L - 1)
    B = fcpa_train_macs(est, n_pairs, len(tr), P["w"])
    fp = fcpa_params(est)
    fcpa_inf = fcpa_inference_macs(est, st["sizes"])
    log(f"  FCPA: {K} eras, {fp['total']} params, train {B:.3g} MACs, inference {fcpa_inf:.3g} MACs/step, T={st['T']}")
    err = {"fcpa": rollout_errors(lambda rows: first_block(est.predict(rows), D), X[te], P, nonfinite["fcpa"], nl)}
    # Global at matched parameters, trained under C2
    p = D  # inputs are the stacked rows (DL), outputs the next x (D)
    width = baselines.solve_width(fp["total"], DL, p)
    n_aff = baselines.global_n_afford(B, DL, p, width, n_pairs)
    n_use = global_n_use(P["c2_mode"], B, DL, p, width, n_pairs)
    keep = np.random.default_rng(seed).permutation(n_pairs)[:n_use]
    g = baselines.GlobalModel(DL, p, width, seed=seed).fit(XL[tr][:, :-1].reshape(-1, DL)[keep], X[tr][:, 1:].reshape(-1, D)[keep])
    g_used = g.fit_macs(n_use)
    err["global"] = rollout_errors(g.step, X[te], P, nonfinite["global"], nl)
    # ASMODEOUS top-m: same K as FCPA, tuned lr and m, steps = floor(B / MACs per step) under C2 (cap 5,000)
    arm = baselines.AsmodeousArm(DL, K, tuned["m"], tuned["lr"], seed=seed, device=device, batch=P["batch"],
                                 window=P["window"], hidden=P["hidden"], D_out=D)
    F = arm.feat_dim
    probe_n = int(np.clip(P["probe_frac"] * B // arm.probe_pair_macs(), 2 * F, 20_000))
    if P["c2_mode"] == "prime":
        probe_n = min(20_000, n_pairs)
    probe_cost = arm.probe_macs(probe_n)
    steps = asmodeous_steps(P["c2_mode"], B, arm.train_step_macs(), probe_cost, P["train_steps"])
    arm.fit(XL[tr], steps, probe_pairs=probe_n)
    a_used = steps * arm.train_step_macs() + arm.probe_macs()
    err["asmodeous"] = rollout_errors(arm.step, X[te], P, nonfinite["asmodeous"], nl)
    # references, no bar: x_hat = x_t0 at every step, and the last true velocity extrapolated
    err["persistence"] = rollout_errors(persistence_step(D), X[te], P, None, nl)
    err["constant_velocity"] = rollout_errors(constant_velocity_step(D), X[te], P, None, nl)
    cross = crosses_switch(reg[te], P)
    e50 = {a: e.mean(1) for a, e in err.items()}
    res = dict(
        seed=seed, n_lags=nl, input_dim=DL, output_dim=D, n_eras=K, emap={str(k): v for k, v in st["emap"].items()}, T=st["T"], width_global=width,
        e50={a: v.tolist() for a, v in e50.items()},
        e50_seed={a: float(v.mean()) for a, v in e50.items()},
        e50_within={a: float(e[~cross].mean()) if (~cross).any() else None for a, e in err.items()},
        e50_crossing={a: float(e[cross].mean()) if cross.any() else None for a, e in err.items()},
        id=dict(correct=st["correct"], total=st["total"], score=sum(st["correct"]) / max(sum(st["total"]), 1)),
        cov_dev=st["dev"], coverage=1 - float(st["errs"].mean()), mean_C=float(st["sizes"].mean()),
        mean_C_traj=st["size_traj"].tolist(),
        params=dict(fcpa=fp, fcpa_total=fp["total"], global_=g.n_params, asmodeous=arm.n_params),
        macs=dict(fcpa_inf=fcpa_inf, global_inf=g.inference_macs(), asmodeous_inf=arm.inference_macs(), fcpa_train=B,
                  global_train=g_used, asmodeous_train=a_used),
        c2_budget=dict(budget=B, global_samples=n_use, global_samples_afforded=n_aff, global_samples_available=n_pairs,
                       asmodeous_steps=steps, asmodeous_steps_cap=P["train_steps"], asmodeous_probe_pairs=probe_n),
        nonfinite_rollouts={a: c[0] for a, c in nonfinite.items()}, seconds=time.time() - t0)
    log(f"  E50 fcpa {res['e50_seed']['fcpa']:.4f} global {res['e50_seed']['global']:.4f} asmodeous {res['e50_seed']['asmodeous']:.4f}"
        f"; P3 {res['id']['score']:.3f}; |cov-0.9| {res['cov_dev']:.5f}; mean|C| {res['mean_C']:.2f}; asm steps {steps}")
    return res


# ------------------------------------------------------------------ the bars
def assemble(per_seed, tuned, P, B_boot):
    """Primary bars from the per-seed results (list ordered by seed)."""
    f = [np.array(r["e50"]["fcpa"]) for r in per_seed]
    glob = [np.array(r["e50"]["global"]) for r in per_seed]
    asm = [np.array(r["e50"]["asmodeous"]) for r in per_seed]
    p1, p2 = bars.p12(f, glob, B_boot), bars.p12(f, asm, B_boot)
    for q, a in ((p1, "global"), (p2, "asmodeous")):
        q["status"] = "PASS" if q["pass"] else "FAIL"
        q["per_seed_rho"] = [float(np.mean(x) / np.mean(y)) for x, y in zip(f, glob if a == "global" else asm)]
    c1s = {a: all(bars.c1(r["macs"]["fcpa_inf"], r["macs"][a + "_inf"]) for r in per_seed) for a in ("global", "asmodeous")}
    mode = P["c2_mode"]
    ratio_train = {a: float(np.mean([r["macs"][a + "_train"] / r["macs"]["fcpa_train"] for r in per_seed])) for a in ("global", "asmodeous")}
    if mode == "prime":  # C2': a reporting rule, no pass/fail, not in the verdict
        c2s = {a: True for a in ("global", "asmodeous")}
        cc2 = dict(status="REPORTED", train_macs_over_fcpa=ratio_train)
    else:  # the draft's C2, as a DIAGNOSTIC
        c2s = {a: all(bars.c2(r["macs"][a + "_train"], r["macs"]["fcpa_train"]) for r in per_seed) for a in ("global", "asmodeous")}
        cc2 = dict(per_baseline=c2s, train_macs_over_fcpa=ratio_train, status="PASS" if all(c2s.values()) else "FAIL")
    ids = [r["id"]["score"] for r in per_seed]
    p3 = bars.p3(ids)
    p3["lower95"] = bars.id_lower95([r["id"]["correct"] for r in per_seed], [r["id"]["total"] for r in per_seed], B_boot)
    devs = [r["cov_dev"] for r in per_seed]
    T = min(r["T"] for r in per_seed)
    pre = bars.p4_precondition(T)
    p4 = dict(devs=devs, T=T, bound=(max(bars.ALPHA1, 1 - bars.ALPHA1) + bars.GAMMA) / (bars.GAMMA * T),
              precondition=pre, status=("PASS" if bars.p4_pass(devs) else "FAIL") if pre else "VOID")
    p3["status"] = "PASS" if p3["pass"] else "FAIL"
    cc1 = dict(per_baseline=c1s, status="PASS" if all(c1s.values()) else "FAIL")
    verdict = bars.primary_verdict(p1["pass"], p2["pass"], all(c1s.values()), all(c2s.values()), p3["pass"],
                                   (p4["status"] == "PASS") if pre else None)
    kills = bars.kills(rho_global=p1["rho"], rho_asm=p2["rho"], c_global=c1s["global"] and c2s["global"],
                       c_asm=c1s["asmodeous"] and c2s["asmodeous"], id_mean=p3["mean"], cov_dev=devs)
    return dict(verdict=verdict, c2_mode=mode, diagnostic=mode == "strict",
                bars=dict(P1=p1, P2=p2, C1=cc1, C2=cc2, P3=p3, P4=p4), kills=kills)


# ------------------------------------------------------------------ report
def _md(res):
    b, S = res["bars"], res["seeds"]
    tag = "SMOKE, not evidence" if res["smoke"] else "full evaluation"
    diag = ", DIAGNOSTIC: the draft C2, not the verdict of record" if res["diagnostic"] else ""
    L = [f"# FCPA primary claim ({tag}; C2 mode {res['c2_mode']}{diag})", "",
         f"**Verdict: {res['verdict']}.** FREEZE hashes verified before the run ({', '.join(res['freeze_files'])}): {res['freeze_verified']}. "
         f"Seeds {list(S)}; tuning seed 3 never evaluated; bootstrap B = {res['B_boot']}, seed {bars.BOOT_SEED}.", "",
         "## Bars", "", "| bar | status | value |", "|---|---|---|"]
    for k in ("P1", "P2"):
        q = b[k]
        L.append(f"| {k} | {q['status']} | rho {q['rho']:.4f}, 95% upper {q['ub']:.4f}, wins {q['wins']}/{q['n_seeds']} |")
    L += [f"| C1 | {b['C1']['status']} | {b['C1']['per_baseline']} |", f"| {'C2prime' if res['c2_mode'] == 'prime' else 'C2 (diagnostic)'} | {b['C2']['status']} | baseline training MACs / FCPA's: {b['C2']['train_macs_over_fcpa']} |",
          f"| P3 | {b['P3']['status']} | seed-mean {b['P3']['mean']:.4f}, lower95 {b['P3']['lower95']:.4f}, worst {b['P3']['worst']:.4f} |",
          f"| P4 | {b['P4']['status']} | max dev {max(b['P4']['devs']):.5f} (bound {b['P4']['bound']:.5f}), T {b['P4']['T']}, precondition T>=950: {b['P4']['precondition']} |",
          "", "## Per seed", "", "| seed | eras | E50 fcpa | E50 global | E50 asmodeous | E50 persistence (ref) | E50 constant velocity (ref) | P3 | cov dev | mean abs C | asm steps |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in S.items():
        e = r["e50_seed"]
        L.append(f"| {s} | {r['n_eras']} | {e['fcpa']:.4f} | {e['global']:.4f} | {e['asmodeous']:.4f} | {e['persistence']:.4f} | {e['constant_velocity']:.4f} | {r['id']['score']:.3f} | "
                 f"{r['cov_dev']:.5f} | {r['mean_C']:.2f} | {r['c2_budget']['asmodeous_steps']} |")
    L += ["", "## Parameters and MACs (per seed)", "", "| seed | params fcpa / global / asmodeous | inference MACs fcpa / global / asmodeous | train MACs fcpa / global / asmodeous |", "|---|---|---|---|"]
    for s, r in S.items():
        p, m = r["params"], r["macs"]
        L.append(f"| {s} | {p['fcpa_total']} / {p['global_']} / {p['asmodeous']} | {m['fcpa_inf']:.3g} / {m['global_inf']:.3g} / {m['asmodeous_inf']:.3g} | "
                 f"{m['fcpa_train']:.3g} / {m['global_train']:.3g} / {m['asmodeous_train']:.3g} |")
    L += ["", "## Reported, no bar", "", "| seed | E50 within-era (fcpa/global/asm) | E50 crossing a switch (fcpa/global/asm) | non-finite rollout rows |", "|---|---|---|---|"]
    f3 = lambda d: " / ".join("n/a" if d[a] is None else f"{d[a]:.4f}" for a in ("fcpa", "global", "asmodeous"))
    for s, r in S.items():
        L.append(f"| {s} | {f3(r['e50_within'])} | {f3(r['e50_crossing'])} | {r['nonfinite_rollouts']} |")
    L += ["", f"Learned-era count at end of fit: {[r['n_eras'] for r in S.values()]}.", "",
          "## Tuning (seed 3, bed seed 2003)", "", f"```\n{json.dumps(res['tuned'], indent=1, default=float)}\n```", "",
          "## Kill rules (point estimates)", "", f"{res['kills']}", "",
          "## Secondary bars S1-S5", "", "NOT RUN by this driver (they need the learned-operator arm, null streams, certified-change events and the "
          "Benettin exponents). A bar that was not run is not a pass.", "", "## Limits", "",
          "- C2' (Amendment 11a): Global fits the full Train split and ASMODEOUS takes the pinned 5,000 steps; training MACs are reported, not equalised, and C2' has no pass/fail. "
          "Under `--c2 strict` the baselines are trained to B_FCPA (steps = min(5000, floor((B_FCPA - probe cost) / MACs per step))) and the run is a labelled DIAGNOSTIC.",
          "- Global is closed-form: one fit on the full Train split (strict: as many pairs as B_FCPA affords).",
          "- FCPA MACs come from an analytic model of the estimator internals (`fcpa_train_macs`, `fcpa_inference_macs`); the crossing S is folded into the era chart by the estimator.",
          "- Every arm sees the lag stack [x_t, x_{t-1}] (equal information); the first row of a trajectory pads with x_0, so T and the pair count are unchanged. FCPA models increments (y = None); the references are persistence and constant velocity (last true velocity extrapolated), neither a bar.",
          "- The ASMODEOUS arm is scored through a closed-form ridge probe from its predicted latent state to the next encoded state; the probe cost is charged.",
          "- Non-finite rollout rows are replaced by the zero vector (relative error 1) and counted."]
    if res["smoke"]:
        L += ["- SMOKE: tiny sizes (8 trajectories, L = 110, 4 origins, 2 seeds); T is far below 950, so P4 is VOID. Nothing here is evidence."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main
def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny sizes (< 2 min); P4 is VOID by construction")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--c2", choices=("prime", "strict"), default="prime",
                    help="prime: C2' of Amendment 11a (baselines at their pinned budgets). strict: the draft's C2 (baselines trained to B_FCPA), a labelled DIAGNOSTIC")
    ap.add_argument("--tune-only", action="store_true")
    ap.add_argument("--tuned", default=None, help="reuse a frozen tuned.json instead of tuning")
    a = ap.parse_args(argv)
    try:
        verify_freeze()
    except FreezeError as e:
        raise SystemExit(f"REFUSED: {e}")
    P = dict(SMOKE if a.smoke else PINS, c2_mode=a.c2)
    seeds = tuple(a.seeds) if a.seeds else (SMOKE_SEEDS if a.smoke else bars.EVAL_SEEDS)
    try:
        check_seeds(seeds, a.smoke)
    except ValueError as e:
        raise SystemExit(f"REFUSED: {e}")
    out = Path(a.out) if a.out else ROOT / "runs" / ("fcpa_primary_smoke" if a.smoke else "fcpa_primary")
    out.mkdir(parents=True, exist_ok=True)
    B_boot = 500 if a.smoke else bars.B_BOOT
    log = lambda s: print(s, flush=True)
    if not a.smoke:  # the precondition of P4 is checked before any model is scored
        T = (P["n_traj"] // 4) * P["L"]
        assert bars.p4_precondition(T), f"T = {T} < 950 voids P4"
    if a.tuned:
        try:
            tuned = load_tuned(a.tuned, P)
        except FreezeError as e:
            raise SystemExit(f"REFUSED: {e}")
    else:
        log("tuning on seed 3 (bed seed 2003)")
        tuned = dict(tune(P, a.workers, out, a.device, log), fingerprint=tuned_fingerprint(P))
        (out / "tuned.json").write_text(json.dumps(tuned, indent=1, default=_json))
    log(f"tuned: lr {tuned['lr']:g}, m {tuned['m']}, FCPA eras {tuned['K']}")
    if a.tune_only:
        return dict(tuned=tuned)
    per_seed = []
    for s in seeds:
        r = evaluate(s, P, tuned, out, a.workers, a.device, log)
        (out / f"seed_{s}.json").write_text(json.dumps(r, indent=1, default=_json))
        per_seed.append(r)
    res = dict(smoke=a.smoke, n_lags=P["n_lags"], freeze_verified=True, freeze_files=list(FREEZE_FILES), B_boot=B_boot, pins={k: v for k, v in P.items()}, tuned=tuned,
               seeds={str(r["seed"]): r for r in per_seed}, secondary={k: "NOT RUN" for k in ("S1", "S2", "S3", "S4", "S5")})
    res.update(assemble(per_seed, tuned, P, B_boot))
    (out / "results.json").write_text(json.dumps(res, indent=1, default=_json))
    (out / "report.md").write_text(_md(res), encoding="utf-8")
    log(f"VERDICT {res['verdict']}; wrote {out}")
    return res


if __name__ == "__main__":
    main()

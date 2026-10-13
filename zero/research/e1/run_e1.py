"""E1 runner (E1_PREREG.md). From zero/research/e1:

  python run_e1.py era            ERA (E0c bank + MOM, alpha 0.003) and ORACLE, numpy, CPU
  python run_e1.py online         GRU-ONLINE-S/L: tune lr, then evaluate
  python run_e1.py meta GRU-META  meta-train + evaluate (likewise TF-META)
  python run_e1.py score          bars -> RESULTS_E1.md
  python run_e1.py all            everything in order

Add --smoke for a tiny end-to-end check (out_smoke/, a few steps and seeds; numbers mean nothing).
"""
import json
import math
import os
import platform
import sys
import time
from multiprocessing import Pool

import numpy as np

import common as C

SMOKE = "--smoke" in sys.argv
ARGS = [a for a in sys.argv[1:] if a != "--smoke"]
OUT = C.OUT + ("_smoke" if SMOKE else "")
EVAL = C.EVAL[:1] if SMOKE else C.EVAL
TUNE = C.TUNE[::5] if SMOKE else C.TUNE
NEURAL = ("GRU-ONLINE-S", "GRU-ONLINE-L", "GRU-META", "TF-META")
LRS, SIZES = (1e-3, 3e-3, 1e-2), {"GRU-ONLINE-S": 16, "GRU-ONLINE-L": 64}
DELTA, W, HAZ, NMIN, ALPHA = 0.05, 32, 1 / 400, 64, 0.003  # E0c bank, MOM pick from E0e


def path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def env():
    import torch
    return {"torch": torch.__version__, "device": device(), "gpu": torch.cuda.get_device_name(0) if device() == "cuda"
            else None, "platform": platform.platform()}


# ---- ERA and ORACLE (numpy) ----

def era_job(args):
    import arms
    import deciders as D
    bed, seed = args
    x, r, _ = C.stream(bed, seed)
    t0 = time.time()
    bl, bvec = arms.bocpd(x, C.V, HAZ, return_vec=True)
    mom = D.Momentum(2, ALPHA)
    res = arms.fcpa(x, C.V, DELTA, W, ext=np.exp(-bl), merge="global", n_min=NMIN, deciders=[mom], ext_vec=bvec,
                    return_info=True)
    return bed, seed, {"ERA": float(res["dec"][mom].mean()), "ERA_s": time.time() - t0,
                       "ORACLE": float(arms.oracle(x, r, C.V).mean()), "eras": res["n_eras"]}


def run_era():
    with Pool() as pool:
        rows = pool.map(era_job, [(b, s) for b in C.BEDS for s in EVAL])
    json.dump({f"{b}/{s}": o for b, s, o in rows}, open(path("era.json"), "w"), indent=1)
    print(f"era: {len(rows)} streams")


# ---- online GRUs ----

def online_job(args):
    import torch
    from online import prequential
    bed, seed, hid, lr, dev = args
    if dev == "cpu":
        torch.set_num_threads(1)
    t0 = time.time()
    return float(prequential(C.stream(bed, seed)[0], hid, lr, dev).mean()), time.time() - t0


def run_online():
    from nets import GRUNet, n_params
    dev = device()
    out = {"env": env(), "tune": {}, "picks": {}, "eval": {}}

    def jobs(js):  # CPU: one stream per core; GPU: in order
        if dev == "cpu":
            with Pool() as pool:
                return pool.map(online_job, js)
        return [online_job(j) for j in js]

    grid = [(arm, hid, lr) for arm, hid in SIZES.items() for lr in LRS]
    res = jobs([(b, s, hid, lr, dev) for _, hid, lr in grid for b, s in TUNE])
    for g, (arm, hid, lr) in enumerate(grid):
        rows = res[g * len(TUNE):(g + 1) * len(TUNE)]
        per_bed = {b: np.mean([l for (bb, _), (l, _) in zip(TUNE, rows) if bb == b]) for b in dict.fromkeys(b for b, _ in TUNE)}
        out["tune"][f"{arm}/{lr}"] = float(np.mean(list(per_bed.values())))
    for arm in SIZES:
        out["picks"][arm] = min(LRS, key=lambda lr: out["tune"][f"{arm}/{lr}"])
    keys = [(arm, b, s) for arm in SIZES for b in C.BEDS for s in EVAL]
    res = jobs([(b, s, SIZES[arm], out["picks"][arm], dev) for arm, b, s in keys])
    for (arm, b, s), (l, sec) in zip(keys, res):
        out["eval"][f"{arm}/{b}/{s}"] = {"loss": l, "s": sec}
    out["params"] = {arm: n_params(GRUNet(h)) for arm, h in SIZES.items()}
    json.dump(out, open(path("online.json"), "w"), indent=1)
    print("online picks", out["picks"])


# ---- meta-trained arms ----

def gen(seed_base_n):
    base, i = seed_base_n
    return C.meta_stream(base + i, base).astype(np.uint8)


def meta_data():
    import meta
    steps = 3 if SMOKE else math.ceil(meta.TOKENS / (meta.B * meta.L))
    per = (20_000 - 1) // meta.L
    n = meta.B * math.ceil(steps / per)
    f = path(f"meta_train_{n}.npy")
    if not os.path.exists(f):
        with Pool() as pool:
            np.save(f, np.stack(pool.map(gen, [(C.TRAIN_SEED0, i) for i in range(n)], chunksize=16)))
            np.save(path("meta_val.npy"), np.stack(pool.map(gen, [(C.VAL_SEEDS[0], i) for i in
                                                                 range(2 if SMOKE else len(C.VAL_SEEDS))])))
    return np.load(f), np.load(path("meta_val.npy")), steps


def run_meta(arch):
    import torch
    import meta
    from nets import n_params, stream_loss
    data, val, steps = meta_data()
    dev = device()
    model, info = meta.train(arch, data, val, dev, path(f"{arch}.pt"), steps=steps, every=1 if SMOKE else 100)
    model.eval()
    info.update(env=env(), params=n_params(model), eval={})
    for b in C.BEDS:
        for s in EVAL:
            t0 = time.time()
            info["eval"][f"{b}/{s}"] = {"loss": float(stream_loss(arch, model, C.stream(b, s)[0], dev).mean()),
                                        "s": time.time() - t0}
    json.dump(info, open(path(f"meta_{arch}.json"), "w"), indent=1)
    torch.cuda.empty_cache() if dev == "cuda" else None


# ---- bars ----

def run_score():
    era = json.load(open(path("era.json")))
    on = json.load(open(path("online.json")))
    me = {a: json.load(open(path(f"meta_{a}.json"))) for a in ("GRU-META", "TF-META")}

    def loss(arm, b, s):
        if arm == "ERA":
            return era[f"{b}/{s}"]["ERA"]
        if arm in SIZES:
            return on["eval"][f"{arm}/{b}/{s}"]["loss"]
        return me[arm]["eval"][f"{b}/{s}"]["loss"]

    reg = {(a, b): np.array([loss(a, b, s) - era[f"{b}/{s}"]["ORACLE"] for s in EVAL]) for a in ("ERA",) + NEURAL
           for b in C.BEDS}
    need = math.ceil(0.8 * len(EVAL))
    L = [f"# E1 results: era bank vs context-reading models (fresh seeds {EVAL[0]}-{EVAL[-1]})", "",
         "Regret vs ORACLE, nats/token, mean over seeds. Online lr picks: "
         + ", ".join(f"{a} {on['picks'][a]}" for a in SIZES) + ".", "",
         "| arm | params | " + " | ".join(C.BEDS) + " |", "|---|---|" + "---|" * len(C.BEDS)]
    params = {"ERA": "counts", **on["params"], **{a: me[a]["params"] for a in me}}
    for a in ("ERA",) + NEURAL:
        L.append(f"| {a} | {params[a]} | " + " | ".join(f"{reg[a, b].mean():.4f}" for b in C.BEDS) + " |")
    L += ["", "| bar | bed | ERA ≤ 0.5 × best neural on | verdict |", "|---|---|---|---|"]
    verdict = {}
    for bar, b in (("E1-B1", "REC"), ("E1-B2", "CYC-S")):
        best = np.min(np.stack([reg[a, b] for a in NEURAL]), 0)
        k = int((reg["ERA", b] <= 0.5 * best).sum())
        verdict[bar] = k >= need
        L.append(f"| {bar} | {b} | {k}/{len(EVAL)} seeds | {'**pass**' if verdict[bar] else '**fail: C5 withdrawn on ' + b + '**'} |")
    L += ["", "Wall time: ERA " + f"{np.mean([v['ERA_s'] for v in era.values()]):.1f} s/stream (CPU); "
          + "; ".join(f"{a} meta-train {me[a]['train_s']:.0f} s, eval "
                      f"{np.mean([v['s'] for v in me[a]['eval'].values()]):.2f} s/stream" for a in me)
          + "; " + "; ".join(f"{a} {np.mean([v['s'] for k, v in on['eval'].items() if k.startswith(a + '/')]):.1f} s/stream"
                             for a in SIZES) + ".",
          f"Neural device: {on['env']['gpu'] or on['env']['device']}. Meta tokens: "
          + ", ".join(f"{a} {me[a]['tokens']:,} (best val {me[a]['best_val']:.4f})" for a in me) + ".", ""]
    open(path("RESULTS_E1.md"), "w").write("\n".join(L))
    json.dump({"verdict": verdict, "regret": {f"{a}/{b}": reg[a, b].tolist() for a, b in reg}},
              open(path("score.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    cmd = ARGS[0] if ARGS else "all"
    if cmd in ("era", "all"):
        run_era()
    if cmd in ("online", "all"):
        run_online()
    if cmd == "meta":
        run_meta(ARGS[1])
    if cmd == "all":
        for a in ("GRU-META", "TF-META"):
            run_meta(a)
    if cmd in ("score", "all"):
        run_score()

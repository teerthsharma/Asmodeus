"""Throughput report (ROADMAP stage 6): the era engine on E0 beds vs the numpy reference, and the era LM.

  python -m eralm.bench --out runs/bench.json [--streams 256] [--ref-seeds 2]
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import EraLMConfig, build, throughput
from . import filter as F
from .engine import EraBank, bocpd

E0 = Path(__file__).resolve().parents[1] / "research" / "e0"


def _e0(name):
    spec = importlib.util.spec_from_file_location(f"e0_{name}", E0 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"e0_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def engine(dev, S, ref_seeds):
    bed, arms, D = _e0("bed"), _e0("arms"), _e0("deciders")
    kinds = ["REC", "CYC", "NOREC"]
    xs = [bed.make_bed(1000 + s, kind=kinds[s % 3])[0] for s in range(S)]
    x = torch.tensor(np.stack(xs), device=dev)
    T = x.shape[1]
    t0 = time.perf_counter()
    pe = bocpd(x, bed.V, 1 / 400)
    _sync(dev)
    t1 = time.perf_counter()
    out = EraBank(bed.V, merge_rule="bf1").run(x, pe)
    _sync(dev)
    t2 = time.perf_counter()
    F.run(out, F.Decider("MOM", 0.003))
    _sync(dev)
    t3 = time.perf_counter()
    tok = S * (T - 1)
    r0 = time.perf_counter()                                     # reference: one process, MOM decider, same beds
    for s in range(ref_seeds):
        bl, bvec = arms.bocpd(xs[s], bed.V, 1 / 400, return_vec=True)
        arms.fcpa(xs[s], bed.V, 0.05, 32, ext=np.exp(-bl), merge="global", n_min=64, deciders=[D.Momentum(2, 0.003)],
                  ext_vec=bvec)
    ref_tps = ref_seeds * (T - 1) / (time.perf_counter() - r0)
    tps = tok / (t3 - t0)
    return {"device": str(dev), "streams": S, "tokens": tok, "bocpd_s": t1 - t0, "bank_s": t2 - t1, "filter_s": t3 - t2,
            "tokens_per_sec": tps, "reference_tokens_per_sec_1proc": ref_tps, "speedup_vs_1proc": tps / ref_tps}


def lm(dev, compile_):
    rows = []
    for name, kw in (("tt_only", dict(bank=False)), ("era", dict(bank=True))):
        for train in (True, False):
            for batch in (32, 64):
                cfg = EraLMConfig(compile=compile_, **kw)
                m = build(cfg, dev)
                if kw["bank"]:                                   # a bank with 8 eras, as after a real build
                    m.eval()
                    m.reset_streams(batch)
                    with torch.no_grad():
                        for _ in range(3):
                            x = torch.randint(0, cfg.vocab, (batch, 257), device=dev)
                            m.step(x[:, :-1], x[:, 1:])
                    m.reset_streams(None)
                    while m.bank.n_alive < 8:
                        k = int((~m.bank.alive).nonzero()[0])
                        m.bank.alive[k] = True
                        m.bank.rank[k] = 100 + k
                    m.online = train
                try:
                    r = throughput(m, batch=batch, seq=256, steps=10, train=train)
                except torch.cuda.OutOfMemoryError:
                    r = {"oom": True}
                rows.append({"model": name, "compile": compile_, "n_params": m.n_params(), "n_eras": m.bank.n_alive, **r})
                print(json.dumps(rows[-1]), flush=True)
                del m
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", type=int, default=256)
    ap.add_argument("--ref-seeds", type=int, default=2)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--skip-engine", action="store_true")
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    res = {"device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", "torch": torch.__version__}
    if not a.skip_engine:
        res["engine"] = engine(dev, a.streams, a.ref_seeds)
        print(json.dumps(res["engine"]), flush=True)
    res["lm"] = lm(dev, a.compile)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

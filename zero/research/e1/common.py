"""E1 shared pieces (E1_PREREG.md): beds, seeds, the e0 import path."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e0"))

from bed import V, make_bed  # noqa: E402

BEDS = {"REC": ("REC", {}), "CYC-S": ("CYC", {"seg_min": 20, "seg_mean": 100}), "NOREC": ("NOREC", {})}
META_KINDS = ("REC", "CYC-S", "NOREC")  # meta stream i uses META_KINDS[i % 3]: equal split
EVAL = list(range(50, 60))
TUNE = [("REC", s) for s in range(100, 105)] + [("CYC-S", s) for s in range(200, 205)] + [("NOREC", s) for s in range(300, 305)]
TRAIN_SEED0, VAL_SEEDS = 100_000, list(range(90_000, 90_020))
OUT = os.path.join(HERE, "out")


def stream(bed, seed):
    kind, kw = BEDS[bed]
    return make_bed(seed, kind=kind, **kw)


def meta_stream(seed, base):
    """Meta-training/validation stream: kind by position from its base seed."""
    return stream(META_KINDS[(seed - base) % 3], seed)[0]

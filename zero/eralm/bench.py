"""Benchmark and profiling CLI script for EraLM.

CLI Interface:
python -m eralm.bench [--device cpu|cuda] [--batch 32] [--seq 256] [--steps 20]
"""

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from eralm.config import EraLMConfig
from eralm.model import build, throughput


def main():
    parser = argparse.ArgumentParser(description="Profile and benchmark EraLM model performance")
    parser.add_argument("--device", type=str, default="cpu", help="Device to profile on (cpu or cuda)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--seq", type=int, default=256, help="Sequence length")
    parser.add_argument("--steps", type=int, default=20, help="Number of benchmark steps")
    parser.add_argument("--vocab", type=int, default=4096, help="Vocab size")
    parser.add_argument("--d", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")

    args = parser.parse_args()

    cfg = EraLMConfig(
        vocab=args.vocab,
        d=args.d,
        layers=args.layers,
    )

    model = build(cfg, device=args.device)

    print(f"Benchmarking EraLM on {args.device} ({model.n_params()} params)...")
    res_train = throughput(model, batch=args.batch, seq=args.seq, steps=args.steps, train=True)
    res_eval = throughput(model, batch=args.batch, seq=args.seq, steps=args.steps, train=False)

    print("--- Train Throughput ---")
    print(f"Tokens / sec: {res_train['tokens_per_sec']:.2f}")
    print(f"ms / step   : {res_train['ms_per_step']:.2f}")
    if args.device == "cuda":
        print(f"Peak Mem    : {res_train['peak_mem_bytes'] / (1024**2):.2f} MB")

    print("\n--- Eval Throughput ---")
    print(f"Tokens / sec: {res_eval['tokens_per_sec']:.2f}")
    print(f"ms / step   : {res_eval['ms_per_step']:.2f}")


if __name__ == "__main__":
    main()

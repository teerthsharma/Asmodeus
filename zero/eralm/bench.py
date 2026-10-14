"""CLI for profiling EraLM performance."""
import argparse
import torch

from eralm.config import EraLMConfig
from eralm.engine import throughput
from eralm.model import build


def main():
    parser = argparse.ArgumentParser(description="Profile EraLM throughput.")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--seq", type=int, default=256, help="Sequence length")
    parser.add_argument("--steps", type=int, default=10, help="Benchmark steps")
    args = parser.parse_args()

    cfg = EraLMConfig(vocab=4096, d=256, layers=4, device=args.device)
    model = build(cfg, device=args.device)

    res = throughput(model, batch=args.batch, seq=args.seq, steps=args.steps, train=True)
    print(f"Throughput on {args.device}: {res['tokens_per_sec']:.2f} tokens/sec ({res['ms_per_step']:.2f} ms/step)")
    print(f"Peak memory: {res['peak_mem_bytes'] / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()

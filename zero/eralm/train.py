"""CLI for training EraLM."""
import argparse
import os
import torch

from eralm.config import EraLMConfig
from eralm.engine import train_step
from eralm.model import build


def main():
    parser = argparse.ArgumentParser(description="Train EraLM.")
    parser.add_argument("--data", type=str, required=True, help="Data path")
    parser.add_argument("--sha256", type=str, required=True, help="Data SHA256")
    parser.add_argument("--tok", type=str, required=True, help="Tokenizer manifest path")
    parser.add_argument("--out", type=str, default="runs/era0", help="Output directory")
    parser.add_argument("--steps", type=int, default=10, help="Training steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    cfg = EraLMConfig(vocab=4096, d=256, layers=4, device=args.device)
    model = build(cfg, device=args.device)
    opt = model.configure_optimizer()

    print(f"Training EraLM on {args.device} ({model.n_params()} params)...")
    os.makedirs(args.out, exist_ok=True)

    dummy_batch = torch.randint(0, cfg.vocab, (4, 128), device=args.device)
    for step in range(1, args.steps + 1):
        info = train_step(model, opt, dummy_batch)
        if step % 2 == 0 or step == args.steps:
            print(f"Step {step}/{args.steps} | Loss: {info['loss']:.4f} | TT Loss: {info['tt_loss']:.4f}")

    save_path = os.path.join(args.out, "ckpt.pt")
    model.save(save_path, opt=opt, extra={"step": args.steps})
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()

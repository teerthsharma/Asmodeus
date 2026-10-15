"""Training CLI script for EraLM.

CLI Interface:
python -m eralm.train --data <data_path> --sha256 <hex> --tok <tok_manifest_or_json> --out runs/era0
"""

import argparse
import json
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from beds.lang import TokenStream, load_tokenizer, pin
from eralm.config import EraLMConfig
from eralm.model import build, train_step, eval_loss


def main():
    parser = argparse.ArgumentParser(description="Train EraLM model")
    parser.add_argument("--data", type=str, required=True, help="Path to text dataset")
    parser.add_argument("--sha256", type=str, required=True, help="Expected SHA256 hex hash of the dataset")
    parser.add_argument("--tok", type=str, required=True, help="Path to tokenizer json or manifest.json")
    parser.add_argument("--out", type=str, required=True, help="Output directory for checkpoints and logs")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--seq", type=int, default=256, help="Sequence length")
    parser.add_argument("--vocab", type=int, default=4096, help="Vocab size")
    parser.add_argument("--d", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on")

    args = parser.parse_args()

    # Pin data
    pin(args.data, args.sha256)

    tok_path = Path(args.tok)
    if tok_path.name == "manifest.json":
        tokenizer = load_tokenizer(tok_path, args.vocab, args.data)
    else:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tok_path))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EraLMConfig(
        vocab=args.vocab,
        d=args.d,
        layers=args.layers,
        lr=args.lr,
    )

    model = build(cfg, device=args.device)
    opt = model.configure_optimizer(lr=args.lr)

    train_stream = TokenStream(
        path=args.data,
        tokenizer=tokenizer,
        split="train",
        seq_len=args.seq,
        batch=args.batch,
        seed=42,
    )

    val_stream = TokenStream(
        path=args.data,
        tokenizer=tokenizer,
        split="val",
        seq_len=args.seq,
        batch=args.batch,
        seed=100,
    )

    print(f"Training EraLM ({model.n_params()} params) for {args.steps} steps...")

    for step, batch in enumerate(train_stream):
        if step >= args.steps:
            break
        metrics = train_step(model, opt, batch)
        if (step + 1) % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"Step {step+1}/{args.steps} | Loss: {metrics['loss']:.4f} | TT-Loss: {metrics['tt_loss']:.4f}")

    val_batches = [next(val_stream) for _ in range(5)]
    val_res = eval_loss(model, val_batches)
    print(f"Validation NLL: {val_res['nll']:.4f} | PPL: {val_res['ppl']:.2f}")

    checkpoint_path = out_dir / "ckpt.pt"
    model.save(str(checkpoint_path), opt=opt, extra={"val_res": val_res})
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

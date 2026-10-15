"""Tokenizer CLI script for EraLM.

CLI Interface:
python -m eralm.tok --data <data_path> --sha256 <hex> --out zero/data/tok [--sizes 4096]
"""

import argparse
import sys
from pathlib import Path

# Add repo root and zero to sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from beds.lang import build_tokenizers, pin


def main():
    parser = argparse.ArgumentParser(description="Train and pin BPE tokenizers for EraLM")
    parser.add_argument("--data", type=str, required=True, help="Path to text dataset")
    parser.add_argument("--sha256", type=str, required=True, help="Expected SHA256 hex hash of the dataset")
    parser.add_argument("--out", type=str, required=True, help="Output directory for tokenizers and manifest")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096], help="Vocab sizes to train")

    args = parser.parse_args()

    # Pin dataset hash
    pin(args.data, args.sha256)

    # Train tokenizers and create manifest
    man = build_tokenizers(args.data, args.out, sizes=tuple(args.sizes))
    print(f"Tokenizers built in {args.out}: {man}")


if __name__ == "__main__":
    main()

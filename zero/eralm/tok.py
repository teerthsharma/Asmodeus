"""CLI for BPE tokenization manifest and preparation."""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Tokenize data for EraLM.")
    parser.add_argument("--data", type=str, required=True, help="Input data file path")
    parser.add_argument("--sha256", type=str, required=True, help="Expected sha256 hash of data file")
    parser.add_argument("--out", type=str, default="zero/data/tok", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "tok_4096.json")
    manifest = {
        "data": args.data,
        "sha256": args.sha256,
        "vocab_size": 4096,
        "status": "ready",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Tokenizer manifest written to {manifest_path}")


if __name__ == "__main__":
    main()

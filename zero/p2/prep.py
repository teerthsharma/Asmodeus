"""Phase 2: tokenize TinyStories once into uint16 token files per lang-v1 split (train.bin, val.bin).

Each doc is encoded and followed by <|endoftext|>, docs in file order. The split is beds.lang.split_of
(sha256-keyed 90/5/5), the tokenizer is the pinned BPE-4096 in p2/tok. Writes <out>/<split>.bin and
<out>/meta.json (corpus sha, tokenizer sha, token counts).

  python p2/prep.py --data ../../Asmodeus/zero/data/TinyStories-train.txt --out ../../Asmodeus/zero/data/p2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beds.lang import EOT, iter_docs, load_tokenizer, split_of  # noqa: E402

CORPUS_SHA = "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f"
TOK = Path(__file__).parent / "tok" / "manifest.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=20000)
    a = ap.parse_args(argv)
    tok = load_tokenizer(TOK, 4096, a.data)          # pins corpus and tokenizer hashes
    assert json.loads(TOK.read_text())["corpus_sha256"] == CORPUS_SHA
    eot = tok.token_to_id(EOT)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: open(out / f"{s}.bin.tmp", "wb") for s in ("train", "val", "test")}
    counts = {s: 0 for s in files}
    docs = {s: 0 for s in files}
    buf = {s: [] for s in files}

    def flush(s):
        if buf[s]:
            enc = tok.encode_batch(buf[s])
            arr = np.fromiter((i for e in enc for i in (*e.ids, eot)), dtype=np.uint16)
            arr.tofile(files[s])
            counts[s] += arr.size
            docs[s] += len(buf[s])
            buf[s] = []

    for d in iter_docs(a.data):
        s = split_of(d)
        buf[s].append(d)
        if len(buf[s]) >= a.chunk:
            flush(s)
    for s in files:
        flush(s)
        files[s].close()
        (out / f"{s}.bin.tmp").replace(out / f"{s}.bin")
    man = json.loads(TOK.read_text())
    meta = dict(corpus_sha256=man["corpus_sha256"], tokenizer_sha256=man["tokenizers"]["4096"]["sha256"],
                split=man["split"], tokens=counts, docs=docs, dtype="uint16", eot=eot)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()

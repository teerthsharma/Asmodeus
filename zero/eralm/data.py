"""Token data: TinyStories (or any EOT-separated text) tokenised once into uint16 bins per lang-v1 split, pinned by
the corpus and tokenizer hashes; random windows for training, persistent parallel streams for the era bank and eval.

Bin format (shared with the phase-2 prep): <split>.bin = docs in file order, each followed by <|endoftext|>;
meta.json = corpus_sha256, tokenizer_sha256, split, token and doc counts, dtype, eot id.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from beds.lang import EOT, iter_docs, load_tokenizer, pin, split_of

TOK_DIR = Path(__file__).parent / "tok"
CORPUS_SHA = "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f"   # TinyStories V1 train


def tokenizer_sha():
    return json.loads((TOK_DIR / "manifest.json").read_text())["tokenizers"]["4096"]["sha256"]


def prep(data, out, chunk=20000):
    """Tokenise `data` into out/{train,val,test}.bin + meta.json (pins corpus + tokenizer)."""
    tok = load_tokenizer(TOK_DIR / "manifest.json", 4096, data)
    eot = tok.token_to_id(EOT)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = {s: open(out / f"{s}.bin.tmp", "wb") for s in ("train", "val", "test")}
    counts, docs, buf = {s: 0 for s in files}, {s: 0 for s in files}, {s: [] for s in files}

    def flush(s):
        if buf[s]:
            enc = tok.encode_batch(buf[s])
            arr = np.fromiter((i for e in enc for i in (*e.ids, eot)), dtype=np.uint16)
            arr.tofile(files[s])
            counts[s] += arr.size
            docs[s] += len(buf[s])
            buf[s] = []

    for d in iter_docs(data):
        s = split_of(d)
        buf[s].append(d)
        if len(buf[s]) >= chunk:
            flush(s)
    for s in files:
        flush(s)
        files[s].close()
        (out / f"{s}.bin.tmp").replace(out / f"{s}.bin")
    man = json.loads((TOK_DIR / "manifest.json").read_text())
    meta = dict(corpus_sha256=man["corpus_sha256"], tokenizer_sha256=tokenizer_sha(), split=man["split"],
                tokens=counts, docs=docs, dtype="uint16", eot=eot)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_bins(bins, corpus_sha=CORPUS_SHA):
    """-> (meta, {split: np.memmap uint16}); refuses bins built from another corpus or tokenizer."""
    bins = Path(bins)
    meta = json.loads((bins / "meta.json").read_text())
    if meta["corpus_sha256"] != corpus_sha:
        raise ValueError(f"bins corpus {meta['corpus_sha256']} != {corpus_sha}")
    if meta["tokenizer_sha256"] != tokenizer_sha():
        raise ValueError(f"bins tokenizer {meta['tokenizer_sha256']} != {tokenizer_sha()}")
    arrs = {}
    for s in ("train", "val", "test"):
        f = bins / f"{s}.bin"
        if f.exists():
            arrs[s] = np.memmap(f, dtype=np.uint16, mode="r")
            if s in meta["tokens"] and arrs[s].size != meta["tokens"][s]:
                raise ValueError(f"{f}: {arrs[s].size} tokens, meta says {meta['tokens'][s]}")
    return meta, arrs


class RandomWindows:
    """IID training batches: B windows of T+1 tokens at uniform offsets. Resumable (numpy Generator state)."""

    def __init__(self, arr, batch, seq, seed, device="cpu"):
        self.arr, self.batch, self.seq, self.device = arr, batch, seq, device
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self):
        off = self.rng.integers(0, self.arr.size - self.seq - 1, self.batch)
        x = np.stack([self.arr[o:o + self.seq + 1] for o in off]).astype(np.int64)
        t = torch.from_numpy(x)
        dev = torch.device(self.device)
        return t.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else t

    def state(self):
        return self.rng.bit_generator.state

    def load(self, st):
        self.rng.bit_generator.state = st


class Streams:
    """B persistent streams: the split cut into B contiguous shards; each call yields the next [B, T+1] window, and
    consecutive windows overlap by one token (so every token after the first is predicted exactly once)."""

    def __init__(self, arr, batch, seq, limit=None, device="cpu"):
        n = arr.size if limit is None else min(arr.size, limit)
        self.arr, self.batch, self.seq, self.device = arr, batch, seq, device
        self.shard = n // batch
        self.starts = np.arange(batch) * self.shard
        self.pos = 0

    def __len__(self):
        return (self.shard - 1) // self.seq

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos + self.seq + 1 > self.shard:
            raise StopIteration
        x = np.stack([self.arr[s + self.pos: s + self.pos + self.seq + 1] for s in self.starts]).astype(np.int64)
        self.pos += self.seq
        return torch.from_numpy(x).to(self.device)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()

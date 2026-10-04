"""Row 8: language-bed plumbing on a local text file (streaming docs, hash-keyed splits,
held-out-clean BPE, pinned corpus hash, n-gram contamination, resumable token stream)."""
import codecs
import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

EOT = "<|endoftext|>"
SPLIT_SALT, SPLIT_FRACTIONS = "lang-v1", (0.90, 0.05, 0.05)


def iter_docs(path, sep=EOT, chunk_bytes=1 << 20):
    """Stream stripped, non-empty documents; holds at most one chunk plus one partial doc."""
    dec = codecs.getincrementaldecoder("utf-8")()
    rest = ""
    with open(path, "rb") as f:
        while True:
            raw = f.read(chunk_bytes)
            rest += dec.decode(raw, final=not raw)
            *done, rest = rest.split(sep)       # last piece may hold a separator prefix: keep it
            for d in done:
                if d.strip():
                    yield d.strip()
            if not raw:
                break
    if rest.strip():
        yield rest.strip()


def normalise(text):
    return " ".join(text.split())


def split_of(doc, fractions=SPLIT_FRACTIONS, salt=SPLIT_SALT):
    h = hashlib.sha256((salt + normalise(doc)).encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    acc = 0.0
    for name, fr in zip(("train", "val", "test"), fractions):
        acc += fr
        if u < acc:
            return name
    return "test"                               # float slack at the top edge


def train_bpe(path, vocab_size, out, splits=("train",), specials=(EOT, "<pad>")):
    """Byte-level BPE trained only on docs whose split is in `splits`."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=list(specials),
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.train_from_iterator((d for d in iter_docs(path) if split_of(d) in splits), trainer)
    tok.save(str(out))
    return tok


def corpus_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def pin(path, expected):
    got = corpus_hash(path)
    if got != expected:
        raise ValueError(f"corpus hash mismatch for {path}: {got} != {expected}")


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_tokenizers(path, out_dir, sizes=(1024, 4096)):
    """One BPE per size, all trained on the same train split; hash-pinned manifest -> returned dict."""
    out_dir = Path(out_dir)
    toks = {}
    for v in sizes:
        f = out_dir / f"tok_{v}.json"
        train_bpe(path, v, f)
        toks[str(v)] = {"file": f.name, "sha256": _file_sha(f)}
    man = {"corpus_sha256": corpus_hash(path), "split": {"salt": SPLIT_SALT, "fractions": list(SPLIT_FRACTIONS)},
           "tokenizers": toks}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def load_tokenizer(manifest_path, vocab, data):
    manifest_path = Path(manifest_path)
    man = json.loads(manifest_path.read_text())
    ent = man["tokenizers"][str(vocab)]                        # KeyError for an unknown size
    pin(data, man["corpus_sha256"])
    f = manifest_path.parent / ent["file"]
    if _file_sha(f) != ent["sha256"]:
        raise ValueError(f"tokenizer hash mismatch for {f}")
    return Tokenizer.from_file(str(f))


def _ngrams(doc, n):
    w = normalise(doc).lower().split()
    return {hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


def contamination(train_docs, eval_docs, n=13):
    """Fraction of eval docs sharing >=1 word n-gram with any train doc."""
    # ponytail: in-memory set of n-gram hashes; swap for a Bloom filter or sharded set past ~1e8 n-grams.
    seen = set()
    for d in train_docs:
        seen |= _ngrams(d, n)
    eval_docs = list(eval_docs)
    hit = sum(1 for d in eval_docs if not seen.isdisjoint(_ngrams(d, n)))
    return hit / len(eval_docs) if eval_docs else 0.0


class TokenStream:
    """Iterator of LongTensor [batch, seq_len+1] from one split; deterministic, wraps, resumable."""

    SHUFFLE = 256

    def __init__(self, path, tokenizer, split, seq_len, batch, seed):
        self.path, self.tok, self.split = path, tokenizer, split
        self.seq_len, self.batch = seq_len, batch
        self.eot = tokenizer.token_to_id(EOT)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.epoch, self.consumed, self.buf, self.toks = 0, 0, [], []
        self._open()

    def _open(self):
        self._gen = (d for d in iter_docs(self.path) if split_of(d) == self.split)

    def _pull(self):
        d = next(self._gen, None)
        if d is not None:
            self.consumed += 1
        return d

    def _next_doc(self):
        while len(self.buf) < self.SHUFFLE and (d := self._pull()) is not None:
            self.buf.append(d)
        if not self.buf:                        # epoch over: reopen and reshuffle with the same rng
            self.epoch, self.consumed = self.epoch + 1, 0
            self._open()
            while len(self.buf) < self.SHUFFLE and (d := self._pull()) is not None:
                self.buf.append(d)
            if not self.buf:
                raise ValueError(f"no documents in split {self.split!r}")
        return self.buf.pop(int(self.rng.integers(len(self.buf))))

    def __iter__(self):
        return self

    def __next__(self):
        need = self.batch * (self.seq_len + 1)
        while len(self.toks) < need:
            self.toks += self.tok.encode(self._next_doc()).ids + [self.eot]
        out, self.toks = self.toks[:need], self.toks[need:]
        return torch.tensor(out, dtype=torch.long).view(self.batch, self.seq_len + 1)

    def state(self):
        # ponytail: resume re-reads the file to skip `consumed` split docs, O(file) per load;
        # store a byte offset instead if corpora get large enough for that to matter.
        return copy.deepcopy({"epoch": self.epoch, "consumed": self.consumed, "buf": self.buf,
                              "toks": self.toks, "rng": self.rng.bit_generator.state})

    def load(self, st):
        st = copy.deepcopy(st)
        self.epoch, self.consumed, self.buf, self.toks = st["epoch"], st["consumed"], st["buf"], st["toks"]
        self.rng.bit_generator.state = st["rng"]
        self._open()
        for _ in range(self.consumed):
            next(self._gen)

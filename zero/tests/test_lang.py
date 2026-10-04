"""Row 8 (ACHIEVABLE.md): language bed plumbing on a local text file. No downloads."""
from pathlib import Path

import pytest

from beds.lang import iter_docs, split_of, train_bpe, corpus_hash, pin, contamination, TokenStream

FIX = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"
EOT = "<|endoftext|>"


def docs():
    return list(iter_docs(FIX))


def test_streaming_reader_matches_full_read():
    full = [d.strip() for d in FIX.read_text(encoding="utf-8").split(EOT) if d.strip()]
    assert docs() == full and len(full) == 402
    assert list(iter_docs(FIX, chunk_bytes=37)) == full                     # chunk boundary inside separators


def test_splits_disjoint_deterministic_document_keyed():
    ds = docs()
    sp = [split_of(d) for d in ds]
    assert sp == [split_of(d) for d in ds] and set(sp) <= {"train", "val", "test"}
    assert split_of(ds[3]) == split_of(ds[-1]) == split_of(ds[-2])          # duplicates share a split
    fr = {s: sp.count(s) / len(sp) for s in ("train", "val", "test")}
    assert 0.85 < fr["train"] < 0.95 and fr["val"] > 0.02 and fr["test"] > 0.02  # default 0.90/0.05/0.05
    assert split_of(ds[0] + " ") == split_of(ds[0])                          # keyed on normalised text


def test_tokenizer_never_sees_heldout_docs(tmp_path):
    ds = docs()
    leak = [d for d in (f"The glorbix {i} glorbix glorbix sang." for i in range(400)) if split_of(d) != "train"]
    assert len(leak) > 20
    corpus = tmp_path / "c.txt"
    corpus.write_text(f"\n{EOT}\n".join(ds + leak) + f"\n{EOT}\n", encoding="utf-8")
    tok = train_bpe(corpus, vocab_size=400, out=tmp_path / "tok.json")    # 300 = 256 bytes+2 specials+42 merges: control cannot bite
    assert not any("glorbix" in t for t in tok.get_vocab())                  # val/test text never trained on
    every = train_bpe(corpus, vocab_size=400, out=tmp_path / "all.json", splits=("train", "val", "test"))
    assert any("glorbix" in t for t in every.get_vocab())                    # control: the test bites
    assert tok.token_to_id(EOT) is not None and (tmp_path / "tok.json").exists()
    assert tok.decode(tok.encode("Tom saw the kite.").ids) == "Tom saw the kite."


def test_hash_pin(tmp_path):
    h = corpus_hash(FIX)
    assert len(h) == 64 and h == corpus_hash(FIX)
    pin(FIX, h)
    other = tmp_path / "x.txt"
    other.write_bytes(FIX.read_bytes() + b"x")
    with pytest.raises(ValueError):
        pin(other, h)


def test_contamination():
    ds = docs()
    train = [d for d in ds if split_of(d) == "train"]
    held = [d for d in ds if split_of(d) != "train"]
    assert contamination(train, train[:10], n=8) == 1.0
    assert contamination(train, ["zz yy xx ww vv uu tt ss rr qq pp"], n=8) == 0.0
    c = contamination(train, held, n=8)                                      # templated stories share 8-grams:
    assert 0.0 <= c <= 1.0                                                   # reported, never silently zero


def test_token_stream_split_shapes_determinism_resume(tmp_path):
    tok = train_bpe(FIX, vocab_size=300, out=tmp_path / "tok.json")
    mk = lambda split, seed: TokenStream(FIX, tok, split=split, seq_len=32, batch=4, seed=seed)
    s = mk("train", 0)
    b = [next(s) for _ in range(3)]
    assert b[0].shape == (4, 33) and int(b[0].max()) < tok.get_vocab_size() and int(b[0].min()) >= 0
    s2 = mk("train", 0)
    assert all((x == next(s2)).all() for x in b)                             # deterministic per seed
    assert not (next(mk("train", 1)) == b[0]).all()
    s3 = mk("train", 0)
    next(s3); st = s3.state()
    want = next(s3)
    s4 = mk("train", 0); s4.load(st)
    assert (next(s4) == want).all()                                          # resumable cursor (checkpoints)
    val_text = tok.decode(next(mk("val", 0))[0].tolist())
    held_stories = {d.split("The end of story ")[-1] for d in docs() if split_of(d) == "val"}
    nums = [w.rstrip(".") for w in val_text.split("story ")[1:]]
    assert all(n.split()[0].rstrip(".") in {h.rstrip(".") for h in held_stories} for n in nums if n)

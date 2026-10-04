import sys

import pytest
import torch

import chat
from beds.lang import train_bpe


def _ckpt(tmp_path):
    from lm.cores import build

    m = build("gpt", 256, 32, layers=1)
    p = tmp_path / "c.pt"
    torch.save({"core": "gpt", "vocab": 256, "d": 32, "kw": {"layers": 1}, "state_dict": m.state_dict()}, p)
    return str(p)


def test_load_tokenizer_roundtrip(tmp_path):
    corp = tmp_path / "c.txt"
    corp.write_text("".join(f"Once upon a time there was story {i}. The end.<|endoftext|>" for i in range(40)))
    out = tmp_path / "tok.json"
    train_bpe(str(corp), 300, out, splits=("train", "val", "test"))
    tok = chat.load_tokenizer(str(out))
    assert tok.decode(tok.encode("Once upon a time.")) == "Once upon a time."


def _run(monkeypatch, tmp_path, inputs):
    it = iter(inputs)

    def fake_input(prompt=""):
        v = next(it)
        if isinstance(v, BaseException):
            raise v
        return v

    monkeypatch.setattr(sys, "argv", ["chat", _ckpt(tmp_path), "--temperature", "0", "--max-new", "4", "--seed", "0"])
    monkeypatch.setattr("builtins.input", fake_input)
    chat.main()


def test_main_turn_then_quit(monkeypatch, tmp_path, capsys):
    _run(monkeypatch, tmp_path, ["hello", "/quit"])
    assert "think steps" in capsys.readouterr().out


def test_main_eof_exits_cleanly(monkeypatch, tmp_path, capsys):
    _run(monkeypatch, tmp_path, ["hello", EOFError()])
    assert "think steps" in capsys.readouterr().out
    _run(monkeypatch, tmp_path, [EOFError()])

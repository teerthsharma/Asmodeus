"""Unit tests for Stage 6 EraLM implementation."""

import sys
import tempfile
from pathlib import Path

import torch
import pytest

# Ensure zero is on sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from eralm import EraLMConfig, EraLM, build, train_step, eval_loss, throughput, generate


def test_eralm_config():
    cfg = EraLMConfig(vocab=512, d=128, layers=2)
    assert cfg.vocab == 512
    assert cfg.d == 128
    assert cfg.layers == 2
    assert cfg.alpha == 0.003


def test_eralm_build_and_forward():
    cfg = EraLMConfig(vocab=100, d=32, layers=2)
    model = build(cfg, device="cpu")
    assert isinstance(model, EraLM)

    tokens = torch.randint(0, 100, (2, 16))
    model.train()
    out = model(tokens)

    assert "nll" in out
    assert "tt_nll" in out
    assert "logits" in out
    assert out["nll"].shape == (2, 16)
    assert out["tt_nll"].shape == (2, 16)
    assert out["logits"].shape == (2, 16, 100)


def test_eralm_metrics_and_flops():
    cfg = EraLMConfig(vocab=100, d=32, layers=2)
    model = build(cfg)

    assert model.n_params() > 0
    assert model.n_stats() >= 0
    assert model.active_flops_per_token(128) > 0


def test_eralm_train_step_and_eval_loss():
    cfg = EraLMConfig(vocab=100, d=32, layers=2)
    model = build(cfg)
    opt = model.configure_optimizer()

    batch = torch.randint(0, 100, (2, 16))
    train_res = train_step(model, opt, batch)

    assert "loss" in train_res
    assert "tt_loss" in train_res
    assert "n_eras" in train_res
    assert "single_frac" in train_res
    assert "lr" in train_res

    eval_res = eval_loss(model, [batch])
    assert "nll" in eval_res
    assert "ppl" in eval_res
    assert "tt_nll" in eval_res
    assert "tt_ppl" in eval_res
    assert eval_res["tokens"] == 2 * 15


def test_eralm_throughput_and_generate():
    cfg = EraLMConfig(vocab=100, d=32, layers=2)
    model = build(cfg)

    tp_res = throughput(model, batch=2, seq=16, steps=2, train=False)
    assert "tokens_per_sec" in tp_res
    assert "ms_per_step" in tp_res

    prompt = torch.randint(0, 100, (1, 5))
    gen_tokens = generate(model, prompt, n_new=5, temperature=1.0)
    assert gen_tokens.shape == (1, 10)


def test_eralm_save_and_load():
    cfg = EraLMConfig(vocab=100, d=32, layers=2)
    model = build(cfg)
    tokens = torch.randint(0, 100, (2, 16))
    model.train()
    _ = model(tokens)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(Path(tmpdir) / "ckpt.pt")
        model.save(ckpt_path)

        loaded_model = EraLM.load(ckpt_path)
        assert loaded_model.cfg.vocab == 100
        assert loaded_model.n_stats() == model.n_stats()

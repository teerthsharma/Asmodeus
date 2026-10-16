import os
import tempfile
import torch
import pytest

from eralm import EraLMConfig, build, train_step, eval_loss, throughput, generate

def test_eralm_config_defaults():
    cfg = EraLMConfig()
    assert cfg.vocab == 4096
    assert cfg.d == 256
    assert cfg.layers == 4
    assert cfg.n_heads == 8

def test_eralm_build_and_forward():
    cfg = EraLMConfig(vocab=100, d=32, layers=2, n_heads=2)
    model = build(cfg, device="cpu")
    assert model.n_params() > 0
    assert model.n_stats() == 0

    batch = torch.randint(0, 100, (2, 16))
    out = model(batch)
    assert "nll" in out
    assert "tt_nll" in out
    assert out["nll"].shape == (2, 16)
    assert out["tt_nll"].shape == (2, 16)

def test_eralm_train_eval_throughput_generate():
    cfg = EraLMConfig(vocab=100, d=32, layers=2, n_heads=2)
    model = build(cfg, device="cpu")
    opt = model.configure_optimizer(lr=1e-3)

    batch = torch.randint(0, 100, (4, 32))
    metrics = train_step(model, opt, batch)
    assert "loss" in metrics
    assert "tt_loss" in metrics
    assert metrics["loss"] > 0.0
    assert model.n_stats() > 0

    eval_m = eval_loss(model, [batch, batch])
    assert "nll" in eval_m
    assert "ppl" in eval_m
    assert eval_m["tokens"] == 4 * 31 * 2

    tp = throughput(model, batch=2, seq=16, steps=5, train=True)
    assert tp["tokens_per_sec"] > 0
    assert tp["ms_per_step"] > 0

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    gen = generate(model, prompt, n_new=5)
    assert gen.shape == (1, 8)

def test_eralm_checkpoint():
    cfg = EraLMConfig(vocab=100, d=32, layers=2, n_heads=2)
    model = build(cfg, device="cpu")
    opt = model.configure_optimizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.pt")
        model.save(ckpt_path, opt=opt, extra={"step": 10})
        assert os.path.exists(ckpt_path)

        loaded_model = model.__class__.load(ckpt_path, device="cpu")
        assert loaded_model.cfg.vocab == cfg.vocab
        assert loaded_model.n_params() == model.n_params()

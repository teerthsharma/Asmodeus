"""Unit tests for EraLM (zero/eralm module)."""
import os
import tempfile
import torch

from eralm import EraLMConfig, build, train_step, eval_loss, throughput, generate
from eralm.model import EraLM


def test_eralm_config_defaults():
    cfg = EraLMConfig()
    assert cfg.vocab == 4096
    assert cfg.d == 256
    assert cfg.layers == 4
    assert cfg.alpha == 0.003


def test_eralm_build_and_counters():
    cfg = EraLMConfig(vocab=128, d=64, layers=2)
    model = build(cfg, device="cpu")
    assert isinstance(model, EraLM)
    assert model.n_params() > 0
    assert model.n_stats() == 0
    assert model.active_flops_per_token(128) > 0


def test_eralm_forward_and_train_step():
    cfg = EraLMConfig(vocab=128, d=64, layers=2)
    model = build(cfg, device="cpu")
    opt = model.configure_optimizer(lr=1e-3)

    tokens = torch.randint(0, cfg.vocab, (2, 16))
    out = model(tokens)
    assert "nll" in out
    assert "tt_nll" in out
    assert out["nll"].shape == (2, 16)
    assert out["tt_nll"].shape == (2, 16)

    res = train_step(model, opt, tokens)
    assert "loss" in res
    assert "tt_loss" in res
    assert res["loss"] >= 0.0
    assert model.n_stats() >= 0


def test_eralm_eval_loss_and_throughput():
    cfg = EraLMConfig(vocab=128, d=64, layers=2)
    model = build(cfg, device="cpu")

    batches = [torch.randint(0, cfg.vocab, (2, 16)) for _ in range(2)]
    eval_res = eval_loss(model, batches)
    assert "nll" in eval_res
    assert "ppl" in eval_res
    assert eval_res["tokens"] == 2 * 2 * 15

    bench_res = throughput(model, batch=2, seq=16, steps=2, train=True)
    assert "tokens_per_sec" in bench_res
    assert bench_res["tokens_per_sec"] > 0


def test_eralm_generate():
    cfg = EraLMConfig(vocab=128, d=64, layers=2)
    model = build(cfg, device="cpu")

    prompt = torch.randint(0, cfg.vocab, (1, 8))
    generated = generate(model, prompt, n_new=4, temperature=1.0)
    assert generated.shape == (1, 12)
    assert torch.equal(generated[:, :8], prompt)


def test_eralm_save_load():
    cfg = EraLMConfig(vocab=128, d=64, layers=2)
    model = build(cfg, device="cpu")

    tokens = torch.randint(0, cfg.vocab, (2, 16))
    model.train()
    _ = model(tokens) # update stats

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        model.save(ckpt_path, extra={"test_key": "test_val"})

        loaded_model = EraLM.load(ckpt_path, device="cpu")
        assert loaded_model.cfg.vocab == cfg.vocab
        assert loaded_model.n_params() == model.n_params()
        assert loaded_model.n_stats() == model.n_stats()

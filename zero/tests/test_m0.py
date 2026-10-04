"""M0 (END_GOAL.md §5): LM cores + harness. CPU, fixture corpus, tiny configs."""
import json
import sys
from pathlib import Path

import pytest
import torch

torch.set_num_threads(4)

from beds.lang import TokenStream, train_bpe  # noqa: E402
from lm.cores import build, match_params  # noqa: E402
import m0  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"
V = 300


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    return train_bpe(FIX, V, tmp_path_factory.mktemp("tok") / "tok.json")


class DummyCore(torch.nn.Module):
    """External-core plug-in proof: causal running mean of embeddings; returns its own nll."""

    def __init__(self, vocab, d, **kw):
        super().__init__()
        self.emb, self.head = torch.nn.Embedding(vocab, d), torch.nn.Linear(d, vocab)

    def forward(self, tokens):
        e = self.emb(tokens)
        h = e.cumsum(1) / torch.arange(1, tokens.shape[1] + 1)[None, :, None]
        lp = self.head(h).log_softmax(-1)
        nll = torch.zeros(tokens.shape)
        nll[:, 1:] = -lp[:, :-1].gather(-1, tokens[:, 1:, None])[..., 0]
        return {"logits": lp, "nll": nll}

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def active_flops_per_token(self, seq=256):
        return 2 * self.n_params()


DUMMY = "test_m0:DummyCore"


def mk(core, seed=0):
    torch.manual_seed(seed)
    return build(core, V, 32, **({"layers": 2, "max_seq": 33} if core == "gpt" else {})).eval()


@pytest.mark.parametrize("core", ["gpt", DUMMY])
def test_causality(core):
    lm = mk(core)
    x = torch.randint(0, V, (2, 24))
    y = x.clone()
    y[:, 13:] = torch.randint(0, V, (2, 11))
    with torch.no_grad():
        a, b = lm(x), lm(y)
    assert a["logits"].shape == (2, 24, V)
    assert torch.allclose(a["logits"][:, :13], b["logits"][:, :13], atol=1e-5)
    assert not torch.allclose(a["logits"][:, 13:], b["logits"][:, 13:], atol=1e-3)   # control: it bites


def test_asmo_is_stub():
    with pytest.raises(NotImplementedError, match="QF-LM"):
        build("asmo", V, 32)


def test_external_core_nll_used_by_harness(tok):
    lm = mk(DUMMY)
    batch = torch.randint(0, V, (2, 9))
    out = lm(batch)
    assert torch.allclose(m0.token_nll(lm, batch).mean(), out["nll"][:, 1:].mean())
    assert isinstance(lm, DummyCore)


def test_match_params_within_5pct():
    for core, kw in (("gpt", dict(layers=2)), (DUMMY, {})):
        d, n = match_params(core, 4e6, 4096, **kw)
        assert d % 8 == 0 and abs(n / 4e6 - 1) < 0.05, (core, d, n)


@pytest.mark.parametrize("core", ["gpt", DUMMY])
def test_training_lowers_loss(core, tok):
    lm = mk(core)
    st = TokenStream(FIX, tok, "train", 32, 8, seed=0)
    losses = m0.train(lm, st, steps=30, lr=3e-3, warmup=3)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.3


@pytest.mark.parametrize("core", ["gpt", DUMMY])
def test_resume_determinism(core, tok, tmp_path):
    def run(stop_at=None):
        lm = mk(core)
        st = TokenStream(FIX, tok, "train", 32, 4, seed=1)
        m0.train(lm, st, steps=10, lr=3e-3, warmup=2, ckpt=tmp_path / f"{core}.pt", stop_at=stop_at)
        return lm
    (tmp_path / "x").mkdir()
    straight = mk(core)
    m0.train(straight, TokenStream(FIX, tok, "train", 32, 4, seed=1), steps=10, lr=3e-3, warmup=2)
    run(stop_at=5)
    resumed = run()
    for a, b in zip(straight.parameters(), resumed.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_probe_prints_both(capsys):
    m0.main(["--probe", "--probe-cores", f"gpt,{DUMMY}", "--params", "200000", "--vocab", "300", "--seq", "32", "--batch", "4", "--steps", "3"])
    out = json.loads(capsys.readouterr().out)
    assert out["gpt"]["tokens_per_s"] > 0 and out[DUMMY]["tokens_per_s"] > 0


def test_missing_data_exits_nonzero(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        m0.main(["--core", "gpt", "--data", str(tmp_path / "nope.txt")])
    assert e.value.code not in (0, None)
    assert "nope.txt" in str(e.value.code) + capsys.readouterr().err

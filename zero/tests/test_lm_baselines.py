"""Owned LM baselines for the QF-LM bake-off (PLAN.md item 5): GPT-style decoder and GRU LM."""
import inspect
import math
from pathlib import Path

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from arms.lm_baselines import GPTLM, GRULM
from beds.lang import TokenStream, train_bpe

FIX = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"
V, T = 300, 16
MAKERS = {
    "gpt": lambda **k: GPTLM(V, d_model=32, n_layer=2, n_head=4, seq_len=T, **k),
    "gru": lambda **k: GRULM(V, d_emb=32, hidden=64, n_layer=2, **k),
}
both = pytest.mark.parametrize("kind", MAKERS)


def toks(B=3, n=T, seed=0):
    return torch.randint(0, V, (B, n + 1), generator=torch.Generator().manual_seed(seed))


@both
def test_shapes_and_loss(kind):
    m, x = MAKERS[kind](), toks()
    out = m(x)
    assert out["logits"].shape == (3, T, V)
    assert out["loss"].dim() == 0 and torch.isfinite(out["loss"])
    nll = m.nll(x)
    assert nll.shape == (3, T)
    assert torch.allclose(out["loss"], nll.mean(), atol=1e-6)


@both
def test_causality(kind):
    m = MAKERS[kind]().eval()
    a = toks()
    b = a.clone()
    t = 7
    b[:, t + 1:-1] = (b[:, t + 1:-1] + 1) % V  # inputs are tokens[:, :-1]; change positions > t
    with torch.no_grad():
        la, lb = m(a)["logits"], m(b)["logits"]
    assert torch.equal(la[:, :t + 1], lb[:, :t + 1])
    assert not torch.equal(la[:, t + 1:], lb[:, t + 1:])  # control: the edit is visible downstream


def test_active_flops_hand_count_gpt():
    d, L = 32, 2
    lin = L * (3 * d * d + d * d + 2 * 4 * d * d) + d * V
    attn = L * 2 * d * (T + 1) // 2
    assert MAKERS["gpt"]().active_flops() == lin + attn
    assert MAKERS["gpt"]().active_flops(T=8) == lin + L * d * 9


def test_active_flops_hand_count_gru():
    e, h = 32, 64
    gru = 3 * (e * h + h * h) + 3 * (h * h + h * h)
    assert MAKERS["gru"]().active_flops() == gru + h * e + e * V


@both
def test_active_flops_vs_flop_counter(kind):
    m = MAKERS[kind]().eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(toks(1))
    got = fc.get_total_flops() / 2 / T
    # The counter does not see SDPA on CPU (measured: GPT total == Linear-only part), so this compares the
    # Linear-only part (attention is ~3% of active_flops at this size); the attention term itself is
    # asserted by formula in test_active_flops_hand_count_gpt.
    assert got == pytest.approx(m.active_flops(), rel=0.10)


def test_n_params_tying_counts_once():
    for make, d in ((MAKERS["gpt"], 32), (MAKERS["gru"], 32)):
        assert make(tie=False).n_params() - make(tie=True).n_params() == V * d


@both
def test_generate(kind):
    m = MAKERS[kind]().eval()
    p = torch.tensor([[5, 9, 11]])
    n = T + 5  # longer than seq_len: exercises the context crop
    a, b = m.generate(p, n, temperature=0), m.generate(p, n, temperature=0)
    assert torch.equal(a, b) and a.shape == (1, 3 + n)
    assert torch.equal(a[:, :3], p) and 0 <= int(a.min()) and int(a.max()) < V
    g = lambda: torch.Generator().manual_seed(1)
    s1 = m.generate(p, 8, top_k=5, generator=g())
    assert torch.equal(s1, m.generate(p, 8, top_k=5, generator=g())) and s1.shape == (1, 11)


@both
def test_forward_signature(kind):
    assert list(inspect.signature(MAKERS[kind]().forward).parameters) == ["tokens"]


@pytest.mark.parametrize("kind", ["gpt", "gru"])
def test_learns_on_fixture_corpus(kind, tmp_path):
    torch.manual_seed(0)
    tok = train_bpe(FIX, vocab_size=V, out=tmp_path / "tok.json")
    assert tok.get_vocab_size() == V
    m = (GPTLM(V, 64, 2, 4, 32) if kind == "gpt" else GRULM(V, 64, 128, 1))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    s = TokenStream(FIX, tok, split="train", seq_len=32, batch=8, seed=0)
    losses = []
    for _ in range(150):
        loss = m(next(s))["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10 - 0.5
    m.eval()
    v = TokenStream(FIX, tok, split="val", seq_len=32, batch=8, seed=0)
    with torch.no_grad():
        val = sum(m.nll(next(v)).mean().item() for _ in range(4)) / 4
    assert val < math.log(V) - 1.0

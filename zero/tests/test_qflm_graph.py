import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

torch.set_num_threads(2)

from qflm import _w
from qflm.model import QFLM
from qflm_graph import GraphedQFLM

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def mk(dev, B, N=16, V=12, T=1.0, null_logit=None, **kw):
    m = QFLM(N, V, seed=3).to(dev)
    if null_logit is not None:
        m.logits.data[:, V] = null_logit  # fixture: tiny p_null so every round w_s>=1 is below TAU
    return GraphedQFLM(m, B, T, **kw)


@pytest.mark.parametrize("B", [1, 4])
def test_q_rounds_match_model_emit_cpu(B):
    g = mk("cpu", B, external_noise=True)
    g.step_eager()
    g.model.reset(B)
    marg, rounds = g.model.dist()
    assert torch.allclose(g.q, marg, atol=1e-5) and torch.equal(g.st, rounds)


def test_tau_forced_emission_cpu():
    # tiny p_null: P(null^1) < TAU, so every row is forced to emit at round 1 (renormalised by 1-p_null)
    g = mk("cpu", 3, null_logit=-30.0)
    g.step_eager()
    assert (g.st == 1).all()
    assert torch.allclose(g.q.sum(1), torch.ones(3), atol=1e-5)
    g2 = mk("cpu", 3, null_logit=2.0)  # null-heavy model: some rows must run later rounds than the tiny-null model
    g2.step_eager()
    assert g2.st.max() > 1


def test_teacher_forced_matches_model_forward_cpu():
    B, n = 3, 8
    g = mk("cpu", B, external_noise=True)
    m = g.model
    toks = torch.randint(0, g.V, (B, n), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        nl = m.nll_per_token(toks)
        _, states = m(toks, return_states=True)
    for t in range(n):
        g.force_tok = toks[:, t]
        g.step_eager()
        assert torch.allclose(-g.q.gather(1, toks[:, t:t + 1])[:, 0].log(), nl[:, t], atol=1e-4)
        assert torch.allclose(g.psi_buf, states[:, t], atol=1e-5)


@cuda
@pytest.mark.parametrize("B", [1, 4])
def test_graph_equals_eager_same_noise(B):
    g = mk("cuda", B, external_noise=True)
    n = 64
    noise = torch.rand(n, B, g.V, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    outs = {}
    for mode in ("eager", "graph"):
        g.reset()
        toks = []
        for t in range(n):
            g.noise.copy_(noise[t])
            g.step_eager() if mode == "eager" else g.replay()
            toks.append(g.tok.clone())
        outs[mode] = torch.stack(toks)
    assert torch.equal(outs["eager"], outs["graph"])


@cuda
def test_ingraph_rng_advances():
    g = mk("cuda", 4)
    seqs = []
    for _ in range(2):
        g.reset()
        seqs.append(torch.stack([(g.replay(), g.tok.clone())[1] for _ in range(32)]))
    assert not torch.equal(seqs[0], seqs[1])


@pytest.mark.parametrize("dev", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_sum_q_and_norm_every_step(dev):
    g = mk(dev, 4)
    g.reset()
    if dev == "cuda":
        g.capture()
    run = g.replay if dev == "cuda" else g.step_eager
    for _ in range(64):
        run()
        assert (g.q.sum(1) - 1).abs().max() < 1e-5
        assert (_w(g.psi_buf).sum(1) - 1).abs().max() < 1e-5


def test_sample_is_gumbel_argmax_of_q_cpu():
    g = mk("cpu", 6, external_noise=True)
    g.noise.copy_(torch.rand(6, g.V, generator=torch.Generator().manual_seed(5)))
    g.step_eager()
    u = g.noise.clamp(1e-20, 1 - 1e-7)
    assert torch.equal(g.tok, (g.q.log() - torch.log(-torch.log(u))).argmax(1))

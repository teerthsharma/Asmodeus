import pytest
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from qflm.generate import NULL, QFLMStepper, generate
from qflm.model import QFLM

T = 6


def mk(arm="qflm"):
    return QFLM(N=16, V=8, s_max=4, arm=arm, seed=0)


@pytest.mark.parametrize("arm", ["qflm", "nonull", "nothink", "embed", "softmax"])
@torch.no_grad()
def test_nll_parity(arm):
    m = mk(arm)
    toks = torch.randint(0, 8, (1, T), generator=torch.Generator().manual_seed(1))
    st, nll = QFLMStepper(m), 0.0
    for t in range(T):
        st.reset(toks[0, :t].tolist())
        nll -= float(st.marginal()[toks[0, t]].log())
    assert nll == pytest.approx(float(m(toks)) * T, rel=1e-4)


@torch.no_grad()
def test_protocol():
    m = mk()
    st = QFLMStepper(m)
    st.reset([1, 2])
    p, pn = st.dist()
    assert float(p.sum()) + pn == pytest.approx(1, abs=1e-5)
    for _ in range(m.s_max):
        st.observe(NULL)
    p, pn = st.dist()
    assert pn == 0.0 and float(p.sum()) == pytest.approx(1, abs=1e-5)
    with pytest.raises(RuntimeError):
        st.observe(NULL)


@torch.no_grad()
def test_generate():
    m = mk()
    r = [generate(QFLMStepper(m), [1, 2], max_new=10, think_budget=m.s_max, seed=0) for _ in range(2)]
    assert r[0] == r[1] and len(r[0]["ids"]) == 10
    assert all(0 <= i < 8 for i in r[0]["ids"]) and max(r[0]["think_steps"]) <= m.s_max
    st = QFLMStepper(m)
    generate(st, [1, 2], max_new=10, seed=0)
    assert float((st.psi.abs() ** 2).sum()) == pytest.approx(1, abs=1e-5)


@torch.no_grad()
def test_marginal_pure():
    m = mk()
    st = QFLMStepper(m)
    st.reset([3])
    a = st.dist()[0].clone()
    assert float(st.marginal().sum()) == pytest.approx(1, abs=1e-5)
    assert torch.equal(a, st.dist()[0])


@torch.no_grad()
def test_nll_parity_high_null():
    m = mk()
    m.logits[:, 8] += 3.0
    toks = torch.randint(0, 8, (1, T), generator=torch.Generator().manual_seed(0))
    st, nll = QFLMStepper(m), 0.0
    for t in range(T):
        st.reset(toks[0, :t].tolist())
        nll -= float(st.marginal()[toks[0, t]].log())
    assert nll == pytest.approx(float(m(toks)) * T, rel=1e-4)


@torch.no_grad()
def test_marginal_matches_model_recurrence():
    m = mk()
    prompt = [1, 2, 5]
    st = QFLMStepper(m)
    st.reset(prompt)
    m.reset(1)
    for x in prompt:
        m.observe(torch.tensor([x]))
    assert torch.allclose(st.marginal(), m.dist()[0][0], atol=1e-6)


@torch.no_grad()
def test_tau_forcing():
    m = mk()
    m.logits[:, m.V] -= 8.0
    st = QFLMStepper(m)
    st.reset([1, 2])
    assert st.dist()[1] < 1e-3  # P(null^1) = p_null < TAU: forced at s=1
    st.observe(NULL)
    p, pn = st.dist()
    assert pn == 0.0 and float(p.sum()) == pytest.approx(1, abs=1e-5)
    with pytest.raises(RuntimeError):
        st.observe(NULL)


@torch.no_grad()
def test_chat_load_wraps_qflm(tmp_path, monkeypatch):
    import lm.cores
    from chat import load

    monkeypatch.setattr(lm.cores, "build", lambda *a, **k: mk())
    p = tmp_path / "ck.pt"
    torch.save({"core": "qflm", "vocab": 8, "d": 16, "state_dict": mk().state_dict()}, p)
    assert isinstance(load(str(p)), QFLMStepper)

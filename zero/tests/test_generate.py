import pytest
import torch

from qflm.generate import NULL, ContextStepper, generate, sample_next

P = torch.tensor([0.1, 0.6, 0.2, 0.1, 0.0])


def draws(n, seed=0, p=P, pn=0.0, **kw):
    g = torch.Generator().manual_seed(seed)
    return [sample_next(p, pn, generator=g, **kw) for _ in range(n)]


class FixedStepper:
    def __init__(self, p, p_null=0.0, decay=1.0):
        self.p, self.pn0, self.pn, self.decay = p, p_null, p_null, decay
        self.nulls, self.toks = 0, []

    def reset(self, prompt_ids):
        self.nulls, self.toks, self.pn = 0, [], self.pn0

    def dist(self):
        return self.p * (1 - self.pn) / self.p.sum(), self.pn

    def observe(self, o):
        if o == NULL:
            self.nulls += 1
            self.pn *= self.decay
        else:
            self.toks.append(o)
            self.pn = self.pn0


def test_temperature_zero_greedy():
    assert set(draws(20, temperature=0)) == {1}


def test_low_temperature_sharpens():
    assert draws(500, temperature=0.05).count(1) >= 495


def test_temperature_one_matches_p():
    d = draws(2000, seed=1, temperature=1.0)
    for v in range(5):
        assert abs(d.count(v) / 2000 - P[v].item()) < 0.05


def test_top_p_and_top_k():
    assert set(draws(300, top_p=0.7)) <= {1, 2}
    assert set(draws(50, top_k=1)) == {1}


def test_zero_prob_never_sampled_hot():
    assert 4 not in draws(2000, temperature=5.0)


def test_no_repeat_ngram_and_penalty():
    assert sample_next(torch.tensor([0.0, 0.1, 0.9]), 0.0, temperature=0, history=[1, 2, 1], no_repeat_ngram=2) != 2
    assert sample_next(torch.tensor([0.3, 0.7]), 0.0, temperature=0, history=[1], repetition_penalty=100.0) == 0


def test_all_banned_falls_back():
    assert sample_next(torch.tensor([0.0, 1.0]), 0.0, temperature=0, history=[1, 1], no_repeat_ngram=1) == 1


def test_null_budget():
    s = FixedStepper(P, p_null=0.9)
    r = generate(s, [0], max_new=20, think_budget=3, seed=0)
    assert len(r["think_steps"]) == len(r["ids"]) == 20
    assert max(r["think_steps"]) <= 3 and s.nulls == sum(r["think_steps"]) > 0
    s = FixedStepper(P, p_null=0.9)
    generate(s, [0], max_new=20, think_budget=0, seed=0)
    assert s.nulls == 0


def test_state_dependent_think_varies():
    s = FixedStepper(P, p_null=0.9, decay=0.3)
    r = generate(s, [0], max_new=40, think_budget=6, seed=3)
    assert len(set(r["think_steps"])) > 1


def test_stop_ids_and_determinism():
    a = generate(FixedStepper(P), [0], max_new=30, seed=7)
    b = generate(FixedStepper(P), [0], max_new=30, seed=7)
    assert a == b and all(0 <= i < 5 for i in a["ids"])
    r = generate(FixedStepper(P), [0], max_new=30, temperature=0, stop_ids=(1,))
    assert r["ids"] == [1]


def test_context_stepper_real_model():
    from lm.cores import build

    m = build("gpt", vocab=16, d=32, layers=1).eval()
    s = ContextStepper(m)
    s.reset([1, 2, 3])
    p, pn = s.dist()
    assert abs(p.sum().item() + pn - 1) < 1e-5 and p.shape == (16,)
    r = generate(s, [1, 2, 3], max_new=10, seed=0)
    assert len(r["ids"]) == 10 and all(0 <= i < 16 for i in r["ids"])
    with pytest.raises(ValueError):
        s.reset([])

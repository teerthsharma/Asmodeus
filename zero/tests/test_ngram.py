import random

import torch

import chat
from lm.ngram import NGramLM
from qflm.generate import generate


def test_dist_normalised_positive():
    m = NGramLM(8)
    m.fit([[1, 2, 3, 4, 5]] * 5)
    for prompt in ([], [1], [1, 2, 3], [7, 7, 7]):
        m.reset(prompt)
        p, pn = m.dist()
        assert pn == 0.0 and p.shape == (8,)
        assert abs(p.sum().item() - 1) < 1e-6 and (p > 0).all()


def test_greedy_continues_pattern():
    m = NGramLM(8)
    m.fit([[1, 2, 3, 4]] * 50)
    m.reset([1, 2, 3])
    p, _ = m.dist()
    assert p.argmax().item() == 4 and p[4] > 0.4


def test_fourgram_beats_unigram_heldout():
    rng = random.Random(0)

    def seq():
        s = [rng.randrange(6)]
        for _ in range(30):
            s.append((s[-1] + 1) % 6 if rng.random() < 0.9 else rng.randrange(6))
        return s

    tr, te = [seq() for _ in range(100)], [seq() for _ in range(20)]
    a, b = NGramLM(6), NGramLM(6, lambdas=(1, 0, 0, 0))
    a.fit(tr), b.fit(tr)
    na, nb = sum(a.nll(s) for s in te), sum(b.nll(s) for s in te)
    assert na < nb


def test_generate_in_range():
    m = NGramLM(8)
    m.fit([[1, 2, 3, 4]] * 10)
    out = generate(m, [1, 2], max_new=10, seed=0)
    assert len(out["ids"]) == 10 and all(0 <= i < 8 for i in out["ids"])


def test_chat_reply_str():
    m = NGramLM(256)
    m.fit([list(f"User: hi {i}\nAssistant: hello {i}\n".encode()) for i in range(5)])
    out = chat.reply(m, chat.ByteTok(), [], "hi", temperature=0, max_new=8)
    assert isinstance(out, str)

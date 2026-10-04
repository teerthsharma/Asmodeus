from arms.lm_baselines import GPTLM
from arms.match import match_lm

V, T = 4096, 256
ref = GPTLM(V, 256, 6, 4, T)
F, P = ref.active_flops(), ref.n_params()
within = lambda a, b: abs(a / b - 1) <= 0.05


def test_gpt_both_recovers_reference():
    r = match_lm("gpt", F, P, vocab=V, n_head=4, seq_len=T)
    assert r["both"] and within(r["both"]["flops"], F) and within(r["both"]["params"], P)
    assert within(r["flop_matched"]["flops"], F) and within(r["param_matched"]["params"], P)


def test_gru_each_axis_matched():
    r = match_lm("gru", F, P, vocab=V)
    assert within(r["flop_matched"]["flops"], F) and within(r["param_matched"]["params"], P)
    if r["both"]:
        assert within(r["both"]["flops"], F) and within(r["both"]["params"], P)
    else:
        assert "note" in r

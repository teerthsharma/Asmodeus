"""Era LM: streaming exactness, filter exactness, bank algebra, harness contract, owned n-gram parity."""
import numpy as np
import torch

from eralm import EraLMConfig, build, eval_loss, generate, train_step
from eralm import filter as F
from eralm.data import RandomWindows, Streams
from eralm.ngram import JMNGram
from eralm.ridge import RidgeBank, gauss_logml
from eralm.tt import TT, _g


def test_mingru_scan_equals_recurrence():
    torch.manual_seed(0)
    m = TT(32, d=16, layers=1).double()
    mix = m.blocks[0].mix
    u = torch.randn(2, 9, 16, dtype=torch.float64)
    y, _ = mix(u)
    xc = torch.cat([u.new_zeros(2, 16, 3), u.transpose(1, 2)], 2)
    uu = torch.nn.functional.silu(mix.conv(xc)).transpose(1, 2)
    k, c = mix.zc(uu).chunk(2, -1)
    z, h, hs = torch.sigmoid(k), torch.zeros(2, mix.h, dtype=torch.float64), []
    for t in range(9):
        h = (1 - z[:, t]) * h + z[:, t] * _g(c[:, t])
        hs.append(h)
    assert (y - mix.out(torch.stack(hs, 1))).abs().max() < 1e-5


def test_window_filter_equals_token_loop():
    torch.manual_seed(1)
    K, nt = 5, 2
    alive = torch.tensor([True, False, True, True, False])
    S = torch.rand(K, K, dtype=torch.float64)
    for d in (F.Decider("MOM", 0.003), F.Decider("SOFT", 0.01)):
        M = F._share(alive, S, d, nt)
        assert torch.allclose(M.sum(1)[torch.cat([alive, torch.ones(nt, dtype=torch.bool)])], torch.ones(5, dtype=M.dtype))
        l = torch.rand(3, 50, K + nt, dtype=torch.float64)
        w0 = torch.rand(3, K + nt, dtype=torch.float64) * torch.cat([alive, torch.ones(nt, dtype=torch.bool)])
        w0 = w0 / w0.sum(1, keepdim=True)
        logv, lwe = F.window_filter(l.log(), F.log_share(alive, S, d, nt), w0.clamp_min(1e-300).log())
        w = w0
        for t in range(50):
            v = w @ M
            assert (logv[:, t].exp() - v).abs().max() < 1e-12
            w = v * l[:, t] / (v * l[:, t]).sum(1, keepdim=True)
        assert (lwe.exp() - w).abs().max() < 1e-12


def _cfg(**kw):
    return EraLMConfig(**{**dict(vocab=64, d=16, layers=1, n_max=4, cache=8, amp=False), **kw})


def test_streaming_equals_one_pass():
    torch.manual_seed(2)
    m = build(_cfg()).double().eval()
    m.online = False
    x = torch.randint(0, 64, (2, 41))
    m.reset_streams(2)
    a, b = m.step(x[:, :20], x[:, 1:21]), m.step(x[:, 20:40], x[:, 21:41])
    m.reset_streams(2)
    c = m.step(x[:, :40], x[:, 1:41])
    assert (torch.cat([a["nll"], b["nll"]], 1) - c["nll"]).abs().max() < 1e-5


def test_harness_contract_and_training():
    torch.manual_seed(3)
    m = build(_cfg())
    x = torch.randint(0, 64, (4, 17))
    out = m(x)
    assert out["nll"].shape == (4, 17) and float(out["nll"][:, 0].abs().max()) == 0.0
    opt = m.configure_optimizer(lr=1e-2)
    l0 = train_step(m, opt, x)["loss"]
    for _ in range(30):
        s = train_step(m, opt, x)
    assert s["loss"] < l0
    assert m.n_params() > 0 and m.active_flops_per_token(16) > 0
    assert generate(m, x[:, :3], n_new=4).shape == (4, 7)


def test_mixture_never_far_worse_than_tt():
    """L38 with the TT as comparator: mixture NLL <= TT NLL + log(n/(1-alpha+rho)) / T + log(1/(1-alpha))."""
    torch.manual_seed(4)
    m = build(_cfg(alpha=0.003)).eval()
    x = torch.randint(0, 64, (3, 200))
    m.reset_streams(3)
    for i in range(3):
        o = m.step(x[:, :-1], x[:, 1:])
    m.online = False
    m.reset_streams(3)
    o = m.step(x[:, :-1], x[:, 1:])
    n = m.cfg.n_max + 2
    T = x.shape[1] - 1
    gap = (o["nll"].sum(1) - o["tt_nll"].sum(1)) / T
    assert float(gap.max()) <= np.log(n / (1 - 0.003 + 0.0015)) / T - np.log(1 - 0.003) + 1e-4


def test_bank_statistics_are_additive_and_merge_same_law():
    torch.manual_seed(5)
    g = torch.Generator().manual_seed(0)
    W = torch.randn(4, 3, generator=g, dtype=torch.float64)

    def seg(n):
        phi = torch.randn(n, 4, generator=g, dtype=torch.float64)
        rho = phi @ W + 0.1 * torch.randn(n, 3, generator=g, dtype=torch.float64)
        return phi.T @ phi, phi.T @ rho, (rho * rho).sum(), torch.tensor(float(n), dtype=torch.float64)

    s1, s2 = seg(200), seg(200)
    whole = [a + b for a, b in zip(s1, s2)]
    lm = lambda s: float(gauss_logml(s[0], s[1], s[2], s[3], 1.0, 0.01))
    # chain rule: log ML(1+2) = log ML(1) + [log ML(1+2) - log ML(1)] (address score is the prequential term)
    assert np.isfinite(lm(whole)) and lm(whole) > lm(s1) + lm(s2)        # same law: Bayes factor > 1
    bank = RidgeBank(4, 3, n_max=4, sigma2=0.01, merge_rule="bf1")
    st = [torch.stack(v) for v in zip(s1, s2)]
    bank.credit(st[0], st[1], st[2], st[3], torch.tensor([-1, -1]))
    assert bank.n_alive == 1                                            # same law: one era
    assert torch.allclose(bank.A[bank.order()[0]], whole[0])


def test_ngram_matches_lm_ngram():
    from lm.ngram import NGramLM
    rng = np.random.default_rng(0)
    x = rng.integers(0, 20, 2000)
    x[::7] = 3
    ref = NGramLM(20, order=4)
    ref.fit([x.tolist()])
    m = JMNGram(20, 4).fit(x)
    test = x[100:150]
    lp = m.logprob(test)
    refl = [torch.log(ref._p(test[:t].tolist())[test[t]]).item() for t in range(1, len(test))]
    assert np.abs(lp - np.array(refl)).max() < 1e-5


def test_streams_cover_each_token_once():
    arr = np.arange(1000, dtype=np.uint16)
    s = Streams(arr, 4, 10)
    rows = [b for b in s]
    seen = torch.cat([r[:, 1:] for r in rows], 1)
    assert seen.shape == (4, 10 * len(s))
    assert all(int(rows[i + 1][0, 0]) == int(rows[i][0, -1]) for i in range(len(rows) - 1))
    w = RandomWindows(arr, 3, 5, 0)
    st = w.state()
    a = next(w)
    w.load(st)
    assert torch.equal(a, next(w))

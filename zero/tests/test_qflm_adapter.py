"""Foreman, PLAN item 5: qflm/adapter.py wraps Kutner's QFLM (imported, never edited) in the lm/cores contract
used by run_lm.py: forward -> {"nll" [B,T], "steps" [B,T], "flops" [B,T]}, step(state, token), n_params,
active_flops_per_token. Plus the two deciding controls in qflm/controls.py (dephased HMM, real psi)."""
import json
import math
from pathlib import Path

import pytest
import torch

torch.set_num_threads(2)

from qflm import butterfly_matrix, _w  # noqa: E402
from qflm.model import QFLM  # noqa: E402
from qflm.adapter import QF  # noqa: E402
from qflm.controls import classical_butterfly, DephasedQFLM, RealQFLM  # noqa: E402

N, V = 8, 11
torch.manual_seed(0)


def toks(B=3, T=7, seed=1):
    return torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(seed))


def qf(**kw):
    a = QF(vocab=V, d=N, seed=0, **kw)
    if kw.get("arm") == "embed":
        torch.nn.init.normal_(a.m.write_phase, std=0.3, generator=torch.Generator().manual_seed(2))
    return a


# ---------- adapter ----------

@pytest.mark.parametrize("arm", ["qflm", "nothink", "nonull", "embed", "softmax"])
def test_mirrors_kutner_forward(arm):
    a = qf(arm=arm)
    x = toks()
    out = a(x)
    assert out["nll"].shape == out["steps"].shape == out["flops"].shape == x.shape
    assert torch.allclose(out["nll"].mean(), a.m(x), atol=1e-5)              # same semantics as QFLM.forward
    assert isinstance(a.m, QFLM)


def test_causal_per_token():
    a, x = qf(), toks(T=9)
    y = x.clone()
    y[:, 5:] = (y[:, 5:] + 1) % V
    with torch.no_grad():
        p, q = a(x)["nll"], a(y)["nll"]
    assert torch.allclose(p[:, :5], q[:, :5], atol=1e-6) and not torch.allclose(p[:, 5:], q[:, 5:], atol=1e-4)


def test_steps_are_pre_observation_expected_think_rounds():
    a = qf()
    s = a(toks())["steps"]
    assert (s >= 0).all() and (s <= a.m.s_max).all()
    assert (qf(arm="nonull")(toks())["steps"] == 0).all()
    with torch.no_grad():
        a.m.logits[:, V] = 30.0                                              # null almost certain: every round runs
    assert torch.allclose(a(toks())["steps"], torch.full((3, 7), float(a.m.s_max)), atol=1e-3)
    y = toks(seed=5)                                                         # pre-observation: steps[:, t] blind to token t
    x = y.clone(); x[:, 4] = (x[:, 4] + 1) % V
    b = qf()
    with torch.no_grad():
        assert torch.allclose(b(x)["steps"][:, :5], b(y)["steps"][:, :5])


def test_fixed_k():
    x = toks()
    f0 = qf(arm="qflm", fixed_k=0)(x)
    assert torch.allclose(f0["nll"], qf(arm="nonull")(x)["nll"], atol=1e-6)   # k=0 == Kutner's nonull arm
    f2 = qf(arm="qflm", fixed_k=2)(x)
    assert (f2["steps"] == 2).all() and (f2["flops"] > f0["flops"]).all()
    assert not torch.allclose(f2["nll"], f0["nll"])


def test_flops_and_params():
    a = qf()
    out = a(toks())
    assert (out["flops"] > 0).all()
    assert out["flops"].max() <= a.active_flops_per_token(7) * 1.0001      # nominal = all rounds (training cost)
    assert a.n_params() == a.m.n_params()                                    # Kutner's count (complex = 2 reals)


def test_step_matches_forward_and_state_is_flat():
    a, x = qf(), toks(B=1, T=6)
    with torch.no_grad():
        want = a(x)["nll"][0]
        st, got = None, []
        for t in range(6):
            st, lp = a.step(st, x[:, t])
            got.append(-lp.reshape(()))
            assert st.shape == (1, N)
    assert torch.allclose(torch.stack(got), want, atol=1e-5)
    import run_lm
    assert run_lm.state_flat(a, V, lengths=(8, 32)) is True


# ---------- controls ----------

@pytest.mark.parametrize("batched", [False, True])
def test_dephased_butterfly_is_unistochastic_layerwise(batched):
    """Butterfly = unique path between any input/output pair, so |Q|^2 (elementwise) = product of |layer|^2."""
    g = torch.Generator().manual_seed(3)
    L = 3
    th, ph = torch.rand(L, N // 2, generator=g) * 6, torch.rand(L, N // 2, generator=g) * 6
    b = torch.softmax(torch.randn(2, N, generator=g), 1)
    T = _w(butterfly_matrix(th, ph, N))
    assert torch.allclose(T.sum(0), torch.ones(N), atol=1e-5) and torch.allclose(T.sum(1), torch.ones(N), atol=1e-5)
    thb = th.expand(2, L, N // 2) if batched else th
    assert torch.allclose(classical_butterfly(b, thb), b @ T.T, atol=1e-5)


def _dense_dephased_reference(m, x):
    """Density-matrix filter with full dephasing after every update; exact classical mixture over null records;
    Kutner's halting (Amendment 7): a row emits (forced, renormalised over tokens) once P(null^s) < TAU or s = s_max."""
    from qflm.model import TAU
    M, Vn = m.M(), m.V
    Q = butterfly_matrix(m.theta, m.phi, m.N)
    b0 = m.psi0()
    rho = torch.diag(_w(b0) if b0.is_complex() else b0).to(Q.dtype)
    nll = []
    for v in x.tolist():
        logw, terms, posts = 0.0, [], []
        for s in range(m.s_max + 1):
            if s:
                th0, ph0, At, Ap = m._tp()
                w = torch.diagonal(rho).real
                Qt = butterfly_matrix(th0 + (w @ At.T).reshape(th0.shape), ph0 + (w @ Ap.T).reshape(ph0.shape), m.N)
                rho = torch.diag(torch.diagonal(Qt @ rho @ Qt.conj().T))
            r = torch.diag(torch.diagonal(Q @ rho @ Q.conj().T)).real
            p = torch.diagonal(r) @ M
            forced = s == m.s_max or math.exp(float(logw)) < TAU
            px = p[v] / (1 - p[Vn]) if forced else p[v]
            terms.append(logw + torch.log(px)); posts.append(torch.diag(M[:, v] * torch.diagonal(r)) / (torch.diagonal(r) @ M[:, v]))
            if forced:
                break
            if s < m.s_max:
                logw = logw + torch.log(p[Vn])
                rho = (torch.diag(M[:, Vn] * torch.diagonal(r)) / p[Vn]).to(Q.dtype)
        lt = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in terms])
        nll.append(-torch.logsumexp(lt, 0))
        wts = torch.softmax(lt, 0)
        rho = sum(wi * q for wi, q in zip(wts, posts)).to(Q.dtype)
    return torch.stack(nll)


def test_dephased_matches_density_matrix_reference():
    a = qf(kind="dephased")
    assert isinstance(a.m, DephasedQFLM)
    x = toks(B=1, T=6)
    with torch.no_grad():
        out = a(x)["nll"][0]
        ref = _dense_dephased_reference(a.m, x[0])
    assert torch.allclose(out, ref, atol=1e-4), (out, ref)


def test_dephased_state_is_distribution_and_blind_to_phases():
    a, x = qf(kind="dephased"), toks()
    with torch.no_grad():
        st, _ = a.step(None, x[:1, 0])
        assert not st.is_complex() and (st >= 0).all() and torch.allclose(st.sum(), torch.tensor(1.0), atol=1e-5)
        before = a(x)["nll"]
        a.m.phi.add_(1.0)
        assert torch.allclose(a(x)["nll"], before, atol=1e-6)               # phases drop out of the HMM
        q = qf()
        before_q = q(x)["nll"]
        q.m.phi.add_(1.0)
        assert not torch.allclose(q(x)["nll"], before_q, atol=1e-4)        # control: they matter in QF-LM


def test_real_psi_control():
    a, x = qf(kind="real"), toks()
    assert isinstance(a.m, RealQFLM)
    names = {n for n, _ in a.m.named_parameters()}
    assert not names & {"phi", "phi0", "A_phi", "psi0_im"}
    with torch.no_grad():
        st = None
        for t in range(x.shape[1]):
            st, _ = a.step(st, x[:1, t])
            assert (st.imag.abs() < 1e-7).all() if st.is_complex() else True
        ref = QFLM(N, V, seed=0)                                            # same seed, phases zeroed by hand
        for n in ("phi", "phi0", "A_phi", "psi0_im"):
            getattr(ref, n).zero_()
        assert torch.allclose(a(x)["nll"].mean(), ref(x), atol=1e-5)


# ---------- harness end to end ----------

def test_run_lm_end_to_end_with_qf_arms(tmp_path):
    import run_lm
    from beds.lang import train_bpe, corpus_hash
    FIX = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"
    tok = tmp_path / "tok.json"
    train_bpe(FIX, 300, tok)
    core = "qflm.adapter:QF"
    arms = {"qf": {"N": 8}, "qf_embed": {"N": 8, "arm": "embed"}, "qf_fixed": {"N": 8, "fixed_k": 1},
            "qf_dephased": {"N": 8, "kind": "dephased"}, "qf_real": {"N": 8, "kind": "real"}}
    argv = ["--data", str(FIX), "--sha256", corpus_hash(FIX), "--tok", str(tok),
            "--arms", ",".join(f"{a}={core}" for a in arms) + ",gpt=gpt",
            "--arm-kw", "gpt={\"layers\":1,\"max_seq\":17}", *sum((["--arm-kw", f"{a}={json.dumps(k)}"] for a, k in arms.items()), []),
            "--seeds", "0", "--params", "22900", "--macs-budget", "5e7", "--seq", "16", "--batch", "2",
            "--val-batches", "1", "--out", str(tmp_path / "o")]
    run_lm.main(argv)
    rows = [json.loads(l) for l in (tmp_path / "o" / "results.jsonl").read_text().splitlines()]
    assert {r["arm"] for r in rows} == set(arms) | {"gpt"}
    assert all(r["d"] == 8 for r in rows if r["arm"] != "gpt")
    assert all(r["macs_per_token"] <= r["macs_per_token_worst"] for r in rows)   # expected <= all-rounds worst case              # N given: no width matching
    crit = json.loads((tmp_path / "o" / "criteria.json").read_text())
    for k in ("a", "b", "c", "quantum", "talks_ppl"):
        assert "missing" not in crit[k]["why"], (k, crit[k])
    assert crit["c"]["pass"] is True                                         # O(N) state


def test_dephased_model_never_falls_through_to_quantum_forward():
    """Thirteen: DephasedQFLM inherits QFLM.forward, which runs the quantum nll primitive on psi.
    Called directly, the dephased arm must give the dense dephased reference, not QF-LM's number."""
    x = toks(B=1, T=6)
    d, q = DephasedQFLM(N, V, seed=0), QFLM(N, V, seed=0)                     # identical params, random phases
    with torch.no_grad():
        got, quantum = d(x), q(x)
        ref = _dense_dephased_reference(d, x[0]).mean()
    assert got.ndim == 0 and torch.allclose(got, ref, atol=1e-4), (got, ref)
    assert (got - quantum).abs() > 1e-3, (got, quantum)                      # phase-sensitive input: they differ
    with torch.no_grad():                                                     # stateful paths stay classical too
        d.reset(1); d.observe(x[:, 0]); d.think()
        assert not d.psi.is_complex() and (d.psi >= 0).all() and torch.allclose(d.psi.sum(), torch.tensor(1.0), atol=1e-5)
        with pytest.raises(NotImplementedError):                              # a control, not a talker
            d.generate(3)


def test_think_rank_passthrough_r16_arm():
    """Amendment 7 r16 reported arm: think_rank reaches QFLM; params and MACs follow Kutner's low-rank count."""
    NN = 32
    full, r = QF(vocab=V, d=NN, seed=0), QF(vocab=V, d=NN, seed=0, think_rank=16)
    assert r.m.think_rank == 16 and full.m.think_rank is None
    assert r.n_params() < full.n_params()
    assert r.active_flops_per_token() == 2 * r.m.active_macs(worst_case=True) < full.active_flops_per_token()
    x = toks()
    out = r(x)
    assert torch.allclose(out["nll"].mean(), r.m(x), atol=1e-5)                 # mirror holds for r16
    assert torch.allclose(out["flops"], 2 * r.m.active_macs(rounds=out["steps"] + 1))

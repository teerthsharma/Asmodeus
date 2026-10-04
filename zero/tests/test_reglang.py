import hashlib
import itertools
import json
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

torch.set_num_threads(2)

import run_reglang as rr
from beds import reglang as rl

A, B, EOS, BOS = 0, 1, 2, 3
LN2 = math.log(2)


def member(lang, s, capped):
    body = s[:-1]
    assert s[-1] == EOS and EOS not in body
    if lang == "kgap3":
        return True
    if capped:
        return True
    if lang == "parity":
        return body.count(A) % 2 == 0
    if lang == "last_a":
        return len(body) > 0 and body[-1] == A
    if lang == "astar_bstar":
        return all(not (body[i] == A and B in body[:i]) for i in range(len(body)))
    d = 0
    for x in body:
        d += 1 if x == A else -1
        assert 0 <= d <= 4
    return d == 0


@pytest.mark.parametrize("lang", rl.LANGS)
def test_samples_in_language(lang):
    seqs = rl.sample(lang, 300, 0, max_len=64)
    for s in seqs:
        assert member(lang, s, capped=len(s) == 65)
        assert len(s) <= 65


def test_dyck_depth_always_valid_even_capped():
    for s in rl.sample("dyck4", 300, 0, max_len=64):
        d = 0
        for x in s[:-1]:
            d += 1 if x == A else -1
            assert 0 <= d <= 4


def test_kgap_copy_rate():
    seqs = rl.sample("kgap3", 4000, 0)
    c = n = 0
    for s in seqs:
        for t in range(3, len(s) - 1):
            c += s[t] == s[t - 3]
            n += 1
    assert abs(c / n - 0.9) < 0.02


@pytest.mark.parametrize("lang", rl.LANGS)
def test_true_distribution_normalised(lang):
    L, tot = 6, 0.0
    for n in range(L + 1):
        for body in itertools.product((A, B), repeat=n):
            s = list(body) + [EOS]
            tot += math.exp(rl.true_logprob(lang, s, max_len=L).sum())
    assert abs(tot - 1) < 1e-9


@pytest.mark.parametrize("lang", rl.LANGS)
def test_deterministic_and_entropy(lang):
    a, b = rl.sample(lang, 50, 5), rl.sample(lang, 50, 5)
    assert a == b and a != rl.sample(lang, 50, 6)
    lp = rl.true_logprob(lang, a[0])
    assert lp.shape == (len(a[0]),) and np.all(lp <= 0)
    h = rl.entropy_rate_estimate(lang, a)
    assert h == pytest.approx(-sum(rl.true_logprob(lang, s).sum() for s in a) / sum(map(len, a)))
    assert 0 < h < math.log(3)


def test_splits_and_hash():
    assert rl.data_hash([[0, 1, 2]]) == "c0be322c1ad6af50f418b96232d98fe25a36d5d0a557291833f8248f2084b8ef"
    assert rl.data_hash(rl.sample("parity", 50, 0)) == "2a226684bff4d8de7c9b0ee99ce1b53f0fa39a47b7a1521a5042576bbe9dacdc"
    assert rl.data_hash([[0, 1, 2]]) != rl.data_hash([[0, 1, 2], [2]])
    for lang in ("kgap3", "last_a"):
        tr, va = (rl.sample(lang, n, sd) for n, sd in ((20000, 0), (2000, 1)))
        te = rl.sample(lang, 2000, 2)
        trs = {tuple(s) for s in tr}
        ov = np.mean([tuple(s) in trs for s in va])
        print(lang, "val/train overlap", ov)
        assert ov < 0.5 and rl.data_hash(tr) != rl.data_hash(va) != rl.data_hash(te)


def test_batch_padding_and_mask():
    seqs = [[0, 2], [1, 0, 1, 2]]
    x, m = rl.batch(seqs, bos=True)
    assert x.tolist() == [[3, 0, 2, 2, 2], [3, 1, 0, 1, 2]]
    assert m.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    x, m = rl.batch(seqs, bos=False)
    assert x.tolist() == [[0, 2, 2, 2], [1, 0, 1, 2]] and m.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]


# ---------------- harness ----------------
class Oracle:  # BOS-prefixed padded tokens -> true per-position NLL (pad positions get a junk value)
    def __init__(self, lang): self.lang = lang

    def nll(self, tok):
        out = torch.full((tok.shape[0], tok.shape[1] - 1), 7.0)
        for i, row in enumerate(tok.tolist()):
            s = row[1:]
            s = s[:s.index(EOS) + 1]
            out[i, :len(s)] = torch.tensor(-rl.true_logprob(self.lang, s), dtype=torch.float32)
        return out


class Uniform:
    def nll(self, tok): return torch.full((tok.shape[0], tok.shape[1] - 1), math.log(3))


def test_oracle_excess_zero_and_masking():
    seqs = rl.sample("last_a", 200, 0)
    assert abs(rr.excess(Oracle("last_a"), "lm", "last_a", seqs)) < 1e-5  # junk on padding must be masked


def test_uniform_excess():
    seqs = rl.sample("last_a", 200, 0)
    exp = (len(seqs) and (sum(len(s) for s in seqs) * math.log(3)
                          + sum(rl.true_logprob("last_a", s).sum() for s in seqs)) / sum(map(len, seqs)) / LN2)
    assert exp > 0
    assert rr.excess(Uniform(), "lm", "last_a", seqs) == pytest.approx(exp, abs=1e-5)


def test_qflm_scoring_matches_forward_and_grouping():
    m = rr.build("qflm", 8, 0)
    eq = rl.sample("parity", 400, 0)
    L3 = [s for s in eq if len(s) == 5][:6]
    assert len(L3) == 6
    x = torch.tensor(L3)
    with torch.no_grad():
        want = m(x).item() * x.numel()
    assert rr.qflm_nll_sum(m, L3) == pytest.approx(want, abs=1e-4)
    mixed = eq[:12]
    with torch.no_grad():
        want = sum(m(torch.tensor([s])).item() * len(s) for s in mixed)
    assert rr.qflm_nll_sum(m, mixed) == pytest.approx(want, rel=1e-5)
    assert rr.excess(m, "qflm", "parity", mixed) > 0


def test_refuses_without_amendment5(tmp_path, monkeypatch):
    p = tmp_path / "PREREG.md"
    p.write_text("# PREREG\n## Amendment 3 (pre-result)\n")
    monkeypatch.setattr(rr, "PREREG", p)
    with pytest.raises(SystemExit):
        rr.main(["--procs", "1", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        rr.main(["--tune", "--out", str(tmp_path / "o")])
    rr.main(["--dry-run", "--out", str(tmp_path / "o")])  # dry-run works regardless
    p.write_text("# PREREG\n## Amendment 5 (pre-result)\n")
    rr.require_amendment()


def test_dry_run_grid(capsys, tmp_path):
    rr.main(["--dry-run", "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert len([l for l in out.splitlines() if l.startswith(("todo ", "done "))]) == 195
    assert len([l for l in out.splitlines() if l.startswith(("tune-todo ", "tune-done "))]) == 35
    assert "reuses qflm" in out and "nothink" in out
    assert len(rr.plan(rr.LANGS, rr.ARMS, (0, 1, 2), tmp_path)) == 195
    assert not [a for a in rr.ARMS if a[0] == "embed" and a[1] != 16]
    with pytest.raises(SystemExit):
        rr.main(["--dry-run", "--procs", "4"])


def test_eval_seed_3_raises(tmp_path):
    with pytest.raises(ValueError):
        rr.plan(rr.LANGS, rr.ARMS, (0, 3), tmp_path)
    with pytest.raises(ValueError):
        rr.main(["--dry-run", "--seeds", "3"])


def _fake_tune(monkeypatch, tmp_path, val_by_lr):
    monkeypatch.setattr(rr, "TUNE_DIR", tmp_path / "tune")
    seen = []

    def split(lang, which):
        seen.append(which)
        if which == "test":
            raise AssertionError("tune touched the test split")
        return [[0, 2]]

    monkeypatch.setattr(rr, "load_split", split)
    monkeypatch.setattr(rr, "clean_sha", lambda cwd=None, ignore=(): "abc123")
    monkeypatch.setattr(rr, "build", lambda arm, N, seed: ("m", seed))
    monkeypatch.setattr(rr, "kind_of", lambda m: "lm")
    cur = {}
    monkeypatch.setattr(rr, "train", lambda m, lang, tr, steps, lr, batch, seed: (cur.update(lr=lr, seed=seed), m)[1])
    monkeypatch.setattr(rr, "excess", lambda m, kind, lang, seqs, max_len=64: val_by_lr[cur["lr"]])
    return seen, cur


def test_tune_picks_min_val_excess(monkeypatch, tmp_path):
    seen, cur = _fake_tune(monkeypatch, tmp_path, {1e-3: 0.5, 3e-3: 0.1, 1e-2: 0.3})
    assert rr.tune("gru", 0, "parity", 8) == 3e-3
    assert cur["seed"] == rr.TUNE_SEED == 3
    assert "test" not in seen
    assert rr.tuned_lr("gru", 0, "parity") == 3e-3
    assert (tmp_path / "tune" / "gru_N0_parity.json").exists()


def test_tune_tie_prefers_smaller_lr(monkeypatch, tmp_path):
    _fake_tune(monkeypatch, tmp_path, {1e-3: 0.2, 3e-3: 0.1, 1e-2: 0.1})
    assert rr.tune("gru", 0, "parity", 8) == 3e-3


def test_run_job_requires_tune_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(rr, "TUNE_DIR", tmp_path / "tune")
    job = dict(lang="parity", arm="gru", N=0, seed=0, key="k", path=tmp_path / "k.json")
    with pytest.raises(Exception):
        rr.run_job(job, "sha", steps=1)


@pytest.mark.parametrize("arm,N", [("qflm", 8), ("gru", 0), ("gpt", 0)])
def test_smoke_build_train(arm, N):
    m = rr.build(arm, N, 0)
    seqs = rl.sample("dyck4", 64, 0, max_len=12)
    kind = rr.kind_of(m)
    rr.train(m, "dyck4", seqs, steps=3, lr=3e-3, batch=8, seed=0)
    e = rr.excess(m, kind, "dyck4", seqs[:8], max_len=12)
    assert math.isfinite(e)


def test_dephased_builds_and_is_not_quantum_path():
    from qflm.controls import DephasedQFLM
    from qflm.model import QFLM
    d = rr.build("dephased", 16, 0)
    assert isinstance(d, DephasedQFLM)
    q = QFLM(16, 3, seed=0)
    q.load_state_dict(d.state_dict(), strict=False)
    pool = rl.sample("dyck4", 400, 0, max_len=12)
    L = max({len(s) for s in pool}, key=lambda n: sum(len(s) == n for s in pool))
    seqs = [s for s in pool if len(s) == L][:4]
    assert len(seqs) == 4
    x = torch.tensor(seqs)
    with torch.no_grad():
        assert abs(d(x).item() - q(x).item()) > 1e-6


def test_tune_stamps_sha_and_code_hash(monkeypatch, tmp_path):
    _fake_tune(monkeypatch, tmp_path, {1e-3: 0.5, 3e-3: 0.1, 1e-2: 0.3})
    rr.tune("gru", 0, "parity", 8)
    rec = json.loads((tmp_path / "tune" / "gru_N0_parity.json").read_text())
    assert rec["sha"] == "abc123"
    h = hashlib.sha256()
    for f in ("run_reglang.py", "qflm/model.py", "qflm/controls.py", "beds/reglang.py"):
        h.update((rr.ZERO / f).read_bytes())
    assert rec["code_hash"] == h.hexdigest()


def test_tune_refuses_dirty_tree(monkeypatch, tmp_path):
    _fake_tune(monkeypatch, tmp_path, {1e-3: 0.5, 3e-3: 0.1, 1e-2: 0.3})

    def dirty(cwd=None, ignore=()):
        raise SystemExit("refusing: dirty tree")

    monkeypatch.setattr(rr, "clean_sha", dirty)
    with pytest.raises(SystemExit):
        rr.tune("gru", 0, "parity", 8)
    assert not (tmp_path / "tune").exists()


def test_run_job_rejects_tune_sha_mismatch(monkeypatch, tmp_path):
    _fake_tune(monkeypatch, tmp_path, {1e-3: 0.5, 3e-3: 0.1, 1e-2: 0.3})
    rr.tune("gru", 0, "parity", 8)
    job = dict(lang="parity", arm="gru", N=0, seed=0, key="k", path=tmp_path / "k.json")
    with pytest.raises(Exception, match="sha"):
        rr.run_job(job, "other_sha", steps=1)


def test_tune_uses_tune_steps_not_eval_steps(monkeypatch, tmp_path):
    _fake_tune(monkeypatch, tmp_path, {1e-3: 0.5, 3e-3: 0.1, 1e-2: 0.3})
    got = []
    monkeypatch.setattr(rr, "train", lambda m, lang, tr, steps, lr, batch, seed: (got.append(steps), m)[1])
    monkeypatch.setattr(rr, "excess", lambda m, kind, lang, seqs, max_len=64: 0.1)
    rr.tune("gru", 0, "parity", 8)
    assert rr.TUNE_STEPS == 1000 and got == [1000] * 3
    assert json.loads((tmp_path / "tune" / "gru_N0_parity.json").read_text())["steps"] == 1000


def test_nothink_nonull_reuse_qflm_lr(monkeypatch, tmp_path):
    _fake_tune(monkeypatch, tmp_path, {1e-3: 0.5, 3e-3: 0.1, 1e-2: 0.3})
    with pytest.raises(FileNotFoundError):
        rr.tuned_lr("nothink", 8, "parity")
    rr.tune("qflm", 8, "parity", 8)
    for a in ("nothink", "nonull"):
        assert rr.tuned_lr(a, 8, "parity") == rr.tuned_lr("qflm", 8, "parity") == 3e-3
    assert not [j for j in rr.tune_plan(rr.LANGS, rr.ARMS) if j["arm"] in ("nothink", "nonull")]


def test_amendment9_gates_qf_arms(monkeypatch, tmp_path):
    p = tmp_path / "PREREG.md"
    p.write_text("# PREREG\n## Amendment 5 (pre-result)\n")
    monkeypatch.setattr(rr, "PREREG", p)
    ran = []
    monkeypatch.setattr(rr, "_tune_worker", lambda a: (ran.append(a[0]["arm"]), (a[0]["key"], 0.0))[1])
    base = ["--tune", "--langs", "parity", "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit):
        rr.main(base + ["--arms", "qflm"])
    rr.main(base + ["--arms", "gru"])
    assert ran == ["gru"]
    p.write_text("# PREREG\n## Amendment 5 (pre-result)\n## Amendment 9 (pre-result)\n")
    rr.main(base + ["--arms", "qflm"])
    assert "qflm" in ran

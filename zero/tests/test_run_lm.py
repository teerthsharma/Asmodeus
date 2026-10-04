"""PLAN.md item 5 harness (Foreman): run_lm.py bake-off. CPU, fixture corpus, tiny configs, no downloads.
Arm roles are fixed names: qf (QF-LM), qf_embed (+embed ablation), qf_fixed (fixed think steps), gpt."""
import json
from pathlib import Path

import pytest
import torch

torch.set_num_threads(2)

from beds.lang import TokenStream, train_bpe, corpus_hash  # noqa: E402
import run_lm  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"
V = 300


@pytest.fixture(scope="module")
def tok_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("tok") / "tok.json"
    train_bpe(FIX, V, p)
    return p


class ToyQF(torch.nn.Module):
    """Stand-in for a QF-LM arm: GRU state, emits per-token 'steps' and 'flops', exposes step() with an O(1) state."""

    def __init__(self, vocab, d, **kw):
        super().__init__()
        self.emb, self.rnn, self.head = torch.nn.Embedding(vocab, d), torch.nn.GRU(d, d, batch_first=True), torch.nn.Linear(d, vocab)
        self.d, self.vocab = d, vocab

    def forward(self, tokens):
        h, _ = self.rnn(self.emb(tokens))
        logits = self.head(h)
        ent = -(logits.log_softmax(-1).exp() * logits.log_softmax(-1)).sum(-1).detach()
        steps = torch.zeros_like(ent)
        steps[:, 1:] = ent[:, :-1]                            # think steps for token t, chosen before seeing it
        return {"logits": logits, "steps": steps, "flops": 2.0 * self.n_params() * (1 + steps)}

    def step(self, state, token):
        h = self.rnn(self.emb(token)[:, None], state)[1]
        return h, self.head(h[-1]).log_softmax(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def active_flops_per_token(self, seq=256):
        return 2 * self.n_params()


class ToyCache(ToyQF):
    """Same, but its step state grows with context (a KV-cache): criterion (c) must reject it."""

    def step(self, state, token):
        state = (state or []) + [self.emb(token)]
        return state, self.head(torch.stack(state).mean(0)).log_softmax(-1)


TOY, CACHE = "test_run_lm:ToyQF", "test_run_lm:ToyCache"


def row(arm, seed, ppl, flops=100.0, rho=None, flat=None):
    return dict(arm=arm, seed=seed, val_ppl=ppl, val_flops_per_token=flops, spearman_steps_surprisal=rho, state_flat=flat)


def test_spearman_and_steps_for_budget():
    assert run_lm.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert run_lm.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    a = run_lm.steps_for_budget(1e9, flops_per_token=1000, batch=4, seq=32)
    b = run_lm.steps_for_budget(1e9, flops_per_token=2000, batch=4, seq=32)
    assert a == int(1e9 // (3 * 1000 * 4 * 32)) and abs(a - 2 * b) <= 1     # fwd+bwd = 3x fwd; equal total FLOPs


def test_criteria_from_rows():
    ok = [row("gpt", s, 10.0) for s in range(3)]
    ok += [row("qf", s, 10.4, rho=0.4, flat=True) for s in range(3)]
    ok += [row("qf_embed", s, 10.3) for s in range(3)]                     # embed wins by <2%: write=read survives
    ok += [row("qf_fixed", s, 11.0, flops=101.0) for s in range(3)]
    c = run_lm.criteria(ok)
    assert c["a"]["pass"] and c["b"]["pass"] and c["c"]["pass"] and c["talks_ppl"]["pass"] and c["new"]
    assert c["talks_ppl"]["ratio"] == pytest.approx(1.04)

    bad = [dict(r) for r in ok]
    for r in bad:
        if r["arm"] == "qf_embed" and r["seed"] < 2:
            r["val_ppl"] = 10.0                                             # embed beats qf by 4% on 2/3 seeds
    assert not run_lm.criteria(bad)["a"]["pass"] and not run_lm.criteria(bad)["new"]

    for mut, key in ((lambda r: r.update(spearman_steps_surprisal=0.1) if r["arm"] == "qf" else None, "b"),
                     (lambda r: r.update(val_flops_per_token=150.0) if r["arm"] == "qf_fixed" else None, "b"),  # unequal FLOPs
                     (lambda r: r.update(val_ppl=10.0) if r["arm"] == "qf_fixed" else None, "b"),             # fixed beats adaptive
                     (lambda r: r.update(state_flat=False) if r["arm"] == "qf" else None, "c"),
                     (lambda r: r.update(val_ppl=11.0) if r["arm"] == "qf" else None, "talks_ppl")):
        rows = [dict(r) for r in ok]
        for r in rows:
            mut(r)
        assert not run_lm.criteria(rows)[key]["pass"], key

    missing = run_lm.criteria([row("gpt", 0, 10.0), row("qf", 0, 10.0)])
    assert missing["a"]["pass"] is False and "missing" in missing["a"]["why"]  # absent arm never passes silently


def test_state_flat():
    assert run_lm.state_flat(ToyQF(V, 16), V, lengths=(16, 64)) is True
    assert run_lm.state_flat(ToyCache(V, 16), V, lengths=(16, 64)) is False
    from lm.cores import build
    assert run_lm.state_flat(build("gpt", V, 16, layers=1, max_seq=65), V, lengths=(16, 64)) is False  # no step(): not flat


@pytest.mark.parametrize("core", ["gpt", TOY])
def test_cell_resume_bit_identical(core, tok_path, tmp_path):
    kw = dict(core=core, kw={"layers": 1, "max_seq": 33} if core == "gpt" else {}, d=16, data=FIX, tok=tok_path,
              seed=3, steps=8, seq=32, batch=4, lr=3e-3, val_batches=2)
    straight = run_lm.cell(**kw, ckpt=tmp_path / "a.pt")
    run_lm.cell(**kw, ckpt=tmp_path / "b.pt", stop_at=4)                   # fresh objects, then resume
    resumed = run_lm.cell(**kw, ckpt=tmp_path / "b.pt")
    assert resumed["val_ppl"] == straight["val_ppl"] and resumed["steps"] == straight["steps"] == 8
    assert straight["val_ppl"] < V                                          # trained below uniform
    if core == TOY:
        assert straight["spearman_steps_surprisal"] is not None and straight["val_flops_per_token"] > 0
        assert straight["state_flat"] is True


def test_main_end_to_end_and_skip_done(tok_path, tmp_path, capsys):
    out = tmp_path / "bake"
    argv = ["--data", str(FIX), "--sha256", corpus_hash(FIX), "--tok", str(tok_path),
            "--arms", f"gpt=gpt,qf={TOY}", "--arm-kw", "gpt={\"layers\":1,\"max_seq\":33}",
            "--seeds", "0,1", "--params", "34000", "--macs-budget", "1.5e9", "--seq", "32", "--batch", "4",
            "--val-batches", "2", "--out", str(out)]
    run_lm.main(argv)
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    assert {(r["arm"], r["seed"]) for r in rows} == {("gpt", 0), ("gpt", 1), ("qf", 0), ("qf", 1)}
    for r in rows:                                                          # equal FLOPs: steps * fpt matches budget
        # Amendment 6: budget in MACs at the measured expected think steps; worst case reported beside
        assert r["steps"] * 3 * r["macs_per_token"] * 4 * 32 <= 1.5e9 < (r["steps"] + 1) * 3 * r["macs_per_token"] * 4 * 32
        assert r["macs_per_token_worst"] > 0 and r["val_macs_per_token"] > 0
        if r["arm"] == "gpt":
            assert r["macs_per_token"] == r["macs_per_token_worst"]
        assert abs(r["n_params"] / 34000 - 1) < 0.05
    crit = json.loads((out / "criteria.json").read_text())
    assert set(crit) >= {"a", "b", "c", "talks_ppl", "new"}
    run_lm.main(argv)                                                       # rerun: completed cells skipped
    assert len((out / "results.jsonl").read_text().splitlines()) == 4


def test_hash_mismatch_exits_nonzero(tok_path, tmp_path):
    with pytest.raises(SystemExit) as e:
        run_lm.main(["--data", str(FIX), "--sha256", "0" * 64, "--tok", str(tok_path), "--arms", "gpt=gpt",
                     "--out", str(tmp_path / "o")])
    assert e.value.code not in (0, None)


class ToyDyn(ToyQF):
    """QF-like stand-in exposing s_max/generate so cell() takes the telemetry branch."""
    s_max = 2

    def generate(self, *a, **k):
        return None


def test_cell_dynamics_branch():
    """Detects cell() dropping the 'dynamics' key for s_max/generate models or adding it for plain ones."""
    import pytest as _p
    mp = _p.MonkeyPatch()
    mp.setattr(run_lm.telemetry, "dynamics", lambda m, n_tokens, B, seed: {"seed": seed, "n": n_tokens})
    try:
        d = Path(__import__("tempfile").mkdtemp())
        tp = d / "tok.json"; train_bpe(FIX, V, tp)
        kw = dict(kw={}, d=16, data=FIX, tok=tp, seed=3, steps=2, seq=32, batch=4, lr=3e-3, val_batches=1)
        assert run_lm.cell(core="test_run_lm:ToyDyn", ckpt=d / "a.pt", **kw)["dynamics"] == {"seed": 3, "n": 256}
        assert "dynamics" not in run_lm.cell(core=TOY, ckpt=d / "b.pt", **kw)
    finally:
        mp.undo()

"""Inspector gates for PROMPT.md §1 laws: private slugs ignored, no AI attribution, L-PREREG order, L-SHA pin,
fair Mapper baseline. Run before every commit (Security: "test asserts it before every commit")."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

ZERO = Path(__file__).resolve().parents[1]
ROOT = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ZERO, text=True).strip())
REL = ZERO.relative_to(ROOT).as_posix()                 # "zero"
SHA_PIN = "467c2ae25f92121093a60435c3397bd5f1e9fe246a2f602b4fc03bbace40a3fe"
SKIP_DIRS = {".git", ".pytest_cache", ".benchmarks", "__pycache__", "node_modules"}


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace")


def ignored(path):
    return git("check-ignore", "-q", "--no-index", f"{REL}/{path}").returncode == 0


# ---- (a) private slugs ------------------------------------------------------------------------
PRIVATE = ["kaggle.json", "sub/kaggle.json", "hf.token", ".env", "secrets.yaml", "secrets/x", ".venv-t0/x",
           ".venv-gpu/x", "model.ckpt", "runs/ck/model.ckpt", "data/x.npz", "datasets/x.npz"]
PUBLIC = ["PREREG.md", "runs/t2_s2.json", "ref/identities.py", "common.py", "tests/test_laws.py"]


@pytest.mark.parametrize("path", PRIVATE)
def test_private_slug_ignored(path):
    assert ignored(path), f"{REL}/{path} is not gitignored"


def test_no_private_slug_tracked():
    """check-ignore --no-index says a pattern exists; this says nothing matching it was force-added."""
    tracked = git("ls-files", "-ci", "--exclude-standard", "--", REL).stdout.split()
    assert not tracked, f"tracked/staged files match private patterns: {tracked}"


@pytest.mark.parametrize("path", PUBLIC)
def test_record_not_ignored(path):
    assert not ignored(path), f"{REL}/{path} is gitignored but must be committed"


# ---- (b) no AI attribution ----------------------------------------------------------------------
ATTRIB = re.compile("|".join(["Co-" + "Authored-By", "noreply@" + "anthropic", "Generated with " + r"\[?Claude",
                              "Generated (with|by) " + "(ChatGPT|Copilot|GPT-|an AI)"]), re.I)


def text_files():
    for p in ZERO.rglob("*"):
        rel = p.relative_to(ZERO)
        if (p.is_file() and not set(rel.parts) & SKIP_DIRS and not rel.parts[0].startswith(".venv")
                and rel.as_posix() != "PROMPT.md" and p.stat().st_size < 5_000_000):
            yield p


def test_no_ai_attribution_in_files():
    hits = []
    for p in text_files():
        try:
            txt = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits += [f"{p.relative_to(ZERO)}:{i}" for i, ln in enumerate(txt.splitlines(), 1) if ATTRIB.search(ln)]
    assert not hits, hits


def test_no_ai_attribution_in_commits():
    log = git("log", "--format=%H%n%B", "--", REL).stdout
    assert not ATTRIB.search(log), "a commit touching zero/ carries an AI attribution line"


# ---- (c) L-PREREG: PREREG.md committed strictly before any result exists ------------------------
def run_files():
    d = ZERO / "runs"
    return [p for p in d.rglob("*") if p.is_file() and p.name != ".gitkeep"] if d.exists() else []


def test_prereg_before_runs():
    files = run_files()
    run_commits = git("log", "--format=%H", "--", f"{REL}/runs").stdout.split()
    if not files and not run_commits:
        return
    added = git("log", "--format=%H", "--diff-filter=A", "--", f"{REL}/PREREG.md").stdout.split()
    assert added, f"result files exist ({[p.name for p in files][:5]}) but PREREG.md was never committed"
    prereg = added[-1]                                   # first commit that added PREREG.md
    for c in run_commits:
        assert c != prereg, f"commit {c[:9]} adds PREREG.md and results together"
        assert git("merge-base", "--is-ancestor", prereg, c).returncode == 0, \
            f"commit {c[:9]} touches runs/ but is not a descendant of the PREREG commit {prereg[:9]}"


POST_HOC = re.compile(r"\s*#+ [^\n]*post-hoc", re.I)


def test_prereg_read_only():
    """L-AUDIT: once committed, PREREG.md only grows by an appended section headed '... post-hoc ...'."""
    path = f"{REL}/PREREG.md"
    commits = git("log", "--reverse", "--format=%H", "--", path).stdout.split()
    if not commits:
        return
    show = lambda c: git("show", f"{c}:{path}")
    base = show(commits[0]).stdout.replace("\r\n", "\n")
    versions = [(c[:9], show(c)) for c in commits[1:]]
    wt = ZERO / "PREREG.md"
    versions.append(("worktree", subprocess.CompletedProcess([], 0 if wt.exists() else 1,
                                                             wt.read_text(encoding="utf-8") if wt.exists() else "")))
    for name, v in versions:
        assert v.returncode == 0, f"PREREG.md deleted at {name}"
        txt = v.stdout.replace("\r\n", "\n")
        assert txt.startswith(base), f"PREREG.md committed text modified at {name}"
        assert txt == base or POST_HOC.match(txt[len(base):]), f"PREREG.md append at {name} not under a post-hoc heading"


# ---- (d) L-SHA ----------------------------------------------------------------------------------
def test_identities_sha_pin():
    assert hashlib.sha256((ZERO / "ref" / "identities.py").read_bytes()).hexdigest() == SHA_PIN


# ---- Mapper fairness (C2 baseline, L-BR) --------------------------------------------------------
def test_mapper_baseline_fair():
    """Record contract: any run JSON that reports a nerve Betti score ("nerve_betti") must carry
    "mapper": {"n_samples", "cover", "tune_seed"} with n_samples == the record's "nerve_n_samples" and
    tune_seed not among the record's "seeds" (cover tuned on the tuning seed only)."""
    bad = []
    for p in run_files():
        if p.suffix != ".json":
            continue
        recs = json.loads(p.read_text(encoding="utf-8"))
        for r in recs if isinstance(recs, list) else [recs]:
            if not isinstance(r, dict) or "nerve_betti" not in r:
                continue
            m = r.get("mapper")
            if not isinstance(m, dict) or not {"n_samples", "cover", "tune_seed"} <= m.keys():
                bad.append(f"{p.name}: no mapper baseline record")
            elif m["n_samples"] != r.get("nerve_n_samples"):
                bad.append(f"{p.name}: mapper n_samples {m['n_samples']} != nerve {r.get('nerve_n_samples')}")
            elif m["tune_seed"] in r.get("seeds", []):
                bad.append(f"{p.name}: mapper cover tuned on an eval seed {m['tune_seed']}")
    assert not bad, bad


def test_mapper_cover_tunable():
    metrics = pytest.importorskip("metrics", reason="metrics.py (Foreman) not built yet")
    import inspect
    params = set(inspect.signature(metrics.mapper_betti).parameters) - {"X", "x", "filter_dim", "max_dim"}
    assert params, "mapper_betti exposes no cover parameters: cover cannot be tuned (unfair baseline)"

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

QDIR = Path(__file__).resolve().parents[1] / "queue"
sys.path.insert(0, str(QDIR))
import runner  # noqa: E402

PY, GiB = sys.executable, 2**30
LINE = re.compile(r"^\S+ (START \d+|PROGRESS \d+/\d+|FAIL \S+ -?\d+ .*|DONE|HALT .+)$")

OK = "import json,sys,time; time.sleep(float(sys.argv[2])); json.dump({'ok':1}, open(sys.argv[1],'w'))"
# counts its own executions in <out>.runs, then writes out
COUNT = "import json,sys,time; open(sys.argv[1]+'.runs','a').write('x'); time.sleep(float(sys.argv[2])); json.dump({}, open(sys.argv[1],'w'))"
# records start/end wall times in <out>.t, then writes out
TIMED = ("import json,sys,time; t0=time.time(); time.sleep(float(sys.argv[2])); "
         "open(sys.argv[1]+'.t','w').write(f'{t0} {time.time()}'); json.dump({}, open(sys.argv[1],'w'))")
# fails the first N attempts (attempt count kept in <out>.n), then succeeds
FLAKY = ("import json,sys; n=sys.argv[1]+'.n'; open(n,'a').write('x'); k=len(open(n).read()); "
         "sys.exit(1) if k<=int(sys.argv[2]) else json.dump({}, open(sys.argv[1],'w'))")
FAIL = "import sys; open(sys.argv[1]+'.n','a').write('x'); sys.stderr.write('boom\\n'); sys.exit(3)"
PROGRESS = ("import sys,time,json; open(sys.argv[1]+'.progress.jsonl','a').write(json.dumps({'step':4200})+'\\n'); "
            "time.sleep(2.0); json.dump({}, open(sys.argv[1],'w'))")


@pytest.fixture(autouse=True)
def plenty(monkeypatch):
    monkeypatch.setattr(runner, "mem_free", lambda: (16 * GiB, 16 * GiB))
    monkeypatch.setattr(runner, "disk_free", lambda p: 100 * GiB)


def git(repo, *a):
    # drop GIT_* (a commit hook exports GIT_DIR): with it, `git init` here re-inits the real repo as bare
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo), *a], check=True,
                   capture_output=True, env=env)


def mkrepo(path, commit_prereg=True):
    z = path / "zero"
    z.mkdir(parents=True)
    git(path, "init", "-q")
    (z / "a.py").write_text("x = 1\n")
    (z / "PREREG.md").write_text("prereg\n")
    git(path, "add", "zero/a.py", *(["zero/PREREG.md"] if commit_prereg else []))
    git(path, "commit", "-qm", "init")
    return path


@pytest.fixture
def repo(tmp_path):
    return mkrepo(tmp_path / "repo")


def job(tmp, id, code=OK, args=("0.05",), gpu=False, est=1.0, **kw):
    out = tmp / f"{id}.json"
    return dict(id=id, cmd=[PY, "-c", code, str(out), *args], out=str(out), gpu=gpu, est_s=est, **kw)


def go(tmp, repo, jobs, name="state", **kw):
    jf = tmp / "jobs.jsonl"
    jf.write_text("\n".join(json.dumps(j) for j in jobs) + "\n")
    state = tmp / name
    kw = dict(dict(backoff=0.05, poll=0.05, repo=repo), **kw)
    return runner.run(jf, state, **kw), state


def events(state):
    return (state / "events.log").read_text().splitlines()


def overlap(tmp, ids):
    iv = [tuple(map(float, (tmp / f"{i}.json.t").read_text().split())) for i in ids]
    pts = [(s, 1) for s, _ in iv] + [(e, -1) for _, e in iv]
    cur = best = 0
    for _, d in sorted(pts):
        cur += d
        best = max(best, cur)
    return best


def test_concurrency_cap(tmp_path, repo):
    g = [job(tmp_path, f"g{i}", TIMED, ("0.6",), gpu=True) for i in range(4)]
    c = [job(tmp_path, f"c{i}", TIMED, ("0.6",)) for i in range(6)]
    rc, st = go(tmp_path, repo, g + c, max_gpu=1, max_cpu=2)
    assert rc == 0 and events(st)[-1].endswith("DONE")
    assert overlap(tmp_path, [j["id"] for j in g]) == 1  # gpu jobs never overlap
    assert overlap(tmp_path, [j["id"] for j in c]) == 2  # cpu jobs overlap, but never beyond the cap


def test_memory_gate(tmp_path, repo, monkeypatch):
    mem = [(0.5 * GiB, 16 * GiB)]
    monkeypatch.setattr(runner, "mem_free", lambda: mem[0])
    j = job(tmp_path, "m")
    res = []
    t = threading.Thread(target=lambda: res.append(go(tmp_path, repo, [j])))
    t.start()
    time.sleep(1.0)
    assert not Path(j["out"]).exists() and not (tmp_path / "state" / "logs" / "m.err").exists()  # nothing launched
    mem[0] = (16 * GiB, 16 * GiB)
    t.join(30)
    assert Path(j["out"]).exists() and res[0][0] == 0


def test_memory_gate_commit_short(tmp_path, repo, monkeypatch):
    monkeypatch.setattr(runner, "mem_free", lambda: (16 * GiB, 1 * GiB))  # commit below 3 GB blocks on its own
    rc, st = go(tmp_path, repo, [job(tmp_path, "m")], mem_halt_s=0.3)
    assert rc != 0 and not Path(tmp_path / "m.json").exists()


def test_halt_memory_and_disk(tmp_path, repo, monkeypatch):
    monkeypatch.setattr(runner, "mem_free", lambda: (0, 0))
    rc, st = go(tmp_path, repo, [job(tmp_path, "m")], mem_halt_s=0.3)
    assert rc != 0 and re.match(r"\S+ HALT memory", events(st)[-1]) and not Path(tmp_path / "m.json").exists()
    assert runner.status_line(st).startswith("halted: memory")
    monkeypatch.setattr(runner, "mem_free", lambda: (16 * GiB, 16 * GiB))
    monkeypatch.setattr(runner, "disk_free", lambda p: 1 * GiB)
    rc, st = go(tmp_path, repo, [job(tmp_path, "m")], name="state2")
    assert rc != 0 and re.match(r"\S+ HALT disk", events(st)[-1]) and not Path(tmp_path / "m.json").exists()


def test_retry_backoff(tmp_path, repo):
    flaky = job(tmp_path, "flaky", FLAKY, ("2",))
    bad = job(tmp_path, "bad", FAIL, ())
    rc, st = go(tmp_path, repo, [flaky, bad])
    assert Path(flaky["out"]).exists() and len((tmp_path / "flaky.json.n").read_text()) == 3
    assert not Path(bad["out"]).exists() and len((tmp_path / "bad.json.n").read_text()) == 3
    fails = [l for l in events(st) if " FAIL " in l]
    assert len(fails) == 1 and fails[0].endswith("FAIL bad 3 boom")
    assert json.loads((st / "status.json").read_text())["failed"] == ["bad"]


def test_resume_after_stop(tmp_path, repo):
    jobs = [job(tmp_path, f"r{i}", COUNT, ("0.05",)) for i in range(5)]
    rc, st = go(tmp_path, repo, jobs, max_launch=2)
    assert rc != 0 and sum(Path(j["out"]).exists() for j in jobs) == 2
    assert "DONE" not in "".join(events(st))
    rc, st = go(tmp_path, repo, jobs)
    assert rc == 0
    for j in jobs:
        assert Path(j["out"]).exists() and (tmp_path / f"{Path(j['out']).name}.runs").read_text() == "x"  # ran once


def test_resume_adopts_live_child(tmp_path, repo):
    j = job(tmp_path, "orphan", COUNT, ("1.0",))
    child = subprocess.Popen(j["cmd"])  # stands in for a child of a dead runner, still running
    st = tmp_path / "state"
    st.mkdir()
    (st / "status.json").write_text(json.dumps(dict(state="running", pids={"orphan": child.pid})))
    rc, _ = go(tmp_path, repo, [j])
    child.wait()
    assert rc == 0 and Path(j["out"]).exists() and (tmp_path / "orphan.json.runs").read_text() == "x"


def test_dirty_tree_refusal(tmp_path, repo):
    j = job(tmp_path, "x")
    (repo / "zero" / "new.txt").write_text("untracked is fine")
    rc, st = go(tmp_path, repo, [j])
    assert rc == 0 and Path(j["out"]).exists()
    (repo / "zero" / "a.py").write_text("x = 2\n")  # modified tracked file
    Path(j["out"]).unlink()
    rc, st = go(tmp_path, repo, [j])
    assert rc != 0 and re.match(r"\S+ HALT dirty tree: ", events(st)[-1]) and not Path(j["out"]).exists()
    r = subprocess.run([PY, str(QDIR / "runner.py"), "start", str(tmp_path / "jobs.jsonl"), "--state", str(tmp_path / "s2"),
                        "--repo", str(repo)], capture_output=True, text=True)  # the CLI refuses before spawning
    assert r.returncode != 0 and not Path(j["out"]).exists()
    assert "HALT dirty tree" in events(tmp_path / "s2")[-1]
    r2 = mkrepo(tmp_path / "r2", commit_prereg=False)  # PREREG untracked
    rc, st = go(tmp_path, r2, [j], name="s3")
    assert rc != 0 and "HALT dirty tree" in events(st)[-1] and "PREREG" in events(st)[-1]
    r3 = mkrepo(tmp_path / "r3")  # PREREG modified
    (r3 / "zero" / "PREREG.md").write_text("edited\n")
    rc, st = go(tmp_path, r3, [j], name="s4")
    assert rc != 0 and "HALT dirty tree" in events(st)[-1]


def test_events_log_milestones_only(tmp_path, repo):
    jobs = [job(tmp_path, f"ok{i}") for i in range(8)] + [job(tmp_path, "bad", FAIL, ()),
                                                           job(tmp_path, "flaky", FLAKY, ("1",))]
    rc, st = go(tmp_path, repo, jobs)
    ls = events(st)
    assert all(LINE.match(l) for l in ls), ls
    kinds = [l.split()[1] for l in ls]
    assert kinds[0] == "START" and kinds[-1] == "DONE" and kinds.count("START") == kinds.count("DONE") == 1
    assert 1 <= kinds.count("PROGRESS") <= 10 and kinds.count("FAIL") == 1


def test_status_one_line(tmp_path, repo):
    s1 = job(tmp_path, "s1", PROGRESS, ())
    res = []
    t = threading.Thread(target=lambda: res.append(go(tmp_path, repo, [s1, job(tmp_path, "s2")])))
    t.start()
    seen = ""
    for _ in range(100):
        time.sleep(0.1)
        if (tmp_path / "state" / "status.json").exists():
            seen = runner.status_line(tmp_path / "state")
            if "@step 4200" in seen:
                break
    assert seen.startswith("running ") and "s1@step 4200" in seen and len(seen.splitlines()) == 1, seen
    t.join(30)
    r = subprocess.run([PY, str(QDIR / "runner.py"), "status", "--state", str(tmp_path / "state")], capture_output=True,
                       text=True)
    assert len(r.stdout.splitlines()) == 1 and r.stdout.startswith("done 2/2 (0 failed)"), r.stdout


def probe_cmd(tmp, name, vram):
    code = ("import json,os; open(os.path.join(%r,'probe_%s.n'),'a').write('x'); "
            "json.dump({'vram_peak_bytes': %d, 'step_s': 0.01}, open(os.environ['QUEUE_PROBE_OUT'],'w'))"
            % (str(tmp), name, vram))
    return [PY, "-c", code]


def test_vram_probe_admission(tmp_path, repo):
    small = [job(tmp_path, f"s{i}", COUNT, ("0.05",), gpu=True, type="t:small",
                 probe_cmd=probe_cmd(tmp_path, "small", 1 * GiB)) for i in range(2)]
    big = job(tmp_path, "big", COUNT, ("0.05",), gpu=True, type="t:big", probe_cmd=probe_cmd(tmp_path, "big", 9 * GiB))
    rc, st = go(tmp_path, repo, small + [big])
    assert all(Path(j["out"]).exists() for j in small) and not Path(big["out"]).exists()
    assert not (tmp_path / "big.json.runs").exists()  # never launched
    assert [l.split(" ", 1)[1] for l in events(st) if " FAIL " in l] == ["FAIL big -2 vram probe exceeds budget"]
    assert (tmp_path / "probe_small.n").read_text() == "x"  # probed once per type, not per job
    probes = json.loads((st / "status.json").read_text())["probes"]
    assert probes["t:small"]["vram_peak_bytes"] == GiB and probes["t:big"]["vram_peak_bytes"] == 9 * GiB


def test_gpu_longest_first(tmp_path, repo):
    jobs = [job(tmp_path, f"g{e}", TIMED, ("0.1",), gpu=True, est=e) for e in (1, 3, 2)]
    go(tmp_path, repo, jobs, max_gpu=1)
    start = {j["id"]: float((tmp_path / f"{j['id']}.json.t").read_text().split()[0]) for j in jobs}
    assert sorted(start, key=start.get) == ["g3", "g2", "g1"]


def test_after_dependency(tmp_path, repo):
    a, b = job(tmp_path, "a", TIMED, ("0.5",)), job(tmp_path, "b", TIMED, ("0.1",), after=["a"])
    go(tmp_path, repo, [a, b], max_cpu=2)
    a_end = float((tmp_path / "a.json.t").read_text().split()[1])
    assert float((tmp_path / "b.json.t").read_text().split()[0]) >= a_end - 0.05
    bad, c = job(tmp_path, "bad", FAIL, ()), job(tmp_path, "c", after=["bad"])
    rc, st = go(tmp_path, repo, [bad, c], name="s2", retries=0)
    assert not Path(c["out"]).exists() and any(l.endswith("FAIL c -1 dependency failed: bad") for l in events(st))

"""Long-run job queue: resumable (done = `out` parses as JSON), memory/disk gated, retries, never kills anything.

  python queue/runner.py start JOBS.jsonl [--state DIR]   preflight, then relaunch itself detached; prints the pid
  python queue/runner.py run   JOBS.jsonl [--state DIR]   foreground scheduler loop
  python queue/runner.py status [--state DIR]             exactly one line

Job: {"id", "cmd": [argv], "out", "gpu": bool, "est_s", "cwd"?, "type"?, "probe_cmd"?, "steps"?, "after"?: [ids]}.
Child env: OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, QUEUE_JOB_DIR=<dir of out>; probe_cmd also gets QUEUE_PROBE_OUT=<json path>
and must write {"vram_peak_bytes": int, "step_s": float}. Children may append JSONL to <out>.progress.jsonl.
"""
import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
GiB = 2**30
MAX_GPU, MAX_CPU = 1, 2
COMMIT_MIN, PHYS_MIN, DISK_MIN, VRAM_BUDGET = 3 * GiB, 1.5 * GiB, 20 * GiB, 7 * GiB
MEM_HALT_S, RETRIES, PROBE_TIMEOUT_S = 1800, 2, 1800
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class _MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
        (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                         "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]


def mem_free():
    """-> (free physical bytes, free commit bytes); injectable for tests."""
    if hasattr(ctypes, "windll"):
        s = _MS()
        s.dwLength = ctypes.sizeof(s)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullAvailPhys, s.ullAvailPageFile
    try:
        meminfo = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    meminfo[key] = int(val) * 1024
        avail_phys = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        avail_swap = meminfo.get("SwapFree", 0)
        return avail_phys, avail_phys + avail_swap
    except Exception:
        return 10 * GiB, 10 * GiB


def disk_free(path):
    return shutil.disk_usage(path).free


def mem_ok(phys, commit):
    return phys >= PHYS_MIN and commit >= COMMIT_MIN


def pid_alive(pid):  # never os.kill(pid, 0) on Windows: that terminates the process
    if hasattr(ctypes, "windll"):
        k = ctypes.windll.kernel32
        k.OpenProcess.restype = ctypes.c_void_p
        h = k.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code))
        k.CloseHandle(ctypes.c_void_p(h))
        return code.value == 259
    try:
        os.kill(int(pid), 0)
        try:
            with open(f"/proc/{pid}/status", "r") as f:
                for line in f:
                    if line.startswith("State:"):
                        if "Z" in line or "zombie" in line:
                            return False
                        break
        except Exception:
            pass
        return True
    except OSError:
        return False


def stale_since_boot(path):  # status.json older than the last boot: recorded pids may belong to other processes now
    if hasattr(ctypes, "windll"):
        k = ctypes.windll.kernel32
        k.GetTickCount64.restype = ctypes.c_ulonglong
        return os.path.getmtime(path) < time.time() - k.GetTickCount64() / 1000
    try:
        with open("/proc/uptime", "r") as f:
            uptime = float(f.read().split()[0])
        return os.path.getmtime(path) < time.time() - uptime
    except Exception:
        return False


class Adopted:
    """A still-running child of a previous runner (we only know its pid)."""
    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        return None if pid_alive(self.pid) else 0


# ---------------- small helpers ----------------
def now():
    return datetime.now().isoformat(timespec="seconds")


def log_event(state, msg):
    Path(state).mkdir(parents=True, exist_ok=True)
    with open(Path(state) / "events.log", "a") as f:
        f.write(f"{now()} {msg}\n")


def save_status(state, st):
    p, tmp = Path(state) / "status.json", Path(state) / "status.json.tmp"
    tmp.write_text(json.dumps(st))
    for i in range(20):  # a concurrent reader can briefly hold the target open on Windows
        try:
            return os.replace(tmp, p)
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, p)


def safe(s):
    return re.sub(r"[^\w.-]", "_", s)


def abs_out(j):
    p = Path(j["out"])
    return p if p.is_absolute() else Path(j.get("cwd") or os.getcwd()) / p


def is_done(j):
    try:
        json.loads(abs_out(j).read_text())
        return True
    except (OSError, ValueError):
        return False


def preflight(repo, prereg=None):
    """None if clean, else the reason: modified tracked file under zero/, or PREREG untracked/modified."""
    repo = Path(repo)
    prereg = prereg or "zero/PREREG.md"
    # drop GIT_* (a commit hook exports GIT_DIR/GIT_INDEX_FILE): they override -C and point git at another repo
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    st = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "zero/"], capture_output=True, text=True,
                        env=env)
    if st.returncode:
        return f"dirty tree: git status failed: {st.stderr.strip()}"
    bad = [l.strip() for l in st.stdout.splitlines() if not l.startswith("??")]
    if subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", prereg], capture_output=True,
                      env=env).returncode:
        bad.append(f"{prereg} is not tracked")
    return "dirty tree: " + "; ".join(bad) if bad else None


def last_err(state, id):
    try:
        ls = [l for l in (Path(state) / "logs" / f"{safe(id)}.err").read_text(errors="replace").splitlines() if l.strip()]
        return ls[-1].strip() if ls else "(no stderr)"
    except OSError:
        return "(no stderr)"


def read_jobs(path):
    js = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    assert len({j["id"] for j in js}) == len(js), "duplicate job ids"
    return js


# ---------------- scheduler ----------------
def run(jobs_path, state=HERE, *, max_gpu=MAX_GPU, max_cpu=MAX_CPU, backoff=30.0, poll=3.0, repo=None, prereg=None,
        max_launch=None, mem_halt_s=MEM_HALT_S, vram_budget=VRAM_BUDGET, retries=RETRIES):
    """Returns 0 when all jobs are settled (DONE), 2 on HALT, 3 if stopped by max_launch (test hook)."""
    state = Path(state)
    (state / "logs").mkdir(parents=True, exist_ok=True)
    jobs = read_jobs(jobs_path)
    byid = {j["id"]: j for j in jobs}
    prior = {}
    try:
        prior = json.loads((state / "status.json").read_text())
    except (OSError, ValueError):
        pass
    done = {j["id"] for j in jobs if is_done(j)}
    failed, attempts, next_at, active = {}, {}, {}, {}  # active: id -> (proc, gpu)
    probes = dict(prior.get("probes", {}))
    if prior.get("pids") and not stale_since_boot(state / "status.json"):
        for id, pid in prior["pids"].items():  # relaunching these would run one job twice on the same out/checkpoint
            if id in byid and id not in done and pid_alive(pid):
                active[id] = (Adopted(pid), bool(byid[id].get("gpu")))
    st = dict(state="running", total=len(jobs), done=0, failed=[], active=[], pids={}, outs={}, probes=probes,
              started=now(), last="START", halt=None, pid=os.getpid())

    def save():
        st.update(done=len(done), failed=list(failed), active=list(active), probes=probes,
                  pids={i: p.pid for i, (p, _) in active.items()}, outs={i: str(abs_out(byid[i])) for i in active})
        save_status(state, st)

    def ev(msg, last=None):
        log_event(state, msg)
        st["last"] = last or msg
        save()

    def fail(id, code, msg):
        failed[id] = code
        ev(f"FAIL {id} {code} {msg}")

    why = preflight(repo or HERE.parent.parent, prereg)
    if why:
        st.update(state="halted", halt=why)
        ev(f"HALT {why}")
        return 2
    ev(f"START {len(jobs)}", f"START {len(jobs)}")
    boundary, launches, short_since, halted = len(done) * 10 // max(len(jobs), 1), 0, None, False

    def halt(reason):
        nonlocal halted
        halted = True
        st.update(halt=reason)
        ev(f"HALT {reason}")

    def child_env(j):
        d = abs_out(j).parent
        d.mkdir(parents=True, exist_ok=True)
        return dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", QUEUE_JOB_DIR=str(d))

    def probe(j):
        out = state / f"probe_{safe(j['type'])}.json"
        out.unlink(missing_ok=True)
        try:
            r = subprocess.run(j["probe_cmd"], cwd=j.get("cwd"), env=dict(child_env(j), QUEUE_PROBE_OUT=str(out)),
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=NOWIN, timeout=PROBE_TIMEOUT_S)
            p = json.loads(out.read_text()) if r.returncode == 0 else None
        except (OSError, ValueError, subprocess.TimeoutExpired):
            p = None
        if p is None:
            return fail(j["id"], -3, "probe failed")
        probes[j["type"]] = p
        save()

    def prio(j):  # gpu jobs longest-first
        p = probes.get(j.get("type"))
        return p["step_s"] * j["steps"] if p and j.get("steps") else j.get("est_s", 0)

    while True:
        for id, (p, g) in list(active.items()):
            rc = p.poll()
            if rc is None:
                continue
            del active[id]
            if rc == 0 and is_done(byid[id]):
                done.add(id)
                continue
            attempts[id] = attempts.get(id, 0) + 1
            if attempts[id] > retries:
                fail(id, rc, last_err(state, id))
            else:
                next_at[id] = time.time() + backoff * 4 ** (attempts[id] - 1)  # 30 s, 120 s at the default base
        b = len(done) * 10 // max(len(jobs), 1)
        if b > boundary:
            boundary = b
            ev(f"PROGRESS {len(done)}/{len(jobs)}", f"PROGRESS {len(done) * 100 // len(jobs)}%")
        settled = done | set(failed)
        if not active and (halted or len(settled) == len(jobs) or (max_launch is not None and launches >= max_launch)):
            break
        if not halted and disk_free(state) < DISK_MIN:
            halt(f"disk (free < {DISK_MIN // GiB} GB)")
        if not halted and (max_launch is None or launches < max_launch):
            ready = []
            for j in jobs:
                id = j["id"]
                if id in settled or id in active or next_at.get(id, 0) > time.time():
                    continue
                dead = [d for d in j.get("after", []) if d in failed]
                if dead:
                    fail(id, -1, f"dependency failed: {dead[0]}")
                elif all(d in done for d in j.get("after", [])):
                    ready.append(j)
            order = sorted([j for j in ready if j.get("gpu")], key=prio, reverse=True) + [j for j in ready if not j.get("gpu")]
            blocked = False
            for j in order:
                g, id = bool(j.get("gpu")), j["id"]
                ng = sum(a[1] for a in active.values())
                if (g and ng >= max_gpu) or (not g and len(active) - ng >= max_cpu):
                    continue
                if not mem_ok(*mem_free()):
                    blocked = True
                    break
                t = j.get("type")
                if j.get("probe_cmd") and t is not None and t not in probes:
                    if g and ng:
                        continue  # probes of gpu jobs run alone on the gpu
                    probe(j)
                    if id in failed:
                        continue
                if g and t in probes and probes[t].get("vram_peak_bytes", 0) > vram_budget:
                    fail(id, -2, "vram probe exceeds budget")
                    continue
                with open(state / "logs" / f"{safe(id)}.err", "ab") as err:
                    p = subprocess.Popen(j["cmd"], cwd=j.get("cwd"), env=child_env(j), stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL, stderr=err, creationflags=NOWIN)
                active[id] = (p, g)
                launches += 1
                break  # ponytail: one launch per poll so the memory gate sees the previous child's footprint
            short_since = (short_since or time.time()) if blocked else None
            if blocked and time.time() - short_since > mem_halt_s:
                halt(f"memory (free below gate for {mem_halt_s:g}s)")
        save()
        time.sleep(poll)
    if halted or (max_launch is not None and launches >= max_launch and len(settled) < len(jobs)):
        st["state"] = "halted" if halted else "stopped"
        save()
        return 2 if halted else 3
    st["state"] = "done"
    ev("DONE")
    return 0


# ---------------- status / CLI ----------------
def step_of(out):
    try:
        tail = Path(str(out) + ".progress.jsonl").read_bytes()[-4096:].splitlines()
        return json.loads(tail[-1])["step"]
    except (OSError, ValueError, KeyError, IndexError):
        return None


def status_line(state):
    s = None
    for _ in range(10):  # the runner's os.replace can briefly lock the file
        try:
            s = json.loads((Path(state) / "status.json").read_text())
            break
        except (OSError, ValueError):
            time.sleep(0.05)
    if s is None:
        return "no status"
    if s["state"] == "halted":
        return f"halted: {s['halt']}".replace("\n", " ")
    base = f"{s['done']}/{s['total']} done"
    if s["state"] == "done":
        return f"done {s['done']}/{s['total']} ({len(s['failed'])} failed)"
    word = "running" if s["state"] == "running" and pid_alive(s["pid"]) else "stopped"
    steps = [f"{i}@step {n}" for i in s["active"] if (n := step_of(s["outs"].get(i, ""))) is not None]
    return f"{word} {base}, {len(s['active'])} active, {len(s['failed'])} failed, last={s['last']}" + (
        f", active: {' '.join(steps)}" if steps else "")


def common(ap):
    ap.add_argument("--state", default=str(HERE))
    ap.add_argument("--repo", default=None)
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--poll", type=float, default=3.0)
    ap.add_argument("--backoff", type=float, default=30.0)
    ap.add_argument("--max-gpu", type=int, default=MAX_GPU)
    ap.add_argument("--max-cpu", type=int, default=MAX_CPU)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("start", "run"):
        common(p := sub.add_parser(name))
        p.add_argument("jobs")
    common(sub.add_parser("status"))
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print(status_line(a.state))
        return 0
    kw = dict(max_gpu=a.max_gpu, max_cpu=a.max_cpu, backoff=a.backoff, poll=a.poll, repo=a.repo, prereg=a.prereg)
    if a.cmd == "run":
        return run(a.jobs, a.state, **kw)
    state = Path(a.state)
    why = preflight(a.repo or HERE.parent.parent, a.prereg)
    if why:
        log_event(state, f"HALT {why}")
        save_status(state, dict(state="halted", halt=why, total=0, done=0, failed=[], active=[]))
        print(f"HALT {why}", file=sys.stderr)
        return 2
    try:
        s = json.loads((state / "status.json").read_text())
        if s["state"] == "running" and pid_alive(s["pid"]):
            print(f"already running: pid {s['pid']}", file=sys.stderr)
            return 1
    except (OSError, ValueError, KeyError):
        pass
    argv = [sys.executable, str(Path(__file__).resolve()), "run", str(Path(a.jobs).resolve()), "--state", str(state),
            "--poll", str(a.poll), "--backoff", str(a.backoff), "--max-gpu", str(a.max_gpu), "--max-cpu", str(a.max_cpu)]
    argv += [x for k, v in (("--repo", a.repo), ("--prereg", a.prereg)) if v for x in (k, v)]
    state.mkdir(parents=True, exist_ok=True)
    with open(state / "runner.log", "ab") as dbg:
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=dbg, close_fds=True,
                             creationflags=0x8 | subprocess.CREATE_NEW_PROCESS_GROUP | NOWIN)  # DETACHED_PROCESS
    print(p.pid)
    return 0


if __name__ == "__main__":
    sys.exit(main())

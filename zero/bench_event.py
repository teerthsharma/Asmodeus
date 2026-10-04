"""Event-driven T7 coupling (PREREG_DRAFT_event.md "T7 coupling"): trigger rates r(eps) measured on bed data,
and streaming us/step of the event step vs always-compute fused_observe (eager and CUDA graph) at B in {1, 64}.
Rows of a stream are trajectories cycling over T; frame 0 of each cycle always triggers (as EventDriven.triggers),
so the replayed trigger sequence is the data's. Prints JSON lines to stdout; writes no files."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas.event_driven import EventDriven  # noqa: E402
from atlas.observer_atlas import ObserverAtlas  # noqa: E402
from bench_t7 import CFG, CellBackbone, fused_observe, graphed, sm_clock, time_us  # noqa: E402

QS = (0.25, 0.5, 0.75, 0.9)
_DUMMY = EventDriven(torch.nn.Identity())


def prev_deltas(obs):
    """Delta against the PREVIOUS frame, t >= 1: [B, T-1]."""
    return (obs[:, 1:] - obs[:, :-1]).norm(dim=-1) / math.sqrt(obs.shape[-1])


def trigger_rate(obs, eps):
    """Exact trigger fraction over all B*T frames (t = 0 included), Delta vs the last trigger."""
    return float(_DUMMY.triggers(obs, eps)[0].float().mean())


class EventStream:
    """Event step: Delta per row vs its last triggered frame; fused_observe on triggered rows only, sample-and-hold
    for the rest. Row b = trajectory b, frame counter cycles over T. `if trig.any()` is a host sync (honest cost)."""

    def __init__(self, at, cell, obs, act, B, eps):
        self.at, self.cell, self.obs, self.act, self.B, self.eps = at, cell, obs[:B], act[:B], B, eps
        self.T, self.t, self.n_trig, self.n_frames = obs.shape[1], 0, 0, 0
        self.ref = torch.zeros(B, obs.shape[-1], device=obs.device)
        self.g = torch.zeros(B, at.D_g, device=obs.device)
        self.gh = torch.zeros_like(self.g)

    def step(self):
        t = self.t % self.T
        self.t += 1
        self.n_frames += self.B
        o = self.obs[:, t]
        trig = (o - self.ref).norm(dim=-1) / math.sqrt(o.shape[-1]) >= self.eps
        if t == 0:
            trig = torch.ones_like(trig)
        if trig.any():
            idx = trig.nonzero().squeeze(1)
            x = o.index_select(0, idx)
            out = fused_observe(self.at, x[:, None], self.act[:, t].index_select(0, idx)[:, None], self.cell)
            self.g.index_copy_(0, idx, out["g"])
            self.gh.index_copy_(0, idx, out["gh"])
            self.ref.index_copy_(0, idx, x)
            self.n_trig += idx.numel()

    @property
    def r(self):
        return self.n_trig / max(self.n_frames, 1)


class EventStreamGraph:
    """CUDA-graph-capturable event step: static buffers, frame counter and trigger count live on the device,
    no host sync. mode 'masked': fused_observe on ALL rows, g/gh/ref held via torch.where (no work skipped).
    mode 'inkernel': the Triton kernel computes Delta and skips held rows (g/gh/ref written in place)."""

    def __init__(self, at, cell, obs, act, B, eps, mode):
        assert mode in ("masked", "inkernel")
        self.at, self.cell, self.mode, self.eps = at, cell, mode, float(eps)
        self.obs, self.act, self.B, self.T, dev = obs[:B], act[:B], B, obs.shape[1], obs.device
        self.ref = torch.zeros(B, obs.shape[-1], device=dev)
        self.g = torch.zeros(B, at.D_g, device=dev)
        self.gh = torch.zeros_like(self.g)
        self.trig = torch.zeros(B, dtype=torch.int32, device=dev)
        self.first = torch.zeros(1, dtype=torch.int32, device=dev)
        self.cnt = torch.zeros(1, dtype=torch.int64, device=dev)
        self.n_trig = torch.zeros((), dtype=torch.int64, device=dev)

    def reset(self):
        for x in (self.ref, self.g, self.gh, self.trig, self.first, self.cnt, self.n_trig):
            x.zero_()

    def step(self):
        t = self.cnt % self.T
        self.first.copy_(t == 0)
        o, a = self.obs.index_select(1, t), self.act.index_select(1, t)
        if self.mode == "inkernel":
            fused_observe(self.at, o, a, self.cell, event=dict(ref=self.ref, g=self.g, gh=self.gh, trig=self.trig,
                                                                first=self.first, eps=self.eps))
        else:
            out = fused_observe(self.at, o, a, self.cell)
            d = (o[:, 0] - self.ref).norm(dim=-1) / math.sqrt(o.shape[-1])
            trig = (d >= self.eps) | (self.first != 0)
            self.trig.copy_(trig)
            self.g.copy_(torch.where(trig[:, None], out["g"], self.g))
            self.gh.copy_(torch.where(trig[:, None], out["gh"], self.gh))
            self.ref.copy_(torch.where(trig[:, None], o[:, 0], self.ref))
        self.n_trig.add_(self.trig.sum())
        self.cnt.add_(1)

    @property
    def r(self):
        return int(self.n_trig) / max(int(self.cnt) * self.B, 1)


def load_bed(name, n=64, T=64, seed=0):
    if name == "swingup":
        from beds import swingup
        b = swingup.make_dataset(n, T, "lift", seed)
    else:
        from beds import make_bed
        b = make_bed(name, n, T, "lift", seed)
    return torch.from_numpy(b["obs"]), torch.from_numpy(b["act"])


def j(**kw):
    print(json.dumps(kw), flush=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beds", nargs="+", default=["s2", "so3", "swingup"])
    ap.add_argument("--B", type=int, nargs="+", default=[1, 64])
    a = ap.parse_args()
    j(clock_before=sm_clock(), gpu=torch.cuda.get_device_name())
    for bed in a.beds:
        obs, act = load_bed(bed)
        d = prev_deltas(obs)
        eps = {q: float(torch.quantile(d.flatten(), q)) for q in QS}
        for q in QS:
            j(kind="rate", bed=bed, q=q, eps=eps[q], r=trigger_rate(obs, eps[q]))
        obs, act = obs.cuda(), act.cuda()
        torch.manual_seed(0)
        at = ObserverAtlas(K=16, m=2, **{**CFG, "A": act.shape[-1]}).cuda().eval()
        cell = CellBackbone(at.backbone)
        for B in a.B:
            o_t, a_t = obs[:B, :1].contiguous(), act[:B, :1].contiguous()
            run = lambda: fused_observe(at, o_t, a_t, cell)
            eager, graph = time_us(run), time_us(graphed(run))
            for q in QS:
                s = EventStream(at, cell, obs, act, B, eps[q])
                ev = time_us(s.step)
                j(kind="stream", bed=bed, q=q, eps=eps[q], B=B, event_us=ev["median"], event_iqr=ev["iqr"],
                  r_stream=s.r, r_data=trigger_rate(obs[:B].cpu(), eps[q]), always_eager_us=eager["median"],
                  always_graph_us=graph["median"], speedup_vs_graph=graph["median"] / ev["median"],
                  speedup_vs_eager=eager["median"] / ev["median"])
    j(clock_after=sm_clock())


if __name__ == "__main__":
    main()

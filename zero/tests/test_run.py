"""50-step smoke of the harness on a contract-conforming toy arm (temp dir only, no run outputs)."""
import json

import numpy as np
import torch

from beds import make_bed
from common import n_params
from run import train, evaluate, budget_ratio, tune_readout


class ToyJEPA(torch.nn.Module):
    def __init__(self, D, A, d):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(D, 32), torch.nn.GELU(), torch.nn.Linear(32, d))
        self.pred = torch.nn.Linear(d + A, d)
        self.D, self.A, self.d = D, A, d

    def forward(self, obs, act):
        g = self.enc(obs)
        ghat = g[:, :-1] + self.pred(torch.cat([g[:, :-1], act[:, :-1]], -1))
        flat = g.reshape(-1, self.d)
        anti = torch.relu(1 - flat.std(0)).mean()                     # variance floor (anti-collapse)
        loss = ((ghat - g[:, 1:].detach()) ** 2).mean() + anti
        return dict(loss=loss, g=g, ghat=ghat, w=None)

    def rollout(self, obs0, act):
        z, out = self.enc(obs0), []
        for t in range(act.shape[1]):
            z = z + self.pred(torch.cat([z, act[:, t]], -1)); out.append(z)
        return torch.stack(out, 1)

    def active_flops(self):
        return self.D * 32 + 32 * self.d + (self.d + self.A) * self.d

    def n_params(self):
        return n_params(self)


def test_smoke_train_eval(tmp_path):
    bed = make_bed("s2", n_traj=16, T=32, obs="lift", seed=0)
    ev = make_bed("s2", n_traj=16, T=32, obs="lift", seed=1)
    arm = ToyJEPA(256, 3, 4)
    log = tmp_path / "smoke.jsonl"
    train(arm, bed, steps=50, seed=0, batch=8, window=8, log=log, log_every=10)
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 5 and all(np.isfinite(r["loss"]) for r in rows)
    tune = make_bed("s2", n_traj=16, T=32, obs="lift", seed=3)
    ro = tune_readout(arm, tune)
    assert ro["tune_seed"] == 3 and set(ro["mapper"]) == {"n_cover", "overlap"}
    m = evaluate(arm, ev, readout=ro)
    for k in ["pred_h1", "pred_h10", "probe_r2", "disc", "fold", "erank_g", "mapper", "nerve_n_samples",
              "flops", "params"]:
        assert k in m, k
    assert {"n_samples", "cover", "tune_seed", "betti_z2", "betti_q"} <= set(m["mapper"])
    assert m["mapper"]["n_samples"] == m["nerve_n_samples"] == 16 * 32 and m["mapper"]["tune_seed"] == 3
    assert m["erank_chart"] is None and "nerve_betti" not in m                 # no z / no router
    assert m["pred_h1"]["npersist"] < np.inf and m["flops"] == arm.active_flops()
    assert budget_ratio(arm, arm) == {"flops": 1.0, "params": 1.0}


def test_train_is_deterministic(tmp_path):
    bed = make_bed("r2", n_traj=8, T=16, obs="lift", seed=0)
    outs = []
    for i in range(2):
        torch.manual_seed(0)
        arm = ToyJEPA(256, 2, 2)
        torch.manual_seed(123 + i); np.random.seed(i)                  # global RNG state must not matter
        train(arm, bed, steps=5, seed=0, batch=4, window=4, log=tmp_path / f"d{i}.jsonl", log_every=5)
        outs.append(torch.cat([p.detach().flatten() for p in arm.parameters()]))
    assert torch.equal(outs[0], outs[1])


class ToyRouted(ToyJEPA):
    def __init__(self, *a):
        super().__init__(*a)
        self.router = torch.nn.Linear(self.d, 6)

    def forward(self, obs, act):
        out = super().forward(obs, act)
        ws = torch.softmax(self.router(out["g"]), -1)
        top, idx = ws.topk(2, -1)
        out["w"] = torch.zeros_like(ws).scatter(-1, idx, top / top.sum(-1, keepdim=True))
        out["w_soft"], out["zidx"] = ws, idx
        out["z"] = out["g"][..., None, :2].expand(-1, -1, 2, -1)
        return out


def test_nerve_readout_frozen(tmp_path):
    arm = ToyRouted(256, 3, 4)
    train(arm, make_bed("s2", 16, 32, "lift", 0), steps=5, seed=0, batch=4, window=8)
    ro = tune_readout(arm, make_bed("s2", 16, 32, "lift", 7))
    m = evaluate(arm, make_bed("s2", 16, 32, "lift", 1), readout=ro)
    assert {"tau", "tau_flag", "tau_soft"} <= set(ro)
    assert {"nerve_betti", "nerve_betti_flag", "nerve_betti_soft", "erank_chart", "seeds"} <= set(m)
    assert m["seeds"] == [1] and set(m["erank_chart"]) <= {str(k) for k in range(6)}            # JSON keys
    import pytest
    with pytest.raises(AssertionError):
        evaluate(arm, make_bed("s2", 16, 32, "lift", 7), readout=ro)       # eval on the tuning seed


def _sd(arm, opt_path):
    st = torch.load(opt_path, weights_only=False)
    return [p.detach().clone() for p in arm.state_dict().values()], st


def test_checkpoint_resume_bit_identical(tmp_path):
    """Wilson's API (signed off by Foreman): train(..., ckpt=path) resumes from the saved step; params, EMA
    targets, AdamW state, torch+numpy RNG (SIGReg slices, batch cursor) restored -> bit-identical."""
    from atlas.phase_sphere import PhaseSphere
    from run import save_state, load_state
    bed = make_bed("s2", n_traj=8, T=16, obs="lift", seed=0)
    mk = lambda s: (torch.manual_seed(s), PhaseSphere(D=256, A=3, K=4, m=2, N=4, d_c=2, D_g=6))[1]
    straight = train(mk(0), bed, steps=10, seed=0, batch=4, window=8)
    ck = tmp_path / "job.ckpt"
    train(mk(0), bed, steps=5, seed=0, batch=4, window=8, ckpt=ck, ckpt_every_s=0)
    assert ck.exists() and not list(tmp_path.glob("*.tmp"))                    # atomic write
    torch.manual_seed(999); np.random.seed(999)                                # fresh objects, polluted RNG
    resumed = train(mk(7), bed, steps=10, seed=0, batch=4, window=8, ckpt=ck, ckpt_every_s=0)
    a, b = straight.state_dict(), resumed.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)   # incl. tgt_* EMA
    st = torch.load(ck, weights_only=False)
    assert st["step"] == 10 and {"arm", "opt", "torch_rng", "np_rng"} <= set(st)


def test_probe_reports_memory_and_speed():
    from run import probe
    r = probe(ToyJEPA(256, 3, 4), make_bed("s2", 8, 16, "lift", 0), steps=3)
    assert set(r) >= {"vram_peak_bytes", "step_s"} and r["step_s"] > 0

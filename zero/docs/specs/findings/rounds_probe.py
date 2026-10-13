"""Rounds-vs-surprisal probe for the ctc arm (finding: ctc_rounds_sign.md). Not under PREREG.
Usage: python rounds_probe.py train ctc 0,1,2,3,4   then   python rounds_probe.py analyse"""
import sys, math, json, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
import torch, numpy as np
from scipy.stats import spearmanr
torch.set_num_threads(1)
from qflm import model as qm
from qflm.model import QFLM

txt = open(os.path.join(ROOT, "tests", "fixtures", "tiny_corpus.txt")).read()
chars = sorted(set(txt)); stoi = {c: i for i, c in enumerate(chars)}
ids = torch.tensor([stoi[c] for c in txt]); V = len(chars)
cut = int(0.9 * len(ids)); tr, va = ids[:cut], ids[cut:]
freq = torch.bincount(tr, minlength=V).float(); freq /= freq.sum()

def batch(src, B, T, g):
    i = torch.randint(0, len(src) - T, (B,), generator=g)
    return torch.stack([src[j:j + T] for j in i])

def train(arm="ctc", seed=0, steps=150, lr=3e-2, B=8, T=32, N=64, r=16, ckpts=()):
    torch.manual_seed(seed)
    m = QFLM(N, V, s_max=4, arm=arm, seed=seed, think_rank=r)
    opt = torch.optim.Adam(m.parameters(), lr=lr); g = torch.Generator().manual_seed(100 + seed)
    snaps = {}
    for s in range(steps + 1):
        if s in ckpts: snaps[s] = {k: v.clone() for k, v in m.state_dict().items()}
        if s == steps: break
        x = batch(tr, B, T, g); opt.zero_grad()
        loss = m(x) + (m.ctc_penalty() if arm == "ctc_hinge" else 0)
        loss.backward(); qm.clip(m); opt.step()
    return m, snaps

HELD = batch(va, 8, 32, torch.Generator().manual_seed(7))  # 256 heldout tokens

@torch.no_grad()
def probe(m, held=HELD, s_max=4, extra=8):
    """Per token: emitted surprisal, residual trajectory (s_max+extra F-steps), features."""
    m.s_max = s_max
    B, T = held.shape; ctx = m._ctx(); M, sqT, aq, _ = ctx
    psi = m.psi0().expand(B, m.N); rows = []
    c = m._ctc_scale()
    for t in range(T):
        marg, rounds, fin = m._emit(psi, ctx); info = dict(m._info)
        x = held[:, t]
        lp, psi_next, _ = fin(x)
        # free-running trajectory for residual ratios
        q0 = qm.butterfly(psi, m.theta, m.phi, aq); p = m._pq(q0, M); res = []; ps = [p]
        for s in range(s_max + extra):
            pn, _ = m._ctc_F(p, psi, ctx, c); res.append((pn - p).abs().sum(1)); p = pn; ps.append(p)
        res = torch.stack(res, 1)
        p0 = ps[0]; pstar = ps[-1]
        pm = lambda P: P[:, :V] / (1 - P[:, V:])
        s0 = -pm(p0).gather(1, x[:, None])[:, 0].log()
        sstar = -pm(pstar).gather(1, x[:, None])[:, 0].log()
        Hs = -(pm(pstar) * pm(pstar).clamp_min(1e-30).log()).sum(1)
        w = qm._w(psi); pr = 1 / (w ** 2).sum(1)
        for b in range(B):
            rows.append(dict(t=t, x=int(x[b]), surpr=float(-lp[b]), s0=float(s0[b]), sstar=float(sstar[b]),
                             H=float(Hs[b]), pmax=float(pm(pstar)[b].max()), pnull=float(pstar[b, V]),
                             rounds=int(info["rounds"][b]), halted=bool(info["halted"][b]),
                             d0=float(res[b, 0]), lam=float((res[b, 3] / res[b, 1]).sqrt()),  # mean contraction rounds 1->3
                             res=res[b].tolist(), pr=float(pr[b]), freq=float(freq[x[b]]),
                             prevx=int(held[b, t - 1]) if t else -1))
        psi = psi_next
    return rows

def sp(a, b):
    r = spearmanr(a, b)[0]; return None if r != r else round(float(r), 3)

def partial(a, b, z):
    from scipy.stats import rankdata
    A, Bv = rankdata(a), rankdata(b)
    Z = np.column_stack([np.ones(len(a))] + [rankdata(zz) for zz in z])
    ra = A - Z @ np.linalg.lstsq(Z, A, rcond=None)[0]; rb = Bv - Z @ np.linalg.lstsq(Z, Bv, rcond=None)[0]
    return round(float(np.corrcoef(ra, rb)[0, 1]), 3)

def uncensored_rounds(row, eps=qm.EPS_CTC):
    for i, r in enumerate(row["res"]):
        if r < eps: return i + 1
    return len(row["res"]) + 1

def summary(rows):
    g = lambda k: np.array([r[k] for r in rows], float)
    R = g("rounds"); S = g("surpr"); U = np.array([uncensored_rounds(r) for r in rows], float)
    from collections import Counter
    return dict(n=len(rows), rounds=dict(Counter(R.astype(int).tolist())), halted=float(g("halted").mean()),
        sp_rounds_surpr=sp(R, S), sp_uncens_surpr=sp(U, S), sp_rounds_s0=sp(R, g("s0")),
        sp_uncens_H=sp(U, g("H")), sp_uncens_pmax=sp(U, g("pmax")),
        sp_d0_uncens=sp(g("d0"), U), sp_lam_uncens=sp(g("lam"), U),
        sp_lam_surpr=sp(g("lam"), S), sp_d0_surpr=sp(g("d0"), S),
        partial_uncens_surpr_given_d0=partial(U, S, [g("d0")]),
        partial_uncens_surpr_given_d0_lam=partial(U, S, [g("d0"), g("lam")]),
        partial_uncens_surpr_given_t=partial(U, S, [g("t")]),
        partial_uncens_surpr_given_freq=partial(U, S, [g("freq")]),
        sp_freq_uncens=sp(g("freq"), U), sp_pr_uncens=sp(g("pr"), U), sp_pnull_uncens=sp(g("pnull"), U),
        mean_lam=float(g("lam").mean()))


def train_cmd(arm, seeds):
    for seed in seeds:
        m, _ = train(arm=arm, seed=seed, lr=3e-2)
        rows = probe(m)
        g = lambda k: np.array([r[k] for r in rows], float)
        R, S, S0, SS = g("rounds"), g("surpr"), g("s0"), g("sstar")
        print(json.dumps(dict(seed=seed, arm=arm, rho_raw=round(m.rho_raw(), 2), nll=round(S.mean(), 3),
            rounds=summary(rows)["rounds"], sp_R_surpr=sp(R, S), sp_R_s0=sp(R, S0), sp_R_d0=sp(R, g("d0")),
            sp_R_lam=sp(R, g("lam")), sp_R_H=sp(R, g("H")), sp_lam_surpr=sp(g("lam"), S),
            partial_R_surpr_d0=partial(R, S, [g("d0")]), partial_R_surpr_d0_lam=partial(R, S, [g("d0"), g("lam")]))), flush=True)
        json.dump(rows, open(f"rows_{arm}_{seed}.json", "w"))


def analyse_cmd():
    import glob
    for f in sorted(glob.glob("rows_ctc_[0-9].json")):
        rows = json.load(open(f)); g = lambda k: np.array([r[k] for r in rows], float)
        S, S0, H, d0 = g("surpr"), g("s0"), g("H"), g("d0")
        rel = np.array([next((i + 1 for i, r in enumerate(row["res"]) if r < 1e-3 * row["res"][0]), 13) for row in rows], float)
        print(f, json.dumps(dict(mean_s0=round(S0.mean(), 2), mean_surpr=round(S.mean(), 2), logV=round(math.log(V), 2),
            sp_d0_H=sp(d0, H), sp_d0_s0=sp(d0, S0), sp_relR_surpr=sp(rel, S))))


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    if sys.argv[1] == "train":
        train_cmd(sys.argv[2], [int(s) for s in sys.argv[3].split(",")])
    else:
        analyse_cmd()

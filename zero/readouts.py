"""Arm-agnostic topology readouts: any arm that follows the common.py contract is scored identically.
Orientation holonomy needs (w, z, zidx); Berry/Chern need a complex psi. Missing keys -> None, never a crash."""
import numpy as np
import torch

import beds
import metrics

KEYS = ("g", "w", "w_soft", "z", "zidx", "psi")


def _np(x):
    return x.detach().cpu().numpy()


def encode_loops(arm, obs_loops, act=None, device="cpu", batch=256):
    """Run arm (eval, no grad) on obs [n, L, D]; zero actions unless act is given. Returns {key: ndarray} for
    the contract keys the arm actually returns (g, w, w_soft, z, zidx, psi)."""
    obs = np.asarray(obs_loops, np.float32)
    if act is None:
        A = getattr(arm, "A", None)
        if A is None:
            raise ValueError("arm has no attribute A: pass act explicitly")
        act = np.zeros(obs.shape[:2] + (A,), np.float32)
    arm.eval().to(device)
    parts = {}
    with torch.no_grad():
        for b in range(0, len(obs), batch):
            out = arm(torch.as_tensor(obs[b:b + batch], device=device),
                      torch.as_tensor(np.asarray(act[b:b + batch], np.float32), device=device))
            for k in KEYS:
                if out.get(k) is not None:
                    parts.setdefault(k, []).append(_np(out[k]))
    return {k: np.concatenate(v) for k, v in parts.items()}


def _acc(signs, truth):
    """Accuracy over ALL loops (None counts as wrong), and the None fraction."""
    n = len(signs)
    hit = sum(s is not None and s == t for s, t in zip(signs, truth))
    return (float(hit / n) if n else None), sum(s is None for s in signs) / max(n, 1)


def to7_score(signs, truth, cls):
    """Per-class accuracy (None = wrong) in order of first appearance, plus balanced mean and min."""
    per = {}
    for c in dict.fromkeys(cls):
        idx = [i for i, k in enumerate(cls) if k == c]
        per[c] = float(sum(signs[i] is not None and signs[i] == truth[i] for i in idx) / len(idx))
    return {**per, "balanced": float(sum(per.values()) / len(per)), "min": float(min(per.values()))}


def loop_orientation(arm, calib_bed, loops, device="cpu"):
    """Chart transitions are fitted on the arm's encoding of calib_bed, then each loop's holonomy is read from
    the arm's w along that loop. Returns dict(signs, acc, none_frac, n_transitions) or None if the arm lacks w/z/zidx."""
    cal = encode_loops(arm, calib_bed["obs"], calib_bed.get("act"), device)
    if not all(k in cal for k in ("w", "z", "zidx")):
        return None
    tr = metrics.chart_transitions(cal["w"], cal["z"], cal["zidx"])
    w = encode_loops(arm, loops["obs"], loops.get("act"), device)["w"]
    signs = [metrics.orientation_holonomy(tr, wl) for wl in w]
    acc, none_frac = _acc(signs, np.asarray(loops["sign"]).tolist())
    out = dict(signs=signs, acc=acc, none_frac=none_frac, n_transitions=len(tr))
    if "cls" in loops:
        out["score"] = to7_score(signs, np.asarray(loops["sign"]).tolist(), list(loops["cls"]))
    return out


def loop_berry(arm, loops, device="cpu"):
    """Sign of the Berry/orientation holonomy of psi along each loop (atlas.phase_sphere.sign_holonomy), or
    None if the arm returns no complex psi."""
    psi = encode_loops(arm, loops["obs"], loops.get("act"), device).get("psi")
    if psi is None or not np.iscomplexobj(psi):
        return None
    from atlas.phase_sphere import sign_holonomy
    signs = [int(s) for s in sign_holonomy(torch.as_tensor(psi, dtype=torch.complex128)).tolist()]
    acc, _ = _acc(signs, np.asarray(loops["sign"]).tolist()) if "sign" in loops else (None, None)
    out = dict(signs=signs, acc=acc)
    if "sign" in loops and "cls" in loops:
        out["score"] = to7_score(signs, np.asarray(loops["sign"]).tolist(), list(loops["cls"]))
    return out


def icosphere(level):
    # copied from tests/test_phase_sphere.py (outward-oriented triangles)
    t = (1 + 5 ** .5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
         (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = [np.array(v) / np.linalg.norm(v) for v in V]
    for _ in range(level):
        mid, F2 = {}, []
        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                v = V[a] + V[b]; V.append(v / np.linalg.norm(v)); mid[k] = len(V) - 1
            return mid[k]
        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    V, F = np.array(V), np.array(F)
    cen = V[F].mean(1)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = (nrm * cen).sum(1) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return V, torch.as_tensor(F)


def _real4(psi):
    return np.stack([psi[:, 0].real, psi[:, 0].imag, psi[:, 1].real, psi[:, 1].imag], -1)


def _chern_psi(arm, bed_name, states, tris, device):
    # vertices are not a trajectory: V independent length-1 sequences, so recurrent arms carry no cross-vertex history
    obs = beds.lift(bed_name, states).astype(np.float32)[:, None]
    psi = encode_loops(arm, obs, device=device).get("psi")
    if psi is None or not np.iscomplexobj(psi):
        return None
    from atlas.phase_sphere import chern_fhs
    return float(chern_fhs(torch.as_tensor(psi[:, 0], dtype=torch.complex128), tris))


def chern_of_arm(arm, bed_name, level=2, device="cpu", seed=0):
    """Chern number of the arm's psi over an icosphere of the S^2 bed. Returns dict(chern, chern_random_phase)
    (chern_random_phase only on 'berry': same states, random per-vertex U(1) phase = gauge sensitivity), or
    None ('N/A: no complex amplitude') when the arm returns no complex psi."""
    if bed_name not in ("s2", "berry", "berry0"):
        raise ValueError(f"chern_of_arm: S^2-based beds only, got {bed_name!r}")
    V, tris = icosphere(level)
    if bed_name == "berry":
        th, ph = np.arccos(np.clip(V[:, 2], -1, 1)), np.arctan2(V[:, 1], V[:, 0])
        psi = np.stack([np.cos(th / 2) + 0j, np.exp(1j * ph) * np.sin(th / 2)], -1)
        c = _chern_psi(arm, bed_name, _real4(psi), tris, device)
        if c is None:
            return None
        chi = np.exp(2j * np.pi * np.random.default_rng(seed).random(len(V)))[:, None]
        return dict(chern=c, chern_random_phase=_chern_psi(arm, bed_name, _real4(psi * chi), tris, device))
    c = _chern_psi(arm, bed_name, V, tris, device)
    return None if c is None else dict(chern=c, chern_random_phase=None)

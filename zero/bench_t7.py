"""T7 systems bench (PROMPT.md §8 T7, §4.4): inference µs per time step on the local GPU.

Arms, one observe() step on obs [B,1,D] (GRU backbone included in the atlas; streaming T=1):
  lazy      OBSERVER-ATLAS top-m, Chase's gather+einsum path (StackedMLP.forward)
  lazy_tri  same model, chart MLPs replaced by the fused per-row Triton kernel (no gathered-weight tensor)
  lazy_grp  same, sort-by-expert grouped kernel (tensor-core dots per expert segment); lazy_grp16 = bf16 weights
  lazy_auto16  grouped bf16 kernel for E (i=256), per-row fp32 for P/phi; lazy_cell = that + GRUCell backbone
  lazy_fused  route + E + P + phi + mix in ONE Triton launch (T=1 inference); lazy_fused_cell = + GRUCell
  dense_c / peer_c / densew_c_H  torch.compile'd baselines (fairness: best equivalent of the atlas kernels)
  gru       the atlas backbone alone (its share of the step)
  soft      atlas, all K charts evaluated, full-softmax mixture (T3 reference)
  dense     SingleJEPA, hidden width bisected to the atlas's active_flops for that (K, m)
  peer      PEERJEPA (n_experts = K), hidden bisected the same way
  route_*   router alone, flat keys O(K d) vs product keys O(sqrt(K) d)
  densew_H  SingleJEPA at fixed widths: the curve T7's "dense JEPA of equal quality" is read from
Each arm eager and CUDA-graph replayed (graph = launch overhead removed; static shapes, top-m is static).

Timing: CUDA events around each of 200 back-to-back steps after >= 0.5 s warmup, synchronize at the end,
median + IQR. Steady state, so a step costs max(CPU launch time, GPU time): launch overhead stays in.
A sync before every step was tried first and rejected: the idle GPU fell to P4 / 510 MHz and medians
became clock-state noise. Clocks are NOT locked (laptop GPU; `nvidia-smi -lgc` needs admin): SM clock,
utilisation and temperature sampled per batch size and printed. Forward only (inference), fp32, TF32 at torch defaults.
Prints JSON lines to stdout; writes no files.
"""
import argparse
import json
import math
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms.peer_jepa import PEERJEPA  # noqa: E402
from arms.single_jepa import SingleJEPA  # noqa: E402
from atlas.observer_atlas import ObserverAtlas, StackedMLP  # noqa: E402

CFG = dict(D=256, A=2, d_c=2, D_g=8, d_h=64, d_key=32, hidden=32)  # D=256 random-lift bed width


@triton.jit
def _chart_mlp(x, idx, W1, b1, W2, b2, out, M, sxn, sxm, I, H, O,
               BI: tl.constexpr, BH: tl.constexpr, BO: tl.constexpr):
    """One program per (row n, slot j): y = silu(x W1[e] + b1[e]) W2[e] + b2[e], e = idx[n, j]."""
    r = tl.program_id(0)
    n, j = r // M, r % M
    e = tl.load(idx + r).to(tl.int64)
    h = tl.arange(0, BH)
    hm = h < H
    acc = tl.zeros([BH], tl.float32)
    for i0 in range(0, I, BI):
        i = i0 + tl.arange(0, BI)
        im = i < I
        xv = tl.load(x + n * sxn + j * sxm + i, mask=im, other=0.0)
        w = tl.load(W1 + e * I * H + i[:, None] * H + h[None, :], mask=im[:, None] & hm[None, :], other=0.0)
        acc += tl.sum(xv[:, None] * w, 0)
    acc += tl.load(b1 + e * H + h, mask=hm, other=0.0)
    acc = acc * tl.sigmoid(acc)
    o = tl.arange(0, BO)
    om = o < O
    w2 = tl.load(W2 + e * H * O + h[:, None] * O + o[None, :], mask=hm[:, None] & om[None, :], other=0.0)
    y = tl.sum(acc[:, None] * w2, 0) + tl.load(b2 + e * O + o, mask=om, other=0.0)
    tl.store(out + r * O + o, y, mask=om)


def chart_mlp(self, x, idx):
    """Drop-in for StackedMLP.forward (fp32, inference): x [N,m,i] (stride-0 m allowed), idx [N,m]."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    N, m, I = x.shape
    K, _, H = self.W1.shape
    O = self.W2.shape[-1]
    out = torch.empty(N, m, O, device=x.device, dtype=torch.float32)
    _chart_mlp[(N * m,)](x, idx.contiguous(), self.W1, self.b1, self.W2, self.b2, out, m, x.stride(0),
                         x.stride(1), I, H, O, BI=min(64, triton.next_power_of_2(I)),
                         BH=triton.next_power_of_2(H), BO=triton.next_power_of_2(O))
    return out


@triton.jit
def _grouped(x, order, eid, W1, b1, W2, b2, out, R, M, sxn, sxm, I, H, O, BM: tl.constexpr, BI: tl.constexpr,
             BH: tl.constexpr, BO: tl.constexpr, PREC: tl.constexpr):
    """Rows sorted by expert; a program takes BM consecutive sorted rows and walks their expert segments.
    Per segment: tile dots on tensor cores, rows outside the segment masked. Many rows per expert (B*m >> K)
    -> near-GEMM; one row per expert (K >> B*m) -> one weight read per row, same bytes as the per-row kernel."""
    pid = tl.program_id(0)
    rr = pid * BM + tl.arange(0, BM)
    rm = rr < R
    row = tl.load(order + rr, mask=rm, other=0)
    er = tl.load(eid + rr, mask=rm, other=-1)
    n, j = row // M, row % M
    h, o = tl.arange(0, BH), tl.arange(0, BO)
    hm, om = h < H, o < O
    y = tl.zeros([BM, BO], tl.float32)
    cur = pid * BM
    end = tl.minimum(pid * BM + BM, R)
    while cur < end:
        e = tl.load(eid + cur)
        sel = er == e
        e = e.to(tl.int64)
        acc = tl.zeros([BM, BH], tl.float32)
        for i0 in range(0, I, BI):
            i = i0 + tl.arange(0, BI)
            im = i < I
            xt = tl.load(x + n[:, None] * sxn + j[:, None] * sxm + i[None, :], mask=sel[:, None] & im[None, :],
                         other=0.0)
            w = tl.load(W1 + e * I * H + i[:, None] * H + h[None, :], mask=im[:, None] & hm[None, :], other=0.0)
            acc = tl.dot(xt.to(w.dtype), w, acc, input_precision=PREC)
        acc += tl.load(b1 + e * H + h, mask=hm, other=0.0)[None, :]
        acc = acc * tl.sigmoid(acc)
        w2 = tl.load(W2 + e * H * O + h[:, None] * O + o[None, :], mask=hm[:, None] & om[None, :], other=0.0)
        ys = tl.dot(acc.to(w2.dtype), w2, input_precision=PREC) + tl.load(b2 + e * O + o, mask=om, other=0.0)[None, :]
        y = tl.where(sel[:, None], ys, y)
        cur += tl.sum(sel.to(tl.int32))
    tl.store(out + row[:, None] * O + o[None, :], y, mask=rm[:, None] & om[None, :])


def _weights(self, dtype):
    """W1, W2 in dtype; bf16 copies cached and re-cast only when an in-place update bumps _version."""
    if dtype == torch.float32:
        return self.W1, self.W2
    key = (dtype, self.W1._version, self.W2._version, self.W1.data_ptr())
    if getattr(self, "_wcache", (None,))[0] != key:
        self._wcache = (key, self.W1.detach().to(dtype), self.W2.detach().to(dtype))
    return self._wcache[1:]


def grouped_chart_mlp(self, x, idx, dtype=torch.float32, BM=16):
    """Drop-in for StackedMLP.forward (inference): sort-by-expert grouped kernel; fp32 out, fp32 accumulate."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    N, m, I = x.shape
    H, O = self.W1.shape[-1], self.W2.shape[-1]
    eid, order = idx.reshape(-1).sort()
    W1, W2 = _weights(self, dtype)
    out = torch.empty(N, m, O, device=x.device, dtype=torch.float32)
    _grouped[(triton.cdiv(N * m, BM),)](
        x, order, eid, W1, self.b1, W2, self.b2, out, N * m, m, x.stride(0), x.stride(1), I, H, O, BM=BM,
        BI=max(16, min(64, triton.next_power_of_2(I))), BH=max(16, triton.next_power_of_2(H)),
        BO=max(16, triton.next_power_of_2(O)), PREC="ieee" if dtype == torch.float32 else "tf32")
    return out


KERNELS = dict(row=lambda mod, x, idx, dtype: chart_mlp(mod, x, idx),
               grouped=lambda mod, x, idx, dtype: grouped_chart_mlp(mod, x, idx, dtype),
               # measured: grouped wins only on wide inputs (E: i=256); P/phi (i<=8) stay per-row
               auto=lambda mod, x, idx, dtype: (grouped_chart_mlp(mod, x, idx, dtype) if x.shape[-1] >= 64
                                                else chart_mlp(mod, x, idx)))


class CellBackbone(torch.nn.Module):
    """Streaming step of the atlas GRU as nn.GRUCell with the same weights: nn.GRU over T=1 from h0=0 equals
    GRUCell(x, 0); a deployed agent carries h_{t-1} instead, same cost. Inference design probe only."""

    def __init__(self, gru):
        super().__init__()
        self.cell = torch.nn.GRUCell(gru.input_size, gru.hidden_size).to(gru.weight_ih_l0.device)
        for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            getattr(self.cell, n).data.copy_(getattr(gru, n + "_l0"))

    def forward(self, obs):
        assert obs.shape[1] == 1, "streaming step: T=1"
        return self.cell(obs[:, 0])[:, None], None


class _Fused(torch.autograd.Function):
    """Forward: Triton kernel. Backward: recompute Chase's gather reference in fp32 and differentiate it
    (ponytail: backward costs a reference forward+backward; a fused backward kernel only if training speed
    on the lazy path ever matters). With bf16 the grads are those of the fp32 function (straight-through)."""

    @staticmethod
    def forward(ctx, x, idx, W1, b1, W2, b2, mod, kind, dtype):
        ctx.save_for_backward(x, idx, W1, b1, W2, b2)
        return KERNELS[kind](mod, x, idx, dtype)

    @staticmethod
    def backward(ctx, gy):
        x, idx, *ws = ctx.saved_tensors
        need = [ctx.needs_input_grad[0], *ctx.needs_input_grad[2:6]]
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(r) for t, r in zip([x, *ws], need)]
            ref = types.SimpleNamespace(W1=leaves[1], b1=leaves[2], W2=leaves[3], b2=leaves[4])
            y = StackedMLP.forward(ref, leaves[0], idx)
            got = iter(torch.autograd.grad(y, [t for t in leaves if t.requires_grad], gy))
        g = [next(got) if r else None for r in need]
        return g[0], None, *g[1:], None, None, None


def fused(mod, x, idx, kind="row", dtype=torch.float32):
    if torch.is_grad_enabled() and (x.requires_grad or mod.W1.requires_grad):
        return _Fused.apply(x, idx, mod.W1, mod.b1, mod.W2, mod.b2, mod, kind, dtype)
    return KERNELS[kind](mod, x, idx, dtype)


def use_triton(atlas, kind="row", dtype=torch.float32):
    for mod in (atlas.E, atlas.P, atlas.phi, atlas.E_tgt):
        mod.forward = types.MethodType(lambda self, x, idx: fused(self, x, idx, kind, dtype), mod)
    return atlas


@triton.jit
def _vm(x, W, I, O, BI: tl.constexpr, BO: tl.constexpr):
    """x [BI] (zeros past I) @ W[:I, :O] (row stride O) -> [BO]."""
    i, o = tl.arange(0, BI), tl.arange(0, BO)
    w = tl.load(W + i[:, None] * O + o[None, :], mask=(i[:, None] < I) & (o[None, :] < O), other=0.0)
    return tl.sum(x[:, None] * w, 0)


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _phi(x, e, F1, Fb1, F2, Fb2, DC, H, DG, BC: tl.constexpr, BH: tl.constexpr, BG: tl.constexpr):
    h, g = tl.arange(0, BH), tl.arange(0, BG)
    hid = _silu(_vm(x, F1 + e * DC * H, DC, H, BC, BH) + tl.load(Fb1 + e * H + h, mask=h < H, other=0.0))
    return _vm(hid, F2 + e * H * DG, H, DG, BH, BG) + tl.load(Fb2 + e * DG + g, mask=g < DG, other=0.0)


@triton.jit
def _chart(orow, av, e, E1, Eb1, E2, Eb2, P1, Pb1, P2, Pb2, F1, Fb1, F2, Fb2, D, A, H, DC, DG,
           BD: tl.constexpr, BA: tl.constexpr, BH: tl.constexpr, BC: tl.constexpr, BG: tl.constexpr):
    """Chart e on one row: z = E_e(o), zh = z + P_e([z, a]) -> phi_e(z), phi_e(zh)."""
    e = e.to(tl.int64)
    h, c = tl.arange(0, BH), tl.arange(0, BC)
    hm, cm = h < H, c < DC
    acc = tl.zeros([BH], tl.float32)
    for i0 in range(0, D, BD):
        i = i0 + tl.arange(0, BD)
        xv = tl.load(orow + i, mask=i < D, other=0.0)
        w = tl.load(E1 + e * D * H + i[:, None] * H + h[None, :], mask=(i[:, None] < D) & hm[None, :], other=0.0)
        acc += tl.sum(xv[:, None] * w, 0)
    hid = _silu(acc + tl.load(Eb1 + e * H + h, mask=hm, other=0.0))
    z = _vm(hid, E2 + e * H * DC, H, DC, BH, BC) + tl.load(Eb2 + e * DC + c, mask=cm, other=0.0)
    IP = DC + A
    hp = _vm(z, P1 + e * IP * H, DC, H, BC, BH) + _vm(av, P1 + e * IP * H + DC * H, A, H, BA, BH)
    hp = _silu(hp + tl.load(Pb1 + e * H + h, mask=hm, other=0.0))
    zh = z + _vm(hp, P2 + e * H * DC, H, DC, BH, BC) + tl.load(Pb2 + e * DC + c, mask=cm, other=0.0)
    return (_phi(z, e, F1, Fb1, F2, Fb2, DC, H, DG, BC, BH, BG),
            _phi(zh, e, F1, Fb1, F2, Fb2, DC, H, DG, BC, BH, BG))


@triton.jit
def _argmax_flat(S, fi, BIG: tl.constexpr):
    v = tl.max(tl.max(S, 1), 0)
    return v, tl.min(tl.min(tl.where(S == v, fi, BIG), 1), 0)


@triton.jit
def _step(c, o, a, Wq, keys, Wg, bg, E1, Eb1, E2, Eb2, P1, Pb1, P2, Pb2, F1, Fb1, F2, Fb2,
          g_out, gh_out, idx_out, w_out, gate_out, D, A, DH, QN, K, SIDE, H, DC, DG, scale,
          ref, eps, first, trig_out, EV: tl.constexpr,
          M: tl.constexpr, PK: tl.constexpr, BD: tl.constexpr, BA: tl.constexpr, BDH: tl.constexpr,
          BQ: tl.constexpr, BS: tl.constexpr, BH: tl.constexpr, BC: tl.constexpr, BG: tl.constexpr):
    """One program per batch row: query, gate, exact top-m (flat or product keys), softmax, m charts, mix."""
    n = tl.program_id(0).to(tl.int64)
    if EV:  # event mode: Delta = ||o - ref|| / sqrt(D); a held row (not first, Delta < eps) leaves g/gh/ref untouched
        dacc = tl.zeros([BD], tl.float32)
        for i0 in range(0, D, BD):
            i = i0 + tl.arange(0, BD)
            dv = tl.load(o + n * D + i, mask=i < D, other=0.0) - tl.load(ref + n * D + i, mask=i < D, other=0.0)
            dacc += dv * dv
        run = (tl.load(first) != 0) | (tl.sqrt(tl.sum(dacc, 0)) / tl.sqrt(D * 1.0) >= eps)
        tl.store(trig_out + n, run.to(tl.int32))
        if not run:
            return
    d, q = tl.arange(0, BDH), tl.arange(0, BQ)
    cv = tl.load(c + n * DH + d, mask=d < DH, other=0.0)
    wq = tl.load(Wq + q[:, None] * DH + d[None, :], mask=(q[:, None] < QN) & (d[None, :] < DH), other=0.0)
    qa = tl.sum(wq * cv[None, :], 1)                                   # [BQ] query (first half if PK)
    gate = tl.sigmoid(tl.sum(tl.load(Wg + d, mask=d < DH, other=0.0) * cv, 0) + tl.load(bg))
    r = tl.arange(0, BS)
    if PK:
        wq2 = tl.load(Wq + (QN + q[:, None]) * DH + d[None, :], mask=(q[:, None] < QN) & (d[None, :] < DH), other=0.0)
        qb = tl.sum(wq2 * cv[None, :], 1)
        km = (r[:, None] < SIDE) & (q[None, :] < QN)
        s1 = tl.sum(tl.load(keys + r[:, None] * QN + q[None, :], mask=km, other=0.0) * qa[None, :], 1)
        s2 = tl.sum(tl.load(keys + SIDE * QN + r[:, None] * QN + q[None, :], mask=km, other=0.0) * qb[None, :], 1)
        S = s1[:, None] + s2[None, :]
        fi = r[:, None] * SIDE + r[None, :]
        S = tl.where((r[:, None] < SIDE) & (r[None, :] < SIDE), S, float("-inf"))
    else:  # ponytail: flat keys in one block (atlas uses flat only below K=256); tile + running top-m if needed
        km = (r[:, None] < K) & (q[None, :] < QN)
        s = tl.sum(tl.load(keys + r[:, None] * QN + q[None, :], mask=km, other=0.0) * qa[None, :], 1)
        S = tl.where(r < K, s, float("-inf"))[:, None]
        fi = r[:, None]
    v0, i0 = _argmax_flat(S, fi, 2 ** 30)
    orow, av_i = o + n * D, tl.arange(0, BA)
    av = tl.load(a + n * A + av_i, mask=av_i < A, other=0.0)
    p0, ph0 = _chart(orow, av, i0, E1, Eb1, E2, Eb2, P1, Pb1, P2, Pb2, F1, Fb1, F2, Fb2, D, A, H, DC, DG,
                     BD, BA, BH, BC, BG)
    gg = tl.arange(0, BG)
    if M == 2:
        v1, i1 = _argmax_flat(tl.where(fi == i0, float("-inf"), S), fi, 2 ** 30)
        w1 = 1.0 / (1.0 + tl.exp((v0 - v1) * scale))
        w0 = 1.0 - w1
        p1, ph1 = _chart(orow, av, i1, E1, Eb1, E2, Eb2, P1, Pb1, P2, Pb2, F1, Fb1, F2, Fb2, D, A, H, DC, DG,
                         BD, BA, BH, BC, BG)
        g, gh = w0 * p0 + w1 * p1, w0 * ph0 + w1 * ph1
        tl.store(idx_out + n * 2 + 1, i1.to(tl.int64))
        tl.store(w_out + n * 2 + 1, w1)
    else:
        w0, g, gh = 1.0, p0, ph0
    tl.store(idx_out + n * M, i0.to(tl.int64))
    tl.store(w_out + n * M, w0)
    tl.store(gate_out + n, gate)
    tl.store(g_out + n * DG + gg, gate * g, mask=gg < DG)
    tl.store(gh_out + n * DG + gg, gate * gh, mask=gg < DG)
    if EV:
        for i0 in range(0, D, BD):
            i = i0 + tl.arange(0, BD)
            tl.store(ref + n * D + i, tl.load(o + n * D + i, mask=i < D, other=0.0), mask=i < D)


def fused_observe(at, obs, act, backbone=None, event=None):
    """ObserverAtlas.observe(mode='lazy') at T=1 with route + charts + mix in ONE kernel launch (inference).
    backbone: optional replacement for at.backbone (e.g. CellBackbone); the GRU stays outside the kernel.
    event: dict(ref [B,D], g, gh [B,D_g], trig [B] int32, first [1] int32, eps float) of persistent buffers: the
    kernel skips held rows and writes g/gh/ref in place for triggered rows only; out["g"], out["gh"] are those buffers."""
    assert obs.shape[1] == 1 and at.m <= 2
    B, dev = obs.shape[0], obs.device
    c = (backbone or at.backbone)(obs)[0].reshape(B, -1).contiguous()
    o, a = obs.reshape(B, -1).contiguous(), act.reshape(B, -1).contiguous()
    pk = at.pk
    QN = at.d_key // 2 if pk else at.d_key
    side = at.side if pk else at.K
    assert pk or at.K <= 256, "flat keys fused only up to K=256"
    out = dict(g=torch.empty(B, at.D_g, device=dev), gh=torch.empty(B, at.D_g, device=dev),
               idx=torch.empty(B, at.m, dtype=torch.long, device=dev), w=torch.empty(B, at.m, device=dev),
               gate=torch.empty(B, 1, device=dev))
    ev = event or dict(ref=c, g=out["g"], gh=out["gh"], trig=c, first=c, eps=0.0)
    if event:
        out["g"], out["gh"] = event["g"], event["gh"]
    p2 = triton.next_power_of_2
    _step[(B,)](c, o, a, at.W_q.weight, at.sub_keys if pk else at.keys, at.gate.weight, at.gate.bias,
                at.E.W1, at.E.b1, at.E.W2, at.E.b2, at.P.W1, at.P.b1, at.P.W2, at.P.b2,
                at.phi.W1, at.phi.b1, at.phi.W2, at.phi.b2, ev["g"], ev["gh"], out["idx"], out["w"], out["gate"],
                at.D, at.A, at.d_h, QN, at.K, side, at.hidden, at.d_c, at.D_g, 1 / math.sqrt(at.d_key),
                ev["ref"], ev["eps"], ev["first"], ev["trig"], EV=event is not None, M=at.m, PK=pk, BD=min(64, p2(at.D)), BA=max(2, p2(at.A)), BDH=p2(at.d_h), BQ=p2(QN),
                BS=p2(side), BH=p2(at.hidden), BC=max(2, p2(at.d_c)), BG=p2(at.D_g))
    return out


def match_flops(cls, target, **kw):
    """Arm of class cls (SingleJEPA signature) whose hidden width puts active_flops closest to target."""
    make = lambda h: cls(CFG["D"], CFG["A"], CFG["D_g"], hidden=h, **kw)
    lo, hi = 1, 8192
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if make(mid).active_flops() <= target else (lo, mid)
    return min((make(lo), make(hi)), key=lambda a: abs(a.active_flops() - target))


def time_us(fn, warmup=25, iters=200, warm_s=0.5):
    """warm_s of back-to-back steps first: the laptop GPU needs ~100s of ms to leave a low P-state, and 200
    small steps last only ~16 ms, so an iteration-count warmup alone left medians stuck at 360-800 us."""
    t0, n = time.perf_counter(), 0
    while n < warmup or time.perf_counter() - t0 < warm_s:
        fn()
        n += 1
        if n % 25 == 0:  # bound the queue so wall time tracks GPU time
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:  # back-to-back: per-step time = max(CPU launch, GPU exec); GPU stays at boost clock
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    q1, med, q3 = statistics.quantiles([s.elapsed_time(e) * 1e3 for s, e in ev], n=4)
    return dict(median=med, q1=q1, q3=q3, iqr=q3 - q1, n=iters)


def graphed(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def sm_clock():
    q = ["nvidia-smi", "--query-gpu=clocks.sm,utilization.gpu,temperature.gpu", "--format=csv,noheader"]
    return subprocess.run(q, capture_output=True, text=True).stdout.strip()


def bench(name, fn, **tags):
    for mode in ("eager", "graph"):
        try:
            r = time_us(fn if mode == "eager" else graphed(fn))
            r["clock"] = sm_clock()  # sampled right after timing: the P-state the median was taken at
        except torch.OutOfMemoryError:
            r = dict(error="oom")
        except RuntimeError as err:  # graph capture failure is a result, not a crash
            r = dict(error=str(err).splitlines()[0][:120])
        torch.cuda.empty_cache()
        print(json.dumps(dict(arm=name, mode=mode, **tags, **r)), flush=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[16, 64, 256, 1024, 4096])
    ap.add_argument("--m", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--B", type=int, nargs="+", default=[1, 64, 1024])
    ap.add_argument("--widths", type=int, nargs="*", default=[64, 128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--d_h", type=int, default=CFG["d_h"], help="backbone GRU width (design-note sweeps)")
    ap.add_argument("--arms", nargs="+", default=list(ARMS) + ["route", "densew"])
    ap.add_argument("--compile", action="store_true", help="also torch.compile the dense width sweep")
    a = ap.parse_args()
    cfg, dev = {**CFG, "d_h": a.d_h}, "cuda"
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                          clock_before=sm_clock(), cfg=cfg)), flush=True)
    for B in a.B:
        print(json.dumps(dict(B=B, clock=sm_clock())), flush=True)
        obs, act = torch.randn(B, 1, cfg["D"], device=dev), torch.randn(B, 1, cfg["A"], device=dev)
        for H in a.widths if "densew" in a.arms else []:
            dw = SingleJEPA(cfg["D"], cfg["A"], cfg["D_g"], hidden=H).to(dev).eval()
            bench(f"densew_{H}", lambda: dw.observe(obs, act), B=B, flops=dw.active_flops())
            if a.compile:
                dwc = torch.compile(dw.observe)
                bench(f"densew_c_{H}", lambda: dwc(obs, act), B=B, flops=dw.active_flops())
        for K in a.K:
            pk = K >= 256  # PROMPT §8 T3: product keys from K=256
            c = torch.randn(B, cfg["d_h"], device=dev)
            for kind in ("flat", "pk") if "route" in a.arms else ():
                r = ObserverAtlas(K=K, m=2, product_keys=kind == "pk", **cfg).to(dev).eval()
                bench(f"route_{kind}", lambda: r.route(c), B=B, K=K)
            for m in a.m:
                torch.manual_seed(0)
                at = ObserverAtlas(K=K, m=m, product_keys=pk, **cfg).to(dev).eval()
                F = at.active_flops()
                tags = dict(K=K, m=m, B=B, flops=F, d_h=cfg["d_h"])
                for name in a.arms:
                    if name in ARMS:
                        fn, extra = ARMS[name](at, obs, act, F, K, m, pk, cfg)
                        bench(name, fn, **tags, **extra)
                del at
    print(json.dumps(dict(clock_after=sm_clock())), flush=True)


def _patched(kind, dtype, cell=False):
    def make(at, obs, act, F, K, m, pk, cfg):
        tri = use_triton(ObserverAtlas(K=K, m=m, product_keys=pk, **cfg).to(at.W_q.weight.device).eval(), kind, dtype)
        tri.load_state_dict(at.state_dict())
        if cell:
            tri.backbone = CellBackbone(tri.backbone)
        return (lambda: tri.observe(obs, act, "lazy")), {}
    return make


def _cell_fused(at, obs, act):
    cell = CellBackbone(at.backbone)
    return lambda: fused_observe(at, obs, act, cell)


def _matched(cls, compiled=False):
    """Fairness: the *_c variants give dense/PEER torch.compile (inductor-fused pointwise + cuBLAS) on top of
    eager/graph, the equivalent of the atlas's custom kernels."""
    def make(at, obs, act, F, K, m, pk, cfg):
        extra = dict(n_experts=K, topk=min(8, math.isqrt(K))) if cls is PEERJEPA else {}
        arm = match_flops(cls, F, **extra).to(obs.device).eval()
        f = torch.compile(arm.observe) if compiled else arm.observe
        return (lambda: f(obs, act)), dict(arm_flops=arm.active_flops())
    return make


ARMS = dict(
    lazy=lambda at, obs, act, *_: ((lambda: at.observe(obs, act, "lazy")), {}),
    lazy_tri=_patched("row", torch.float32),
    lazy_grp=_patched("grouped", torch.float32),
    lazy_grp16=_patched("grouped", torch.bfloat16),
    lazy_auto16=_patched("auto", torch.bfloat16),
    lazy_cell=_patched("auto", torch.bfloat16, cell=True),
    lazy_fused=lambda at, obs, act, *_: ((lambda: fused_observe(at, obs, act)), {}),
    lazy_fused_cell=lambda at, obs, act, *_: (_cell_fused(at, obs, act), {}),
    soft=lambda at, obs, act, *_: ((lambda: at.observe(obs, act, "soft")), {}),
    gru=lambda at, obs, act, *_: ((lambda: at.backbone(obs)), {}),  # backbone alone: its share of a step
    dense=_matched(SingleJEPA),
    peer=_matched(PEERJEPA),
    dense_c=_matched(SingleJEPA, compiled=True),
    peer_c=_matched(PEERJEPA, compiled=True),
)


if __name__ == "__main__":
    main()

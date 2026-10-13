"""EraLM: the TT (a flat-state softmax network) on a chronal fluid of ridge eras.

Experts per token: [era_1..era_K, TT alone, cache]. The decider holds w_t, the TT's weight over them (the chronal
fluid density), persistent per stream across windows; the bank is shared and persistent across windows and batches
(E1: the gain is memory beyond the context window). Inside a window the bank is frozen, so the filter is one
log-space prefix scan (filter.window_filter); after the window each stream's segment is addressed and credited
(RidgeBank.credit). Answer = linear mixture sum_i v_t(i) p_i(y), so the L38 regret bound applies: never worse than
the best expert path by more than log(n / (1 - alpha + rho)) + T log(1/(1 - alpha)) + switch costs.

Contract (lm/cores.py): model(tokens [B, T]) -> dict(nll [B, T], tt_nll [B, T]); entry 0 is 0 and ignored.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .config import EraLMConfig
from .filter import Decider, window_filter
from .ridge import RidgeBank
from .tt import TT

N_TAIL = 2   # TT alone, cache


class EraLM(nn.Module):
    def __init__(self, cfg: EraLMConfig = None, vocab=None, d=None, **kw):
        super().__init__()
        if cfg is None:                                              # lm/cores.py external-core signature
            cfg = EraLMConfig(**({"vocab": vocab} if vocab else {}), **({"d": d} if d else {}), **kw)
        self.cfg = cfg
        self.tt = TT(cfg.vocab, cfg.d, cfg.layers, cfg.expand, cfg.conv, cfg.grad_ckpt)
        self.bank = RidgeBank(cfg.d + 1, cfg.d, cfg.n_max, cfg.ridge, merge_rule=cfg.merge_rule,
                              merge_delta=cfg.merge_delta)
        self.decider = Decider(cfg.decider, cfg.alpha, cfg.decider_p)
        self.online = True            # credit segments to the bank in train mode (and in eval when stream_online)
        self.stream_online = True     # prequential eval: the bank keeps learning from the stream it predicts
        self.streams = None
        self._feat = self.tt.features
        if cfg.compile:
            self._feat = torch.compile(self.tt.features, dynamic=False)
        self.monitor = {}

    # ---- sizes -------------------------------------------------------------------------------------------------
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def n_stats(self):
        return self.bank.n_stats()

    def active_flops_per_token(self, seq=256):
        """Forward FLOPs/token: TT + one head per alive era + bank statistics (phi phi^T, p E)."""
        c, d = self.cfg, self.cfg.d
        macs = self.tt.macs_per_token()
        if c.bank:
            k = max(self.bank.n_alive, 1)
            macs += k * ((d + 1) * d + d * c.vocab) + (d + 1) ** 2 + (d + 1) * d + c.vocab * d
        return 2 * macs

    # ---- streams -----------------------------------------------------------------------------------------------
    def reset_streams(self, B=None):
        """B: start B persistent streams (TT state, decider weights, cache, last era). None: stateless calls."""
        self.streams = None if B is None else {"B": B}

    def _stream_state(self, B, dev):
        st = self.streams
        if st is None:
            return None
        if "w" not in st:
            st.update(tt=None, w=None, prev=torch.full((B,), -1, dtype=torch.long, device=dev), ctx_ready=False)
        return st

    # ---- forward -----------------------------------------------------------------------------------------------
    def forward(self, tokens):
        """Harness contract: predict tokens[:, 1:] from tokens[:, :-1]. -> dict(nll [B, T], tt_nll [B, T])."""
        out = self.step(tokens[:, :-1], tokens[:, 1:])
        z = out["nll"].new_zeros(tokens.shape[0], 1)
        return {"nll": torch.cat([z, out["nll"]], 1), "tt_nll": torch.cat([z, out["tt_nll"]], 1)}

    def _autocast(self, dev):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.amp and dev.type == "cuda")

    def step(self, x, y):
        """x, y [B, T] (y = next tokens). In stream mode x continues the previous call's stream.
        -> dict(nll [B, T], tt_nll [B, T]) and self.monitor updated."""
        B, T = x.shape
        dev = x.device
        st = self._stream_state(B, dev)
        with self._autocast(dev):
            h, tt_state = self._feat(x, st["tt"] if st else None)
        if st is not None:
            st["tt"] = [tuple(s.detach() for s in layer) for layer in tt_state]
        h = h.float()
        E = self.tt.emb.weight.float()
        z = h @ E.T                                                  # TT logits [B, T, V]
        logp_tt = torch.log_softmax(z, -1)
        lt = logp_tt.gather(-1, y[..., None])[..., 0]                # [B, T]
        res = {"tt_nll": -lt}
        if not self.cfg.bank:
            res["nll"] = -lt
            return res
        c = self.cfg
        K = c.n_max
        alive = self.bank.alive
        phi = torch.cat([h, h.new_ones(B, T, 1)], -1)
        # eras: log p_k(y) = log softmax((h + phi W_k) E^T)[y]; no gradient into the bank's statistics
        le = lt.new_full((B, T, K), -1e4)
        W = self.bank.W.clone()                                      # the bank is credited before backward
        for k in self.bank.order().tolist():
            hk = h + phi @ W[k]
            zk = hk @ E.T
            le[..., k] = zk.gather(-1, y[..., None])[..., 0] - torch.logsumexp(zk, -1)
        # cache: Polya urn over the last c.cache tokens with the TT as base measure
        if c.cache > 0:
            C = c.cache
            ctx = st["ctx"] if st is not None and st.get("ctx_ready") else None
            if ctx is None:                                          # stream start: history is just x_0
                ctx = torch.cat([x.new_full((B, C - 1), -1), x[:, :1]], 1)
            S = torch.cat([ctx, y], 1)                               # ctx ends with x_0; y_t is predicted from S[:C+t]
            win = S.unfold(1, C, 1)[:, :T]                           # [B, T, C]: the last C tokens before y_t
            cnt = (win == y[..., None]).sum(-1).float()
            tot = (win >= 0).sum(-1).float()
            lc = torch.log(cnt + c.cache_a * lt.exp()) - torch.log(tot + c.cache_a)
            if st is not None:
                st["ctx"], st["ctx_ready"] = S[:, -C:].clone(), True
        else:
            lc = lt
        logl = torch.cat([le, lt[..., None], lc[..., None]], -1)     # [B, T, n]
        logM = self.bank.log_share(self.decider, N_TAIL).to(logl.dtype)
        n = K + N_TAIL
        al = torch.cat([alive, alive.new_ones(N_TAIL)]).clone()
        if st is not None and st["w"] is not None:
            w0 = st["w"]
            w0 = torch.where(al, w0, torch.full_like(w0, -1e30))     # eras folded since: mass already moved
            w0 = w0 - torch.logsumexp(w0, -1, keepdim=True)
        else:
            w0 = torch.where(al, torch.zeros(n, device=dev), torch.full((n,), -1e30, device=dev))
            w0 = (w0 - torch.logsumexp(w0, -1)).expand(B, n)
        with torch.no_grad():
            logv, w_end = window_filter(logl.detach().double(), logM.double(), w0.double(), self.decider.eta)
        logv = logv.float()
        if self.decider.name == "HARD":
            vm, km = logv.max(-1)
            single = vm.exp() >= self.decider.p
            mix = torch.where(single, logl.gather(-1, km[..., None])[..., 0], torch.logsumexp(logv + logl, -1))
        elif self.decider.name == "TEMP":
            u = logv / self.decider.p
            mix = torch.logsumexp(u - torch.logsumexp(u, -1, keepdim=True) + logl, -1)
            single = torch.zeros_like(mix, dtype=torch.bool)
        else:
            mix = torch.logsumexp(logv + logl, -1)
            single = torch.zeros_like(mix, dtype=torch.bool)
        res["nll"] = -mix
        v = logv.detach().exp()
        self.monitor = {"era_mass": v[..., :K].sum(-1).mean().item(), "tt_mass": v[..., K].mean().item(),
                        "cache_mass": v[..., K + 1].mean().item(), "single_frac": single.float().mean().item(),
                        "era_usage": v[..., :K].mean((0, 1)).tolist(), "n_eras": self.bank.n_alive}
        eras = torch.tensor(self.monitor["era_usage"])
        tot = float(eras.sum())
        self.monitor["collapse"] = bool(self.bank.n_alive >= 2 and tot > 0 and float(eras.max()) / tot > 0.9)
        if st is not None:
            st["w"] = w_end.float()
        if self.online and (self.training or (st is not None and self.stream_online)):
            self._credit(h.detach(), phi.detach(), logp_tt.detach(), y, E.detach(), st, B, dev)
        return res

    @torch.no_grad()
    def _credit(self, h, phi, logp_tt, y, E, st, B, dev):
        """Each stream's window is one segment: Bohning residual targets rho, additive statistics, address."""
        G = E.T.double() @ E.double()
        G += 1e-6 * torch.trace(G) / G.shape[0] * torch.eye(G.shape[0], dtype=G.dtype, device=dev)
        pE = logp_tt.exp() @ E                                       # [B, T, d]
        r = (E[y] - pE).double()                                    # (e_y - p) E
        rho = 2 * torch.linalg.solve(G, r.reshape(-1, r.shape[-1]).T).T.reshape(r.shape)
        ph = phi.double()
        A = ph.transpose(1, 2) @ ph
        Bs = ph.transpose(1, 2) @ rho
        cs = (rho * rho).sum((1, 2))
        ns = torch.full((B,), float(y.shape[1]), dtype=torch.float64, device=dev)
        prev = st["prev"] if st is not None else torch.full((B,), -1, dtype=torch.long, device=dev)
        new = self.bank.credit(A, Bs, cs, ns, prev)
        if st is not None:
            st["prev"] = new

    # ---- checkpoints -------------------------------------------------------------------------------------------
    def save(self, path, opt=None, extra=None):
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({"cfg": self.cfg.to_dict(), "model": self.state_dict(), "opt": opt.state_dict() if opt else None,
                    "extra": extra or {}}, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        m = cls(EraLMConfig(**ck["cfg"])).to(device)
        m.load_state_dict(ck["model"])
        return m, ck

    def configure_optimizer(self, lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95)):
        decay = [p for n, p in self.tt.named_parameters() if p.dim() >= 2]
        rest = [p for n, p in self.tt.named_parameters() if p.dim() < 2]
        fused = next(self.parameters()).is_cuda
        return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}],
                                 lr=lr, betas=betas, fused=fused)

    # ---- generation --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt, n_new=128, temperature=1.0, top_k=None, generator=None):
        """prompt [B, P] long -> [B, P + n_new]. Samples from the decider's mixture; with HARD routing a token whose
        top expert holds >= thr of the weight is drawn from that expert alone (one head computed)."""
        was = self.training
        self.eval()
        online = self.online
        self.online = False
        B = prompt.shape[0]
        self.reset_streams(B)
        out = prompt
        if prompt.shape[1] > 1:
            self.step(prompt[:, :-1], prompt[:, 1:])
        last = prompt[:, -1:]
        for _ in range(n_new):
            p = self._next_dist(last)
            if temperature != 1.0:
                p = torch.softmax(torch.log(p.clamp_min(1e-30)) / temperature, -1)
            if top_k:
                v, _ = p.topk(top_k, -1)
                p = torch.where(p < v[:, -1:], torch.zeros_like(p), p)
            nxt = torch.multinomial(p / p.sum(-1, keepdim=True), 1, generator=generator)
            self.step(last, nxt)                                     # advances TT state, weights and cache
            out = torch.cat([out, nxt], 1)
            last = nxt
        self.online = online
        self.train(was)
        self.reset_streams(None)
        return out

    def _next_dist(self, last):
        """Mixture distribution of the token after `last` [B, 1], without advancing the streams."""
        st = self.streams
        dev = last.device
        saved = {k: v for k, v in st.items()}
        with self._autocast(dev):
            h, _ = self._feat(last, st.get("tt"))
        h = h.float()[:, -1]
        E = self.tt.emb.weight.float()
        p_tt = torch.softmax(h @ E.T, -1)
        if not self.cfg.bank:
            return p_tt
        K = self.cfg.n_max
        al = torch.cat([self.bank.alive, self.bank.alive.new_ones(N_TAIL)])
        w = st.get("w")
        n = K + N_TAIL
        if w is None:
            w = torch.where(al, torch.zeros(n, device=dev), torch.full((n,), -1e30, device=dev)).expand(last.shape[0], n)
        logM = self.bank.log_share(self.decider, N_TAIL).float()
        v = torch.logsumexp(w[:, :, None] + logM[None], 1).softmax(-1)    # predictive weights for the next token
        phi = torch.cat([h, h.new_ones(h.shape[0], 1)], -1)
        top_v, top_k = v.max(-1)
        hard = self.decider.name == "HARD"
        out = torch.zeros_like(p_tt)
        for b in range(last.shape[0]):
            ks = [int(top_k[b])] if hard and float(top_v[b]) >= self.decider.p else al.nonzero().flatten().tolist()
            vb = v[b, ks] / v[b, ks].sum()
            for wk, k in zip(vb.tolist(), ks):
                if k < K:
                    pk = torch.softmax((h[b] + phi[b] @ self.bank.W[k]) @ E.T, -1)
                elif k == K:
                    pk = p_tt[b]
                else:
                    C = self.cfg.cache
                    ctx = st["ctx"][b] if st.get("ctx_ready") else last[b]
                    ctx = ctx[ctx >= 0][-C:]
                    cnt = torch.bincount(ctx, minlength=self.cfg.vocab).float()
                    pk = (cnt + self.cfg.cache_a * p_tt[b]) / (ctx.numel() + self.cfg.cache_a)
                out[b] += wk * pk
        self.streams.clear()
        self.streams.update(saved)
        return out

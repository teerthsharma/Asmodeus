"""T4b graph-protocol adapters: every learned arm behind the controls_t4b API
  fit(ctx [M,2], nxt [M,2]) -> self;  embed(xy [C,2]) -> [C,d];  predict(ctx [C,2]) -> (heads [C,K,d], pi [C,K])
Edges: beds/ogbench_bed.decode_edges(rule='centred' PRIMARY | 'paper' SECONDARY) (L-GIVEN: arms never import beds).

Training mirrors arms/mop_jepa.MoPGraph (the MoP adapter IS MoPGraph): pairs (ctx, nxt) as T=2 sequences with
A=0 (action-marginalised, v3 S10), inputs standardised by fit-set mean/std, Adam, uniform minibatches, seeded.
Successor proposals per arm:
  single           one head: tgt-space prediction z + P(z); embed = EMA target encoder
  mop              K heads (MoPGraph); embed = EMA target encoder
  atlas            top-m charts: phi_k(zhat_k) for k in S(c), pi = observer weights; embed = glued g
  lazy_chart_ae    top-m charts: D_k(zhat_k), pi = renormalised membership; embed = glued g
  codebook_latent  MoP-JEPA whose heads and router see zeros: K learned latent codes, same for every context
                   (v1's input-agnostic codebook control, in latent space; encoder trained by SIGReg only)
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from arms.lazy_chart_ae import LazyChartAEJEPA
from arms.mop_jepa import MoPGraph, MoPJEPA
from arms.single_jepa import SingleJEPA
from atlas.observer_atlas import ObserverAtlas

# v3 S10: K=8, 30k updates, 512-d latents. Atlas/chart-AE widths are PREREG choices (L-BR matching), not the paper's.
FULL = dict(
    single=dict(d=512, hidden=128),
    mop=dict(K=8, d=512),
    atlas=dict(K=8, m=2, d_c=2, D_g=64, d_h=64, d_key=32, hidden=32),
    lazy_chart_ae=dict(K=8, m=2, d_c=2, D_g=64, hidden=128, chart_hidden=32),
    codebook_latent=dict(K=8, d=512),
)
TINY = dict(
    single=dict(d=4, hidden=8),
    mop=dict(K=3, d=4, hidden=8, head_hidden=8, router_hidden=8),
    atlas=dict(K=4, m=2, d_c=2, D_g=6, d_h=8, d_key=8, hidden=8),
    lazy_chart_ae=dict(K=4, m=2, d_c=2, D_g=6, hidden=8, chart_hidden=8),
    codebook_latent=dict(K=3, d=4, hidden=8, head_hidden=8, router_hidden=8),
)


class _Graph:
    def __init__(self, steps=30000, batch=256, lr=1e-3, seed=0, **kw):
        self.steps, self.batch, self.lr, self.seed, self.kw = steps, batch, lr, seed, kw

    def _x(self, xy):
        return torch.as_tensor((np.asarray(xy, np.float64) - self.mu) / self.sd, dtype=torch.float32)

    def fit(self, ctx, nxt):
        pts = np.concatenate([ctx, nxt]).astype(np.float64)
        self.mu, self.sd = pts.mean(0), pts.std(0) + 1e-8
        torch.manual_seed(self.seed)
        self.model = self.build().train()
        obs = torch.stack([self._x(ctx), self._x(nxt)], 1)          # [M, 2, 2]; misaligned pairs raise
        act = obs.new_zeros(len(obs), 2, 0)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.steps):
            i = torch.randint(0, len(obs), (min(self.batch, len(obs)),), generator=g)
            opt.zero_grad()
            self.model(obs[i], act[i])["loss"].backward()
            opt.step()
        self.model.eval()
        return self

    def _one(self, xy):                                              # [C,2] -> obs [C,1,2], act [C,1,0]
        x = self._x(xy)[:, None]
        return x, x.new_zeros(len(x), 1, 0)


class SingleGraph(_Graph):
    def build(self):
        return SingleJEPA(D=2, A=0, **self.kw)

    @torch.no_grad()
    def embed(self, xy):
        return self.model.tgt_enc(self._x(xy)).numpy()

    @torch.no_grad()
    def predict(self, ctx):
        x = self._x(ctx)
        h = self.model.step(self.model.enc(x), x.new_zeros(len(x), 0))[:, None]
        return h.numpy(), np.ones((len(x), 1), np.float32)


class AtlasGraph(_Graph):
    def build(self):
        return ObserverAtlas(D=2, A=0, **self.kw)

    @torch.no_grad()
    def embed(self, xy):
        return self.model.observe(*self._one(xy))["g"].numpy()

    @torch.no_grad()
    def predict(self, ctx):
        r = self.model.observe(*self._one(ctx))
        return self.model.phi(r["zh"], r["idx"]).numpy(), r["w"].numpy()


class LazyChartGraph(_Graph):
    def build(self):
        return LazyChartAEJEPA(D=2, A=0, **self.kw)

    @torch.no_grad()
    def embed(self, xy):
        return self.model._parts(*self._one(xy))[6][:, 0].numpy()

    @torch.no_grad()
    def predict(self, ctx):
        m = self.model
        _, _, idx, w, _, zh, _, _ = m._parts(*self._one(ctx))
        return m._one(m.Dc, zh[:, 0], idx[:, 0]).numpy(), w[:, 0].numpy()


class _Blind(nn.Module):
    """Feeds zeros to the wrapped module: its output cannot depend on the context."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(torch.zeros_like(x))


class CodebookGraph(_Graph):
    def build(self):
        m = MoPJEPA(D=2, A=0, **self.kw)
        m.heads, m.router = _Blind(m.heads), _Blind(m.router)
        return m

    @torch.no_grad()
    def embed(self, xy):
        return F.normalize(self.model.tgt_enc(self._x(xy)), dim=-1).numpy()

    @torch.no_grad()
    def predict(self, ctx):
        z = F.normalize(self.model.enc(self._x(ctx)), dim=-1)
        return F.normalize(self.model.heads(z), dim=-1).numpy(), self.model.router(z).softmax(-1).numpy()


CLASSES = dict(single=SingleGraph, mop=MoPGraph, atlas=AtlasGraph, lazy_chart_ae=LazyChartGraph,
               codebook_latent=CodebookGraph)


def make(kind, seed=0, tiny=False, **train):
    """train: steps / batch / lr overrides. tiny=True: test-size widths (TINY), else FULL."""
    return CLASSES[kind](seed=seed, **train, **(TINY if tiny else FULL)[kind])

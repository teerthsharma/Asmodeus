"""CUDA-graphed QFLM serving step. The emission rule (rounds, forced emission below TAU, marginal q, mixture
post-state) is QFLM._emit, shared with training/forward/generate; this file adds only the device-side Gumbel-max
sample from q, the persistent buffers and the CUDA graph. No host sync, no data-dependent shapes.

Noise: external_noise=False draws u~U(0,1) in-graph from the default CUDA generator (graph-safe philox);
external_noise=True leaves `noise` [B,V] to the caller (for eager-vs-graph equality tests)."""
import contextlib

import torch

import qflm.model as qm
from qflm.model import TAU  # noqa: F401  (re-exported; the rule itself lives in QFLM._emit)


def _renorm_nosync(psi):
    """qm.renorm_psi minus its host-side zero-row check (.nonzero().tolist() cannot be captured)."""
    psi = psi / psi.abs().amax(-1, keepdim=True)
    return psi / torch.linalg.vector_norm(psi, dim=-1, keepdim=True)


@contextlib.contextmanager
def _graph_safe():
    orig, qm.renorm_psi = qm.renorm_psi, _renorm_nosync
    try:
        yield
    finally:
        qm.renorm_psi = orig


class GraphedQFLM:
    def __init__(self, model, B, temperature=1.0, external_noise=False):
        self.model, self.B, self.T, self.ext = model, B, temperature, external_noise
        self.V, self.S = model.V, model.s_max + 1
        dev = model.logits.device
        self.noise = torch.rand(B, self.V, device=dev)
        self.psi_buf = torch.zeros(B, model.N, dtype=torch.complex64, device=dev)
        self.tok = torch.zeros(B, dtype=torch.long, device=dev)
        self.q = torch.zeros(B, self.V, device=dev)  # token marginal of the last step
        self.st = torch.zeros(B, dtype=torch.long, device=dev)  # round at which each row was (forced to) emit
        self.force_tok = None  # set to a [B] long tensor for teacher forcing (eager only)
        self.graph = None
        self.reset()

    @torch.no_grad()
    def reset(self):
        self.psi_buf.copy_(self.model.psi0().detach().expand_as(self.psi_buf))

    @torch.no_grad()
    def step_eager(self):
        marg, rounds, fin = self.model._emit(self.psi_buf)
        if self.force_tok is not None:
            x = self.force_tok
        else:
            if not self.ext:
                self.noise.uniform_()
            u = self.noise.clamp(1e-20, 1 - 1e-7)
            x = (marg.clamp_min(1e-30).log() / self.T - torch.log(-torch.log(u))).argmax(1)
        self.q.copy_(marg)
        self.st.copy_(rounds)
        self.psi_buf.copy_(fin(x)[1])
        self.tok.copy_(x)

    def capture(self):
        saved = self.psi_buf.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), _graph_safe():
            for _ in range(3):
                self.step_eager()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with _graph_safe(), torch.cuda.graph(self.graph):
            self.step_eager()
        self.psi_buf.copy_(saved)

    def replay(self):
        if self.graph is None:
            self.capture()
        self.graph.replay()

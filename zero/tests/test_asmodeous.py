"""ASMODEOUS (ASMODEOUS.md "Contract and invariants"): RED-first contract tests for atlas/asmodeous.py::Asmodeous.
CPU only, seeded, small sizes. Ground truth is recomputed in the test from the model's own psi and V (the POVM
algebra is exact, so no reference implementation is needed). No bed structure is given to the model (L-GIVEN)."""
import inspect
import re
import sys
import tokenize
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import beds  # noqa: E402
import readouts  # noqa: E402
from atlas.asmodeous import Asmodeous  # noqa: E402

torch.set_num_threads(4)

ZERO = Path(__file__).resolve().parents[1]
D, A, K, m, N, D_g, B, T = 16, 2, 8, 2, 8, 8, 3, 6
MODES = ("loop", "one_pass", "amortised")


def make(seed=0, **kw):
    cfg = dict(D=D, A=A, K=K, m=m, N=N, d_c=2, D_g=D_g, hidden=16)
    cfg.update(kw)
    torch.manual_seed(seed)
    return Asmodeous(**cfg)


def data(seed=1, Dd=D, Aa=A, b=B, t=T):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, Dd, generator=g), torch.randn(b, t, Aa, generator=g)


def run(model, obs, act, mode="loop", seed=7):
    torch.manual_seed(seed)  # SIGReg / Haar slices and any internal starts are seeded
    return model(obs, act, mode=mode)


def povm_probs(model, psi):
    """All K+1 outcome probabilities |<v_k, psi>|^2 recomputed from V() and the exposed psi."""
    with torch.no_grad():
        return torch.einsum("kn,btn->btk", model.V(), psi).abs().pow(2)


# ---- 1. contract: keys, shapes, dtypes, in all three modes ---------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_contract_keys_shapes_dtypes(mode):
    model = make().eval()
    obs, act = data()
    out = run(model, obs, act, mode)
    for k in ("loss", "g", "ghat", "w", "gate", "psi", "ctc_iters", "ctc_residual", "terms"):
        assert k in out, k
    assert out["loss"].dim() == 0 and out["loss"].dtype == torch.float32 and torch.isfinite(out["loss"])
    assert out["g"].shape == (B, T, D_g) and out["g"].dtype == torch.float32
    assert out["ghat"].shape == (B, T - 1, D_g) and out["ghat"].dtype == torch.float32
    assert out["w"].shape == (B, T, K) and out["w"].dtype == torch.float32
    assert out["gate"].shape == (B, T, 1) and out["gate"].dtype == torch.float32
    assert ((out["gate"] >= 0) & (out["gate"] <= 1)).all()
    assert out["psi"].shape == (B, T, N) and out["psi"].dtype == torch.complex64
    assert torch.allclose(out["psi"].abs().pow(2).sum(-1), torch.ones(B, T), atol=1e-5)
    assert out["ctc_residual"].shape == (B, T) and torch.isfinite(out["ctc_residual"]).all()
    assert isinstance(out["ctc_iters"], int) or int(out["ctc_iters"]) == out["ctc_iters"]
    assert isinstance(out["terms"], dict) and len(out["terms"]) > 0
    # exact POVM weights on the m read charts, zeros elsewhere, NOT renormalised
    assert (out["w"] >= 0).all()
    assert ((out["w"] > 0).sum(-1) <= m).all() and (out["w"] > 0).any(-1).all()
    assert (out["w"].sum(-1) <= 1 + 1e-5).all()
    if mode == "loop":
        assert 1 <= int(out["ctc_iters"]) <= model.loop_iters
    else:
        assert int(out["ctc_iters"]) == 0  # no loop is run in one_pass / amortised


def test_contract_n_params_excludes_target_and_flops_int():
    model = make()
    named = dict(model.named_parameters())
    tgt = [n for n in named if n.startswith("tgt_")]
    assert tgt, "EMA target encoder parameters must be named tgt_*"
    assert model.n_params() == sum(p.numel() for n, p in named.items() if not n.startswith("tgt_")) > 0
    assert isinstance(model.active_flops(), int) and isinstance(model.n_params(), int)
    assert model.ema == 0.996 and model.loop_iters == 8 and model.loop_eps == 0.1 and model.loop_tol == 1e-6
    assert model.A == A  # readouts.encode_loops builds zero actions from arm.A


# ---- 2. decider exactness -------------------------------------------------------------------------
def test_V_is_an_isometry():
    for seed in range(3):
        V = make(seed=seed).V()
        assert V.shape == (K + 1, N) and V.dtype == torch.complex64
        assert torch.allclose(V.conj().T @ V, torch.eye(N, dtype=torch.complex64), atol=1e-5)


def test_V_stays_isometric_after_optimiser_steps():
    model = make()
    opt = torch.optim.Adam([p for n, p in model.named_parameters() if not n.startswith("tgt_")], lr=1e-2)
    obs, act = data()
    for _ in range(5):
        opt.zero_grad()
        run(model, obs, act)["loss"].backward()
        opt.step()
    V = model.V()
    assert torch.allclose(V.conj().T @ V, torch.eye(N, dtype=torch.complex64), atol=1e-5)


@pytest.mark.parametrize("mode", ("one_pass", "loop"))
def test_povm_probabilities_complete_and_w_exact(mode):
    model = make().eval()
    obs, act = data()
    out = run(model, obs, act, mode)
    p = povm_probs(model, out["psi"])  # [B, T, K+1], last = null effect
    assert p.shape == (B, T, K + 1)
    assert torch.allclose(p.sum(-1), torch.ones(B, T), atol=1e-5)  # sum over K read+unread plus p_null
    assert torch.allclose(p[..., :K].sum(-1) + p[..., K], torch.ones(B, T), atol=1e-5)
    read = out["w"] > 0
    assert torch.allclose(out["w"][read], p[..., :K][read], atol=1e-5)  # exact, not renormalised
    assert (out["w"][~read] == 0).all()
    assert torch.allclose(out["gate"][..., 0], 1 - p[..., K], atol=1e-5)
    scatter = lambda idx: torch.zeros(B, T, K, dtype=torch.bool).scatter_(-1, idx, True)
    if mode == "one_pass":  # laziness in K: read set is the top-m of the Born probs (no softmax)
        assert torch.equal(scatter(p[..., :K].topk(m, -1).indices), read)
    else:  # the loop may re-select inside the top-M candidates: read set = top-m of w* among them
        cand, wstar = out["cand"], out["wstar"]
        assert cand.shape == wstar.shape == (B, T, model.M)
        assert torch.equal(cand, p[..., :K].topk(model.M, -1).indices)
        top = wstar.topk(m, -1).indices
        assert torch.equal(scatter(cand.gather(-1, top)), read)


def test_loop_can_reselect_charts_one_pass_would_not_read():
    model = make().eval()
    obs, act = data(b=8, t=25)  # 200 (b, t) instances, fixed seed
    lo, op = run(model, obs, act, "loop"), run(model, obs, act, "one_pass")
    assert (lo["w"] > 0).ne(op["w"] > 0).any(), "loop read set never differs from top-m of p"
    # whatever the loop reads is still inside the candidate set
    assert torch.equal(lo["cand"].gather(-1, lo["wstar"].topk(m, -1).indices).sort(-1).values,
                       lo["zidx"].sort(-1).values)


# ---- 3. lazy == dense --------------------------------------------------------------------------
def test_lazy_matches_dense_on_read_charts():
    lazy = make(dense=False).eval()
    dense = make(dense=True).eval()
    dense.load_state_dict(lazy.state_dict())
    obs, act = data()
    a, b = run(lazy, obs, act), run(dense, obs, act)
    for k in ("g", "ghat", "w"):
        assert torch.allclose(a[k], b[k], atol=1e-5), k


# ---- 4. loop contraction --------------------------------------------------------------------------
def test_loop_contracts_and_is_deterministic():
    # loop_eps=0.3, not 0.1: damping alone contracts at (1 - eps)^iters, and 0.9^50 ~ 5e-3 cannot reach the 1e-4
    # residual bar, while 0.7^50 ~ 1.8e-8 can.
    model = make(loop_iters=50, loop_eps=0.3).eval()
    obs, act = data(b=8, t=16)
    out = run(model, obs, act, seed=7)
    res = out["ctc_residual"]
    assert res.shape == (8, 16)
    assert float((res < 1e-4).float().mean()) >= 0.99, float((res < 1e-4).float().mean())
    # same seed: bit-identical. Different seed (any random internal start): w* agrees to the contraction
    # bound |x - x*| <= res / (1 - L) <= 1e-4 / 0.3 = 3.4e-4 (<= the 2e-3 atol).
    again = run(model, obs, act, seed=7)
    assert torch.equal(out["w"], again["w"])
    other = run(model, obs, act, seed=12345)
    assert torch.allclose(out["w"], other["w"], atol=2e-3)
    # NOTE: the API cannot inject two chosen starts, so "unique fixed point from two starts" is tested only
    # through the internal-seed change above plus the residual bound.


def phase_patch(model, seed=99):
    """Wrap model.encode so psi -> e^{i phi} psi with a fixed phi per (b, t). Returns phi [B, T]."""
    orig, box = model.encode, {}

    def phased(o):
        psi = orig(o)
        gen = torch.Generator().manual_seed(seed)
        phi = 2 * torch.pi * torch.rand(psi.shape[:-1], generator=gen)
        box["phi"] = phi
        return psi * torch.exp(1j * phi)[..., None].to(psi.dtype)

    model.encode = phased
    return box


def code_only(path):
    """Source lines with comment tails removed."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for t in tokenize.generate_tokens(path.open(encoding="utf-8").readline):
        if t.type == tokenize.COMMENT:
            row, col = t.start
            lines[row - 1] = lines[row - 1][:col]
    return "\n".join(lines)


# ---- 5. gauge invariance ----------------------------------------------------------------------------
def test_global_phase_of_psi_changes_nothing():
    model = make().eval()
    obs, act = data()
    ref = run(model, obs, act)
    phase_patch(model)
    out = run(model, obs, act)
    assert not torch.allclose(out["psi"], ref["psi"], atol=1e-3), "control: encode patch is not live in forward"
    for k in ("g", "w", "loss"):
        assert torch.allclose(out[k], ref[k], atol=1e-5), k
    assert torch.allclose(out["ghat"], ref["ghat"], atol=1e-5)


# ---- 6. real-psi ablation, 7. complex psi exposed -----------------------------------------------
def _act_dim(bed_name):
    return beds.make_bed(bed_name, 2, 4)["act"].shape[-1]


def test_real_psi_ablation_has_zero_chern():
    model = make(D=256, A=_act_dim("s2"), real=True).eval()
    obs, act = data(Dd=256, Aa=_act_dim("s2"))
    assert not torch.is_complex(run(model, obs, act)["psi"])
    assert run(model, obs, act)["psi"].dtype == torch.float32
    r = readouts.chern_of_arm(model, "s2", level=1)
    assert r is None or abs(r["chern"]) < 1e-9, r


def test_complex_psi_is_exposed_to_chern_readout():
    model = make(D=256, A=_act_dim("berry")).eval()
    r = readouts.chern_of_arm(model, "berry", level=1)  # runs forward on length-1 sequences (T = 1)
    assert isinstance(r, dict) and torch.isfinite(torch.tensor(float(r["chern"])))
    assert r["chern_random_phase"] is not None
    # also T = 1 directly: ghat is empty, nothing crashes
    out = model(torch.randn(2, 1, 256), torch.zeros(2, 1, _act_dim("berry")))
    assert out["ghat"].shape == (2, 0, D_g) and torch.isfinite(out["loss"])


# ---- 8. links ----------------------------------------------------------------------------------------
def test_link_is_unitary():
    model = make().eval()
    g = torch.Generator().manual_seed(5)
    zj = torch.complex(torch.randn(7, 2, generator=g), torch.randn(7, 2, generator=g))  # d_c = 2
    zk = torch.complex(torch.randn(7, 2, generator=g), torch.randn(7, 2, generator=g))
    with torch.no_grad():
        G = model.link(zj, zk)
    assert G.shape == (7, 2, 2) and torch.is_complex(G)
    eye = torch.eye(2, dtype=G.dtype).expand(7, 2, 2)
    assert torch.allclose(G.conj().transpose(-1, -2) @ G, eye, atol=1e-5)


# ---- 9. gradients ------------------------------------------------------------------------------------
GROUPS = {
    "encoder": r"enc",
    "V (Stiefel isometry)": r"(^|[._])V([._]|$)|stiefel|isometry|frame",
    "links": r"link",
    "amortised head": r"amort|head",
}


def test_loss_reaches_encoder_V_links_and_amortised_head():
    model = make()
    obs, act = data()
    run(model, obs, act, "loop")["loss"].backward()
    params = {n: p for n, p in model.named_parameters() if not n.startswith("tgt_")}
    for gname, pat in GROUPS.items():
        names = [n for n in params if re.search(pat, n, re.I) and "tgt_" not in n]
        assert names, f"no parameter name matches {gname} ({pat!r}): name them so"
        live = [n for n in names if params[n].grad is not None and params[n].grad.abs().sum() > 0]
        assert live, f"{gname}: no gradient reached any of {names}"
    assert all(torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None)
    for n, p in model.named_parameters():  # EMA target is stop-gradient
        if n.startswith("tgt_"):
            assert p.grad is None or p.grad.abs().sum() == 0, n


# ---- 10. L-GIVEN -------------------------------------------------------------------------------------
def test_signatures_obs_act_only():
    fp = list(inspect.signature(Asmodeous.forward).parameters.values())
    assert [p.name for p in fp] == ["self", "obs", "act", "mode"]
    assert fp[3].default in MODES
    rp = list(inspect.signature(Asmodeous.rollout).parameters)
    assert rp == ["self", "obs0", "act"]


def test_source_never_reads_bed_state():
    code = code_only(ZERO / "atlas" / "asmodeous.py")
    hits = re.findall(r"""\[\s*['"]state['"]\s*\]|get\(\s*['"]state['"]""", code)
    assert not hits, hits


# ---- 11. active FLOPs ---------------------------------------------------------------------------------
def test_active_flops_loop_vs_one_pass_and_linear_in_iters():
    model = make()
    assert model.active_flops(mode="loop") > model.active_flops(mode="one_pass")
    f = lambda it: make(loop_iters=it).active_flops(mode="loop")
    inc = f(8) - f(4)
    assert inc > 0 and f(12) - f(8) == inc and f(16) - f(12) == inc


def test_active_flops_linear_in_K_through_decider_only():
    f = lambda k: make(K=k).active_flops(mode="loop")
    d = f(16) - f(8)
    assert d > 0 and f(24) - f(16) == d  # K enters only via the K*N decider term (and is linear)
    assert all(isinstance(f(k), int) for k in (8, 16))


# ---- 12. rollout ----------------------------------------------------------------------------------------
def test_rollout_shape_and_action_dependence():
    model = make().eval()
    obs, act = data()
    H = 4
    with torch.no_grad():
        r1 = model.rollout(obs[:, 0], act[:, :H])
        r2 = model.rollout(obs[:, 0], act[:, :H] + 1.0)
    assert r1.shape == (B, H, D_g) and torch.isfinite(r1).all()
    assert (r1 - r2).abs().max() > 1e-6


def test_source_does_not_import_beds():
    code = code_only(ZERO / "atlas" / "asmodeous.py")
    hits = re.findall(r"^\s*(?:import\s+beds\b|from\s+beds\b)", code, re.M)
    assert not hits, hits


# ---- 13. readout keys for Thirteen: phase, chi, zidx, link -------------------------------------------------
def wrap(x):
    return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi


def test_readout_keys_shapes_and_antisymmetric_chi():
    model = make().eval()
    obs, act = data()
    out = run(model, obs, act)
    assert out["phase"].shape == (B, T, m) and out["phase"].dtype == torch.float32
    assert out["chi"].shape == (B, T, m, m) and out["chi"].dtype == torch.float32
    assert out["zidx"].shape == (B, T, m) and out["zidx"].dtype == torch.long
    assert out["link"].shape == (B, T, m, m, 2, 2) and torch.is_complex(out["link"])  # d_c = 2
    # zidx are the read charts: exactly the support of w
    assert torch.equal(torch.zeros(B, T, K, dtype=torch.bool).scatter_(-1, out["zidx"], True), out["w"] > 0)
    # phase = arg alpha_k, alpha = V psi at the read charts; chi_jk = arg(alpha_j conj alpha_k)
    alpha = torch.einsum("kn,btn->btk", model.V(), out["psi"]).gather(-1, out["zidx"]).detach()
    assert torch.allclose(wrap(out["phase"] - alpha.angle()), torch.zeros(B, T, m), atol=1e-4)
    chi_ref = (alpha[..., :, None] * alpha[..., None, :].conj()).angle()
    assert torch.allclose(wrap(out["chi"] - chi_ref), torch.zeros(B, T, m, m), atol=1e-4)
    assert (out["chi"].abs() <= torch.pi + 1e-6).all()
    assert torch.allclose(wrap(out["chi"] + out["chi"].transpose(-1, -2)), torch.zeros(B, T, m, m), atol=1e-4)
    assert torch.allclose(out["chi"].diagonal(dim1=-2, dim2=-1), torch.zeros(B, T, m), atol=1e-5)


def test_forward_links_unitary_with_identity_diagonal():
    model = make().eval()
    obs, act = data()
    G = run(model, obs, act)["link"]
    eye = torch.eye(2, dtype=G.dtype).expand_as(G)
    assert torch.allclose(G.conj().transpose(-1, -2) @ G, eye, atol=1e-5)
    diag = G.diagonal(dim1=2, dim2=3).movedim(-1, 2)  # [B, T, m, d_c, d_c]
    assert torch.allclose(diag, torch.eye(2, dtype=G.dtype).expand_as(diag), atol=1e-5)


def test_phase_shifts_by_phi_and_chi_is_gauge_invariant():
    model = make().eval()
    obs, act = data()
    ref = run(model, obs, act)
    box = phase_patch(model)
    out = run(model, obs, act)
    assert torch.equal(out["zidx"], ref["zidx"])
    dphi = out["phase"] - ref["phase"]
    assert torch.allclose(wrap(dphi - box["phi"][..., None]), torch.zeros(B, T, m), atol=1e-4)
    assert torch.allclose(wrap(out["chi"] - ref["chi"]), torch.zeros(B, T, m, m), atol=1e-4)
    assert torch.allclose(out["link"], ref["link"], atol=1e-5)

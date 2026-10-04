"""T7 bench contracts: fused Triton chart MLP == StackedMLP gather path; timer stats; FLOP matching."""
import pytest
import torch

from atlas.observer_atlas import ObserverAtlas, StackedMLP

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("K,m,N,i,h,o", [(16, 1, 1, 256, 32, 2), (4096, 2, 1000, 256, 32, 2),
                                         (64, 2, 33, 4, 32, 2), (1024, 2, 7, 2, 32, 8), (16, 2, 5, 5, 20, 3)])
def test_fused_chart_mlp_matches_gather_reference(K, m, N, i, h, o):
    from bench_t7 import chart_mlp
    torch.manual_seed(0)
    mod = StackedMLP(K, i, h, o).cuda()
    idx = torch.randint(0, K, (N, m), device="cuda")
    for x in (torch.randn(N, m, i, device="cuda"), torch.randn(N, i, device="cuda")[:, None].expand(-1, m, -1)):
        with torch.no_grad():
            ref, got = mod(x, idx), chart_mlp(mod, x, idx)
        assert got.shape == ref.shape and torch.allclose(got, ref, atol=1e-4, rtol=1e-4)


SHAPES = [(16, 1, 1, 256, 32, 2), (4096, 2, 1000, 256, 32, 2), (64, 2, 33, 4, 32, 2), (1024, 2, 7, 2, 32, 8),
          (16, 2, 5, 5, 20, 3), (4, 2, 3000, 256, 32, 8)]  # last: few experts, long segments (grouped path)


@cuda
@pytest.mark.parametrize("K,m,N,i,h,o", SHAPES)
def test_grouped_kernel_fp32_and_bf16(K, m, N, i, h, o):
    """fp32 weights: ieee dot, tol 1e-4 vs the gather path.
    bf16 weights (x and hidden also rounded to bf16, fp32 accumulate): unit roundoff u = 2^-9 ~ 2e-3 on four
    roundings (x, W1, hidden, W2) bounds the relative error near 4u ~ 8e-3 of the output scale; the test allows
    2e-2 * max|ref| (2.5x margin), checked against the fp32 gather reference."""
    from bench_t7 import grouped_chart_mlp
    torch.manual_seed(0)
    mod = StackedMLP(K, i, h, o).cuda()
    idx = torch.randint(0, K, (N, m), device="cuda")
    for x in (torch.randn(N, m, i, device="cuda"), torch.randn(N, i, device="cuda")[:, None].expand(-1, m, -1)):
        with torch.no_grad():
            ref = mod(x, idx)
            f32 = grouped_chart_mlp(mod, x, idx)
            b16 = grouped_chart_mlp(mod, x, idx, dtype=torch.bfloat16)
        assert f32.dtype == b16.dtype == torch.float32 and f32.shape == ref.shape
        assert torch.allclose(f32, ref, atol=1e-4, rtol=1e-4)
        assert (b16 - ref).abs().max() <= 2e-2 * ref.abs().max()


@cuda
@pytest.mark.parametrize("kind", ["row", "grouped"])
def test_fused_paths_backprop_like_reference(kind):
    """Training through a patched atlas must not silently drop grads: backward recomputes the reference."""
    from bench_t7 import fused
    torch.manual_seed(0)
    mod = StackedMLP(8, 6, 16, 3).cuda()
    x = torch.randn(5, 6, device="cuda", requires_grad=True)
    idx = torch.randint(0, 8, (5, 2), device="cuda")
    xe = x[:, None].expand(-1, 2, -1)
    gy = torch.randn(5, 2, 3, device="cuda")
    ref = torch.autograd.grad(mod(xe, idx), [x, *mod.parameters()], gy)
    got = torch.autograd.grad(fused(mod, xe, idx, kind), [x, *mod.parameters()], gy)
    for a, b in zip(ref, got):
        assert torch.allclose(a, b, atol=1e-5)


@cuda
@pytest.mark.parametrize("K,pk", [(16, False), (256, True)])
@pytest.mark.parametrize("kind", ["row", "grouped", "auto"])
def test_triton_atlas_observe_matches_reference(K, pk, kind):
    from bench_t7 import CFG, use_triton
    torch.manual_seed(0)
    ref = ObserverAtlas(K=K, m=2, product_keys=pk, **CFG).cuda().eval()
    tri = use_triton(ObserverAtlas(K=K, m=2, product_keys=pk, **CFG).cuda().eval(), kind)
    tri.load_state_dict(ref.state_dict())
    obs, act = torch.randn(64, 1, CFG["D"], device="cuda"), torch.randn(64, 1, CFG["A"], device="cuda")
    with torch.no_grad():
        a, b = ref.observe(obs, act, "lazy"), tri.observe(obs, act, "lazy")
    for k in ("g", "gh", "idx", "w"):
        assert torch.allclose(a[k].float(), b[k].float(), atol=1e-4), k


@cuda
def test_time_us_stats():
    from bench_t7 import time_us
    x = torch.randn(256, 256, device="cuda")
    s = time_us(lambda: x @ x, warmup=5, iters=100)
    assert s["n"] == 100 and 0 < s["q1"] <= s["median"] <= s["q3"] and s["iqr"] == s["q3"] - s["q1"]


@pytest.mark.parametrize("K,pk", [(16, False), (4096, True)])
def test_dense_and_peer_match_atlas_active_flops(K, pk):
    from bench_t7 import CFG, match_flops
    from arms.peer_jepa import PEERJEPA
    from arms.single_jepa import SingleJEPA
    target = ObserverAtlas(K=K, m=2, product_keys=pk, **CFG).active_flops()
    for cls, kw in ((SingleJEPA, {}), (PEERJEPA, dict(n_experts=K, topk=min(8, int(K ** 0.5))))):
        arm = match_flops(cls, target, **kw)
        assert abs(arm.active_flops() - target) / target < 0.05, cls.__name__


@cuda
def test_cell_backbone_equals_gru_step():
    from bench_t7 import CFG, CellBackbone
    torch.manual_seed(0)
    at = ObserverAtlas(K=16, m=2, **CFG).cuda().eval()
    obs = torch.randn(64, 1, CFG["D"], device="cuda")
    with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):  # cuDNN GRU defaults to TF32 (7e-4 off)
        assert torch.allclose(CellBackbone(at.backbone)(obs)[0], at.backbone(obs)[0], atol=1e-5)


@cuda
@pytest.mark.parametrize("K,m,pk", [(16, 1, False), (64, 2, False), (256, 2, True), (4096, 1, True), (4096, 2, True)])
@pytest.mark.parametrize("B", [1, 64])
def test_fused_step_matches_observe(K, m, pk, B):
    """One-launch route + E + P + phi + mix == ObserverAtlas.observe(mode='lazy') at T=1 (fp32)."""
    from bench_t7 import CFG, fused_observe
    torch.manual_seed(0)
    at = ObserverAtlas(K=K, m=m, product_keys=pk, **CFG).cuda().eval()
    obs, act = torch.randn(B, 1, CFG["D"], device="cuda"), torch.randn(B, 1, CFG["A"], device="cuda")
    with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
        ref, got = at.observe(obs, act, "lazy"), fused_observe(at, obs, act)
    order = lambda r: r["idx"].sort(-1)
    (ri, rp), (gi, gp) = order(ref), order(got)
    assert torch.equal(ri, gi)
    assert torch.allclose(ref["w"].gather(-1, rp), got["w"].gather(-1, gp), atol=1e-5)
    for k in ("g", "gh", "gate"):
        assert torch.allclose(ref[k], got[k], atol=1e-4), k

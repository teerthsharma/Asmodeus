"""E1 checks: parameter counts match the preregistration, models are causal, the sliding window scores every
token once with the stated context, and the online arm scores each chunk before it learns from it."""
import numpy as np
import torch

from common import stream
from nets import GRUNet, TFNet, n_params, tf_stream_loss, gru_stream_loss
from online import prequential


def test_params_match_prereg():
    assert abs(n_params(GRUNet(16)) - 1.9e3) < 0.1e3
    assert abs(n_params(GRUNet(64)) - 17e3) < 1e3
    assert abs(n_params(GRUNet(256)) - 0.2e6) < 0.03e6
    assert abs(n_params(TFNet()) - 0.8e6) < 0.1e6


def test_causal():
    torch.manual_seed(0)
    for m in (GRUNet(32), TFNet(ctx=64)):
        m.eval()
        x = torch.randint(0, 16, (1, 64))
        y = x.clone()
        y[0, 40:] = (y[0, 40:] + 1) % 16
        with torch.no_grad():
            assert torch.allclose(m(x)[0][0, :40], m(y)[0][0, :40], atol=1e-5)


def test_sliding_window_matches_full_and_partial():
    torch.manual_seed(0)
    m = TFNet(ctx=64).eval()
    x = np.random.default_rng(0).integers(0, 16, 300)
    a = tf_stream_loss(m, x, "cpu", stride=32)
    assert a.shape == (299,) and np.isfinite(a).all()
    with torch.no_grad():  # first window: identical to a direct pass
        lo = m(torch.as_tensor(x[:64])[None])[0][0]
        d = torch.nn.functional.cross_entropy(lo, torch.as_tensor(x[1:65]), reduction="none").numpy()
    assert np.allclose(a[:64], d, atol=1e-5)
    short = x[:40]  # stream shorter than the context: one window
    assert tf_stream_loss(m, short, "cpu", stride=32).shape == (39,)


def test_gru_stream_loss_shape():
    x = stream("REC", 1)[0][:500]
    assert gru_stream_loss(GRUNet(16).eval(), x, "cpu").shape == (499,)


def test_prequential_scores_before_learning():
    x = stream("REC", 1)[0][:200]
    out = prequential(x, 16, 1e-2)
    torch.manual_seed(0)
    m = GRUNet(16)
    with torch.no_grad():  # the first chunk is scored by the untouched initial model
        lo = m(torch.as_tensor(x[:32])[None])[0][0]
        d = torch.nn.functional.cross_entropy(lo, torch.as_tensor(x[1:33]), reduction="none").numpy()
    assert out.shape == (199,) and np.allclose(out[:32], d, atol=1e-5)

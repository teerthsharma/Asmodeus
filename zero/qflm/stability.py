import torch


def renorm_psi(psi, bad=None):
    s = psi.abs().amax(-1, keepdim=True)  # max-abs scale first: fp32 norm of 1e-30 amps underflows
    zero = (s == 0).squeeze(-1)
    if bad is not None:
        bad |= zero  # sync-free device flag; zero rows come out NaN (0/0), no eps
    elif not (psi.is_cuda and torch.cuda.is_current_stream_capturing()):
        idx = zero.reshape(-1).nonzero().flatten().tolist()
        if idx:
            raise ValueError(f"renorm_psi: zero-norm rows {idx}")
    psi = psi / s
    return psi / torch.linalg.vector_norm(psi, dim=-1, keepdim=True)


def log_born(amp):
    a = amp.to(torch.complex64 if amp.is_complex() else torch.float32)  # upcast BEFORE squaring
    return torch.logsumexp(2 * torch.log(a.abs()), dim=-1)


def null_record_logp(log_p_null, log_p_x):
    z = log_p_null.new_zeros(*log_p_null.shape[:-1], 1)
    prefix = torch.cat([z, torch.cumsum(log_p_null, -1)], -1)  # exclusive cumsum
    return torch.logsumexp(prefix + log_p_x, dim=-1)


def butterfly_apply(x, angles):
    L, h = angles.shape
    N = x.shape[-1]
    rd = x.real.dtype if x.is_complex() else x.dtype
    for l in range(L):
        s = 1 << l
        y = x.reshape(*x.shape[:-1], N // (2 * s), 2, s)
        c = torch.cos(angles[l].float()).to(rd).reshape(N // (2 * s), s)
        sn = torch.sin(angles[l].float()).to(rd).reshape(N // (2 * s), s)
        a, b = y[..., 0, :], y[..., 1, :]
        x = torch.stack([c * a - sn * b, sn * a + c * b], -2).reshape(x.shape)
    return x


def orth_error(B):
    I = torch.eye(B.shape[-1], dtype=B.dtype)
    return (B.mH @ B - I).abs().max()

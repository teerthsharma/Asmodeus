"""GRU-ONLINE arms: prequential 32-token chunks. Each chunk is scored with the parameters and carried state
from before it, then one Adam step is taken on that chunk's loss."""
import numpy as np
import torch
import torch.nn.functional as F

from common import V
from nets import GRUNet

CHUNK = 32


def prequential(x, hidden, lr, device="cpu", init_seed=0):
    torch.manual_seed(init_seed)
    model = GRUNet(hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.as_tensor(x, device=device).long()[None]
    out, h = np.empty(len(x) - 1), None
    for a in range(0, len(x) - 1, CHUNK):
        b = min(a + CHUNK, len(x) - 1)
        logits, h_new = model(xt[:, a:b], h)
        loss = F.cross_entropy(logits[0], xt[0, a + 1:b + 1], reduction="none")
        out[a:b] = loss.detach().cpu().numpy()
        opt.zero_grad()
        loss.mean().backward()
        opt.step()
        h = h_new.detach()
    return out

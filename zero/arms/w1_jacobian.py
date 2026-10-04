"""w1 (orientability) from Jacobian-determinant signs of learned chart transition maps (Paluzo-Hidalgo & Ike, arXiv 2602.22873).

Transition j -> k of a multi-chart autoencoder is tau_jk(u) = sigmoid(Ec_k(Dc_j(u))); sign det J_tau_jk is an edge cocycle,
and w1 = 0 iff that cocycle is a coboundary (every cycle has sign product +1).
"""
import torch
from torch.func import jacrev, vmap


def jac_sign(f, u):
    """f: [N,d] -> [N,d] row-wise; returns int [N] of sign(det J_f(u_n))."""
    J = vmap(jacrev(lambda x: f(x[None])[0]))(u)
    return torch.linalg.det(J).sign().long()


def transition_fn(model, j, k):
    """u -> sigmoid(Ec_k(Dc_j(u))); only charts j and k evaluate."""
    idx = lambda u, c: torch.full((u.shape[0],), c, dtype=torch.long)
    return lambda u: torch.sigmoid(model.Ec.one(model.Dc.one(u, idx(u, j)), idx(u, k)))


def edge_signs(model, obs, tau=0.2, min_points=8):
    """{(j,k): (sign, agreement)} for j<k with >= min_points rows where both memberships >= tau."""
    with torch.no_grad():
        z = model.E(obs).reshape(-1, model.Ec.W1.shape[1])
        p, zc = model.member(z).softmax(-1), torch.sigmoid(model.Ec(z))
    out = {}
    for j in range(model.K):
        for k in range(j + 1, model.K):
            rows = (p[:, j] >= tau) & (p[:, k] >= tau)
            if rows.sum() < min_points:
                continue
            s = jac_sign(transition_fn(model, j, k), zc[rows, j].detach())
            maj = 1 if s.sum() >= 0 else -1
            out[(j, k)] = (maj, (s == maj).double().mean().item())
    return out


def w1_class(edges):
    """edges {(j,k): sign}. 0 if vertex signs s exist with sign_jk = s_j*s_k on every edge (BFS 2-colouring), else 1."""
    adj = {}
    for (j, k), s in edges.items():
        adj.setdefault(j, []).append((k, s))
        adj.setdefault(k, []).append((j, s))
    col = {}
    for root in adj:
        if root in col:
            continue
        col[root], stack = 1, [root]
        while stack:
            v = stack.pop()
            for w, s in adj[v]:
                if w not in col:
                    col[w] = col[v] * s
                    stack.append(w)
                elif col[w] != col[v] * s:
                    return 1
    return 0


def loop_sign(edges, charts):
    """Product of edge signs around the closed loop charts[0] -> ... -> charts[0]; 0 if any transition's edge is missing."""
    seq = [c for i, c in enumerate(charts) if i == 0 or c != charts[i - 1]]
    prod = 1
    for a, b in zip(seq, seq[1:] + seq[:1]):
        if a != b:
            prod *= edges.get((min(a, b), max(a, b)), 0)
    return prod

"""E0d tournament: candidate TT deciders over the same era bank (PROGRAM.md Addendum 2, E0d_PREREG.md).

Every decider holds w_t, the TT's softmax weight over experts [eras..., window, BOCPD] (the chronal fluid density),
and answers p(x_t) from the experts' predictive vectors Y (n_experts x V). The era bank, births and merges are shared
(fcpa bookkeeping); deciders differ only in how w_t moves and how the answer is formed.
"""
import numpy as np


class Soft:
    """D0 incumbent: fixed share (Herbster-Warmuth) + Bayes step, 'after multiplication and normalisation'."""
    name = "SOFT"

    def __init__(self, n_tail, alpha):
        self.n_tail, self.alpha, self.w = n_tail, alpha, np.ones(1 + n_tail) / (1 + n_tail)
        self.single = 0  # steps answered by one expert (compute proxy)

    def share(self):
        return (1 - self.alpha) * self.w + self.alpha / len(self.w)

    def answer(self, v, Y, cur):
        return float(v @ Y[:, cur])

    def step(self, Y, cur):
        v = self.share()
        p = self.answer(v, Y, cur)
        post = v * Y[:, cur]
        self.w = post / post.sum()
        return p

    def birth(self):
        al, nt = self.alpha, self.n_tail
        self.w = np.concatenate([self.w[:-nt] * (1 - al), [al], self.w[-nt:] * (1 - al)])
        self.w /= self.w.sum()

    def fold(self, i, j):
        self.w[i] += self.w[j]
        self.w = np.delete(self.w, j)

    def switch(self, a, b):
        pass


class Momentum(Soft):
    """D1: the shared mass of era i moves along a learned era-to-era kernel Q (FCPA D3, Markov law over observer
    outcomes), half by Q and half uniformly; Q = Dirichlet(1)-smoothed counts of observed era switches."""
    name = "MOM"

    def __init__(self, n_tail, alpha):
        super().__init__(n_tail, alpha)
        self.S = np.zeros((1, 1))

    def share(self):
        n_e = len(self.w) - self.n_tail
        out = (1 - self.alpha) * self.w + 0.5 * self.alpha / len(self.w)  # half the shared mass: uniform
        moved = 0.5 * self.alpha * self.w  # other half: era mass travels along Q, tail mass stays
        if n_e > 1:
            Q = self.S[:n_e, :n_e] + 1.0
            np.fill_diagonal(Q, 0.0)
            Q /= Q.sum(1, keepdims=True)
            moved[:n_e] = moved[:n_e] @ Q
        return out + moved

    def birth(self):
        super().birth()
        n = self.S.shape[0]
        S = np.zeros((n + 1, n + 1))
        S[:n, :n] = self.S
        self.S = S

    def fold(self, i, j):
        super().fold(i, j)
        S = self.S
        S[i] += S[j]
        S[:, i] += S[:, j]
        S[i, i] = 0
        self.S = np.delete(np.delete(S, j, 0), j, 1)

    def switch(self, a, b):
        if a != b:
            self.S[a, b] += 1


class Tempered(Soft):
    """D3: answer with w^(1/T) renormalised (L4 post-selection temperature); memory update unchanged."""
    name = "TEMP"

    def __init__(self, n_tail, alpha, T):
        super().__init__(n_tail, alpha)
        self.T = T

    def answer(self, v, Y, cur):
        u = v ** (1.0 / self.T)
        return float((u / u.sum()) @ Y[:, cur])


class GeneralBayes(Soft):
    """D6: tempered likelihood (generalised/safe Bayes), w <- w * p^eta; answer as SOFT."""
    name = "GBAYES"

    def __init__(self, n_tail, alpha, eta):
        super().__init__(n_tail, alpha)
        self.eta = eta

    def step(self, Y, cur):
        v = self.share()
        p = self.answer(v, Y, cur)
        post = v * Y[:, cur] ** self.eta
        self.w = post / post.sum()
        return p


class HardJump(Soft):
    """D4: answer with the top expert alone when its weight >= thr (one expert computed), else the blend (L5 spirit)."""
    name = "HARD"

    def __init__(self, n_tail, alpha, thr):
        super().__init__(n_tail, alpha)
        self.thr = thr

    def answer(self, v, Y, cur):
        k = int(v.argmax())
        if v[k] >= self.thr:
            self.single += 1
            return float(Y[k, cur])
        return float(v @ Y[:, cur])


class CTC(Soft):
    """D2: answer with the CTC fixed point u* = G(u*), G(u)_i ∝ v_i exp(-c |Y_i - sum_k u_k Y_k|^2), c = 2q/Δ²
    (L32(e): Hilbert contraction with constant q < 1 for any v > 0). Memory update unchanged."""
    name = "CTC"

    def __init__(self, n_tail, alpha, q, tol=1e-6, max_iter=200):
        super().__init__(n_tail, alpha)
        self.q, self.tol, self.max_iter = q, tol, max_iter

    def answer(self, v, Y, cur):
        D = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
        d2 = D.max()
        if d2 <= 0:
            return float(v @ Y[:, cur])
        c = 2 * self.q / d2
        lv = np.log(v)
        u = v
        for _ in range(self.max_iter):
            g = lv - c * (D @ u)  # Lemma 0: |Y_i - Ybar_u|^2 = (Du)_i - const
            g = np.exp(g - g.max())
            g /= g.sum()
            if np.ptp(np.log(g) - np.log(u)) < self.tol:
                u = g
                break
            u = g
        return float(u @ Y[:, cur])


class MomentumGB(Momentum):
    """D7: momentum share (D1) + tempered likelihood (D6)."""
    name = "MOM+GB"

    def __init__(self, n_tail, alpha, eta):
        super().__init__(n_tail, alpha)
        self.eta = eta

    def step(self, Y, cur):
        v = self.share()
        p = self.answer(v, Y, cur)
        post = v * Y[:, cur] ** self.eta
        self.w = post / post.sum()
        return p

"""QF-LM sampling: next token = measurement outcome over V tokens + NULL ("still thinking")."""
import torch

NULL = -1


class ContextStepper:
    """Adapts any lm/cores.py-contract model: softmax of last logits, p_null = 0."""

    def __init__(self, model):
        self.model, self.toks = model, []

    def reset(self, prompt_ids):
        if not len(prompt_ids):
            raise ValueError("prompt must be non-empty")
        self.toks = list(prompt_ids)

    @torch.no_grad()
    def dist(self):
        lg = self.model(torch.tensor([self.toks]))["logits"][0, -1]
        return torch.softmax(lg.float(), -1), 0.0

    def observe(self, outcome):
        if outcome != NULL:
            self.toks.append(outcome)


def sample_next(p_tok, p_null, *, temperature=1.0, top_k=0, top_p=1.0, history=(),
                repetition_penalty=1.0, no_repeat_ngram=0, allow_null=True, generator=None):
    V = p_tok.numel()
    q = torch.cat([p_tok.float(), torch.tensor([float(p_null) if allow_null else 0.0])])
    hist = list(history)
    if repetition_penalty != 1.0:
        for v in set(hist):
            if 0 <= v < V:
                q[v] /= repetition_penalty
    n = no_repeat_ngram
    if n > 0 and len(hist) >= n - 1:
        pre = tuple(hist[len(hist) - (n - 1):]) if n > 1 else ()
        for i in range(len(hist) - n + 1):
            if tuple(hist[i:i + n - 1]) == pre and 0 <= hist[i + n - 1] < V:
                q[hist[i + n - 1]] = 0.0
    if q.sum() <= 0:  # everything banned: unguarded argmax, never crash
        return int(p_tok.argmax())
    if temperature == 0:
        i = int(q.argmax())
    else:
        pos = q > 0
        q = torch.where(pos, torch.exp(q.clamp_min(1e-30).log() / temperature), torch.zeros_like(q))
        q = q / q.sum()
        s, idx = q.sort(descending=True)
        keep = torch.ones_like(s, dtype=torch.bool)
        if top_k > 0:
            keep[top_k:] = False
        if top_p < 1.0:
            cum = s.cumsum(0)
            keep &= (cum - s) < top_p  # smallest prefix reaching top_p, top one always kept
        s = s * keep
        i = int(idx[torch.multinomial(s / s.sum(), 1, generator=generator)])
    return NULL if i == V else i


def generate(stepper, prompt_ids, max_new=64, *, think_budget=4, stop_ids=(), seed=None, **sample_kw):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    prompt, out, thinks_log = list(prompt_ids), [], []
    stepper.reset(prompt)
    while len(out) < max_new:
        thinks = 0
        while True:
            p, pn = stepper.dist()
            o = sample_next(p, pn, history=prompt + out, allow_null=thinks < think_budget,
                            generator=g, **sample_kw)
            stepper.observe(o)
            if o != NULL:
                break
            thinks += 1
        out.append(o)
        thinks_log.append(thinks)
        if o in stop_ids:
            break
    return {"ids": out, "think_steps": thinks_log}

"""Minimal local chat loop over a QF-LM / lm.cores checkpoint."""
import argparse

import torch

from qflm.generate import ContextStepper, QFLMStepper, generate


class ByteTok:
    vocab = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", "replace")


class _HFTok:
    def __init__(self, tk):
        self.tk = tk

    def encode(self, text):
        return self.tk.encode(text).ids

    def decode(self, ids):
        return self.tk.decode(list(ids), skip_special_tokens=False)


def load_tokenizer(path):
    from tokenizers import Tokenizer

    return _HFTok(Tokenizer.from_file(path))


def render(history, user_text):
    turns = list(history) + [("User", user_text)]
    return "".join(f"{r}: {t}\n" for r, t in turns) + "Assistant: "


def reply(stepper, tok, history, user_text, **gen_kw):
    ids = tok.encode(render(history, user_text))
    out = generate(stepper, ids, **gen_kw)
    reply.last_think = out["think_steps"]
    return tok.decode(out["ids"]).split("\nUser:")[0].strip()


reply.last_think = []


def load(path):
    from lm.cores import build

    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build(ck["core"], ck["vocab"], ck["d"], **ck.get("kw", {}))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    if hasattr(m, "_emit"):  # QFLM: batched-tensor API, wrap into the scalar protocol
        return QFLMStepper(m)
    return m if all(hasattr(m, a) for a in ("reset", "dist", "observe")) else ContextStepper(m)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("checkpoint")
    a.add_argument("--tokenizer")
    a.add_argument("--temperature", type=float, default=0.8)
    a.add_argument("--top-p", type=float, default=0.9)
    a.add_argument("--think-budget", type=int, default=4)
    a.add_argument("--max-new", type=int, default=128)
    a.add_argument("--seed", type=int)
    a = a.parse_args()
    st = load(a.checkpoint)
    tok = load_tokenizer(a.tokenizer) if a.tokenizer else ByteTok()
    hist = []
    while True:
        try:
            u = input("You: ")
        except EOFError:
            break
        if u.strip() == "/quit":
            break
        r = reply(st, tok, hist, u, temperature=a.temperature, top_p=a.top_p,
                  think_budget=a.think_budget, max_new=a.max_new, seed=a.seed)
        hist += [("User", u), ("Assistant", r)]
        t = reply.last_think
        print(f"Assistant: {r}\nthink steps: {sum(t) / max(len(t), 1):.2f}")


if __name__ == "__main__":
    main()

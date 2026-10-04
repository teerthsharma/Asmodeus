import torch

from chat import ByteTok, load, render, reply
from qflm.generate import NULL


class Script:
    def __init__(self, text):
        self.ids, self.i = list(text.encode()), 0

    def reset(self, prompt_ids):
        self.i = 0

    def dist(self):
        p = torch.zeros(256)
        p[self.ids[min(self.i, len(self.ids) - 1)]] = 1.0
        return p, 0.0

    def observe(self, o):
        if o != NULL:
            self.i += 1


def test_bytetok_and_template():
    t = ByteTok()
    assert t.decode(t.encode("héllo")) == "héllo"
    assert render([("User", "a"), ("Assistant", "b")], "c") == "User: a\nAssistant: b\nUser: c\nAssistant: "


def test_reply_cuts_at_user_turn():
    out = reply(Script("hi there\nUser: x"), ByteTok(), [], "yo", temperature=0, max_new=16, think_budget=0)
    assert out == "hi there"


def test_load_checkpoint(tmp_path):
    from lm.cores import build

    m = build("gpt", 256, 32, layers=1)
    p = tmp_path / "c.pt"
    torch.save({"core": "gpt", "vocab": 256, "d": 32, "kw": {"layers": 1}, "state_dict": m.state_dict()}, p)
    out = reply(load(str(p)), ByteTok(), [], "hello", temperature=0, max_new=8)
    assert isinstance(out, str)

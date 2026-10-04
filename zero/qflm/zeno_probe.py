"""Repeat-loop escape probe: after emitting argmax token v, does P(v) fall as think steps are inserted?"""
from qflm.generate import NULL


def repeat_escape(stepper, prompt_ids, max_think=4):
    stepper.reset(list(prompt_ids))
    v = int(stepper.dist()[0].argmax())
    stepper.observe(v)
    rep, nul = [], []
    for s in range(max_think + 1):
        p, pn = stepper.dist()
        rep.append(float(p[v]))
        nul.append(float(pn))
        if s < max_think:
            stepper.observe(NULL)
    return {"token": v, "p_repeat": rep, "p_null": nul, "escape": 1 - rep[-1]}


def summary(stepper, prompts, max_think=4):
    curves = [repeat_escape(stepper, p, max_think)["p_repeat"] for p in prompts]
    return [sum(c[i] for c in curves) / len(curves) for i in range(max_think + 1)]

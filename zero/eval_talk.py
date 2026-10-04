"""Blind human/LLM evaluation harness for model "talk" quality (A vs B).

The rubric (RUBRIC_FIELDS, 1-5 Likert) and pairwise preference are the real
measures; proxy_scores are cheap automatic stand-ins. Stdlib only.
"""
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

PROMPTS_SHA256 = "ff9a955c3a864d7fa9b4daca3bbae308b524fddffe65c7182986d892de264c7e"
SHEET = Path(__file__).resolve().parent / "ref" / "prompts_50.json"
RUBRIC_FIELDS = ("on_topic", "grammatical", "non_repeating")


def load_prompts(path=SHEET):
    prompts = json.loads(Path(path).read_text(encoding="utf-8"))["prompts"]
    h = hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()
    if len(prompts) != 50 or h != PROMPTS_SHA256:
        raise ValueError(f"prompt sheet altered (n={len(prompts)}, sha256={h})")
    return prompts


def blind_pairs(prompts, replies_a, replies_b, seed):
    """Shuffle items and randomise sides. key[id] = model ("A"/"B") on the LEFT."""
    if not len(prompts) == len(replies_a) == len(replies_b):
        raise ValueError("prompts/replies_a/replies_b lengths differ")
    rng = random.Random(seed)
    order = list(range(len(prompts)))
    rng.shuffle(order)
    items, key = [], {}
    for n, i in enumerate(order):
        a_left = rng.random() < 0.5
        l, r = (replies_a[i], replies_b[i]) if a_left else (replies_b[i], replies_a[i])
        iid = f"i{n:02d}"
        items.append({"id": iid, "prompt": prompts[i], "left": l, "right": r})
        key[iid] = "A" if a_left else "B"
    return items, key


def unblind(judgements, key):
    """judgements: id -> left/right/tie  =>  id -> A/B/tie."""
    flip = {"A": "B", "B": "A"}
    out = {}
    for iid, j in judgements.items():
        if j == "tie":
            out[iid] = "tie"
        elif j in ("left", "right"):
            out[iid] = key[iid] if j == "left" else flip[key[iid]]
        else:
            raise ValueError(f"bad judgement {j!r}")
    return out


def validate_rubric(scores):
    for f in RUBRIC_FIELDS:
        v = scores.get(f)
        if not (isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5):
            raise ValueError(f"{f} must be an int in 1..5, got {v!r}")


# ponytail: proxies below are NOT the rubric; human/LLM raters are. Crude, gameable.
_STOP = set("""about above after again also been before being both could does down each
from have here into just like more most once only other over same some such than that
their them then there these they this those through under until upon very were what when
where which while will with would your""".split())


def _words(t):
    return re.findall(r"\w+", t.lower())


def proxy_scores(prompt, reply):
    w = _words(reply)
    g4 = list(zip(w, w[1:], w[2:], w[3:]))
    rep = 1 - len(set(g4)) / len(g4) if g4 else 0.0
    g2 = list(zip(w, w[1:]))
    d2 = len(set(g2)) / len(g2) if g2 else 0.0
    # grammar: fraction of sentences starting capitalised and ending in . ! ? or a
    # closing quote, minus 0.2 per run of 3+ identical consecutive tokens; clipped [0,1].
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", reply.strip()) if s.strip()]
    ok = sum(1 for s in sents
             if re.match(r"""["'“‘]*[A-Z]""", s)
             and re.search(r"""[.!?"'”’]$""", s))
    base = ok / len(sents) if sents else 0.0
    runs = sum(1 for i in range(len(w) - 2) if w[i] == w[i + 1] == w[i + 2]
               and (i == 0 or w[i - 1] != w[i]))
    gram = max(0.0, min(1.0, base - 0.2 * runs))
    pw = {x for x in _words(prompt) if len(x) >= 4 and x not in _STOP}
    on = len(pw & set(w)) / len(pw) if pw else 0.0
    return {"repetition": rep, "distinct_2": d2, "grammar_proxy": gram, "on_topic_proxy": on}


def cohen_kappa(r1, r2):
    n = len(r1)
    if n != len(r2) or n == 0:
        raise ValueError("raters need equal non-zero length")
    po = sum(a == b for a, b in zip(r1, r2)) / n
    c1, c2 = Counter(r1), Counter(r2)
    pe = sum(c1[k] * c2[k] for k in c1) / n ** 2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def weighted_kappa(r1, r2, weights="quadratic", categories=range(1, 6)):
    cats = list(categories)
    k, n = len(cats), len(r1)
    if n != len(r2) or n == 0:
        raise ValueError("raters need equal non-zero length")
    if weights not in ("quadratic", "linear"):
        raise ValueError(weights)
    pw = 2 if weights == "quadratic" else 1
    ix = {c: i for i, c in enumerate(cats)}
    w = lambda i, j: (abs(i - j) / (k - 1)) ** pw
    obs = [[0.0] * k for _ in range(k)]
    for a, b in zip(r1, r2):
        obs[ix[a]][ix[b]] += 1 / n
    m1 = [sum(row) for row in obs]
    m2 = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    do = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    de = sum(w(i, j) * m1[i] * m2[j] for i in range(k) for j in range(k))
    return 1.0 if de == 0 else 1 - do / de


def krippendorff_alpha(data, level="nominal"):
    """data: list of raters, each a list over units, None = missing.
    Units with <2 ratings are ignored. Coincidence-matrix formulation;
    ordinal uses Krippendorff's rank-based metric."""
    units = [[r[u] for r in data if r[u] is not None] for u in range(len(data[0]))]
    o = Counter()
    for v in units:
        m = len(v)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(v[i], v[j])] += 1 / (m - 1)
    n_c = Counter()
    for (c, _), x in o.items():
        n_c[c] += x
    n = sum(n_c.values())
    vals = sorted(n_c)
    if level == "nominal":
        d = lambda c, k: float(c != k)
    elif level == "interval":
        d = lambda c, k: (c - k) ** 2
    elif level == "ordinal":
        def d(c, k):
            lo, hi = sorted((c, k))
            s = sum(n_c[g] for g in vals if lo <= g <= hi)
            return (s - (n_c[c] + n_c[k]) / 2) ** 2
    else:
        raise ValueError(level)
    do = sum(x * d(c, k) for (c, k), x in o.items())
    de = sum(n_c[c] * n_c[k] * d(c, k) for c in vals for k in vals)
    return 1.0 if de == 0 else 1 - (n - 1) * do / de


def _binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def _root(f):
    """Root of increasing f on [0,1] by bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(mid) > 0 else (mid, hi)
    return lo


def _clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (0.0, 1.0)
    lower = 0.0 if k == 0 else _root(lambda p: (1 - _binom_cdf(k - 1, n, p)) - alpha / 2)
    upper = 1.0 if k == n else _root(lambda p: alpha / 2 - _binom_cdf(k, n, p))
    return (lower, upper)


def tie_test(prefs):
    """Exact two-sided sign test on non-ties + exact 95% CI of A's win-rate.
    sign_test_nonsig=True iff p>=0.05 AND CI contains 0.5 (weak rule, NOT the tie
    decision: see equivalence_tie, PREREG Amendment 1). Low power: with ~50 items a
    non-significant result is weak evidence of a tie, hence n_nontie is returned."""
    a, b = prefs.count("A"), prefs.count("B")
    n = a + b
    p = 1.0 if n == 0 else min(1.0, 2 * _binom_cdf(min(a, b), n, 0.5))
    lo, hi = _clopper_pearson(a, n)
    return {"wins_a": a, "wins_b": b, "ties": prefs.count("tie"), "n_nontie": n,
            "p_two_sided": p, "ci95": (lo, hi), "sign_test_nonsig": p >= 0.05 and lo <= 0.5 <= hi}


def equivalence_tie(prefs_by_prompt, margin=(0.40, 0.60), n_boot=2000, seed=0, ci=0.90):
    """TOST tie rule (PREREG Amendment 1): cluster-bootstrap (prompts resampled) percentile
    CI of A's pooled non-tie win-rate must lie strictly inside `margin`."""
    import numpy as np
    if not prefs_by_prompt:
        raise ValueError("no prompts")
    rows = []
    for g in prefs_by_prompt:
        bad = set(g) - {"A", "B", "tie"}
        if bad:
            raise ValueError(f"unknown label(s): {sorted(bad)}")
        rows.append((g.count("A"), g.count("A") + g.count("B")))
    wins, nn = np.array(rows, dtype=float).T
    if nn.sum() == 0:
        raise ValueError("no non-tie judgements")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(n_boot, len(rows)))
    w, n = wins[idx].sum(1), nn[idx].sum(1)
    rates = w[n > 0] / n[n > 0]
    q = (1 - ci) / 2
    lo, hi = (float(x) for x in np.quantile(rates, [q, 1 - q]))
    return {"win_rate": float(wins.sum() / nn.sum()), "ci": (lo, hi), "n_prompts": len(rows),
            "n_nontie": int(nn.sum()), "ties": sum(g.count("tie") for g in prefs_by_prompt),
            "tie": margin[0] < lo and hi < margin[1]}

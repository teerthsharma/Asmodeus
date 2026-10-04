import json
import shutil
from pathlib import Path

import pytest

import eval_talk as et

SHEET = Path(__file__).resolve().parent.parent / "ref" / "prompts_50.json"
HASH = "ff9a955c3a864d7fa9b4daca3bbae308b524fddffe65c7182986d892de264c7e"


def test_sheet_hash():
    assert et.PROMPTS_SHA256 == HASH
    assert len(et.load_prompts()) == 50


def test_tampered_sheet_raises(tmp_path):
    p = tmp_path / "p.json"
    shutil.copy(SHEET, p)
    d = json.loads(p.read_text(encoding="utf-8"))
    d["prompts"][0] += "x"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError):
        et.load_prompts(p)
    d["prompts"] = d["prompts"][:49]
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError):
        et.load_prompts(p)


def _blind(seed):
    prompts = [f"p{i}" for i in range(50)]
    a = [f"A-reply-{i}" for i in range(50)]
    b = [f"B-reply-{i}" for i in range(50)]
    items, key = et.blind_pairs(prompts, a, b, seed)
    return prompts, a, b, items, key


def test_blind_deterministic_and_seeded():
    assert _blind(0)[3:] == _blind(0)[3:]
    assert _blind(0)[3] != _blind(1)[3]


def test_blind_each_reply_once_and_roundtrip():
    prompts, a, b, items, key = _blind(0)
    assert len(items) == 50
    seen = [x for it in items for x in (it["left"], it["right"])]
    assert sorted(seen) == sorted(a + b)
    for it in items:
        i = prompts.index(it["prompt"])
        left_model = key[it["id"]]
        assert it["left"] == (a if left_model == "A" else b)[i]
        assert it["right"] == (b if left_model == "A" else a)[i]
    # judge "left" everywhere -> unblinded equals the key
    assert et.unblind({it["id"]: "left" for it in items}, key) == {
        k: v for k, v in key.items()}
    flip = et.unblind({it["id"]: "right" for it in items}, key)
    assert all(flip[k] == ("B" if key[k] == "A" else "A") for k in key)
    assert set(et.unblind({"i00": "tie"}, {"i00": "A"}).values()) == {"tie"}


def test_blind_ids_opaque_and_balanced():
    _, _, _, items, key = _blind(0)
    for it in items:
        assert it["id"] not in ("A", "B") and "A-" not in it["id"]
        assert it["id"].startswith("i")
    assert 15 <= sum(v == "A" for v in key.values()) <= 35
    assert [it["id"] for it in items] == [f"i{n:02d}" for n in range(50)]


def test_blind_length_mismatch():
    with pytest.raises(ValueError):
        et.blind_pairs(["a", "b"], ["x"], ["y", "z"], 0)


def test_rubric():
    assert et.RUBRIC_FIELDS == ("on_topic", "grammatical", "non_repeating")
    et.validate_rubric({"on_topic": 5, "grammatical": 1, "non_repeating": 3})
    for bad in ({"on_topic": 6, "grammatical": 1, "non_repeating": 3},
                {"on_topic": 2.5, "grammatical": 1, "non_repeating": 3},
                {"on_topic": 2, "grammatical": 1}):
        with pytest.raises(ValueError):
            et.validate_rubric(bad)


def test_cohen_kappa():
    assert et.cohen_kappa([1, 0, 1, 2], [1, 0, 1, 2]) == 1.0
    r1 = ["y"] * 25 + ["n"] * 25
    r2 = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    # po = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5 -> 0.4
    assert et.cohen_kappa(r1, r2) == pytest.approx(0.4)


def test_weighted_kappa():
    a = [1, 2, 3, 4, 5, 3, 2]
    b = [2, 2, 3, 5, 5, 1, 2]
    assert et.weighted_kappa(a, a) == 1.0
    assert et.weighted_kappa(a, b) == pytest.approx(et.weighted_kappa(b, a))
    assert et.weighted_kappa(a, b) < 1.0
    # one-off disagreements are penalised less than far ones
    base = [1, 3, 5, 2, 4]
    near = [2, 3, 5, 2, 4]
    far = [5, 3, 5, 2, 4]
    assert et.weighted_kappa(base, near) > et.weighted_kappa(base, far)


def test_krippendorff():
    assert et.krippendorff_alpha([[1, 2, 1, 2], [1, 2, 1, 2]]) == 1.0
    # coincidence matrix: o11=2, o12=o21=1, o22=4, n=8, n1=3, n2=5
    # alpha = 1 - (n-1)*sum_off o / sum_off n_c n_k = 1 - 7*2/30 = 8/15
    d = [[1, 1, 2, 2], [1, 2, 2, 2]]
    assert et.krippendorff_alpha(d, "nominal") == pytest.approx(8 / 15)
    assert et.krippendorff_alpha(d, "interval") == pytest.approx(8 / 15)
    # ordinal with 2 categories reduces to the same thing
    assert et.krippendorff_alpha(d, "ordinal") == pytest.approx(8 / 15)
    # missing values: unit 3 has a single rating and is ignored
    full = [[1, 2, 3], [1, 2, 4]]
    miss = [[1, 2, 3, None], [1, 2, 4, 5]]
    assert et.krippendorff_alpha(miss, "interval") == pytest.approx(
        et.krippendorff_alpha(full, "interval"))
    # interval hand case: units (1,1),(2,2),(3,4): n=6, values 1:2,2:2,3:1,4:1
    # Do-sum = 2*1 = 2 ; De-sum = 2*(4*1+2*4+2*9+2*1+2*4+1*1) = 2*41 = 82
    # alpha = 1 - 5*2/82 = 36/41
    assert et.krippendorff_alpha(full, "interval") == pytest.approx(36 / 41)


def test_proxy():
    rep = et.proxy_scores("The dog", "the dog ran. the dog ran. the dog ran. the dog ran.")
    assert rep["repetition"] > 0.5
    ok = et.proxy_scores("x", "Every morning the small fox quietly crossed a frozen river.")
    assert ok["repetition"] == 0 and ok["distinct_2"] == 1.0
    assert et.proxy_scores("x", "hi there")["repetition"] == 0
    p = "Once upon a time, there was a little girl named Lily. She loved to"
    on = et.proxy_scores(p, "Lily was a little girl who loved to dance.")
    off = et.proxy_scores(p, "The truck drove down a long road.")
    assert on["on_topic_proxy"] > off["on_topic_proxy"]
    good = et.proxy_scores("x", "The cat sat. It was happy!")
    bad = et.proxy_scores("x", "the cat sat sat sat sat it was happy")
    assert good["grammar_proxy"] > bad["grammar_proxy"]
    for v in (rep, ok, on, off, good, bad):
        assert 0 <= v["grammar_proxy"] <= 1 and 0 <= v["on_topic_proxy"] <= 1


def test_tie_test():
    t = et.tie_test(["A"] * 25 + ["B"] * 25)
    assert t["p_two_sided"] == pytest.approx(1.0) and t["tie"] is True
    assert t["n_nontie"] == 50
    t = et.tie_test(["A"] * 40 + ["B"] * 10)
    assert t["p_two_sided"] < 0.001 and t["tie"] is False
    assert t["ci95"][0] > 0.5
    t = et.tie_test(["A"] * 9 + ["B"] + ["tie"] * 5)
    assert t["p_two_sided"] == pytest.approx(22 / 1024)
    assert (t["wins_a"], t["wins_b"], t["ties"], t["n_nontie"]) == (9, 1, 5, 10)
    lo, hi = t["ci95"]
    assert lo < 0.9 < hi
    assert et.tie_test(["tie"] * 3)["p_two_sided"] == 1.0

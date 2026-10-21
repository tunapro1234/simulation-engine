import math
import re
import statistics
import unittest

import numpy as np

import simengine.persona as sp
from simengine.llm import FakeLLM
from simengine.persona import bigfive as bf
from simengine.persona import validate as va


class TestSentences(unittest.TestCase):
    def test_integer_levels_give_three_sentences(self):
        for d in bf.DOMAINS:
            for lv in range(-3, 4):
                self.assertEqual(bf.blend_sentences(d, lv), bf.SENTENCES[d][lv])
                self.assertEqual(len(bf.blend_sentences(d, lv)), 3)

    def test_fractional_blend_count(self):
        for w in (0.2, 0.5, 1.4, -1.7, 2.9, -0.34):
            lo = math.floor(w)
            n_hi = round((w - lo) * 3)
            got = bf.blend_sentences("agreeableness", w)
            self.assertEqual(len(got), 3)
            self.assertEqual(sum(s in bf.SENTENCES["agreeableness"][lo + 1] for s in got), n_hi, w)
            self.assertEqual(sum(s in bf.SENTENCES["agreeableness"][lo] for s in got), 3 - n_hi, w)
        self.assertEqual(bf.blend_sentences("openness", 7), bf.SENTENCES["openness"][3])   # clipped

    def test_score_mapping_and_text(self):
        self.assertEqual(bf.score_to_weight(5), 3.0)
        self.assertEqual(bf.score_to_weight(1), -3.0)
        self.assertAlmostEqual(bf.weight_to_score(bf.score_to_weight(3.7)), 3.7)
        t = bf.bfi2_text({"agreeableness": 5.0, "neuroticism": 1.0})
        lines = t.split("\n")
        self.assertEqual(len(lines), 15)
        self.assertEqual(lines[3:6], bf.SENTENCES["agreeableness"][3])
        self.assertEqual(lines[9:12], bf.SENTENCES["neuroticism"][-3])
        self.assertEqual(lines[0:3], bf.SENTENCES["extraversion"][0])
        with self.assertRaises(ValueError):
            bf.bfi2_text({"honesty": 3})
        self.assertAlmostEqual(bf.profile_scores("cooperative")["agreeableness"], 4.333, places=3)

    def test_traits(self):
        tr = sp.bigfive_traits({"openness": 4.123})
        self.assertEqual([t.label for t in tr], bf.DOMAINS)
        o = tr[-1]
        self.assertIsInstance(o, sp.Trait)
        self.assertEqual((o.value, o.group, o.extra["level"]), (4.12, "big_five", "high"))
        self.assertEqual(o.to_dict()["scale"], "1-5")


class TestSampling(unittest.TestCase):
    def test_marginals_match_norms(self):
        people = sp.sample_bigfive(5000, seed=3)
        for d, (mean, sd) in bf.US_ADULT_NORMS.items():
            xs = [p["scores"][d] for p in people]
            self.assertAlmostEqual(statistics.fmean(xs), mean, delta=0.05, msg=d)
            self.assertAlmostEqual(statistics.pstdev(xs), sd, delta=0.05, msg=d)
            self.assertTrue(all(1 <= x <= 5 for x in xs))

    def test_overrides_and_sd_scale(self):
        low = sp.sample_bigfive(2000, seed=1, overrides={"agreeableness": 2.0})
        self.assertAlmostEqual(statistics.fmean(p["scores"]["agreeableness"] for p in low), 2.0, delta=0.06)
        self.assertAlmostEqual(statistics.fmean(p["scores"]["openness"] for p in low), 3.6, delta=0.06)
        fixed = sp.sample_bigfive(5, seed=1, sd_scale=0.0, overrides={"neuroticism": 1.0})
        self.assertTrue(all(p["scores"]["neuroticism"] == 1.0 for p in fixed))
        self.assertEqual(len({p["text"] for p in fixed}), 1)
        wide = sp.sample_bigfive(2000, seed=1, sd_scale=1.5)
        narrow = sp.sample_bigfive(2000, seed=1, sd_scale=0.5)
        sw = statistics.pstdev(p["scores"]["extraversion"] for p in wide)
        sn = statistics.pstdev(p["scores"]["extraversion"] for p in narrow)
        self.assertAlmostEqual(sn, 0.375, delta=0.03)
        self.assertGreater(sw, 2 * sn)
        with self.assertRaises(ValueError):
            sp.sample_bigfive(2, overrides={"grit": 3})

    def test_persona_dict_and_determinism(self):
        a = sp.sample_bigfive(3, seed=9)
        self.assertEqual(a, sp.sample_bigfive(3, seed=9))
        self.assertNotEqual(a[0]["scores"], sp.sample_bigfive(3, seed=10)[0]["scores"])
        p = a[0]
        self.assertEqual(p["id"], "bfi2-9-0")
        self.assertEqual(p["persona_method"], "bfi2")
        self.assertEqual(p["text"], bf.bfi2_text(p["scores"]))
        self.assertEqual(p["params"]["scores"], p["scores"])

    def test_correlated_sampling(self):
        people = sp.sample_bigfive(5000, seed=2, norms=bf.SOTO_JOHN_2017_NORMS, corr=bf.SOTO_JOHN_2017_CORR)
        x = np.array([[p["scores"][d] for d in bf.DOMAINS] for p in people])
        r = np.corrcoef(x, rowvar=False)
        e, n = bf.DOMAINS.index("extraversion"), bf.DOMAINS.index("neuroticism")
        self.assertAlmostEqual(r[e, n], -0.34, delta=0.05)
        a, c = bf.DOMAINS.index("agreeableness"), bf.DOMAINS.index("conscientiousness")
        self.assertAlmostEqual(r[a, c], 0.28, delta=0.05)

    def test_audience_strength_knob(self):
        spec = sp.AudienceSpec(target_description="test", audience=sp.Audience(
            income_tier=["low", "middle"], trait_requirements=[{"dimension": "agreeableness", "values": ["high"]}]))
        flat = sp.bigfive_audience(spec, 10, seed=0, strength=0.0)
        self.assertTrue(all(all(v == 3.0 for v in p["scores"].values()) for p in flat))
        self.assertTrue(all("sometimes is compassionate, but not always" in p["text"] for p in flat))
        full = sp.bigfive_audience(spec, 2000, seed=0, strength=1.0)
        mean_a = statistics.fmean(p["scores"]["agreeableness"] for p in full)
        self.assertAlmostEqual(mean_a, bf.weight_to_score(2), delta=0.08)
        self.assertEqual(collections_count(p["income_tier"] for p in full), {"low": 1000, "middle": 1000})
        p = full[0]
        self.assertEqual(p["segment"], p["income_tier"])
        self.assertTrue(p["text"].split("\n")[0].startswith("age "))
        self.assertIn(f"income_tier: {p['income_tier']}", p["text"].split("\n")[0])
        strong = sp.bigfive_audience(spec, 2000, seed=0, strength=2.0, targets="selfish")
        self.assertLess(statistics.fmean(q["scores"]["agreeableness"] for q in strong), 1.6)
        from simengine.sim.personas import as_persona
        self.assertIn(bf.SENTENCES["openness"][0][0][:20], as_persona(full[0]).text() + bf.SENTENCES["openness"][0][0])


def collections_count(it):
    out = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


# ---------------------------------------------------------------- Mini-IPIP + validation
ITEM_BY_TEXT = {it["text"]: it for it in va.MINI_IPIP}


def _item(user):
    return ITEM_BY_TEXT[re.search(r'"(.*)"', user).group(1)]


def tagged(scores):
    """Builder whose text is unique per agent (BFI-2 sentences are quantized, so two agents can share a text)."""
    return bf.bfi2_text(scores) + "\n#" + repr(sorted(scores.items()))


def truth_responder(n, seed, shift=None, sample_kw=None, fmt="{:.4f}", builder=None):
    """FakeLLM responder that answers each item with the persona's assigned domain score (+shift)."""
    people = sp.sample_bigfive(n, seed=seed, **(sample_kw or {}))
    by_sys = {va.system_prompt((builder or bf.bfi2_text)(p["scores"])): p["scores"] for p in people}
    if builder is None:
        assert len(by_sys) == n, "duplicate persona texts: use builder=tagged"
    shift = shift or {}

    def respond(system, user):
        scores = by_sys[system]
        it = _item(user)
        x = scores[it["domain"]] + shift.get(it["domain"], 0.0)
        return fmt.format(6 - x if it["reverse"] else x)
    return respond


class TestScoring(unittest.TestCase):
    def test_parse_likert(self):
        cases = {"4": 4.0, " 3.5 ": 3.5, "4 - Moderately accurate": 4.0, "5 Very accurate": 5.0,
                 '{"answer": 2}': 2.0, "```json\n{\"rating\": \"1\"}\n```": 1.0, '{"a": 1, "b": 2}': None, "I would say 4.": None,
                 "": None, None: None, "7": None, "0": None, "1 to 5, I'd say 4": None, "maybe": None,
                 "[3]": 3.0, "2/5": 2.0, "I would say 7 or 4": None,
                 "4 out of 5": 4.0, "Answer: 3 out of 5.": 3.0, "4.": 4.0,
                 "Not 5": None, "I would never pick 1": None, "Definitely not a 2.": None, "never 1": None,
                 "5. Actually no, 2": None, "3 (Neutral)": 3.0}
        for text, want in cases.items():
            self.assertEqual(va.parse_likert(text), want, repr(text))
        self.assertEqual(va.parse_likert(4), 4.0)

    def test_score_reverse_and_missing(self):
        ans = {f"ipip_{it['id']}": (2 if it["reverse"] else 4) for it in va.MINI_IPIP}
        s = sp.score_mini_ipip(ans)
        self.assertTrue(all(v == 4.0 for v in s.values()))
        ans.pop("ipip_1")
        s = va.score_mini_ipip(ans)
        self.assertTrue(math.isnan(s["extraversion"]))
        self.assertEqual(s["agreeableness"], 4.0)
        self.assertAlmostEqual(va.score_mini_ipip(ans, min_items=3)["extraversion"], 4.0)
        bad = {it["id"]: "" for it in va.MINI_IPIP}
        self.assertTrue(all(math.isnan(v) for v in va.score_mini_ipip(bad).values()))

    def test_compute_metrics(self):
        m = va.compute_metrics([1, 2, 3], [2, 3, 4])
        self.assertAlmostEqual(m["r"], 1.0)
        self.assertAlmostEqual(m["bias"], 1.0)
        self.assertAlmostEqual(m["mae"], 1.0)
        self.assertTrue(math.isnan(va.compute_metrics([1], [2])["r"]))
        self.assertEqual(va.compute_metrics([], [])["n"], 0)


class TestValidate(unittest.TestCase):
    def test_truthful_answers_give_r1_bias0(self):
        llm = FakeLLM(truth_responder(12, 42))
        res = sp.validate_bigfive(llm, n=12, seed=42)
        self.assertEqual(len(llm.calls), 12 * 20)
        self.assertAlmostEqual(res["pooled"]["r"], 1.0, places=6)
        self.assertAlmostEqual(res["pooled"]["bias"], 0.0, places=3)
        for d in bf.DOMAINS:
            self.assertAlmostEqual(res["metrics"][d]["r"], 1.0, places=4)
            self.assertAlmostEqual(res["metrics"][d]["bias"], 0.0, places=3)
        self.assertEqual((res["n_pairs"], res["n_missing"], res["n_missing_pairs"]), (60, 0, 0))
        sys_p, user = llm.calls[0]
        self.assertTrue(sys_p.startswith(va.VALIDATION_INSTRUCTION))
        self.assertIn("Am the life of the party", user)
        row = res["rows"][0]
        for k in ("persona_method", "agent_id", "seed", "params", "max_tokens", "wording_id", "valid"):
            self.assertIn(k, row)
        self.assertEqual(row["persona_method"], "bfi2")

    def test_integer_answers_still_track(self):
        res = sp.validate_bigfive(FakeLLM(truth_responder(20, 42, fmt="{:.0f}")), n=20, seed=42)
        self.assertGreater(res["pooled"]["r"], 0.9)

    def test_shift_on_agreeableness(self):
        kw = {"overrides": {"agreeableness": 3.0}, "sd_scale": 0.4}
        llm = FakeLLM(truth_responder(15, 7, shift={"agreeableness": 1.0}, sample_kw=kw, builder=tagged))
        res = sp.validate_bigfive(llm, n=15, seed=7, sample_kw=kw, builder=tagged)
        self.assertEqual(res["builder"], "tagged")
        self.assertAlmostEqual(res["metrics"]["agreeableness"]["bias"], 1.0, places=3)
        self.assertAlmostEqual(res["metrics"]["agreeableness"]["r"], 1.0, places=4)
        self.assertAlmostEqual(res["metrics"]["extraversion"]["bias"], 0.0, places=3)
        self.assertAlmostEqual(res["pooled"]["bias"], 0.2, places=3)

    def test_empty_answers_are_missing_not_imputed(self):
        res = sp.validate_bigfive(FakeLLM(lambda s, u: ""), n=4, seed=1)
        self.assertEqual(res["n_missing"], 80)
        self.assertEqual(res["n_missing_pairs"], 20)
        self.assertEqual(res["n_pairs"], 0)
        self.assertTrue(math.isnan(res["pooled"]["r"]))
        self.assertTrue(all(p["measured"] is None for p in res["paired"]))
        self.assertTrue(all(r["value"] is None and not r["valid"] for r in res["rows"]))
        card = sp.to_card_row(res)
        self.assertEqual(card["completion"], 0.0)

    def test_partial_missing(self):
        base = truth_responder(6, 3)

        def respond(system, user):
            return "" if _item(user)["id"] == 1 else base(system, user)
        res = sp.validate_bigfive(FakeLLM(respond), n=6, seed=3)
        self.assertEqual(res["n_missing"], 6)
        self.assertEqual(res["n_missing_pairs"], 6)
        self.assertEqual(res["metrics"]["extraversion"]["n"], 0)
        self.assertEqual(res["metrics"]["openness"]["n"], 6)
        self.assertEqual(res["n_pairs"], 24)

    def test_big5_scaler_builder(self):
        from simengine.persona.methods import big5
        from simengine.persona.methods.helpers import as_rng
        rng = as_rng(5)
        dials = [big5.SAMPLER.dials(rng) for _ in range(8)]
        by_sys = {va.system_prompt(big5.render(d)): {t: big5.dial_to_score(d[t]) for t in big5.TRAITS} for d in dials}

        def respond(system, user):
            it = _item(user)
            x = by_sys[system][it["domain"]]
            return f"{6 - x if it['reverse'] else x:.3f}"
        res = sp.validate_bigfive(FakeLLM(respond), n=8, seed=5, builder="big5_scaler")
        self.assertEqual(res["builder"], "big5_scaler")
        self.assertAlmostEqual(res["pooled"]["bias"], 0.0, places=3)
        self.assertGreater(res["pooled"]["r"], 0.999)

    def test_card_row(self):
        res = sp.validate_bigfive(FakeLLM(truth_responder(5, 0)), n=5, seed=0)
        row = sp.to_card_row(res)
        for k in ("model", "builder", "pooled_r", "pooled_bias", "agreeableness_bias", "neuroticism_r", "n_pairs",
                  "n_missing", "completion"):
            self.assertIn(k, row)
        self.assertEqual(row["model"], "fake")
        self.assertEqual(row["completion"], 1.0)
        self.assertFalse(any(isinstance(v, (dict, list)) for v in row.values()))

    def test_unknown_builder(self):
        with self.assertRaises(ValueError):
            sp.validate_bigfive(FakeLLM(), n=1, builder="nope")


if __name__ == "__main__":
    unittest.main()

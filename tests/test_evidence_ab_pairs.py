"""evidence.ab: A/B pair table -> variant frame (G0b #3, G3 #2) and the A/B memorization probe (G0b #7, G3 #6)."""
import hashlib
import json
import re
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine.evidence as ev
from simengine.llm import FakeLLM

LINE = re.compile(r'^(\d+)\. A: "(.*)" \| B: "(.*)"$', re.M)
NAMED = "Headline tests from the Upworthy Research Archive (Matias et al. 2021)."
NEUTRAL = "A/B test of two headlines shown to real users; outcome = click-through rate."


def _pairs(n_tests=60, per_test=1, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tests):
        for k in range(per_test):
            rows.append({"test_id": f"t{t}", "headline_a": f"story {t} variant {k} first",
                         "headline_b": f"story {t} variant {k} second", "y": int(rng.integers(0, 2))})
    return pd.DataFrame(rows)


def _ab_responder(winners: set, skill: float = 0.6):
    """LLMDirectAB prompt -> JSON. Named context: 'remembers' the winner (p 0.9 / 0.1). Otherwise a weak, stable
    guess that is right on a `skill` share of texts (taste, the same in every arm)."""
    def resp(system, user):
        named = "Upworthy" in user.split("\n", 1)[0]
        out = []
        for i, a, b in LINE.findall(user):
            if named and (a in winners or b in winners):
                p = 0.9 if a in winners else 0.1
            else:
                h = int(hashlib.sha256((a + "|" + b).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                h2 = int(hashlib.sha256((min(a, b)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                right = h2 < skill
                a_true = a in winners
                p = (0.7 if a_true == right else 0.3) if (a in winners or b in winners) else 0.4 + 0.2 * h
            out.append({"i": int(i), "p_a": p})
        return json.dumps(out)
    return resp


def _winners(pairs):
    return {r.headline_a if r.y == 1 else r.headline_b for r in pairs.itertuples()}


class PairsFrameTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.n = 200
        y = rng.integers(0, 2, self.n)
        good = rng.random(self.n) < 0.72
        p = np.where(good == (y == 1), 0.7, 0.3)
        p[:25] = 0.5
        self.pairs = pd.DataFrame({"test_id": np.arange(self.n) // 2, "y": y, "p_llm": p,
                                   "p_coin": rng.random(self.n), "split": "test"})
        self.direct = float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == (y == 1))))

    def test_shape_and_rows(self):
        f = ev.ab_pairs_frame(self.pairs, ["p_llm", "p_coin"], keep=["split"])
        self.assertEqual(len(f), 2 * self.n)
        self.assertEqual(list(f.columns), ["unit", "item", "group", "arm", "truth", "pred_llm", "pred_coin", "split"])
        a, b = f.iloc[0], f.iloc[1]
        self.assertEqual((a.arm, b.arm), ("A", "B"))
        self.assertEqual(a["item"], b["item"])
        self.assertAlmostEqual(a.truth + b.truth, 1.0)
        self.assertAlmostEqual(a.pred_llm + b.pred_llm, 1.0)
        self.assertEqual(f["item"].nunique(), self.n)        # item = pair, so two pairs of one test never cross
        self.assertEqual(f["group"].nunique(), self.n // 2)  # cluster = test

    def test_pair_sign_equals_direct_accuracy(self):
        f = ev.ab_pairs_frame(self.pairs, {"llm": "p_llm", "coin": self.pairs.p_coin.to_numpy()})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = ev.summary(ev.evaluate(f, task="ab", ours="llm", raw="coin", B=200, B_aggregate=50))
        self.assertEqual(s["primary_metric"], "pair_sign")
        self.assertAlmostEqual(s["ours"]["value"], self.direct, places=9)

    def test_ci_is_binomial_width_one_cluster_per_pair(self):
        f = ev.ab_pairs_frame(self.pairs, ["p_llm", "p_coin"], group=None)
        self.assertNotIn("group", f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o = ev.summary(ev.evaluate(f, task="ab", ours="llm", raw="coin", B=200, B_aggregate=100))["ours"]
        self.assertAlmostEqual(o["value"], self.direct, places=9)
        binom = 2 * 1.96 * np.sqrt(self.direct * (1 - self.direct) / self.n)
        self.assertLess(abs((o["hi"] - o["lo"]) / binom - 1), 0.3)

    def test_labels_ties_and_errors(self):
        d = pd.DataFrame({"test_id": [1, 1, 2, 3], "winner": ["a", "b", "tie", "a"], "p": [0.8, 0.6, 0.5, 0.5]})
        f = ev.ab_pairs_frame(d, "p", y="winner", a_wins="a")
        self.assertEqual(len(f), 6)                           # the tie is dropped
        self.assertEqual(f.truth.tolist(), [1, 0, 0, 1, 1, 0])
        self.assertEqual(len(ev.ab_pairs_frame(d, "p", y="winner", a_wins="a", drop_ties=False)), 8)
        with self.assertRaises(ValueError):
            ev.ab_pairs_frame(d.assign(y=[1, 2, 0, 1]), "p")
        with self.assertRaises(ValueError):
            ev.ab_pairs_frame(d.assign(y=1, p=[1.2, 0, 0, 0]), "p")
        with self.assertRaises(KeyError):
            ev.ab_pairs_frame(d.assign(y=1), "p", group="test")


    def test_label_case_and_unknown_labels(self):
        d = pd.DataFrame({"test_id": [1, 2, 3, 4], "winner": ["A", "B", "Tie", "a"], "p": [0.8, 0.6, 0.5, 0.4]})
        f = ev.ab_pairs_frame(d, "p", y="winner", a_wins="a")      # "A" == "a": not silently a B win
        self.assertEqual(f.truth.tolist(), [1, 0, 0, 1, 1, 0])
        f2 = ev.ab_pairs_frame(d, "p", y="winner")                 # default labels a / b / tie
        self.assertEqual(f2.truth.tolist(), f.truth.tolist())
        with self.assertRaises(ValueError):                         # "1"/"0" strings would all read as B won
            ev.ab_pairs_frame(d.assign(winner=["1", "0", "1", "0"]), "p", y="winner")


class MemorizationTest(unittest.TestCase):
    def test_items(self):
        pairs = _pairs(40)
        items = ev.ab_memorization_items(pairs, n=20, seed=1)
        real = [o for o in items if not o["control"]]
        ctrl = [o for o in items if o["control"]]
        self.assertEqual((len(real), len(ctrl)), (20, 20))
        win = _winners(pairs)
        self.assertTrue(all(o["expected"] in win for o in real))
        for o in ctrl:                                       # control texts come from two different tests
            ta, tb = re.findall(r'^[AB]: "story (\d+) ', o["prompt"], re.M)
            self.assertNotEqual(ta, tb)
        self.assertEqual(items, ev.ab_memorization_items(pairs, n=20, seed=1))
        # a model that knows the winners: memorization_audit (completion gap) flags it
        llm = FakeLLM(lambda s, u: next((w for w in win if f'"{w}"' in u), "none"))
        r = ev.memorization_audit(items, llm, B=300)
        self.assertEqual(r["durum"], "uyari")

    def test_named_arm_remembers(self):
        pairs = _pairs(60)
        llm = FakeLLM(_ab_responder(_winners(pairs)))
        r = ev.ab_memorization(llm, pairs, NAMED, NEUTRAL, n=None, B=400)
        self.assertEqual(r["durum"], "uyari")
        self.assertGreater(r["deger"], 0.2)
        self.assertGreater(r["alt"], 0)
        self.assertEqual(r["accuracy_named"], 1.0)
        self.assertAlmostEqual(r["deger"], r["accuracy_named"] - r["accuracy_neutral"])
        self.assertEqual((r["n_pairs"], r["n_tests"], r["esik"]), (60, 60, 0.0))
        self.assertEqual(r["calls"], len(llm.calls))
        self.assertIn("adlı − nötr", r["ayrinti"])
        # the dict is what evaluate(memorization=...) reads
        f = ev.ab_pairs_frame(pairs.assign(p=0.6), "p", group="test_id")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = ev.evaluate(f, task="ab", ours="p", memorization=r, B=200, B_aggregate=50)
        aud = [a for a in ev.summary(rep)["audits"] if "ezber" in a["check"]]
        self.assertEqual(aud[0]["status"], "warning")

    def test_honest_model_no_gap(self):
        pairs = _pairs(60)
        llm = FakeLLM(_ab_responder(set()))                 # knows nothing: both arms the same
        r = ev.ab_memorization(llm, pairs, NAMED, NEUTRAL, n=40, B=300)
        self.assertEqual(r["durum"], "tamam")
        self.assertEqual(r["deger"], 0.0)
        self.assertEqual(r["n_pairs"], 40)

    def test_paraphrase_arm_and_limits(self):
        pairs = _pairs(30, per_test=2)                     # two pairs per test: cluster = test
        llm = FakeLLM(_ab_responder(_winners(pairs)))
        r = ev.ab_memorization(llm, pairs, NAMED, NEUTRAL, n=None, B=300, paraphrase=str.upper)
        self.assertEqual(r["n_tests"], 30)
        self.assertLess(r["accuracy_paraphrase"], r["accuracy_named"])
        self.assertGreater(r["paraphrase_gap_ci"][0], 0)
        self.assertEqual(r["durum"], "uyari")
        n0 = len(llm.calls)
        with self.assertRaises(ValueError):
            ev.ab_memorization(llm, pairs, NAMED, NEUTRAL, n=None, max_calls=5)
        self.assertEqual(len(llm.calls), n0)                # checked before any call
        self.assertEqual(ev.ab_memorization(llm, pairs.head(3), NAMED, NEUTRAL)["durum"], "atlandi")

    def test_items_gap_is_taste_not_memory(self):
        pairs = _pairs(80)
        win = _winners(pairs)
        taste = FakeLLM(lambda s, u: next((w for w in win if f'"{w}"' in u), "none")
                        if int(hashlib.sha256(u.encode()).hexdigest()[:8], 16) % 10 < 7
                        else re.findall(r'^[AB]: "(.*)"$', u, re.M)[0])
        # a model with taste and no memory (same answer in either context) beats the coin-flip controls ...
        items = ev.ab_memorization_items(pairs, n=None, seed=0)
        self.assertEqual(ev.memorization_audit(items, taste, B=300)["durum"], "uyari")
        # ... while the named - neutral probe, which cancels taste, has nothing to flag
        llm = FakeLLM(_ab_responder(win, skill=0.75))              # the responder only "remembers" on "Upworthy"
        r = ev.ab_memorization(llm, pairs, "Headline tests from a public archive (Matias et al.).", NEUTRAL,
                               n=None, B=300)
        self.assertGreater(r["accuracy_neutral"], 0.6)
        self.assertEqual((r["deger"], r["durum"]), (0.0, "tamam"))

    def test_paraphrase_share(self):
        pairs = _pairs(20)
        llm = FakeLLM(_ab_responder(set()))
        r = ev.ab_memorization(llm, pairs, NAMED, NEUTRAL, n=None, B=200, paraphrase=lambda t: t)
        self.assertEqual(r["paraphrased_share"], 0.0)               # nothing rewritten: paraphrase_gap is not evidence
        self.assertIn("yeniden yazıldı", r["ayrinti"])


if __name__ == "__main__":
    unittest.main()

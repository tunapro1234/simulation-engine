"""simengine.evidence.why_match: N-way 'why' recognition (generated reason vs N real candidate reasons).
Synthetic texts only; the default TF-IDF encoder is pure numpy and a toy callable stands in for a sentence model."""
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine.evidence as ev
from simengine.evidence.why import RR_PREFIX, TfidfEncoder, tfidf_encoder, why_match, why_summary

TOPICS = ["battery life is long", "the screen is too dim", "shipping took two weeks", "great value for the price",
          "the strap broke quickly", "sound quality is clear", "setup was confusing", "it runs very quiet"]


def _panel(n=80, N=4, seed=0):
    """Each person's true reason is one topic; distractors are other topics; the 'good' arm paraphrases the truth,
    the 'blind' arm writes text sharing nothing with any candidate."""
    rng = np.random.default_rng(seed)
    cands, truth, good, blind = [], [], [], []
    for _ in range(n):
        pick = rng.choice(len(TOPICS), N, replace=False)
        t = int(rng.integers(N))
        cands.append([TOPICS[k] for k in pick])
        truth.append(t)
        good.append("honestly " + TOPICS[pick[t]] + " for me")
        blind.append("zq xv")                       # shares no word or char n-gram with any candidate
    return cands, truth, good, blind


class TestWhyMatch(unittest.TestCase):
    def test_frame_columns_and_recognition(self):
        cands, truth, good, blind = _panel()
        f = why_match({"good": good, "blind": blind}, cands, truth, ids=[f"p{i}" for i in range(len(truth))])
        for c in ("unit", "group", "truth", "n_candidates", "chance", "pred_good", "pred_blind",
                  RR_PREFIX + "good", RR_PREFIX + "blind"):
            self.assertIn(c, f.columns)
        self.assertTrue((f["pred_good"] == f["truth"]).all())
        self.assertTrue(np.allclose(f["rr_good"], 1.0))
        s = why_summary(f)
        self.assertAlmostEqual(s["chance_accuracy"], 0.25)
        self.assertAlmostEqual(s["chance_mrr"], (1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)
        self.assertEqual(s["arms"]["good"]["accuracy"], 1.0)
        # a text with no overlap scores chance on average (random tie-break, average-rank RR), not candidate 0
        self.assertAlmostEqual(s["arms"]["blind"]["mrr"], s["chance_mrr"], places=6)
        self.assertGreater(f["pred_blind"].nunique(), 1)
        self.assertEqual(f.attrs["why"]["encoder"], "tfidf")

    def test_tie_rr_and_seed(self):
        f1 = why_match(["", ""], [["a b", "c d", "e f"]] * 2, 1, seed=3)
        f2 = why_match(["", ""], [["a b", "c d", "e f"]] * 2, 1, seed=3)
        pd.testing.assert_frame_equal(f1, f2)
        self.assertTrue(np.allclose(f1["rr_gen"], (1 + 1 / 2 + 1 / 3) / 3))

    def test_partial_rank(self):
        cands = [["red apples taste sweet", "red cars drive fast", "blue sky at noon"]]
        f = why_match({"x": ["red apples"]}, cands, [1])          # truth is 2nd best
        self.assertEqual(int(f["pred_x"][0]), 0)
        self.assertAlmostEqual(float(f["rr_x"][0]), 0.5)

    def test_callable_encoder_and_embed_object(self):
        def toy(texts):                                           # bag of letters, no model download
            return np.array([[t.count(ch) for ch in "abcdefghijklmnopqrstuvwxyz"] for t in texts], float)

        class Model:
            def embed(self, texts):
                return toy(texts)
        cands, truth, good, _ = _panel(n=30, seed=2)
        f = why_match({"good": good}, cands, truth, encoder=toy)
        g = why_match({"good": good}, cands, truth, encoder=Model())
        pd.testing.assert_frame_equal(f, g)
        self.assertGreater(why_summary(f)["arms"]["good"]["accuracy"], 0.25)

    def test_tfidf_encoder(self):
        enc = tfidf_encoder(["the cat sat", "a dog ran"])
        self.assertIsInstance(enc, TfidfEncoder)
        E = enc(["the cat sat", "the cat sat", "zzz"])
        self.assertAlmostEqual(float(E[0] @ E[1]), 1.0)          # unit norm
        self.assertEqual(float(np.abs(E[2]).sum()), 0.0)          # unseen terms ignored
        f = why_match({"a": ["cat"]}, [["the cat sat", "a dog ran"]], 0, encoder=enc)
        self.assertEqual(int(f["pred_a"][0]), 0)

    def test_dataframe_arms_variable_n(self):
        gen = pd.DataFrame({"m1": ["dim screen", "slow shipping"], "m2": ["price", "price"]})
        cands = [["the screen is too dim", "great value for the price"],
                 ["shipping took two weeks", "great value for the price", "setup was confusing"]]
        f = why_match(gen, cands, [0, 0])
        self.assertEqual(list(f["n_candidates"]), [2, 3])
        self.assertTrue(np.allclose(f["chance"], [1 / 2, 1 / 3]))
        self.assertEqual(list(f["pred_m1"]), [0, 0])
        self.assertEqual(list(f["pred_m2"]), [1, 1])

    def test_repeated_ids_groups_and_constant_truth(self):
        cands, truth, good, _ = _panel(n=12, seed=6)
        person = [f"p{i // 3}" for i in range(12)]                  # person x product: 3 rows per person
        f = why_match({"good": good}, cands, truth, ids=person)
        self.assertTrue(f["unit"].is_unique)
        self.assertEqual(list(f["group"]), person)
        g = why_match({"good": good}, cands, truth, ids=list(range(12)), groups=person)
        self.assertEqual(list(g["unit"]), list(range(12)))
        self.assertEqual(list(g["group"]), person)
        with self.assertWarns(UserWarning):
            why_match({"good": good}, cands, 0)
        with self.assertRaises(ValueError):
            why_match({"good": good}, cands, truth, groups=person[:3])

    def test_errors(self):
        with self.assertRaises(ValueError):
            why_match(["a"], [["x", "y"], ["x", "y"]], 0)
        with self.assertRaises(ValueError):
            why_match(["a"], [["x"]], 0)
        with self.assertRaises(ValueError):
            why_match(["a"], [["x", "y"]], 2)
        with self.assertRaises(ValueError):
            why_match(["a"], [["x", "y"]], 0, encoder="bge")

    def test_feeds_evaluate(self):
        cands, truth, good, blind = _panel(n=60, seed=4)
        f = why_match({"good": good, "blind": blind}, cands, truth)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ev.evaluate(f, task="multiclass", primary="accuracy", ours="good", raw="blind", B=100)
        s = ev.summary(r)
        self.assertIsInstance(s, dict)

    def test_exported(self):
        self.assertIs(ev.why_match, why_match)
        self.assertIs(ev.why_summary, why_summary)


if __name__ == "__main__":
    unittest.main()

"""Tests for simengine.evidence.extras.say_do (synthetic data only, offline)."""
import json
import unittest

import numpy as np
import pandas as pd

from simengine.evidence import extras


def _frame(seed=0, persons=150, items=2, gap=1.0, noise=0.3):
    """Persons say ~gap more than they do; pred_say tracks say, pred_do tracks do, pred_mid sits halfway."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(persons):
        base = rng.normal(3, 1)
        for k in range(items):
            do = base + rng.normal(0, 0.5)
            say = do + gap + rng.normal(0, 0.2)
            rows.append({"person": f"p{i}", "item": f"q{k}", "said": say, "did": do,
                         "pred_say": say + rng.normal(0, noise), "pred_do": do + rng.normal(0, noise),
                         "pred_mid": (say + do) / 2 + rng.normal(0, noise)})
    return pd.DataFrame(rows)


def _by(sec, col):
    return next(r for r in sec["predictions"] if r["prediction"] == col)


class SayDo(unittest.TestCase):
    def test_verdicts(self):
        sec = extras.say_do(_frame(), "said", "did", preds=["say", "pred_do", "mid"], B=300)
        self.assertEqual(sec["kind"], "say_do")
        self.assertTrue(sec["anchors"]["apart"])
        self.assertAlmostEqual(sec["anchors"]["gap"], -1.0, delta=0.1)
        s, d, m = _by(sec, "pred_say"), _by(sec, "pred_do"), _by(sec, "pred_mid")
        self.assertEqual(s["verdict"], extras.SAY)
        self.assertLess(s["hi"], 0)
        self.assertLess(s["mae_hi"], 0)
        self.assertEqual(d["verdict"], extras.DO)
        self.assertGreater(d["lo"], 0)
        self.assertEqual(m["verdict"], extras.UNDECIDED)
        self.assertLess(m["lo"], 0)
        self.assertGreater(m["hi"], 0)
        self.assertIn("closer to what people say", s["card_line"])
        self.assertEqual(sec["n_clusters"], 150)
        self.assertEqual(len(sec["rows"]), 4)           # anchors + 3 predictions
        json.dumps(sec)                                  # JSON-friendly

    def test_point_values(self):
        f = pd.DataFrame({"person": ["a", "a", "b", "b"], "said": [4.0, 4, 2, 2], "did": [1.0, 1, 1, 1],
                          "pred_x": [3.0, 3, 2, 2]})
        r = _by(extras.say_do(f, "said", "did", B=50), "pred_x")
        self.assertAlmostEqual(r["mean"], 2.5)
        self.assertAlmostEqual(r["dist_say"], 0.5)
        self.assertAlmostEqual(r["dist_do"], 1.5)
        self.assertAlmostEqual(r["diff"], -1.0)
        self.assertAlmostEqual(r["mae_say"], 0.5)
        self.assertAlmostEqual(r["mae_do"], 1.5)
        self.assertEqual(r["clusters"], 2)

    def test_default_preds_seed_and_on_mae(self):
        f = _frame(persons=60)
        a = extras.say_do(f, "said", "did", B=200, seed=3)
        b = extras.say_do(f, "said", "did", B=200, seed=3)
        self.assertEqual([r["prediction"] for r in a["predictions"]], ["pred_say", "pred_do", "pred_mid"])
        self.assertEqual(a["rows"], b["rows"])
        c = extras.say_do(f, "said", "did", preds="say", on="mae", B=200)
        self.assertEqual(c["on"], "mae")
        self.assertEqual(c["predictions"][0]["verdict"], extras.SAY)

    def test_anchors_not_apart_is_undecided(self):
        sec = extras.say_do(_frame(gap=0.0, noise=0.05), "said", "did", B=300)
        self.assertFalse(sec["anchors"]["apart"])
        self.assertTrue(all(r["verdict"] == extras.UNDECIDED for r in sec["predictions"]))
        self.assertIn("not apart", " ".join(sec["notes"]))

    def test_cluster_bootstrap_is_wider_than_rows(self):
        f = _frame(persons=40, items=6, noise=0.0)
        shift = dict(zip(f["person"].unique(), np.random.default_rng(1).normal(0, 1, 40)))
        f["pred_y"] = f["did"] + 0.5 + f["person"].map(shift)              # person-level error: rows tied
        tied = _by(extras.say_do(f, "said", "did", preds=["y"], B=400), "pred_y")
        rows = _by(extras.say_do(f, "said", "did", preds=["y"], cluster=None, B=400), "pred_y")
        self.assertGreater(tied["hi"] - tied["lo"], rows["hi"] - rows["lo"])

    def test_cluster_falls_back_to_group(self):
        f = _frame(persons=40, items=6, noise=0.0)
        shift = dict(zip(f["person"].unique(), np.random.default_rng(1).normal(0, 1, 40)))
        f["pred_y"] = f["did"] + 0.5 + f["person"].map(shift)
        tied = _by(extras.say_do(f, "said", "did", preds=["y"], B=400), "pred_y")
        g = f.rename(columns={"person": "group"})          # survey_frame / EvalFrame convention
        viag = extras.say_do(g, "said", "did", preds=["y"], B=400)
        self.assertEqual(viag["cluster"], "group")
        self.assertEqual(viag["n_clusters"], 40)
        self.assertAlmostEqual(_by(viag, "pred_y")["hi"], tied["hi"])
        tr = extras.say_do(f.rename(columns={"person": "grup"}), "said", "did", preds=["y"], cluster="grup", B=50)
        self.assertEqual(tr["cluster"], "group")
        with self.assertRaises(KeyError):                   # a named cluster column that is absent is an error
            extras.say_do(f, "said", "did", preds=["y"], cluster="persona_id", B=50)

    def test_missing_and_aliases(self):
        f = _frame(persons=50).rename(columns={"pred_say": "tahmin_say"})
        f.loc[:4, "tahmin_say"] = np.nan
        f.loc[5, "did"] = np.nan
        sec = extras.say_do(f, "said", "did", preds=["tahmin_say"], B=100)
        r = sec["predictions"][0]
        self.assertEqual(r["prediction"], "pred_say")
        self.assertEqual(r["n_missing"], 6)
        self.assertIn("not imputed", " ".join(sec["notes"]))
        nocl = extras.say_do(f.drop(columns="person"), "said", "did", preds=["say"], B=50)
        self.assertIsNone(nocl["cluster"])
        self.assertIn("own cluster", " ".join(nocl["notes"]))

    def test_errors_and_render(self):
        f = _frame(persons=20)
        with self.assertRaises(KeyError):
            extras.say_do(f, "said", "nope")
        with self.assertRaises(KeyError):
            extras.say_do(f, "said", "did", preds=["nope"])
        with self.assertRaises(ValueError):
            extras.say_do(f, "said", "did", on="median")
        with self.assertRaises(ValueError):
            extras.say_do(f, "said", "did", B=0)
        g = f.copy()
        g.loc[0, "person"] = None
        with self.assertRaises(ValueError):
            extras.say_do(g, "said", "did")
        sec = extras.say_do(f, "said", "did", B=50)
        self.assertIn("| anchors (do − say) |", extras.section_markdown(sec))
        self.assertIn("se-say_do", extras.section_html(sec))
        self.assertIn("say_do", extras.__all__)


if __name__ == "__main__":
    unittest.main()

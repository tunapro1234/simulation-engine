"""Tests for simengine.evidence.extras (synthetic data only, offline)."""
import json
import math
import unittest

import numpy as np
import pandas as pd

import simengine.evidence as ev
from simengine.evidence import extras


def _frame(seed=0, n=60, items=3, shrink=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(items):
        y = rng.normal(3 + i, 1 + 0.5 * i, n)
        p = y.mean() + shrink * (y - y.mean())
        rows += [{"item": f"q{i}", "truth": a, "pred_sim": b} for a, b in zip(y, p)]
    return pd.DataFrame(rows)


class VarianceRatio(unittest.TestCase):
    def test_identical_is_one(self):
        r = ev.variance_ratio(_frame(), "sim", B=200)
        self.assertAlmostEqual(r["pooled"]["ratio"], 1.0, places=12)
        self.assertAlmostEqual(r["pooled"]["lo"], 1.0, places=12)
        self.assertAlmostEqual(r["pooled"]["hi"], 1.0, places=12)
        for it in r["items"]:
            self.assertAlmostEqual(it["ratio"], 1.0, places=12)
        self.assertIsNone(r["flag"])

    def test_halved_spread_is_half(self):
        r = ev.variance_ratio(_frame(shrink=0.5), "sim", B=200)
        self.assertAlmostEqual(r["pooled"]["ratio"], 0.5, places=12)
        self.assertAlmostEqual(r["pooled"]["lo"], 0.5, places=10)
        self.assertAlmostEqual(r["pooled"]["hi"], 0.5, places=10)
        self.assertTrue(all(abs(it["ratio"] - 0.5) < 1e-12 for it in r["items"]))
        self.assertEqual(r["flag"], "homogenisation")
        self.assertEqual(r["median_item_ratio"], 0.5)

    def test_unpaired_long_format(self):
        rng = np.random.default_rng(1)
        h = rng.normal(0, 2, 400)
        s = rng.normal(0, 1, 300)
        df = pd.DataFrame({"source": ["human"] * 400 + ["llm"] * 300, "value": np.r_[h, s]})
        r = ev.variance_ratio(df, "llm", source="source", B=300)
        self.assertFalse(r["paired"])
        self.assertAlmostEqual(r["pooled"]["ratio"], np.std(s, ddof=1) / np.std(h, ddof=1), places=12)
        self.assertLess(r["pooled"]["lo"], r["pooled"]["ratio"])
        self.assertGreater(r["pooled"]["hi"], r["pooled"]["ratio"])
        self.assertTrue(0.45 < r["pooled"]["lo"] and r["pooled"]["hi"] < 0.7)

    def test_missing_counted_not_imputed_and_degenerate_items(self):
        df = _frame(items=2, n=20)
        df.loc[0, "pred_sim"] = np.nan
        df = pd.concat([df, pd.DataFrame({"item": ["flat"] * 3, "truth": [1.0] * 3, "pred_sim": [1.0, 2.0, 3.0]}),
                        pd.DataFrame({"item": ["tiny"], "truth": [1.0], "pred_sim": [2.0]})])
        r = ev.variance_ratio(df, "sim", B=50)
        self.assertEqual(r["n_missing_sim"], 1)
        by = {it["item"]: it for it in r["items"]}
        self.assertEqual(by["q0"]["n_sim"], 19)
        self.assertIsNone(by["flat"]["ratio"])
        self.assertEqual(by["flat"]["reason"], "human SD is 0")
        self.assertIsNone(by["tiny"]["ratio"])
        self.assertEqual(r["pooled"]["n_items"], 2)
        self.assertTrue(any("not imputed" in n for n in r["notes"]))

    def test_turkish_alias_and_errors(self):
        df = _frame().rename(columns={"truth": "gercek", "item": "madde", "pred_sim": "tahmin_sim"})
        self.assertAlmostEqual(ev.variance_ratio(df, "sim", B=20)["pooled"]["ratio"], 1.0)
        with self.assertRaises(KeyError):
            ev.variance_ratio(_frame(), "nope")
        bad = _frame()
        bad["pred_sim"] = bad["pred_sim"].astype(object)
        bad.loc[3, "pred_sim"] = "abc"
        with self.assertRaises(ValueError):
            ev.variance_ratio(bad, "sim")

    def test_deterministic_and_json(self):
        a = ev.variance_ratio(_frame(shrink=0.7), "sim", B=100, seed=3)
        b = ev.variance_ratio(_frame(shrink=0.7), "sim", B=100, seed=3)
        self.assertEqual(json.dumps(a, default=str), json.dumps(b, default=str))
        json.dumps(a)  # JSON-friendly without a default hook


class Completion(unittest.TestCase):
    def setUp(self):
        # counts from the replicant Ertan vignette run (regime page): 50/50, 50/50, 30/50
        self.sec = ev.completion_row(by_condition={"default": (50, 50, 0), "random": (50, 50, 0),
                                                   "cooperative": {"planned": 50, "completed": 30,
                                                                   "parse_failed": 20}})

    def test_counts_and_test(self):
        s = self.sec
        self.assertEqual(s["overall"]["n_planned"], 150)
        self.assertEqual(s["overall"]["n_completed"], 130)
        self.assertEqual(s["overall"]["n_parse_failed"], 20)
        self.assertEqual(s["overall"]["n_dropout"], 0)
        self.assertEqual(s["worst_condition"], "cooperative")
        self.assertEqual(s["test"]["test"], "chi2")
        self.assertLess(s["test"]["p_value"], 0.001)
        lo, hi = s["by_condition"][2]["lo"], s["by_condition"][2]["hi"]
        self.assertTrue(lo < 0.6 < hi)

    def test_renders_md_and_html(self):
        md = ev.section_markdown(self.sec)
        self.assertIn("| condition | planned | completed |", md)
        self.assertIn("| cooperative | 50 | 30 | 20 | 0 | 0.600 |", md)
        self.assertIn("never imputed", md)
        page = ev.section_html(self.sec)
        self.assertIn('<section class="se-extra se-completion">', page)
        self.assertIn("<td>cooperative</td>", page)
        self.assertIn("<th>parse failed</th>", page)
        self.assertNotIn("<script", page)

    def test_totals_only_and_validation(self):
        s = ev.completion_row(50, 7)
        self.assertAlmostEqual(s["overall"]["rate"], 0.14)
        self.assertIsNone(s["test"])
        self.assertIsNone(s["overall"]["n_dropout"])
        with self.assertRaises(ValueError):
            ev.completion_row(10, 11)
        with self.assertRaises(ValueError):
            ev.completion_row(10, 8, n_parse_failed=3)
        with self.assertRaises(ValueError):
            ev.completion_row()

    def test_two_conditions_fisher_and_identical(self):
        s = ev.completion_row(by_condition={"a": (20, 20), "b": (20, 10)})
        self.assertEqual(s["test"]["test"], "fisher_exact")
        same = ev.completion_row(by_condition={"a": (5, 5), "b": (7, 7)})
        self.assertEqual(same["test"]["test"], "none")

    def test_from_frame(self):
        df = pd.DataFrame({"cond": ["x"] * 8 + ["y"] * 5, "valid": [True] * 7 + [False] + [True] * 3 + [False] * 2})
        s = ev.completion_from_frame(df, {"x": 10, "y": 5}, condition="cond")
        x = s["by_condition"][0]
        self.assertEqual((x["n_completed"], x["n_parse_failed"], x["n_dropout"]), (7, 1, 2))
        t = ev.completion_from_frame(df, 13)
        self.assertEqual((t["overall"]["n_completed"], t["overall"]["n_parse_failed"]), (10, 3))


class PublishedFindings(unittest.TestCase):
    def test_delta_table(self):
        s = ev.published_findings([
            {"finding": "vote punish-low", "published": 0.85, "simulated": 0.94, "source": "Ertan et al. 2009",
             "tolerance": 0.10},
            {"finding": "vote punish-high", "published": 0.0, "simulated": 0.0, "source": "Ertan et al. 2009",
             "tolerance": 0.05},
            {"finding": "baseline contribution", "published": 7.5, "simulated": None, "source": "Ertan et al. 2009"},
            {"finding": "p1 share", "published": 0.73, "simulated": 0.5, "published_ci": (0.69, 0.77),
             "source": "Ertan et al. 2009"},
        ])
        f = s["findings"]
        self.assertAlmostEqual(f[0]["delta"], 0.09)
        self.assertTrue(f[0]["match"])
        self.assertIsNone(f[1]["rel_delta"])
        self.assertIsNone(f[2]["delta"])
        self.assertFalse(f[3]["inside_published_ci"])
        self.assertEqual(s["summary"]["n_missing"], 1)
        self.assertEqual((s["summary"]["n_judged"], s["summary"]["n_match"]), (3, 2))
        md = ev.section_markdown(s)
        self.assertIn("| baseline contribution | 7.500 | — | — |", md)
        self.assertIn("Ertan et al. 2009", ev.section_html(s))

    def test_requires_published(self):
        with self.assertRaises(ValueError):
            ev.published_findings([{"finding": "x", "published": None, "simulated": 1}])


class StylizedFacts(unittest.TestCase):
    def test_na_excluded(self):
        s = ev.stylized_facts_score([
            {"name": "a", "value": 1.0, "weight": 3},
            {"name": "b", "value": 0.0, "weight": 1},
            {"name": "c", "value": None, "weight": 2},
            {"name": "d", "value": float("nan"), "weight": 2},
        ])
        self.assertAlmostEqual(s["index"], 0.75)
        self.assertEqual(s["excluded"], ["c", "d"])
        self.assertEqual(s["n_applicable"], 2)
        self.assertAlmostEqual(s["weight_coverage"], 0.5)
        self.assertIn("N/A", ev.section_markdown(s))

    def test_matches_who_to_punish_rule(self):
        # weighted mean over non-None criteria, as in edsl-test benchmark/score_wtp.py
        crit = [("x", 3, 1.0), ("y", 2, 0.5), ("z", 1, None)]
        s = ev.stylized_facts_score([{"name": n, "weight": w, "value": v} for n, w, v in crit])
        self.assertAlmostEqual(s["index"], (3 * 1.0 + 2 * 0.5) / 5)

    def test_all_na_and_validation(self):
        self.assertIsNone(ev.stylized_facts_score([{"name": "a", "value": None}])["index"])
        with self.assertRaises(ValueError):
            ev.stylized_facts_score([{"name": "a", "value": 1.2}])
        with self.assertRaises(ValueError):
            ev.stylized_facts_score([{"name": "a", "value": 0.5, "weight": 0}])
        with self.assertRaises(ValueError):
            ev.stylized_facts_score([])


class CoefficientRecovery(unittest.TestCase):
    def _data(self, seed, b1, b2, n=400, noise=1.0):
        rng = np.random.default_rng(seed)
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        y = 1 + b1 * x1 + b2 * x2 + rng.normal(scale=noise, size=n)
        return pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    def test_same_process_recovers(self):
        s = ev.coefficient_recovery(self._data(1, 2.0, -1.0), self._data(2, 2.0, -1.0), ("y", "x1", "x2"))
        self.assertEqual(s["summary"]["sign_agreement"], 1.0)
        self.assertEqual(s["summary"]["share_differ"], 0.0)
        self.assertEqual(s["summary"]["sign_flip_when_human_significant"], 0.0)
        self.assertEqual(s["summary"]["n_coefficients"], 2)

    def test_flipped_sign_detected(self):
        s = ev.coefficient_recovery(self._data(1, 2.0, 1.0), self._data(2, 2.0, -1.0), {"y": "y", "x": ["x1", "x2"]})
        self.assertEqual(s["summary"]["sign_agreement"], 0.5)
        self.assertEqual(s["summary"]["share_differ"], 0.5)
        self.assertEqual(s["summary"]["sign_flip_when_human_significant"], 0.5)
        by = {c["name"]: c for c in s["coefficients"]}
        self.assertTrue(by["x2"]["differs"])
        self.assertFalse(by["x2"]["same_sign"])

    def test_listwise_drop_counted_and_errors(self):
        sim = self._data(1, 1.0, 1.0, n=50)
        sim.loc[0, "x1"] = np.nan
        s = ev.coefficient_recovery(sim, self._data(2, 1.0, 1.0, n=50), ("y", "x1", "x2"))
        self.assertEqual(s["summary"]["dropped_sim"], 1)
        self.assertEqual(s["summary"]["n_sim"], 49)
        col = self._data(3, 1, 1, n=30)
        col["x2"] = col["x1"] * 2
        with self.assertRaises(ValueError):
            ev.coefficient_recovery(col, self._data(2, 1, 1), ("y", "x1", "x2"))
        with self.assertRaises(ValueError):
            ev.coefficient_recovery(sim, sim, ("y",))
        self.assertIn("<table>", ev.section_html(s))


class Rendering(unittest.TestCase):
    def test_list_and_escape(self):
        a = ev.stylized_facts_score([{"name": "<b>x|y</b>", "value": 0.5}])
        b = ev.completion_row(4, 3)
        md = ev.section_markdown([a, b], level=2)
        self.assertEqual(md.count("\n## "), 1)
        self.assertTrue(md.startswith("## "))
        self.assertIn("x\\|y", md)
        page = ev.section_html([a, b])
        self.assertIn("&lt;b&gt;", page)
        self.assertEqual(page.count("<section"), 2)

    def test_exported(self):
        pending = {"say_do"}            # in extras.__all__; package-level export lands with the evidence/__init__ task
        for name in extras.__all__:
            if name in pending and name not in ev.__all__:
                continue
            self.assertIn(name, ev.__all__)
            self.assertTrue(callable(getattr(ev, name)))
        self.assertTrue(math.isfinite(0.0))


if __name__ == "__main__":
    unittest.main()

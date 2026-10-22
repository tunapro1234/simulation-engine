"""open_ecommerce usability: the fully observed rule in errors and layout, survey column checks, meta naming."""
import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine import bench
from simengine.bench.tasks import open_ecommerce as oe
from simengine.sim.prediction import AMAZON_QUESTIONS


def _write(root, n_users=50, last_year=2023, survey_cols=None, seed=0):
    d = os.path.join(root, "open-ecommerce")
    os.makedirs(d)
    rng = np.random.default_rng(seed)
    cats = [f"CAT_{i:02d}" for i in range(6)]
    rows = []
    for u in range(n_users):
        days = list(rng.integers(0, 365 * 4, 25)) + [365 * 4 + 200]   # 2021-12 .. 2022-07
        for dd in days:
            dt = pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(dd))
            rows.append({"Order Date": dt.strftime("%Y-%m-%d"), "Purchase Price Per Unit": "5.0", "Quantity": "1",
                         "Category": rng.choice(cats), "Survey ResponseID": f"R_{u:03d}"})
        rows.append({"Order Date": f"{last_year}-02-01", "Purchase Price Per Unit": "5.0", "Quantity": "1",
                     "Category": rng.choice(cats), "Survey ResponseID": f"R_{u:03d}"})
    pd.DataFrame(rows).to_csv(os.path.join(d, "amazon-purchases.csv"), index=False)
    cols = survey_cols if survey_cols is not None else [c for c in oe.SURVEY_COLS if c != "Survey ResponseID"]
    sv = {"Survey ResponseID": [f"R_{u:03d}" for u in range(n_users)]}
    for c in cols:
        sv[c] = rng.choice(["a", "b"], n_users)
    pd.DataFrame(sv).to_csv(os.path.join(d, "survey.csv"), index=False)


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()


class TestRule(_Tmp):
    def test_error_names_rule_when_no_post_2023_purchases(self):
        _write(self.root, last_year=2022)
        with self.assertRaises(ValueError) as cm:
            bench.get("open_ecommerce", n_persons=20, n_categories=4).load(self.root)
        msg = str(cm.exception)
        self.assertIn("only 0 fully observed persons", msg)
        self.assertIn("on or after 2023-01-01", msg)

    def test_layout_lists_rule_and_raw_columns(self):
        lay = bench.get("open_ecommerce").describe()["layout"]
        self.assertIn("on or after 2023-01-01", lay)
        for c in oe.SURVEY_RAW_REQUIRED:
            self.assertIn(c, lay)
        self.assertTrue(set(oe.SURVEY_RAW_REQUIRED) <= set(AMAZON_QUESTIONS))


class TestDemosOnlySurvey(_Tmp):
    def test_baselines_warn_and_skip_survey_lgbm(self):
        demos = [c for c in oe.SURVEY_COLS if c.startswith("Q-demos-")]
        _write(self.root, survey_cols=demos)
        task = bench.get("open_ecommerce", n_persons=20, n_categories=4)
        data = task.load(self.root)
        self.assertIn("Q-sexual-orientation", data.meta["survey_missing"])
        self.assertIn("Q-life-changes", data.meta["survey_missing"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            preds = task.baselines(data)
        self.assertNotIn("survey_lgbm", preds)
        msgs = [str(x.message) for x in w if "survey_lgbm" in str(x.message)]
        self.assertEqual(len(msgs), 1)
        self.assertIn("Q-substance-use-cigarettes", msgs[0])
        self.assertNotIn("sigara", msgs[0])
        frame = task.frame(data, preds)
        self.assertEqual(len(frame), 20 * 4)

    def test_full_survey_does_not_warn(self):
        _write(self.root)
        task = bench.get("open_ecommerce", n_persons=20, n_categories=4)
        data = task.load(self.root)
        self.assertEqual(data.meta["survey_missing"], [])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            task.baselines(data)
        self.assertFalse([x for x in w if "survey_lgbm" in str(x.message)])


class TestMeta(_Tmp):
    def setUp(self):
        super().setUp()
        _write(self.root)
        self.task = bench.get("open_ecommerce", n_persons=30, n_categories=4)
        self.data = self.task.load(self.root)

    def test_split_follows_n_persons(self):
        self.assertIn("30-person sample", self.data.meta["split"])
        self.assertNotIn("400-person", self.data.meta["split"])

    def test_raw_survey_matches_amazon_questions(self):
        raw = self.data.meta["survey_raw"]
        self.assertIn("Survey ResponseID", raw.columns)
        self.assertTrue(set(AMAZON_QUESTIONS) <= set(raw.columns))
        self.assertIn("yas", self.data.meta["survey"].columns)          # internal names kept
        people = oe.survey_people(self.data)
        self.assertEqual(sorted(people), self.data.meta["sample"])
        self.assertTrue(set(next(iter(people.values()))) <= set(AMAZON_QUESTIONS))

    def test_truth_dict_and_frame_columns(self):
        truth = oe.truth_dict(self.data)
        self.assertEqual(sorted(truth), self.data.meta["sample"])
        self.assertEqual(sum(len(v) for v in truth.values()), len(self.data.test))
        frame = self.task.frame(self.data, {"never": np.zeros(len(self.data.test))})
        for c in ("unit", "group", "item", "uid", "category", "truth", "balanced", "tiebreak", "segment", "split"):
            self.assertIn(c, frame.columns)
        r = frame.iloc[0]
        self.assertEqual(truth[r["uid"]][r["category"]], r["truth"])


if __name__ == "__main__":
    unittest.main()

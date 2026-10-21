"""Synthetic fixtures for every bench adapter: tiny files in the documented layout, written to a temp dir."""
import json
import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine import bench
from simengine.types import Task


def _gss(root):
    d = os.path.join(root, "gss")
    os.makedirs(d)
    rng = np.random.default_rng(0)
    items = ["polviews", "natenvir", "cappun"]
    for k, a in enumerate(["panel2006", "panel2008", "panel2010"]):
        n = 160
        lat = rng.normal(size=n)
        cols = {"age_1": rng.integers(18, 80, n).astype(float), "age_3": np.nan, "sex_1": rng.integers(1, 3, n),
                "sex_3": rng.integers(1, 3, n)}
        for j, it in enumerate(items):
            x = np.clip(np.round(2.5 + lat + rng.normal(0, .7, n)), 1, 4)
            y = np.clip(np.round(2.5 + lat + rng.normal(0, .7, n)), 1, 4)
            cols[f"{it}_1"], cols[f"{it}_2"], cols[f"{it}_3"] = x, np.nan, y
        pd.DataFrame(cols).to_parquet(os.path.join(d, f"{a}.parquet"))
    n = 260
    lat = rng.normal(size=n)
    st = np.where(np.arange(n) < 130, 2016, 2018)
    cols = {"panstat": np.ones(n), "samptype": st, "age_1a": rng.integers(18, 80, n).astype(float),
            "age_1b": rng.integers(18, 80, n).astype(float), "age_2": np.nan,
            "sex_1a": rng.integers(1, 3, n), "sex_1b": rng.integers(1, 3, n), "sex_2": rng.integers(1, 3, n)}
    for it in items:
        cols[f"{it}_1a"] = np.where(st == 2016, np.clip(np.round(2.5 + lat + rng.normal(0, .7, n)), 1, 4), np.nan)
        cols[f"{it}_1b"] = np.where(st == 2018, np.clip(np.round(2.5 + lat + rng.normal(0, .7, n)), 1, 4), np.nan)
        cols[f"{it}_2"] = np.clip(np.round(2.5 + lat + rng.normal(0, .7, n)), 1, 4)
    pd.DataFrame(cols).to_parquet(os.path.join(d, "panel2016_2020.parquet"))


GROUPS = {"SEX": ["Male", "Female"], "AGE": ["18-29", "65+"]}


def _oqa(root):
    o = os.path.join(root, "opinionqa", "islenmis")
    s = os.path.join(root, "subpop-opinionqa", "data")
    os.makedirs(o)
    os.makedirs(os.path.join(s, "subpopulation_metadata"))
    os.makedirs(os.path.join(s, "opinionqa", "processed"))
    pd.DataFrame({"attribute": list(GROUPS), "group": [str(v) for v in GROUPS.values()]}).to_csv(
        os.path.join(s, "subpopulation_metadata", "demographics_22.csv"), index=False)
    rng = np.random.default_rng(1)
    topics = ["economy jobs", "crime safety", "climate energy", "health care", "economy taxes", "school safety"]
    qrows, drows = [], []
    for i, t in enumerate(topics):
        q, w = f"Q{i}_W{1 + i % 2}", f"W{1 + i % 2}"
        qrows.append({"wave": w, "qkey": q, "question": f"How worried are you about {t} in your community",
                      "options": np.array(["Very", "Somewhat", "Not at all", "Refused"], dtype=object),
                      "ordinal": np.array([1.0, 2.0, 3.0]), "dis500": False})
        for a, gs in [("Overall", ["Overall"])] + list(GROUPS.items()):
            for g in gs:
                drows.append({"qkey": q, "attribute": a, "group": g, "responses": rng.dirichlet([3, 2, 1, .1])})
    tests = []
    for i, t in enumerate(["economy prices", "crime police"]):
        q, w = f"T{i}_W{1 + i}", f"W{1 + i}"
        qrows.append({"wave": w, "qkey": q, "question": f"How worried are you about {t} in your community",
                      "options": np.array(["Very", "Somewhat", "Not at all", "Refused"], dtype=object),
                      "ordinal": np.array([1.0, 2.0, 3.0]), "dis500": True})
        for a, gs in [("Overall", ["Overall"])] + list(GROUPS.items()):
            for g in gs:
                tests.append({"qkey": q, "attribute": a, "group": g, "responses": str([float(x) for x in rng.dirichlet([3, 2, 1])]),
                              "refusal_rate": 0.0, "ordinal": "[1.0, 2.0, 3.0]",
                              "question": f"How worried are you about {t} in your community",
                              "options": "['Very', 'Somewhat', 'Not at all', 'Refused']"})
    pd.DataFrame(qrows).to_parquet(os.path.join(o, "sorular.parquet"))
    pd.DataFrame(drows).to_parquet(os.path.join(o, "dagilim.parquet"))
    pd.DataFrame(tests).to_csv(os.path.join(s, "opinionqa", "processed", "opinionqa.csv"), index=False)


TWIN_COLS = ["Q157", "Q158", "Q164", "Q166", "Q168", "Q170", "1_Q295", "Q181"]


def _twin(root):
    t = os.path.join(root, "twin2k500")
    q = os.path.join(t, "question_catalog_and_human_response_csv")
    f = os.path.join(t, "LLM_simulation_results", "GPT4.1-mini-simulation-llm-vs-human")
    os.makedirs(q)
    os.makedirs(f)
    rng = np.random.default_rng(2)
    n = 160
    ids = np.arange(1, n + 1)
    lat = rng.normal(size=n)
    feats = {f"QID{i}": np.round(3 + lat * (i % 3) + rng.normal(size=n)) for i in range(1, 9)}
    feats["QID99_1"] = rng.integers(1, 6, n)  # target in catalogue form, excluded from features
    pd.DataFrame({"pid": ids, **feats}).to_csv(os.path.join(q, "wave1_3_response.csv"), index=False)
    json.dump([{"formatted_column": "Q157", "QuestionID": "QID99", "catalog_csv_column": "QID99_1"}],
              open(os.path.join(t, "LLM_simulation_results", "wave4_formatted_to_catalog_mapping.json"), "w"))
    json.dump([{"QuestionID": "QID1", "Settings": "{'Selector': 'SAVR'}", "csv_columns": "['QID1']"}],
              open(os.path.join(q, "question_catalog.json"), "w"))
    rng_ = bench.tasks.twin.column_ranges()

    def answers(noise):
        d = {"TWIN_ID": ids}
        for c in TWIN_COLS:
            lo, hi = rng_[c]
            if c in ("Q164", "Q166", "Q168", "Q170"):
                d[c] = np.round(np.exp(4 + 0.5 * lat + rng.normal(0, noise, n)))
            else:
                d[c] = np.clip(np.round((lo + hi) / 2 + lat + rng.normal(0, noise, n)), lo, hi)
        return pd.DataFrame(d)
    for name, noise in (("responses_wave1_3_formatted.csv", .5), ("responses_wave4_formatted.csv", .7)):
        df = answers(noise)
        with open(os.path.join(f, name), "w") as fh:
            fh.write(",".join(df.columns) + "\n")
            fh.write(",".join(["question text"] * len(df.columns)) + "\n")  # formatted files: 2nd row = question text
            df.to_csv(fh, header=False, index=False)


def _oe(root):
    d = os.path.join(root, "open-ecommerce")
    os.makedirs(d)
    rng = np.random.default_rng(3)
    cats = [f"CAT_{i:02d}" for i in range(8)]
    rows = []
    for u in range(70):
        uid = f"R_{u:04d}"
        taste = rng.dirichlet(np.ones(len(cats)) * .5)
        days = rng.integers(0, 365 * 5, rng.integers(15, 60))
        last = 0 if u < 60 else -400  # 60 fully observed users (last purchase in 2023)
        for dd in list(days) + [365 * 5 + 10 + last]:
            dt = pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(dd))
            rows.append({"Order Date": dt.strftime("%Y-%m-%d"), "Purchase Price Per Unit": "9.99", "Quantity": "1.0",
                         "Shipping Address State": "NJ", "Title": "x", "ASIN/ISBN (Product Code)": "B0",
                         "Category": rng.choice(cats, p=taste) if rng.random() > .05 else "",
                         "Survey ResponseID": uid})
    pd.DataFrame(rows).to_csv(os.path.join(d, "amazon-purchases.csv"), index=False)
    sv = {"Survey ResponseID": [f"R_{u:04d}" for u in range(70)]}
    for c in bench.tasks.open_ecommerce.SURVEY_COLS:
        if c != "Survey ResponseID":
            sv[c] = rng.choice(["a", "b", ""], 70)
    sv["Q-life-changes"] = rng.choice(["Moved place of residence", "Lost a job,Had a child", ""], 70)
    pd.DataFrame(sv).to_csv(os.path.join(d, "survey.csv"), index=False)


def _pairs(rng, n_tests=60):
    rows = []
    good, bad = ["amazing", "you", "won't believe"], ["report", "policy", "update"]
    for t in range(n_tests):
        date = pd.Timestamp("2013-01-01") + pd.Timedelta(days=7 * t)
        for k in range(3):
            win = rng.random() < .5
            hw = f"{rng.choice(good)} story {t} {k} {rng.choice(good)}"
            hl = f"{rng.choice(bad)} story {t} {k} {rng.choice(bad)}"
            a, b = (hw, hl) if win else (hl, hw)
            rows.append({"test_id": f"t{t:03d}", "split": "holdout" if t >= 45 else "exploratory", "test_date": date,
                         "headline_a": a, "headline_b": b, "winner": "a" if win else "b", "headline_only": True,
                         "signif_fdr": bool(rng.random() < .8)})
    return pd.DataFrame(rows)


def _upworthy(root, raw=False):
    d = os.path.join(root, "upworthy")
    rng = np.random.default_rng(4)
    if not raw:
        os.makedirs(os.path.join(d, "islenmis"))
        _pairs(rng).to_parquet(os.path.join(d, "islenmis", "pairwise.parquet"))
        return
    os.makedirs(os.path.join(d, "ham"))
    for s, off in (("exploratory", 0), ("confirmatory", 20), ("holdout", 40)):
        rows = []
        for t in range(20):
            for k, h in enumerate(["you won't believe this amazing thing", "a policy report on things", "an update"]):
                imp = 2000
                rows.append({"created_at": str(pd.Timestamp("2013-01-01") + pd.Timedelta(days=7 * (t + off))),
                             "clickability_test_id": f"{s}{t}", "headline": f"{h} {t}", "excerpt": "e", "lede": "l",
                             "eyecatcher_id": "img", "impressions": imp,
                             "clicks": int(imp * [0.06, 0.02, 0.03][k]) + int(rng.integers(0, 5))})
        pd.DataFrame(rows).to_csv(os.path.join(d, "ham", f"upworthy-archive-{s}-packages-03.12.2020.csv"))


def _simbench(root):
    d = os.path.join(root, "simbench")
    os.makedirs(d)
    rng = np.random.default_rng(5)
    for f, n_ds in (("SimBenchPop.csv", 3), ("SimBenchGrouped.csv", 2)):
        rows = []
        for ds in range(n_ds):
            for q in range(12):
                for g in range(2 if f.startswith("SimBenchG") else 1):
                    p = rng.dirichlet([4, 2, 1]) * 100
                    rows.append({"group_prompt_template": "You are {who}.", "group_prompt_variable_map": str({"who": f"g{g}"}),
                                 "input_template": f"Question {q} of dataset {ds}?", "auxiliary": "{}", "group_size": 100,
                                 "human_answer": str({"A": float(p[0]), "B": float(p[1]), "C": float(p[2])}), "dataset_name": f"DS{ds}"})
        pd.DataFrame(rows).to_csv(os.path.join(d, f))


def _bank(root):
    d = os.path.join(root, "bank-marketing", "bank-additional")
    os.makedirs(d)
    rng = np.random.default_rng(6)
    n = 1000
    cols = {c: rng.choice(["a", "b", "c"], n) for c in bench.tasks.bank.CATEGORICAL}
    cols["poutcome"] = rng.choice(["nonexistent", "failure", "success"], n, p=[.8, .15, .05])
    for c in bench.tasks.bank.NUMERIC:
        cols[c] = rng.normal(size=n).round(3)
    cols["duration"] = rng.integers(1, 500, n)
    z = -2 + 1.5 * (cols["poutcome"] == "success") + cols["euribor3m"] + np.linspace(0, 1.5, n)
    cols["y"] = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "yes", "no")
    pd.DataFrame(cols).to_csv(os.path.join(d, "bank-additional-full.csv"), sep=";", index=False)


class _Base(unittest.TestCase):
    builder = None

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = cls._tmp.name
        cls.builder(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def check_task(self, task, data, frame, expect_methods):
        self.assertIsInstance(task, Task)
        self.assertEqual(len(frame), len(data.test))
        self.assertTrue((frame["split"] == "test").all())
        self.assertIn("truth", frame.columns)
        for m in expect_methods:
            self.assertIn(f"pred_{m}", frame.columns)
            r = task.score(frame, m, B=50)
            self.assertEqual(set(r) >= {"metric", "value", "lo", "hi", "n", "higher_is_better"}, True)
            self.assertTrue(np.isfinite(r["value"]), (m, r))
        self.assertIn("split", data.meta)


class TestGSS(_Base):
    builder = staticmethod(_gss)

    def test_load_split_baselines(self):
        task = bench.get("gss")
        data = task.load(self.root)
        self.assertEqual(sorted(data.meta["targets"]), ["cappun", "natenvir", "polviews"])  # 'sex' is factual
        self.assertTrue(data.test["person"].str.startswith("panel2016_2020").all())
        self.assertFalse(data.train["person"].str.startswith("panel2016_2020").any())
        frame = task.run(data=data)
        self.check_task(task, data, frame, ["median", "persistence", "ridge", "pca_ridge"])
        self.assertEqual(task.score(frame, "median")["value"], 0.0)  # constant prediction -> r = 0 by rule
        self.assertGreater(task.score(frame, "pca_ridge")["value"], 0.2)

    def test_holdout_panel_and_dev(self):
        data = bench.get("gss", holdout_panel="panel2008").load(self.root)
        self.assertTrue(data.test["person"].str.startswith("panel2008").all())
        self.assertFalse(data.train["person"].str.startswith("panel2008").any())
        r = bench.tasks.gss.cross_panel_dev(self.root, {"yontem": "kendi"}, items="all", B=50)
        self.assertEqual(r["mean_z"]["n"], 3)
        with self.assertRaises(ValueError):
            bench.get("gss", holdout_panel="panel1999").load(self.root)

    def test_env_var_root(self):
        old = os.environ.get("SIMENGINE_DATA")
        os.environ["SIMENGINE_DATA"] = self.root
        try:
            self.assertGreater(len(bench.get("gss").load().test), 0)
        finally:
            if old is None:
                del os.environ["SIMENGINE_DATA"]
            else:
                os.environ["SIMENGINE_DATA"] = old


class TestOpinionQA(_Base):
    builder = staticmethod(_oqa)

    def test_setting_a(self):
        task = bench.get("opinionqa_groups")
        data = task.load(self.root)
        self.assertEqual(data.meta["n_questions"], 2)
        self.assertEqual(len(data.test), 2 * 4)
        frame = task.run(data=data)
        self.check_task(task, data, frame, ["uniform", "position", "knn"])
        self.assertFalse(task.primary.higher_is_better)
        for p in frame["pred_position"]:
            self.assertAlmostEqual(float(np.sum(p)), 1.0, places=6)
        with self.assertRaises(ValueError):
            task.predict_with(data, "overall")  # needs setting B

    def test_wave_out_rule_and_setting_b(self):
        task = bench.get("opinionqa_groups", split="dev", setting="B")
        data = task.load(self.root)
        self.assertEqual(data.meta["n_questions"], 6)
        h = task.history(data, "W1")
        self.assertTrue((h["wave"] != "W1").all())
        frame = task.run(data=data)
        self.check_task(task, data, frame, ["uniform", "position", "overall", "sibling"])
        # position baseline for a W1 question must equal the mean of W2 questions' group distributions
        q = data.test[data.test.wave == "W1"].iloc[0]
        w2 = data.train[(data.train.wave == "W2") & (data.train.attribute == q.attribute) & (data.train.group == q.group)]
        exp = np.mean(np.stack(w2["dist"].to_list()), 0)
        got = frame.loc[(frame["item"] == q.qkey) & (frame["unit"].str.endswith(f"{q.attribute}:{q.group}")),
                        "pred_position"].iloc[0]
        np.testing.assert_allclose(got, exp)


class TestTwin(_Base):
    builder = staticmethod(_twin)

    def test_person_split_and_scoring(self):
        task = bench.get("twin2k500")
        data = task.load(self.root)
        self.assertFalse(set(data.meta["train_ids"]) & set(data.meta["test_ids"]))
        self.assertNotIn("QID99_1", data.meta["features"].columns)
        self.assertTrue(data.test["retest"].notna().all())
        frame = task.run(data=data)
        self.check_task(task, data, frame, ["median", "ridge"])
        anchors = frame[frame["item"].isin(["Q164", "Q166"])]
        self.assertTrue(anchors["truth"].between(1, 10).all())  # deciles
        self.assertEqual(task.score(frame, "median")["value"], 0.0)
        self.assertGreater(task.score(frame, "ridge")["value"], 0.1)
        ceiling = task.score(task.frame(data, {"retest": data.test["retest"].to_numpy()}), "retest", "acc", B=50)
        self.assertGreater(ceiling["value"], 0.5)

    def test_dev_split_never_uses_test_persons(self):
        full = bench.get("twin2k500").load(self.root)
        dev = bench.get("twin2k500", split="dev").load(self.root)
        used = set(dev.meta["train_ids"]) | set(dev.meta["test_ids"])
        self.assertFalse(used & set(full.meta["test_ids"]))


class TestOpenEcommerce(_Base):
    builder = staticmethod(_oe)

    def test_panel_and_baselines(self):
        task = bench.get("open_ecommerce", seed=1, n_persons=40, n_categories=5)
        data = task.load(self.root)
        self.assertEqual(len(data.test), 40 * 5)
        self.assertEqual(data.meta["n_fully_observed"], 60)
        self.assertFalse(set(data.meta["sample"]) & set(data.meta["train_persons"]))
        frame = task.run(data=data)
        self.check_task(task, data, frame, ["never", "popularity", "persistence", "decayed_count"])
        self.assertAlmostEqual(task.score(frame, "never", "person_auc")["value"], 0.5)
        m = data.meta["mask"]
        Y = data.test["y"].to_numpy().reshape(m.shape)
        self.assertTrue(((m & (Y == 1)).sum(1) == (m & (Y == 0)).sum(1)).all())  # balanced subset
        with self.assertRaises(ValueError):
            bench.get("open_ecommerce", n_persons=400).load(self.root)


class TestUpworthy(_Base):
    builder = staticmethod(_upworthy)

    def test_temporal_and_holdout(self):
        for split in ("temporal", "holdout"):
            task = bench.get("upworthy", split=split)
            data = task.load(self.root)
            self.assertTrue(data.test["signif_fdr"].all())
            if split == "temporal":
                self.assertLess(data.train["test_date"].max(), data.test["test_date"].min())
            frame = task.run(data=data)
            self.check_task(task, data, frame, ["coin", "longer_headline", "tfidf_bt"])
            self.assertEqual(task.score(frame, "coin")["value"], 0.5)

    def test_raw_fallback(self):
        with tempfile.TemporaryDirectory() as r:
            _upworthy(r, raw=True)
            data = bench.get("upworthy", split="holdout", test_signif="none").load(r)
            self.assertGreater(len(data.test), 0)
            self.assertTrue((data.train.split != "holdout").all())

    def test_bh(self):
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
        self.assertEqual(bench.tasks.upworthy.bh_reject(p).sum(), 2)


class TestSimBench(_Base):
    builder = staticmethod(_simbench)

    def test_split_and_score(self):
        task = bench.get("simbench")
        data = task.load(self.root)
        self.assertFalse(set(data.test["qkey"]) & set(data.train["qkey"]))
        frame = task.run(data=data)
        self.check_task(task, data, frame, ["uniform", "dev_marginal"])
        # uniform on every row of a dataset scores exactly 0
        allr = data.meta["all_rows"]
        fr = task.frame(type(data)(data.train, allr.reset_index(drop=True), data.meta),
                        {"u": [np.full(len(p), 1 / len(p)) for p in allr.P]})
        self.assertAlmostEqual(task.score(fr, "u", B=20)["value"], 0.0, places=9)
        grp = bench.get("simbench", part="grp").load(self.root)
        self.assertTrue((grp.test["part"] == "grp").all())


class TestBank(_Base):
    builder = staticmethod(_bank)

    def test_time_split(self):
        task = bench.get("bank_marketing")
        data = task.load(self.root)
        self.assertEqual((len(data.train), len(data.test)), (800, 200))
        self.assertNotIn("duration", data.test.columns)
        frame = task.run(data=data)
        self.check_task(task, data, frame, ["constant", "persistence", "logistic"])
        self.assertEqual(task.score(frame, "constant")["value"], 0.5)
        dev = bench.get("bank_marketing", split="dev", lgbm=False).load(self.root)
        self.assertEqual((len(dev.train), len(dev.test)), (600, 200))


if __name__ == "__main__":
    unittest.main()

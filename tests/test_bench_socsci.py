"""socsci210 bench task: synthetic HF-style shards and mapping files (no real rows); the real-data check runs only
when $SIMENGINE_DATA holds socsci210."""
import importlib.util
import json
import os
import re
import tempfile
import unittest

import numpy as np
import pandas as pd

from simengine.bench.base import DataNotFound
from simengine.bench.tasks import socsci210 as ss
from simengine.llm.fake import FakeLLM
from simengine.types import Task

# parquet needs pyarrow or fastparquet; without one the classes that use it skip (nothing gets installed)
PARQUET = any(importlib.util.find_spec(m) for m in ("pyarrow", "fastparquet"))
NO_PARQUET = "no parquet engine (pyarrow / fastparquet)"

SEEN, UNSEEN = ["s1", "s2", "s3", "s4"], ["s5", "s6"]
LIKERT = "Only return an integer from 1 to 5, where 1 means 'not at all' and 5 means 'very much', nothing else."
BINARY = "Only return 1 for Option A or 2 for Option B, nothing else."
OPEN = "Only return an integer representing dollars, nothing else."


def make_rows(seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """6 studies x 2 outcomes x 4 conditions; outcome 1 on 1-5, outcome 2 binary; the study moves the answers more
    than the condition does (as in most survey experiments).
    Study s3 adds an open-amount outcome 3, study s4's binary outcome is coded 3/4 (off its stated scale)."""
    rng = np.random.default_rng(seed)
    rows, maps = [], {"participant": {"seen": SEEN, "unseen": UNSEEN}, "condition": {}, "task": {}}
    for si, s in enumerate(SEEN + UNSEEN):
        sid, cmap, tmap = 0, {"train": [], "eval": []}, {"train": [], "eval": []}
        topic = ["taxes", "housing", "climate", "school", "health", "transit"][si]
        tasks = [1, 2, 3] if s == "s3" else [1, 2]
        for c in range(1, 5):
            for t in tasks:
                n = int(rng.integers(30, 60))
                if t == 1:
                    p = np.exp(-0.5 * (np.arange(5) - 0.7 * si - 0.25 * (c - 1)) ** 2)
                    r = rng.choice(np.arange(1, 6), n, p=p / p.sum())
                    q = f"Story about {topic} variant {c}. How much do you support the {topic} plan? {LIKERT}"
                elif t == 2:
                    a = 0.1 + 0.12 * si + 0.04 * c
                    r = rng.choice([1, 2], n, p=[a, 1 - a]) + (2 if s == "s4" else 0)
                    q = f"Story about {topic} variant {c}. Which {topic} option do you pick? {BINARY}"
                else:
                    r = rng.integers(0, 500, n)
                    q = f"Story about {topic} variant {c}. How much would you pay? {OPEN}"
                ids = list(range(sid, sid + n))
                sid += n
                cmap["eval" if c == 4 else "train"] += ids
                tmap["eval" if t == 2 else "train"] += ids
                rows += [{"study_id": s, "task_num": t, "condition_num": c, "sample_id": i, "response": int(x),
                          "stimuli": q, "participant": f"p{i}"} for i, x in zip(ids, r)]
        if s in ("s1", "s2", "s3", "s5"):
            maps["condition"][s] = cmap
        if s in ("s1", "s2", "s5"):
            maps["task"][s] = tmap
    return pd.DataFrame(rows), maps


def write_fixture(root: str, n_shards: int = 2, seed: int = 0) -> pd.DataFrame:
    d, maps = make_rows(seed)
    os.makedirs(os.path.join(root, "ham", "data"), exist_ok=True)
    os.makedirs(os.path.join(root, "ham", "metadata"), exist_ok=True)
    d = d.sample(frac=1, random_state=seed).reset_index(drop=True)   # a cell's answers span both shards
    for k, part in enumerate(np.array_split(np.arange(len(d)), n_shards)):
        d.iloc[part].to_parquet(os.path.join(root, "ham", "data", f"train-{k:05d}-of-{n_shards:05d}.parquet"))
    for name, m in maps.items():
        with open(os.path.join(root, "ham", "metadata", f"{name}_mapping.json"), "w") as f:
            json.dump(m, f)
    return d


class _Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        cls.raw = write_fixture(os.path.join(cls.root, "socsci210"))
        cls.task = ss.SocSci210()
        cls.data = cls.task.load(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestScaleAndDistances(unittest.TestCase):
    def test_parse_scale(self):
        cases = {
            "Q? Only return an integer from 1 to 7, nothing else.": (1, 7),
            "Q? Only return an integer from 1 (Strongly disagree) to 7 (Strongly agree).": (1, 7),
            "Q? Only return a number between 0 and 100, nothing else.": (0, 100),
            "Q? Only return an integer from 1 to 5, where 1 = 'never' and 5 = 'always'.": (1, 5),
            "Q? Only return 1 for Yes or 2 for No, nothing else.": (1, 2),
            "Q? Only return 1 if you choose A; 2 if you choose B, nothing else.": (1, 2),
            "Q? Only return 1 to choose the first person; 2 to choose the second person, nothing else.": (1, 2),
            "Q? Only return 1, 2, 3, or 4, nothing else.": (1, 4),
            "Q? Only return 1–4, nothing else.": (1, 4),
            "How do you feel, on a scale from 1 (very bad) to 7 (very good)?": (1, 7),
            "Q? Only return an integer representing dollars, nothing else.": None,
            "Q? Only return one of the integers 5, 50, or 500, nothing else.": None,
            "Q? Only return an integer from 3 to 3.": None,
        }
        for text, want in cases.items():
            self.assertEqual(ss.parse_scale(text), want, text)
        self.assertIsNone(ss.parse_scale(None))
        # the instruction wins over numbers in the story
        self.assertEqual(ss.parse_scale("An integer from 0 to 10 in the story. Only return 1 for Yes or 2 for No."),
                         (1, 2))

    def test_w1_matches_scipy_and_bounds(self):
        from scipy.stats import wasserstein_distance
        rng = np.random.default_rng(1)
        for k in (2, 5, 11):
            p, q = rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(k))
            pos = ss.positions(k)
            self.assertAlmostEqual(ss.w1(p, q), wasserstein_distance(pos, pos, p, q), places=12)
        self.assertAlmostEqual(ss.w1([1, 0, 0], [0, 0, 1]), 1.0)
        self.assertEqual(ss.w1([.2, .8], [.2, .8]), 0.0)

    def test_w1_uniform01_exact(self):
        rng = np.random.default_rng(2)
        x = (np.arange(200000) + 0.5) / 200000           # midpoint rule for int |F(x) - x| dx
        for k in (2, 3, 7, 101):
            p, pos = rng.dirichlet(np.ones(k)), ss.positions(k)
            F = np.cumsum(p)[np.searchsorted(pos, x, side="right") - 1]
            self.assertAlmostEqual(ss.w1_uniform01(p), float(np.abs(F - x).mean()), places=5)
        self.assertAlmostEqual(ss.w1_uniform01([0.5, 0.5]), 0.25)

    def test_empirical_best_shrinks_with_n(self):
        p = np.array([.1, .2, .4, .2, .1])
        small = ss.empirical_best(p, 20, rng=np.random.default_rng(0))
        big = ss.empirical_best(p, 2000, rng=np.random.default_rng(0))
        self.assertGreater(small, 3 * big)
        self.assertTrue(np.isnan(ss.empirical_best(p, 0)))
        self.assertEqual(ss.empirical_best([0, 1, 0], 50), 0.0)

    def test_dist_from_responses(self):
        np.testing.assert_allclose(ss.dist_from_responses([1, 2, 2, 5, 9, np.nan, 2.4], 1, 5),
                                   [.2, .6, 0, 0, .2])
        np.testing.assert_allclose(ss.dist_from_responses([], 0, 3), [.25] * 4)


@unittest.skipUnless(PARQUET, NO_PARQUET)
class TestHist(_Fixture):
    def test_build_merges_shards(self):
        h, src = ss.load_hist(os.path.join(self.root, "socsci210"))
        self.assertIn("2 shard(s)", src)
        self.assertEqual(len(h), self.raw.groupby(["study_id", "task_num", "condition_num"]).ngroups)
        g = self.raw[(self.raw.study_id == "s2") & (self.raw.task_num == 1) & (self.raw.condition_num == 3)]
        row = h[(h.study_id == "s2") & (h.task_num == 1) & (h.condition_num == 3)].iloc[0]
        np.testing.assert_array_equal(row.counts, np.bincount(g.response - 1, minlength=5))
        self.assertEqual((row.lo, row.hi, row.levels, row.n, row.n_dropped), (1, 5, 5, len(g), 0))
        self.assertTrue(row.seen)
        self.assertEqual(row.cond_eval, 0.0)

    def test_open_and_off_scale_cells(self):
        h, _ = ss.load_hist(os.path.join(self.root, "socsci210"))
        opn = h[h.task_num == 3]
        self.assertTrue((opn.levels == 0).all() and (opn.n == 0).all() and (opn.in_scale == 0).all())
        off = h[(h.study_id == "s4") & (h.task_num == 2)]
        self.assertTrue(((off.levels == 2) & (off.in_scale < .9)).all())
        self.assertEqual(self.data.meta["cells_unparsed"], 4)
        self.assertGreaterEqual(self.data.meta["cells_off_scale"], 4)
        kept = pd.concat([self.data.train, self.data.test])
        self.assertFalse(((kept.study_id == "s4") & (kept.task_num == 2)).any())
        self.assertFalse((kept.task_num == 3).any())

    def test_write_hist_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "socsci210")
            write_fixture(sub)
            h = ss.write_hist(sub, os.path.join(sub, ss.HIST_FILE))
            self.assertIn("answers", h.columns)
            data = ss.SocSci210().load(d)
            self.assertTrue(data.meta["data"].endswith(ss.HIST_FILE))
            pd.testing.assert_frame_equal(data.test.drop(columns=["P", "counts", "answers", "answer_counts"]),
                                          self.data.test.drop(columns=["P", "counts", "answers", "answer_counts"]),
                                          check_dtype=False)
            for a, b in zip(data.test.P, self.data.test.P):
                np.testing.assert_allclose(a, b)

    def test_old_hist_rejected_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({"study_id": ["x"], "counts": [[1, 2]]}).to_parquet(os.path.join(d, ss.HIST_FILE))
            with self.assertRaises(DataNotFound):
                ss.load_hist(d)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(DataNotFound):
                ss.SocSci210().load(d)
            with self.assertRaises(DataNotFound):
                ss.build_hist([])


@unittest.skipUnless(PARQUET, NO_PARQUET)
class TestTask(_Fixture):
    def test_protocol_and_options(self):
        self.assertIsInstance(self.task, Task)
        self.assertEqual(self.task.kind, "distribution")
        with self.assertRaises(ValueError):
            ss.SocSci210(splitt="study")
        with self.assertRaises(ValueError):
            ss.SocSci210(split="person").load(self.root)
        with self.assertRaises(ValueError):
            ss.SocSci210(baselines="uniform+oracle").baselines(self.data)
        self.assertIn("hist.parquet", self.task.describe()["layout"])

    def test_study_split(self):
        tr, te = self.data.train, self.data.test
        self.assertEqual(set(tr.study_id), set(SEEN))
        self.assertEqual(set(te.study_id), set(UNSEEN))
        self.assertEqual(self.data.meta["studies_test"], 2)
        self.assertTrue(all(len(p) == k and abs(p.sum() - 1) < 1e-12 for p, k in zip(te.P, te.levels)))

    def test_condition_and_outcome_splits(self):
        dc = ss.SocSci210(split="condition").load(self.root)
        self.assertEqual(set(dc.test.condition_num), {4})
        self.assertEqual(set(dc.train.condition_num), {1, 2, 3})
        self.assertEqual(set(dc.test.study_id), {"s1", "s2", "s3", "s5"})
        self.assertEqual(dc.meta["cells_mixed_split"], 0)
        do = ss.SocSci210(split="outcome").load(self.root)
        self.assertEqual(set(do.test.task_num), {2})
        self.assertEqual(set(do.test.study_id) | set(do.train.study_id), {"s1", "s2", "s5"})

    def test_mixed_cell_left_out_of_both_sides(self):
        """A cell whose answers the mapping puts partly in train and partly in eval is counted, not used."""
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "socsci210")
            raw = write_fixture(sub)
            path = os.path.join(sub, "ham", "metadata", "condition_mapping.json")
            with open(path) as f:
                m = json.load(f)
            ids = sorted(raw[(raw.study_id == "s1") & (raw.task_num == 1) & (raw.condition_num == 2)].sample_id)
            m["s1"]["train"] = [i for i in m["s1"]["train"] if i not in ids[:5]]
            m["s1"]["eval"] += ids[:5]
            with open(path, "w") as f:
                json.dump(m, f)
            dc = ss.SocSci210(split="condition").load(d)
            self.assertEqual(dc.meta["cells_mixed_split"], 1)
            both = pd.concat([dc.train, dc.test])
            self.assertFalse(((both.study_id == "s1") & (both.task_num == 1) & (both.condition_num == 2)).any())
            self.assertTrue(((dc.train.study_id == "s1") & (dc.train.task_num == 2) &
                             (dc.train.condition_num == 2)).any())

    def test_filters(self):
        d = ss.SocSci210(min_n=10 ** 6).load(self.root)
        self.assertEqual(len(d.test), 0)
        d = ss.SocSci210(max_levels=2).load(self.root)
        self.assertTrue((d.test.levels == 2).all())
        # s4's binary outcome has no answer on its stated scale: an empty truth vector is never kept, even with
        # both filters off (it used to enter as an all-zero "distribution")
        d = ss.SocSci210(min_in_scale=0, min_n=0).load(self.root)
        kept = pd.concat([d.train, d.test])
        self.assertFalse(((kept.study_id == "s4") & (kept.task_num == 2)).any())
        self.assertTrue((kept.n >= 1).all() and all(abs(p.sum() - 1) < 1e-12 for p in kept.P))

    def test_baselines_and_metrics(self):
        fr = self.task.run(data=self.data)
        self.assertEqual(set(m[5:] for m in fr.columns if m.startswith("pred_")),
                         {"uniform", "scale_marginal", "tfidf_knn"})          # pooled needs same-study train cells
        for col in ("unit", "item", "group", "levels", "segment", "truth", "ref_empirical_best", "ref_uniform01"):
            self.assertIn(col, fr.columns)
        tab = self.task.score_all(fr, B=200)
        self.assertEqual(set(tab.metric), {"w1", "w1_cell", "mean_err"})
        self.assertTrue(np.isfinite(tab.value).all() and (tab.value >= 0).all())
        # the paper's averaging: cell -> study -> overall
        v = np.array([ss.w1(t, np.full(k, 1 / k)) for t, k in zip(fr.truth, fr.levels)])
        want = pd.Series(v).groupby(fr.group.to_numpy()).mean().mean()
        self.assertAlmostEqual(self.task.score(fr, "uniform")["value"], want, places=12)
        self.assertEqual(self.task.score(fr, "uniform")["n"], 2)

    def test_oracle_and_reference(self):
        fr = self.task.run(methods={"oracle": lambda d: list(d.test.P)}, data=self.data, baselines=["uniform"])
        for m in ("w1", "w1_cell", "mean_err"):
            self.assertAlmostEqual(self.task.score(fr, "oracle", m)["value"], 0.0)
        eb = self.task.reference(fr, "empirical_best")
        self.assertTrue(0 < eb["value"] < self.task.score(fr, "uniform")["value"])
        u01 = self.task.reference(fr, "uniform01")
        self.assertGreater(u01["value"], 0)
        with self.assertRaises(KeyError):
            self.task.reference(fr, "best")

    def test_prediction_checks(self):
        bad = lambda d: [np.ones(3)] * len(d.test)   # noqa: E731  wrong length for 5-level and binary cells
        fr = self.task.run(methods={"bad": bad}, data=self.data, baselines=False)
        with self.assertRaises(ValueError):
            self.task.score(fr, "bad")
        fr = self.task.run(methods={"zeros": lambda d: [np.zeros(k) for k in d.test.levels]}, data=self.data,
                           baselines=["uniform"])
        self.assertAlmostEqual(self.task.score(fr, "zeros")["value"], self.task.score(fr, "uniform")["value"])

    def test_pooled_on_condition_split(self):
        task = ss.SocSci210(split="condition")
        d = task.load(self.root)
        fr = task.run(data=d)
        s = {m: task.score(fr, m, B=100)["value"] for m in ("uniform", "pooled", "tfidf_knn")}
        self.assertLess(s["pooled"], s["uniform"])
        self.assertLess(s["tfidf_knn"], s["uniform"])
        only = ss.SocSci210(split="condition", baselines="pooled").baselines(d)
        self.assertEqual(list(only), ["pooled"])

    def test_tfidf_knn_per_item_cap(self):
        tr = pd.DataFrame({"study_id": ["a", "a", "a", "b"], "task_num": [1, 1, 1, 1], "levels": 2,
                           "stimuli": ["red apple pie", "red apple pie", "red apple pie", "red apple tart"],
                           "P": [np.array([1., 0.])] * 3 + [np.array([0., 1.])]})
        te = pd.DataFrame({"study_id": ["c"], "task_num": [1], "levels": [2], "stimuli": ["red apple pie"]})
        np.testing.assert_allclose(ss.tfidf_knn(tr, te, k=2, per_item=1)[0], [.5, .5])
        np.testing.assert_allclose(ss.tfidf_knn(tr, te, k=2, per_item=0)[0], [1., 0.])
        te3 = te.assign(levels=3)
        np.testing.assert_allclose(ss.tfidf_knn(tr, te3, k=2)[0], [1 / 3] * 3)   # no 3-level train cell

    def test_llm_panel_method(self):
        """A persona-panel style method: sample answers from a FakeLLM per test cell, turn them into a vector."""
        def responder(system, user):
            return "2" if "Option A" in user else ("4" if system.endswith("1") else "5")
        llm = FakeLLM(responder)

        def panel(data):
            out = []
            for stim, lo, hi in zip(data.test.stimuli, data.test.lo, data.test.hi):
                answers = [r.text for r in llm.map([(f"persona {i}", stim) for i in range(4)])]
                out.append(ss.dist_from_responses([int(re.findall(r"-?\d+", a)[0]) for a in answers],
                                                  int(lo), int(hi)))
            return out
        fr = self.task.run(methods={"panel": panel}, data=self.data, baselines=["uniform"])
        self.assertEqual(len(llm.calls), 4 * len(self.data.test))
        likert = fr[fr.levels == 5].pred_panel.iloc[0]
        np.testing.assert_allclose(likert, [0, 0, 0, .25, .75])
        self.assertTrue(np.isfinite(self.task.score(fr, "panel")["value"]))


@unittest.skipUnless(PARQUET, NO_PARQUET)
@unittest.skipUnless(os.environ.get("SIMENGINE_DATA"), "$SIMENGINE_DATA not set")
class TestRealData(unittest.TestCase):
    def test_study_split_shape(self):
        task = ss.SocSci210(baselines="uniform")
        try:
            data = task.load()
        except DataNotFound:
            self.skipTest("socsci210 not under $SIMENGINE_DATA")
        self.assertGreater(len(data.test), 0)
        self.assertTrue(set(data.train.study_id).isdisjoint(data.test.study_id))
        if data.meta["data"].endswith(ss.HIST_FILE):              # all 17 shards aggregated
            self.assertLessEqual(data.meta["studies_test"], 40)
        fr = task.run(data=data)
        r = task.score(fr, "uniform", B=100)
        self.assertTrue(0 < r["value"] < 0.5)


if __name__ == "__main__":
    unittest.main()

"""Persona builders: codebook value labels in `from_table` / `from_row` (codes -> labels before the enum mapping and
the text, one warning per column for unknown codes), `read_codebook`, and persona pool selection (`pool_select`,
`pool_curve`)."""
import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.persona import (POOL_METHODS, Mapping, apply_value_labels, from_row, from_table, pool_curve,
                               pool_select, read_codebook)

GENDER = {1: "male", 2: "female", 3: "other"}
EDU = {1: "primary", 2: "secondary", 3: "university"}


def survey():
    return pd.DataFrame({"rid": [1, 2, 3, 4, 5],
                         "Gender": [2, 1, "2", 2.0, 9],               # mixed types; 9 has no label
                         "Education": ["01", 3, None, 2, 3],
                         "visits": [3, 5, 1, 0, 2],
                         "target": [1, 0, 1, 0, 1]})


def user_warnings(w):
    return [str(x.message) for x in w if issubclass(x.category, UserWarning)]


class TestValueLabels(unittest.TestCase):
    def test_codes_become_labels_before_enum_and_text(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ps = from_table(survey(), id_col="rid", forbidden=["target"],
                            value_labels={"Gender": GENDER, "Education": EDU})
        self.assertEqual([p.identity.gender for p in ps[:4]], ["female", "male", "female", "female"])
        self.assertIsNone(ps[4].identity.gender)                    # 9 kept as is -> not a TALP gender
        self.assertEqual(ps[0].facets["Education"], "primary")     # "01" == 1
        self.assertNotIn("Education", ps[2].facets)                # empty stays empty
        txt = ps[0].text()
        self.assertIn("female", txt)
        self.assertNotIn("Gender: 2", txt)
        self.assertIn("primary", txt)
        msgs = [m for m in user_warnings(w) if "value_labels" in m]
        self.assertEqual(len(msgs), 1)                               # one warning per column, not per row
        self.assertIn("'Gender'", msgs[0])
        self.assertIn("1 of 5 rows", msgs[0])
        self.assertIn("'9' x1", msgs[0])

    def test_string_keys_match_numeric_cells_and_ids_untouched(self):
        df = survey().assign(rid=[2, 1, 3, 4, 5])
        ps = from_table(df.iloc[:4], id_col="rid", forbidden=["target"],
                        value_labels={"Gender": {"1": "male", "2.0": "female"}, "rid": {1: "x", 2: "y"}})
        self.assertEqual([p.identity.gender for p in ps], ["female", "male", "female", "female"])
        self.assertEqual([p.pid for p in ps], ["2", "1", "3", "4"])

    def test_mapping_value_labels_and_kwarg_merge(self):
        m = Mapping(id_col="rid", fields={"gender": "Gender"}, facets={"edu": "Education"},
                    value_labels={"Gender": GENDER, "Education": {1: "old"}})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ps = from_table(survey(), m, value_labels={"Education": EDU})
        self.assertEqual(ps[1].identity.gender, "male")
        self.assertEqual(ps[0].facets["edu"], "primary")    # kwarg wins per column
        self.assertEqual(m.value_labels["Education"], {1: "old"})                      # caller's Mapping unchanged

    def test_from_row_dict_and_series(self):
        m = Mapping(id_col="rid", fields={"gender": "Gender"}, value_labels={"Gender": GENDER})
        self.assertEqual(from_row({"rid": "a", "Gender": 2}, m).identity.gender, "female")
        self.assertEqual(from_row(pd.Series({"rid": "b", "Gender": np.int64(1)}), m).identity.gender, "male")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p = from_row({"rid": "c", "Gender": 7}, m)
        self.assertIsNone(p.identity.gender)
        self.assertEqual(len([m for m in user_warnings(w) if "value_labels" in m]), 1)

    def test_no_labels_keeps_old_behaviour(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ps = from_table(survey(), id_col="rid", forbidden=["target"])
        self.assertIsNone(ps[0].identity.gender)                    # code 2 is not a gender spelling
        self.assertEqual(ps[0].facets["Education"], "01")
        self.assertFalse([m for m in user_warnings(w) if "value_labels" in m])

    def test_codebook_gender_wordings_reach_the_enum(self):
        # survey codebooks spell the labels their own way; auto_mapping's gender_value must still read them
        labels = {"Gender": {1: "Male", 2: "Female", 3: "Non-binary / third gender", 4: "Prefer not to say"}}
        df = pd.DataFrame({"rid": [1, 2, 3, 4], "Gender": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
        ps = from_table(df, id_col="rid", forbidden=["target"], value_labels=labels)
        self.assertEqual([p.identity.gender for p in ps], ["male", "female", "other", None])

    def test_conflicting_codes_raise(self):
        with self.assertRaises(ValueError):
            from_table(survey(), id_col="rid", value_labels={"Gender": {1: "male", "1": "female"}})

    def test_apply_value_labels(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = apply_value_labels(survey(), {"Gender": GENDER, "Missing": {1: "a"}}, skip=("rid",))
        self.assertEqual(out["Gender"].tolist(), ["female", "male", "female", "female", 9])
        self.assertEqual(survey()["Gender"].tolist()[0], 2)
        msgs = user_warnings(w)
        self.assertTrue(any("'Missing'" in m for m in msgs))
        self.assertTrue(any("'9' x1" in m for m in msgs))


class TestCodebook(unittest.TestCase):
    def test_three_columns_with_blank_variable_cells(self):
        cb = pd.DataFrame({"variable": ["Gender", "", "", "Education", None],
                           "code": ["1", "2", "3", "01", 2],
                           "label": ["male", "female ", "other", "primary", "secondary"]})
        labels = read_codebook(cb)
        self.assertEqual(labels, {"Gender": {1: "male", 2: "female", 3: "other"},
                                  "Education": {1: "primary", 2: "secondary"}})
        ps = from_table(survey().iloc[[0, 3]], id_col="rid", value_labels=labels, forbidden=["target"])
        self.assertEqual(ps[0].identity.gender, "female")

    def test_two_columns_and_csv(self):
        cb = pd.DataFrame({"code": [1, 2], "label": ["male", "female"]})
        self.assertEqual(read_codebook(cb, variable="Gender"), {"Gender": {1: "male", 2: "female"}})
        with self.assertRaises(ValueError):
            read_codebook(cb)
        wide = pd.DataFrame({"note": ["x", "y"], "var": ["G", "G"], "c": ["1", "2"], "lab": ["m", "f"]})
        self.assertEqual(read_codebook(wide, columns=("var", "c", "lab")), {"G": {1: "m", 2: "f"}})
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cb.csv")
            pd.DataFrame({"variable": ["G", "G"], "code": ["01", "02"], "label": ["m", "f"]}).to_csv(p, index=False)
            self.assertEqual(read_codebook(p), {"G": {1: "m", 2: "f"}})

    def test_two_labels_for_one_code_raise(self):
        with self.assertRaises(ValueError):
            read_codebook(pd.DataFrame({"v": ["G", "G"], "c": ["1", "1.0"], "l": ["m", "f"]}))


def people(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"uid": [f"u{i}" for i in range(n)],
                         "region": rng.choice(["north", "south", "east"], n, p=[0.6, 0.3, 0.1]),
                         "visits": rng.poisson(4, n), "spend": rng.gamma(2.0, 50.0, n)},
                        index=pd.RangeIndex(100, 100 + n))


class TestPoolSelect(unittest.TestCase):
    def test_random_is_seeded_nested_and_in_table_order(self):
        df = people()
        a, b = pool_select(df, 20, seed=3), pool_select(df, 20, seed=3)
        self.assertEqual(list(a), list(b))
        self.assertEqual(len(set(a)), 20)
        self.assertTrue(set(pool_select(df, 10, seed=3)) <= set(a))     # nested pools for one seed
        self.assertEqual(list(a), sorted(a))
        self.assertTrue(set(a) <= set(df.index))
        self.assertNotEqual(list(a), list(pool_select(df, 20, seed=4)))

    def test_stratified_is_proportional(self):
        df = people()
        idx = pool_select(df, 20, method="stratified", strata="region", seed=1)
        share = df["region"].value_counts(normalize=True)
        got = df.loc[idx, "region"].value_counts()
        for k, p in share.items():
            self.assertLessEqual(abs(got.get(k, 0) - 20 * p), 1)
        idx2 = pool_select(df, 30, method="stratified", strata=["region", "visits"], seed=1)
        self.assertEqual(len(set(idx2)), 30)
        with self.assertRaises(ValueError):
            pool_select(df, 5, method="stratified")

    def test_kmeans_picks_one_row_per_cluster(self):
        rng = np.random.default_rng(0)
        centers = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
        X = np.vstack([c + rng.normal(0, 0.3, (25, 2)) for c in centers])
        df = pd.DataFrame(X, columns=["a", "b"]).assign(kind=["x"] * 100)
        idx = pool_select(df, 4, method="kmeans", features=["a", "b"], seed=0)
        self.assertEqual(len(idx), 4)
        blob = sorted(int(i) // 25 for i in idx)
        self.assertEqual(blob, [0, 1, 2, 3])                          # every blob represented once
        self.assertEqual(list(idx), list(pool_select(df, 4, method="kmeans", features=["a", "b"], seed=0)))
        with self.assertRaises(ValueError):
            pool_select(df, 4, method="kmeans")                        # features are required

    def test_kmeans_text_features_nan_and_duplicates(self):
        df = pd.DataFrame({"region": ["n", "n", "s", "s", None, "e"], "v": [1.0, 1.0, 1.0, np.nan, 1.0, 1.0]})
        idx = pool_select(df, 5, method="kmeans", features=["region", "v"], seed=2)
        self.assertEqual(len(set(idx)), 5)
        same = pd.DataFrame({"v": [1.0] * 10})
        self.assertEqual(len(set(pool_select(same, 4, method="kmeans", features="v"))), 4)

    def test_edges(self):
        df = people(30)
        self.assertEqual(list(pool_select(df, 30)), list(df.index))
        self.assertEqual(list(pool_select(df, 99, method="kmeans", features=["visits"])), list(df.index))
        for bad in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                pool_select(df, bad)
        with self.assertRaises(ValueError):
            pool_select(df, 3, method="greedy")
        with self.assertRaises(ValueError):
            pool_select(pd.concat([df, df]), 3)                       # duplicated index labels
        self.assertEqual(POOL_METHODS, ("random", "stratified", "kmeans"))

    def test_pool_feeds_from_table(self):
        df = people()
        ps = from_table(df.loc[pool_select(df, 12, method="stratified", strata="region")], id_col="uid")
        self.assertEqual(len(ps), 12)


class TestPoolCurve(unittest.TestCase):
    def test_summary_and_long(self):
        df = people()
        truth = df["region"].value_counts(normalize=True)

        def score(pool):                                              # 1 - TVD of the region mix
            got = pool["region"].value_counts(normalize=True).reindex(truth.index, fill_value=0)
            return 1 - 0.5 * float((got - truth).abs().sum())

        curve = pool_curve(df, [5, 20, 80, 500], score, method=["random", "stratified"], draws=6, strata="region")
        self.assertEqual(list(curve.columns), ["method", "size", "draws", "mean", "lo", "hi"])
        self.assertEqual(len(curve), 8)
        full = curve[curve["size"] == 500]
        self.assertTrue((full["draws"] == 1).all())
        self.assertTrue(np.allclose(full["mean"], 1.0))
        r = curve[curve["method"] == "random"].set_index("size")
        self.assertGreater(r.loc[80, "mean"], r.loc[5, "mean"])
        self.assertTrue((curve["lo"] <= curve["mean"]).all() and (curve["mean"] <= curve["hi"]).all())
        raw = pool_curve(df, [5, 20], score, draws=3, long=True)
        self.assertEqual(list(raw.columns), ["method", "size", "seed", "score"])
        self.assertEqual(len(raw), 6)
        self.assertEqual(raw["seed"].tolist(), [0, 1, 2, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()

"""simengine.methods.sessions, D13-W8 rehearsal gaps: label column name (y_col=), unlabeled prefix rows
(label=None), several labels in one pass (label={name: kind}), the train / test split column and prefix_features
(window popularity of the prefix items, hour, weekday, same item)."""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.methods.sessions import cutoff_windows, prefix_features, sessionize

T0 = pd.Timestamp("2026-01-01")


def _log(seed=0, users=60, days=120, items=15):
    """Random click log: per user a few sessions of views, sometimes an add-to-cart and a purchase."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(users):
        for _ in range(rng.integers(1, 8)):
            t = T0 + pd.Timedelta(minutes=int(rng.integers(0, days * 1440)))
            for _ in range(rng.integers(1, 6)):
                t += pd.Timedelta(minutes=int(rng.integers(0, 10)))
                it = int(rng.integers(0, items))
                rows.append((u, t, "view", it))
                if rng.random() < 0.2:
                    rows.append((u, t, "addtocart", it))
                    if rng.random() < 0.4:
                        rows.append((u, t, "transaction", it))
    return pd.DataFrame(rows, columns=["user", "time", "event", "item"])


class _Base(unittest.TestCase):
    def setUp(self):
        self.ev = sessionize(_log(), "user", "time", kind_col="event")
        self.cuts = [T0 + pd.Timedelta(days=d) for d in range(28, 100, 7)]

    def _rows(self, **kw):
        a = dict(prefix_len=2, label="transaction", kind_col="event", item_col="item")
        a.update(kw)
        return cutoff_windows(self.ev, self.cuts, "21D", **a)


class TestLabelColumn(_Base):
    def test_y_col_names_the_label(self):
        # D13-W8 #1: evaluate reads `truth`; the label landed in `y` and the card came out empty without a word
        r = self._rows()
        t = self._rows(y_col="truth")
        self.assertIn("y", r.columns)                                          # default unchanged
        self.assertNotIn("truth", r.columns)
        self.assertNotIn("y", t.columns)
        self.assertEqual(t["truth"].tolist(), r["y"].tolist())
        with self.assertRaises(ValueError):
            self._rows(y_col="cutoff")                                          # would overwrite a column

    def test_label_none_gives_prefix_rows(self):
        # #3: test prefixes (first two events, no future) -> feature rows without a label and without the censor
        # warning, which only concerns labels
        r = self._rows()
        n = self._rows(label=None)
        self.assertNotIn("y", n.columns)
        pd.testing.assert_frame_equal(n, r.drop(columns="y"))
        late = self.ev["time"].max() - pd.Timedelta("5D")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p = cutoff_windows(self.ev, [late], "21D", label=None, kind_col="event", item_col="item")
        self.assertFalse([x for x in w if "censored" in str(x.message)])
        self.assertGreater(len(p), 0)
        # without kind_col: no kind label, so label=None needs none
        cutoff_windows(self.ev, self.cuts, "21D", label=None)

    def test_label_dict_one_pass(self):
        # #4: p_buy and p_cart needed two passes (features computed twice)
        calls = []

        def feats(hist, rows, at):
            calls.append(at)
            return pd.DataFrame({"n_hist": len(hist)}, index=rows.index)
        lab = {"buy": "transaction", "cart": "addtocart", "n": lambda rest: len(rest)}
        d = self._rows(label=lab, features=feats)
        self.assertEqual(len(calls), len(self.cuts))
        self.assertNotIn("y", d.columns)
        self.assertEqual(d["buy"].tolist(), self._rows()["y"].tolist())
        self.assertEqual(d["cart"].tolist(), self._rows(label="addtocart")["y"].tolist())
        self.assertEqual(d["n"].tolist(), d["n_rest"].tolist())
        self.assertGreater(d["cart"].sum(), 0)                                   # add-to-carts occur
        tr, dv = self._rows(label=lab, eval_gap="0D")
        for c in ("buy", "cart", "n"):
            self.assertIn(c, tr.columns)
            self.assertIn(c, dv.columns)
        with self.assertRaises(ValueError):
            self._rows(label=lab, y_col="truth")                               # a dict names its own columns
        with self.assertRaises(ValueError):
            self._rows(label={})
        with self.assertRaises(ValueError):
            self._rows(label={"session_id": "transaction"})
        with self.assertRaises(TypeError):
            self._rows(label={"buy": 3})
        with self.assertRaises(ValueError):
            cutoff_windows(self.ev, self.cuts, "21D", label={"buy": "transaction"})   # kind label needs kind_col


class TestSplitColumn(_Base):
    def test_split_column_with_eval_gap(self):
        # #8: the card flagged "split information skipped" because the time split never reached it
        tr, dv = self._rows(eval_gap="0D")
        self.assertEqual(set(tr["split"]), {"train"})
        self.assertEqual(set(dv["split"]), {"test"})
        self.assertIn("cutoff", dv.columns)
        self.assertNotIn("split", self._rows().columns)                        # no eval_gap: no split
        tr2, dv2 = self._rows(eval_gap="0D", split_col=None)
        self.assertNotIn("split", tr2.columns)
        self.assertNotIn("split", dv2.columns)
        _, dv3 = self._rows(eval_gap="0D", split_col="bolum")
        self.assertEqual(set(dv3["bolum"]), {"test"})

    def test_split_values_are_ones_evaluate_reads(self):
        # 'eval' would not be a test name for evaluate: the dev rows would be read as training rows
        from simengine.evidence.api import COLUMN_TO_INTERNAL, SPLIT_TO_INTERNAL
        from simengine.evidence.sema import TEST_ADLARI
        self.assertEqual(COLUMN_TO_INTERNAL["split"], "bolum")
        self.assertIn("test", TEST_ADLARI)
        self.assertNotIn(SPLIT_TO_INTERNAL.get("train", "train"), TEST_ADLARI)


class TestPrefixFeatures(_Base):
    def test_columns_and_values(self):
        r = self._rows()
        f = prefix_features(r, item_col="item")
        for c in ("prefix_pop_item0", "prefix_pop_item1", "prefix_hour", "prefix_weekday", "prefix_same_item",
                  "prefix_days_after_cutoff"):
            self.assertIn(c, f.columns)
        self.assertEqual(len(f), len(r))
        pd.testing.assert_frame_equal(f[r.columns], r)                          # rows kept, only columns added
        self.assertEqual(f["prefix_same_item"].tolist(), (r["item0"] == r["item1"]).astype(int).tolist())
        self.assertEqual(f["prefix_hour"].tolist(), (r["time0"].dt.hour + r["time0"].dt.minute / 60
                                                     + r["time0"].dt.second / 3600).tolist())
        self.assertEqual(f["prefix_weekday"].tolist(), r["time0"].dt.weekday.tolist())
        # popularity: share of the cutoff's rows whose prefix holds the item (each prefix position counted)
        T = r["cutoff"].iloc[0]
        g = r[r["cutoff"] == T]
        cnt = pd.concat([g["item0"], g["item1"]]).value_counts()
        row = f[f["cutoff"] == T].iloc[0]
        self.assertAlmostEqual(row["prefix_pop_item0"], cnt[row["item0"]] / len(g))
        self.assertTrue(((f["prefix_days_after_cutoff"] >= 0) & (f["prefix_days_after_cutoff"] < 21)).all())

    def test_popularity_stays_inside_the_cutoff(self):
        r = self._rows()
        f = prefix_features(r, item_col="item")
        T = r["cutoff"].iloc[0]
        g = prefix_features(r[r["cutoff"] == T].reset_index(drop=True), item_col="item")
        pd.testing.assert_series_equal(f.loc[f["cutoff"] == T, "prefix_pop_item0"].reset_index(drop=True),
                                       g["prefix_pop_item0"])

    def test_numeric_time(self):
        log = _log().assign(time=lambda d: (d["time"] - pd.Timestamp(0)) // pd.Timedelta("1ms"))
        ev = sessionize(log, "user", "time", kind_col="event", time_unit="ms")
        cuts = [(c - pd.Timestamp(0)) // pd.Timedelta("1ms") for c in self.cuts]
        r = cutoff_windows(ev, cuts, "21D", label="transaction", kind_col="event", item_col="item", time_unit="ms")
        f = prefix_features(r, item_col="item", time_unit="ms")
        g = prefix_features(self._rows(), item_col="item")
        np.testing.assert_allclose(f["prefix_hour"], g["prefix_hour"])
        self.assertEqual(f["prefix_weekday"].tolist(), g["prefix_weekday"].tolist())
        np.testing.assert_allclose(f["prefix_days_after_cutoff"], g["prefix_days_after_cutoff"])
        with self.assertRaises(ValueError):
            prefix_features(r, item_col="item")                                # epoch unit unknown

    def test_without_items(self):
        r = cutoff_windows(self.ev, self.cuts, "21D", label=None)
        f = prefix_features(r)
        self.assertIn("prefix_hour", f.columns)
        self.assertNotIn("prefix_same_item", f.columns)

    def test_empty_windows(self):
        # review: a window with no session gave a float64 cutoff column and prefix_features raised on datetime - float
        far = self.ev["time"].max() + pd.Timedelta("400D")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for lab in (None, "transaction", {"buy": "transaction", "n": len}):
                r = cutoff_windows(self.ev, [far], "21D", label=lab, kind_col="event", item_col="item")
                self.assertEqual(len(r), 0)
                self.assertTrue(pd.api.types.is_datetime64_any_dtype(r["cutoff"]))
                f = prefix_features(r, item_col="item")
                self.assertEqual(len(f), 0)
                self.assertIn("prefix_days_after_cutoff", f.columns)

    def test_missing_item_is_not_same(self):
        # review: nunique(axis=1) skipped NaN, so (NaN, "c") counted as one distinct item -> same_item 1
        r = self._rows().head(4).copy()
        r["item1"] = r["item0"]
        r.loc[0, "item0"] = np.nan
        r.loc[1, ["item0", "item1"]] = np.nan
        f = prefix_features(r, item_col="item")
        self.assertEqual(f["prefix_same_item"].tolist(), [0, 0, 1, 1])
        self.assertTrue(np.isnan(f.loc[0, "prefix_pop_item0"]))                # no item, no popularity


if __name__ == "__main__":
    unittest.main()

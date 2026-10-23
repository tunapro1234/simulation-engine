"""simengine.methods.sessions on synthetic click logs: sessionize, cutoff windows (+ non-overlapping eval split),
pre-cutoff history features (no future leak) and rate rescaling."""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.methods import cutoff_windows, history_features, rescale_to_rates, sessionize

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
    df = pd.DataFrame(rows, columns=["user", "time", "event", "item"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)      # shuffled, as logs often are


class TestSessionize(unittest.TestCase):
    def test_gap_and_users(self):
        m = pd.Timedelta(minutes=1)
        df = pd.DataFrame({"user": [1, 1, 1, 2, 1], "time": [T0, T0 + 10 * m, T0 + 41 * m, T0 + 11 * m, T0 + 70 * m]})
        ev = sessionize(df, "user", "time", gap="30min")
        self.assertEqual(ev["session_id"].tolist(), [0, 0, 1, 1, 2])
        self.assertEqual(ev["user"].tolist(), [1, 1, 1, 1, 2])
        self.assertEqual(ev["seq"].tolist(), [0, 1, 0, 1, 0])
        self.assertEqual(ev["session_start"].iloc[3], T0 + 41 * m)           # 70 - 41 = 29 min: same session
        self.assertEqual(len(df), 5)
        self.assertNotIn("session_id", df.columns)                              # input not modified

    def test_gap_is_strict(self):
        df = pd.DataFrame({"user": [1, 1], "time": [T0, T0 + pd.Timedelta(minutes=30)]})
        self.assertEqual(sessionize(df, "user", "time")["session_id"].nunique(), 1)

    def test_ties_ordered_by_kind(self):
        df = pd.DataFrame({"user": [7, 7, 7, 7], "time": [T0] * 3 + [T0 - pd.Timedelta(seconds=1)],
                           "event": ["transaction", "view", "addtocart", "view"]})
        ev = sessionize(df, "user", "time", kind_col="event")
        self.assertEqual(ev["event"].tolist(), ["view", "view", "addtocart", "transaction"])
        unk = sessionize(df.assign(event=["transaction", "zoom", "addtocart", "view"]), "user", "time",
                         kind_col="event")
        self.assertEqual(unk["event"].tolist(), ["view", "addtocart", "transaction", "zoom"])  # unknown kind last

    def test_numeric_time(self):
        df = pd.DataFrame({"user": [1, 1, 1], "time": [0, 29 * 60_000, 60 * 60_000]})
        self.assertEqual(sessionize(df, "user", "time", gap="30min", time_unit="ms")["session_id"].tolist(),
                         [0, 0, 1])
        self.assertEqual(sessionize(df, "user", "time", gap=30 * 60_000)["session_id"].tolist(), [0, 0, 1])
        with self.assertRaises(ValueError):
            sessionize(df, "user", "time", gap="30min")                          # unit unknown: refuse


class TestCutoffWindows(unittest.TestCase):
    def setUp(self):
        self.ev = sessionize(_log(), "user", "time", kind_col="event")
        self.cuts = [T0 + pd.Timedelta(days=d) for d in range(28, 100, 7)]

    def _rows(self, **kw):
        a = dict(prefix_len=2, label="transaction", kind_col="event", item_col="item")
        a.update(kw)
        return cutoff_windows(self.ev, self.cuts, "21D", **a)

    def test_one_row_per_cutoff_user_and_window(self):
        r = self._rows()
        self.assertGreater(len(r), 0)
        self.assertFalse(r.duplicated(["cutoff", "user"]).any())
        self.assertTrue((r["session_start"] >= r["cutoff"]).all())
        self.assertTrue((r["session_start"] < r["cutoff"] + pd.Timedelta("21D")).all())
        self.assertTrue((r["label_end"] < r["cutoff"] + pd.Timedelta("21D")).all())  # truncated at window end
        for c in ("time0", "time1", "item0", "item1", "event0", "event1", "prefix_span", "y", "n_rest"):
            self.assertIn(c, r.columns)

    def test_first_session_after_cutoff(self):
        r = self._rows()
        heads = self.ev[self.ev["seq"] == 0]
        for _, row in r.sample(20, random_state=1, replace=True).iterrows():
            h = heads[(heads["user"] == row["user"]) & (heads["session_start"] >= row["cutoff"])]
            self.assertEqual(h["session_start"].min(), row["session_start"])

    def test_label_from_rest_and_prefix_kinds(self):
        r = self._rows(prefix_kinds=("view",))
        self.assertTrue(((r["event0"] == "view") & (r["event1"] == "view")).all())
        for _, row in r.head(30).iterrows():
            s = self.ev[(self.ev["session_id"] == row["session_id"]) & (self.ev["seq"] >= 2)
                        & (self.ev["time"] < row["cutoff"] + pd.Timedelta("21D"))]
            self.assertEqual(row["y"], int((s["event"] == "transaction").any()))
        rc = self._rows(prefix_kinds=("view",), label=lambda rest: len(rest))
        self.assertEqual(rc["y"].tolist(), rc["n_rest"].tolist())

    def test_max_end_offset(self):
        r = self._rows(max_end_offset="3D")
        self.assertTrue((r["session_start"] <= r["cutoff"] + pd.Timedelta("18D")).all())

    def test_features_hook_sees_only_history(self):
        seen = []

        def feats(hist, rows, at):
            seen.append(hist["time"].max() < at)
            return pd.DataFrame({"n_hist": [len(hist)] * len(rows)}, index=rows.index)
        r = self._rows(features=feats)
        self.assertTrue(all(seen) and len(seen) == len(self.cuts))
        self.assertIn("n_hist", r.columns)

    def test_overlapping_windows_leak_without_eval_gap(self):
        # the D13 trap: 21-day windows every 7 days; eval = last cutoff, train = the rest -> shared sessions
        r = self._rows()
        last = r["cutoff"].max()
        shared = set(r.loc[r["cutoff"] < last, "session_id"]) & set(r.loc[r["cutoff"] == last, "session_id"])
        self.assertGreater(len(shared), 0)

    def test_eval_gap_split_does_not_overlap(self):
        train, ev = self._rows(eval_gap="0D")
        te = ev["cutoff"].iloc[0]
        self.assertTrue((ev["cutoff"] == te).all())
        self.assertFalse(set(train["session_id"]) & set(ev["session_id"]))
        self.assertTrue((train["label_end"] < te).all())
        self.assertTrue((train["cutoff"] + pd.Timedelta("21D") <= te).all())
        self.assertEqual(sorted(train.attrs["dropped_cutoffs"]), [te - pd.Timedelta(days=d) for d in (14, 7)])
        train2, _ = self._rows(eval_gap="7D")
        self.assertTrue((train2["cutoff"] + pd.Timedelta("28D") <= te).all())
        with self.assertRaises(ValueError):
            self._rows(eval_gap="0D", truncate=False)          # labels could reach into the eval range
        with self.assertRaises(ValueError):
            cutoff_windows(self.ev, self.cuts[-2:], "21D", label="transaction", kind_col="event", eval_gap="0D")

    def test_needs_sessionized_events(self):
        with self.assertRaises(ValueError):
            cutoff_windows(_log(), self.cuts, "21D", label="transaction", kind_col="event")


class TestHistoryFeatures(unittest.TestCase):
    def setUp(self):
        self.ev = sessionize(_log(1), "user", "time", kind_col="event")
        self.at = T0 + pd.Timedelta(days=70)
        self.kinds = ("view", "addtocart", "transaction")

    def test_counts_rates_days(self):
        f = history_features(self.ev, self.at, kinds=self.kinds)
        h = self.ev[self.ev["time"] < self.at]
        u = f["user"]
        self.assertEqual(int(u["user_n_view"].sum()), int((h["event"] == "view").sum()))
        u0 = h[h["user"] == u.index[0]]
        self.assertEqual(u.iloc[0]["user_n_transaction"], (u0["event"] == "transaction").sum())
        self.assertAlmostEqual(u.iloc[0]["user_days_since_last"], (self.at - u0["time"].max()) / pd.Timedelta("1D"))
        self.assertTrue((u["user_days_since_last"] > 0).all())
        h28 = h[h["time"] >= self.at - pd.Timedelta("28D")]
        self.assertEqual(int(f["item"]["item_n_view_28D"].sum()), int((h28["event"] == "view").sum()))
        it = f["item"]
        prior = (h["event"] == "transaction").sum() / (h["event"] == "view").sum()
        exp = (it["item_n_transaction"] + 2.0 * prior) / (it["item_n_view"] + 2.0)
        np.testing.assert_allclose(it["item_rate_transaction"], exp)
        pair = f[("user", "item")]
        self.assertEqual(int(pair["user_item_n_view"].sum()), int((h["event"] == "view").sum()))
        self.assertIn("user_n_sessions", u.columns)

    def test_attach_to_rows(self):
        rows = pd.DataFrame({"user": [0, 1, 10_000], "item0": [1, 2, 3], "item1": [4, 99_999, 5]})
        r = history_features(self.ev, self.at, rows, on={"user": "user", "item": ["item0", "item1"]},
                             kinds=self.kinds)
        self.assertEqual(len(r), 3)
        for c in ("user_n_view", "item_n_view_item0", "item_rate_transaction_item1", "user_item_n_view_item0",
                  "user_seen", "item_seen_item1"):
            self.assertIn(c, r.columns)
        unseen = r.iloc[2]
        self.assertEqual(unseen["user_n_view"], 0)
        self.assertEqual(unseen["user_seen"], 0)
        self.assertTrue(np.isnan(unseen["user_days_since_last"]))
        self.assertEqual(r.iloc[1]["item_n_view_item1"], 0)
        self.assertGreater(r.iloc[1]["item_rate_transaction_item1"], 0)          # unseen item -> prior rate

    def test_no_future_leak(self):
        rows = pd.DataFrame({"user": range(60), "item0": np.arange(60) % 15})
        on = {"user": "user", "item": "item0"}
        a = history_features(self.ev, self.at, rows, on=on, kinds=self.kinds)
        fut = self.ev["time"] >= self.at
        self.assertGreater(int(fut.sum()), 0)
        rng = np.random.default_rng(5)
        ev2 = self.ev.copy()
        ev2.loc[fut, "event"] = rng.choice(list(self.kinds), int(fut.sum()))
        ev2.loc[fut, "item"] = rng.integers(0, 15, int(fut.sum()))
        extra = ev2[fut].assign(time=self.at)                                     # events AT the cutoff are future
        b = history_features(pd.concat([ev2, extra], ignore_index=True), self.at, rows, on=on, kinds=self.kinds)
        pd.testing.assert_frame_equal(a, b)

    def test_window_and_numeric_time(self):
        ev = self.ev.assign(time=(self.ev["time"] - T0) // pd.Timedelta("1ms"))
        at = (self.at - T0) // pd.Timedelta("1ms")
        f = history_features(ev, at, kinds=self.kinds, time_unit="ms")
        g = history_features(self.ev, self.at, kinds=self.kinds)
        pd.testing.assert_frame_equal(f["user"], g["user"])
        with self.assertRaises(ValueError):
            history_features(ev, at, kinds=self.kinds)                             # days need a unit


class TestRescale(unittest.TestCase):
    def test_segment_means_and_ranking(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 0.1, 400)
        seg = np.repeat(["a", "b"], 200)
        q = rescale_to_rates(p, seg, {"a": 0.02, "b": 0.01})
        self.assertAlmostEqual(q[:200].mean(), 0.02)
        self.assertAlmostEqual(q[200:].mean(), 0.01)
        for s in (slice(0, 200), slice(200, 400)):
            self.assertTrue((np.argsort(q[s], kind="stable") == np.argsort(p[s], kind="stable")).all())

    def test_total_series_and_logit(self):
        p = pd.Series(np.linspace(0.01, 0.99, 50), index=range(100, 150))
        q = rescale_to_rates(p, None, 0.05)
        self.assertIsInstance(q, pd.Series)
        self.assertTrue(q.index.equals(p.index))
        self.assertAlmostEqual(q.mean(), 0.05)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = rescale_to_rates(p, None, 0.9)                                    # scale would pass 1 -> clip
        self.assertTrue(any("clip" in str(x.message) for x in w))
        self.assertLessEqual(c.max(), 1.0)
        lg = rescale_to_rates(p, None, 0.9, method="logit")
        self.assertAlmostEqual(lg.mean(), 0.9, places=6)
        self.assertTrue((np.diff(lg.to_numpy()) > 0).all())

    def test_missing_segment_and_default(self):
        p = np.array([0.1, 0.2, 0.3, 0.4])
        seg = ["a", "a", "z", "z"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            q = rescale_to_rates(p, seg, {"a": 0.3})
        self.assertTrue(any("z" in str(x.message) for x in w))
        np.testing.assert_allclose(q, [0.2, 0.4, 0.3, 0.4])
        q2 = rescale_to_rates(p, seg, {"a": 0.3}, default=0.07)
        self.assertAlmostEqual(q2[2:].mean(), 0.07)
        with self.assertRaises(TypeError):
            rescale_to_rates(p, seg, 0.1)                                         # scalar target with segments


if __name__ == "__main__":
    unittest.main()

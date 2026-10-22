"""retailrocket bench tasks on synthetic events (hand-made sessions + random filler); the real-data check is skipped
unless $SIMENGINE_DATA is set."""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from simengine import bench
from simengine.bench.base import DataNotFound
from simengine.bench.tasks import retailrocket as rr
from simengine.bench.tasks.retailrocket import RetailRocketNext, RetailRocketPurchase

MIN = 60_000
T_TEST = rr._ms(pd.Timestamp("2015-09-06 10:00"))


def _hand():
    """visitor -> (minute offset, event, item) in the test window."""
    rows = []
    add = lambda v, m, e, i: rows.append((T_TEST + v * 10**7 + int(m * MIN), v, e, i))
    for m, e, i in [(0, "view", 1), (1, "view", 2), (2, "addtocart", 2), (3, "transaction", 2), (120, "view", 3)]:
        add(1, m, e, i)                               # session 0: prefix 3, y=1; session 1: one event
    add(2, 0, "view", 1); add(2, 1, "transaction", 1)  # prefix 1 -> not evaluated (leak filter)
    for m, i in [(0, 1), (1, 1), (2, 2), (3, 2)]:
        add(3, m, "view", i)                          # runs 1,2 -> target 2, input_end 1, last 1, new
    for m, i in [(0, 1), (30, 2), (31, 1)]:
        add(4, m, "view", i)                          # gap of exactly 30 min stays one session; target 1 = repeat
    add(5, 0, "transaction", 4); add(5, 1, "view", 4)  # first event transaction -> prefix 0
    add(6, 0, "view", 1); add(6, 31, "view", 2)        # gap > 30 min -> two one-event sessions
    return pd.DataFrame(rows, columns=["timestamp", "visitorid", "event", "itemid"])


def _train_hand():
    """visitor 7 in the train window: two views of item 999 (seen by nobody else), then a purchase."""
    t = rr._ms(pd.Timestamp("2015-07-10 12:00"))
    return pd.DataFrame([(t, 7, "view", 999), (t + MIN, 7, "view", 999), (t + 2 * MIN, 7, "transaction", 999)],
                        columns=["timestamp", "visitorid", "event", "itemid"])


def _filler(n_vis=900, seed=0):
    rng = np.random.default_rng(seed)
    t0, t1 = rr._ms(pd.Timestamp("2015-07-01")), rr._ms(pd.Timestamp("2015-09-17"))
    rows = []
    for v in range(100, 100 + n_vis):
        for _ in range(rng.integers(1, 4)):
            t = int(rng.integers(t0, t1))
            items = rng.choice(40, rng.integers(2, 7)) + 10
            atc = rng.random() < 0.3
            for j, it in enumerate(items):
                rows.append((t + j * 2 * MIN, v, "view", int(it)))
            if atc:
                rows.append((t + len(items) * 2 * MIN, v, "addtocart", int(items[-1])))
                if rng.random() < 0.6:
                    rows.append((t + (len(items) + 1) * 2 * MIN, v, "transaction", int(items[-1])))
            elif rng.random() < 0.03:
                rows.append((t + len(items) * 2 * MIN, v, "transaction", int(items[0])))
    return pd.DataFrame(rows, columns=["timestamp", "visitorid", "event", "itemid"])


def _write(root, df):
    d = os.path.join(root, "retailrocket")
    os.makedirs(d)
    out = df.sample(frac=1, random_state=1).assign(transactionid="")   # file order must not matter
    out[["timestamp", "visitorid", "event", "itemid", "transactionid"]].to_csv(os.path.join(d, "events.csv"),
                                                                             index=False)


class TestSessions(unittest.TestCase):
    def setUp(self):
        self.ev, self.s = rr.sessionize(_hand())

    def test_sessions_and_prefix(self):
        s = self.s.set_index("session_id")
        self.assertEqual(list(s.loc[["1_0", "1_1", "2_0", "3_0", "4_0", "5_0", "6_0", "6_1"], "n_events"]),
                         [4, 1, 2, 4, 3, 2, 1, 1])
        self.assertEqual(s.loc["1_0", "prefix_len"], 3)
        self.assertEqual(s.loc["1_0", "y"], 1)
        self.assertEqual(s.loc["2_0", "prefix_len"], 1)
        self.assertEqual(s.loc["5_0", "prefix_len"], 0)
        self.assertEqual(s.loc["3_0", "y"], 0)

    def test_event_order_within_same_ms(self):
        e = pd.DataFrame({"timestamp": [5, 5, 5], "visitorid": [1, 1, 1], "event": ["transaction", "view", "addtocart"],
                          "itemid": [9, 9, 9]})
        ev, s = rr.sessionize(e)
        self.assertEqual(list(ev["event"]), ["view", "addtocart", "transaction"])
        self.assertEqual(s["prefix_len"].iloc[0], 2)

    def test_next_item_targets(self):
        sid = dict(zip(self.s["session_id"], self.s.index))
        pl = self.s["prefix_len"].to_numpy()
        ok = pl >= 2
        e = self.ev
        pre = e[ok[e["session"]] & (e["seq"] < pl[e["session"]])]
        t = rr.next_item_targets(pre)
        r3, r4 = t.loc[sid["3_0"]], t.loc[sid["4_0"]]
        self.assertEqual((r3["target"], r3["input_end"], r3["last_item"], r3["repeat"]), (2, 1, 1, 0))
        self.assertEqual((r4["target"], r4["input_end"], r4["last_item"], r4["repeat"]), (1, 1, 2, 1))
        self.assertEqual(t.loc[sid["1_0"], "target"], 2)      # prefix 1,2,atc 2 -> runs 1 | 2 2
        self.assertNotIn(sid["2_0"], t.index)

    def test_visitor_history_uses_earlier_sessions_only(self):
        h = rr.visitor_history(self.s)
        s = self.s.set_index("session_id")
        i0, i1 = s.index.get_loc("1_0"), s.index.get_loc("1_1")
        self.assertEqual((h.iloc[i0]["g_sessions"], h.iloc[i0]["g_tx"]), (0, 0))
        self.assertEqual((h.iloc[i1]["g_sessions"], h.iloc[i1]["g_tx"], h.iloc[i1]["g_events"]), (1, 1, 4))

    def test_unknown_event_raises(self):
        with tempfile.TemporaryDirectory() as root:
            _write(root, _hand().assign(event="click"))
            with self.assertRaises(ValueError):
                RetailRocketPurchase(lgbm=False).load(root)


class TestTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        _write(cls.tmp.name, pd.concat([_hand(), _train_hand(), _filler()], ignore_index=True))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_registered(self):
        self.assertIn("retailrocket_purchase", bench.list_tasks())
        self.assertIsInstance(bench.get("retailrocket_next", split="valid"), RetailRocketNext)
        with self.assertRaises(ValueError):
            RetailRocketPurchase(spilt="test")

    def test_purchase_split_and_no_leak(self):
        task = RetailRocketPurchase(lgbm=False)
        d = task.load(self.tmp.name)
        self.assertTrue((d.train["start"] < rr.VALID_START).all())
        self.assertTrue((d.test["start"] >= rr.TEST_START).all())
        self.assertTrue((d.test["prefix_len"] >= 2).all())
        self.assertNotIn("2_0", set(d.test["session_id"]))
        self.assertEqual((d.meta["prefix"]["event"] == "transaction").sum(), 0)
        self.assertTrue((d.meta["history"]["time"] < rr.VALID_START).all())
        self.assertNotIn("n_tx", d.test.columns)
        dv = RetailRocketPurchase(split="valid", history="all").load(self.tmp.name)
        self.assertTrue(dv.test["start"].between(rr.VALID_START, rr.TEST_START, inclusive="left").all())
        self.assertTrue((dv.meta["history"]["time"] < rr.VALID_START).all())
        dt = RetailRocketPurchase(history="all").load(self.tmp.name)
        self.assertGreater(dt.meta["history"]["time"].max(), rr.VALID_START)
        self.assertTrue((dt.meta["history"]["time"] < rr.TEST_START).all())
        with self.assertRaises(ValueError):
            RetailRocketPurchase(split="future").load(self.tmp.name)

    def test_purchase_baselines_and_score(self):
        task = RetailRocketPurchase()
        d = task.load(self.tmp.name)
        fr = task.run(data=d)
        names = {"constant", "has_addtocart", "last_item_conversion", "counts_logistic"}
        self.assertTrue(names <= set(bench.methods_in(fr)))
        self.assertEqual(len(fr), len(d.test))
        for c in ["truth", "group", "unit", "segment", "split"]:
            self.assertIn(c, fr.columns)
        r = task.score(fr, "has_addtocart", B=50)
        self.assertGreater(r["value"], fr["truth"].mean())          # atc is a strong signal in the fixture
        self.assertTrue(r["lo"] <= r["value"] <= r["hi"])
        self.assertAlmostEqual(task.score(fr, "constant", "auc", B=20)["value"], 0.5)
        tab = task.score_all(fr, ["constant", "counts_logistic"], B=20)
        self.assertEqual(len(tab), 6)

    def test_train_item_features_are_out_of_fold(self):
        task = RetailRocketPurchase(lgbm=False)
        d = task.load(self.tmp.name)
        xtr, _ = task.features(d)
        i = int(np.flatnonzero(d.train["session_id"].to_numpy() == "7_0")[0])
        self.assertEqual(d.train["y"].iloc[i], 1)
        # item 999 only occurs in this visitor's own session -> unseen out of fold, not "sold"
        self.assertEqual((xtr.loc[i, "item_sold_n"], xtr.loc[i, "item_new_n"]), (0.0, 2.0))

    def test_average_precision_matches_sklearn(self):
        from sklearn.metrics import average_precision_score
        rng = np.random.default_rng(3)
        y = (rng.random(300) < 0.2).astype(float)
        p = np.round(rng.random(300), 1)                              # many ties
        self.assertAlmostEqual(rr._ap(y, p), average_precision_score(y, p), places=12)

    def test_next_baselines_and_metric(self):
        task = RetailRocketNext()
        d = task.load(self.tmp.name)
        inp = d.meta["input"].merge(d.test[["session", "input_end"]], on="session")
        self.assertTrue((inp["seq"] <= inp["input_end"]).all())
        self.assertTrue(set(d.test["session"]) <= set(d.meta["input"]["session"]))
        base = task.baselines(d)
        self.assertEqual(set(base), {"popularity", "last_viewed", "cooc", "transition"})
        for m, p in base.items():
            self.assertEqual(p.shape, (len(d.test), rr.K))
            self.assertFalse((p == d.test["last_item"].to_numpy()[:, None]).any(), m)   # last item never proposed
        oracle = d.test["target"].to_numpy()[:, None]
        fr = task.run({"oracle": lambda data: oracle}, data=d)
        self.assertEqual(task.score(fr, "oracle", B=20)["value"], 1.0)
        self.assertEqual(task.score(fr, "oracle", "mrr@20", B=20)["value"], 1.0)
        rep = d.test["repeat"].to_numpy() == 1
        lv = task.score(fr, "last_viewed", B=20)["value"]
        self.assertGreaterEqual(lv, 0.0)
        self.assertLessEqual(task.score(fr, "last_viewed", "recall@20_new", B=20)["value"],
                         task.score(fr, "popularity", "recall@20_new", B=20)["value"])   # new targets: only the (shorter) pop fill can hit
        self.assertTrue(rep.any())
        with self.assertRaises(ValueError):
            task.frame(d, {"short": oracle[:-1]})

    def test_next_metric_hand_ranks(self):
        task = RetailRocketNext(k=3)
        fr = pd.DataFrame({"truth": [5, 5, 5, 5, 5], "group": [1, 2, 3, 4, 5], "segment": "new", "split": "test",
                           "pred_m": pd.Series([np.array([5, 1, 2]), np.array([1, 2, 5]), np.array([1, 2, 3, 5]),
                                                np.array([1, 1, 1, 2, 5]), np.array([-1, 1, -1, 5])], dtype=object)})
        hit, rr_ = rr._hits(fr, "pred_m", 3)
        # duplicates count once and -1 is padding (source metric: unique ids per session, then rank)
        np.testing.assert_allclose(hit, [1, 1, 0, 1, 1])
        np.testing.assert_allclose(rr_, [1, 1 / 3, 0, 1 / 3, 1 / 2])
        r = task.score(fr, "m", "mrr@3", B=20)
        self.assertAlmostEqual(r["value"], (1 + 1 / 3 + 1 / 3 + 1 / 2) / 5)
        with self.assertRaises(ValueError):
            RetailRocketNext(k=0)

    def test_next_hand_session_in_test(self):
        d = RetailRocketNext().load(self.tmp.name)
        t = d.test.set_index("session_id")
        self.assertEqual(int(t.loc["4_0", "target"]), 1)
        self.assertEqual(int(t.loc["4_0", "repeat"]), 1)
        self.assertNotIn("6_0", t.index)

    def test_custom_k(self):
        task = RetailRocketNext(k=5)
        self.assertEqual(task.primary.name, "recall@5")
        d = task.load(self.tmp.name)
        self.assertEqual(task.baselines(d)["cooc"].shape[1], 5)

    def test_missing_data(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(DataNotFound):
                RetailRocketPurchase().load(root)


@unittest.skipUnless(os.environ.get("SIMENGINE_DATA"), "set $SIMENGINE_DATA to run on the real RetailRocket file")
class TestRealData(unittest.TestCase):
    def test_counts(self):
        try:
            d = RetailRocketPurchase(lgbm=False).load()
        except DataNotFound:
            self.skipTest("retailrocket not under $SIMENGINE_DATA")
        self.assertEqual(d.meta["n_events"], 2_756_101)
        self.assertEqual(d.meta["n_sessions"], 1_761_675)
        self.assertEqual((len(d.train), len(d.test), int(d.test["y"].sum())), (312_600, 34_223, 1_013))


if __name__ == "__main__":
    unittest.main()

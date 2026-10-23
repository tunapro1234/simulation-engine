"""instacart bench task: synthetic Kaggle-layout files in a temp dir; real data only when $SIMENGINE_DATA is set."""
import importlib.util
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from simengine.bench.base import ENV_DATA, DataNotFound
from simengine.bench.tasks import instacart as ic
from simengine.bench.tasks.instacart import Instacart

# the loader reads parquet through pyarrow; without it these classes skip (nothing gets installed)
PARQUET = importlib.util.find_spec("pyarrow") is not None
NO_PARQUET = "pyarrow not installed"


def _write(root, csv=False, n_users=60, seed=0):
    """Users with a few 'staple' products they reorder often plus random one-offs; last order = 'train' for most
    users, 'test' (unlabelled) for a few."""
    d = os.path.join(root, "instacart")
    os.makedirs(d)
    rng = np.random.default_rng(seed)
    n_prod = 40
    orders, lines = [], {"prior": [], "train": []}
    oid = 1
    for u in range(1, n_users + 1):
        staples = rng.choice(n_prod, 4, replace=False) + 1
        n_ord = int(rng.integers(4, 9))
        kind = "test" if u % 10 == 0 else "train"
        for k in range(1, n_ord + 1):
            es = "prior" if k < n_ord else kind
            orders.append((oid, u, es, k, int(rng.integers(0, 7)), int(rng.integers(0, 24)),
                           np.nan if k == 1 else float(rng.integers(1, 31))))
            basket = [p for p in staples if rng.random() < 0.7] + list(rng.choice(n_prod, 2) + 1)
            if es in lines:
                for j, p in enumerate(dict.fromkeys(int(x) for x in basket)):
                    lines[es].append([oid, p, j + 1, 0, u])
            oid += 1
    o = pd.DataFrame(orders, columns=ic.FILES["orders"])
    # reordered = product bought by the user in an earlier order
    for es in ("prior", "train"):
        lines[es] = pd.DataFrame(lines[es], columns=["order_id", "product_id", "add_to_cart_order", "reordered", "u"])
    allp = pd.concat([lines["prior"].assign(es=0), lines["train"].assign(es=1)])
    allp = allp.merge(o[["order_id", "order_number"]], on="order_id").sort_values(["u", "order_number"])
    allp["reordered"] = allp.duplicated(["u", "product_id"]).astype(int)
    for i, es in enumerate(("prior", "train")):
        lines[es] = allp[allp["es"] == i][ic.FILES["order_products__prior"]]
    prods = pd.DataFrame({"product_id": np.arange(1, n_prod + 1), "product_name": [f"item {i}" for i in range(n_prod)],
                          "aisle_id": np.arange(n_prod) % 5 + 1, "department_id": np.arange(n_prod) % 3 + 1})
    tabs = {"orders": o, "order_products__prior": lines["prior"], "order_products__train": lines["train"],
            "products": prods, "aisles": pd.DataFrame({"aisle_id": range(1, 6), "aisle": list("abcde")}),
            "departments": pd.DataFrame({"department_id": [1, 2, 3], "department": ["produce", "dairy", "bakery"]})}
    for k, t in tabs.items():
        if csv:
            t.to_csv(os.path.join(d, f"{k}.csv"), index=False)
        else:
            t.to_parquet(os.path.join(d, f"{k}.parquet"), index=False)
    return o, lines


class TestMetrics(unittest.TestCase):
    def test_per_user_f1_none_rule(self):
        inv = np.array([0, 0, 1, 1, 2, 2])
        y = np.array([1, 0, 0, 0, 1, 1], bool)
        d = np.array([1, 1, 0, 0, 0, 0], bool)
        f = ic.per_user_f1(inv, y, d)
        np.testing.assert_allclose(f, [2 / 3, 1.0, 0.0])

    def test_weighted_auc_matches_rank_auc(self):
        from simengine.bench.stats import auc
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 500)
        s = np.round(rng.random(500) + 0.3 * y, 1)          # many ties
        o, gid, ys = ic.weighted_auc_setup(y, s)
        self.assertAlmostEqual(ic.weighted_auc(np.ones(500), gid, ys), auc(y, s), places=12)
        w = rng.integers(0, 3, 500).astype(float)
        rep = np.repeat(np.arange(500), w.astype(int))
        self.assertAlmostEqual(ic.weighted_auc(w[o], gid, ys), auc(y[rep], s[rep]), places=12)


@unittest.skipUnless(PARQUET, NO_PARQUET)
class TestInstacart(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.orders, self.lines = _write(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_split_and_labels(self):
        data = Instacart(lgbm=False).load(self.root)
        te, tr = data.test, data.train
        o = self.orders
        labelled = set(o[o.eval_set == "train"].user_id)
        self.assertEqual(set(te.user_id), labelled)
        self.assertEqual(data.meta["n_users"], len(labelled))
        # candidates = distinct prior products per user; truth = in the last ('train') order
        pr = self.lines["prior"].merge(o, on="order_id")
        pr = pr[pr.user_id.isin(labelled)]
        self.assertEqual(len(te), len(pr.drop_duplicates(["user_id", "product_id"])))
        last = self.lines["train"].merge(o, on="order_id")
        want = set(zip(last.user_id, last.product_id)) & set(zip(pr.user_id, pr.product_id))
        got = set(zip(te.user_id[te.y == 1], te.product_id[te.y == 1]))
        self.assertEqual(got, want)
        # train labels: last prior order; history strictly before it (up_n never counts the label order)
        lp = pr[pr.order_number == pr.groupby("user_id").order_number.transform("max")]
        before = pr[pr.order_number < pr.groupby("user_id").order_number.transform("max")]
        self.assertEqual(len(tr), len(before.drop_duplicates(["user_id", "product_id"])))
        self.assertEqual(set(zip(tr.user_id[tr.y == 1], tr.product_id[tr.y == 1])),
                         set(zip(lp.user_id, lp.product_id)) & set(zip(before.user_id, before.product_id)))
        self.assertTrue((tr.u_orders < te.groupby("user_id").u_orders.max().reindex(tr.user_id).to_numpy()).all())
        self.assertEqual(list(te.columns[:3]), ["user_id", "product_id", "y"])
        self.assertFalse(te[ic.FEATURES].drop(columns=["o_days"]).isna().any().any())

    def test_features_hand_checked(self):
        data = Instacart(lgbm=False).load(self.root)
        te = data.test
        o = self.orders
        pr = self.lines["prior"].merge(o, on="order_id")
        row = te.iloc[0]
        u, p = int(row.user_id), int(row.product_id)
        mine = pr[pr.user_id == u]
        self.assertEqual(row.u_orders, mine.order_number.max())
        self.assertEqual(row.up_n, (mine.product_id == p).sum())
        self.assertAlmostEqual(row.up_rate, (mine.product_id == p).sum() / mine.order_number.max(), places=6)
        self.assertAlmostEqual(row.p_reorder_rate, pr[pr.product_id == p].reordered.mean(), places=6)
        lo = o[(o.user_id == u) & (o.eval_set == "train")].iloc[0]
        self.assertEqual(row.o_days, lo.days_since_prior_order)
        uo = o[o.user_id == u].sort_values("order_number")
        cum = uo.days_since_prior_order.fillna(0).cumsum().to_numpy()
        lastk = mine[mine.product_id == p].order_number.max()
        self.assertAlmostEqual(row.up_days_since_last, cum[lo.order_number - 1] - cum[lastk - 1], places=5)

    def test_train_cut_excludes_label_orders(self):
        # train-cut features count only orders before each user's last prior order, for every user (also the
        # unlabelled 'test' users that feed product statistics); test-cut product statistics use all prior orders
        data = Instacart(lgbm=False).load(self.root)
        tr, te = data.train, data.test
        pr = self.lines["prior"].merge(self.orders, on="order_id")
        before = pr[pr.order_number < pr.groupby("user_id").order_number.transform("max")]
        for p in tr.product_id.unique()[:10]:
            self.assertEqual(tr[tr.product_id == p].p_n.iloc[0], (before.product_id == p).sum())
            self.assertEqual(te[te.product_id == p].p_n.iloc[0], (pr.product_id == p).sum())
        bu = before[before.user_id.isin(tr.user_id)]
        want = bu.groupby(["user_id", "product_id"]).size()
        got = tr.set_index(["user_id", "product_id"]).up_n
        np.testing.assert_array_equal(got.reindex(want.index).to_numpy(), want.to_numpy())

    def test_option_validation(self):
        with self.assertRaises(ValueError):
            Instacart(n_users=0)
        with self.assertRaises(ValueError):
            Instacart(threshold=float("nan"))

    def test_baselines_frame_score(self):
        task = Instacart(lgbm=False)
        data = task.load(self.root)
        frame = task.run(data=data)
        self.assertEqual(len(frame), len(data.test))
        for m in ("user_reorder_rate", "product_reorder_rate", "user_product_rate"):
            self.assertIn("pred_" + m, frame.columns)
            for metric in ("f1", "auc"):
                r = task.score(frame, m, metric, B=50)
                self.assertTrue(0 <= r["value"] <= 1, (m, metric, r))
                self.assertLessEqual(r["lo"], r["hi"])
                self.assertEqual(r["n"], data.meta["n_users"])
        self.assertGreater(task.score(frame, "user_product_rate", "auc", B=20)["value"], 0.6)
        # F1 decision rule follows the threshold option
        ones = task.frame(data, {"all": np.ones(len(data.test)), "none": np.zeros(len(data.test))})
        f_none = task.score(ones, "none", B=20)["value"]
        no_reorder = (data.test.groupby("user_id").y.sum() == 0).mean()
        self.assertAlmostEqual(f_none, no_reorder)
        hi = Instacart(lgbm=False, threshold=1.5)
        self.assertAlmostEqual(hi.score(ones, "all", B=20)["value"], no_reorder)

    def test_lgbm_baseline(self):
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("lightgbm not installed")
        task = Instacart()
        data = task.load(self.root)
        p = task.baselines(data)["lgbm"]
        self.assertEqual(len(p), len(data.test))
        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_options_and_csv_layout(self):
        with self.assertRaises(ValueError):
            Instacart(n_user=5)
        tmp = tempfile.TemporaryDirectory()
        _write(tmp.name, csv=True)
        data = Instacart(n_users=10, seed=1, lgbm=False).load(os.path.join(tmp.name, "instacart"))
        self.assertEqual(data.test.user_id.nunique(), 10)
        self.assertEqual(sorted(data.meta["users"]), sorted(data.test.user_id.unique()))
        with self.assertRaises(ValueError):
            Instacart(n_users=10000, lgbm=False).load(tmp.name)
        tmp.cleanup()
        with self.assertRaises(DataNotFound):
            Instacart().load(os.path.join(self.root, "nope"))
        empty = tempfile.TemporaryDirectory()
        with self.assertRaises(DataNotFound):
            Instacart().load(empty.name)
        empty.cleanup()


@unittest.skipUnless(PARQUET, NO_PARQUET)
@unittest.skipUnless(os.environ.get(ENV_DATA), f"${ENV_DATA} not set")
class TestInstacartReal(unittest.TestCase):
    def test_small_sample(self):
        root = os.environ[ENV_DATA]
        if not os.path.isdir(os.path.join(root, "instacart")):
            self.skipTest("no instacart/ under $SIMENGINE_DATA")
        task = Instacart(n_users=500, lgbm=False)
        data = task.load(root)
        frame = task.run(data=data)
        self.assertEqual(frame.group.nunique(), 500)
        self.assertGreater(task.score(frame, "user_product_rate", "auc", B=20)["value"], 0.6)


if __name__ == "__main__":
    unittest.main()

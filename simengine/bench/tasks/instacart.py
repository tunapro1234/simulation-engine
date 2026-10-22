"""instacart: which previously bought products will a shopper reorder in their next Instacart order (Instacart Online
Grocery Shopping Dataset 2017, the Kaggle "Instacart Market Basket Analysis" setup).

    from simengine import bench
    from simengine.bench.tasks.instacart import Instacart
    task = Instacart(n_users=20000, seed=0)             # default: every user with a labelled last order
    data = task.load("/path/to/catalogue")              # or $SIMENGINE_DATA; <root>/instacart/*.parquet|csv
    frame = task.run(data=data)                         # baselines: user / product / user x product rate, lgbm
    task.score(frame, "lgbm")                           # mean per-user F1 at p >= threshold, user bootstrap
    task.score(frame, "lgbm", "auc")

Users = those whose last order is in `eval_set == "train"` (the Kaggle test orders have no labels). One row per
(user, product the user bought in a prior order); truth = the product is in the user's last order. The last order per
user is the test; the training table uses each user's last *prior* order as label with features from the orders
before it (same users, one order earlier, so no test label is seen). Product statistics are computed from all
users' history before the label order (train) or all prior orders (test).

Metrics: `f1` (primary) = mean per-user F1 of the set {p >= threshold} against the reordered set; an empty truth
set matched by an empty prediction scores 1 (the competition's "None" token), one empty side scores 0. `auc` = ROC
AUC pooled over all candidate rows. Both use a user bootstrap (default_rng(seed), B = 1000) via resampling weights.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, DataNotFound, Metric, register
from ..stats import ci

THRESHOLD = 0.2
FILES = {"orders": ["order_id", "user_id", "eval_set", "order_number", "order_dow", "order_hour_of_day",
                    "days_since_prior_order"],
         "order_products__prior": ["order_id", "product_id", "add_to_cart_order", "reordered"],
         "order_products__train": ["order_id", "product_id", "add_to_cart_order", "reordered"],
         "products": ["product_id", "product_name", "aisle_id", "department_id"],
         "departments": ["department_id", "department"],
         "aisles": ["aisle_id", "aisle"]}
FEATURES = ["u_orders", "u_items", "u_distinct", "u_basket", "u_reorder_rate", "u_days_mean",
            "p_n", "p_users", "p_reorder_rate", "p_repeat_rate", "p_cart_mean",
            "up_n", "up_rate", "up_rate_since_first", "up_orders_since_last", "up_days_since_last", "up_cart_mean",
            "up_in_last", "o_days", "o_dow", "o_hour", "aisle_id", "department_id"]
LGBM_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9,
                   bagging_fraction=0.9, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)


def _slim(d: pd.DataFrame) -> pd.DataFrame:
    for c in d.columns:
        if pd.api.types.is_integer_dtype(d[c]):
            d[c] = pd.to_numeric(d[c], downcast="integer")
    return d


def read_table(root: str, name: str) -> pd.DataFrame:
    """<root>/<name>.parquet, else <root>/<name>.csv (Kaggle original), with the columns the task uses. Read in
    row groups / chunks with integer columns downcast (the prior file has 32M rows; ~0.8 GB instead of ~5 GB)."""
    cols = FILES[name]
    p = os.path.join(root, f"{name}.parquet")
    if os.path.exists(p):
        import pyarrow.parquet as pq
        f = pq.ParquetFile(p)
        parts = [_slim(f.read_row_group(i, columns=cols).to_pandas()) for i in range(f.metadata.num_row_groups)]
        return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)
    p = os.path.join(root, f"{name}.csv")
    if os.path.exists(p):
        return pd.concat([_slim(c) for c in pd.read_csv(p, usecols=cols, chunksize=2_000_000)], ignore_index=True)[cols]
    raise DataNotFound(f"missing file: {os.path.join(root, name)}.parquet|csv")


def history(orders: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """Prior order lines with their order's user_id and order_number: user_id, order_number, product_id,
    add_to_cart_order, reordered (order lookup by array index, no merge copy)."""
    o = orders[orders["eval_set"] == "prior"]
    ids = o["order_id"].to_numpy()
    uid = np.full(int(ids.max()) + 1, -1, np.int32)
    num = np.zeros(int(ids.max()) + 1, np.int16)
    uid[ids], num[ids] = o["user_id"].to_numpy(), o["order_number"].to_numpy()
    oid = prior["order_id"].to_numpy()
    ok = oid <= ids.max()
    ok[ok] = uid[oid[ok]] >= 0
    oid = oid[ok]
    return pd.DataFrame({"user_id": uid[oid], "order_number": num[oid],
                         "product_id": prior["product_id"].to_numpy()[ok].astype(np.int32),
                         "add_to_cart_order": prior["add_to_cart_order"].to_numpy()[ok].astype(np.int16),
                         "reordered": prior["reordered"].to_numpy()[ok].astype(np.int8)})


def order_days(orders: pd.DataFrame) -> pd.DataFrame:
    """user_id, order_number, cum_days (cumulative days_since_prior_order from the user's first order)."""
    o = orders.sort_values(["user_id", "order_number"])
    cum = o["days_since_prior_order"].fillna(0).groupby(o["user_id"]).cumsum()
    return pd.DataFrame({"user_id": o["user_id"].to_numpy("int32"), "order_number": o["order_number"].to_numpy("int16"),
                         "cum_days": cum.to_numpy("float32")})


def build_table(h: pd.DataFrame, label_orders: pd.DataFrame, labels: pd.DataFrame, days: pd.DataFrame,
                products: pd.DataFrame) -> pd.DataFrame:
    """Candidate table for one cut. h = history lines before each user's label order (all users, for product
    statistics); label_orders = user_id, order_number, o_days, o_dow, o_hour of the label order of task users;
    labels = user_id, product_id of the label order lines. Rows sorted by (user_id, product_id)."""
    # product statistics over every user's history
    pg = h.groupby("product_id")
    up_all = h.groupby(["user_id", "product_id"]).size()
    rep = (up_all >= 2).groupby(level="product_id").mean()
    prod = pd.DataFrame({"p_n": pg.size(), "p_users": up_all.groupby(level="product_id").size(),
                         "p_reorder_rate": pg["reordered"].mean(), "p_repeat_rate": rep,
                         "p_cart_mean": pg["add_to_cart_order"].mean()}).astype("float32")
    hu = h[h["user_id"].isin(label_orders["user_id"].to_numpy())]
    ug = hu.groupby("user_id")
    later = hu[hu["order_number"] > 1]
    user = pd.DataFrame({"u_orders": ug["order_number"].max(), "u_items": ug.size(),
                         "u_distinct": ug["product_id"].nunique(),
                         "u_reorder_rate": later.groupby("user_id")["reordered"].mean()})
    user["u_basket"] = user["u_items"] / user["u_orders"]
    lo = label_orders.set_index("user_id")
    # mean gap between the user's history orders (order 1 has no gap)
    hd = days.merge(lo[["order_number"]].rename(columns={"order_number": "cut"}), left_on="user_id", right_index=True)
    hd = hd[hd["order_number"] < hd["cut"]].sort_values(["user_id", "order_number"])
    user["u_days_mean"] = (hd.groupby("user_id")["cum_days"].max() / (user["u_orders"] - 1).clip(lower=1))
    user = user.astype("float32")
    upg = hu.groupby(["user_id", "product_id"])           # keys come out sorted by (user_id, product_id)
    n_up, first = upg.size(), upg["order_number"].min()
    last, cart = upg["order_number"].max(), upg["add_to_cart_order"].mean()
    del upg, hu, later
    u = n_up.index.get_level_values(0).to_numpy()
    p = n_up.index.get_level_values(1).to_numpy()
    f32 = np.float32

    def take(tab, key, col):  # column of a keyed table aligned with rows, float32
        return tab[col].reindex(key).to_numpy(f32)
    out = {"user_id": u, "product_id": p}
    lab = labels["user_id"].to_numpy(np.int64) * 100000 + labels["product_id"].to_numpy(np.int64)
    out["y"] = np.isin(u.astype(np.int64) * 100000 + p, lab).astype(np.int8)
    for c in ("u_orders", "u_items", "u_distinct", "u_basket", "u_reorder_rate", "u_days_mean"):
        out[c] = take(user, u, c)
    for c in ("p_n", "p_users", "p_reorder_rate", "p_repeat_rate", "p_cart_mean"):
        out[c] = take(prod, p, c)
    nu, fi, la = n_up.to_numpy(f32), first.to_numpy(f32), last.to_numpy(f32)
    uo = out["u_orders"]
    # cum_days on a dense (user, order_number) grid
    grid = np.full((int(days["user_id"].max()) + 1, int(days["order_number"].max()) + 1), np.nan, f32)
    grid[days["user_id"].to_numpy(), days["order_number"].to_numpy()] = days["cum_days"].to_numpy()
    lab_no = lo["order_number"].reindex(u).to_numpy()
    out.update(up_n=nu, up_rate=nu / uo, up_rate_since_first=nu / (uo - fi + 1), up_orders_since_last=uo - la,
               up_days_since_last=grid[u, lab_no] - grid[u, la.astype(np.int64)], up_cart_mean=cart.to_numpy(f32),
               up_in_last=(la == uo).astype(f32))
    for c in ("o_days", "o_dow", "o_hour"):
        out[c] = take(lo, u, c)
    pi = products.set_index("product_id")
    out["aisle_id"], out["department_id"] = take(pi, p, "aisle_id"), take(pi, p, "department_id")
    return pd.DataFrame(out)[["user_id", "product_id", "y"] + FEATURES]


# ---------------------------------------------------------------------- metrics

def _user_index(frame):
    g = frame["group"].to_numpy()
    _, inv = np.unique(g, return_inverse=True)
    return inv


def _boot_counts(N, B, seed):
    rng = np.random.default_rng(seed)
    for _ in range(B):
        yield np.bincount(rng.integers(0, N, N), minlength=N).astype(float)


def per_user_f1(inv: np.ndarray, y: np.ndarray, d: np.ndarray) -> np.ndarray:
    """F1 per user of predicted set d against truth y; both empty -> 1 ("None" matched)."""
    N = inv.max() + 1
    tp = np.bincount(inv, weights=(y & d).astype(float), minlength=N)
    s = np.bincount(inv, weights=y.astype(float), minlength=N) + np.bincount(inv, weights=d.astype(float), minlength=N)
    return np.where(s > 0, 2 * tp / np.where(s > 0, s, 1), 1.0)


def _f1(task, frame, col, B, seed):
    inv = _user_index(frame)
    y = frame["truth"].to_numpy(int).astype(bool)
    d = frame[col].to_numpy(float) >= float(task.options.get("threshold", THRESHOLD))
    f = per_user_f1(inv, y, d)
    N = len(f)
    boots = np.array([w @ f / N for w in _boot_counts(N, B, seed)])
    lo, hi = ci(float(f.mean()), boots)
    return {"value": float(f.mean()), "lo": lo, "hi": hi, "n": int(N)}


def weighted_auc_setup(y: np.ndarray, s: np.ndarray):
    """Sort once; returns (row order, tie-group id per sorted row, y sorted) for weighted_auc."""
    o = np.argsort(s, kind="mergesort")
    ss = s[o]
    gid = np.r_[0, np.cumsum(ss[1:] != ss[:-1])]
    return o, gid, y[o].astype(bool)


def weighted_auc(w_sorted: np.ndarray, gid: np.ndarray, ys: np.ndarray) -> float:
    """ROC AUC with row weights (ties count 0.5): sum over tie groups of Wpos * (Wneg below + Wneg in group / 2)."""
    G = gid[-1] + 1
    wp = np.bincount(gid, weights=w_sorted * ys, minlength=G)
    wn = np.bincount(gid, weights=w_sorted * ~ys, minlength=G)
    P, Nn = wp.sum(), wn.sum()
    if P == 0 or Nn == 0:
        return float("nan")
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (P * Nn))


def _auc(task, frame, col, B, seed):
    inv = _user_index(frame)
    y = frame["truth"].to_numpy(int)
    o, gid, ys = weighted_auc_setup(y, frame[col].to_numpy(float))
    inv_s = inv[o]
    point = weighted_auc(np.ones(len(y)), gid, ys)
    N = inv.max() + 1
    boots = np.array([weighted_auc(c[inv_s], gid, ys) for c in _boot_counts(N, B, seed)])
    lo, hi = ci(point, boots)
    return {"value": point, "lo": lo, "hi": hi, "n": int(N)}


@register()
class Instacart(BenchTask):
    """Options: n_users=None (all labelled users; an int samples that many with default_rng(seed)), seed=0,
    threshold=0.2 (F1 decision rule p >= threshold), lgbm=True (LightGBM baseline, skipped if lightgbm is absent)."""

    name = "instacart"
    accepted_options = {
        "n_users": "int | None (default None = all users with a labelled last order)",
        "seed": "int (default 0)",
        "threshold": "float (default 0.2)",
        "lgbm": "true | false (default true)",
    }
    kind = "binary"
    data_subdir = "instacart"
    layout = """
    <root>/instacart/orders.parquet|csv                  order_id, user_id, eval_set, order_number, order_dow,
                                                         order_hour_of_day, days_since_prior_order
    <root>/instacart/order_products__prior.parquet|csv   order_id, product_id, add_to_cart_order, reordered
    <root>/instacart/order_products__train.parquet|csv   same columns; the labelled last orders
    <root>/instacart/products.parquet|csv                product_id, product_name, aisle_id, department_id
    <root>/instacart/aisles.parquet|csv, departments.parquet|csv
    """
    source = "Instacart Online Grocery Shopping Dataset 2017 (Kaggle 'Instacart Market Basket Analysis' file layout)"
    metrics = (Metric("f1", True, _f1, "mean per-user F1 of {p >= threshold} vs reordered set (empty = 'None')"),
               Metric("auc", True, _auc, "ROC AUC pooled over (user, product) candidate rows"))

    def __init__(self, **options):
        super().__init__(**options)
        n, th = options.get("n_users"), options.get("threshold", THRESHOLD)
        if n is not None and int(n) < 1:
            raise ValueError(f"instacart: n_users must be >= 1 or None, got {n}")
        if not np.isfinite(float(th)):  # NaN would silently predict nothing for every user
            raise ValueError(f"instacart: threshold must be a finite number, got {th}")

    def _load(self, root: str) -> TaskData:
        s = int(self.options.get("seed", 0))
        orders = read_table(root, "orders")
        products = read_table(root, "products")
        depts = read_table(root, "departments").set_index("department_id")["department"]
        tr_orders = orders[orders["eval_set"] == "train"]
        users = np.sort(tr_orders["user_id"].unique())
        n = self.options.get("n_users")
        if n is not None:
            if int(n) > len(users):
                raise ValueError(f"only {len(users)} labelled users, need n_users={n}")
            users = np.sort(np.random.default_rng(s).choice(users, int(n), replace=False))
        h = history(orders, read_table(root, "order_products__prior"))
        days = order_days(orders[orders["eval_set"] != "test"])
        lines = read_table(root, "order_products__train").merge(tr_orders[["order_id", "user_id"]], on="order_id")

        def ctx(o):
            return pd.DataFrame({"user_id": o["user_id"].to_numpy("int32"),
                                 "order_number": o["order_number"].to_numpy("int16"),
                                 "o_days": o["days_since_prior_order"].to_numpy("float32"),
                                 "o_dow": o["order_dow"].to_numpy("float32"),
                                 "o_hour": o["order_hour_of_day"].to_numpy("float32")})
        uset = set(users.tolist())
        te_lo = ctx(tr_orders[tr_orders["user_id"].isin(uset)])
        test = build_table(h, te_lo, lines[lines["user_id"].isin(uset)], days, products)
        # training cut: each user's last prior order is the label, history = orders before it
        last = h.groupby("user_id")["order_number"].transform("max")
        h_tr = h[h["order_number"] < last]
        pr = orders[orders["eval_set"] == "prior"]
        pr_last = pr[pr["order_number"] == pr.groupby("user_id")["order_number"].transform("max")]
        tr_lo = ctx(pr_last[pr_last["user_id"].isin(uset)])
        tr_lines = h[(h["order_number"] == last) & h["user_id"].isin(uset)]
        train = build_table(h_tr, tr_lo, tr_lines, days, products)
        del h_tr, last
        prod = products.assign(department=products["department_id"].map(depts))
        meta = {"split": "test: each user's last order (eval_set 'train'), candidates = products from prior orders; "
                         "train: each user's last prior order as label, features from the orders before it",
                "seed": s, "n_users": int(len(users)), "users": users, "threshold": float(self.options.get(
                    "threshold", THRESHOLD)), "features": list(FEATURES), "products": prod, "history": h,
                "test_base_rate": float(test["y"].mean()), "train_base_rate": float(train["y"].mean()),
                "source": "The Instacart Online Grocery Shopping Dataset 2017 (Kaggle competition files); "
                          "Instacart terms: non-commercial use"}
        return TaskData(train=train, test=test, meta=meta)

    # ------------------------------------------------------------------ baselines
    def _lgbm(self, data: TaskData) -> np.ndarray:
        import lightgbm as lgb  # optional dependency
        s = data.meta["seed"]
        tr, te = data.train, data.test
        uu = np.unique(tr["user_id"].to_numpy())
        inner = np.random.default_rng(100 + s).choice(uu, max(1, int(0.1 * len(uu))), replace=False)
        m = np.isin(tr["user_id"].to_numpy(), inner)
        X, y = tr[FEATURES], tr["y"].to_numpy()
        p = dict(LGBM_PARAMS, seed=s)
        cat = ["aisle_id", "department_id"]
        b = lgb.train(p, lgb.Dataset(X[~m], y[~m], categorical_feature=cat), 1000,
                      valid_sets=[lgb.Dataset(X[m], y[m], categorical_feature=cat)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        return np.clip(b.predict(te[FEATURES], num_iteration=b.best_iteration), 0, 1)

    def baselines(self, data: TaskData) -> dict:
        te = data.test
        out = {"user_reorder_rate": te["u_reorder_rate"].fillna(0).to_numpy(float),
               "product_reorder_rate": te["p_reorder_rate"].fillna(0).to_numpy(float),
               "user_product_rate": te["up_rate"].to_numpy(float)}
        if self.options.get("lgbm", True):
            try:
                import lightgbm  # noqa: F401
            except ImportError:
                return out
            out["lgbm"] = self._lgbm(data)
        return out

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        dep = data.meta["products"].set_index("product_id")["department"]
        u, p = t["user_id"].to_numpy("int64"), t["product_id"].to_numpy("int64")
        return pd.DataFrame({"unit": u * 100000 + p, "group": u, "item": p, "truth": t["y"].to_numpy(int),
                             "segment": pd.Categorical(t["product_id"].map(dep).fillna("NA"))})

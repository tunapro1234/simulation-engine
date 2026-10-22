"""retailrocket: RetailRocket e-commerce events (view / addtocart / transaction, 2015-05-03 -> 2015-09-18), two
session tasks on one temporal split: will this session end in a purchase, and which item does the visitor open next.

    from simengine.bench.tasks.retailrocket import RetailRocketPurchase, RetailRocketNext
    task = RetailRocketPurchase()                       # registered as "retailrocket_purchase"
    data = task.load("/path/to/catalogue")              # or $SIMENGINE_DATA; reads <root>/retailrocket/events.csv
    frame = task.run(data=data)                         # constant, has_addtocart, last_item_conversion, counts_logistic, lgbm
    task.score(frame, "counts_logistic")                # PR-AUC with a visitor bootstrap CI
    nxt = RetailRocketNext()                            # "retailrocket_next"; predictions = ranked item-id lists
    nxt.score(nxt.run(data=nxt.load()), "cooc")         # recall@20 (= hit rate, one target per session)

Ported from the intent branch (`kod/eval/hazirla_rr.py`, `kod/eval/baseline_rr.py`, `kod/eval/sonraki_metrik.py`,
PROTOKOL.md "RR-oturum", leak fix in BASELINE_RR.md). Definitions:
  * time = timestamp ms, UTC. Events sorted by visitor, time, event order (view < addtocart < transaction), item,
    file row. A gap > 30 min starts a new session; session_id = "<visitorid>_<n>" (n from 0 per visitor).
  * prefix = the session's events before its first transaction (all events if none). A session is evaluated when its
    prefix has >= 2 events (">= 2 events in the session" would make prefix length 1 imply y = 1: a leak).
  * purchase: y = the session has >= 1 transaction; the model sees the prefix only.
  * split by session start: train < 2015-08-21, valid [2015-08-21, 2015-09-04), test >= 2015-09-04.
  * next item (evaluated sessions with >= 2 runs of distinct consecutive items in the prefix): the last run is held
    out; target = its item, input = prefix events with seq <= input_end (the event just before that run). The target
    is therefore the first item after the input that differs from the input's last item.
Item tables (popularity, conversion, co-occurrence) use events before 2015-08-21 by default (history="train", as the
source); history="all" uses every event before the evaluation window.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, Metric, as_pred_col, need, register
from ..stats import auc, ci, cluster_bootstrap

EVENT_ORDER = {"view": 0, "addtocart": 1, "transaction": 2}
GAP_MS = 30 * 60 * 1000
VALID_START = pd.Timestamp("2015-08-21")     # naive UTC
TEST_START = pd.Timestamp("2015-09-04")
ES_START = pd.Timestamp("2015-08-07")        # lgbm early-stopping slice = last 14 days of train
K = 20
PRIOR = 20.0                                 # pseudo-views for smoothed item rates
NFOLD = 5
COOC_CAP = 50                                # distinct items per history session used for co-occurrence
LR_FEATURES = ["n_events", "n_view", "n_atc", "n_items", "duration_s", "g_sessions", "g_tx"]
END_POSITION = ["last_is_atc", "events_after_atc", "secs_after_atc", "item_conv_last"]
LGBM_FEATURES = LR_FEATURES + ["n_atc_items", "last_is_atc", "events_after_atc", "secs_after_atc", "hour", "weekday",
                               "repeat_consecutive", "events_per_item", "mean_gap_s", "g_events", "g_atc",
                               "g_tx_sessions", "g_days_since_last", "item_conv_max", "item_conv_mean",
                               "item_conv_last", "item_atc_max", "item_atc_mean", "item_logpop_mean",
                               "item_logpop_min", "item_sold_n", "item_new_n"]
LGBM_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1, seed=42)
LAYOUT = """
    <root>/retailrocket/events.csv   timestamp (ms, UTC), visitorid, event (view | addtocart | transaction), itemid,
                                     transactionid (Kaggle "Retailrocket recommender system dataset"; 2,756,101 rows)
    """
SOURCE = ("talp-simulation-hackathon-2026 uretim/intent/kod/eval/hazirla_rr.py, baseline_rr.py, sonraki_metrik.py; "
          "uretim/intent/PROTOKOL.md (RR-oturum), kod/eval/BASELINE_RR.md")
LICENCE = "RetailRocket recommender system dataset (Kaggle, Retail Rocket); CC BY-NC-SA 4.0"
OPTIONS = {"split": '"test" | "valid" (default "test")',
           "history": '"train" (events < 2015-08-21, default) | "all" (events before the evaluation window)'}


def _ms(t: pd.Timestamp) -> int:
    return int(t.value // 1_000_000)


# ---------------------------------------------------------------------- loading
def read_events(root: str) -> pd.DataFrame:
    """timestamp (int ms), visitorid, event, itemid, in file order."""
    e = pd.read_csv(need(os.path.join(root, "events.csv")), usecols=["timestamp", "visitorid", "event", "itemid"],
                    dtype={"timestamp": "int64", "visitorid": "int64", "event": str, "itemid": "int64"})
    bad = ~e["event"].isin(list(EVENT_ORDER))
    if bad.any():
        raise ValueError(f"retailrocket: {int(bad.sum())} rows with an unknown event type")
    return e


def sessionize(e: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(events, sessions). events: session (int), seq, visitorid, ts_ms, time, event, er, itemid, sorted.
    sessions (one row per session, index = session): session_id, visitorid, start, end, n_events, n_view, n_atc, n_tx,
    prefix_len, y."""
    er = e["event"].map(EVENT_ORDER).to_numpy(np.int8)
    v, t, it = (e[c].to_numpy() for c in ("visitorid", "timestamp", "itemid"))
    o = np.lexsort((np.arange(len(e)), it, er, t, v))
    v, t, it, er = v[o], t[o], it[o], er[o]
    new = np.r_[True, (v[1:] != v[:-1]) | (np.diff(t) > GAP_MS)]
    sess = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    seq = np.arange(len(v)) - starts[sess]
    first_of_visitor = np.r_[True, v[1:] != v[:-1]][starts]
    n_in_visitor = np.arange(len(starts)) - np.maximum.accumulate(np.where(first_of_visitor, np.arange(len(starts)), 0))
    ev = pd.DataFrame({"session": sess, "seq": seq, "visitorid": v, "ts_ms": t,
                       "time": pd.to_datetime(t, unit="ms"), "event": np.array(list(EVENT_ORDER))[er], "er": er,
                       "itemid": it})
    S = len(starts)
    n_ev = np.diff(np.r_[starts, len(v)])
    cnt = [np.bincount(sess, weights=(er == k), minlength=S).astype(int) for k in range(3)]
    tx = np.flatnonzero(er == 2)
    first_tx = np.full(S, -1)
    u, i0 = np.unique(sess[tx], return_index=True)
    first_tx[u] = seq[tx[i0]]
    s = pd.DataFrame({"session_id": pd.Series(v[starts]).astype(str).str.cat(pd.Series(n_in_visitor).astype(str), "_"),
                      "visitorid": v[starts], "start": pd.to_datetime(t[starts], unit="ms"),
                      "end": pd.to_datetime(t[starts + n_ev - 1], unit="ms"), "n_events": n_ev, "n_view": cnt[0],
                      "n_atc": cnt[1], "n_tx": cnt[2], "prefix_len": np.where(first_tx < 0, n_ev, first_tx),
                      "y": (cnt[2] > 0).astype(np.int8)})
    return ev, s


def visitor_history(s: pd.DataFrame) -> pd.DataFrame:
    """Per session: totals over the same visitor's EARLIER sessions (sessions never overlap). Index = session."""
    vis = s["visitorid"].to_numpy()
    o = np.lexsort((s["start"].to_numpy(), vis))
    d = s.iloc[o]
    g = d.groupby("visitorid", sort=False)
    prev = lambda c: (g[c].cumsum() - d[c]).to_numpy()
    txs = (d["n_tx"] > 0).astype(int)
    out = pd.DataFrame({"g_sessions": g.cumcount().to_numpy(), "g_events": prev("n_events"), "g_atc": prev("n_atc"),
                        "g_tx": prev("n_tx"), "g_tx_sessions": (txs.groupby(d["visitorid"]).cumsum() - txs).to_numpy(),
                        "g_days_since_last": ((d["start"] - g["end"].shift(1)).dt.total_seconds() / 86400.0).to_numpy()},
                       index=d.index)
    return out.sort_index()


def _period(start: pd.Series) -> np.ndarray:
    return np.where(start >= TEST_START, "test", np.where(start >= VALID_START, "valid", "train"))


class _RetailRocket(BenchTask):
    data_subdir = "retailrocket"
    layout = LAYOUT
    source = SOURCE

    def _split(self) -> tuple[str, pd.Timestamp, pd.Timestamp]:
        split = self.options.get("split", "test")
        if split not in ("test", "valid"):
            raise ValueError("split must be 'test' or 'valid'")
        hist = self.options.get("history", "train")
        if hist not in ("train", "all"):
            raise ValueError("history must be 'train' or 'all'")
        win = TEST_START if split == "test" else VALID_START
        return split, win, (VALID_START if hist == "train" else win)

    def _base(self, root: str):
        split, win, cut = self._split()
        ev, s = sessionize(read_events(root))
        s["period"] = _period(s["start"])
        ok = (s["prefix_len"] >= 2).to_numpy()
        history = ev[ev["ts_ms"] < _ms(cut)].reset_index(drop=True)
        meta = {"split": f"{split}: sessions starting in the {split} window; train sessions start < {VALID_START.date()}"
                         f"; item tables from events < {cut.date()} (history option)", "window_start": str(win.date()),
                "history_end": str(cut.date()), "history": history, "n_events": len(ev), "n_sessions": len(s),
                "n_visitors": int(ev["visitorid"].nunique()), "source": LICENCE}
        return split, ev, s, ok, meta

    @staticmethod
    def _rows(s: pd.DataFrame, mask) -> pd.DataFrame:
        cols = ["session_id", "visitorid", "start", "prefix_len", "y"]
        return s.loc[mask, cols].rename_axis("session").reset_index()


# ---------------------------------------------------------------------- purchase task
def prefix_features(prefix: pd.DataFrame) -> pd.DataFrame:
    """Session features from prefix events only (source `onek_ozellik`). Index = session."""
    p = prefix.sort_values(["session", "seq"])
    g = p.groupby("session", sort=True)
    atc = p[p["er"] == 1]
    ga = atc.groupby("session")
    first, last = g["ts_ms"].min(), g["ts_ms"].max()
    n = g.size()
    f = pd.DataFrame({"n_events": n, "n_view": (p["er"] == 0).groupby(p["session"]).sum(),
                      "n_atc": (p["er"] == 1).groupby(p["session"]).sum(), "n_items": g["itemid"].nunique(),
                      "duration_s": (last - first) / 1000.0})
    f["n_atc_items"] = ga["itemid"].nunique().reindex(f.index).fillna(0)
    f["last_is_atc"] = (g["er"].last() == 1).astype(int)
    f["events_after_atc"] = n - 1 - ga["seq"].max().reindex(f.index)
    f["secs_after_atc"] = (last - ga["ts_ms"].max().reindex(f.index)) / 1000.0
    t0 = pd.to_datetime(first, unit="ms")
    f["hour"], f["weekday"] = t0.dt.hour.to_numpy(), t0.dt.dayofweek.to_numpy()
    same = (p["itemid"].to_numpy()[1:] == p["itemid"].to_numpy()[:-1]) & (p["session"].to_numpy()[1:] ==
                                                                          p["session"].to_numpy()[:-1])
    f["repeat_consecutive"] = pd.Series(np.r_[False, same].astype(int), index=p.index).groupby(p["session"]).sum()
    f["events_per_item"] = f["n_events"] / f["n_items"]
    f["mean_gap_s"] = f["duration_s"] / f["n_events"]
    return f


def item_stats(history: pd.DataFrame) -> pd.DataFrame:
    """itemid -> view / addtocart / transaction counts."""
    c = pd.crosstab(history["itemid"], history["er"])
    return pd.DataFrame({k: c[i] if i in c.columns else 0 for i, k in enumerate(("iv", "ia", "it"))})


def item_features(prefix: pd.DataFrame, st: pd.DataFrame, scale: float, tot: tuple[float, float, float]) -> pd.DataFrame:
    """Session summaries of the prefix items' smoothed rates (source `oge_ozellik`). Index = session."""
    gv, ga, gt = tot
    d = prefix[["session", "seq", "itemid"]].join(st, on="itemid").sort_values(["session", "seq"])
    iv, ia, it = (d[c].fillna(0).to_numpy(float) * scale for c in ("iv", "ia", "it"))
    d = d.assign(conv=(it + PRIOR * gt / gv) / (iv + PRIOR), atc=(ia + PRIOR * ga / gv) / (iv + PRIOR),
                 logpop=np.log(iv + 1), sold=(it > 0).astype(int), new=(iv == 0).astype(int))
    g = d.groupby("session", sort=True)
    return pd.DataFrame({"item_conv_max": g["conv"].max(), "item_conv_mean": g["conv"].mean(),
                         "item_conv_last": g["conv"].last(), "item_atc_max": g["atc"].max(),
                         "item_atc_mean": g["atc"].mean(), "item_logpop_mean": g["logpop"].mean(),
                         "item_logpop_min": g["logpop"].min(), "item_sold_n": g["sold"].sum(),
                         "item_new_n": g["new"].sum()})


def _ap(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision, step-wise with tied scores as one threshold (= sklearn average_precision_score)."""
    o = np.argsort(-p, kind="mergesort")
    y, p = y[o], p[o]
    distinct = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tp = np.cumsum(y)[distinct]
    if len(tp) == 0 or tp[-1] == 0:
        return float("nan")
    prec, rec = tp / (distinct + 1), tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _visitor_boot(fn):
    def metric(task, frame, col, B, seed):
        y, p = frame["truth"].to_numpy(float), frame[col].to_numpy(float)
        r = cluster_bootstrap(lambda i: fn(y[i], p[i]), frame["group"].to_numpy(), B=B, seed=seed)
        return {**r, "n": len(y), "n_visitors": r["n"]}
    return metric


@register()
class RetailRocketPurchase(_RetailRocket):
    """Session -> purchase (binary). Options: split, history (see module docstring), lgbm=True.
    data.train / data.test: session, session_id, visitorid, start, prefix_len, y, plus g_* visitor-history columns
    (earlier sessions only). meta["prefix"]: prefix events of every train / test session (the only per-session input);
    meta["history"]: all events before the history cut (includes transactions; for train rows use out-of-fold stats)."""

    name = "retailrocket_purchase"
    kind = "binary"
    accepted_options = {**OPTIONS, "lgbm": "true | false (default true; skipped if lightgbm is absent)"}
    metrics = (Metric("pr_auc", True, _visitor_boot(_ap), "average precision (positive rate ~0.03)"),
               Metric("auc", True, _visitor_boot(auc), "ROC AUC"),
               Metric("logloss", False, _visitor_boot(_logloss), "log loss (clipped at 1e-15)"))

    def _load(self, root: str) -> TaskData:
        split, ev, s, ok, meta = self._base(root)
        per = s["period"].to_numpy()
        gh = visitor_history(s)
        tr, te = ok & (per == "train"), ok & (per == split)
        rows = tr | te
        pl = s["prefix_len"].to_numpy()
        prefix = ev[rows[ev["session"].to_numpy()] & (ev["seq"].to_numpy() < pl[ev["session"].to_numpy()])]
        meta.update(prefix=prefix.reset_index(drop=True), n_train=int(tr.sum()), n_test=int(te.sum()),
                    positive_rate={"train": float(s.loc[tr, "y"].mean()), "test": float(s.loc[te, "y"].mean())})
        mk = lambda m: self._rows(s, m).join(gh, on="session")
        return TaskData(train=mk(tr), test=mk(te), meta=meta)

    # ------------------------------------------------------------------ baselines
    def features(self, data: TaskData) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(train, test) feature tables (LGBM_FEATURES), rows aligned with data.train / data.test. Item statistics
        for train rows are out-of-fold by visitor (NFOLD folds, visitorid % NFOLD), for test rows from all history."""
        pre, hist = data.meta["prefix"], data.meta["history"]
        pf = prefix_features(pre)
        full = item_stats(hist)
        tot = tuple(float(full[c].sum()) for c in ("iv", "ia", "it"))
        trs = data.train["session"].to_numpy()
        fold_of = pd.Series(data.train["visitorid"].to_numpy() % NFOLD, index=trs)
        hfold = hist["visitorid"].to_numpy() % NFOLD
        parts = []
        for k in range(NFOLD):
            ids = fold_of.index[fold_of.to_numpy() == k]
            parts.append(item_features(pre[pre["session"].isin(ids)], item_stats(hist[hfold != k]),
                                       NFOLD / (NFOLD - 1), tot))
        it_tr = pd.concat(parts)
        it_te = item_features(pre[pre["session"].isin(data.test["session"])], full, 1.0, tot)

        def table(rows, items):
            x = rows[["session"] + [c for c in rows.columns if c.startswith("g_")]].join(pf, on="session")
            return x.join(items, on="session")[LGBM_FEATURES].astype(float).reset_index(drop=True)
        return table(data.train, it_tr), table(data.test, it_te)

    def baselines(self, data: TaskData) -> dict:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        xtr, xte = self.features(data)
        ytr = data.train["y"].to_numpy()
        n = len(data.test)
        rate = pd.Series(ytr).groupby(xtr["n_atc"].to_numpy() > 0).mean()
        out = {"constant": np.full(n, float(ytr.mean())),
               "has_addtocart": (xte["n_atc"] > 0).map(rate).fillna(float(ytr.mean())).to_numpy(float),
               "last_item_conversion": xte["item_conv_last"].to_numpy(float)}
        lx = lambda x: np.log1p(x[LR_FEATURES].fillna(0).clip(lower=0).to_numpy())
        sc = StandardScaler().fit(lx(xtr))
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(lx(xtr)), ytr)
        out["counts_logistic"] = lr.predict_proba(sc.transform(lx(xte)))[:, 1]
        if self.options.get("lgbm", True):
            try:
                import lightgbm  # noqa: F401
            except ImportError:
                return out
            early = (data.train["start"] >= ES_START).to_numpy()
            out["lgbm"], data.meta["lgbm_rounds"] = self._lgbm(xtr, ytr, xte, early, LGBM_FEATURES)
            keep = [c for c in LGBM_FEATURES if c not in END_POSITION]
            out["lgbm_no_end_position"] = self._lgbm(xtr, ytr, xte, early, keep, data.meta["lgbm_rounds"])[0]
        return out

    @staticmethod
    def _lgbm(xtr, ytr, xte, early, cols, rounds=None):
        """Early stopping on the last 14 days of train (valid untouched), refit on all train with 1.1 x best rounds."""
        import lightgbm as lgb
        if rounds is None:
            if early.all() or not early.any() or len(np.unique(ytr[early])) < 2:
                rounds = 200
            else:
                m = lgb.train(LGBM_PARAMS, lgb.Dataset(xtr.loc[~early, cols], ytr[~early]), 3000,
                              valid_sets=[lgb.Dataset(xtr.loc[early, cols], ytr[early])],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
                rounds = max(50, int(m.best_iteration * 1.1))
        m = lgb.train(LGBM_PARAMS, lgb.Dataset(xtr[cols], ytr), rounds)
        return np.clip(m.predict(xte[cols]), 0, 1), rounds

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        pl = t["prefix_len"].to_numpy()
        pre = data.meta["prefix"]
        atc = t["session"].isin(pre.loc[pre["er"] == 1, "session"]).astype(int).to_numpy()
        return pd.DataFrame({"unit": t["session_id"], "group": t["visitorid"].to_numpy(), "truth": t["y"].to_numpy(int),
                             "time": t["start"].to_numpy(),
                             "segment": np.where(pl <= 2, "2", np.where(pl <= 4, "3-4", "5+")),
                             "seg_addtocart": atc})


# ---------------------------------------------------------------------- next-item task
def next_item_targets(prefix: pd.DataFrame) -> pd.DataFrame:
    """Per session with >= 2 runs of consecutive identical items: target (item of the last run), input_end (seq of
    the event just before that run), input_len, last_item (input's last item), repeat (target seen in the input)."""
    p = prefix.sort_values(["session", "seq"])
    s, it, sq = (p[c].to_numpy() for c in ("session", "itemid", "seq"))
    brk = np.r_[True, (s[1:] != s[:-1]) | (it[1:] != it[:-1])]
    run = pd.Series(brk.astype(int)).groupby(s).cumsum().to_numpy()
    n_runs = pd.Series(run).groupby(s).transform("max").to_numpy()
    start_last = brk & (run == n_runs) & (n_runs >= 2)
    tgt = pd.DataFrame({"session": s[start_last], "target": it[start_last], "input_end": sq[start_last] - 1})
    i = np.flatnonzero(start_last) - 1
    tgt["last_item"] = it[i]
    tgt["input_len"] = tgt["input_end"] + 1
    inp = p.merge(tgt[["session", "input_end", "target"]], on="session")
    inp = inp[inp["seq"] <= inp["input_end"]]
    seen = set(inp.loc[inp["itemid"] == inp["target"], "session"])
    tgt["repeat"] = tgt["session"].isin(seen).astype(int)
    return tgt.set_index("session")


def _rank_lists(cands: pd.DataFrame, n: int, k: int) -> np.ndarray:
    """cands: row, itemid, score -> (n, k) int array, best first; ties by smaller item id; padding -1."""
    c = cands.groupby(["row", "itemid"], as_index=False)["score"].max()
    c = c.sort_values(["row", "score", "itemid"], ascending=[True, False, True])
    c = c[c.groupby("row").cumcount() < k]
    out = np.full((n, k), -1, dtype=np.int64)
    out[c["row"].to_numpy(), c.groupby("row").cumcount().to_numpy()] = c["itemid"].to_numpy()
    return out


def _hits(frame: pd.DataFrame, col: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(hit, reciprocal rank) per row, top k of the ranked list."""
    truth = frame["truth"].to_numpy(np.int64)
    rank = np.zeros(len(frame))
    for r, (lst, t) in enumerate(zip(frame[col], truth)):
        a = np.asarray(lst, np.int64)[:k]
        j = np.flatnonzero(a == t)
        rank[r] = j[0] + 1 if len(j) else 0
    return (rank > 0).astype(float), np.where(rank > 0, 1.0 / np.maximum(rank, 1), 0.0)


def _group_mean(v: np.ndarray, groups: np.ndarray, B: int, seed: int) -> dict:
    """Mean of row values with a visitor (cluster) bootstrap; same draws as stats.cluster_bootstrap."""
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    sums, cnt = np.bincount(inv, v, G), np.bincount(inv, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, G, G), minlength=G)
        boots[b] = (w @ sums) / max(w @ cnt, 1)
    point = float(v.mean()) if len(v) else float("nan")
    lo, hi = ci(point, boots)
    return {"value": point, "lo": lo, "hi": hi, "n": int(len(v)), "n_visitors": G}


def _next_metric(which: str, subset: str | None = None):
    def metric(task, frame, col, B, seed):
        f = frame if subset is None else frame[frame["segment"] == subset]
        hit, rr = _hits(f, col, task.k)
        return _group_mean(hit if which == "recall" else rr, f["group"].to_numpy(), B, seed)
    return metric


@register()
class RetailRocketNext(_RetailRocket):
    """Next distinct item in a session. Options: split, history (see module docstring), k=20.
    A prediction is one ranked list of item ids per data.test row (array (n, >= k) or list of sequences; best first);
    pred_<method> frame cells hold int arrays and truth is the target item id. data.test: session, session_id,
    visitorid, start, prefix_len, y, target, input_end, input_len, last_item, repeat. meta["input"]: the input events of
    every train / test row (seq <= input_end; anything later in the session is the answer)."""

    name = "retailrocket_next"
    kind = "multiclass"
    accepted_options = {**OPTIONS, "k": "int (default 20)"}
    metrics = (Metric("recall@20", True, _next_metric("recall"), "target in the top k (= hit rate; one target)"),
               Metric("mrr@20", True, _next_metric("mrr"), "mean reciprocal rank within the top k"),
               Metric("recall@20_new", True, _next_metric("recall", "new"), "recall on rows whose target is not in "
                                                                              "the input"))

    def __init__(self, **options):
        super().__init__(**options)
        self.k = int(self.options.get("k", K))
        if self.k != K:
            self.metrics = tuple(Metric(m.name.replace("@20", f"@{self.k}"), m.higher_is_better, m.fn, m.description)
                                 for m in type(self).metrics)

    def _load(self, root: str) -> TaskData:
        split, ev, s, ok, meta = self._base(root)
        per = s["period"].to_numpy()
        rows = ok & np.isin(per, ["train", split])
        sid = ev["session"].to_numpy()
        pl = s["prefix_len"].to_numpy()
        prefix = ev[rows[sid] & (ev["seq"].to_numpy() < pl[sid])]
        tg = next_item_targets(prefix)
        has = np.zeros(len(s), bool)
        has[tg.index.to_numpy()] = True
        tr, te = has & ok & (per == "train"), has & ok & (per == split)
        inp = prefix.join(tg["input_end"], on="session", how="inner")
        inp = inp[inp["seq"] <= inp["input_end"]].drop(columns="input_end").reset_index(drop=True)
        meta.update(input=inp, n_train=int(tr.sum()), n_test=int(te.sum()),
                    n_test_sessions_evaluated=int((ok & (per == split)).sum()),
                    repeat_rate=float(tg.loc[np.flatnonzero(te), "repeat"].mean()) if te.any() else float("nan"))
        mk = lambda m: self._rows(s, m).join(tg, on="session")
        return TaskData(train=mk(tr), test=mk(te), meta=meta)

    # ------------------------------------------------------------------ baselines
    def baselines(self, data: TaskData) -> dict:
        k, t = self.k, data.test
        n = len(t)
        hist = data.meta["history"]
        row_of = pd.Series(np.arange(n), index=t["session"].to_numpy())
        last = pd.DataFrame({"row": np.arange(n), "a": t["last_item"].to_numpy()})
        pc = hist.groupby("itemid").size()
        pop = pc.sort_values(ascending=False, kind="mergesort")  # ties: smaller item id first (groupby index sorted)
        top = pop.index.to_numpy()[:k + 1]
        popc = last.merge(pd.DataFrame({"itemid": top, "score": (k + 1 - np.arange(len(top))) * 1e-6}), how="cross")
        popc = popc[popc["itemid"] != popc["a"]][["row", "itemid", "score"]]
        # last_viewed: earlier input items, most recent first
        inp = data.meta["input"]
        inp = inp[inp["session"].isin(row_of.index)]
        lv = inp.groupby(["session", "itemid"], as_index=False)["seq"].max()
        lv["row"] = lv["session"].map(row_of).to_numpy()
        lv = lv.merge(last, on="row")
        lv = lv[lv["itemid"] != lv["a"]].assign(score=lambda d: 1.0 + d["seq"])[["row", "itemid", "score"]]
        # co-occurrence with the input's last item in history sessions (distinct items, first COOC_CAP per session)
        need_a = np.unique(last["a"])
        h = hist.sort_values(["session", "seq"]).drop_duplicates(["session", "itemid"])
        h = h[h.groupby("session").cumcount() < COOC_CAP][["session", "itemid"]]
        h = h[h.groupby("session")["itemid"].transform("size") >= 2]
        ha = h[h["itemid"].isin(need_a)].rename(columns={"itemid": "a"})
        pairs = ha.merge(h, on="session")
        pairs = pairs[pairs["itemid"] != pairs["a"]]
        cooc = self._top(pairs.groupby(["a", "itemid"]).size(), k)
        # transitions a -> next different item in history sessions
        hs = hist.sort_values(["session", "seq"])
        nx = hs["itemid"].shift(-1).to_numpy()
        same_s = hs["session"].to_numpy() == hs["session"].shift(-1).to_numpy()
        tr = pd.DataFrame({"a": hs["itemid"].to_numpy(), "itemid": nx})[same_s & (hs["itemid"].to_numpy() != nx)]
        tr = tr[tr["a"].isin(need_a)].astype(np.int64)
        trans = self._top(tr.groupby(["a", "itemid"]).size(), k)
        cc = last.merge(cooc, on="a")[["row", "itemid", "score"]]
        tc = last.merge(trans, on="a").assign(score=lambda d: d["score"] * 1e6)[["row", "itemid", "score"]]
        return {"popularity": _rank_lists(popc, n, k),
                "last_viewed": _rank_lists(pd.concat([lv, popc]), n, k),
                "cooc": _rank_lists(pd.concat([cc, popc]), n, k),
                "transition": _rank_lists(pd.concat([tc, cc, popc]), n, k)}

    @staticmethod
    def _top(counts: pd.Series, k: int) -> pd.DataFrame:
        d = counts.rename("score").reset_index().sort_values(["a", "score", "itemid"], ascending=[True, False, True])
        return d[d.groupby("a").cumcount() < k].astype({"score": float})

    def frame(self, data: TaskData, preds: dict) -> pd.DataFrame:
        base = self._frame(data).reset_index(drop=True)
        n = len(base)
        for m, p in preds.items():
            vals = [np.asarray(x, np.int64).ravel() for x in p]
            if len(vals) != n:
                raise ValueError(f"{m}: {len(vals)} ranked lists for {n} test rows")
            base[as_pred_col(m)] = pd.Series(vals, dtype=object)
        base["split"] = "test"
        return base

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        il = t["input_len"].to_numpy()
        return pd.DataFrame({"unit": t["session_id"], "group": t["visitorid"].to_numpy(),
                             "truth": t["target"].to_numpy(np.int64), "item": t["last_item"].to_numpy(np.int64),
                             "time": t["start"].to_numpy(), "segment": np.where(t["repeat"] == 1, "repeat", "new"),
                             "seg_input_len": np.where(il <= 1, "1", np.where(il <= 3, "2-3", "4+"))})

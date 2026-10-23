"""m5: Walmart daily unit sales (M5 Accuracy, Kaggle 2020) as a bench task: 28-day forecasts for item x store series,
scored by WRMSSE over the competition's 12 aggregation levels and by bottom-level WMAPE.

    from simengine import bench
    from simengine.bench.tasks import m5
    from simengine.methods import forecast as fc
    task = m5.M5(n_series=3000, baselines="snaive")   # or bench.get("m5", stores="CA_1"); default: all 30,490 series
    data = task.load("/path/to/catalogue")            # or $SIMENGINE_DATA
    mine = lambda d: m5.align(d, fc.moving_mean(m5.future_panel(d), d.meta["origin"], 28, window=28, **m5.COLS))
    frame = task.run(methods={"ma28": mine}, data=data)  # fits only the `baselines` option (unset: all five)
    task.score(frame, "ma28")                         # {'metric': 'wrmsse', 'value', 'lo', 'hi', 'n'}
    m5.wrmsse_by_level(frame, "ma28")                 # one row per aggregation level (compare M5 paper Table B)

Data: one row per (series, day) in long form, the panel layout of `simengine.methods.forecast` (`COLS`: id_col="id",
time_col="d", y_col="sales"; d is the integer M5 day index, d_1 = 2011-01-29). `data.train` holds the last `history`
days up to the forecast origin, `data.test` the 28 days after it (`h` = 1..28, `sales` = truth). Both carry the series
keys (item_id, dept_id, cat_id, store_id, state_id; categorical) and the known-in-advance covariates of the target day:
`sell_price` (NaN before an item's first price), `wday` (1 = Saturday, M5 calendar), `snap` (SNAP day in the series'
state) and `event` (any named event). `future_panel(data)` = train + test rows with sales NaN, the input a forecaster
in `methods.forecast` expects; `align(data, fr)` maps its long output back onto data.test rows.

Splits (the last training day is the origin):

    evaluation  origin d_1941, test d_1942-1969 (the competition's final, private-leaderboard window; default)
    validation  origin d_1913, test d_1914-1941 (public-leaderboard window)
    dev         origin d_1885, test d_1886-1913 (never touches either competition window)

Metrics (Makridakis, Spiliotis & Assimakopoulos 2022, IJF 38(4), section 3.3, checked against the paper):

    wrmsse  (primary) RMSSE_s = sqrt(mean_h (y - y_hat)^2 / scale_s), scale_s = mean squared one-step difference of
            series s over its training history from its first non-zero sale to the origin (all days of the file, not
            only `history`); bottom forecasts are summed to the 42,840 series of the 12 levels (bottom-up);
            w_s = the series' dollar sales (units x sell_price) over the last 28 training days, normalised within each
            level; WRMSSE = mean over levels of sum_s w_s RMSSE_s. On a subsample (n_series, stores) the hierarchy,
            weights and scales are those of the subsample, so values are not comparable with full-M5 numbers.
    wmape   sum |y - y_hat| / sum y over the bottom-level test rows (the D4 rehearsal metric, `forecast.wmape`)

CI: 95% percentile bootstrap, default_rng(42), B = 1000. WRMSSE resamples the aggregate series within each level
(w and RMSSE of a resampled series move together, level value = sum c w RMSSE / sum c w); the levels are resampled
independently and the hierarchy's scales are held fixed, so levels 1-5 (1-10 series) add almost no width. WMAPE
resamples bottom series.
A method has to forecast every test row: one NaN or inf prediction makes both metrics (and the CI) NaN.

Baselines (all bottom level, aggregated bottom-up by the metric, as the M5 benchmarks):

    naive      last value (`forecast.persistence_forecast`)
    snaive     same weekday last week (`forecast.seasonal_naive`, season = 7; M5 benchmark sNaive)
    ses        simple exponential smoothing, one global alpha tuned by `forecast.ses` (WMAPE on earlier origins); the
               M5 benchmark SES picks alpha in [0.1, 0.3] per series by in-sample MSE instead
    ets_lite   ses level x a day-of-week index pooled over store x dept from the last 8 weeks (`ets_lite`); a cheap
               stand-in for the M5 ES_bu benchmark (per-series ETS), not an ETS model
    lightgbm   `forecast.DirectForecaster`: one Tweedie LightGBM per horizon (28 models), lags 1/2/3/7/14/28, rolling
               mean / non-zero count over 7/28/56/112 days, same-weekday-last-week lag, target-day price, wday, SNAP,
               event, store/dept/cat/state codes; trained on the last `lgb_origins` daily origins (origin_step = 1:
               with step 7 `DirectForecaster` takes origins p - h - 7k, whose weekday differs from the forecast
               origin's for h not a multiple of 7, and the fixture's WRMSSE rose from 0.49 to 1.50)
"""
from __future__ import annotations

import io
import os
import warnings
import zipfile
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ...methods import forecast as fc
from ...types import TaskData
from ..base import BenchTask, DataNotFound, Metric, register
from ..stats import ci

HORIZON = 28
COLS = {"id_col": "id", "time_col": "d", "y_col": "sales"}
KEYS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
SPLITS = {"evaluation": 1941, "validation": 1913, "dev": 1885}
LAST_TRAIN_FILE_DAY = 1941             # sales_train_evaluation.csv ends here; sales_test_evaluation.csv holds 1942-1969
# M5 aggregation levels in the competition's order (Competitors' Guide; weights_*.csv Level1..Level12)
LEVELS = ((), ("state_id",), ("store_id",), ("cat_id",), ("dept_id",), ("state_id", "cat_id"),
          ("state_id", "dept_id"), ("store_id", "cat_id"), ("store_id", "dept_id"), ("item_id",),
          ("state_id", "item_id"), ("item_id", "store_id"))
BASELINES = ("naive", "snaive", "ses", "ets_lite", "lightgbm")
COVARIATES = ("sell_price", "wday", "snap", "event")
FILES = ("sales_train_evaluation.csv", "sales_test_evaluation.csv", "calendar.csv", "sell_prices.csv")


# ---------------------------------------------------------------------- files
class _Files:
    """CSV files of the M5 release, either extracted in `root` (or root/m5) or inside root/m5.zip."""

    def __init__(self, root: str):
        self.dir, self.zip = None, None
        for d in (root, os.path.join(root, "m5")):
            if os.path.exists(os.path.join(d, "sales_train_evaluation.csv")):
                self.dir = d
                return
        for z in (os.path.join(root, "m5.zip"), os.path.join(root, "m5", "m5.zip")):
            if os.path.exists(z):
                self.zip = z
                return
        raise DataNotFound(f"M5 files not found under {root} (sales_train_evaluation.csv or m5.zip)")

    def has(self, name: str) -> bool:
        if self.dir:
            return os.path.exists(os.path.join(self.dir, name))
        with zipfile.ZipFile(self.zip) as z:
            return name in z.namelist()

    def read(self, name: str, **kw) -> pd.DataFrame:
        if not self.has(name):
            raise DataNotFound(f"missing file: {name} in {self.dir or self.zip}")
        if self.dir:
            return pd.read_csv(os.path.join(self.dir, name), **kw)
        with zipfile.ZipFile(self.zip) as z, z.open(name) as f:
            return pd.read_csv(io.BufferedReader(f), **kw)


def _day_cols(cols) -> list[str]:
    return [c for c in cols if isinstance(c, str) and c.startswith("d_")]


def read_sales(files: _Files, last_day: int) -> tuple[pd.DataFrame, np.ndarray]:
    """(keys[id, item_id, dept_id, cat_id, store_id, state_id], Y[n_series, last_day]) for days d_1..d_last_day;
    id = item_id + "_" + store_id. Days after d_1941 come from sales_test_evaluation.csv."""
    hdr = files.read("sales_train_evaluation.csv", nrows=0).columns
    days = _day_cols(hdr)
    tr = files.read("sales_train_evaluation.csv", usecols=list(KEYS) + days[:last_day],
                    dtype={**{c: "int16" for c in days}, **{k: str for k in KEYS}})
    Y = tr[days[:last_day]].to_numpy(np.float32)
    keys = tr[list(KEYS)].reset_index(drop=True)
    if last_day > len(days):
        te = files.read("sales_test_evaluation.csv", dtype={k: str for k in KEYS})
        te = keys.merge(te, on=list(KEYS), how="left", validate="1:1")
        extra = [f"d_{d}" for d in range(len(days) + 1, last_day + 1)]
        miss = [c for c in extra if c not in te.columns]
        if miss:
            raise DataNotFound(f"sales_test_evaluation.csv lacks {miss[:3]}...")
        Y = np.hstack([Y, te[extra].to_numpy(np.float32)])
    keys.insert(0, "id", keys["item_id"] + "_" + keys["store_id"])
    return keys, Y


def read_calendar(files: _Files) -> pd.DataFrame:
    """calendar.csv indexed by the integer day d (row k -> d = k + 1, or the file's own d_ column), plus `event`
    (1 if any named event) ."""
    c = files.read("calendar.csv")
    c["d"] = c["d"].str[2:].astype(int) if "d" in c.columns else np.arange(1, len(c) + 1)
    ev = [x for x in ("event_name_1", "event_name_2") if x in c.columns]
    c["event"] = c[ev].notna().any(axis=1).astype(np.int8) if ev else np.int8(0)
    c["date"] = pd.to_datetime(c["date"])
    return c.set_index("d")


def price_matrix(files: _Files, keys: pd.DataFrame, cal: pd.DataFrame, days: np.ndarray) -> np.ndarray:
    """sell_price of every series on every day in `days` (n_series x len(days), float32, NaN where not on sale)."""
    p = files.read("sell_prices.csv", dtype={"store_id": str, "item_id": str, "wm_yr_wk": np.int32,
                                             "sell_price": np.float32})
    weeks = np.unique(cal.loc[days, "wm_yr_wk"].to_numpy())
    p = p[p["wm_yr_wk"].isin(weeks)]
    si = pd.MultiIndex.from_frame(keys[["store_id", "item_id"]]).get_indexer(
        pd.MultiIndex.from_frame(p[["store_id", "item_id"]]))
    ok = si >= 0
    W = np.full((len(keys), len(weeks)), np.nan, np.float32)
    W[si[ok], np.searchsorted(weeks, p["wm_yr_wk"].to_numpy()[ok])] = p["sell_price"].to_numpy()[ok]
    return W[:, np.searchsorted(weeks, cal.loc[days, "wm_yr_wk"].to_numpy())]


# ---------------------------------------------------------------------- hierarchy, scales, weights
def level_codes(keys: pd.DataFrame, level: Sequence[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """(code per series, aggregate labels) for one aggregation level; () = the total."""
    if not level:
        return np.zeros(len(keys), np.int64), pd.DataFrame({"key": ["Total"]})
    g = keys.groupby(list(level), sort=True, observed=True)
    return g.ngroup().to_numpy(np.int64), g.size().reset_index()[list(level)]


def rmsse_scale(A: np.ndarray) -> np.ndarray:
    """Per row: mean of squared one-step differences from the first non-zero value on (M5 RMSSE denominator).
    NaN for rows that are never non-zero or have a single active day."""
    A = np.asarray(A, float)
    nz = A > 0
    first = np.where(nz.any(1), nz.argmax(1), A.shape[1])
    D2 = np.diff(A, axis=1) ** 2
    act = np.arange(1, A.shape[1])[None, :] > first[:, None]     # difference t-1 -> t counts when t-1 >= first
    n = act.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (D2 * act).sum(1) / n, np.nan)


def hierarchy(keys: pd.DataFrame, Y_hist: np.ndarray, dollars: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(levels, per_series). levels: one row per aggregate series (level 1-12, key labels, dollar_sales, weight,
    scale); per_series: by bottom id, w_L<l> and sc_L<l> = weight and scale of the series' level-l aggregate."""
    from scipy.sparse import csr_matrix
    rows, per = [], {}
    for lv, level in enumerate(LEVELS, 1):
        code, lab = level_codes(keys, level)
        G = int(code.max()) + 1
        if lv == len(LEVELS) and G == len(keys):
            A = Y_hist[np.argsort(code)]
        else:
            S = csr_matrix((np.ones(len(code), np.float32), (code, np.arange(len(code)))), shape=(G, len(code)))
            A = np.asarray(S @ Y_hist)
        sc = rmsse_scale(A)
        dol = np.bincount(code, dollars, minlength=G)
        w = dol / dol.sum() if dol.sum() > 0 else np.full(G, 1.0 / G)
        lab = lab.assign(level=lv, n_bottom=np.bincount(code, minlength=G), dollar_sales=dol, weight=w, scale=sc)
        rows.append(lab)
        per[f"w_L{lv}"], per[f"sc_L{lv}"] = w[code], sc[code]
    levels = pd.concat(rows, ignore_index=True)
    levels = levels[["level", *[c for c in ("key", *KEYS) if c in levels.columns], "n_bottom", "dollar_sales",
                     "weight", "scale"]]
    return levels, pd.DataFrame(per, index=keys["id"].to_numpy())


# ---------------------------------------------------------------------- metrics
def _level_codes_frame(frame: pd.DataFrame, level: Sequence[str]) -> np.ndarray:
    if not level:
        return np.zeros(len(frame), np.int64)
    return frame.groupby(list(level), sort=False, observed=True).ngroup().to_numpy(np.int64)


def level_scores(frame: pd.DataFrame, col: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per level: (w, RMSSE) arrays over the aggregate series present in the frame (bottom-up sums of truth and of
    `col`). A frame row is one (bottom series, h) with w_L<l> / sc_L<l> columns."""
    e = frame["truth"].to_numpy(float) - frame[col].to_numpy(float)
    h = frame["h"].to_numpy(np.int64)
    hs, hpos = np.unique(h, return_inverse=True)
    out = []
    for lv, level in enumerate(LEVELS, 1):
        g = _level_codes_frame(frame, level)
        G, H = int(g.max()) + 1 if len(g) else 0, len(hs)
        E = np.bincount(g * H + hpos, e, minlength=G * H).reshape(G, H)
        seen = np.bincount(g * H + hpos, minlength=G * H).reshape(G, H) > 0
        cnt = np.bincount(g, minlength=G)
        w = np.bincount(g, frame[f"w_L{lv}"].to_numpy(float), minlength=G) / cnt
        sc = np.bincount(g, frame[f"sc_L{lv}"].to_numpy(float), minlength=G) / cnt
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.sqrt((E ** 2).sum(1) / seen.sum(1) / sc)
        out.append((w, r))
    return out


def _complete(frame: pd.DataFrame, col: str) -> bool:
    """Every test row has a finite prediction. Both metrics are NaN otherwise: a method is not scored on the subset
    of series it chose to forecast (`level_scores` drops every aggregate holding a gap, `fc.wmape` the rows)."""
    return bool(np.isfinite(frame[col].to_numpy(float)).all())


def _weighted(w: np.ndarray, r: np.ndarray) -> float:
    ok = np.isfinite(r) & np.isfinite(w)
    return float((w[ok] * r[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else float("nan")


def wrmsse(frame: pd.DataFrame, method: str) -> float:
    """WRMSSE of one pred_<method> column (point value, no CI)."""
    col = method if method in frame.columns else f"pred_{method}"
    if not _complete(frame, col):
        return float("nan")
    return float(np.mean([_weighted(w, r) for w, r in level_scores(frame, col)]))


def wrmsse_by_level(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    """level, keys, n_series, wrmsse per aggregation level, and an 'average' row (= WRMSSE)."""
    col = method if method in frame.columns else f"pred_{method}"
    sc, ok = level_scores(frame, col), _complete(frame, col)
    rows = [{"level": lv, "keys": "+".join(LEVELS[lv - 1]) or "total", "n_series": len(w),
             "wrmsse": _weighted(w, r) if ok else float("nan")}
            for lv, (w, r) in enumerate(sc, 1)]
    rows.append({"level": "average", "keys": "", "n_series": sum(len(w) for w, _ in sc),
                 "wrmsse": float(np.mean([x["wrmsse"] for x in rows]))})
    return pd.DataFrame(rows)


def _ratio_boot(num: np.ndarray, den: np.ndarray, rng, B: int, block: int = 100) -> np.ndarray:
    """B replicates of sum(num[idx]) / sum(den[idx]), idx a with-replacement resample of the units."""
    n = len(num)
    out = np.empty(B)
    for lo in range(0, B, block):
        idx = rng.integers(0, n, (min(block, B - lo), n))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[lo:lo + len(idx)] = num[idx].sum(1) / den[idx].sum(1)
    return out


def _wrmsse_metric(task, frame, col, B, seed):
    if not _complete(frame, col):
        return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(frame["group"].nunique())}
    sc = level_scores(frame, col)
    point = float(np.mean([_weighted(w, r) for w, r in sc]))
    rng = np.random.default_rng(seed)
    reps = np.zeros(B)
    for w, r in sc:
        ok = np.isfinite(r) & np.isfinite(w)
        reps += _ratio_boot(w[ok] * r[ok], w[ok], rng, B) if ok.any() else np.nan
    lo, hi = ci(point, reps / len(sc))
    return {"value": point, "lo": lo, "hi": hi, "n": int(frame["group"].nunique())}


def _wmape_metric(task, frame, col, B, seed):
    y, p = frame["truth"].to_numpy(float), frame[col].to_numpy(float)
    g = pd.factorize(frame["group"])[0]
    if not _complete(frame, col):
        return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(g.max() + 1)}
    ok = np.isfinite(y)
    a = np.bincount(g[ok], np.abs(y - p)[ok], minlength=g.max() + 1)
    t = np.bincount(g[ok], y[ok], minlength=g.max() + 1)
    point = fc.wmape(y, p)
    lo, hi = ci(point, _ratio_boot(a, t, np.random.default_rng(seed), B))
    return {"value": point, "lo": lo, "hi": hi, "n": int(g.max() + 1)}


METRICS = (Metric("wrmsse", False, _wrmsse_metric,
                  "weighted RMSSE over the 12 M5 aggregation levels (bottom-up); bootstrap over aggregate series"),
           Metric("wmape", False, _wmape_metric, "sum|y - y_hat| / sum y over bottom rows; series bootstrap"))


# ---------------------------------------------------------------------- forecaster glue
def future_panel(data: TaskData) -> pd.DataFrame:
    """data.train + data.test rows with sales = NaN (covariates kept): the panel a `methods.forecast` forecaster
    takes at origin data.meta['origin']."""
    out = pd.concat([data.train, data.test[data.train.columns]], ignore_index=True)
    out.loc[len(data.train):, "sales"] = np.nan
    return out


def align(data: TaskData, fr: pd.DataFrame, *, pred_col: str = "pred") -> np.ndarray:
    """Map a long forecast frame [id, d (or origin + h), pred] onto data.test rows (NaN where missing)."""
    f = fr if "d" in fr.columns else fr.assign(d=fr["origin"] + fr["h"])
    key = pd.MultiIndex.from_arrays([f["id"].astype(str).to_numpy(), f["d"].to_numpy(np.int64)])
    if key.has_duplicates:
        raise ValueError("align: forecast frame has duplicate (id, d) rows")
    at = key.get_indexer(pd.MultiIndex.from_arrays([data.test["id"].astype(str).to_numpy(),
                                                    data.test["d"].to_numpy(np.int64)]))
    v = f[pred_col].to_numpy(float)
    return np.where(at >= 0, v[np.maximum(at, 0)], np.nan)


def ets_lite(panel: pd.DataFrame, origin: int, horizon: int = HORIZON, *, weeks: int = 8,
             group_cols: Sequence[str] = ("store_id", "dept_id"), dow_col: str = "wday", alpha: float | None = None,
             id_col: str = "id", time_col: str = "d", y_col: str = "sales") -> pd.DataFrame:
    """SES level (`forecast.ses`, alpha tuned when None) x day-of-week index of the target day. The index is
    7 * sum(y on that weekday) / sum(y) over the last `weeks` weeks up to the origin, pooled over the series of each
    `group_cols` group (1 when the group sold nothing). The target's weekday is the one 7 * ceil(h / 7) days earlier."""
    fr = fc.ses(panel, origin, horizon, alpha=alpha, id_col=id_col, time_col=time_col, y_col=y_col)
    gc = list(group_cols)
    hist = panel[(panel[time_col] > origin - 7 * weeks) & (panel[time_col] <= origin)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        by = hist.groupby(gc + [dow_col], observed=True)[y_col].sum()
        tot = hist.groupby(gc, observed=True)[y_col].sum()
        grp = panel.groupby(id_col, observed=True)[gc].first()
    idx = (7 * by / tot.reindex(by.index.droplevel(-1)).to_numpy()).replace([np.inf, -np.inf], np.nan)
    dow = hist.drop_duplicates(time_col).set_index(time_col)[dow_col]
    t = fr[time_col].to_numpy(np.int64)
    src = t - 7 * np.ceil((t - origin) / 7).astype(np.int64)
    grp.index = grp.index.astype(str)
    g = grp.reindex(fr[id_col].astype(str).to_numpy())
    look = pd.MultiIndex.from_arrays([*(g[c].to_numpy() for c in gc), dow.reindex(src).to_numpy()])
    f = idx.reindex(look).to_numpy(float)
    fr["pred"] = fr["pred"].to_numpy(float) * np.where(np.isfinite(f), f, 1.0)
    return fr


def _lightgbm(data: TaskData, *, origins: int = 28, jobs: int | None = None, seed: int = 0) -> np.ndarray:
    model: Any = "lightgbm"
    if jobs:
        model = fc._default_model("lightgbm", None, True, seed)
        model.set_params(**({"n_jobs": int(jobs)} if "n_jobs" in model.get_params() else {}))
    f = fc.DirectForecaster(model, HORIZON, lags=(1, 2, 3, 7, 14, 28), windows=(7, 28, 56, 112), season=7,
                            exog=list(COVARIATES), static=["store_id", "dept_id", "cat_id", "state_id"],
                            n_train_origins=origins, origin_step=1, clip_min=0.0, seed=seed, **COLS)
    return align(data, f(future_panel(data), data.meta["origin"]))


# ---------------------------------------------------------------------- task
@register()
class M5(BenchTask):
    """Options: split, origin, history, n_series, stores, seed, baselines, lgb_origins, lgb_jobs (see
    accepted_options)."""

    name = "m5"
    kind = "continuous"
    data_subdir = "m5-walmart"
    metrics = METRICS
    accepted_options = {
        "split": '"evaluation" | "validation" | "dev" (default "evaluation": origin d_1941, test d_1942-1969)',
        "origin": "int last training day (overrides split; needs origin + 28 <= last day in the files)",
        "history": "int days of training rows kept in data.train (default 730; scales and weights use all days)",
        "n_series": "int: random subsample of item x store series, default_rng(seed) (default all 30,490)",
        "stores": 'store ids, "+"-joined str or list, e.g. "CA_1+TX_2" (applied before n_series; default all)',
        "seed": "int (default 0; subsample and LightGBM seed)",
        "baselines": '"+"-joined subset of naive, snaive, ses, ets_lite, lightgbm (default all)',
        "lgb_origins": "int consecutive daily training origins for the LightGBM baseline (default 28)",
        "lgb_jobs": "int LightGBM threads (default 1, the forecast module's setting)",
    }
    layout = """
    <root>/m5-walmart/m5.zip  (Kaggle M5 release as zipped by the Nixtla m5-forecasts mirror) or the extracted
        sales_train_evaluation.csv, sales_test_evaluation.csv (only for split="evaluation"), calendar.csv,
        sell_prices.csv in <root>/m5-walmart/ (or its m5/ subdirectory)
    """
    source = ("M5 Forecasting - Accuracy (Kaggle / University of Nicosia 2020; Walmart unit sales); competition "
              "data terms apply, no licence file in the mirror: internal research use")

    def __init__(self, **options):
        super().__init__(**options)
        sp = self.options.get("split", "evaluation")
        if sp not in SPLITS:
            raise ValueError(f"m5: split must be one of {tuple(SPLITS)}")
        self.origin = int(self.options.get("origin", SPLITS[sp]))
        self.history = int(self.options.get("history", 730))
        if self.history < 1:
            raise ValueError("m5: history must be >= 1")
        b = self.options.get("baselines", "+".join(BASELINES))
        self.baseline_names = tuple(b.split("+") if isinstance(b, str) else b)
        bad = sorted(set(self.baseline_names) - set(BASELINES))
        if bad:
            raise ValueError(f"m5: unknown baselines {bad}; have {BASELINES}")

    # ------------------------------------------------------------------ load
    def _select(self, keys: pd.DataFrame) -> np.ndarray:
        sel = np.arange(len(keys))
        st = self.options.get("stores")
        if st:
            want = st.split("+") if isinstance(st, str) else list(st)
            sel = sel[keys["store_id"].isin(want).to_numpy()]
            if not len(sel):
                raise ValueError(f"m5: no series in stores {want}")
        n = self.options.get("n_series")
        if n is not None and int(n) < len(sel):
            rng = np.random.default_rng(int(self.options.get("seed", 0)))
            sel = np.sort(rng.choice(sel, int(n), replace=False))
        return sel

    def _load(self, root: str) -> TaskData:
        files = _Files(root)
        o, H = self.origin, HORIZON
        cal = read_calendar(files)
        if o < 2 or o + H > len(cal):
            raise ValueError(f"m5: origin {o} + {H} outside the calendar (1..{len(cal)})")
        keys, Y = read_sales(files, o + H)
        n_all = len(keys)
        sel = self._select(keys)
        keys, Y = keys.iloc[sel].reset_index(drop=True), Y[sel]
        d0 = max(o - self.history + 1, 1)
        days = np.arange(min(d0, o - H + 1), o + H + 1)            # covers the weight window and the test window
        P = price_matrix(files, keys, cal, days)
        wsl = slice(o - H + 1 - days[0], o + 1 - days[0])           # last 28 training days
        dollars = np.nansum(Y[:, o - H:o] * P[:, wsl], axis=1)
        levels, per = hierarchy(keys, Y[:, :o], dollars)

        def long(lo: int, hi: int) -> pd.DataFrame:
            dd = np.arange(lo, hi + 1)
            n, T = len(keys), len(dd)
            c = cal.loc[dd]
            out = {"id": np.repeat(keys["id"].to_numpy(), T)}
            for k in KEYS:
                out[k] = np.repeat(keys[k].to_numpy(), T)
            out["d"] = np.tile(dd.astype(np.int32), n)
            out["sales"] = Y[:, lo - 1:hi].ravel()
            out["sell_price"] = P[:, lo - days[0]:hi - days[0] + 1].ravel()
            out["wday"] = np.tile(c["wday"].to_numpy(np.int8), n)
            snap = np.zeros((n, T), np.int8)
            for s in keys["state_id"].unique():
                if f"snap_{s}" in c.columns:
                    snap[(keys["state_id"] == s).to_numpy()] = c[f"snap_{s}"].to_numpy(np.int8)
            out["snap"] = snap.ravel()
            out["event"] = np.tile(c["event"].to_numpy(np.int8), n)
            df = pd.DataFrame(out)
            for k in ("id", *KEYS):
                df[k] = pd.Categorical(df[k], categories=sorted(keys[k].unique()))
            return df

        train, test = long(d0, o), long(o + 1, o + H)
        test.insert(test.columns.get_loc("d") + 1, "h", (test["d"] - o).astype(np.int16))
        sp = self.options.get("split", "evaluation") if "origin" not in self.options else f"origin d_{o}"
        meta = {"split": f"{sp}: train d_{d0}-d_{o} (history {o - d0 + 1} days), test d_{o + 1}-d_{o + H}",
                "origin": o, "horizon": H, "n_series": len(keys), "subsample": len(keys) < n_all,
                "calendar": cal, "levels": levels, "series": per, "keys": keys,
                "scale_rule": "M5 RMSSE denominator over d_1..origin from the first non-zero sale",
                "weight_rule": f"dollar sales d_{o - H + 1}-d_{o} (units x sell_price), normalised per level",
                "source": self.source}
        return TaskData(train=train, test=test, meta=meta)

    # ------------------------------------------------------------------ baselines
    def baselines(self, data: TaskData) -> dict:
        panel, o = future_panel(data), data.meta["origin"]
        seed = int(self.options.get("seed", 0))
        out: dict[str, np.ndarray] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)              # pandas observed= default on categoricals
            for b in self.baseline_names:
                if b == "naive":
                    fr = fc.persistence_forecast(panel, o, HORIZON, **COLS)
                elif b == "snaive":
                    fr = fc.seasonal_naive(panel, o, HORIZON, season=7, **COLS)
                elif b == "ses":
                    fr = fc.ses(panel, o, HORIZON, **COLS)
                elif b == "ets_lite":
                    fr = ets_lite(panel, o, HORIZON)
                else:
                    out[b] = _lightgbm(data, origins=int(self.options.get("lgb_origins", 28)),
                                       jobs=self.options.get("lgb_jobs"), seed=seed)
                    continue
                out[b] = align(data, fr)
        return out

    # ------------------------------------------------------------------ frame
    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        cal, per = data.meta["calendar"], data.meta["series"]
        ids = t["id"].astype(str).to_numpy()
        base = pd.DataFrame({"unit": t["id"].astype(str) + "|d_" + t["d"].astype(str), "group": ids,
                             "item": t["item_id"].astype(str).to_numpy(), "time": cal["date"].reindex(t["d"]).to_numpy(),
                             "d": t["d"].to_numpy(), "h": t["h"].to_numpy(), "truth": t["sales"].to_numpy(float),
                             "segment": t["cat_id"].astype(str).to_numpy(),
                             "seg_store": t["store_id"].astype(str).to_numpy(),
                             "seg_state": t["state_id"].astype(str).to_numpy()})
        for k in KEYS:
            base[k] = t[k].to_numpy()
        at = per.index.get_indexer(ids)
        for c in per.columns:
            base[c] = per[c].to_numpy()[at]
        return base

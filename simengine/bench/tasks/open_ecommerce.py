"""open_ecommerce: which of 20 Amazon categories will a person buy from in the next 6 months (TALP "73%" setup).

Ported from the talp73 branch (`PROTOKOL.md` v1.0, `kod/panel/panel.py`, `kod/klasik/klasik.py`, `kod/degerlendir.py`).
Persons = users fully observed through 2022 (last purchase >= 2023-01-01); history = purchases before 2022-07-01;
test window 2022-07-01 -> 2022-12-31 (2022-H2 holdout). The 20 categories have the most distinct buyers in history.
A sample of 400 persons (seed s: default_rng(s).choice(sorted(uids), 400)) x 20 categories = 8,000 test rows.
Classic baselines train at cut 2022-01-01 (labels 2022-01-01 -> 2022-06-30) on persons outside the sample.

Primary metric `acc_1b_c`: balanced within-person accuracy. For each person k = min(5, #bought, #not bought)
bought + k not-bought categories are drawn (default_rng(1000 + s)); the half with the highest p is called "yes"
(ties: a separate U(0,1) matrix from default_rng(1000 + s)). Chance = 0.5. Person bootstrap via a weight matrix
(default_rng(42), B = 1000), as in the source.

    task = bench.get("open_ecommerce", seed=0)
    data = task.load(root)                     # ValueError names the "fully observed" rule if too few persons
    frame = task.run(data=data)                # baselines; survey_lgbm is skipped (warning) if Q-* columns lack
    truth = truth_dict(data)                   # {uid: {category: 0/1}} for sim.prediction_frame(truth=...)
    people = survey_people(data)               # {uid: {Q-* column: answer}} for sim.predict_people (sim.AMAZON)
"""
from __future__ import annotations

import os
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, Metric, need, register

HISTORY_END = date(2022, 7, 1)     # history < this; test window starts here
TEST_END = date(2023, 1, 1)
TRAIN_CUT = date(2022, 1, 1)       # classic training: features before, labels in [cut, cut + 181 d)
LABEL_DAYS = 181
N_CAT, N_PERSONS = 20, 400
SURVEY_COLS = {
    "Survey ResponseID": "uid",
    "Q-demos-age": "yas", "Q-demos-hispanic": "hispanik", "Q-demos-race": "irk", "Q-demos-education": "egitim",
    "Q-demos-income": "gelir", "Q-demos-gender": "cinsiyet", "Q-sexual-orientation": "cinsel_yonelim",
    "Q-demos-state": "eyalet", "Q-amazon-use-howmany": "hesap_kisi", "Q-amazon-use-hh-size": "hane_buyuklugu",
    "Q-amazon-use-how-oft": "siklik_beyan", "Q-substance-use-cigarettes": "sigara", "Q-substance-use-marijuana": "esrar",
    "Q-substance-use-alcohol": "alkol", "Q-personal-diabetes": "diyabet", "Q-personal-wheelchair": "tekerlekli_sandalye",
    "Q-life-changes": "hayat_degisimi_2021", "Q-sell-YOUR-data": "veri_satis_kendi",
    "Q-sell-consumer-data": "veri_satis_genel", "Q-small-biz-use": "kucuk_isletme_erisim",
    "Q-census-use": "nufus_sayimi_kullanim", "Q-research-society": "arastirma_kullanim",
}
DEMOGRAPHICS = ["yas", "hispanik", "irk", "egitim", "gelir", "cinsiyet", "cinsel_yonelim", "eyalet", "hesap_kisi",
                "hane_buyuklugu"]
LIFE = {"Moved place of residence": "hd_tasindi", "Lost a job": "hd_is_kaybi", "Had a child": "hd_cocuk",
        "Became pregnant": "hd_hamile", "Divorce": "hd_bosanma"}
STATEMENTS = ["siklik_beyan", "sigara", "esrar", "alkol", "diyabet", "tekerlekli_sandalye", "veri_satis_kendi",
              "veri_satis_genel", "kucuk_isletme_erisim", "nufus_sayimi_kullanim", "arastirma_kullanim"] + list(LIFE.values())
SURVEY_FEATURES = DEMOGRAPHICS + STATEMENTS
# internal feature name -> raw survey.csv column (the LIFE flags all come from Q-life-changes)
RAW_COLUMN = {**{v: k for k, v in SURVEY_COLS.items()}, **{v: "Q-life-changes" for v in LIFE.values()}}
SURVEY_RAW_REQUIRED = sorted({RAW_COLUMN[f] for f in SURVEY_FEATURES})   # needed by the survey_lgbm baseline
FULLY_OBSERVED_RULE = f"a person is fully observed if their last purchase is on or after {TEST_END}"
LGBM_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)


def read_purchases(root: str) -> pd.DataFrame:
    """uid, date (datetime64), price, qty, category (may be NaN)."""
    a = pd.read_csv(need(os.path.join(root, "amazon-purchases.csv")), dtype=str, keep_default_na=False,
                    na_values=[""], usecols=["Order Date", "Purchase Price Per Unit", "Quantity", "Category",
                                             "Survey ResponseID"])
    return pd.DataFrame({"uid": a["Survey ResponseID"],
                         "date": pd.to_datetime(a["Order Date"], format="%Y-%m-%d", errors="coerce"),
                         "price": pd.to_numeric(a["Purchase Price Per Unit"], errors="coerce"),
                         "qty": pd.to_numeric(a["Quantity"], errors="coerce"),
                         "category": a["Category"]})


def read_survey_raw(root: str) -> pd.DataFrame:
    """survey.csv as is: Survey ResponseID + raw Q-* columns (the keys of sim.AMAZON.questions)."""
    return pd.read_csv(need(os.path.join(root, "survey.csv")), dtype=str, keep_default_na=False, na_values=[""])


def survey_internal(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw survey -> internal (talp73) names: uid, yas, cinsiyet, ... plus hd_* life-change flags when
    Q-life-changes is present. Columns missing in the raw file stay missing."""
    s = raw.rename(columns=SURVEY_COLS)
    if "hayat_degisimi_2021" in s.columns:
        hd = s["hayat_degisimi_2021"].fillna("")
        for k, v in LIFE.items():
            s[v] = hd.str.contains(k, regex=False).astype(int).astype(str)
    return s


def read_survey(root: str) -> pd.DataFrame:
    """Survey with internal names (see survey_internal); read_survey_raw keeps the raw Q-* names."""
    return survey_internal(read_survey_raw(root))


def missing_survey_columns(survey: pd.DataFrame, features: list | None = None) -> list:
    """Raw Q-* columns needed for `features` (default SURVEY_FEATURES) that `survey` (internal names) lacks."""
    return sorted({RAW_COLUMN[f] for f in (features or SURVEY_FEATURES) if f not in survey.columns})


def _ts(d: date) -> pd.Timestamp:
    return pd.Timestamp(d)


def history_features(a: pd.DataFrame, cats: list, uids: list, cut: date) -> pd.DataFrame:
    """uid x category features from purchases before `cut` (talp73 `gecmis_ozellik`), rows = uids x cats in order."""
    c = _ts(cut)
    g = a[a["date"] < c].copy()
    g["age"] = (c - g["date"]).dt.days
    g["spend"] = g["price"].fillna(0) * g["qty"].fillna(1)
    gk = g[g["category"].isin(cats)]
    last_y = (c - timedelta(days=365), c - timedelta(days=365 - LABEL_DAYS))
    gk = gk.assign(**{f"_k{d}": (gk["age"] <= d).astype(int) for d in (30, 90, 180, 365, 730)})
    grp = gk.groupby(["uid", "category"])
    f = pd.DataFrame({f"k_{d}": grp[f"_k{d}"].sum() for d in (30, 90, 180, 365, 730)})
    f["k_tum"] = grp.size()
    f["gun_tum"] = grp["date"].nunique()
    f["gun_365"] = gk[gk["age"] <= 365].groupby(["uid", "category"])["date"].nunique()
    f["son_yas"] = grp["age"].min()
    f["ilk_yas"] = grp["age"].max()
    for t in (90, 180, 365):
        f[f"son_{t}"] = np.exp(-gk["age"] / t).groupby([gk["uid"], gk["category"]]).sum()
    f["k_gecen_yil_ayni"] = ((gk["date"] >= last_y[0]) & (gk["date"] < last_y[1])).groupby(
        [gk["uid"], gk["category"]]).sum()
    f["harcama_tum"] = grp["spend"].sum()
    y365 = g[g["age"] <= 365]
    ug = g.groupby("uid")
    u = pd.DataFrame({"u_k_365": y365.groupby("uid").size(), "u_k_tum": ug.size(),
                      "u_kat_365": y365.groupby("uid")["category"].nunique(dropna=False),
                      "u_ilk_yas": ug["age"].max(), "u_son_yas": ug["age"].min(),
                      "u_harcama_365": y365.groupby("uid")["spend"].sum()})
    grid = pd.MultiIndex.from_product([uids, cats], names=["uid", "category"])
    d = f.reindex(grid)
    for col in d.columns:
        if col not in ("son_yas", "ilk_yas"):
            d[col] = d[col].fillna(0)
    d = d.reset_index()
    d = d.join(u.reindex(d["uid"]).reset_index(drop=True))
    for col in ("u_k_365", "u_k_tum", "u_kat_365", "u_harcama_365"):
        d[col] = d[col].fillna(0)
    d["pay_365"] = d["k_365"] / d["u_k_365"].clip(lower=1)
    return d


HISTORY_FEATURES = ["k_30", "k_90", "k_180", "k_365", "k_730", "k_tum", "gun_tum", "gun_365", "son_yas", "ilk_yas",
                    "son_90", "son_180", "son_365", "k_gecen_yil_ayni", "harcama_tum", "u_k_365", "u_k_tum", "u_kat_365",
                    "u_ilk_yas", "u_son_yas", "u_harcama_365", "pay_365"]


def labels(a: pd.DataFrame, cats: list, uids: list, start: date, end: date) -> np.ndarray:
    w = a[(a["date"] >= _ts(start)) & (a["date"] < _ts(end)) & a["category"].isin(cats)]
    bought = set(zip(w["uid"], w["category"]))
    return np.array([int((u, k) in bought) for u in uids for k in cats], dtype=np.int8)


def balanced_mask(Y: np.ndarray, s: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + s)
    M = np.zeros_like(Y, dtype=bool)
    for i in range(Y.shape[0]):
        pos, neg = np.flatnonzero(Y[i] == 1), np.flatnonzero(Y[i] == 0)
        k = min(5, len(pos), len(neg))
        if k:
            M[i, rng.choice(pos, k, replace=False)] = True
            M[i, rng.choice(neg, k, replace=False)] = True
    return M


def half_decision(P, M, G):
    D = np.zeros_like(M, dtype=np.int8)
    for i in range(M.shape[0]):
        j = np.flatnonzero(M[i])
        if len(j):
            order = j[np.lexsort((-G[i, j], -P[i, j]))]
            D[i, order[:len(j) // 2]] = 1
    return D


def person_weights(N: int, B: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = np.zeros((B, N))
    for b in range(B):
        np.add.at(W[b], rng.integers(0, N, N), 1)
    return W


def _matrices(frame, col):
    persons = pd.unique(frame["group"])
    N = len(persons)
    K = len(frame) // N
    sh = (N, K)
    return (frame[col].to_numpy(float).reshape(sh), frame["truth"].to_numpy(int).reshape(sh),
            frame["balanced"].to_numpy(bool).reshape(sh), frame["tiebreak"].to_numpy(float).reshape(sh))


def _weighted(fn, frame, col, B, seed):
    P, Y, M, G = _matrices(frame, col)
    N = P.shape[0]
    point = float(fn(np.ones((1, N)), P, Y, M, G)[0])
    boots = fn(person_weights(N, B, seed), P, Y, M, G)
    b = boots[np.isfinite(boots)]
    lo, hi = (np.percentile(b, [2.5, 97.5]) if len(b) >= 10 else (np.nan, np.nan))
    return {"value": point, "lo": float(lo), "hi": float(hi), "n": N}


def _acc_1b_c(task, frame, col, B, seed):
    def f(W, P, Y, M, G):
        D = half_decision(P, M, G)
        return (W @ ((D == Y) & M).sum(1)) / (W @ M.sum(1))
    return _weighted(f, frame, col, B, seed)


def _acc_1a(task, frame, col, B, seed):
    def f(W, P, Y, M, G):
        return (W @ ((P >= 0.5).astype(int) == Y).sum(1)) / (W.sum(1) * Y.shape[1])
    return _weighted(f, frame, col, B, seed)


def _person_auc(task, frame, col, B, seed):
    def f(W, P, Y, M, G):
        pos, neg = Y == 1, Y == 0
        diff = P[:, :, None] - P[:, None, :]
        pair = pos[:, :, None] & neg[:, None, :]
        win = (((diff > 0) + 0.5 * (diff == 0)) * pair).sum((1, 2))
        nc = pair.sum((1, 2)).astype(float)
        ok = nc > 0
        auc = np.where(ok, win / np.where(ok, nc, 1), 0.0)
        return (W @ auc) / (W @ ok)
    return _weighted(f, frame, col, B, seed)


@register()
class OpenEcommerce(BenchTask):
    """Options: seed=0 (sample seed s in {0, 1, 2} of the source; any int works); n_persons=400, n_categories=20
    (protocol values; smaller only for toy data); lgbm=True (LightGBM baselines, skipped if lightgbm is absent)."""

    name = "open_ecommerce"
    accepted_options = {
        'seed': 'int (default 0)',
        'n_categories': 'int (default 20)',
        'n_persons': 'int (default 400)',
        'lgbm': 'true | false (default true)',
    }
    kind = "binary"
    data_subdir = "open-ecommerce"
    layout = f"""
    <root>/open-ecommerce/amazon-purchases.csv   Order Date, Purchase Price Per Unit, Quantity, Category,
                                                 Survey ResponseID (Berke et al. 2024 Open e-commerce 1.0)
    <root>/open-ecommerce/survey.csv             Survey ResponseID + Q-demos-* / Q-* survey answers
    persons: {FULLY_OBSERVED_RULE}; at least n_persons (default {N_PERSONS})
             such persons are needed. History < {HISTORY_END}, test window {HISTORY_END} -> {TEST_END} (exclusive).
    survey_lgbm baseline needs these survey.csv columns (skipped with a warning otherwise):
             {", ".join(SURVEY_RAW_REQUIRED)}
    """
    source = "talp-simulation-hackathon-2026 uretim/talp73/PROTOKOL.md, kod/panel/panel.py, kod/klasik/klasik.py, kod/degerlendir.py"
    metrics = (Metric("acc_1b_c", True, _acc_1b_c, "balanced within-person top-half accuracy (chance 0.5)"),
               Metric("acc_1a", True, _acc_1a, "accuracy of p >= 0.5 on all test rows (8,000 in the protocol; 'never buys' ~0.77)"),
               Metric("person_auc", True, _person_auc, "mean within-person AUC (bought vs not bought)"))

    def load(self, data_root: str | None = None) -> TaskData:
        """TaskData for the 2022-H2 holdout.

        test: uid, category, y (bought in [2022-07-01, 2023-01-01)), n_persons x n_categories rows, person-major.
        train: uid, category, y (bought in [2022-01-01, 2022-07-01)) for persons outside the sample.
        meta keys: sample (sorted test uids), train_persons, categories (top categories by distinct buyers in
        history), prior_rate (Series category -> train buy rate), purchases (all purchases: uid, date, price, qty,
        category), survey (internal talp73 names: uid, yas, cinsiyet, ..., hd_*), survey_raw (raw Survey
        ResponseID / Q-* columns as keyed in sim.AMAZON.questions), survey_missing (raw columns absent from
        survey.csv), n_fully_observed, history_end, train_cut, mask (n_persons x n_categories bool, balanced
        subset), tiebreak (same shape, U(0,1)), split (text), seed, source.
        Raises ValueError naming the fully observed rule when fewer than n_persons persons qualify."""
        return super().load(data_root)

    def _load(self, root: str) -> TaskData:
        s = int(self.options.get("seed", 0))
        a = read_purchases(root)
        last = a.groupby("uid")["date"].max()
        persons = sorted(last.index[last >= _ts(TEST_END)])
        persons_prev = sorted(last.index[last >= _ts(HISTORY_END)])
        hist = a[a["uid"].isin(set(persons)) & (a["date"] < _ts(HISTORY_END)) & a["category"].notna()]
        n_cat, n_per = int(self.options.get("n_categories", N_CAT)), int(self.options.get("n_persons", N_PERSONS))
        if len(persons) < n_per:
            raise ValueError(f"only {len(persons)} fully observed persons, need n_persons={n_per} "
                             f"({FULLY_OBSERVED_RULE}; {len(persons_prev)} persons have a purchase on or after "
                             f"{HISTORY_END}). Lower n_persons for toy data.")
        buyers = hist.drop_duplicates(["uid", "category"]).groupby("category").size().rename("n").reset_index()
        cats = buyers.sort_values(["n", "category"], ascending=[False, True]).head(n_cat)["category"].tolist()
        sample = sorted(np.random.default_rng(s).choice(persons, n_per, replace=False).tolist())
        y = labels(a, cats, sample, HISTORY_END, TEST_END)
        test = pd.DataFrame({"uid": np.repeat(sample, len(cats)), "category": np.tile(cats, len(sample)), "y": y})
        out = sorted(set(persons_prev) - set(sample))
        w = a[a["uid"].isin(set(out)) & (a["date"] >= _ts(TRAIN_CUT)) & (a["date"] < _ts(HISTORY_END))
              & a["category"].isin(cats)].drop_duplicates(["uid", "category"])
        prior = (w.groupby("category").size().reindex(cats).fillna(0) / len(out)).rename("rate")
        y_tr = labels(a, cats, out, TRAIN_CUT, TRAIN_CUT + timedelta(days=LABEL_DAYS))
        train = pd.DataFrame({"uid": np.repeat(out, len(cats)), "category": np.tile(cats, len(out)), "y": y_tr})
        Y = y.reshape(len(sample), len(cats))
        raw = read_survey_raw(root)
        survey = survey_internal(raw)
        meta = {"split": f"test: 2022-07-01 -> 2022-12-31 purchases of a {n_per}-person sample; history < 2022-07-01; "
                         "train: labels 2022-01-01 -> 2022-06-30 for persons outside the sample (features < 2022-01-01)",
                "seed": s, "categories": cats, "sample": sample, "train_persons": out, "prior_rate": prior,
                "purchases": a, "survey": survey, "survey_raw": raw,
                "survey_missing": missing_survey_columns(survey), "n_fully_observed": len(persons),
                "history_end": str(HISTORY_END), "train_cut": str(TRAIN_CUT),
                "mask": balanced_mask(Y, s), "tiebreak": np.random.default_rng(1000 + s).random(Y.shape),
                "source": "Open e-commerce 1.0 (Berke et al. 2024, Harvard Dataverse); CC BY 4.0"}
        return TaskData(train=train, test=test, meta=meta)

    # ------------------------------------------------------------------ baselines
    def history_features(self, data: TaskData, uids: list, cut: date) -> pd.DataFrame:
        return history_features(data.meta["purchases"], data.meta["categories"], uids, cut)

    def _lgbm(self, data, cols, s):
        import lightgbm as lgb  # optional dependency
        cats, sv = data.meta["categories"], data.meta["survey"]
        used = [c for c in SURVEY_FEATURES if c in cols]   # history_lgbm needs no survey columns
        surv = sv[["uid"] + used].drop_duplicates("uid").set_index("uid")
        levels = {c: sorted(surv[c].dropna().unique().tolist()) for c in used}

        def table(uids, cut):
            d = self.history_features(data, uids, cut)
            d = d.join(surv.reindex(d["uid"]).reset_index(drop=True))
            x = d[cols].copy()
            x["category"] = pd.Categorical(x["category"], categories=cats)
            for c in cols:
                if c in levels:
                    x[c] = pd.Categorical(x[c], categories=levels[c])
            return x, d["uid"].to_numpy()

        xe, kisi = table(data.meta["train_persons"], TRAIN_CUT)
        ye = data.train["y"].to_numpy()
        xt, _ = table(data.meta["sample"], HISTORY_END)
        rng = np.random.default_rng(100 + s)
        ku = np.array(sorted(set(kisi)))
        inner = set(rng.choice(ku, int(0.2 * len(ku)), replace=False))
        m = np.array([k in inner for k in kisi])
        p = dict(LGBM_PARAMS, seed=s)
        b = lgb.train(p, lgb.Dataset(xe[~m], ye[~m]), 3000, valid_sets=[lgb.Dataset(xe[m], ye[m])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        final = lgb.train(p, lgb.Dataset(xe, ye), b.best_iteration)
        return np.clip(final.predict(xt), 0, 1)

    def baselines(self, data: TaskData) -> dict:
        """{name: p per data.test row}: never, constant_rate, random, popularity (prior_rate of the category),
        persistence (bought in the last 180 days of history), decayed_count; with lgbm=True and lightgbm installed
        also survey_lgbm (category + survey features) and history_lgbm (category + history features).
        survey_lgbm is skipped with a warning naming the raw Q-* columns when survey.csv lacks them."""
        s = data.meta["seed"]
        n = len(data.test)
        cats = data.meta["categories"]
        ft = self.history_features(data, data.meta["sample"], HISTORY_END)
        out = {"never": np.zeros(n), "constant_rate": np.full(n, float(data.train["y"].mean())),
               "random": np.random.default_rng(s).random(n),
               "popularity": data.test["category"].map(data.meta["prior_rate"]).to_numpy(float),
               "persistence": (ft["k_180"].to_numpy() > 0).astype(float),
               "decayed_count": 1 - np.exp(-ft["son_180"].to_numpy(float))}
        if self.options.get("lgbm", True):
            missing = missing_survey_columns(data.meta["survey"])
            if missing:
                warnings.warn(f"open_ecommerce: skipping survey_lgbm, survey.csv lacks {missing}", stacklevel=2)
            try:
                import lightgbm  # noqa: F401
            except ImportError:
                return out
            hist = ["category"] + HISTORY_FEATURES
            if not missing:
                out["survey_lgbm"] = self._lgbm(data, ["category"] + SURVEY_FEATURES, s)
            out["history_lgbm"] = self._lgbm(data, hist, s)
        assert len(cats) * len(data.meta["sample"]) == n
        return out

    def frame(self, data: TaskData, preds: dict) -> pd.DataFrame:
        """EvalFrame, one row per data.test row: unit ("uid|category"), group (= uid), item (= category),
        uid, category, truth (0/1), balanced (row is in the acc_1b_c subset), tiebreak (U(0,1) for equal p),
        segment (age group, Q-demos-age), seg_gender (Q-demos-gender), split ("test"), pred_<method>."""
        return super().frame(data, preds)

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        sv = data.meta["survey"].drop_duplicates("uid").set_index("uid")

        def seg(col):
            return t["uid"].map(sv[col]).fillna("NA").to_numpy() if col in sv.columns else np.full(len(t), "NA")
        return pd.DataFrame({"unit": t["uid"] + "|" + t["category"], "group": t["uid"], "item": t["category"],
                             "uid": t["uid"].to_numpy(), "category": t["category"].to_numpy(),
                             "truth": t["y"].to_numpy(int), "balanced": data.meta["mask"].ravel(),
                             "tiebreak": data.meta["tiebreak"].ravel(),
                             "segment": seg("yas"), "seg_gender": seg("cinsiyet")})


def truth_dict(data: TaskData) -> dict:
    """{uid: {category: 0/1}} of data.test (the `truth` argument of sim.prediction_frame)."""
    out: dict = {}
    for u, k, y in zip(data.test["uid"], data.test["category"], data.test["y"]):
        out.setdefault(u, {})[k] = int(y)
    return out


def survey_people(data: TaskData, uids: list | None = None) -> dict:
    """{uid: {raw Q-* column: answer}} for `uids` (default the test sample), missing answers left out; the
    `people` argument of sim.predict_people with spec=sim.AMAZON (whose questions are keyed by these columns)."""
    raw = data.meta["survey_raw"].drop_duplicates("Survey ResponseID").set_index("Survey ResponseID")
    out = {}
    for u in (data.meta["sample"] if uids is None else uids):
        if u in raw.index:
            out[u] = {c: v for c, v in raw.loc[u].items() if isinstance(v, str)}
    return out

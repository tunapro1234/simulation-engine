"""uplift: rank customers by how much a treatment (an e-mail, an ad, an SMS) raises their chance to respond.

Three randomised-experiment datasets, one task each:

    hillstrom_uplift   MineThatData e-mail challenge (Hillstrom 2008): 64,000 customers, mens / womens / no e-mail,
                       outcome visit (default) or conversion within two weeks
    criteo_uplift      Criteo Uplift v2.1 (Diemert et al. 2021): 13,979,592 users, treatment = targetable for ads
                       (85%), outcome visit (default) or conversion; 12 anonymised dense features
    x5_uplift          X5 RetailHero (ODS.ai 2019-20): 200,039 clients, treatment = SMS (50%), outcome = purchase;
                       client profile features only (age, gender, card issue / first redeem dates, a first redeem
                       after the 2019-03-18 end of the pre-campaign window masked as post-treatment)

    from simengine import bench
    from simengine.bench.tasks import uplift
    task = uplift.HillstromUplift(arm="womens")                  # or bench.get("hillstrom_uplift", arm="womens")
    data = task.load("/path/to/catalogue")                       # or $SIMENGINE_DATA
    frame = task.run(methods={"mine": lambda d: my_uplift(d.train, d.test)}, data=data)
    task.score(frame, "mine")                                    # {'metric': 'qini', 'value', 'lo', 'hi', 'n'}

A prediction is an uplift *score* per test row (higher = treat first); only its ranking matters. The EvalFrame has
`truth` (outcome 0/1) and `treatment` (0/1); `pred_<method>` columns are scores, not probabilities, so the generic
binary metrics in `simengine.evidence` do not apply: score with `task.score`.

Split: stratified random split by (treatment, outcome), test_frac = 0.2 (the 80/20 protocol of Diemert et al. 2021),
default_rng(seed = 0). Metrics, all with a row bootstrap (default_rng(42), B = 1000; replicates are multinomial
row-count weights on a fixed sort, so a replicate is O(n)):

    qini     normalised Qini coefficient: (area under the Qini curve - random) / (perfect - random), the perfect curve
             allowing negative effects (Radcliffe 2007; same definition as scikit-uplift `qini_auc_score`)
    auuc     normalised area under the uplift curve, same normalisation (scikit-uplift `uplift_auc_score`)
    uplift@30  treated minus control response rate among the top int(0.3 n) rows by score (scikit-uplift
               `uplift_at_k`, strategy "overall")

Baselines: random, two-model / T-learner (logistic, GBDT), class transformation (Jaskowski & Jaroszewicz 2012;
treated / control reweighted to 50:50 when the treatment share is not one half), S-learner (GBDT with treatment as a
feature). GBDT = LightGBM with fixed parameters (sklearn HistGradientBoosting if LightGBM is missing).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, DataNotFound, Metric, need, register
from ..stats import ci

TOP_K = 0.3
GBDT_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.9, random_state=0, verbose=-1, n_jobs=4)


# ---------------------------------------------------------------------- curves and metrics
def _thresholds(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descending order and the last index of every distinct-score block (ties move together)."""
    order = np.argsort(score, kind="mergesort")[::-1]
    s = score[order]
    return order, np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]


def _cum(y, t, w, order, thr):
    """Weighted cumulative counts at the thresholds: (n_all, n_treated, y_treated, n_control, y_control)."""
    y, t, w = y[order], t[order], w[order]
    na, nt = np.cumsum(w)[thr], np.cumsum(w * t)[thr]
    yt, yc = np.cumsum(w * y * t)[thr], np.cumsum(w * y * (1 - t))[thr]
    return na, nt, yt, na - nt, yc


def _div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def _curve(kind, y, t, w, order, thr):
    na, nt, yt, nc, yc = _cum(y, t, w, order, thr)
    v = yt - yc * _div(nt, nc) if kind == "qini" else (_div(yt, nt) - _div(yc, nc)) * na
    return np.r_[0.0, na], np.r_[0.0, v]


def _area(x, v):
    return float(np.sum(np.diff(x) * (v[1:] + v[:-1]) / 2))


def _norm_area(kind, y, t, w, model, perfect):
    """(area(model) - area(random)) / (area(perfect) - area(random)); random = straight line to the end point."""
    x, v = _curve(kind, y, t, w, *model)
    xp, vp = _curve(kind, y, t, w, *perfect)
    base = 0.5 * xp[-1] * vp[-1]
    den = _area(xp, vp) - base
    return (_area(x, v) - base) / den if den != 0 else float("nan")


def _perfect_qini_score(y, t):
    return y * t - y * (1 - t)  # treated responders first, control responders last


def _perfect_uplift_scores(y, t):
    """The two candidate perfect orderings of scikit-uplift `perfect_uplift_curve` (chosen per sample)."""
    eq = 2.0 * (y == t)
    return eq + y, eq + t  # summand y if control responders > treated non-responders, else t


def qini_curve(y, treatment, score, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """Qini curve: x = number targeted, value = Y_t(x) - Y_c(x) * N_t(x) / N_c(x) (Radcliffe 2007), at every
    distinct score, starting at (0, 0)."""
    y, t, s = (np.asarray(a, float) for a in (y, treatment, score))
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    return _curve("qini", y, t, w, *_thresholds(s))


def uplift_curve(y, treatment, score, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """Uplift curve: x = number targeted, value = (Y_t / N_t - Y_c / N_c)(x) * x, starting at (0, 0)."""
    y, t, s = (np.asarray(a, float) for a in (y, treatment, score))
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    return _curve("uplift", y, t, w, *_thresholds(s))


def qini_coefficient(y, treatment, score, weights=None) -> float:
    """Normalised Qini coefficient: 1 = perfect ranking (negative effects allowed), 0 = random."""
    y, t, s = (np.asarray(a, float) for a in (y, treatment, score))
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    return _norm_area("qini", y, t, w, _thresholds(s), _thresholds(_perfect_qini_score(y, t)))


def auuc(y, treatment, score, weights=None) -> float:
    """Normalised area under the uplift curve: 1 = perfect ranking, 0 = random."""
    y, t, s = (np.asarray(a, float) for a in (y, treatment, score))
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    py, pt = _perfect_uplift_scores(y, t)
    return _auuc_w(y, t, w, _thresholds(s), _thresholds(py), _thresholds(pt))


def _auuc_w(y, t, w, model, perf_y, perf_t):
    cr = float(np.sum(w * y * (1 - t)))          # control responders
    tn = float(np.sum(w * (1 - y) * t))          # treated non-responders
    return _norm_area("uplift", y, t, w, model, perf_y if cr > tn else perf_t)


def uplift_at_k(y, treatment, score, k: float = TOP_K, weights=None) -> float:
    """Treated minus control response rate among the first int(k n) rows by score (ties broken by sort order)."""
    y, t, s = (np.asarray(a, float) for a in (y, treatment, score))
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    return _uak(y, t, w, np.argsort(s, kind="mergesort")[::-1], k)


def _uak(y, t, w, order, k):
    y, t, w = y[order], t[order], w[order]
    top = np.cumsum(w) <= k * w.sum()  # unit weights: the first int(k n) rows, as scikit-uplift
    wt, wc = w * top * t, w * top * (1 - t)
    if wt.sum() == 0 or wc.sum() == 0:
        return float("nan")
    return float((wt * y).sum() / wt.sum() - (wc * y).sum() / wc.sum())


def _boot(stat, n, B, seed):
    """Point (unit weights) and B multinomial-weight replicates = a row bootstrap without re-sorting."""
    point = stat(np.ones(n))
    rng = np.random.default_rng(seed)
    reps = np.array([stat(np.bincount(rng.integers(0, n, n), minlength=n).astype(float)) for _ in range(B)])
    lo, hi = ci(point, reps)
    return {"value": float(point), "lo": lo, "hi": hi, "n": int(n)}


def _arrays(frame, col):
    return (frame["truth"].to_numpy(float), frame["treatment"].to_numpy(float), frame[col].to_numpy(float))


def _m_qini(task, frame, col, B, seed):
    y, t, s = _arrays(frame, col)
    model, perf = _thresholds(s), _thresholds(_perfect_qini_score(y, t))
    return _boot(lambda w: _norm_area("qini", y, t, w, model, perf), len(y), B, seed)


def _m_auuc(task, frame, col, B, seed):
    y, t, s = _arrays(frame, col)
    py, pt = _perfect_uplift_scores(y, t)
    model, perf_y, perf_t = _thresholds(s), _thresholds(py), _thresholds(pt)
    return _boot(lambda w: _auuc_w(y, t, w, model, perf_y, perf_t), len(y), B, seed)


def _m_uak(task, frame, col, B, seed):
    y, t, s = _arrays(frame, col)
    order = np.argsort(s, kind="mergesort")[::-1]
    return _boot(lambda w: _uak(y, t, w, order, TOP_K), len(y), B, seed)


# ---------------------------------------------------------------------- learners
def _gbdt():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(**GBDT_PARAMS)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        warnings.warn("lightgbm not installed; using sklearn HistGradientBoostingClassifier", RuntimeWarning,
                      stacklevel=3)
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=200, random_state=0)


def _logistic(categorical, numeric):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    parts = [("s", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric)]
    if categorical:
        parts.append(("k", OneHotEncoder(handle_unknown="ignore"), categorical))
    return make_pipeline(ColumnTransformer(parts), LogisticRegression(C=1.0, max_iter=2000))


def _make(model, categorical, numeric):
    if callable(model):
        return model()
    if model == "logistic":
        return _logistic(categorical, numeric)
    if model == "gbdt":
        return _gbdt()
    raise ValueError(f"model must be 'logistic', 'gbdt' or a factory; got {model!r}")


def _X(df, categorical, numeric, model, vocab):
    x = df[numeric + categorical].copy()
    if model != "logistic":  # tree models: categories as integer codes on the training vocabulary
        for c in categorical:
            x[c] = pd.Categorical(x[c].astype(str), categories=vocab[c]).codes
    return x


def _vocab(train, categorical):
    return {c: sorted(train[c].astype(str).unique()) for c in categorical}


def _p1(m, X):
    return m.predict_proba(X)[:, 1]


def two_model(train, test, *, treatment: str, outcome: str, categorical: list[str], numeric: list[str],
              model="logistic") -> np.ndarray:
    """T-learner: separate outcome models on treated and control rows; uplift = P(y | x, t=1) - P(y | x, t=0)."""
    tr = train[treatment].to_numpy() == 1
    v = _vocab(train, categorical)
    Xa, Xt = _X(train, categorical, numeric, model, v), _X(test, categorical, numeric, model, v)
    m1 = _make(model, categorical, numeric).fit(Xa[tr], train[outcome].to_numpy()[tr])
    m0 = _make(model, categorical, numeric).fit(Xa[~tr], train[outcome].to_numpy()[~tr])
    return _p1(m1, Xt) - _p1(m0, Xt)


def s_learner(train, test, *, treatment: str, outcome: str, categorical: list[str], numeric: list[str],
              model="gbdt") -> np.ndarray:
    """S-learner: one outcome model with the treatment flag as a feature; uplift = f(x, 1) - f(x, 0)."""
    num, v = numeric + ["_t"], _vocab(train, categorical)
    Xa = _X(train.assign(_t=train[treatment].astype(float)), categorical, num, model, v)
    m = _make(model, categorical, num).fit(Xa, train[outcome].to_numpy())
    Xt = _X(test.assign(_t=1.0), categorical, num, model, v)
    return _p1(m, Xt) - _p1(m, Xt.assign(_t=0.0))


def class_transformation(train, test, *, treatment: str, outcome: str, categorical: list[str], numeric: list[str],
                         model="logistic") -> np.ndarray:
    """Class variable transformation (Jaskowski & Jaroszewicz 2012): z = y t + (1 - y)(1 - t), uplift = 2 P(z=1|x) - 1.
    The identity needs P(t=1) = 1/2; with any other share the rows are weighted so treated and control each carry
    half of the total weight (then 2 P_w(z=1|x) - 1 equals the uplift again)."""
    t, y = train[treatment].to_numpy(float), train[outcome].to_numpy(float)
    z = (y * t + (1 - y) * (1 - t)).astype(int)
    e = t.mean()
    w = np.where(t == 1, 0.5 / e, 0.5 / (1 - e))
    m, v = _make(model, categorical, numeric), _vocab(train, categorical)
    Xa = _X(train, categorical, numeric, model, v)
    if hasattr(m, "steps"):
        m.fit(Xa, z, **{f"{m.steps[-1][0]}__sample_weight": w})
    else:
        m.fit(Xa, z, sample_weight=w)
    return 2 * _p1(m, _X(test, categorical, numeric, model, v)) - 1


def stratified_split(df: pd.DataFrame, by: list[str], test_frac: float, seed: int) -> np.ndarray:
    """Boolean test mask: within every `by` stratum (taken in sorted order) a random round(test_frac * n) rows
    (default_rng(seed))."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(df), bool)
    for _, i in sorted(df.reset_index(drop=True).groupby(by, sort=True).indices.items()):
        test[rng.permutation(i)[: int(round(test_frac * len(i)))]] = True
    return test


# ---------------------------------------------------------------------- tasks
class UpliftTask(BenchTask):
    """Shared load / baselines / frame for the uplift tasks. Subclasses implement `_read(root) -> DataFrame` with
    columns `t` (0/1), the outcome columns, CATEGORICAL and NUMERIC, and set OUTCOMES (first = default)."""

    kind = "binary"
    CATEGORICAL: list[str] = []
    NUMERIC: list[str] = []
    OUTCOMES: tuple[str, ...] = ()
    SEGMENT: str | None = None
    LICENCE = ""
    metrics = (Metric("qini", True, _m_qini, "normalised Qini coefficient (perfect = 1, random = 0); row bootstrap"),
               Metric("auuc", True, _m_auuc, "normalised area under the uplift curve (perfect = 1, random = 0)"),
               Metric("uplift@30", True, _m_uak, "treated - control response rate in the top 30% by score"))
    BASELINES = ("random", "t_logistic", "t_gbdt", "class_transform", "s_gbdt")

    def _read(self, root: str) -> pd.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError

    def _load(self, root: str) -> TaskData:
        d = self._read(root)
        outcome = self.options.get("outcome", self.OUTCOMES[0])
        if outcome not in self.OUTCOMES:
            raise ValueError(f"{self.name}: outcome must be one of {self.OUTCOMES}")
        d = d.assign(y=d[outcome].astype(int), t=d["t"].astype(int))
        frac, seed = float(self.options.get("test_frac", 0.2)), int(self.options.get("seed", 0))
        if not 0 < frac < 1:
            raise ValueError("test_frac must be in (0, 1)")
        te = stratified_split(d, ["t", "y"], frac, seed)
        tr, test = d[~te].reset_index(drop=True), d[te].reset_index(drop=True)
        meta = {"split": f"stratified random by (treatment, outcome), test_frac={frac}, default_rng({seed})",
                "outcome": outcome, "treatment": "t", "categorical": list(self.CATEGORICAL),
                "numeric": list(self.NUMERIC), "n_train": len(tr), "n_test": len(test),
                "treated_share": float(d["t"].mean()),
                "ate_test": float(test.y[test.t == 1].mean() - test.y[test.t == 0].mean()),
                "source": self.LICENCE}
        return TaskData(train=tr, test=test, meta=meta)

    def baselines(self, data: TaskData) -> dict:
        tr, te = data.train, data.test
        kw = dict(treatment="t", outcome="y", categorical=list(self.CATEGORICAL), numeric=list(self.NUMERIC))
        keep = self.options.get("baselines")
        want = set(self.BASELINES if keep is None else
                   (keep.replace("+", ",").split(",") if isinstance(keep, str) else keep))
        unknown = want - set(self.BASELINES)
        if unknown:
            raise ValueError(f"{self.name}: unknown baseline(s) {sorted(unknown)}; have {self.BASELINES}")
        out = {}
        if "random" in want:
            out["random"] = np.random.default_rng(0).random(len(te))
        if "t_logistic" in want:
            out["t_logistic"] = two_model(tr, te, model="logistic", **kw)
        if "t_gbdt" in want:
            out["t_gbdt"] = two_model(tr, te, model="gbdt", **kw)
        if "class_transform" in want:
            out["class_transform"] = class_transformation(tr, te, model="logistic", **kw)
        if "s_gbdt" in want:
            out["s_gbdt"] = s_learner(tr, te, model="gbdt", **kw)
        return out

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        seg = t[self.SEGMENT].astype(str).to_numpy() if self.SEGMENT else np.full(len(t), "all")
        return pd.DataFrame({"unit": np.arange(len(t)), "truth": t["y"].to_numpy(int),
                             "treatment": t["t"].to_numpy(int), "segment": seg})


_COMMON_OPTIONS = {
    "test_frac": "float in (0, 1) (default 0.2)",
    "seed": "int, split seed (default 0)",
    "baselines": '"+"-joined subset of random, t_logistic, t_gbdt, class_transform, s_gbdt (default all)',
}


@register()
class HillstromUplift(UpliftTask):
    """Options: arm='any' (default; either e-mail vs none) | 'mens' | 'womens' (that e-mail vs none);
    outcome='visit' (default) | 'conversion'; test_frac, seed, baselines."""

    name = "hillstrom_uplift"
    data_subdir = "hillstrom"
    accepted_options = {"arm": '"any" | "mens" | "womens" (default "any")',
                        "outcome": '"visit" | "conversion" (default "visit")', **_COMMON_OPTIONS}
    layout = """
    <root>/hillstrom/ham/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
        (64,000 rows: recency, history_segment, history, mens, womens, zip_code, newbie, channel, segment, visit,
        conversion, spend; the file may also sit directly under <root>/hillstrom/)
    """
    source = "MineThatData E-Mail Analytics and Data Mining Challenge (Hillstrom 2008)"
    LICENCE = "MineThatData e-mail challenge (Hillstrom 2008); no licence text at the source (public challenge data)"
    CATEGORICAL = ["history_segment", "zip_code", "channel"]
    NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
    OUTCOMES = ("visit", "conversion")
    SEGMENT = "channel"
    FILE = "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    ARMS = {"mens": "Mens E-Mail", "womens": "Womens E-Mail"}

    def _read(self, root: str) -> pd.DataFrame:
        path = os.path.join(root, "ham", self.FILE)
        d = pd.read_csv(need(path if os.path.exists(path) else os.path.join(root, self.FILE)))
        arm = self.options.get("arm", "any")
        if arm == "any":
            pass
        elif arm in self.ARMS:
            d = d[d.segment.isin([self.ARMS[arm], "No E-Mail"])]
        else:
            raise ValueError("arm must be 'any', 'mens' or 'womens'")
        return d.assign(t=(d.segment != "No E-Mail").astype(int)).reset_index(drop=True)


@register()
class CriteoUplift(UpliftTask):
    """Options: outcome='visit' (default) | 'conversion'; n_rows=None (default, all 13,979,592) | int (random
    subsample of the file before the split, default_rng(seed)); test_frac, seed, baselines."""

    name = "criteo_uplift"
    data_subdir = "criteo-uplift"
    accepted_options = {"outcome": '"visit" | "conversion" (default "visit")',
                        "n_rows": "int, subsample before the split (default: all rows)", **_COMMON_OPTIONS}
    layout = """
    <root>/criteo-uplift/islenmis/criteo.parquet            f0..f11, treatment, conversion, visit, exposure, or
    <root>/criteo-uplift/ham/criteo-research-uplift-v2.1.csv.gz   (Criteo Uplift v2.1, 13,979,592 rows)
    """
    source = "Criteo Uplift Prediction Dataset v2.1 (Diemert, Betlei, Renaudin, Amini, Gregoir, Rahier 2021)"
    LICENCE = "Criteo Uplift v2.1 (Diemert et al. 2021); CC BY-NC-SA 4.0 (non-commercial)"
    NUMERIC = [f"f{i}" for i in range(12)]
    OUTCOMES = ("visit", "conversion")
    COLS = NUMERIC + ["treatment", "visit", "conversion"]

    def _read(self, root: str) -> pd.DataFrame:
        pq = os.path.join(root, "islenmis", "criteo.parquet")
        csv = os.path.join(root, "ham", "criteo-research-uplift-v2.1.csv.gz")
        if os.path.exists(pq):
            d = pd.read_parquet(pq, columns=self.COLS)
        elif os.path.exists(csv):
            d = pd.read_csv(csv, usecols=self.COLS)
        else:
            raise DataNotFound(f"neither {pq} nor {csv} found")
        n = self.options.get("n_rows")
        if n is not None and int(n) < len(d):
            keep = np.sort(np.random.default_rng(int(self.options.get("seed", 0))).choice(len(d), int(n), replace=False))
            d = d.iloc[keep]
        return d.rename(columns={"treatment": "t"}).reset_index(drop=True)


@register()
class X5Uplift(UpliftTask):
    """Options: test_frac, seed, baselines. Outcome is `target` (purchase after the SMS campaign)."""

    name = "x5_uplift"
    data_subdir = "x5-uplift"
    accepted_options = dict(_COMMON_OPTIONS)
    layout = """
    <root>/x5-uplift/ham/uplift_train.csv.gz   client_id, treatment_flg, target (200,039 rows)
    <root>/x5-uplift/ham/clients.csv.gz        client_id, first_issue_date, first_redeem_date, age, gender
        (X5 RetailHero uplift data, scikit-uplift `fetch_x5` mirror; purchases.csv.gz is not used)
    """
    source = "X5 RetailHero Uplift Modeling (ODS.ai 2019-2020), via the scikit-uplift fetch_x5 mirror"
    LICENCE = "X5 RetailHero (ODS.ai competition data); no licence stated, research / evaluation use only"
    CATEGORICAL = ["gender"]
    NUMERIC = ["age", "issue_day", "redeem_day", "redeem_delay"]
    OUTCOMES = ("target",)
    SEGMENT = "gender"
    # first_redeem_date runs to 2019-11 and spikes on 2019-03-21/22 (the SMS); the purchase history ends 2019-03-18.
    # A later first redemption is post-treatment (more treated clients have one), so it is masked as "none yet".
    CUTOFF = pd.Timestamp("2019-03-19")

    def _read(self, root: str) -> pd.DataFrame:
        base = os.path.join(root, "ham") if os.path.isdir(os.path.join(root, "ham")) else root
        u = pd.read_csv(need(os.path.join(base, "uplift_train.csv.gz")))
        c = pd.read_csv(need(os.path.join(base, "clients.csv.gz")))
        d = u.merge(c, on="client_id", how="left", validate="one_to_one")
        ref = pd.Timestamp("2017-01-01")
        issue = pd.to_datetime(d.first_issue_date, errors="coerce")
        redeem = pd.to_datetime(d.first_redeem_date, errors="coerce")
        redeem = redeem.where(redeem < self.CUTOFF)
        age = d.age.astype(float)
        return pd.DataFrame({
            "t": d.treatment_flg.astype(int), "target": d.target.astype(int),
            "age": age.where((age >= 10) & (age <= 100)),  # the file has ages < 0 and > 1000
            "gender": d.gender.fillna("U").astype(str),
            "issue_day": (issue - ref).dt.total_seconds() / 86400, "redeem_day": (redeem - ref).dt.total_seconds() / 86400,
            "redeem_delay": (redeem - issue).dt.total_seconds() / 86400})


__all__ = ["CriteoUplift", "HillstromUplift", "UpliftTask", "X5Uplift", "auuc", "class_transformation",
           "qini_coefficient", "qini_curve", "s_learner", "stratified_split", "two_model", "uplift_at_k",
           "uplift_curve"]

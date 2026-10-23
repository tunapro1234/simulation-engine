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

import numpy as np
import pandas as pd

from ...methods.uplift import (GBDT_PARAMS, TOP_K, _auuc_w, _norm_area, _perfect_qini_score,  # noqa: F401
                               _perfect_uplift_scores, _thresholds, _uak, auuc, class_transformation,
                               qini_coefficient, qini_curve, s_learner, two_model, uplift_at_k, uplift_curve)
from ...types import TaskData
from ..base import BenchTask, DataNotFound, Metric, need, register
from ..stats import ci


# ---------------------------------------------------------------------- bench metrics (curves, metrics and
# learners live in simengine.methods.uplift; these wrap them in the row bootstrap)
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

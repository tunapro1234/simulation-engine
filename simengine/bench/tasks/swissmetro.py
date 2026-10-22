"""swissmetro: stated mode choice between train, Swissmetro and car (Bierlaire et al. 2001; the Biogeme example data).

    from simengine.bench.tasks import swissmetro as S
    task = S.SwissmetroModeChoice()                 # bench name "swissmetro"; 20% of respondents held out
    data = task.load("/path/to/catalogue")          # or $SIMENGINE_DATA
    frame = task.run(data=data)                     # uniform, shares, mnl, mnl_rich, nested, gbdt
    task.score_all(frame)                           # loglik (primary), accuracy, rho2, share_tvd
    fit = S.fit_mnl(S.prepare(raw))                 # Biogeme b01 logit: coef, se, robust_se, loglik
    S.persona_text(data.test.iloc[0]); S.situation_text(data.test.iloc[0])
    task.run(methods={"llm": S.llm_method(llm)}, data=data)   # one prompt per test choice, parsed to probabilities

1,192 respondents (441 rail travellers interviewed on the train in Mar 1998, 751 car drivers reached by licence plates
recorded in Sept 1997) each answered nine stated choice situations: train, Swissmetro (a planned mag-lev) and, for car owners,
car, described by door-to-door time, cost, headway and seat layout. One unit is one answered situation; `truth` is
the chosen alternative coded 0 = train, 1 = Swissmetro, 2 = car (the dataset's CHOICE - 1). A prediction is one
probability vector over (train, sm, car) per test row (array (n, 3)); the metrics zero unavailable alternatives,
renormalise and clip the chosen alternative's probability at 1e-6 for the log-likelihood.

sample='biogeme' (default) applies the Biogeme example filter: commuter or business trips (PURPOSE 1 or 3) with a
known choice, 6,768 choices of 752 respondents. sample='all' keeps every known choice (10,719 of 1,191).
split='respondent' (default) holds out `test_frac` of respondents (all nine situations; default_rng(seed));
split='situation' holds out each respondent's last `n_test` situations in file order (default 2; positions follow a
fixed design order, not a known presentation order), so a respondent's other answers are available (own_shares).

Baselines. uniform (equal over available); shares (train market shares over available); mnl: the Biogeme b01 logit
(ASC train, ASC car, generic time and cost / 100, train and Swissmetro cost 0 for GA holders; Swissmetro ASC = 0)
fitted by maximum likelihood on the train rows; mnl_rich: mnl plus alternative-specific time, headway, seats and
respondent covariates (GA, car-user survey group, first class, employer pays, luggage, age, income, gender, business
trip); nested: the Biogeme b09 nested logit (train and car in one nest, nest parameter mu >= 1) on the b01 utilities;
gbdt: LightGBM multiclass on attributes + respondent covariates (fixed parameters); own_shares (split='situation').
Metrics: mean log-likelihood per choice (primary), accuracy (argmax, ties split), rho2 = 1 - LL / LL(equal shares
over available) and share_tvd (aggregate market-share error); cluster bootstrap over respondents.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from ...types import TaskData
from ..base import BenchTask, Metric, as_pred_col, need, register
from ..stats import cluster_bootstrap

ALTS = ("train", "sm", "car")
ALT_LABEL = {"train": "Train", "sm": "Swissmetro", "car": "Car"}
EPS = 1e-6

PURPOSE = {1: "commuting", 2: "shopping", 3: "business", 4: "leisure", 5: "return from work",
           6: "return from shopping", 7: "return from business", 8: "return from leisure", 9: "other"}
TICKET = {0: "no ticket", 1: "a two-way ticket with a half-price card", 2: "a one-way ticket with a half-price card",
          3: "a two-way full-price ticket", 4: "a one-way full-price ticket", 5: "a half-day ticket",
          6: "an annual season ticket", 7: "a junior or senior annual season ticket",
          8: "a free-travel-after-7pm card", 9: "a group ticket", 10: "another ticket"}
WHO = {0: None, 1: "you pay for the trip yourself", 2: "your employer pays for the trip",
       3: "you and your employer split the cost"}
LUGGAGE = {0: "no luggage", 1: "one piece of luggage", 3: "several pieces of luggage"}
AGE = {1: "24 or younger", 2: "25-39", 3: "40-54", 4: "55-65", 5: "over 65", 6: None}
INCOME = {0: "under 50,000 CHF", 1: "under 50,000 CHF", 2: "50,000-100,000 CHF", 3: "over 100,000 CHF", 4: None}
CANTON = dict(enumerate("ZH BE LU UR SZ OW NW GL ZG FR SO BS BL SH AR AI SG GR AG TG TI VD VS NE GE JU".split(), 1))

# (parameter, alternative, column or None = constant). A parameter on several alternatives is generic.
Term = tuple[str, str, "str | None"]
B01: tuple[Term, ...] = (
    ("asc_train", "train", None), ("asc_car", "car", None),
    ("b_time", "train", "TRAIN_TT_SCALED"), ("b_time", "sm", "SM_TT_SCALED"), ("b_time", "car", "CAR_TT_SCALED"),
    ("b_cost", "train", "TRAIN_COST_SCALED"), ("b_cost", "sm", "SM_COST_SCALED"), ("b_cost", "car", "CAR_CO_SCALED"))
RICH: tuple[Term, ...] = (
    ("asc_train", "train", None), ("asc_car", "car", None),
    ("b_time_train", "train", "TRAIN_TT_SCALED"), ("b_time_sm", "sm", "SM_TT_SCALED"),
    ("b_time_car", "car", "CAR_TT_SCALED"),
    ("b_cost", "train", "TRAIN_COST_SCALED"), ("b_cost", "sm", "SM_COST_SCALED"), ("b_cost", "car", "CAR_CO_SCALED"),
    ("b_headway", "train", "TRAIN_HE_SCALED"), ("b_headway", "sm", "SM_HE_SCALED"), ("b_seats", "sm", "SM_SEATS"),
    ("b_ga_train", "train", "ga"), ("b_ga_sm", "sm", "ga"),
    ("b_caruser_car", "car", "car_user"), ("b_caruser_train", "train", "car_user"),
    ("b_first_car", "car", "first"), ("b_employer_sm", "sm", "employer_pays"),
    ("b_luggage_car", "car", "luggage"), ("b_old_train", "train", "age_55plus"),
    ("b_highinc_sm", "sm", "income_high"), ("b_male_car", "car", "male"), ("b_business_sm", "sm", "business"))
NESTS_B09 = {"existing": ("train", "car"), "sm": ("sm",)}

GBDT_FEATURES = ["PURPOSE", "FIRST", "TICKET", "WHO", "LUGGAGE", "AGE", "MALE", "INCOME", "GA", "ORIGIN", "DEST",
                 "SURVEY", "TRAIN_TT", "TRAIN_COST", "TRAIN_HE", "SM_TT", "SM_COST", "SM_HE", "SM_SEATS", "CAR_TT",
                 "CAR_CO", "av_car"]


# ---------------------------------------------------------------------- data
def read_raw(root: str) -> pd.DataFrame:
    """swissmetro.dat (tab separated, 28 columns) from root or root/ham."""
    for p in (os.path.join(root, "swissmetro.dat"), os.path.join(root, "ham", "swissmetro.dat")):
        if os.path.exists(p):
            return pd.read_csv(p, sep="\t")
    return pd.read_csv(need(os.path.join(root, "swissmetro.dat")), sep="\t")


def prepare(raw: pd.DataFrame, sample: str = "biogeme") -> pd.DataFrame:
    """Filter and add the Biogeme derived variables (swissmetro_data.py): GA holders pay 0 for train and Swissmetro,
    times / costs / headways divided by 100, availabilities x (SP != 0). Adds y (0 train, 1 sm, 2 car), av_<alt>,
    respondent (= ID), situation (0-8 within respondent) and 0/1 respondent covariates."""
    d = raw.copy()
    d["situation"] = d.groupby("ID").cumcount()
    keep = d["CHOICE"] != 0
    if sample == "biogeme":
        keep &= d["PURPOSE"].isin([1, 3])
    elif sample != "all":
        raise ValueError("sample must be 'biogeme' or 'all'")
    d = d[keep].reset_index(drop=True)
    sp = (d["SP"] != 0).astype(float)
    nga = (d["GA"] == 0).astype(float)
    d["TRAIN_COST"], d["SM_COST"] = d["TRAIN_CO"] * nga, d["SM_CO"] * nga
    for c in ("TRAIN_TT", "TRAIN_COST", "TRAIN_HE", "SM_TT", "SM_COST", "SM_HE", "CAR_TT", "CAR_CO"):
        d[c + "_SCALED"] = d[c] / 100.0
    d["av_train"], d["av_sm"], d["av_car"] = d["TRAIN_AV"] * sp, d["SM_AV"].astype(float), d["CAR_AV"] * sp
    d["y"] = (d["CHOICE"] - 1).astype(int)
    d["respondent"] = d["ID"].astype(int)
    d["ga"] = d["GA"].astype(float)
    d["car_user"] = (d["SURVEY"] == 1).astype(float)
    d["first"] = d["FIRST"].astype(float)
    d["male"] = d["MALE"].astype(float)
    d["employer_pays"] = (d["WHO"] == 2).astype(float)
    d["luggage"] = (d["LUGGAGE"] > 0).astype(float)
    d["age_55plus"] = d["AGE"].isin([4, 5]).astype(float)
    d["income_high"] = (d["INCOME"] == 3).astype(float)
    d["business"] = d["PURPOSE"].isin([3, 7]).astype(float)
    return d


def availability(df: pd.DataFrame) -> np.ndarray:
    return df[[f"av_{a}" for a in ALTS]].to_numpy(float) > 0


def respondent_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0) -> np.ndarray:
    """Boolean test mask: round(test_frac * n_respondents) respondents drawn by default_rng(seed) from the sorted ids."""
    ids = np.unique(df["respondent"].to_numpy())
    k = int(round(test_frac * len(ids)))
    if not 0 < k < len(ids):
        raise ValueError(f"test_frac={test_frac} leaves {k} of {len(ids)} respondents in test; need 1 .. n-1")
    test_ids = np.random.default_rng(seed).choice(ids, k, replace=False)
    return df["respondent"].isin(test_ids).to_numpy()


def situation_split(df: pd.DataFrame, n_test: int = 2) -> np.ndarray:
    """Boolean test mask: each respondent's last n_test answered situations (by situation index = row order in the
    file; the file has no presentation-order column, and positions follow a fixed design order)."""
    if n_test < 1:
        raise ValueError("n_test must be >= 1")
    rank = df.groupby("respondent")["situation"].rank(ascending=False, method="first")
    te = (rank <= n_test).to_numpy()
    if te.all():
        raise ValueError(f"n_test={n_test} leaves no training situations")
    return te


# ---------------------------------------------------------------------- logit models
def design(df: pd.DataFrame, terms: Sequence[Term]) -> tuple[np.ndarray, list[str]]:
    """X (n, J, K) for V = X @ beta, parameters in first-appearance order."""
    names = list(dict.fromkeys(t[0] for t in terms))
    X = np.zeros((len(df), len(ALTS), len(names)))
    for p, alt, col in terms:
        X[:, ALTS.index(alt), names.index(p)] += 1.0 if col is None else df[col].to_numpy(float)
    return X, names


@dataclass
class LogitFit:
    """Estimated logit. coef / se / robust_se: pd.Series by parameter; loglik, null_loglik (equal shares over
    available), n; nests: {} for MNL, else {nest: alternatives}. predict_proba(df) -> (n, 3)."""
    terms: tuple[Term, ...]
    coef: pd.Series
    se: pd.Series
    robust_se: pd.Series
    loglik: float
    null_loglik: float
    n: int
    converged: bool
    nests: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def rho2(self) -> float:
        return 1 - self.loglik / self.null_loglik

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = design(df, self.terms)
        beta = self.coef.to_numpy()
        return np.exp(_log_probs(X, availability(df), beta, self.nests, len(self.coef) - len(_nest_params(self.nests))))

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"value": self.coef, "se": self.se, "robust_se": self.robust_se,
                             "robust_t": self.coef / self.robust_se})


def _nest_params(nests: Mapping[str, Sequence[str]]) -> list[str]:
    return [f"mu_{k}" for k, alts in nests.items() if len(alts) > 1]


def _log_probs(X: np.ndarray, av: np.ndarray, theta: np.ndarray, nests: Mapping[str, Sequence[str]],
               k_beta: int) -> np.ndarray:
    """(n, J) log choice probabilities (-inf where unavailable). MNL when nests is empty; else nested logit with the
    Ben-Akiva / Biogeme normalisation: P(i) = P(i | m) P(m), P(i | m) ~ exp(mu_m V_i), P(m) ~ exp(I_m),
    I_m = log sum_{j in m} exp(mu_m V_j) / mu_m (upper level scale 1)."""
    V = np.where(av, X @ theta[:k_beta], -np.inf)
    if not nests:
        return V - logsumexp(V, axis=1, keepdims=True)
    mus = iter(theta[k_beta:])
    out = np.empty_like(V)
    inc, members = [], []
    with np.errstate(divide="ignore", invalid="ignore"):
        for alts in nests.values():
            idx = [ALTS.index(a) for a in alts]
            mu = next(mus) if len(idx) > 1 else 1.0
            lse = logsumexp(mu * V[:, idx], axis=1)
            out[:, idx] = mu * V[:, idx] - lse[:, None]           # log P(i | m)
            inc.append(lse / mu)
            members.append(idx)
        I = np.column_stack(inc)
        lpm = I - logsumexp(I, axis=1, keepdims=True)
    for k, idx in enumerate(members):
        out[:, idx] += lpm[:, [k]]
    return np.where(av, out, -np.inf)


def _fit(df: pd.DataFrame, terms: Sequence[Term], nests: Mapping[str, Sequence[str]] | None) -> LogitFit:
    terms, nests = tuple(terms), dict(nests or {})
    X, names = design(df, terms)
    av = availability(df)
    y = df["y"].to_numpy(int)
    rows = np.arange(len(y))
    if not av[rows, y].all():
        raise ValueError("a chosen alternative is unavailable")
    K = len(names)
    mu_names = _nest_params(nests)

    def ll_obs(theta):
        return _log_probs(X, av, theta, nests, K)[rows, y]

    if not nests:
        def f(b):
            lp = _log_probs(X, av, b, nests, K)
            p = np.exp(lp)
            xbar = np.einsum("nj,njk->nk", p, X)
            return -lp[rows, y].sum(), -(X[rows, y] - xbar).sum(0)
        r = minimize(f, np.zeros(K), jac=True, method="BFGS", options={"gtol": 1e-7, "maxiter": 2000})
        theta = r.x
        p = np.exp(_log_probs(X, av, theta, nests, K))
        xbar = np.einsum("nj,njk->nk", p, X)
        dx = X - xbar[:, None, :]
        H = np.einsum("nj,njk,njl->kl", p, dx, dx)                 # information matrix (negative Hessian)
        S = X[rows, y] - xbar                                       # per-observation scores
        ok = bool(r.success) or float(np.abs(S.sum(0)).max()) < 1e-4  # BFGS may stop on precision loss at the optimum
    else:
        x0 = np.r_[np.zeros(K), np.ones(len(mu_names))]
        bounds = [(None, None)] * K + [(1.0, None)] * len(mu_names)
        r = minimize(lambda t: -ll_obs(t).sum(), x0, jac="3-point", method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-8})
        theta = r.x
        S = _num_jac(ll_obs, theta)
        H = -_num_hess(lambda t: ll_obs(t).sum(), theta)
        ok = bool(r.success)
    Hi = np.linalg.pinv(H)
    rob = Hi @ (S.T @ S) @ Hi
    idx = names + mu_names
    null = float(-np.log(av.sum(1)).sum())
    return LogitFit(terms=terms, coef=pd.Series(theta, index=idx), se=pd.Series(np.sqrt(np.diag(Hi)), index=idx),
                    robust_se=pd.Series(np.sqrt(np.diag(rob)), index=idx), loglik=float(ll_obs(theta).sum()),
                    null_loglik=null, n=len(y), converged=ok, nests=nests)


def _num_jac(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    cols = []
    for k in range(len(x)):
        e = np.zeros_like(x)
        e[k] = h
        cols.append((f(x + e) - f(x - e)) / (2 * h))
    return np.column_stack(cols)


def _num_hess(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    K = len(x)
    H = np.empty((K, K))
    for i in range(K):
        for j in range(i, K):
            ei, ej = np.zeros(K), np.zeros(K)
            ei[i], ej[j] = h, h
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def fit_mnl(df: pd.DataFrame, terms: Sequence[Term] = B01) -> LogitFit:
    """Maximum-likelihood multinomial logit (BFGS, analytic gradient; classic and robust sandwich standard errors)
    on a `prepare`d frame. Default terms = Biogeme b01."""
    return _fit(df, terms, None)


def fit_nested(df: pd.DataFrame, terms: Sequence[Term] = B01,
               nests: Mapping[str, Sequence[str]] = NESTS_B09) -> LogitFit:
    """Nested logit, every alternative in exactly one nest, mu_m >= 1 for nests with two or more alternatives
    (L-BFGS-B, numerical gradient; standard errors from a numerical Hessian). Default = Biogeme b09 (which also caps
    the nest parameter at 3; no upper bound here, the b09 optimum 2.05 is interior)."""
    seen = sorted(a for alts in nests.values() for a in alts)
    if seen != sorted(ALTS):
        raise ValueError(f"nests must partition {ALTS}, got {dict(nests)}")
    return _fit(df, terms, nests)


def mask_probs(p: Any, av: np.ndarray) -> np.ndarray:
    """(n, 3) probabilities with unavailable alternatives zeroed and rows renormalised (uniform over available when a
    row has no mass left)."""
    p = np.clip(np.asarray(p, float).reshape(len(av), len(ALTS)), 0, None) * av
    s = p.sum(1, keepdims=True)
    return np.where(s > 0, p / np.where(s > 0, s, 1), av / av.sum(1, keepdims=True))


def gbdt_probs(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """LightGBM multiclass (300 trees, lr 0.05, 15 leaves, min_child_samples 50, subsample 0.8 / colsample 0.9);
    sklearn HistGradientBoosting if LightGBM is missing. Unavailable alternatives masked afterwards."""
    Xtr, Xte = train[GBDT_FEATURES].to_numpy(float), test[GBDT_FEATURES].to_numpy(float)
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.05, num_leaves=15,
                               min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                               random_state=seed, verbose=-1, n_jobs=4)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)
    m.fit(Xtr, train["y"].to_numpy(int))
    p = np.zeros((len(test), len(ALTS)))
    p[:, m.classes_.astype(int)] = m.predict_proba(Xte)
    return mask_probs(p, availability(test))


# ---------------------------------------------------------------------- persona / prompt text
def persona_text(row: Mapping) -> str:
    """Second-person profile from the respondent attributes (constant over a respondent's nine situations)."""
    parts = []
    sex = "man" if int(row["MALE"]) == 1 else "woman"
    age = AGE.get(int(row["AGE"]))
    parts.append(f"You are a {sex}" + (f" aged {age}" if age else "") + " living in Switzerland.")
    inc = INCOME.get(int(row["INCOME"]))
    if inc:
        parts.append(f"Your yearly income is {inc}.")
    o, de = CANTON.get(int(row["ORIGIN"]), "?"), CANTON.get(int(row["DEST"]), "?")
    trip = PURPOSE.get(int(row["PURPOSE"]), "other")
    if int(row["SURVEY"]) == 1:
        parts.append(f"You regularly make this trip by car (from canton {o} to canton {de}); the purpose is {trip}.")
    else:
        parts.append(f"You are making this trip by train (from canton {o} to canton {de}); the purpose is {trip}.")
        parts.append(f"You travel in {'first' if int(row['FIRST']) == 1 else 'second'} class with "
                     f"{TICKET.get(int(row['TICKET']), 'another ticket')}.")
    who = WHO.get(int(row["WHO"]))
    if who:
        parts.append(who[0].upper() + who[1:] + ".")
    parts.append(f"You carry {LUGGAGE.get(int(row['LUGGAGE']), 'some luggage')}.")
    parts.append("You own a GA (Swiss annual travel card), so train trips cost you nothing extra." if int(row["GA"]) == 1
                 else "You do not own a GA annual travel card.")
    if float(row.get("av_car", row.get("CAR_AV", 0))) == 0:
        parts.append("You do not have a car available for this trip.")
    return " ".join(parts)


def situation_text(row: Mapping) -> str:
    """The available alternatives of one stated choice situation, with time, cost and headway. Cost follows the
    Biogeme coding (train and Swissmetro free with a GA); the codebook says SM_CO ignores the GA and TRAIN_CO holds the
    annual ticket price for GA holders, so what GA holders were actually shown is not recorded."""
    ga = int(row["GA"]) == 1
    lines = []
    if float(row["av_train"]) > 0:
        cost = "covered by your GA" if ga else f"{row['TRAIN_CO']:.0f} CHF"
        lines.append(f"Train: {row['TRAIN_TT']:.0f} minutes door to door, {cost}, one train every "
                     f"{row['TRAIN_HE']:.0f} minutes.")
    if float(row["av_sm"]) > 0:
        cost = "covered by your GA" if ga else f"{row['SM_CO']:.0f} CHF"
        seats = "airline-style seats" if int(row["SM_SEATS"]) == 1 else "regular seats"
        lines.append(f"Swissmetro (new underground mag-lev at up to 500 km/h): {row['SM_TT']:.0f} minutes door to "
                     f"door, {cost}, one every {row['SM_HE']:.0f} minutes, {seats}.")
    if float(row["av_car"]) > 0:
        lines.append(f"Car: {row['CAR_TT']:.0f} minutes, {row['CAR_CO']:.0f} CHF.")
    return "\n".join(lines)


SYSTEM = ("You answer a travel survey as the person described. Reply with JSON only: the probability that you would "
          "choose each available option, e.g. {\"train\": 0.2, \"sm\": 0.5, \"car\": 0.3}.")


def prompts(df: pd.DataFrame, system: str = SYSTEM) -> list[tuple[str, str]]:
    """(system, user) per row: persona, then the situation, then the question."""
    out = []
    for _, r in df.iterrows():
        keys = ", ".join(f'"{a}"' for a in ALTS if float(r[f"av_{a}"]) > 0)
        out.append((system, f"{persona_text(r)}\n\nFor this trip you can choose between:\n{situation_text(r)}\n\n"
                            f"Which would you choose? Give probabilities for {keys}."))
    return out


_WORD = {"train": 0, "rail": 0, "swissmetro": 1, "sm": 1, "metro": 1, "car": 2}


_PAIR = re.compile(r"\b(swissmetro|train|rail|metro|car|sm)\b\"?\s*[:=]\s*\"?\s*(\d+(?:\.\d*)?|\.\d+)\s*(%?)", re.I)


def _num(x: Any) -> float | None:
    """Finite non-negative number from a JSON value (0.2, "0.2", "20%"); None otherwise."""
    try:
        s = str(x).strip()
        v = float(s[:-1]) if s.endswith("%") else float(s)
    except (ValueError, TypeError):
        return None
    return v if np.isfinite(v) and v >= 0 else None


def parse_choice(text: str) -> np.ndarray | None:
    """Probability vector (train, sm, car) from a reply: a JSON object (keys train / sm / swissmetro / car; values may
    be "20%" strings), else `key: number` pairs, else the first alternative word. A reply that has an object with
    alternative keys but no usable positive value is None (not read as its first key); None if nothing is recognised.
    Not renormalised (mask_probs does that)."""
    text = text or ""
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            v = np.zeros(len(ALTS))
            for k, x in obj.items():
                j, f = _WORD.get(str(k).strip().lower()), _num(x)
                if j is not None and f is not None:
                    v[j] += f
            if v.sum() > 0:
                return v
    pairs = _PAIR.findall(text)
    if pairs:
        v = np.zeros(len(ALTS))
        for k, x, _ in pairs:
            v[_WORD[k.lower()]] += float(x)
        return v if v.sum() > 0 else None
    if m and re.search(r"\b(swissmetro|train|rail|car|sm)\b", m.group(0).lower()):
        return None                                                  # structured answer without usable numbers
    w = re.search(r"\b(swissmetro|train|rail|car|sm)\b", text.lower())
    if w:
        v = np.zeros(len(ALTS))
        v[_WORD[w.group(1)]] = 1.0
        return v
    return None


def llm_method(llm: Any, *, system: str = SYSTEM, smooth: float = 0.05, max_tokens: int = 64,
               parallel: int = 4) -> Callable[[TaskData], np.ndarray]:
    """Method for task.run: one prompt per data.test row through llm.map; replies parsed by parse_choice, mixed with
    `smooth` of the uniform-over-available vector (a one-hot answer would otherwise score log 1e-6 when wrong),
    unparseable replies and replies with no mass on an available alternative -> uniform. The method's
    `last_unparsed` attribute counts them after each call."""
    def method(data: TaskData) -> np.ndarray:
        t = data.test
        av = availability(t)
        res = llm.map(prompts(t, system), max_tokens=max_tokens, parallel=parallel)
        parsed = [parse_choice(r.text) for r in res]
        parsed = [None if v is None or (v * av[i]).sum() <= 0 else v for i, v in enumerate(parsed)]
        method.last_unparsed = sum(p is None for p in parsed)
        u = av / av.sum(1, keepdims=True)
        p = mask_probs(np.vstack([u[i] if v is None else v for i, v in enumerate(parsed)]), av)
        return (1 - smooth) * p + smooth * u
    method.last_unparsed = 0
    return method


# ---------------------------------------------------------------------- metrics
def _probs(frame: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    av = availability(frame)
    return mask_probs(np.vstack(frame[col].to_numpy()), av), frame["truth"].to_numpy(int), av


def _metric(which: str):
    def fn(task, frame, col, B, seed):
        p, y, av = _probs(frame, col)
        n = len(y)
        rows = np.arange(n)
        lpy = np.log(np.clip(p[rows, y], EPS, 1.0))
        if which == "loglik":
            v = lpy
            f = lambda i: v[i].mean()
        elif which == "accuracy":
            top = np.where(av, p, -1.0)
            best = np.isclose(top, top.max(1, keepdims=True), rtol=0, atol=1e-12)
            v = best[rows, y] / best.sum(1)
            f = lambda i: v[i].mean()
        elif which == "rho2":
            ll, l0 = lpy, -np.log(av.sum(1))
            f = lambda i: 1 - ll[i].sum() / l0[i].sum()
        else:
            Y = np.eye(len(ALTS))[y]
            f = lambda i: 0.5 * np.abs(p[i].mean(0) - Y[i].mean(0)).sum()
        r = cluster_bootstrap(f, frame["group"].to_numpy(), B=B, seed=seed)
        return {**r, "n": n, "n_respondents": r["n"]}
    return fn


# ---------------------------------------------------------------------- task
@register()
class SwissmetroModeChoice(BenchTask):
    """Options: sample='biogeme' | 'all'; split='respondent' | 'situation'; test_frac=0.2 and seed=0 (respondent
    split); n_test=2 (situation split); lgbm=True; nested=True."""

    name = "swissmetro"
    kind = "multiclass"
    data_subdir = "swissmetro"
    accepted_options = {
        "sample": '"biogeme" (PURPOSE 1 or 3, 6,768 choices) | "all" (10,719) (default "biogeme")',
        "split": '"respondent" | "situation" (default "respondent")',
        "test_frac": "float, share of respondents held out (default 0.2)",
        "seed": "int (default 0; respondent split and gbdt)",
        "n_test": "int, last situations per respondent held out with split='situation' (default 2)",
        "lgbm": "true | false (default true)",
        "nested": "true | false (default true)",
    }
    layout = """
    <root>/swissmetro/swissmetro.dat   Biogeme Swissmetro data (tab separated, 10,728 rows x 28 columns;
                                       ham/swissmetro.dat also accepted)
    """
    source = ("Bierlaire, Axhausen, Abay (2001) 'The acceptance of modal innovation: the case of Swissmetro', STRC; "
              "data and model specifications from the Biogeme examples b01a_logit and b09_nested "
              "(transp-or.epfl.ch/data/swissmetro.dat)")
    metrics = (Metric("loglik", True, _metric("loglik"), "mean log-likelihood per choice"),
               Metric("accuracy", True, _metric("accuracy"), "argmax over available alternatives, ties split"),
               Metric("rho2", True, _metric("rho2"), "1 - LL / LL(equal shares over available)"),
               Metric("share_tvd", False, _metric("share_tvd"), "TVD between mean predicted and observed shares"))

    def _load(self, root: str) -> TaskData:
        o = self.options
        sample, split = o.get("sample", "biogeme"), o.get("split", "respondent")
        d = prepare(read_raw(root), sample)
        if split == "respondent":
            frac, seed = float(o.get("test_frac", 0.2)), int(o.get("seed", 0))
            te = respondent_split(d, frac, seed)
            rule = f"respondent: {frac:.0%} of respondents held out, default_rng({seed})"
        elif split == "situation":
            k = int(o.get("n_test", 2))
            te = situation_split(d, k)
            rule = f"situation: each respondent's last {k} answered situations held out"
        else:
            raise ValueError("split must be 'respondent' or 'situation'")
        tr, ts = d[~te].reset_index(drop=True), d[te].reset_index(drop=True)
        meta = {"split": rule, "sample": sample, "n_train": len(tr), "n_test": len(ts),
                "n_train_respondents": int(tr.respondent.nunique()), "n_test_respondents": int(ts.respondent.nunique()),
                "alternatives": list(ALTS), "test_shares": np.bincount(ts.y, minlength=3).tolist(),
                "source": "Swissmetro SP survey (Bierlaire et al. 2001), Biogeme example data; free download from "
                          "transp-or.epfl.ch, no licence text"}
        return TaskData(train=tr, test=ts, meta=meta)

    def baselines(self, data: TaskData) -> dict:
        tr, te = data.train, data.test
        av = availability(te)
        out = {"uniform": mask_probs(np.ones((len(te), 3)), av),
               "shares": mask_probs(np.tile(np.bincount(tr.y, minlength=3) / len(tr), (len(te), 1)), av)}
        fits = {"mnl": fit_mnl(tr), "mnl_rich": fit_mnl(tr, RICH)}
        if self.options.get("nested", True):
            fits["nested"] = fit_nested(tr)
        for k, f in fits.items():
            out[k] = f.predict_proba(te)
        data.meta["fits"] = {k: f.table().round(6).to_dict("index") | {"_loglik_train": f.loglik}
                             for k, f in fits.items()}
        if self.options.get("lgbm", True):
            out["gbdt"] = gbdt_probs(tr, te, int(self.options.get("seed", 0)))
        if self.options.get("split", "respondent") == "situation":
            cnt = np.zeros((len(te), 3))
            c = tr.groupby("respondent")["y"].value_counts().unstack(fill_value=0).reindex(columns=range(3),
                                                                                             fill_value=0)
            cnt[:] = c.reindex(te.respondent).fillna(0).to_numpy(float)
            out["own_shares"] = mask_probs(cnt + 1.0, av)           # Laplace over available
        return out

    def frame(self, data: TaskData, preds: dict[str, Any]) -> pd.DataFrame:
        base = self._frame(data).reset_index(drop=True)
        n = len(base)
        for m, p in preds.items():
            a = np.asarray(p, float)
            if a.shape != (n, len(ALTS)):
                raise ValueError(f"{m}: predictions must have shape ({n}, {len(ALTS)}), got {a.shape}")
            if not np.isfinite(a).all() or (a < 0).any():
                raise ValueError(f"{m}: NaN, infinite or negative probabilities")
            base[as_pred_col(m)] = pd.Series(list(a), dtype=object)
        base["split"] = "test"
        return base

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        return pd.DataFrame({"unit": np.arange(len(t)), "group": t["respondent"].to_numpy(),
                             "item": t["situation"].to_numpy(), "truth": t["y"].to_numpy(int),
                             "segment": np.where(t["car_user"] > 0, "car_user", "rail_user"),
                             "seg_ga": np.where(t["ga"] > 0, "ga", "no_ga"),
                             "seg_purpose": t["PURPOSE"].map(PURPOSE).to_numpy(),
                             "av_train": t["av_train"].to_numpy(), "av_sm": t["av_sm"].to_numpy(),
                             "av_car": t["av_car"].to_numpy()})

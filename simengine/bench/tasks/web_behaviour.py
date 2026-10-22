"""web_behaviour: human web-shopping behaviour on WebShop (MTurk trajectories), two step-level tasks on one split:
which action does the shopper take next, and does the shopper buy the item on the current page now.

    from simengine.bench.tasks.web_behaviour import WebShopNextAction, WebShopBuy, llm_method
    task = WebShopNextAction()                     # registered as "webshop_next_action"
    data = task.load("/path/to/catalogue")         # or $SIMENGINE_DATA; reads <root>/webshop-insan/all_trajs/*.jsonl
    frame = task.run(data=data)                    # uniform, frequency, markov, markov2, logistic, uxagent_markov
    task.score(frame, "markov")                    # mean log-likelihood per action, session bootstrap CI
    buy = WebShopBuy()                             # "webshop_buy": item-page rows, P(next action = buy)
    buy.score(buy.run(data=buy.load()), "logistic")          # ROC AUC
    task.run(methods={"llm": llm_method(se.FakeLLM())}, data=data)   # one prompt per test row

Data: the human demonstrations of WebShop (Yao, Chen, Yang & Narasimhan 2022, arXiv:2207.01206): MTurk workers,
selected by a qualification test, buy an instructed product on a simulated Amazon-like site. A trajectory file holds
one JSON line per page visited (page type, url, goal, content); the last line carries the episode reward. Actions are
not logged; they are derived from consecutive pages (`classify_step`):
  * -> done: buy;  -> item_sub_page: view_detail (description / features / reviews);  -> index: back_to_search
  * index -> search_results, or any page -> results for other keywords (a new query after a browser back, which
    the server does not log): search;  search_results -> search_results: next_page / prev_page by the page number
  * search_results -> item_page, or item_page -> item_page of another product: click_item
  * item_page -> item_page, same product, other options: select_option;  item_page / sub page -> previous page: back
  * anything else (reloads: same page, same content; index -> item_page): other
Rows: every action except each session's first one (`search` from the start page; in 84 of the 1,643 human
sessions a start-page reload, `other`, whose search then is the first row). A row sees the goal, the
pages and actions before it, and the current page; truth = the action taken there.
Split (option split="batch", default): sessions of the collection batches dated before 2022-05-13 train, the
2022-05-13 / 2022-05-14 batches (posted after a scoring fix and a server speed-up) test. split="random" holds out
test_frac of sessions (default_rng(seed)).
UXAgent (Lu et al. 2025, arXiv:2502.12561; optional, <root>/uxagent): 60 shopping sessions run by LLM agents with
generated personas on the authors' own shop site, mapped onto the same action names (`map_uxagent_action`). They are
never evaluated; they feed `uxagent_markov`, a chain learnt from LLM agents and scored on humans.
"""
from __future__ import annotations

import ast
import glob
import json
import os
import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, Metric, as_pred_col, need, register
from ..stats import auc, cluster_bootstrap

ACTIONS = ("search", "click_item", "select_option", "view_detail", "next_page", "prev_page", "back",
           "back_to_search", "buy", "other")
A = {a: i for i, a in enumerate(ACTIONS)}
K = len(ACTIONS)
PAGES = ("index", "search_results", "item_page", "item_sub_page", "done")
TEST_FROM = "20220513"          # batch date prefix; batches from this date on are the test set
EPS = 1e-6
PRIOR2 = 10.0                   # pseudo-counts of the first-order row in the second-order chain
NUM_FEATURES = ["log_step", "n_search", "n_items", "n_details", "n_opts", "n_goal_opts", "opts_done",
                "log_results_page"]
LAYOUT = """
    <root>/webshop-insan/all_trajs/<batch>_<n>.jsonl   one WebShop human trajectory per file (1,643 files): one JSON
                                                     line per page (page, url, goal, content); last line reward
    <root>/uxagent/data/train-00000-of-00001.parquet   optional: UXAgent LLM-agent sessions (60 rows; action_trace)
    """
SOURCE = ("Yao, Chen, Yang, Narasimhan (2022) 'WebShop: Towards Scalable Real-World Web Interaction with Grounded "
          "Language Agents', arXiv:2207.01206, human trajectories (github princeton-nlp/WebShop, MIT); Lu et al. (2025) "
          "'UXAgent', arXiv:2502.12561, HF NEU-HAI/UXAgent (MIT)")
LICENCE = "WebShop human trajectories: MIT (princeton-nlp/WebShop); UXAgent traces: MIT (HF dataset card)"
OPTIONS = {"split": '"batch" (default) | "random"',
           "test_frac": "float, share of sessions held out with split='random' (default 0.3)",
           "seed": "int (default 0; random split)",
           "uxagent": "true | false (default true; skipped if the parquet is absent)"}


# ---------------------------------------------------------------------- parsing
def _opts(c: dict | None) -> dict:
    o = (c or {}).get("options") or {}
    return o if isinstance(o, dict) else {}


def _pageno(c: dict | None) -> int:
    try:
        return int((c or {}).get("page", 0))
    except (TypeError, ValueError):
        return 0


def _kw(c: dict | None) -> tuple | None:
    """Search keywords of a page: a list on results pages, its repr string on item pages; None if absent."""
    k = (c or {}).get("keywords")
    if isinstance(k, str):
        try:
            k = ast.literal_eval(k)
        except (ValueError, SyntaxError):
            return (k,)
    return tuple(str(x) for x in k) if isinstance(k, (list, tuple)) else None


def classify_step(pa: str, pb: str, ca: dict | None = None, cb: dict | None = None) -> str:
    """Action name for the move from page `pa` (content ca) to page `pb` (content cb); rules in the module docstring."""
    if pb == "done":
        return "buy"
    if pb == "item_sub_page":
        return "view_detail"
    if pb == "index":
        return "other" if pa == "index" else "back_to_search"
    if pb == "search_results":
        if pa == "index":
            return "search"
        ka, kb = _kw(ca), _kw(cb)
        if ka is not None and kb is not None and ka != kb:
            return "search"                   # new query typed after a browser back to the (unlogged) start page
        if pa == "search_results":
            d = _pageno(cb) - _pageno(ca)
            return "next_page" if d > 0 else "prev_page" if d < 0 else "other"
        return "back" if pa in ("item_page", "item_sub_page") else "other"
    if pb == "item_page":
        if pa == "search_results":
            return "click_item"
        if pa == "item_page":
            if (ca or {}).get("asin") != (cb or {}).get("asin"):
                return "click_item"
            return "select_option" if _opts(ca) != _opts(cb) else "other"
        return "back" if pa == "item_sub_page" else "other"
    return "other"


def parse_trajectory(lines: list[dict], session: str = "") -> tuple[pd.DataFrame, dict]:
    """(steps, summary) for one trajectory (list of page records). steps: one row per action t with the state it was
    taken in: session, t, page, action, y, prev, prev2, history (earlier action names), n_search, n_items, n_details,
    n_opts, n_goal_opts, opts_done, results_page, step features. summary: session, n_actions, bought, reward, price,
    instruction, n_goal_opts, price_upper."""
    pages = [str(x.get("page", "")) for x in lines]
    cont = [x.get("content") if isinstance(x.get("content"), dict) else {} for x in lines]
    goal = next((x["goal"] for x in lines if isinstance(x.get("goal"), dict)), {})
    n_goal = len(goal.get("goal_options") or [])
    acts = [classify_step(pages[i], pages[i + 1], cont[i], cont[i + 1]) for i in range(len(lines) - 1)]
    rows, seen, n_search, n_det, rp = [], set(), 0, 0, 0
    for t, a in enumerate(acts):
        p, c = pages[t], cont[t]
        if p in ("item_page", "item_sub_page") and c.get("asin"):
            seen.add(c["asin"])
        if p == "search_results":
            rp = _pageno(c)
        n_opts = len(_opts(c)) if p in ("item_page", "item_sub_page") else 0
        rows.append({"session": session, "t": t, "page": p, "action": a, "y": A[a],
                     "prev": acts[t - 1] if t >= 1 else "none", "prev2": acts[t - 2] if t >= 2 else "none",
                     "history": " ".join(acts[:t]), "n_search": n_search, "n_items": len(seen), "n_details": n_det,
                     "n_opts": n_opts, "n_goal_opts": n_goal, "opts_done": int(n_opts >= n_goal),
                     "results_page": rp})
        n_search += a == "search"
        n_det += a == "view_detail"
    last = lines[-1] if lines else {}
    try:
        reward = float(last.get("reward"))
    except (TypeError, ValueError):
        reward = float("nan")
    summ = {"session": session, "n_actions": len(acts), "bought": int(bool(acts) and acts[-1] == "buy"),
            "reward": reward, "price": (cont[-1] or {}).get("price", np.nan) if pages and pages[-1] == "done" else np.nan,
            "instruction": str(goal.get("instruction_text", "")), "n_goal_opts": n_goal,
            "price_upper": goal.get("price_upper", np.nan)}
    steps = pd.DataFrame(rows, columns=["session", "t", "page", "action", "y", "prev", "prev2", "history", "n_search",
                                        "n_items", "n_details", "n_opts", "n_goal_opts", "opts_done", "results_page"])
    return steps, summ


def read_webshop(root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(steps, sessions) for every all_trajs/*.jsonl under root, files in sorted order. sessions adds batch (file
    stem without the trailing number) and date (its first 8 characters)."""
    d = need(os.path.join(root, "all_trajs"))
    files = sorted(glob.glob(os.path.join(d, "*.jsonl")))
    if not files:
        raise ValueError(f"web_behaviour: no *.jsonl trajectories in {d}")
    steps, summ = [], []
    for f in files:
        with open(f) as fh:
            lines = [json.loads(x) for x in fh if x.strip()]
        sid = os.path.basename(f)[:-len(".jsonl")]
        st, sm = parse_trajectory(lines, sid)
        steps.append(st)
        summ.append(sm)
    s = pd.DataFrame(summ)
    s["batch"] = s["session"].str.replace(r"_\d+$", "", regex=True)
    s["date"] = s["session"].str[:8]
    return pd.concat(steps, ignore_index=True), s


# ---------------------------------------------------------------------- UXAgent
_UX_OTHER = ("filter", "sorter", "cart", "proceed_to_checkout", "navigate", "terminate")


def map_uxagent_action(a: dict) -> str | None:
    """UXAgent action dict (type, name) -> action name, None for `terminate` (end of session). add_to_cart counts as
    buy (the agents' purchase); cart / checkout (also typing into its fields), filters, sorting and URL navigation
    are `other`."""
    typ, name = str(a.get("type") or ""), str(a.get("name") or "").lower()
    if typ == "terminate":
        return None
    if typ == "back" or name == "back":
        return "back"
    if "search_box" in name or name.endswith(("search_button", "search_input")):
        return "search"
    if typ in ("type_and_submit", "input_text"):
        return "other"                            # typing into checkout / quantity fields
    if name.endswith("view_product") or name.startswith(("related.", "product_link")):
        return "click_item"                       # before the substring tests: product names are free text
    if "add_to_cart" in name:
        return "buy"
    if name.startswith("pager") or "next_page" in name:
        return "next_page" if "prev" not in name else "prev_page"
    if typ == "navigate" or name.startswith(_UX_OTHER) or "checkout" in name or "cart" in name:
        return "other"
    if name.startswith(("product_info", "view_reviews")) or typ == "read":
        return "view_detail"
    if name.startswith(("product_form", "size_", "color_")):
        return "select_option"
    return "other"


def uxagent_sequences(traces) -> list[list[int]]:
    """action_trace column (per session: steps, each a list of action dicts) -> action index lists; consecutive
    `search` inside one step (type, then press the search button) count once."""
    out = []
    for tr in traces:
        seq: list[int] = []
        for step in tr if tr is not None else []:
            prev_in_step = None
            for a in step:
                m = map_uxagent_action(a)
                if m is None or (m == "search" and prev_in_step == "search"):
                    continue
                seq.append(A[m])
                prev_in_step = m
        out.append(seq)
    return out


def read_uxagent(root: str) -> list[list[int]] | None:
    """UXAgent sessions as action index lists, or None if the parquet is not next to / under root."""
    for d in (os.path.join(os.path.dirname(os.path.abspath(root)), "uxagent"), os.path.join(root, "uxagent")):
        f = os.path.join(d, "data", "train-00000-of-00001.parquet")
        if os.path.exists(f):
            return uxagent_sequences(pd.read_parquet(f, columns=["action_trace"])["action_trace"])
    return None


# ---------------------------------------------------------------------- chains and summaries
def transition_counts(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """(K + 1, K) counts; row K is 'none' (no earlier action)."""
    c = np.zeros((K + 1, K))
    np.add.at(c, (np.asarray(prev, int), np.asarray(nxt, int)), 1.0)
    return c


def _idx(names: pd.Series) -> np.ndarray:
    return names.map(lambda a: A.get(a, K)).to_numpy(int)


def sequence_transitions(seqs: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """(prev, next) index arrays over all consecutive pairs of the sequences."""
    pr = [s[i - 1] for s in seqs for i in range(1, len(s))]
    nx = [s[i] for s in seqs for i in range(1, len(s))]
    return np.asarray(pr, int), np.asarray(nx, int)


def session_summary(seqs: list[list[int]]) -> pd.DataFrame:
    """Per session: n_actions, n_search, n_items (click_item), n_details, n_options, bought (any buy)."""
    rows = []
    for s in seqs:
        c = np.bincount(np.asarray(s, int), minlength=K) if s else np.zeros(K, int)
        rows.append({"n_actions": len(s), "n_search": c[A["search"]], "n_items": c[A["click_item"]],
                     "n_details": c[A["view_detail"]], "n_options": c[A["select_option"]],
                     "bought": int(c[A["buy"]] > 0)})
    return pd.DataFrame(rows)


def compare_agents(data: TaskData) -> pd.DataFrame:
    """Action shares and per-session means: human train sessions, human test sessions, UXAgent (if loaded)."""
    seqs = data.meta["sequences"]
    groups = {"human_train": [seqs[s] for s in data.meta["train_sessions"]],
              "human_test": [seqs[s] for s in data.meta["test_sessions"]]}
    if data.meta.get("uxagent") is not None:
        groups["uxagent"] = data.meta["uxagent"]
    out = {}
    for g, ss in groups.items():
        allv = np.concatenate([np.asarray(s, int) for s in ss if s]) if ss else np.zeros(0, int)
        sh = np.bincount(allv, minlength=K) / max(len(allv), 1)
        summ = session_summary(ss).mean()
        out[g] = {"sessions": len(ss), **summ.to_dict(), **{f"share_{a}": sh[i] for i, a in enumerate(ACTIONS)}}
    return pd.DataFrame(out)


# ---------------------------------------------------------------------- task base
class _WebShop(BenchTask):
    data_subdir = "webshop-insan"
    layout = LAYOUT
    source = SOURCE
    accepted_options = OPTIONS

    def _test_sessions(self, s: pd.DataFrame) -> tuple[np.ndarray, str]:
        split = self.options.get("split", "batch")
        if split == "batch":
            return (s["date"] >= TEST_FROM).to_numpy(), (f"batch: collection batches dated before {TEST_FROM} train, "
                                                          f"from {TEST_FROM} on test")
        if split == "random":
            frac, seed = float(self.options.get("test_frac", 0.3)), int(self.options.get("seed", 0))
            te = np.zeros(len(s), bool)
            te[np.random.default_rng(seed).choice(len(s), int(round(frac * len(s))), replace=False)] = True
            return te, f"random: {frac:.0%} of sessions held out, default_rng({seed})"
        raise ValueError("split must be 'batch' or 'random'")

    def _base(self, root: str) -> tuple[pd.DataFrame, dict]:
        steps, s = read_webshop(root)
        te, rule = self._test_sessions(s)
        s["split"] = np.where(te, "test", "train")
        steps = steps.merge(s[["session", "split", "batch", "reward"]], on="session", how="left")
        steps["log_step"] = np.log1p(steps["t"])
        steps["log_results_page"] = np.log1p(steps["results_page"])
        seqs = {k: g.to_list() for k, g in steps.groupby("session", sort=False)["y"]}
        ux = read_uxagent(root) if self.options.get("uxagent", True) else None
        meta = {"split": rule, "actions": list(ACTIONS), "sessions": s.drop(columns=["instruction"]),
                "sequences": seqs, "train_sessions": s.loc[~te, "session"].tolist(),
                "test_sessions": s.loc[te, "session"].tolist(), "uxagent": ux,
                "n_uxagent_sessions": 0 if ux is None else len(ux), "source": LICENCE,
                "instructions": dict(zip(s["session"], s["instruction"]))}
        return steps, meta

    @staticmethod
    def _rows(steps: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
        return steps[mask].reset_index(drop=True)

    # ------------------------------------------------------------------ shared baseline pieces
    @staticmethod
    def _design(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        def oh(df):
            parts = [np.eye(K + 1)[_idx(df[c])] for c in ("prev", "prev2")]
            parts.append(np.eye(len(PAGES))[df["page"].map(lambda p: PAGES.index(p) if p in PAGES else 0).to_numpy()])
            return np.hstack(parts + [df[NUM_FEATURES].to_numpy(float)])
        xtr, xte = oh(train), oh(test)
        mu, sd = xtr.mean(0), xtr.std(0)
        sd[sd == 0] = 1.0
        return (xtr - mu) / sd, (xte - mu) / sd

    @staticmethod
    def _markov(train: pd.DataFrame, test: pd.DataFrame, alpha: float = 1.0) -> np.ndarray:
        """P(a_t | a_{t-1}) from train rows, add-alpha; (n_test, K)."""
        c = transition_counts(_idx(train["prev"]), train["y"].to_numpy(int)) + alpha
        return (c / c.sum(1, keepdims=True))[_idx(test["prev"])]

    @staticmethod
    def _markov2(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """P(a_t | a_{t-2}, a_{t-1}): counts plus PRIOR2 pseudo-counts spread as the first-order row."""
        p1 = _WebShop._markov(train, test)
        key = lambda d: _idx(d["prev2"]) * (K + 1) + _idx(d["prev"])
        c = np.zeros(((K + 1) ** 2, K))
        np.add.at(c, (key(train), train["y"].to_numpy(int)), 1.0)
        ct = c[key(test)]
        return (ct + PRIOR2 * p1) / (ct.sum(1, keepdims=True) + PRIOR2)

    @staticmethod
    def _ux_markov(ux: list[list[int]], test: pd.DataFrame, alpha: float = 1.0) -> np.ndarray:
        c = transition_counts(*sequence_transitions(ux)) + alpha
        return (c / c.sum(1, keepdims=True))[_idx(test["prev"])]


def _probs(frame: pd.DataFrame, col: str) -> np.ndarray:
    p = np.clip(np.vstack(frame[col].to_numpy()).astype(float), 0, None)
    s = p.sum(1, keepdims=True)
    return np.where(s > 0, p / np.where(s > 0, s, 1), 1.0 / K)


def _next_metric(which: str):
    def fn(task, frame, col, B, seed):
        p, y = _probs(frame, col), frame["truth"].to_numpy(int)
        n = len(y)
        rows = np.arange(n)
        if which == "loglik":
            v = np.log(np.clip(p[rows, y], EPS, 1.0))
            f = lambda i: v[i].mean()
        elif which == "accuracy":
            best = np.isclose(p, p.max(1, keepdims=True), rtol=0, atol=1e-12)
            v = best[rows, y] / best.sum(1)
            f = lambda i: v[i].mean()
        else:
            Y = np.eye(K)[y]
            f = lambda i: 0.5 * np.abs(p[i].mean(0) - Y[i].mean(0)).sum()
        r = cluster_bootstrap(f, frame["group"].to_numpy(), B=B, seed=seed)
        return {**r, "n": n, "n_sessions": r["n"]}
    return fn


@register()
class WebShopNextAction(_WebShop):
    """Next action of a human shopper (multiclass over ACTIONS). Options: split, test_frac, seed, uxagent.
    A prediction is an (n_test, K) probability array (columns in ACTIONS order; rows are renormalised, all-zero rows
    become uniform). data.train / data.test: session, t, page, action, y, prev, prev2, history, the NUM_FEATURES
    columns, split, batch, reward. meta: sessions (one row per session), sequences ({session: action indices}),
    instructions ({session: goal text}), uxagent (LLM-agent sequences or None)."""

    name = "webshop_next_action"
    kind = "multiclass"
    metrics = (Metric("loglik", True, _next_metric("loglik"), "mean log-likelihood of the action taken (clip 1e-6)"),
               Metric("accuracy", True, _next_metric("accuracy"), "argmax = action taken, ties split"),
               Metric("share_tvd", False, _next_metric("share_tvd"),
                      "TVD between mean predicted and observed action shares"))

    def _load(self, root: str) -> TaskData:
        steps, meta = self._base(root)
        ok = (steps["t"] >= 1).to_numpy()
        tr, te = ok & (steps["split"] == "train").to_numpy(), ok & (steps["split"] == "test").to_numpy()
        test = self._rows(steps, te)
        meta.update(n_train=int(tr.sum()), n_test=int(te.sum()), n_first_actions_excluded=int((~ok).sum()),
                    test_shares=(np.bincount(test["y"], minlength=K) / max(len(test), 1)).round(4).tolist())
        return TaskData(train=self._rows(steps, tr), test=test, meta=meta)

    def baselines(self, data: TaskData) -> dict:
        from sklearn.linear_model import LogisticRegression
        tr, te = data.train, data.test
        n = len(te)
        freq = np.bincount(tr["y"], minlength=K) + 1.0
        out = {"uniform": np.full((n, K), 1.0 / K), "frequency": np.tile(freq / freq.sum(), (n, 1)),
               "markov": self._markov(tr, te), "markov2": self._markov2(tr, te)}
        xtr, xte = self._design(tr, te)
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(xtr, tr["y"].to_numpy(int))
        p = np.full((n, K), EPS)
        p[:, lr.classes_] = lr.predict_proba(xte)
        out["logistic"] = p / p.sum(1, keepdims=True)
        if data.meta.get("uxagent"):
            out["uxagent_markov"] = self._ux_markov(data.meta["uxagent"], te)
        return out

    def frame(self, data: TaskData, preds: dict[str, Any]) -> pd.DataFrame:
        base = self._frame(data).reset_index(drop=True)
        n = len(base)
        for m, p in preds.items():
            a = np.asarray(p, float)
            if a.shape != (n, K):
                raise ValueError(f"{m}: predictions must have shape ({n}, {K}), got {a.shape}")
            if np.isnan(a).any() or (a < 0).any():
                raise ValueError(f"{m}: NaN or negative probabilities")
            base[as_pred_col(m)] = pd.Series(list(a), dtype=object)
        base["split"] = "test"
        return base

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        return _common_frame(t, t["y"].to_numpy(int))


def _common_frame(t: pd.DataFrame, truth: np.ndarray) -> pd.DataFrame:
    st = t["t"].to_numpy()
    return pd.DataFrame({"unit": t["session"] + ":" + t["t"].astype(str), "group": t["session"].to_numpy(),
                         "truth": truth, "item": t["prev"].to_numpy(), "segment": t["page"].to_numpy(),
                         "seg_step": np.where(st <= 2, "1-2", np.where(st <= 5, "3-5", "6+")),
                         "seg_success": np.where(t["reward"].to_numpy(float) >= 1.0, "reward=1", "reward<1")})


# ---------------------------------------------------------------------- buy task
def _buy_metric(which: str):
    def fn(task, frame, col, B, seed):
        y, p = frame["truth"].to_numpy(float), frame[col].to_numpy(float)
        if which == "auc":
            g = lambda i: auc(y[i], p[i])
        elif which == "logloss":
            q = np.clip(p, EPS, 1 - EPS)
            v = -(y * np.log(q) + (1 - y) * np.log(1 - q))
            g = lambda i: v[i].mean()
        else:
            v = (p - y) ** 2
            g = lambda i: v[i].mean()
        r = cluster_bootstrap(g, frame["group"].to_numpy(), B=B, seed=seed)
        return {**r, "n": len(y), "n_sessions": r["n"]}
    return fn


@register()
class WebShopBuy(_WebShop):
    """Purchase now (binary): rows are the actions taken on an item page (t >= 1), truth = the action is buy.
    A prediction is one probability per data.test row. Same options, columns and meta as WebShopNextAction."""

    name = "webshop_buy"
    kind = "binary"
    metrics = (Metric("auc", True, _buy_metric("auc"), "ROC AUC"),
               Metric("logloss", False, _buy_metric("logloss"), "log loss (clip 1e-6)"),
               Metric("brier", False, _buy_metric("brier"), "Brier score"))

    def _load(self, root: str) -> TaskData:
        steps, meta = self._base(root)
        ok = ((steps["t"] >= 1) & (steps["page"] == "item_page")).to_numpy()
        tr, te = ok & (steps["split"] == "train").to_numpy(), ok & (steps["split"] == "test").to_numpy()
        train, test = self._rows(steps, tr), self._rows(steps, te)
        meta.update(n_train=int(tr.sum()), n_test=int(te.sum()),
                    positive_rate={"train": float((train["action"] == "buy").mean()),
                                   "test": float((test["action"] == "buy").mean())})
        return TaskData(train=train, test=test, meta=meta)

    def baselines(self, data: TaskData) -> dict:
        from sklearn.linear_model import LogisticRegression
        tr, te = data.train, data.test
        ytr = (tr["y"] == A["buy"]).to_numpy(int)
        b = A["buy"]
        out = {"constant": np.full(len(te), float(ytr.mean())), "markov": self._markov(tr, te)[:, b]}
        xtr, xte = self._design(tr, te)
        if len(np.unique(ytr)) == 2:
            out["logistic"] = LogisticRegression(C=1.0, max_iter=3000).fit(xtr, ytr).predict_proba(xte)[:, 1]
        else:
            out["logistic"] = out["constant"].copy()
        if data.meta.get("uxagent"):
            out["uxagent_markov"] = self._ux_markov(data.meta["uxagent"], te)[:, b]
        return out

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        return _common_frame(t, (t["y"] == A["buy"]).to_numpy(int))


# ---------------------------------------------------------------------- LLM method
SYSTEM = ("You predict what a human shopper does next on a simple shopping website. The shopper was given a "
          "shopping instruction and is trying to buy a matching product. Answer only with a JSON object of "
          "probabilities over the listed actions.")
_DESC = {"search": "type a new search query (start page only)", "click_item": "open a product from the results",
         "select_option": "choose a product option (size, colour, ...)",
         "view_detail": "open the description / features / reviews tab", "next_page": "next results page",
         "prev_page": "previous results page", "back": "go back to the previous page",
         "back_to_search": "go back to the start page to search again", "buy": "press Buy Now on this product",
         "other": "anything else (reload, odd navigation)"}


def prompts(df: pd.DataFrame, instructions: dict[str, str], system: str = SYSTEM) -> list[tuple[str, str]]:
    """(system, user) per row: instruction, actions so far, current page and options chosen, the action list."""
    menu = "\n".join(f'- "{a}": {_DESC[a]}' for a in ACTIONS)
    out = []
    for r in df[["session", "history", "page", "n_opts", "n_goal_opts", "n_items"]].itertuples(index=False):
        hist = r.history.replace(" ", ", ") or "(none)"
        out.append((system, f"Instruction: {instructions.get(r.session, '')}\nActions so far: {hist}\n"
                            f"Current page: {r.page}; products opened so far: {r.n_items}; options chosen on this "
                            f"product: {r.n_opts} of {r.n_goal_opts} requested.\n\nPossible actions:\n{menu}\n\n"
                            f"Give the probability of each action as the shopper's next step."))
    return out


def parse_action(text: str) -> np.ndarray | None:
    """Probability vector over ACTIONS from a reply: a JSON object keyed by action names, else the first action name
    in the text (one-hot); None if nothing is recognised. Not renormalised."""
    m = re.search(r"\{.*?\}", text or "", re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = np.zeros(K)
            for k, x in obj.items():
                j = A.get(str(k).strip().lower().replace(" ", "_"))
                if j is not None and np.isfinite(x := float(x)) and x > 0:      # NaN / inf / negative dropped
                    v[j] += x
            if v.sum() > 0:
                return v
        except (ValueError, TypeError, AttributeError):
            pass
    names = sorted(ACTIONS, key=len, reverse=True)       # back_to_search before back
    w = re.search(r"\b(" + "|".join(names) + r")\b", (text or "").lower())
    if w:
        v = np.zeros(K)
        v[A[w.group(1)]] = 1.0
        return v
    return None


def llm_method(llm: Any, *, system: str = SYSTEM, smooth: float = 0.05, max_tokens: int = 128,
               parallel: int = 4) -> Callable[[TaskData], np.ndarray]:
    """Method for task.run on either task: one prompt per data.test row through llm.map, replies parsed by
    parse_action, mixed with `smooth` of the uniform vector; unparseable -> uniform (counted in .last_unparsed).
    Returns (n, K) probabilities for webshop_next_action and the buy column for webshop_buy."""
    def method(data: TaskData) -> np.ndarray:
        res = llm.map(prompts(data.test, data.meta.get("instructions", {}), system), max_tokens=max_tokens,
                      parallel=parallel)
        parsed = [parse_action(r.text) for r in res]
        method.last_unparsed = sum(p is None for p in parsed)
        p = np.vstack([np.full(K, 1.0 / K) if v is None else v / v.sum() for v in parsed]) if parsed \
            else np.zeros((0, K))
        p = (1 - smooth) * p + smooth / K
        return p[:, A["buy"]] if data.meta.get("kind") == "binary" else p
    method.last_unparsed = 0
    return method

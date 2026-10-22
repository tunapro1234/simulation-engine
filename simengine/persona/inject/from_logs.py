"""Personas from behavioural logs: purchase / view / click / rating events rendered raw, retrieved, as a
statistical summary, or as an LLM summary made beforehand.

    import simengine.persona.inject as pi
    from simengine.persona.inject import from_logs as fl
    person = {"id": "u1", "history": [
        {"t": "2024-03-02", "action": "purchase", "item": "i9", "title": "Oat milk 1L", "category": "dairy",
         "brand": "B", "price": 2.1, "qty": 2, "discount": True},
        {"t": "2024-03-05", "action": "rate", "item": "m4", "title": "Heat (1995)", "category": "Action|Crime",
         "rating": 4}]}
    q = {"id": "q1", "text": "Would you buy this oat milk at 2.50?", "options": ["yes", "no"], "cutoff": "2024-04-01"}
    pi.get_injector("log_raw", n=10)(person, q)              # last 10 events in context (Kang et al. 2023)
    pi.get_injector("log_stats")(person, q)                  # RFM / affinity / price / brand statistics
    people = fl.summarize_logs(llm, people, style="paars", cutoff="2024-04-01")   # LLM summaries, offline
    pi.get_injector("log_paars")(people[0], q)

Event dicts in person["history"] (all keys optional): t | time | date | timestamp (ISO string, date or number),
action (purchase, view, click, cart, rate, ...; missing = "rate" when rating is set, else "purchase"), item,
title, category (str, "A|B" or list), brand, price (unit), qty, discount (bool or fraction), rating, text, qid.

Question extensions read here: `item` / `items` (target item ids: their events are hidden) and `cutoff`
(events at or after it are hidden; events without a time are hidden too once a cutoff is set).
LEAKAGE RULE: every injector reads events through `visible_events`; LLM summaries carry the cutoff and item ids
they were built from, and rendering refuses a question whose cutoff is earlier or whose target item was summarised.
Summaries are made by `summarize_logs` before the run, so render() stays pure and dry runs spend nothing.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from datetime import date, datetime
from typing import Any, Iterable, Sequence

import pandas as pd

from ..history import purchase_history_text, summarize_purchases
from ._base import Injector, MissingPersonField, Prompt, register

FAMILY = "from_logs"
TIME_KEYS = ("t", "time", "date", "timestamp")
PURCHASE = ("purchase", "buy", "order", "transaction")
SUMMARY_FIELD = "log_summary"
STYLES = ("paars", "agent4rec", "task")
PAARS_FACTORS = ("products", "price", "value", "product selection", "reviews", "brand reputation", "quality")
PAARS_PROFILE = ("gender", "age range", "relationship status", "education", "industry", "salary range",
                 "home ownership", "parental status", "interests")


# ---------------------------------------------------------------- events
def _ts(v: Any) -> Any:
    """Sortable time: numbers stay numbers, strings/dates become pandas Timestamps, None stays None."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, bool):
        raise ValueError(f"event time must be a date, string or number, not {v!r}")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (str, date, datetime, pd.Timestamp)):
        return pd.Timestamp(v)
    raise ValueError(f"event time must be a date, string or number, not {type(v).__name__}")


def event_time(e: dict) -> Any:
    for k in TIME_KEYS:
        if e.get(k) is not None:
            return _ts(e[k])
    return None


def action(e: dict) -> str:
    a = e.get("action")
    if a:
        return str(a).lower()
    return "rate" if e.get("rating") is not None else "purchase"


def categories(e: dict) -> list[str]:
    c = e.get("category")
    if c is None or c == "":
        return []
    if isinstance(c, (list, tuple, set)):
        return [str(x) for x in c]
    return [x.strip() for x in str(c).split("|") if x.strip()]


def events(person: dict) -> list[dict]:
    """person['history'] as event dicts; (role, text) chat turns are refused with a clear error."""
    h = person.get("history") or []
    bad = [x for x in h if not isinstance(x, dict)]
    if bad:
        raise ValueError(f"from_logs injectors read person['history'] as event dicts (person {person.get('id')!r} "
                         f"has {type(bad[0]).__name__} entries, e.g. chat turns)")
    return list(h)


def _target_items(question: dict) -> set[str]:
    out = {str(x) for x in (question.get("items") or ())}
    if question.get("item") is not None:
        out.add(str(question["item"]))
    return out


def _cut(*vals: Any) -> Any:
    ts = [_ts(v) for v in vals if v is not None]
    if not ts:
        return None
    try:
        return min(ts)
    except TypeError:
        raise ValueError(f"cutoffs mix numbers and dates: {vals}") from None


def visible_events(person: dict, question: dict, cutoff: Any = None, hide_target_item: bool = True) -> list[dict]:
    """Events the model may see for `question`, oldest first: drops events at/after the earlier of
    question['cutoff'] and `cutoff` (and undated events once a cutoff is set), events on the target item(s),
    and events tagged with the target or an `exclude` qid."""
    cut = _cut(question.get("cutoff"), cutoff)
    items = _target_items(question) if hide_target_item else set()
    qids = {str(x) for x in (question.get("exclude") or ())}
    if question.get("id") is not None:
        qids.add(str(question["id"]))
    keep = []
    for i, e in enumerate(events(person)):
        t = event_time(e)
        if cut is not None:
            try:
                if t is None or t >= cut:
                    continue
            except TypeError:
                raise ValueError(f"event time {t!r} and cutoff {cut!r} are not comparable "
                                 "(mix of numbers and dates)") from None
        if e.get("item") is not None and str(e["item"]) in items:
            continue
        if e.get("qid") is not None and str(e["qid"]) in qids:
            continue
        keep.append((i, t, e))
    if any(t is not None for _, t, _ in keep):
        try:
            keep.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0, x[0]))
        except TypeError:
            raise ValueError(f"person {person.get('id')!r} mixes numeric and date event times") from None
    return [e for _, _, e in keep]


def _num(x: Any) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:.2f}".rstrip("0").rstrip(".")


def _when(e: dict) -> str:
    t = event_time(e)
    if t is None:
        return ""
    if isinstance(t, pd.Timestamp):
        return str(t.date()) if t == t.normalize() else str(t)[:16]
    return f"t={_num(t)}"


def event_line(e: dict) -> str:
    """One event as text: '2024-03-02 purchase "Oat milk 1L" (dairy; brand B) 2 x 2.1, on discount, rated 4'."""
    parts = [p for p in (_when(e), action(e)) if p]
    name = e.get("title") or e.get("item")
    if name is not None:
        parts.append(f'"{name}"')
    tags = []
    if categories(e):
        tags.append(", ".join(categories(e)))
    if e.get("brand"):
        tags.append(f"brand {e['brand']}")
    if tags:
        parts.append(f"({'; '.join(tags)})")
    line = " ".join(parts)
    extra = []
    if e.get("price") is not None:
        q = e.get("qty")
        extra.append(f"{_num(q)} x {_num(e['price'])}" if q not in (None, 1) else f"price {_num(e['price'])}")
    elif e.get("qty") not in (None, 1):
        extra.append(f"qty {_num(e['qty'])}")
    d = e.get("discount")
    if d:
        extra.append("on discount" if d is True else f"{_num(float(d) * 100)}% off")
    if e.get("rating") is not None:
        extra.append(f"rated {_num(e['rating'])}")
    if extra:
        line += ", " + ", ".join(extra)
    if e.get("text"):
        line += f': "{e["text"]}"'
    return line


def sessions_text(evs: Sequence[dict]) -> str:
    """Events grouped into daily sessions (PAARS layout); undated events go last under 'Undated'."""
    groups: dict[str, list[str]] = {}
    for e in evs:
        t = event_time(e)
        key = str(t.date()) if isinstance(t, pd.Timestamp) else (f"t={_num(t)}" if t is not None else "Undated")
        line = event_line(e)
        groups.setdefault(key, []).append(line.split(" ", 1)[1] if key != "Undated" and " " in line else line)
    return "\n".join(f"Session {k}:\n" + "\n".join(f"  - {x}" for x in v) for k, v in groups.items())


_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


def bm25_rank(query: str, evs: Sequence[dict], k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Indices of `evs` by BM25 score against `query` (idf over this person's events); ties: most recent first."""
    docs = [_tokens(event_line(e)) for e in evs]
    if not docs:
        return []
    n, avg = len(docs), (sum(map(len, docs)) / len(docs)) or 1.0
    df = Counter(w for d in docs for w in set(d))
    q = set(_tokens(query))
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for w in q & set(tf):
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return sorted(range(n), key=lambda i: (-scores[i], -i))


def _query(question: dict) -> str:
    return " ".join([str(question.get("text", ""))] + [str(o) for o in question.get("options") or ()]
                    + [str(question[k]) for k in ("title", "item") if question.get(k) is not None])


# ---------------------------------------------------------------- statistics
def log_stats(evs: Sequence[dict], *, ref: Any = None, top_k: int = 5, like_at: float = 4) -> dict:
    """RFM-style numbers from visible events: recency, frequency, monetary, category shares, discount share,
    bought/viewed price ratio, top-brand share, repeat-purchase share, rating mean. Missing data -> key absent."""
    buys = [e for e in evs if action(e) in PURCHASE]
    out: dict[str, Any] = {"n_events": len(evs), "n_purchases": len(buys)}
    times = [event_time(e) for e in buys if event_time(e) is not None]
    if times:
        out["first"], out["last"] = min(times), max(times)
        out["purchase_days"] = len({t.normalize() if isinstance(t, pd.Timestamp) else t for t in times})
        ref_t = _ts(ref)
        if ref_t is not None:
            gap = ref_t - out["last"]
            out["recency_days"] = gap.days if isinstance(gap, pd.Timedelta) else gap
        if isinstance(out["first"], pd.Timestamp):
            end = ref_t if isinstance(ref_t, pd.Timestamp) else out["last"]
            months = max((end - out["first"]).days / 30.44, 1.0)
            out["purchases_per_month"] = len(buys) / months
    spend = [float(e["price"]) * float(e.get("qty") or 1) for e in buys if e.get("price") is not None]
    if spend:
        out["spend_total"] = sum(spend)
        out["spend_per_day"] = sum(spend) / max(out.get("purchase_days") or len(spend), 1)
    cats = Counter(c for e in buys for c in categories(e))
    if cats:
        tot = sum(cats.values())
        out["categories"] = [(c, n / tot) for c, n in sorted(cats.items(), key=lambda x: (-x[1], x[0]))[:top_k]]
    disc = [bool(e["discount"]) for e in buys if "discount" in e]
    if disc:
        out["discount_share"] = sum(disc) / len(disc)
    viewed = [float(e["price"]) for e in evs if action(e) not in PURCHASE and action(e) != "rate"
              and e.get("price") is not None]
    bought = [float(e["price"]) for e in buys if e.get("price") is not None]
    if viewed and bought and sum(viewed) > 0:
        out["bought_vs_viewed_price"] = (sum(bought) / len(bought)) / (sum(viewed) / len(viewed))
    brands = Counter(str(e["brand"]) for e in buys if e.get("brand"))
    if brands:
        b, n = sorted(brands.items(), key=lambda x: (-x[1], x[0]))[0]
        out.update(top_brand=b, top_brand_share=n / sum(brands.values()), n_brands=len(brands))
    items = [str(e["item"]) for e in buys if e.get("item") is not None]
    if items:
        seen, rep = set(), 0
        for it in items:
            rep += it in seen
            seen.add(it)
        out["repeat_share"] = rep / len(items)
    rs = [float(e["rating"]) for e in evs if e.get("rating") is not None]
    if rs:
        out.update(n_ratings=len(rs), rating_mean=sum(rs) / len(rs),
                   rating_high_share=sum(r >= like_at for r in rs) / len(rs))
    return out


def stats_text(s: dict, like_at: float = 4) -> list[str]:
    def d(t):
        return str(t.date()) if isinstance(t, pd.Timestamp) else f"t={_num(t)}"
    lines = []
    if "last" in s:
        unit = "days" if isinstance(s["last"], pd.Timestamp) else "time units"
        rec = (f"last purchase {_num(s['recency_days'])} {unit} before now ({d(s['last'])})"
               if "recency_days" in s else f"last purchase {d(s['last'])}")
        lines.append(f"- recency: {rec}")
    if s.get("n_purchases"):
        f = f"- frequency: {s['n_purchases']} purchases"
        if "purchase_days" in s:
            f += f" on {s['purchase_days']} different days"
        if "purchases_per_month" in s:
            f += f" ({s['purchases_per_month']:.1f} per month since {d(s['first'])})"
        lines.append(f)
    if "spend_total" in s:
        lines.append(f"- monetary: total spend {s['spend_total']:.2f}, {s['spend_per_day']:.2f} per shopping day")
    if "categories" in s:
        lines.append("- category affinity: " + ", ".join(f"{c} {p:.0%}" for c, p in s["categories"]))
    price = []
    if "discount_share" in s:
        price.append(f"{s['discount_share']:.0%} of purchases on discount")
    if "bought_vs_viewed_price" in s:
        price.append(f"items bought cost {s['bought_vs_viewed_price']:.0%} of the average price of items viewed")
    if price:
        lines.append("- price sensitivity: " + "; ".join(price))
    brand = []
    if "top_brand" in s:
        brand.append(f"top brand {s['top_brand']} is {s['top_brand_share']:.0%} of branded purchases "
                     f"({s['n_brands']} brands in all)")
    if "repeat_share" in s:
        brand.append(f"{s['repeat_share']:.0%} of purchases repeat an item bought before")
    if brand:
        lines.append("- brand loyalty: " + "; ".join(brand))
    if "n_ratings" in s:
        lines.append(f"- ratings: {s['n_ratings']} given, mean {s['rating_mean']:.1f}, "
                     f"{s['rating_high_share']:.0%} at {_num(like_at)} or above")
    return lines


def category_lines(evs: Sequence[dict], cats: Sequence[str], ref: Any = None) -> str:
    """Per-category purchase lines via persona.history (summarize_purchases + purchase_history_text);
    '' when the events have no calendar dates."""
    rows = [{"id": "p", "category": c, "date": event_time(e), "qty": float(e.get("qty") or 1)}
            for e in evs if action(e) in PURCHASE and isinstance(event_time(e), pd.Timestamp) for c in categories(e)]
    if not rows or not cats:
        return ""
    df = pd.DataFrame(rows)
    ref_t = _ts(ref)
    cutoff = ref_t if isinstance(ref_t, pd.Timestamp) else df["date"].max().normalize() + pd.Timedelta(days=1)
    summ = summarize_purchases(df, id_col="id", category_col="category", date_col="date", qty_col="qty",
                               cutoff=cutoff, categories=list(cats), ids=["p"])
    return purchase_history_text(summ, list(cats))


# ---------------------------------------------------------------- Agent4Rec traits
TRAITS = ("activity", "conformity", "diversity")
TIERS = ("low", "medium", "high")


def agent4rec_traits(evs: Sequence[dict]) -> dict:
    """Agent4Rec social traits (Zhang et al. 2023, sec. 3.1): activity = number of rated/interacted items,
    conformity = mean (r_ui - R_i)^2 over events carrying `item_mean` (None when none do; larger = less
    conforming), diversity = number of distinct categories (genres)."""
    rated = [e for e in evs if e.get("rating") is not None] or list(evs)
    dev = [(float(e["rating"]) - float(e["item_mean"])) ** 2 for e in rated
           if e.get("rating") is not None and e.get("item_mean") is not None]
    return {"activity": len(rated), "conformity": sum(dev) / len(dev) if dev else None,
            "diversity": len({c for e in rated for c in categories(e)})}


def _tier(v: float | None, cuts: Sequence[float] | None) -> str | None:
    if v is None or not cuts:
        return None
    lo, hi = cuts
    return TIERS[0] if v < lo else (TIERS[2] if v >= hi else TIERS[1])


def trait_cuts(people: Iterable[dict], question: dict | None = None) -> dict[str, tuple[float, float]]:
    """Tertile cut points of each Agent4Rec trait over `people` (the paper buckets users into three tiers from
    dataset statistics but does not publish the cut points). Pass the result as log_agent4rec(cuts=...)."""
    vals: dict[str, list[float]] = {t: [] for t in TRAITS}
    for p in people:
        tr = agent4rec_traits(visible_events(p, question or {}))
        for t in TRAITS:
            if tr[t] is not None:
                vals[t].append(float(tr[t]))
    out = {}
    for t, v in vals.items():
        if len(v) >= 3:
            s = pd.Series(v)
            out[t] = (float(s.quantile(1 / 3)), float(s.quantile(2 / 3)))
    return out


def add_item_means(people: Sequence[dict], question: dict | None = None) -> list[dict]:
    """Copies of `people` whose rating events carry `item_mean` (R_i, the mean rating of that item over the
    visible events of all people), needed for Agent4Rec conformity. Pass the question (cutoff/item) so R_i is
    built from the same visible window."""
    acc: dict[str, list[float]] = {}
    for p in people:
        for e in visible_events(p, question or {}):
            if e.get("rating") is not None and e.get("item") is not None:
                acc.setdefault(str(e["item"]), []).append(float(e["rating"]))
    means = {k: sum(v) / len(v) for k, v in acc.items()}
    out = []
    for p in people:
        h = [({**e, "item_mean": means[str(e["item"])]} if isinstance(e, dict) and e.get("item") is not None
              and str(e["item"]) in means else e) for e in (p.get("history") or [])]
        out.append({**p, "history": h})
    return out


def agent4rec_sample(person: dict, evs: Sequence[dict], n: int = 25, seed: int = 0) -> list[dict]:
    """The `n` rated events used for the taste summary: a seeded random sample (Agent4Rec samples 25 items from
    the viewing history), returned oldest first."""
    rated = [e for e in evs if e.get("rating") is not None]
    if len(rated) <= n:
        return rated
    idx = sorted(random.Random(f"{seed}:{person.get('id')}").sample(range(len(rated)), n))
    return [rated[i] for i in idx]


def liked_text(sample: Sequence[dict], like_at: float = 3) -> str:
    def name(e):
        return str(e.get("title") or e.get("item"))
    like = [name(e) for e in sample if float(e["rating"]) >= like_at]
    dislike = [name(e) for e in sample if float(e["rating"]) < like_at]
    return f"Liked: {'; '.join(like) or '(none)'}\nDisliked: {'; '.join(dislike) or '(none)'}"


# ---------------------------------------------------------------- injectors
class _LogInjector(Injector):
    """Shared plumbing: cutoff / target-item hiding, the system header and the person's visible events."""
    family = FAMILY
    needs = ("history",)
    header = "You are a real person. Below is a record of your own past activity."

    def __init__(self, cutoff: Any = None, hide_target_item: bool = True, **params: Any):
        super().__init__(**({"cutoff": cutoff} if cutoff is not None else {}),
                         **({} if hide_target_item else {"hide_target_item": False}), **params)
        self.cutoff = cutoff
        self.hide_target_item = hide_target_item

    def visible(self, person: dict, question: dict) -> list[dict]:
        return visible_events(person, question, self.cutoff, self.hide_target_item)

    def ref(self, question: dict) -> Any:
        return _cut(question.get("cutoff"), self.cutoff)

    def prompt(self, body: str, question: dict, evs: Sequence[dict], **meta: Any) -> Prompt:
        system = f"{self.header}\n\n{body}\n\nAnswer as this person would, consistent with this record."
        return Prompt(system=system, user=self.question_block(question), meta={"n_events": len(evs), **meta})


@register
class LogRaw(_LogInjector):
    """The last `n` events verbatim, oldest first (Kang et al. 2023 rating prediction prompt: the user's
    chronological rating history truncated to the 10 most recent interactions); `recency_note` adds Hou et al.'s
    recency-focused sentence naming the latest item."""
    name = "log_raw"
    paper = ("Kang et al. 2023, arXiv:2305.06474 (recent-history prompt); recency note: Hou et al. 2024, "
             "arXiv:2305.08845")
    description = "Last n events (default 10) as text lines in chronological order; optional recency-focused note."

    def __init__(self, n: int = 10, recency_note: bool = False, order: str = "chronological", **params: Any):
        if order not in ("chronological", "recent_first"):
            raise ValueError("order must be 'chronological' or 'recent_first'")
        if n < 1:
            raise ValueError("n must be >= 1")
        super().__init__(n=n, recency_note=recency_note, order=order, **params)
        self.n, self.recency_note, self.order = n, recency_note, order

    def render(self, person: dict, question: dict) -> Prompt:
        evs = self.visible(person, question)[-self.n:]
        lines = [event_line(e) for e in (evs if self.order == "chronological" else evs[::-1])]
        what = "oldest first" if self.order == "chronological" else "most recent first"
        body = (f"Your last {len(evs)} recorded events ({what}):\n" + "\n".join(f"- {x}" for x in lines)
                if lines else "You have no recorded events yet.")
        if self.recency_note and evs:
            e = evs[-1]
            body += f"\nNote that your most recent event is: {action(e)} \"{e.get('title') or e.get('item')}\"."
        return self.prompt(body, question, evs)


@register
class LogRetrieved(_LogInjector):
    """The `k` events most relevant to the question (LaMP retrieval-augmented personalisation: BM25 over the
    user's profile items, or the most recent items, or random), highest score first."""
    name = "log_retrieved"
    paper = "Salemi et al. 2024 (LaMP), arXiv:2304.11406"
    description = "k events retrieved from the log by BM25 against the question (or recency / seeded random)."

    def __init__(self, k: int = 5, method: str = "bm25", seed: int = 0, **params: Any):
        if method not in ("bm25", "recent", "random"):
            raise ValueError("method must be 'bm25', 'recent' or 'random'")
        if k < 1:
            raise ValueError("k must be >= 1")
        super().__init__(k=k, method=method, **({"seed": seed} if method == "random" else {}), **params)
        self.k, self.method, self.seed = k, method, seed

    def render(self, person: dict, question: dict) -> Prompt:
        evs = self.visible(person, question)
        if self.method == "bm25":
            idx = bm25_rank(_query(question), evs)
        elif self.method == "recent":
            idx = list(range(len(evs)))[::-1]
        else:
            idx = list(range(len(evs)))
            random.Random(f"{self.seed}:{person.get('id')}:{question.get('id')}").shuffle(idx)
        pick = [evs[i] for i in idx[:self.k]]
        body = (f"Your {len(pick)} past events most relevant to the question:\n"
                + "\n".join(f"- {event_line(e)}" for e in pick) if pick else "You have no recorded events yet.")
        return self.prompt(body, question, pick, retrieved=self.method)


@register
class LogStats(_LogInjector):
    """Statistical summary of the log: RFM, category affinity, price sensitivity, brand loyalty, ratings, plus
    per-category purchase lines from persona.history. No LLM involved; the classical marketing summary."""
    name = "log_stats"
    paper = ("statistical-summary baseline: RFM (Hughes 1994, no arXiv); the non-LLM counterpart of purchase-history "
             "personas such as Shi et al. 2025, arXiv:2504.17304")
    description = ("Recency, frequency, monetary, top-category shares, discount share, bought/viewed price ratio, "
                   "top-brand and repeat shares, rating mean; optional per-category lines (persona.history).")

    def __init__(self, top_k: int = 5, per_category: bool = True, like_at: float = 4, **params: Any):
        super().__init__(top_k=top_k, per_category=per_category, like_at=like_at, **params)
        self.top_k, self.per_category, self.like_at = top_k, per_category, like_at

    def render(self, person: dict, question: dict) -> Prompt:
        evs = self.visible(person, question)
        s = log_stats(evs, ref=self.ref(question), top_k=self.top_k, like_at=self.like_at)
        lines = stats_text(s, self.like_at)
        body = (f"Statistics from your own activity record ({s['n_events']} events):\n" + "\n".join(lines)
                if lines else "You have no recorded purchases or ratings yet.")
        if self.per_category and s.get("categories"):
            cat = category_lines(evs, [c for c, _ in s["categories"]], self.ref(question))
            if cat:
                body += "\n\nPurchases per category:\n" + cat
        return self.prompt(body, question, evs, stats={k: v for k, v in s.items() if not isinstance(v, pd.Timestamp)})


class _SummaryInjector(_LogInjector):
    """Base for injectors that show an LLM summary made beforehand by `summarize_logs` (style `style`)."""
    style = ""

    def check(self, person: dict) -> None:
        super().check(person)
        if self.style not in (person.get(SUMMARY_FIELD) or {}):
            raise MissingPersonField(
                f"injector {self.name!r} needs person field(s) {list(self.needs) + [SUMMARY_FIELD]} with a "
                f"{self.style!r} entry (person {person.get('id')!r}); run "
                f"from_logs.summarize_logs(llm, people, style={self.style!r}) first")

    def summary(self, person: dict, question: dict) -> dict:
        """The stored summary, after checking it was built without the target item and not past the cutoff."""
        s = person[SUMMARY_FIELD][self.style]
        hit = _target_items(question) & set(s.get("items") or ())
        if hit and self.hide_target_item:
            raise ValueError(f"leakage: {self.style} summary of person {person.get('id')!r} was built from target "
                             f"item(s) {sorted(hit)}; rebuild it with summarize_logs(..., hide_items=...)")
        cut = self.ref(question)
        if cut is not None:
            sc = _ts(s.get("cutoff"))
            try:
                late = sc is None or sc > cut
            except TypeError:
                late = True
            if late:
                raise ValueError(f"leakage: {self.style} summary of person {person.get('id')!r} has cutoff "
                                 f"{s.get('cutoff')!r}, later than the question's {cut}; rebuild with that cutoff")
        return s


@register
class LogAgent4Rec(_SummaryInjector):
    """Agent4Rec user profile: three social traits computed from the log (activity, conformity, diversity; in
    tiers when `cuts` from `trait_cuts` are given) plus a taste summary distilled by an LLM from 25 sampled rated
    items (like = rating >= 3). taste='items' shows the sampled like/dislike lists instead of the LLM summary."""
    name = "log_agent4rec"
    paper = "Zhang et al. 2024 (Agent4Rec), arXiv:2310.10108"
    description = ("Activity / conformity / diversity traits (tiered with cuts) + taste summary of 25 sampled "
                   "rated items (summarize_logs style 'agent4rec', or taste='items' for the raw like/dislike lists).")
    style = "agent4rec"

    def __init__(self, taste: str = "summary", n_items: int = 25, like_at: float = 3, seed: int = 0,
                 cuts: dict | None = None, **params: Any):
        if taste not in ("summary", "items"):
            raise ValueError("taste must be 'summary' or 'items'")
        super().__init__(taste=taste, n_items=n_items, like_at=like_at, seed=seed,
                         **({"cuts": dict(cuts)} if cuts else {}), **params)
        self.taste, self.n_items, self.like_at, self.seed = taste, n_items, like_at, seed
        self.cuts = dict(cuts or {})

    def check(self, person: dict) -> None:
        if self.taste == "summary":
            super().check(person)
        else:
            _LogInjector.check(self, person)

    def traits_text(self, tr: dict) -> list[str]:
        words = {
            "activity": ("You rate few items", "You rate a moderate number of items", "You are a very active rater"),
            "conformity": ("Your ratings usually agree with the average opinion",
                           "Your ratings sometimes differ from the average opinion",
                           "Your ratings often differ from the average opinion"),
            "diversity": ("You stick to a few genres", "You watch a moderate range of genres",
                          "You watch a very wide range of genres")}
        unit = {"activity": "items rated", "conformity": "mean squared gap to the item average",
                "diversity": "distinct genres"}
        out = []
        for t in TRAITS:
            v = tr[t]
            if v is None:
                continue
            tier = _tier(v, self.cuts.get(t))
            num = f"{_num(v)} {unit[t]}"
            out.append(f"- {t}: {words[t][TIERS.index(tier)]} ({num})." if tier else f"- {t}: {num}.")
        return out

    def render(self, person: dict, question: dict) -> Prompt:
        evs = self.visible(person, question)
        tr = agent4rec_traits(evs)
        body = "Your traits as a user:\n" + "\n".join(self.traits_text(tr))
        if self.taste == "summary":
            body += "\n\nYour tastes:\n" + self.summary(person, question)["text"]
        else:
            sample = agent4rec_sample(person, evs, self.n_items, self.seed)
            body += f"\n\nItems you rated ({len(sample)} sampled):\n" + liked_text(sample, self.like_at)
        return self.prompt(body, question, evs, traits=tr)


@register
class LogPaars(_SummaryInjector):
    """PAARS persona: an LLM-inferred consumer profile and how much each of seven shopping factors matters (two
    prompts, made by summarize_logs style 'paars'), followed by the raw shopping history in daily sessions
    (`include_raw`, the last `n_raw` events), as in the paper's final persona."""
    name = "log_paars"
    paper = "Mansour et al. 2025 (PAARS), arXiv:2503.24228"
    description = ("LLM consumer profile + shopping preferences over seven factors (summarize_logs style 'paars') "
                   "+ raw history as daily sessions.")
    style = "paars"

    def __init__(self, include_raw: bool = True, n_raw: int = 30, **params: Any):
        super().__init__(include_raw=include_raw, n_raw=n_raw, **params)
        self.include_raw, self.n_raw = include_raw, n_raw

    def render(self, person: dict, question: dict) -> Prompt:
        s = self.summary(person, question)
        body = f"Your consumer profile:\n{s['profile']}\n\nYour shopping preferences:\n{s['preferences']}"
        evs = self.visible(person, question)[-self.n_raw:] if self.include_raw else []
        if evs:
            body += "\n\nYour recent shopping history:\n" + sessions_text(evs)
        return self.prompt(body, question, evs)


@register
class LogTaskSummary(_SummaryInjector):
    """Task-aware LLM summary of the log made offline (Richardson et al. 2023), optionally with `k` BM25-retrieved
    events next to it (the paper combines the summary with a smaller retrieval)."""
    name = "log_task_summary"
    paper = "Richardson et al. 2023, arXiv:2310.20081"
    description = ("Offline task-aware LLM summary (summarize_logs style 'task', task=...) + optional k retrieved "
                   "events.")
    style = "task"

    def __init__(self, k: int = 0, **params: Any):
        super().__init__(k=k, **params)
        self.k = k

    def render(self, person: dict, question: dict) -> Prompt:
        body = "Summary of your past activity:\n" + self.summary(person, question)["text"]
        pick: list[dict] = []
        if self.k:
            evs = self.visible(person, question)
            pick = [evs[i] for i in bm25_rank(_query(question), evs)[:self.k]]
            if pick:
                body += "\n\nRelated past events:\n" + "\n".join(f"- {event_line(e)}" for e in pick)
        return self.prompt(body, question, pick)


# ---------------------------------------------------------------- offline LLM summaries
SUMMARY_SYSTEM = "You analyse anonymised activity logs of one person. Be concrete and stay faithful to the log."


def _paars_profile_prompt(hist: str) -> str:
    return (f"Shopping activity of one anonymous shopper, in daily sessions:\n{hist}\n\n"
            f"Infer this shopper's likely {', '.join(PAARS_PROFILE)}. For each, give a value and a one-line reason "
            "drawn from the activity. Reply with JSON only: {\"<field>\": {\"value\": ..., \"reason\": ...}}.")


def _paars_prefs_prompt(hist: str, profile: str) -> str:
    return (f"Shopper profile:\n{profile}\n\nShopping activity, in daily sessions:\n{hist}\n\n"
            f"How important is each of these factors to this shopper when choosing what to buy: "
            f"{', '.join(PAARS_FACTORS)}? Reason from the activity, then write one paragraph to the shopper "
            "(\"You ...\") that says how much each factor matters to them.")


def _agent4rec_prompt(sample_text: str, n: int) -> str:
    return (f"{n} items this user rated (liked = rating 3 or above):\n{sample_text}\n\n"
            "Summarise the user's tastes and rating habits in 3 to 4 sentences written to the user (\"You ...\").")


def _task_prompt(hist: str, task: str) -> str:
    return (f"Activity log of one user, oldest first:\n{hist}\n\nThis summary will be used to {task}. Summarise "
            "what in the log is relevant to that task (preferences, habits, price and brand behaviour) in at most "
            "120 words, written to the user (\"You ...\").")


def summarize_logs(llm, people: Sequence[dict], style: str = "paars", *, cutoff: Any = None,
                   hide_items: Iterable = (), n: int = 50, n_items: int = 25, seed: int = 0, like_at: float = 3,
                   task: str | None = None, max_tokens: int = 400, parallel: int = 4, dry_run: bool = False,
                   **llm_kw: Any) -> list[dict] | dict:
    """Copies of `people` with person['log_summary'][style] = {text, ..., cutoff, items, n_events, model}.

    style 'paars': two calls per person (consumer profile JSON, then how much each of the seven factors matters) over
    the last `n` events in daily sessions. 'agent4rec': one call over `n_items` seeded-sampled rated events
    (like = rating >= like_at). 'task': one call over the last `n` events, steered by `task` (e.g. "predict which
    product categories the user buys next month"). Only events before `cutoff` and not on `hide_items` are used;
    the summary records both so injectors can refuse leaky use. dry_run: {"calls", "input_tokens_est"}, no call."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    if style == "task" and not task:
        raise ValueError("style 'task' needs task='<what the summary will be used for>'")
    window = {"cutoff": cutoff, "items": list(hide_items)}
    plans = []
    for p in people:
        evs = visible_events(p, window)
        if style == "agent4rec":
            used = agent4rec_sample(p, evs, n_items, seed)
            first = _agent4rec_prompt(liked_text(used, like_at), len(used))
        else:
            used = evs[-n:]
            first = (_paars_profile_prompt(sessions_text(used)) if style == "paars"
                     else _task_prompt("\n".join(f"- {event_line(e)}" for e in used), task))
        plans.append((p, used, first))
    if dry_run:
        chars = sum(len(SUMMARY_SYSTEM) + len(u) for _, _, u in plans) * (2 if style == "paars" else 1)
        return {"calls": len(plans) * (2 if style == "paars" else 1), "input_tokens_est": chars // 4,
                "output_tokens_max": len(plans) * (2 if style == "paars" else 1) * max_tokens}

    def run(plan):
        p, used, first = plan
        out = {"text": llm.complete(SUMMARY_SYSTEM, first, max_tokens=max_tokens, **llm_kw).text.strip()}
        if style == "paars":
            prefs = llm.complete(SUMMARY_SYSTEM, _paars_prefs_prompt(sessions_text(used), out["text"]),
                                 max_tokens=max_tokens, **llm_kw).text.strip()
            out = {"profile": out["text"], "preferences": prefs, "text": f"{out['text']}\n\n{prefs}"}
        if style == "task":
            out["task"] = task
        items = sorted({str(e["item"]) for e in used if e.get("item") is not None})
        cut = cutoff if cutoff is None or isinstance(cutoff, (int, float)) else str(cutoff)   # numbers stay numbers
        out.update(cutoff=cut, items=items, n_events=len(used),
                   model=getattr(llm, "model", None))
        return {**p, SUMMARY_FIELD: {**(p.get(SUMMARY_FIELD) or {}), style: out}}

    from ...llm.parallel import parallel_map
    return parallel_map(run, plans, parallel=parallel)


__all__ = ["LogRaw", "LogRetrieved", "LogStats", "LogAgent4Rec", "LogPaars", "LogTaskSummary", "summarize_logs",
           "visible_events", "event_line", "sessions_text", "log_stats", "stats_text", "category_lines",
           "agent4rec_traits", "agent4rec_sample", "trait_cuts", "add_item_means", "bm25_rank", "events",
           "SUMMARY_FIELD", "STYLES", "FAMILY"]

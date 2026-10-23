"""Event logs: sessions, cutoff windows that mimic a test period, pre-cutoff history features and rate rescaling.

Port of the D13 blind rehearsal code (``kod/ozellik.py``: a click log of views / add-to-carts / transactions; does
a visitor's first session after the cutoff, seen through its first two views, end in a purchase, and what are the
total and segment purchase rates) and of the rescaling arm in ``kod/modeller.py``.

Event log layout (long): one row per event with a unit column (``user_col``), a time column (``time_col``:
datetime64 or a number such as epoch ms) and optionally a kind column (``kind_col``) and an item column. Durations
(``gap``, ``horizon``, ``windows``, ...) are pandas offsets ("30min", "21D") for a datetime time column; for a
numeric one pass numbers in the column's unit, or set ``time_unit="ms"`` (or "s") to use offsets.

    ev = sessionize(log, "user", "time", gap="30min", kind_col="event")    # + session_id, session_start, seq
    cuts = [test_start - pd.Timedelta(days=7 * k) for k in range(3, 14)]   # weekly cutoffs, 21-day windows
    train, dev = cutoff_windows(
        ev, cuts, "21D", label="transaction", prefix_len=2, prefix_kinds=("view",), max_end_offset="3h",
        eval_gap="0D", kind_col="event", item_col="item",
        features=lambda hist, rows, at: history_features(hist, at, rows, on={"user": "user",
                                                                             "item": ["item0", "item1"]},
                                                         kind_col="event", kinds=("view", "addtocart", "transaction")))
    p = rescale_to_rates(p_dev, dev["segment"], train.groupby("segment")["y"].mean())   # label-free

Trap (D13 #8): 21-day windows every 7 days overlap, so the same session lands in training rows and in an
early-stopping ``eval_set`` cut from the same windows. ``eval_gap=`` returns a (train, eval) pair whose time ranges
do not overlap.
"""
from __future__ import annotations

import itertools
import numbers
import warnings
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_ORDER = {"view": 0, "addtocart": 1, "transaction": 2}
SELECT = ("first_session_after", "all")


# ----------------------------------------------------------------------------- time helpers
def _is_dt(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def _dur(x, times: pd.Series, time_unit: str | None, what: str):
    """Duration ``x`` in the units of ``times``: a Timedelta for datetime times, a number for numeric ones."""
    if x is None:
        return None
    num = isinstance(x, numbers.Real) and not isinstance(x, bool)
    if _is_dt(times):
        if num:
            raise TypeError(f"{what}={x!r}: datetime time column needs a duration such as '30min' or '21D'")
        return pd.Timedelta(x)
    if num:
        return x
    if time_unit is None:
        raise ValueError(f"{what}={x!r}: numeric time column; pass {what} as a number in its unit or set "
                         f"time_unit= ('ms', 's')")
    return pd.Timedelta(x) / pd.Timedelta(1, unit=time_unit)


def _point(x, times: pd.Series):
    return pd.Timestamp(x) if _is_dt(times) else x


def _seconds(d: pd.Series, times: pd.Series, time_unit: str | None) -> pd.Series:
    """Time differences in seconds (raw units for a numeric time column without ``time_unit``)."""
    if _is_dt(times):
        return d.dt.total_seconds()
    if time_unit is None:
        return d.astype(float)
    return d.astype(float) * (pd.Timedelta(1, unit=time_unit) / pd.Timedelta(1, unit="s"))


def _check_cols(df: pd.DataFrame, cols: Iterable[str | None], what: str) -> None:
    miss = [c for c in cols if c is not None and c not in df.columns]
    if miss:
        raise KeyError(f"{what}: columns {miss} not in the frame ({list(df.columns)[:12]})")


# ----------------------------------------------------------------------------- sessionize
def sessionize(events: pd.DataFrame, user_col: str, time_col: str, gap: Any = "30min",
               order: Mapping[Any, int] | None = None, kind_col: str | None = None, *,
               time_unit: str | None = None) -> pd.DataFrame:
    """Split each user's events into sessions: a gap > ``gap`` (strict) between consecutive events starts a new one.

    Returns a copy sorted by user, time, kind order (``order`` ranks ``kind_col`` values for events with the same
    time: view < addtocart < transaction by default; unknown kinds go last) and input position, index reset, with
    ``session_id`` (int, 0.. in that order), ``session_start`` (time of its first event) and ``seq`` (0.. within the
    session). Without ``kind_col`` ties keep their input order.
    """
    _check_cols(events, [user_col, time_col, kind_col], "sessionize")
    e = events.copy()
    if e[time_col].isna().any():
        raise ValueError(f"sessionize: {int(e[time_col].isna().sum())} rows without a time")
    g = _dur(gap, e[time_col], time_unit, "gap")
    keys = [user_col, time_col]
    e["_pos"] = np.arange(len(e))
    if kind_col is not None:
        order = DEFAULT_ORDER if order is None else order
        last = max(order.values(), default=-1) + 1
        e["_rank"] = e[kind_col].map(order).fillna(last)
        keys.append("_rank")
    e = e.sort_values(keys + ["_pos"], kind="mergesort").reset_index(drop=True)
    u, t = e[user_col], e[time_col]
    new = (u != u.shift()) | (t.diff() > g)
    new.iloc[:1] = True
    sid = new.cumsum().to_numpy(np.int64) - 1
    e["session_id"] = sid
    e["session_start"] = t.to_numpy()[np.flatnonzero(new.to_numpy())][sid]
    e["seq"] = e.groupby("session_id", sort=False).cumcount().to_numpy(np.int64)
    return e.drop(columns=[c for c in ("_pos", "_rank") if c in e.columns])


# ----------------------------------------------------------------------------- cutoff windows
def cutoff_windows(events: pd.DataFrame, cutoffs: Sequence, horizon: Any, *,
                   label: str | Callable[[pd.DataFrame], Any], select: str = "first_session_after",
                   prefix_len: int = 2, prefix_kinds: Sequence | None = None, max_end_offset: Any = None,
                   eval_gap: Any = None, truncate: bool = True,
                   features: Callable[[pd.DataFrame, pd.DataFrame, Any], pd.DataFrame] | None = None,
                   user_col: str = "user", time_col: str = "time", kind_col: str | None = None,
                   item_col: str | None = None, time_unit: str | None = None):
    """Rows that mimic a test period, one per (cutoff, unit), from sessionized ``events`` (``sessionize`` first).

    For each cutoff T the window is [T, T + horizon). ``select="first_session_after"``: each user's first session
    starting in the window (one row per (cutoff, user)); ``"all"``: every session starting in it. ``max_end_offset``
    keeps sessions starting at or before T + horizon - offset (a test set cut "at least 3 h before the log ends").
    The user's selected session is then kept only when its first ``prefix_len`` events exist (and, with
    ``prefix_kinds``, are all of those kinds); the user does not fall through to a later session. With ``truncate``
    (default) events at or after T + horizon are not seen, as in a test log that ends there. A window ending more
    than horizon / 10 after the last event warns (censored labels).

    Columns: cutoff, ``user_col``, session_id, session_start, ``<time_col>k`` / ``<kind_col>k`` / ``<item_col>k``
    for prefix event k, prefix_span (last - first prefix time, seconds), prefix_items (distinct items), n_rest
    (events after the prefix), label_end (last event time seen) and y. ``label`` is a kind ("transaction": y = 1 if
    the rest of the session has one) or a callable on the rest (events with seq >= prefix_len, possibly empty).
    ``features(history, rows, T)`` adds columns; ``history`` holds only events with time < T.

    ``eval_gap`` (a duration, "0D" = adjacent): returns ``(train, eval)``. eval = rows of the latest cutoff Te;
    train = rows of cutoffs with T + horizon + eval_gap <= Te, so no training row sees an event at or after Te
    (overlapping windows otherwise put the same sessions in an early-stopping eval set, D13 #8).
    ``train.attrs["dropped_cutoffs"]`` lists the cutoffs left out.
    """
    miss = {"session_id", "session_start", "seq"} - set(events.columns)
    if miss:
        raise ValueError(f"cutoff_windows: {sorted(miss)} missing; run sessionize(events, ...) first")
    _check_cols(events, [user_col, time_col, kind_col, item_col], "cutoff_windows")
    if select not in SELECT:
        raise ValueError(f"select={select!r}: one of {SELECT}")
    if prefix_len < 1:
        raise ValueError("prefix_len must be >= 1")
    if kind_col is None and (isinstance(label, str) or prefix_kinds is not None):
        raise ValueError("a kind label or prefix_kinds needs kind_col=")
    times = events[time_col]
    h = _dur(horizon, times, time_unit, "horizon")
    off = _dur(max_end_offset, times, time_unit, "max_end_offset")
    gap = _dur(eval_gap, times, time_unit, "eval_gap")
    cuts = sorted({_point(c, times) for c in cutoffs})
    if not cuts:
        raise ValueError("cutoff_windows: no cutoffs")
    if gap is not None:
        if not truncate:
            raise ValueError("eval_gap needs truncate=True: untruncated training labels can reach into the eval "
                             "window")
        all_cuts, te = cuts, cuts[-1]
        keep = [T for T in cuts[:-1] if T + h + gap <= te]
        if not keep:
            raise ValueError(f"eval_gap: no cutoff ends {eval_gap} before the eval cutoff {te}; add earlier cutoffs")
        cuts = keep + [te]
    # a window running past the log's last event censors its labels (late sessions lose their rest, y biased low)
    tmax = times.max()
    late = [T for T in cuts if T + h - tmax > h / 10]
    if late:
        warnings.warn(f"cutoff_windows: {len(late)} window(s) end more than a tenth of the horizon after the last "
                      f"event ({tmax}), e.g. cutoff {late[0]}: their labels are censored; use earlier cutoffs",
                      stacklevel=2)

    heads =events.loc[events["seq"] == 0, ["session_id", user_col, "session_start"]]
    heads = heads.sort_values(["session_start", "session_id"], kind="mergesort")
    L = prefix_len
    frames = []
    for T in cuts:
        end = T + h
        c = heads[(heads["session_start"] >= T) & (heads["session_start"] < end)]
        if off is not None:
            c = c[c["session_start"] <= end - off]
        if select == "first_session_after":
            c = c.drop_duplicates(user_col)
        sub = events[events["session_id"].isin(c["session_id"])]
        if truncate:
            sub = sub[sub[time_col] < end]
        sub = sub.sort_values(["session_id", "seq"], kind="mergesort")
        pre = sub[sub["seq"] < L]
        n = pre.groupby("session_id").size()
        ok = n.index[n == L]
        if prefix_kinds is not None:
            ok = ok.difference(pre.loc[~pre[kind_col].isin(list(prefix_kinds)), "session_id"].unique())
        sub = sub[sub["session_id"].isin(ok)]
        pre, rest = sub[sub["seq"] < L], sub[sub["seq"] >= L]
        sids = pre["session_id"].to_numpy()[::L]
        rows = pd.DataFrame({"cutoff": [T] * len(sids), user_col: pre[user_col].to_numpy()[::L], "session_id": sids,
                             "session_start": pre["session_start"].to_numpy()[::L]})
        for k in range(L):
            for col in (time_col, kind_col, item_col):
                if col is not None:
                    rows[f"{col}{k}"] = pre[col].to_numpy()[k::L]
        rows["prefix_span"] = _seconds(rows[f"{time_col}{L - 1}"] - rows[f"{time_col}0"], times, time_unit)
        if item_col is not None:
            rows["prefix_items"] = pre.groupby("session_id")[item_col].nunique().reindex(sids).to_numpy()
        rows["n_rest"] = rest.groupby("session_id").size().reindex(sids, fill_value=0).to_numpy()
        rows["label_end"] = sub.groupby("session_id")[time_col].max().reindex(sids).to_numpy()
        if isinstance(label, str):
            hit = rest.loc[rest[kind_col] == label, "session_id"].unique()
            rows["y"] = np.isin(sids, hit).astype(np.int64)
        else:
            got = {s: label(g) for s, g in rest.groupby("session_id", sort=False)}
            empty = None
            if len(got) < len(sids):
                empty = label(rest.iloc[0:0])
            rows["y"] = [got.get(s, empty) for s in sids]
        if features is not None:
            hist = events[events[time_col] < T]
            assert not len(hist) or hist[time_col].max() < T, "history reaches the cutoff"
            extra = features(hist, rows, T)
            if not isinstance(extra, pd.DataFrame) or not extra.index.equals(rows.index):
                raise ValueError("features(history, rows, cutoff) must return a DataFrame with the rows' index")
            new = [col for col in extra.columns if col not in rows.columns]
            rows = pd.concat([rows, extra[new]], axis=1)
        frames.append(rows)
    out = pd.concat([f for f in frames if len(f)] or frames[:1], ignore_index=True)   # empty frames: dtype drift
    if gap is None:
        return out
    te = cuts[-1]
    train = out[out["cutoff"] < te].reset_index(drop=True)
    ev = out[out["cutoff"] == te].reset_index(drop=True)
    assert not set(train["session_id"]) & set(ev["session_id"]), "a session is in train and eval"
    assert not len(train) or train["label_end"].max() < te, "a training row sees the eval window"
    train.attrs["dropped_cutoffs"] = [c for c in all_cuts if c not in set(cuts)]
    return train, ev


# ----------------------------------------------------------------------------- history features
def _kind_counts(h: pd.DataFrame, keys: list[str], kinds: Sequence, kind_col: str) -> pd.DataFrame:
    if not len(h):
        idx = pd.MultiIndex.from_tuples([], names=keys) if len(keys) > 1 else pd.Index([], name=keys[0])
        return pd.DataFrame(0, index=idx, columns=list(kinds), dtype=np.int64)
    c = h.groupby(keys + [kind_col], sort=True).size().unstack(kind_col, fill_value=0)
    return c.reindex(columns=list(kinds), fill_value=0).astype(np.int64)


def _entity_block(h, keys, name, kinds, kind_col, time_col, wins, smooth, at, day, others):
    """(frame indexed by ``keys``, {column: fill value for an entity with no history})."""
    cols, fill = {}, {}
    base = kinds[0]
    for sfx, hw in wins:
        c = _kind_counts(hw, keys, kinds, kind_col)
        tot = c.sum()
        for k in kinds:
            cols[f"{name}_n_{k}{sfx}"] = c[k]
            fill[f"{name}_n_{k}{sfx}"] = 0
        for k in kinds[1:]:
            prior = float(tot[k] / tot[base]) if tot[base] > 0 else 0.0
            cols[f"{name}_rate_{k}{sfx}"] = (c[k] + smooth * prior) / (c[base] + smooth)
            fill[f"{name}_rate_{k}{sfx}"] = prior
    g = h.groupby(keys, sort=True)
    cols[f"{name}_days_since_last"] = (at - g[time_col].max()) / day
    cols[f"{name}_days_since_first"] = (at - g[time_col].min()) / day
    for o in others:
        cols[f"{name}_n_unique_{o}"] = g[o].nunique()
        fill[f"{name}_n_unique_{o}"] = 0
    if "session_id" in h.columns:
        cols[f"{name}_n_sessions"] = g["session_id"].nunique()
        fill[f"{name}_n_sessions"] = 0
    out = pd.DataFrame(cols)
    for col, v in fill.items():
        out[col] = out[col].fillna(v)
        if v == 0 and "_rate_" not in col:
            out[col] = out[col].astype(np.int64)
    assert not len(out) or (out[f"{name}_days_since_last"] > 0).all(), "an event at or after `at` was used"
    return out, fill


def history_features(events: pd.DataFrame, at, rows: pd.DataFrame | None = None, *,
                     on: Mapping[str, str | Sequence[str]] | None = None,
                     entity: Sequence[str] = ("user", "item"), kinds: Sequence | None = None,
                     windows: Sequence = (None, "28D"), pair: bool = True, smooth: float = 2.0,
                     kind_col: str = "event", time_col: str = "time", time_unit: str | None = None):
    """Per-entity statistics from events strictly before ``at`` (asserted): counts per kind, smoothed conversion
    rates, days since the last / first event, distinct counts of the other entities, sessions, and the same counts
    for every entity pair (user x item history).

    For each window (None = all history, "28D" = the 28 days before ``at``; suffix ``_28D``):
    ``<e>_n_<kind>`` and, against the first kind (``kinds[0]``, e.g. view), ``<e>_rate_<kind>`` =
    (n_kind + smooth * prior) / (n_base + smooth) with prior = n_kind / n_base over all entities in that window
    (D13 used (t + 0.02) / (v + 2)). ``kinds`` defaults to the kinds seen before ``at``; pass it to keep the columns
    fixed across cutoffs.

    Without ``rows``: a dict {entity: frame indexed by it, (a, b): frame indexed by the pair}. With ``rows``: a copy of
    ``rows`` with the features joined; ``on`` maps an entity to the rows column(s) that hold it (default: the same
    name; only the listed entities are joined). With several columns per entity the feature names get ``_<column>``
    (``item_n_view_item0``). Unseen entities: counts 0, rates = prior, days NaN, ``<e>_seen`` 0.
    """
    _check_cols(events, list(entity) + [kind_col, time_col], "history_features")
    times = events[time_col]
    at = _point(at, times)
    if _is_dt(times):
        day = pd.Timedelta("1D")
    elif time_unit is not None:
        day = pd.Timedelta("1D") / pd.Timedelta(1, unit=time_unit)
    else:
        raise ValueError("history_features: numeric time column; set time_unit= ('ms', 's') for the day features")
    h = events[times < at]
    assert not len(h) or h[time_col].max() < at, "history reaches `at`"
    if kinds is None:
        seen = list(pd.unique(h[kind_col]))
        kinds = sorted(seen, key=lambda k: (DEFAULT_ORDER.get(k, len(DEFAULT_ORDER)), str(k)))
    kinds = list(kinds)
    if not kinds:
        raise ValueError("history_features: no kinds (empty history); pass kinds=")
    wins = []
    for w in windows:
        if w is None:
            wins.append(("", h))
        else:
            wins.append((f"_{w}", h[h[time_col] >= at - _dur(w, times, time_unit, "windows")]))
    ent = list(entity)
    blocks: dict[Any, tuple[pd.DataFrame, dict]] = {}
    for e in ent:
        blocks[e] = _entity_block(h, [e], e, kinds, kind_col, time_col, wins, smooth, at, day,
                                  [o for o in ent if o != e])
    if pair:
        for a, b in itertools.combinations(ent, 2):
            blocks[(a, b)] = _entity_block(h, [a, b], f"{a}_{b}", kinds, kind_col, time_col, wins, smooth, at,
                                           day, [])
    if rows is None:
        return {k: v[0] for k, v in blocks.items()}

    def cols_of(e):
        c = on.get(e) if on is not None else e
        return [] if c is None else ([c] if isinstance(c, str) else list(c))
    parts = []
    for e in ent:
        cs = cols_of(e)
        _check_cols(rows, cs, "history_features rows")
        frame, fill = blocks[e]
        for c in cs:
            sfx = f"_{c}" if len(cs) > 1 else ""
            parts.append(_lookup(frame, fill, [rows[c].to_numpy()], sfx, rows.index, f"{e}_seen{sfx}"))
    if pair:
        for a, b in itertools.combinations(ent, 2):
            ca, cb = cols_of(a), cols_of(b)
            frame, fill = blocks[(a, b)]
            for x, y in itertools.product(ca, cb):
                sfx = "".join(f"_{c}" for c, lst in ((x, ca), (y, cb)) if len(lst) > 1)
                parts.append(_lookup(frame, fill, [rows[x].to_numpy(), rows[y].to_numpy()], sfx, rows.index, None))
    return pd.concat([rows] + parts, axis=1)


def _lookup(frame, fill, keys, sfx, index, seen_col):
    idx = keys[0] if len(keys) == 1 else pd.MultiIndex.from_arrays(keys)
    x = frame.reindex(idx)
    seen = x.notna().any(axis=1).to_numpy() if len(x.columns) else np.zeros(len(x), bool)
    for col, v in fill.items():
        x[col] = x[col].fillna(v)
        if v == 0 and "_rate_" not in col:
            x[col] = x[col].astype(np.int64)
    x.index = index
    x.columns = [f"{c}{sfx}" for c in x.columns]
    if seen_col is not None and len(frame) and len(x) and not seen.any():
        warnings.warn(f"history_features: {seen_col} is 0 for every row (rows key dtype {keys[0].dtype} vs history "
                      f"{frame.index.dtype}?); every feature is the unseen fill", stacklevel=3)
    if seen_col is not None:
        x[seen_col] = seen.astype(np.int64)
    return x


# ----------------------------------------------------------------------------- rate rescaling
def _rescale(x: np.ndarray, r: float, method: str) -> tuple[np.ndarray, bool]:
    if method == "scale":
        m = x.mean()
        if m == 0:
            return np.full_like(x, r), False
        q = x * (r / m)
        if r <= 1 and (q > 1).any():
            return np.minimum(q, 1.0), True
        return q, False
    if not 0 <= r <= 1:
        raise ValueError(f"method='logit' needs a target rate in [0, 1], got {r}")
    if r in (0.0, 1.0):
        return np.full_like(x, r), False
    z = np.log(np.clip(x, 1e-12, 1 - 1e-12)) - np.log1p(-np.clip(x, 1e-12, 1 - 1e-12))
    lo, hi = -60.0, 60.0
    for _ in range(200):                                   # mean(sigmoid(z + b)) is increasing in b
        b = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(z + b)))) < r:
            lo = b
        else:
            hi = b
    return 1 / (1 + np.exp(-(z + (lo + hi) / 2))), False


def rescale_to_rates(p, segment=None, target=None, *, method: str = "scale", default: float | None = None):
    """Move each segment's mean prediction to a target rate while keeping the ranking within the segment.

    ``target``: {segment: rate} (dict / Series, e.g. training rates ``train.groupby("segment").y.mean()``: label-free
    on the scored set) or, with ``segment=None`` (the total version), one rate. ``method="scale"`` multiplies
    (p * target / mean(p), D13 K arms; values that would pass 1 are clipped with a warning, so the mean then falls
    short); ``"logit"`` shifts log-odds by one constant per segment, staying inside (0, 1). Segments with no target
    use ``default`` or stay unchanged (warning). ``p`` and ``segment`` both Series: matched by index. Returns an
    array, or a Series with ``p``'s index.
    """
    if method not in ("scale", "logit"):
        raise ValueError(f"method={method!r}: 'scale' or 'logit'")
    a = np.asarray(p, dtype=float).copy()
    if np.isnan(a).any() or (a < 0).any():
        raise ValueError("rescale_to_rates: p must be non-negative with no NaN")
    scalar = isinstance(target, numbers.Real) and not isinstance(target, bool)
    if segment is None:
        if not scalar:
            raise TypeError("segment=None (total version) needs one target rate")
        codes, uniq, tmap = np.zeros(len(a), int), [None], {None: float(target)}
    else:
        if scalar or target is None:
            raise TypeError("with segment=, target is {segment: rate}; use segment=None for one total rate")
        if isinstance(p, pd.Series) and isinstance(segment, pd.Series) and not segment.index.equals(p.index):
            # two Series: match by index, never silently by position
            if not (segment.index.is_unique and len(segment) == len(p) and segment.index.isin(p.index).all()):
                raise ValueError("rescale_to_rates: p and segment are Series with different indexes")
            segment = segment.reindex(p.index)
        seg = pd.Series(np.asarray(segment, dtype=object))
        if len(seg) != len(a):
            raise ValueError(f"segment has {len(seg)} values, p has {len(a)}")
        codes, uniq = pd.factorize(seg, use_na_sentinel=False)
        tmap = target.to_dict() if isinstance(target, pd.Series) else dict(target)
    clipped, missing = False, []
    for i, g in enumerate(uniq):
        r = tmap.get(g, default)
        if r is None:
            missing.append(g)
            continue
        m = codes == i
        a[m], c = _rescale(a[m], float(r), method)
        clipped |= c
    if missing:
        warnings.warn(f"rescale_to_rates: no target for segments {missing[:10]}; left unchanged (set default=)",
                      stacklevel=2)
    if clipped:
        warnings.warn("rescale_to_rates: scaled values passed 1 and were clipped; the mean is below the target "
                      "(use method='logit')", stacklevel=2)
    return pd.Series(a, index=p.index, name=p.name) if isinstance(p, pd.Series) else a


__all__ = ["sessionize", "cutoff_windows", "history_features", "rescale_to_rates", "DEFAULT_ORDER"]

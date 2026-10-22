"""A/B pair tables for the evidence layer: pair table -> variant frame, and an A/B memorization probe.

    import simengine.evidence as ev
    f = ev.ab_pairs_frame(pairs, {"llm": "p_llm", "stack": "p_stack"}, y="y", group="test_id")
    rep = ev.evaluate(f, task="ab", ours="stack", raw="llm")         # pair_sign = pair accuracy, clustered on tests
    items = ev.ab_memorization_items(pairs, n=100, seed=0)          # real pairs + cross-test shuffled controls
    mem = ev.ab_memorization(llm, pairs, named_context, neutral_context, n=100)
    rep = ev.evaluate(f, task="ab", ours="stack", raw="llm", memorization=mem)

A pair table has one row per (A, B) pair: `text_a`, `text_b` (aliases `headline_a/b`, `metin_a/b`), `y` (1 = A won,
0 = B won, 0.5 or NaN = tie) and a test id; predictions are P(A wins) (bench upworthy, `methods.LLMDirectAB`,
`ABStack`). `evaluate(task="ab")` wants variant rows, so `ab_pairs_frame` writes each pair as two rows
(A: truth y, pred p; B: truth 1 - y, pred 1 - p). The item is the pair, the bootstrap cluster (`group`) is the test,
so pair_sign equals the direct pair accuracy (p = 0.5 counts as a half) and a test with several pairs moves together.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import ezber as _ezber

TEXT_ALIASES = {"headline_a": "text_a", "headline_b": "text_b", "metin_a": "text_a", "metin_b": "text_b"}
TIE_LABELS = ("tie", "draw", "berabere")
REAL_PROMPT = ("Context: {context}\nTwo variants were shown to real users in one A/B test. Which one got the HIGHER "
               "click rate?\nA: \"{a}\"\nB: \"{b}\"\nReply with the winning text exactly, nothing else.")


def _pred_map(pred_cols) -> dict[str, Any]:
    """str | [str] | {arm: column or array} -> {arm: source}; a 'pred_' / 'p_' prefix is dropped from the arm name."""
    if isinstance(pred_cols, str):
        pred_cols = [pred_cols]
    if isinstance(pred_cols, Mapping):
        return {str(k): v for k, v in pred_cols.items()}
    out = {}
    for c in pred_cols:
        arm = re.sub(r"^(pred_|tahmin_|p_)", "", str(c)) or str(c)
        if arm in out:
            raise ValueError(f"two prediction columns map to the arm name {arm!r}; pass a dict {{arm: column}}")
        out[arm] = c
    return out


def _truth(s: pd.Series, a_wins, tie_labels) -> np.ndarray:
    """y column -> P(A won) in {0, 0.5, 1} (NaN kept). Numeric y is used as is; labels need `a_wins`."""
    if a_wins is None and pd.api.types.is_numeric_dtype(s):
        v = s.to_numpy(float)
        bad = np.isfinite(v) & ~np.isin(v, (0.0, 0.5, 1.0))
        if bad.any():
            raise ValueError(f"y must be 1 (A won), 0 (B won) or 0.5 (tie); got {sorted(set(v[bad]))[:5]}. "
                             "For labels pass a_wins=<label meaning A won>")
        return v
    lab = "a" if a_wins is None else a_wins
    v = np.where(s.isna(), np.nan, np.where(s == lab, 1.0, 0.0))
    if isinstance(lab, str):
        v = np.where(s.astype(str).str.lower().isin([t.lower() for t in tie_labels]), 0.5, v)
    return v


def ab_pairs_frame(pairs: pd.DataFrame, pred_cols, y: str = "y", group: str | None = "test_id", *,
                   a_wins: Any = None, tie_labels: Sequence[str] = TIE_LABELS, keep: Sequence[str] = (),
                   drop_ties: bool = True) -> pd.DataFrame:
    """Per-pair table (y = A won, pred = P(A wins)) -> the variant-row EvalFrame `evaluate(task="ab")` expects.

    pred_cols: a column name, a list of names (arm = name without a `pred_` / `p_` prefix) or {arm: column or
        array}. Each becomes `pred_<arm>`: A row p, B row 1 - p. NaN predictions stay NaN on both rows.
    y: truth column. Numeric 1 / 0 / 0.5 (tie), or labels with `a_wins=` (the label meaning A won, e.g. "a" for
        the bench `winner` column; labels in `tie_labels` count as ties, anything else as B won).
    group: test id column (the bootstrap cluster); None makes every pair its own cluster.
    keep: extra pair columns copied to both rows (e.g. "split"). drop_ties: ties carry no pair_sign signal
        (evaluate skips them anyway); True drops them here so the row count matches the scored pairs.
    Output: unit (`<pair>:A|B`), item (the pair: one pair per item, so pairs of one test never cross-compare),
    group (the test, when given), arm ("A" | "B": the arm-majority baseline reads it), truth, pred_<arm>, and the `keep` columns."""
    d = pd.DataFrame(pairs).reset_index(drop=True)
    if y not in d:
        raise KeyError(f"pair table needs the truth column {y!r} (1 = A won)")
    if group is not None and group not in d:
        raise KeyError(f"pair table has no group column {group!r}; pass group=None for one cluster per pair")
    t = _truth(d[y], a_wins, tie_labels)
    arms = {}
    for arm, src in _pred_map(pred_cols).items():
        if isinstance(src, str):
            if src not in d:
                raise KeyError(f"prediction column {src!r} not in the pair table")
            v = d[src].to_numpy(float)
        else:
            v = np.asarray(src, float)
            if v.shape != (len(d),):
                raise ValueError(f"prediction {arm!r} has {v.shape[0] if v.ndim else 0} values for {len(d)} pairs")
        if np.any((v < 0) | (v > 1)):
            raise ValueError(f"prediction {arm!r} must be P(A wins) in [0, 1]")
        arms[arm] = v
    ok = np.isfinite(t) & ((t != 0.5) if drop_ties else True)
    idx = np.flatnonzero(ok)
    pid = (d[group].astype(str) + "#" + d.index.astype(str)).to_numpy() if group is not None else d.index.astype(str).to_numpy()
    n = len(idx)
    out = {"unit": np.concatenate([[f"{p}:A" for p in pid[idx]], [f"{p}:B" for p in pid[idx]]]),
           "item": np.tile(pid[idx], 2)}
    if group is not None:
        out["group"] = np.tile(d[group].to_numpy()[idx], 2)
    out["arm"] = np.repeat(["A", "B"], n)
    out["truth"] = np.concatenate([t[idx], 1 - t[idx]])
    for arm, v in arms.items():
        out[f"pred_{arm}"] = np.concatenate([v[idx], 1 - v[idx]])
    for c in keep:
        out[c] = np.tile(d[c].to_numpy()[idx], 2)
    f = pd.DataFrame(out)
    # A row then B row of each pair, in pair order
    return f.iloc[np.argsort(np.tile(np.arange(n), 2), kind="stable")].reset_index(drop=True)


# ------------------------------------------------------------------ memorization probe
def _texts_frame(pairs: pd.DataFrame, y: str, group: str | None) -> pd.DataFrame:
    d = pd.DataFrame(pairs)
    d = d.rename(columns={k: v for k, v in TEXT_ALIASES.items() if k in d and v not in d})
    for c in ("text_a", "text_b", y):
        if c not in d:
            raise KeyError(f"A/B pair table needs column {c!r} (text aliases: headline_a/b, metin_a/b)")
    if group is not None and group not in d:
        raise KeyError(f"pair table has no group column {group!r}; pass group=None")
    d = d.reset_index(drop=True)
    d["_test"] = d[group].astype(str) if group is not None else d.index.astype(str)
    d["_y"] = _truth(d[y], None if pd.api.types.is_numeric_dtype(d[y]) else "a", TIE_LABELS)
    d["text_a"], d["text_b"] = d.text_a.fillna("").astype(str), d.text_b.fillna("").astype(str)
    return d[np.isin(d["_y"], (0.0, 1.0))].reset_index(drop=True)


def _sample(d: pd.DataFrame, n: int | None, rng) -> pd.DataFrame:
    if n is None or n >= len(d):
        return d
    return d.iloc[np.sort(rng.choice(len(d), int(n), replace=False))].reset_index(drop=True)


def ab_memorization_items(pairs: pd.DataFrame, n: int | None = 100, seed: int = 0, *, y: str = "y",
                          group: str | None = "test_id", context: str = "an online A/B test of headlines") -> list[dict]:
    """Probe items for `memorization_audit`: n real pairs ('which text won?', expected = the real winner's text) plus
    n control pairs whose A and B texts come from two different tests (shuffled across tests), so no one can know
    a winner; the control's expected text is a coin flip. Real hit rate - control hit rate > 0 is the completion gap.
    Note: a model with good taste also beats the controls on real pairs (real pairs have a true winner), so read
    this gap next to `ab_memorization` (named - neutral context on the same real pairs), which cancels taste.
    Items: {"prompt", "expected", "control", "test", "pair"}; A/B order is randomised. n=None uses every pair."""
    rng = np.random.default_rng(seed)
    d = _sample(_texts_frame(pairs, y, group), n, rng)
    if not len(d):
        return []
    full = _texts_frame(pairs, y, group)
    pool = pd.concat([pd.DataFrame({"t": full.text_a, "test": full._test}),
                      pd.DataFrame({"t": full.text_b, "test": full._test})], ignore_index=True)
    if pool.test.nunique() < 2:
        raise ValueError("control pairs need texts from at least two different tests")
    items = []
    for k, r in d.iterrows():
        win = r.text_a if r._y == 1 else r.text_b
        a, b = (r.text_a, r.text_b) if rng.random() < 0.5 else (r.text_b, r.text_a)
        items.append({"prompt": REAL_PROMPT.format(context=context, a=a, b=b), "expected": win, "control": False,
                      "test": r._test, "pair": int(k)})
        # control: two different texts from two different tests (no real winner between them)
        for _ in range(100):
            i, j = rng.integers(0, len(pool), 2)
            if pool.test.iat[i] != pool.test.iat[j] and pool.t.iat[i] != pool.t.iat[j]:
                break
        ca, cb = pool.t.iat[i], pool.t.iat[j]
        items.append({"prompt": REAL_PROMPT.format(context=context, a=ca, b=cb),
                      "expected": ca if rng.random() < 0.5 else cb, "control": True,
                      "test": f"{pool.test.iat[i]}|{pool.test.iat[j]}", "pair": int(k)})
    return items


def _hit(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pair hit: 1 when P(A) is on the winner's side of 0.5, 0.5 when P(A) = 0.5."""
    return np.where(np.isclose(p, 0.5), 0.5, ((p > 0.5) == (y == 1)).astype(float))


def _cluster_ci(vals: dict[str, np.ndarray], cl: np.ndarray, B: int, seed: int) -> dict[str, tuple[float, float]]:
    """Paired cluster bootstrap (same test draws for every series) of each series' mean; 95 % percentile CI."""
    codes, inv = np.unique(cl, return_inverse=True)
    G = len(codes)
    cnt = np.bincount(inv, minlength=G).astype(float)
    sums = {k: np.bincount(inv, v, minlength=G) for k, v in vals.items()}
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(B)]).astype(float)  # B x G
    den = W @ cnt
    return {k: tuple(float(x) for x in np.percentile((W @ s) / den, [2.5, 97.5])) for k, s in sums.items()}


def _paraphrase_llm(llm, texts: list[str], batch: int, max_tokens: int) -> list[str]:
    from ..methods._util import parse_json
    out = list(texts)
    chunks = [list(range(i, min(i + batch, len(texts)))) for i in range(0, len(texts), batch)]
    prompts = [("You rewrite text. Reply with JSON only.",
                "Rewrite each headline with the same meaning in different words (no new facts).\n"
                + "\n".join(f'{j}. "{texts[i]}"' for j, i in enumerate(ch))
                + f"\n\nReturn JSON list: [{{\"i\": 0, \"text\": \"...\"}}, ...] for all {len(ch)} headlines.")
               for ch in chunks]
    for ch, r in zip(chunks, llm.map(prompts, max_tokens=max_tokens)):
        try:
            got = parse_json(r.text)
        except (ValueError, TypeError):
            continue
        for e in got if isinstance(got, list) else []:
            try:
                j, t = int(e["i"]), str(e["text"]).strip()
            except (KeyError, ValueError, TypeError):
                continue
            if 0 <= j < len(ch) and t:
                out[ch[j]] = t
    return out


def ab_memorization(llm, pairs: pd.DataFrame, named_context: str, neutral_context: str, *, n: int | None = 100,
                    paraphrase=False, y: str = "y", group: str | None = "test_id", margin: float = 0.0,
                    seed: int = 0, B: int = 1000, batch: int = 15, reverse: bool = True, max_tokens: int = 1024,
                    max_calls: int = _ezber.EN_FAZLA, **llm_kw: Any) -> dict:
    """A/B memorization probe: the same real pairs asked `LLMDirectAB`-style twice, once with the study / source
    NAMED in the context (e.g. "Upworthy headline tests, Matias et al. 2021 archive") and once NEUTRAL ("A/B test of
    two headlines"). Taste for good headlines helps both arms equally; if naming the source lifts pair accuracy,
    the model may remember the published winners. deger = accuracy(named) - accuracy(neutral), paired cluster
    bootstrap over tests (95 % CI); durum "uyari" when the lower bound > `margin` (the threshold), else "tamam";
    "atlandi" below 5 pairs.

    paraphrase: False (default) | True (the LLM rewrites every text first, batched) | a str -> str callable. When
        set, a third arm asks the named context on the rewritten texts; memorised wording is lost there, so
        accuracy(named) - accuracy(named, paraphrased) is a second gap (`paraphrase_gap`, CI) and also raises
        "uyari" when its lower bound > margin.
    n: pairs drawn at random (seed); None uses all. max_calls: checked before any call (planned calls; cached
        replies are counted). llm_kw go to `llm.map` (e.g. model settings of the backend).
    Returns the dict `evaluate(memorization=...)` reads (durum, ayrinti, deger, alt, ust) plus esik, accuracy_named,
    accuracy_neutral (+ _ci), coverage_named / coverage_neutral, n_pairs, n_tests, calls, yontem."""
    from ..methods.ab import LLMDirectAB
    rng = np.random.default_rng(seed)
    d = _sample(_texts_frame(pairs, y, group), n, rng)
    per_arm = 0 if not len(d) else -(-len(d) // batch) * (2 if reverse else 1)
    n_arms = 3 if paraphrase else 2
    planned = per_arm * n_arms + (-(-2 * len(d) // batch) if paraphrase is True else 0)
    if planned > max_calls:
        raise ValueError(f"ab_memorization needs {planned} calls ({n_arms} arms x {per_arm}"
                         + (" + paraphrase" if paraphrase is True else "") + f") > max_calls {max_calls}; "
                         f"pass max_calls={planned} or a smaller n. No call was made.")
    base = {"yontem": "A/B adlı vs nötr bağlam (çift isabeti)", "esik": float(margin), "n_pairs": len(d),
            "n_tests": int(d._test.nunique()) if len(d) else 0, "calls": planned}
    if len(d) < 5:
        return {**base, "durum": "atlandi", "deger": None, "alt": None, "ust": None,
                "ayrinti": f"A/B ezber sondası: yalnız {len(d)} çift (berabere olmayan); en az 5 gerekli"}
    yy = d._y.to_numpy()
    arms = {}
    cov = {}
    for name, ctx, frame in (("named", named_context, d), ("neutral", neutral_context, d)):
        m = LLMDirectAB(llm, context=ctx, batch=batch, reverse=reverse, max_tokens=max_tokens, **llm_kw)
        arms[name] = _hit(m.predict(frame[["text_a", "text_b"]]), yy)
        cov[name] = m.coverage_
    if paraphrase:
        texts = list(d.text_a) + list(d.text_b)
        rew = (_paraphrase_llm(llm, texts, batch, max_tokens) if paraphrase is True
               else [str(paraphrase(t)) for t in texts])
        pf = pd.DataFrame({"text_a": rew[:len(d)], "text_b": rew[len(d):]})
        m = LLMDirectAB(llm, context=named_context, batch=batch, reverse=reverse, max_tokens=max_tokens, **llm_kw)
        arms["paraphrase"] = _hit(m.predict(pf), yy)
        cov["paraphrase"] = m.coverage_
    series = {f"acc_{k}": v for k, v in arms.items()}
    series["gap"] = arms["named"] - arms["neutral"]
    if paraphrase:
        series["pgap"] = arms["named"] - arms["paraphrase"]
    ci = _cluster_ci(series, d._test.to_numpy(), B, seed)
    gap = float(series["gap"].mean())
    alt, ust = ci["gap"]
    uyari = alt > margin
    f = lambda x: f"{x:+.3f}"  # noqa: E731
    ayrinti = (f"A/B ezber sondası ({len(d)} çift, {base['n_tests']} test): adlı bağlamda çift isabeti "
               f"{arms['named'].mean():.3f}, nötr {arms['neutral'].mean():.3f}; adlı − nötr {f(gap)} "
               f"[{f(alt)}, {f(ust)}] (test kümeli eşleşik bootstrap), eşik {f(margin)}")
    out = {**base, "deger": gap, "alt": alt, "ust": ust,
           "accuracy_named": float(arms["named"].mean()), "accuracy_named_ci": list(ci["acc_named"]),
           "accuracy_neutral": float(arms["neutral"].mean()), "accuracy_neutral_ci": list(ci["acc_neutral"]),
           "coverage_named": cov["named"], "coverage_neutral": cov["neutral"]}
    if paraphrase:
        pg, (pa, pu) = float(series["pgap"].mean()), ci["pgap"]
        uyari = uyari or pa > margin
        ayrinti += (f"; yeniden yazılmış metinde adlı isabet {arms['paraphrase'].mean():.3f}, özgün − yeniden yazılmış "
                    f"{f(pg)} [{f(pa)}, {f(pu)}]")
        out.update({"accuracy_paraphrase": float(arms["paraphrase"].mean()),
                    "accuracy_paraphrase_ci": list(ci["acc_paraphrase"]), "coverage_paraphrase": cov["paraphrase"],
                    "paraphrase_gap": pg, "paraphrase_gap_ci": [pa, pu]})
    out["durum"] = "uyari" if uyari else "tamam"
    out["ayrinti"] = ayrinti + (" · ezber şüphesi" if uyari else " · ezber işareti yok")
    return out

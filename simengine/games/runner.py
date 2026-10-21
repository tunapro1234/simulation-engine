"""`play`: personas x games x conditions x runs through any `simengine.types.LLM`, into the b01 tidy schema.

Ported from edsl-test `benchmark/beebench/b01/run.py`: the CSV schema, the resume key, the cost gate ($5 default,
refuses without confirmation) and the retry of missing decisions. The EDSL wiring is rewritten onto
`LLM.map`: the persona goes in as `system`, the game text plus a JSON answer line as `user`. When a game has a
belief layer, the belief is asked first and the decision prompt shows the stated belief (b01 reviewer finding #5,
EDSL targeted memory).

A missing answer is never a value: an unparseable decision stays None with valid=False (b01 before commit 15b9531
turned None into $0; that trap is closed here). An out-of-range number is kept and flagged valid=False.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from . import library as L
from ._answer import ask_many, instruction

COLUMNS = ["model", "persona_method", "agent_id", "game", "role", "condition", "wording_id", "run", "decision",
           "belief", "valid", "raw_len", "cost_usd"]
EXTRA_COLUMNS = ["seed", "max_tokens", "persona_params", "belief_valid", "attempts", "error", "holdout"]
KEY = ["model", "persona_method", "agent_id", "game", "condition", "wording_id", "run"]
CHARS_PER_TOKEN = 4      # rough prompt-size estimate for the cost gate


class CostGateError(RuntimeError):
    """The estimated cost is above `budget_gate_usd` and `confirm=True` was not given."""

    def __init__(self, estimate: dict):
        self.estimate = estimate
        super().__init__(f"estimated ${estimate['usd']:.2f} for {estimate['calls']} calls exceeds the gate "
                         f"${estimate['gate_usd']:.2f}; pass confirm=True to run anyway")


# ---------------------------------------------------------------- personas

def as_agents(personas: Any, method: str = "custom") -> list[dict]:
    """Normalise personas to [{agent_id, persona_method, text, params}].

    Accepted: {method: {agent_id: text | {"text", "params"}}} (the b01 agent bank), {agent_id: text} (one bank,
    method=`method`), a list of dicts with agent_id/id, persona_method/method, text/persona/system and
    params/traits/dials, or a list of strings. An empty text means no persona (the bare-model baseline)."""
    out: list[dict] = []

    def one(aid, meth, v):
        if isinstance(v, Mapping):
            text = v.get("text", v.get("persona", v.get("system", "")))
            prm = v.get("params", v.get("traits", v.get("dials", {})))
        else:
            text, prm = v, {}
        out.append({"agent_id": str(aid), "persona_method": str(meth), "text": "" if text is None else str(text),
                    "params": dict(prm or {})})

    if isinstance(personas, Mapping):
        vals = list(personas.values())
        if vals and all(isinstance(v, Mapping) and not ({"text", "persona", "system"} & set(v)) for v in vals):
            for meth, bank in personas.items():
                for aid, v in bank.items():
                    one(aid, meth, v)
        else:
            for aid, v in personas.items():
                one(aid, method, v)
    else:
        for i, v in enumerate(personas):
            if isinstance(v, Mapping):
                aid = v.get("agent_id", v.get("id", f"{method}-{i:03d}"))
                one(aid, v.get("persona_method", v.get("method", method)), v)
            else:
                one(f"{method}-{i:03d}", method, v)
    ids = [(a["persona_method"], a["agent_id"]) for a in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate (persona_method, agent_id) in personas")
    return out


def _games(games: Any) -> list[dict]:
    if games is None or games == "all":
        names = L.list_games()
    elif games == "calibration":
        names = L.list_games(holdout=False)
    elif games == "holdout":
        names = L.list_games(holdout=True)
    elif isinstance(games, (str, Mapping)):
        names = [games]
    else:
        names = list(games)
    out = []
    for g in names:
        gd = L.get_game(g)
        probs = L.check_game(gd)
        if probs:
            raise ValueError("invalid GAME dict: " + "; ".join(probs))
        out.append(gd)
    return out


# ---------------------------------------------------------------- planning and cost

def _plan(agents, games, runs, wording_ids, model, seed):
    cells = []
    for g in games:
        for a in agents:
            for c in g["conditions"]:
                for w in wording_ids:
                    for r in range(runs):
                        key = (str(model), a["persona_method"], a["agent_id"], g["game"], str(c["id"]), str(w),
                               str(r))
                        cells.append({"key": key, "game": g, "agent": a, "cond": c, "wording_id": w, "run": r,
                                      "sample": int(seed) * 1000 + r})
    return cells


def _is_fake(llm) -> bool:
    try:
        from ..llm import FakeLLM
        return isinstance(llm, FakeLLM)
    except Exception:   # pragma: no cover
        return False


def estimate_cost(llm, personas: Any, games: Any = "all", *, runs: int = 3, belief: bool = True,
                  max_tokens: int = 1024, wording_ids: Sequence[int] = (0,), est_cost_per_call: float | None = None,
                  done: Iterable[tuple] = (), seed: int = 0, prices: dict | None = None,
                  budget_gate_usd: float = 5.0) -> dict:
    """Upper-bound cost of the first attempt of every missing cell: prompt chars / 4 input tokens and
    `max_tokens` output tokens at list price (`simengine.llm.price_for`), or `est_cost_per_call` per call when
    given. FakeLLM costs 0 unless est_cost_per_call is given. An unpriced real model raises (an unknown price must not silently disable the gate)."""
    agents = as_agents(personas)
    gl = _games(games)
    model = getattr(llm, "model", "unknown")
    done = {tuple(map(str, k)) for k in done}
    cells = [c for c in _plan(agents, gl, runs, list(wording_ids), model, seed) if c["key"] not in done]
    calls, in_chars = 0, 0
    for c in cells:
        g, cond, w = c["game"], c["cond"], c["wording_id"]
        text = L.render(g, cond, w)
        in_chars += len(c["agent"]["text"]) + len(text) + 120
        calls += 1
        if belief and g.get("belief"):
            in_chars += len(c["agent"]["text"]) + len(L.render_belief(g, cond, w) or "") + 120
            in_chars += len(text) // 2
            calls += 1
    if est_cost_per_call is not None:
        per_call = float(est_cost_per_call)
        usd = calls * per_call
    elif _is_fake(llm):
        usd, per_call = 0.0, 0.0
    else:
        from ..llm import cost_usd
        try:
            usd = cost_usd(model, input_tokens=in_chars // CHARS_PER_TOKEN, output_tokens=calls * int(max_tokens),
                           prices=prices)
        except KeyError as e:
            raise ValueError(f"no list price for model {model!r}: pass est_cost_per_call= or prices=") from e
        per_call = usd / calls if calls else 0.0
    return {"calls": calls, "cells": len(cells), "usd": float(usd), "usd_per_call": float(per_call),
            "gate_usd": float(budget_gate_usd), "model": model, "max_tokens": int(max_tokens)}


# ---------------------------------------------------------------- resume

def _load_existing(resume: Any, out: str | os.PathLike | None) -> pd.DataFrame | None:
    frames = []
    if resume is not None:
        frames.append(pd.read_csv(resume, dtype=str, keep_default_na=False) if isinstance(resume, (str, os.PathLike))
                      else resume.copy())
    if out is not None and os.path.exists(out) and os.path.getsize(out) > 0:
        frames.append(_read_results(out))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return df.drop_duplicates(subset=KEY, keep="first").reset_index(drop=True)


def _read_results(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"agent_id": str, "condition": str, "persona_method": str, "model": str})
    if "valid" in df.columns:
        df["valid"] = df["valid"].map(lambda v: str(v).strip().lower() in ("1", "true", "1.0"))
    return df


def read_results(path) -> pd.DataFrame:
    """Read a results CSV written by `play(out=...)` (valid -> bool, ids as strings)."""
    return _read_results(path)


def _keys(df: pd.DataFrame | None) -> set[tuple]:
    if df is None or df.empty:
        return set()
    return {tuple(str(v) for v in row) for row in df[KEY].itertuples(index=False, name=None)}


# ---------------------------------------------------------------- play

def _user(text: str, spec: dict) -> str:
    return f"{text}\n\n{instruction(spec)}"


def play(llm, personas: Any, games: Any = "all", *, runs: int = 3, seed: int = 0, belief: bool = True,
         max_tokens: int = 1024, budget_gate_usd: float = 5.0, confirm: bool = False,
         wording_ids: Sequence[int] = (0,), retries: int = 2, resume: Any = None,
         out: str | os.PathLike | None = None, est_cost_per_call: float | None = None, parallel: int = 4,
         method: str = "custom", prices: dict | None = None, **llm_kw: Any) -> pd.DataFrame:
    """Play every (persona, game, condition, wording, run) cell once and return the tidy results.

    personas: see `as_agents`. games: names, GAME dicts, "all", "calibration" or "holdout".
    runs: repeated samples per cell (passed to the backend as `sample=seed*1000+run`, so cached backends
    give independent draws). belief: ask the belief layer where the game has one.
    budget_gate_usd / confirm: the estimate (`estimate_cost`) must be under the gate, or confirm=True.
    retries: extra attempts for an unusable answer, each with the error text appended to the prompt.
    resume / out: rows already present (a DataFrame, a CSV path, and/or the `out` CSV) are skipped by the key
    (model, persona_method, agent_id, game, condition, wording_id, run); new rows are appended to `out`.
    Returns existing + new rows (no duplicate keys). Columns: COLUMNS + EXTRA_COLUMNS."""
    agents = as_agents(personas, method)
    gl = _games(games)
    model = str(getattr(llm, "model", "unknown"))
    existing = _load_existing(resume, out)
    done = _keys(existing)
    est = estimate_cost(llm, agents, gl, runs=runs, belief=belief, max_tokens=max_tokens, wording_ids=wording_ids,
                        est_cost_per_call=est_cost_per_call, done=done, seed=seed, prices=prices,
                        budget_gate_usd=budget_gate_usd)
    if est["usd"] > budget_gate_usd and not confirm:
        raise CostGateError(est)
    cells = [c for c in _plan(agents, gl, runs, list(wording_ids), model, seed) if c["key"] not in done]

    # stage 1: beliefs (asked BEFORE the decision)
    beliefs: dict[int, Any] = {}
    b_idx = [i for i, c in enumerate(cells) if belief and c["game"].get("belief")]
    if b_idx:
        items = []
        for i in b_idx:
            c = cells[i]
            spec = L.belief_spec(c["game"])
            items.append((c["agent"]["text"], _user(L.render_belief(c["game"], c["cond"], c["wording_id"]), spec),
                          spec, {"sample": c["sample"]}))
        for i, a in zip(b_idx, ask_many(llm, items, retries=retries, max_tokens=max_tokens, parallel=parallel,
                                        **llm_kw)):
            beliefs[i] = a

    # stage 2: decisions (the stated belief is shown, like EDSL targeted memory)
    items = []
    for i, c in enumerate(cells):
        spec = L.answer_spec(c["game"], c["cond"])
        text = L.render(c["game"], c["cond"], c["wording_id"])
        b = beliefs.get(i)
        if b is not None and b.valid:
            text = (f"Before this decision you were asked:\n{L.render_belief(c['game'], c['cond'], c['wording_id'])}"
                    f"\nYour answer was: {b.value}\n\n{text}")
        items.append((c["agent"]["text"], _user(text, spec), spec, {"sample": c["sample"]}))
    answers = ask_many(llm, items, retries=retries, max_tokens=max_tokens, parallel=parallel, **llm_kw) if items else []

    rows = []
    for i, (c, a) in enumerate(zip(cells, answers)):
        b = beliefs.get(i)
        g = c["game"]
        rows.append({
            "model": model, "persona_method": c["agent"]["persona_method"], "agent_id": c["agent"]["agent_id"],
            "game": g["game"], "role": g["role"], "condition": str(c["cond"]["id"]), "wording_id": c["wording_id"],
            "run": c["run"], "decision": a.value, "belief": (b.value if b is not None else None),
            "valid": bool(a.valid), "raw_len": a.raw_len,
            "cost_usd": round(a.cost_usd + (b.cost_usd if b is not None else 0.0), 8),
            "seed": seed, "max_tokens": int(max_tokens),
            "persona_params": json.dumps(c["agent"]["params"], sort_keys=True, ensure_ascii=False)
            if c["agent"]["params"] else "",
            "belief_valid": (bool(b.valid) if b is not None else None),
            "attempts": a.attempts, "error": a.error, "holdout": bool(g["holdout"]),
        })
    new = pd.DataFrame(rows, columns=COLUMNS + EXTRA_COLUMNS)
    if out is not None and len(new):
        header = not (os.path.exists(out) and os.path.getsize(out) > 0)
        new.to_csv(out, mode="a", header=header, index=False)
    if existing is None:
        return new
    if not len(new):
        return existing.reset_index(drop=True)
    return pd.concat([existing, new], ignore_index=True)


__all__ = ["play", "estimate_cost", "as_agents", "read_results", "CostGateError", "COLUMNS", "EXTRA_COLUMNS", "KEY"]

"""Solution tree search for automated modelling: AIDE-style draft / debug / improve with greedy best-node selection
(Jiang et al. 2025, "AIDE: AI-Driven Exploration in the Space of Code", arXiv:2502.13138), plus MCTS-UCT over the
discrete choices of a `Space` (UCB1 tree policy, Kocsis & Szepesvari 2006).

    from simengine.research.space import Space
    from simengine.research.tree_search import ConfigOperators, MCTSSearch, TreeSearch
    space = Space({"k": (1, 50), "method": ["a", "b"]}, defaults={"k": 10, "method": "a"})
    ts = TreeSearch(runner, ConfigOperators(space), primary="err", higher_is_better=False, steps=20,
                    workdir="runs/tree")
    out = ts.run()          # nodes on the dev split; out["best_node"], out["best_payload"]
    fin = ts.finalize()     # best node + first draft run ONCE on the test split
    m = MCTSSearch(runner, space, primary="err", higher_is_better=False, budget=40, workdir="runs/mcts")
    m.run(); m.finalize()

A node holds a payload: a config dict, or {"code": str} for code solutions. The runner follows the research-loop
contract (simengine.types.Runner, see loop.py): `runner({**payload, split_key: split})` returns
{"per_unit": DataFrame[unit, <primary>, ...], "metrics": {...}} (a {"score": float} reply is accepted and marked
score_source="runner"). Scores are computed here from per-unit values. A node is buggy when the runner raises, reports
status "crash", returns no finite score, changes the dev unit count, or the operator produced no valid payload (AIDE:
exception or missing metric = buggy). Nothing here executes code; a code runner must sandbox it.

Search policy (AIDE `Agent.search_policy`, defaults of aideml `config.yaml`: num_drafts 5, debug_prob 0.5,
max_debug_depth 3): draft until `num_drafts` drafts exist; else with probability `debug_prob` debug a random buggy
leaf whose debug depth < max_debug_depth; else improve the best good node (greedy); draft if there is no good node.
Operators: `ConfigOperators` (non-LLM: random draft, one-key repair, one-key local improve) and `LLMOperators`
(any simengine LLM, FakeLLM in tests; prompts carry a tree summary `summarize_tree`, AIDE's Sigma(T)).

The final pick is the max over many noisy dev scores, so the dev score of the winner is optimistic; `finalize()`
reports test - dev for it. Workdir layout matches `Loop`: state.json (resume by running again), JOURNAL.md,
results.tsv, units/<key>.csv.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd

from ..types import LLM
from .loop import _fingerprint, _jsonable
from .proposers import TEST_REFERENCE
from .space import Space, config_diff, config_hash
from .stats import bootstrap_mean, paired_bootstrap_diff

STATE_VERSION = 1
OPS = ("draft", "debug", "improve")


@dataclass
class Node:
    id: int
    parent: int | None
    op: str                                  # draft | debug | improve
    payload: dict | None                     # config or {"code": str}; None = operator gave nothing usable
    plan: str = ""
    status: str = "pending"                  # ok | buggy
    score: float | None = None               # mean per-unit primary on the dev split
    lo: float | None = None
    hi: float | None = None
    n_units: int = 0
    error: str | None = None
    metrics: dict = field(default_factory=dict)
    debug_depth: int = 0                     # 0 unless op == debug: parent.debug_depth + 1
    children: list = field(default_factory=list)
    payload_hash: str | None = None
    repeat_of: int | None = None             # same payload as an earlier node: runner not called again
    diff_parent: dict | None = None          # paired diff vs parent on shared units (info only)
    score_source: str = "per_unit"
    seconds: float = 0.0
    cost_usd: float = 0.0
    eval_key: str | None = None

    @property
    def buggy(self) -> bool:
        return self.status == "buggy"

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SearchContext:
    step: int
    nodes: list
    primary: str
    higher_is_better: bool
    seen: set
    rng: np.random.Generator
    task: str = ""

    def best(self) -> Node | None:
        return best_node(self.nodes, self.higher_is_better)

    def summary(self, k: int = 5) -> str:
        return summarize_tree(self.nodes, self.primary, self.higher_is_better, k=k)


class Operators(Protocol):
    """draft(ctx) / debug(node, ctx) / improve(node, ctx) -> {"payload": dict | None, "plan": str, "error": str | None,
    "cost_usd": float, "raw": str, "prompt": str} or None (nothing left to propose)."""

    def draft(self, ctx: SearchContext) -> dict | None: ...

    def debug(self, node: Node, ctx: SearchContext) -> dict | None: ...

    def improve(self, node: Node, ctx: SearchContext) -> dict | None: ...


def best_node(nodes: Sequence[Node], higher_is_better: bool) -> Node | None:
    """Greedy pick: the good node with the best dev score; ties go to the earlier node."""
    good = [n for n in nodes if n.status == "ok" and n.score is not None]
    if not good:
        return None
    return min(good, key=lambda n: ((-n.score if higher_is_better else n.score), n.id))


def _short(s: str | None, n: int) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[:n - 3] + "..."


def _payload_text(p: dict | None, n: int = 300) -> str:
    if p is None:
        return "(none)"
    if set(p) == {"code"}:
        return f"<code, {len(str(p['code']).splitlines())} lines>"
    return _short(json.dumps(p, sort_keys=True, default=str), n)


def summarize_tree(nodes: Sequence[Node], primary: str, higher_is_better: bool, *, k: int = 5,
                   n_bugs: int = 3) -> str:
    """Compact tree memory for prompts (AIDE's summarization operator): counts, the k best good nodes with plan and
    payload, the last buggy nodes with their errors. Dev scores only."""
    ok = [n for n in nodes if n.status == "ok"]
    bug = [n for n in nodes if n.buggy]
    b = best_node(nodes, higher_is_better)
    L = [f"Tree: {len(nodes)} nodes ({len(ok)} ok, {len(bug)} buggy); {primary}: "
         f"{'higher' if higher_is_better else 'lower'} is better"
         + (f"; best node {b.id} {primary}={b.score:.4f}." if b else "; no working solution yet.")]
    top = sorted(ok, key=lambda n: ((-n.score if higher_is_better else n.score), n.id))[:k]
    if top:
        L.append("Best nodes (dev):")
        for n in top:
            L.append(f"- node {n.id} ({n.op}{'' if n.parent is None else f' of {n.parent}'}) {primary}={n.score:.4f}"
                     f" plan: {_short(n.plan, 160) or '-'} | {_payload_text(n.payload)}")
    if bug:
        L.append("Recent failures:")
        for n in bug[-n_bugs:]:
            L.append(f"- node {n.id} ({n.op}): {_short(n.error, 200)} | {_payload_text(n.payload, 160)}")
    return "\n".join(L)


# ===================================================================== operators


def _random_value(space: Space, k: str, rng: np.random.Generator):
    u = space.params[k]
    if isinstance(u, list):
        u = [x for x in u if x not in space.forbidden.get(k, [])]
        return u[int(rng.integers(len(u)))] if u else None
    if space.is_int_range(k):
        return int(rng.integers(u[0], u[1] + 1))
    return round(float(rng.uniform(u[0], u[1])), 4)


def _local_value(space: Space, k: str, cur, rng: np.random.Generator):
    """A nearby value: another choice, or a Gaussian step of sd (hi - lo) / 4 clipped to the range."""
    u = space.params[k]
    if isinstance(u, list) or cur is None or not isinstance(cur, (int, float)):
        return _random_value(space, k, rng)
    lo, hi = u
    v = float(np.clip(cur + rng.normal(0, (hi - lo) / 4), lo, hi))
    return int(round(v)) if space.is_int_range(k) else round(v, 4)


def _word(w: str, text: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(str(w).lower())}(?![\w.])", text) is not None


class ConfigOperators:
    """Non-LLM operators over a `Space`. draft: the space defaults first (if `default_first`), then uniform random
    configs; improve: one atomic change (one key, local step); debug: one change, preferring keys named in the
    error text. Every payload is validated and unseen; None after `tries` misses."""
    name = "config"

    def __init__(self, space: Space, *, default_first: bool = True, tries: int = 200):
        self.space, self.default_first, self.tries = space, default_first, tries

    def _keys(self, cfg):
        return self.space.used(cfg)

    def draft(self, ctx: SearchContext):
        if self.default_first and not any(n.op == "draft" for n in ctx.nodes):
            full, err = self.space.validate({})
            if not err and config_hash(full) not in ctx.seen:
                return {"payload": full, "plan": "space defaults"}
        base, _ = self.space.validate({})
        for _ in range(self.tries):
            cfg = dict(base)
            for k in self._keys(base):
                cfg[k] = _random_value(self.space, k, ctx.rng)
            full, err = self.space.validate(cfg)
            if not err and config_hash(full) not in ctx.seen:
                return {"payload": full, "plan": "random draft"}
        return None

    def _one_change(self, cfg, ctx, keys, local, label):
        if not keys:
            return None
        for _ in range(self.tries):
            k = keys[int(ctx.rng.integers(len(keys)))]
            v = _local_value(self.space, k, cfg.get(k), ctx.rng) if local else _random_value(self.space, k, ctx.rng)
            if isinstance(v, np.generic):
                v = v.item()
            if v is None or v == cfg.get(k):
                continue
            full, err = self.space.validate({**cfg, k: v})
            if not err and config_hash(full) not in ctx.seen:
                return {"payload": full, "plan": f"{label} {k}: {cfg.get(k)!r} -> {v!r}"}
        return None

    def improve(self, node: Node, ctx: SearchContext):
        return self._one_change(node.payload, ctx, self._keys(node.payload), True, "improve")

    def debug(self, node: Node, ctx: SearchContext):
        if node.payload is None or set(node.payload) == {"code"}:
            return self.draft(ctx)
        keys = self._keys(node.payload)
        err = (node.error or "").lower()
        named = [k for k in keys if _word(k, err) or
                 (isinstance(node.payload.get(k), str) and _word(node.payload[k], err))]
        return self._one_change(node.payload, ctx, named, False, "debug") or \
            self._one_change(node.payload, ctx, keys, False, "debug")


SYSTEM = ("You are an expert machine-learning engineer searching for the best solution to a modelling task. You see "
          "development-split scores only. Follow the reply format exactly.")

_FMT_CONFIG = ('Reply with ONE JSON object: {"plan": "<one or two sentences>", "config": {<full configuration>}}. '
               "Allowed space (lists = choices, pairs = [min, max]): {space}")
_FMT_CODE = ("Reply with a short plan (a few sentences), then the complete solution as ONE ```python code block. "
             "The code must be a single self-contained file.")


def parse_config_reply(text: str) -> tuple[dict | None, str, str | None]:
    """(config, plan, error) from a JSON reply: first '{' to last '}', keys `config` (or `payload`) and `plan`."""
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None, "", "no JSON object in the reply"
    try:
        d = json.loads(text[a:b + 1])
    except (json.JSONDecodeError, ValueError) as e:
        return None, "", f"JSON could not be parsed: {e}"
    if not isinstance(d, dict):
        return None, "", "reply must be a JSON object"
    cfg = d.get("config", d.get("payload"))
    if not isinstance(cfg, dict):
        return None, str(d.get("plan", "")), "reply has no 'config' object"
    return cfg, str(d.get("plan", "")), None


_CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def parse_code_reply(text: str) -> tuple[str | None, str, str | None]:
    """(code, plan, error): the last fenced code block; the plan is the text before the first fence."""
    blocks = _CODE_RE.findall(text)
    if not blocks:
        return None, _short(text, 500), "no ```python code block in the reply"
    return blocks[-1].strip("\n"), text[:text.find("```")].strip(), None


class LLMOperators:
    """AIDE coding operators through an LLM. mode="config": the reply is a JSON config, validated against `space`
    (improve must change exactly one key when `atomic`); mode="code": the reply is a plan + one python block, the
    payload is {"code": ...}. A reply that cannot be used becomes a buggy node carrying the reason, so the debug
    operator can repair it. Replies mentioning the test split are refused in config mode (the regex also matches
    names like train_test_split, so it is off by default for code)."""
    name = "llm"

    def __init__(self, llm: LLM, *, mode: str = "config", space: Space | None = None, task: str = "",
                 data_preview: str = "", atomic: bool = True, max_tokens: int = 2000, system: str = SYSTEM,
                 forbid_test_refs: bool | None = None, summary_k: int = 5, **complete_kw):
        if mode not in ("config", "code"):
            raise ValueError("mode must be 'config' or 'code'")
        if mode == "config" and space is None:
            raise ValueError("mode='config' needs a space")
        self.llm, self.mode, self.space, self.task = llm, mode, space, task
        self.data_preview, self.atomic, self.max_tokens, self.system = data_preview, atomic, max_tokens, system
        self.forbid_test_refs = (mode == "config") if forbid_test_refs is None else forbid_test_refs
        self.summary_k, self.complete_kw = summary_k, complete_kw

    def _fmt(self):
        if self.mode == "code":
            return _FMT_CODE
        return _FMT_CONFIG.replace("{space}", json.dumps(self.space.allowed(), ensure_ascii=False, default=str))

    def _head(self, ctx):
        h = [self.task or ctx.task or "Modelling task."]
        if self.data_preview:
            h.append(f"Data preview:\n{self.data_preview}")
        h.append(f"Search memory:\n{ctx.summary(self.summary_k)}")
        return "\n\n".join(h)

    def _sol(self, node: Node) -> str:
        if node.payload is None:
            return "(the previous reply could not be used)"
        if self.mode == "code":
            return f"```python\n{node.payload.get('code', '')}\n```"
        return json.dumps(node.payload, ensure_ascii=False, default=str)

    def _ask(self, prompt: str, ctx: SearchContext, parent: Node | None, op: str) -> dict:
        r = self.llm.complete(self.system, prompt, max_tokens=self.max_tokens, **self.complete_kw)
        text = r.text or ""
        out = {"raw": text, "prompt": prompt, "cost_usd": float(getattr(r, "cost_usd", 0.0) or 0.0), "payload": None}
        if self.forbid_test_refs:
            m = TEST_REFERENCE.search(text)
            if m:
                return {**out, "plan": "", "error": f"reply refers to the test split ({m.group(0)!r})"}
        if self.mode == "code":
            code, plan, err = parse_code_reply(text)
            return {**out, "plan": plan, "error": err, "payload": None if err else {"code": code}}
        cfg, plan, err = parse_config_reply(text)
        if err:
            return {**out, "plan": plan, "error": err}
        full, errs = self.space.validate(cfg)
        if errs:
            return {**out, "plan": plan, "error": "invalid config: " + "; ".join(errs)}
        if op == "improve" and self.atomic and parent is not None and parent.payload:
            d = config_diff(full, parent.payload)
            if len(d) != 1:
                return {**out, "plan": plan, "error": f"improve must change exactly one key, changed {sorted(d)}"}
        return {**out, "plan": plan, "payload": full}

    def draft(self, ctx: SearchContext):
        p = (f"{self._head(ctx)}\n\nDRAFT: propose a new solution, different from the ones in the memory. Start with "
             f"a simple, robust approach.\n\n{self._fmt()}")
        return self._ask(p, ctx, None, "draft")

    def improve(self, node: Node, ctx: SearchContext):
        what = "exactly ONE configuration key" if self.mode == "config" else "ONE atomic change (one idea)"
        p = (f"{self._head(ctx)}\n\nIMPROVE node {node.id} ({ctx.primary}={node.score:.4f} on dev). Current "
             f"solution:\n{self._sol(node)}\n\nChange {what} so its effect can be measured; say what you expect."
             f"\n\n{self._fmt()}")
        return self._ask(p, ctx, node, "improve")

    def debug(self, node: Node, ctx: SearchContext):
        p = (f"{self._head(ctx)}\n\nDEBUG node {node.id}. It failed with:\n{_short(node.error, 1500)}\n\nSolution:\n"
             f"{self._sol(node)}\n\nFix the bug and keep the approach.\n\n{self._fmt()}")
        return self._ask(p, ctx, node, "debug")


# ===================================================================== shared evaluation / finalize


class _Search:
    kind = "search"

    def __init__(self, runner: Callable[[dict], dict], *, primary: str, higher_is_better: bool, unit_col: str,
                 split_key: str, dev_split: str, test_split: str, workdir, seed: int, n_boot: int, alpha: float,
                 check_runner: bool, verbose: bool):
        self.runner, self.primary, self.hib = runner, primary, higher_is_better
        self.unit_col, self.split_key, self.dev, self.test = unit_col, split_key, dev_split, test_split
        self.workdir = Path(workdir) if workdir else None
        self.seed, self.n_boot, self.alpha = seed, n_boot, alpha
        self.check_runner, self.verbose = check_runner, verbose
        self._units: dict[str, pd.Series] = {}

    # ------------------------------------------------------------ state
    def _base_state(self) -> dict:
        return {"version": STATE_VERSION, "kind": self.kind, "runner_fingerprint": _fingerprint(self.runner),
                "primary": self.primary, "higher_is_better": self.hib, "seed": self.seed, "nodes": [], "evals": {},
                "stopped": None, "final": None, "cost_usd": 0.0, "runner_calls": 0,
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    def _load_state(self) -> dict | None:
        p = self.workdir / "state.json" if self.workdir else None
        if p is None or not p.exists():
            return None
        st = json.loads(p.read_text(encoding="utf-8"))
        if st.get("kind") != self.kind:
            raise RuntimeError(f"workdir holds a {st.get('kind')!r} state, not {self.kind!r}; use a new workdir")
        fp = _fingerprint(self.runner)
        if self.check_runner and st.get("runner_fingerprint") and fp and st["runner_fingerprint"] != fp:
            raise RuntimeError("runner source changed since the search started; start a new workdir or pass "
                               "check_runner=False")
        if st.get("primary") != self.primary or st.get("higher_is_better") != self.hib:
            raise RuntimeError("primary metric or direction differs from the saved state; start a new workdir")
        return st

    def _save(self):
        if self.workdir is None:
            return
        self.workdir.mkdir(parents=True, exist_ok=True)
        tmp = self.workdir / "state.json.tmp"
        tmp.write_text(json.dumps(_jsonable(self.state), indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(self.workdir / "state.json")
        (self.workdir / "JOURNAL.md").write_text(self.journal_markdown(), encoding="utf-8")
        rows = [{k: n.get(k) for k in ("id", "parent", "op", "status", "score", "lo", "hi", "n_units", "debug_depth",
                                       "repeat_of", "payload_hash", "seconds")}
                | {"plan": _short(n.get("plan"), 200), "error": _short(n.get("error"), 200),
                   "payload": _payload_text(n.get("payload"), 400)} for n in self.state["nodes"]]
        pd.DataFrame(rows).to_csv(self.workdir / "results.tsv", sep="\t", index=False, float_format="%.5f")

    @property
    def nodes(self) -> list[Node]:
        return [Node.from_dict(d) for d in self.state["nodes"]]

    # ------------------------------------------------------------ evaluation
    def _units_path(self, key):
        return self.workdir / "units" / f"{key}.csv" if self.workdir else None

    def _per_unit(self, res: dict) -> pd.DataFrame | None:
        pu = res.get("per_unit")
        if pu is None:
            return None
        if isinstance(pu, dict):
            pu = pd.DataFrame({self.unit_col: list(pu), self.primary: list(pu.values())})
        elif isinstance(pu, pd.Series):
            pu = pd.DataFrame({self.unit_col: pu.index, self.primary: pu.values})
        elif not isinstance(pu, pd.DataFrame):
            pu = pd.DataFrame(pu)
        if self.unit_col not in pu.columns:
            pu = pu.reset_index().rename(columns={"index": self.unit_col})
        if self.primary not in pu.columns:
            raise ValueError(f"per_unit has no primary column {self.primary!r}")
        pu = pu.copy()
        pu[self.unit_col] = pu[self.unit_col].astype(str)
        return pu

    def evaluate(self, payload: dict, split: str | None = None, *, _allow_test: bool = False) -> dict:
        """Run the runner once per (payload, split) and cache the summary. The test split is reachable only from
        finalize()."""
        split = split or self.dev
        if split == self.test and not _allow_test:
            raise PermissionError("the test split is evaluated only once, by finalize()")
        key = f"{split}_{config_hash(payload)}"
        if key in self.state["evals"]:
            return self.state["evals"][key]
        t0 = time.time()
        summ = {"key": key, "split": split}
        try:
            self.state["runner_calls"] += 1
            res = self.runner({**payload, self.split_key: split}) or {}
            if not isinstance(res, dict):
                res = {"score": res}
            if res.get("status") == "crash":
                raise RuntimeError(res.get("error", "runner reported crash"))
            pu = self._per_unit(res)
            if pu is not None:
                if len(pu) == 0:
                    raise ValueError("no units")
                if pu[self.unit_col].duplicated().any():
                    raise ValueError("duplicate unit ids")
                v = pd.to_numeric(pu[self.primary], errors="coerce").to_numpy(float)
                if not np.isfinite(v).all():
                    raise ValueError(f"non-finite {self.primary} in {int((~np.isfinite(v)).sum())} units")
                n0 = self.state.get("dev_n")
                if split == self.dev and n0 is not None and len(pu) != n0:
                    raise ValueError(f"unit count {len(pu)} != {n0} of the first scored node")
                m = bootstrap_mean(v, seed=self.seed, B=self.n_boot, alpha=self.alpha)
                summ.update(score=m["value"], lo=m["lo"], hi=m["hi"], n=m["n"], score_source="per_unit")
                s = pd.Series(v, index=pu[self.unit_col].to_numpy())
                self._units[key] = s
                p = self._units_path(key)
                if p is not None:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    pu.to_csv(p, index=False)
                if split == self.dev and n0 is None:
                    self.state["dev_n"] = len(pu)
            else:
                x = res.get("score", (res.get("metrics") or {}).get(self.primary))
                x = float(x) if isinstance(x, (int, float, np.number)) and not isinstance(x, bool) else float("nan")
                if not math.isfinite(x):
                    raise ValueError(f"runner returned no per_unit and no finite {self.primary!r} score")
                summ.update(score=x, lo=None, hi=None, n=None, score_source="runner")
            summ.update(status="ok", metrics=_jsonable(res.get("metrics", {})))
        except Exception as e:  # noqa: BLE001 - a crash is a buggy node, not a reason to stop
            summ.update(status="buggy", error=f"{type(e).__name__}: {str(e)[:1500]}")
        summ["seconds"] = round(time.time() - t0, 3)
        self.state["evals"][key] = summ
        return summ

    def units(self, key: str) -> pd.Series | None:
        if key not in self._units:
            p = self._units_path(key)
            if p is None or not p.exists():
                return None
            pu = pd.read_csv(p, dtype={self.unit_col: str}, float_precision="round_trip")
            self._units[key] = pu.set_index(self.unit_col)[self.primary].astype(float)
        return self._units[key]

    def paired(self, key_a: str, key_b: str, alpha: float | None = None) -> dict | None:
        """mean(a - b) over shared units with a bootstrap CI; None without per-unit scores."""
        a, b = self.units(key_a), self.units(key_b)
        if a is None or b is None:
            return None
        common = a.index.intersection(b.index)
        if not len(common):
            return None
        return paired_bootstrap_diff(a.loc[common].to_numpy(), b.loc[common].to_numpy(), seed=self.seed,
                                     B=self.n_boot, alpha=alpha or self.alpha)

    def best(self) -> Node | None:
        return best_node(self.nodes, self.hib)

    def _reference(self) -> Node | None:
        """The node the winner is compared with on test: the first good node."""
        return next((n for n in self.nodes if n.status == "ok"), None)

    def finalize(self, extra_test: dict[str, dict] | None = None) -> dict:
        """Winner (greedy best dev node), the reference (first good node) and `extra_test` payloads run ONCE on the
        test split; paired winner - reference diffs with 95% and Bonferroni CIs. Idempotent."""
        st = self.state
        if st.get("final"):
            return self.result()
        w = self.best()
        if w is None:
            raise RuntimeError("no good node to finalize")
        ref = self._reference()
        cfgs = {"winner": w.payload, "reference": ref.payload, **(extra_test or {})}
        tests = {a: self.evaluate(p, self.test, _allow_test=True) for a, p in cfgs.items()}
        k = max(len(cfgs) - 1, 1)
        diffs = {}
        for a in cfgs:
            if a == "winner" or tests[a]["status"] != "ok" or tests["winner"]["status"] != "ok":
                continue
            diffs[f"winner-{a}"] = {"ci95": self.paired(tests["winner"]["key"], tests[a]["key"]),
                                    "bonferroni": self.paired(tests["winner"]["key"], tests[a]["key"],
                                                              alpha=self.alpha / k)}
        wt = tests["winner"]
        fin = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "winner_node": w.id, "winner_payload": w.payload,
               "reference_node": ref.id, "winner_dev": w.score,
               "test": {a: {kk: t.get(kk) for kk in ("status", "score", "lo", "hi", "n", "error", "key")}
                        for a, t in tests.items()},
               "test_diff": diffs, "bonferroni_k": k, "n_dev_candidates": sum(n.status == "ok" for n in self.nodes),
               # max over noisy dev scores is optimistic: test - dev of the winner, metric units
               "winner_test_minus_dev": (wt["score"] - w.score) if wt["status"] == "ok" else None}
        st["final"] = fin
        self._save()
        return self.result()

    def result(self) -> dict:
        st, b = self.state, self.best()
        nodes = self.nodes
        return {"kind": self.kind, "stopped": st["stopped"], "n_nodes": len(nodes),
                "best_node": b.id if b else None, "best_payload": b.payload if b else None,
                "best_dev": {"value": b.score, "lo": b.lo, "hi": b.hi, "n": b.n_units} if b else None,
                "counts": {s: sum(n.status == s for n in nodes) for s in ("ok", "buggy")}
                | {op: sum(n.op == op for n in nodes) for op in OPS}
                | {"repeat": sum(n.repeat_of is not None for n in nodes)},
                "runner_calls": st["runner_calls"], "cost_usd": st["cost_usd"], "final": st.get("final"),
                "history": st["nodes"]}

    def _make_node(self, nid, parent, op, prop, debug_depth) -> Node:
        """Evaluate a proposal (or record why there is nothing to evaluate) and return the node."""
        payload = prop.get("payload")
        node = Node(id=nid, parent=parent.id if parent else None, op=op, payload=payload,
                    plan=str(prop.get("plan") or ""), debug_depth=debug_depth,
                    cost_usd=float(prop.get("cost_usd", 0.0) or 0.0))
        if payload is None or prop.get("error"):
            node.status, node.error = "buggy", f"operator: {prop.get('error') or 'no payload'}"
            return node
        node.payload_hash = config_hash(payload)
        for o in self.state["nodes"]:
            if o.get("payload_hash") == node.payload_hash:
                node.repeat_of = o["id"]
                break
        ev = self.evaluate(payload, self.dev)
        node.eval_key, node.status, node.error = ev["key"], ev["status"], ev.get("error")
        node.seconds = ev["seconds"] if node.repeat_of is None else 0.0
        if ev["status"] == "ok":
            node.score, node.lo, node.hi, node.n_units = ev["score"], ev.get("lo"), ev.get("hi"), ev.get("n") or 0
            node.metrics, node.score_source = ev.get("metrics") or {}, ev.get("score_source", "per_unit")
            if parent is not None and parent.status == "ok" and parent.eval_key and node.repeat_of is None:
                node.diff_parent = self.paired(ev["key"], parent.eval_key)
        return node

    def _append(self, node: Node, parent: Node | None):
        self.state["nodes"].append(node.to_dict())
        if parent is not None:
            self.state["nodes"][parent.id]["children"].append(node.id)
        self.state["cost_usd"] += node.cost_usd
        if self.verbose:
            print(f"node {node.id} {node.op} {node.status} {self.primary}={node.score} {_short(node.plan, 80)}",
                  flush=True)

    # ------------------------------------------------------------ journal
    def tree_text(self) -> str:
        nodes = self.nodes
        b = self.best()
        out = []

        def walk(n: Node, depth: int):
            mark = " *best*" if b and n.id == b.id else ""
            s = f"{n.score:.4f}" if n.score is not None else "-"
            out.append(f"{'  ' * depth}- [{n.id}] {n.op} {n.status} {self.primary}={s}{mark}"
                       + (f" (repeat of {n.repeat_of})" if n.repeat_of is not None else "")
                       + (f" · {_short(n.plan, 90)}" if n.plan else "")
                       + (f" · error: {_short(n.error, 90)}" if n.buggy else ""))
            for c in n.children:
                walk(nodes[c], depth + 1)

        for n in nodes:
            if n.parent is None:
                walk(n, 0)
        return "\n".join(out)

    def _journal_head(self) -> list[str]:
        return []

    def journal_markdown(self) -> str:
        st = self.state
        P = self.primary
        L = [f"# {self.kind} journal", "",
             f"Primary `{P}` ({'higher' if self.hib else 'lower'} is better) on split `{self.dev}`, computed from "
             f"per-unit scores. Greedy pick: best dev node. Runner calls: {st['runner_calls']}."]
        L += self._journal_head()
        L += ["", "## Tree", "", self.tree_text() or "(empty)", ""]
        if st["stopped"]:
            L += [f"Stopped: {st['stopped']}", ""]
        fin = st.get("final")
        if fin:
            L += ["## Final (test split evaluated once)", "",
                  f"Winner node {fin['winner_node']} (dev {P} {fin['winner_dev']:.4f}, best of "
                  f"{fin['n_dev_candidates']} good nodes); reference node {fin['reference_node']}.", "",
                  f"| config | test {P} [CI] | n |", "|---|---|---|"]
            for a, t in fin["test"].items():
                ci = f" [{t['lo']:.4f}, {t['hi']:.4f}]" if t.get("lo") is not None else ""
                L.append(f"| {a} | {t['score']:.4f}{ci} | {t.get('n')} |" if t["status"] == "ok"
                         else f"| {a} | buggy: {_short(t.get('error'), 80)} | |")
            for a, d in fin["test_diff"].items():
                if d["ci95"]:
                    f, g = d["ci95"], d["bonferroni"]
                    L.append(f"\n{a}: {f['diff']:+.4f} [{f['lo']:+.4f}, {f['hi']:+.4f}] · Bonferroni (k="
                             f"{fin['bonferroni_k']}) [{g['lo']:+.4f}, {g['hi']:+.4f}]")
            if fin.get("winner_test_minus_dev") is not None:
                L.append(f"\nWinner test - dev: {fin['winner_test_minus_dev']:+.4f} (selection optimism shows here)")
        return "\n".join(L) + "\n"


# ===================================================================== AIDE tree search


class TreeSearch(_Search):
    """AIDE-style solution tree search. Each step: the search policy picks draft / debug / improve and a parent, the
    operator proposes a payload, the runner scores it on dev, the node joins the tree. Stops after `steps` nodes,
    the `budget_usd` operator budget, or when the operators have nothing left. Resume: run() again with the same
    workdir (step n uses default_rng([seed, n]), so a resumed run takes the same decisions)."""
    kind = "tree_search"

    def __init__(self, runner: Callable[[dict], dict], operators: Operators, *, primary: str = "score",
                 higher_is_better: bool = True, steps: int = 20, num_drafts: int = 5, debug_prob: float = 0.5,
                 max_debug_depth: int = 3, budget_usd: float | None = None, task: str = "", unit_col: str = "unit",
                 split_key: str = "split", dev_split: str = "dev", test_split: str = "test",
                 workdir: str | Path | None = None, seed: int = 0, n_boot: int = 1000, alpha: float = 0.05,
                 check_runner: bool = True, verbose: bool = False):
        super().__init__(runner, primary=primary, higher_is_better=higher_is_better, unit_col=unit_col,
                         split_key=split_key, dev_split=dev_split, test_split=test_split, workdir=workdir, seed=seed,
                         n_boot=n_boot, alpha=alpha, check_runner=check_runner, verbose=verbose)
        if not 0 <= debug_prob <= 1:
            raise ValueError("debug_prob must be in [0, 1]")
        self.operators, self.steps, self.num_drafts = operators, steps, num_drafts
        self.debug_prob, self.max_debug_depth, self.budget_usd, self.task = debug_prob, max_debug_depth, budget_usd, task
        self.state = self._load_state() or self._base_state()
        self.state.setdefault("policy", {"steps": steps, "num_drafts": num_drafts, "debug_prob": debug_prob,
                                         "max_debug_depth": max_debug_depth,
                                         "operators": getattr(operators, "name", type(operators).__name__)})

    def policy(self, nodes: list[Node], rng: np.random.Generator) -> tuple[str, Node | None]:
        """AIDE search policy: (op, parent)."""
        if sum(n.op == "draft" for n in nodes) < self.num_drafts:
            return "draft", None
        if rng.random() < self.debug_prob:
            cands = [n for n in nodes if n.buggy and n.is_leaf and n.debug_depth < self.max_debug_depth]
            if cands:
                return "debug", cands[int(rng.integers(len(cands)))]
        b = best_node(nodes, self.hib)
        return ("draft", None) if b is None else ("improve", b)

    def step(self) -> Node | None:
        st = self.state
        nodes = self.nodes
        n = len(nodes)
        rng = np.random.default_rng([self.seed, n])
        op, parent = self.policy(nodes, rng)
        ctx = SearchContext(step=n, nodes=nodes, primary=self.primary, higher_is_better=self.hib,
                            seen={x.payload_hash for x in nodes if x.payload_hash}, rng=rng, task=self.task)
        prop = (self.operators.draft(ctx) if op == "draft" else
                self.operators.debug(parent, ctx) if op == "debug" else self.operators.improve(parent, ctx))
        if prop is None and op != "draft":    # nothing left from this parent: fall back to a draft
            op, parent = "draft", None
            prop = self.operators.draft(ctx)
        if prop is None:
            st["stopped"] = f"operators have nothing left to propose (step {n})"
            return None
        depth = parent.debug_depth + 1 if op == "debug" else 0
        node = self._make_node(n, parent, op, prop, depth)
        self._log_raw(node, prop)
        self._append(node, parent)
        return node

    def _log_raw(self, node: Node, prop: dict):
        if self.workdir is None or "raw" not in prop:
            return
        d = self.workdir / "operator"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"node{node.id}_{node.op}_prompt.txt").write_text(prop.get("prompt", ""), encoding="utf-8")
        (d / f"node{node.id}_{node.op}_reply.txt").write_text(prop.get("raw") or "", encoding="utf-8")

    def run(self, finalize: bool = False, extra_test: dict[str, dict] | None = None) -> dict:
        """Run (or resume) steps; a resumed search with a larger `steps` continues past an old step limit."""
        st = self.state
        if (st["stopped"] or "").startswith("step limit") and len(st["nodes"]) < self.steps and not st.get("final"):
            st["stopped"] = None
        while not st["stopped"]:
            if len(st["nodes"]) >= self.steps:
                st["stopped"] = f"step limit {self.steps}"
                break
            if self.budget_usd is not None and st["cost_usd"] >= self.budget_usd:
                st["stopped"] = f"operator budget ${self.budget_usd:.2f} used (${st['cost_usd']:.4f})"
                break
            self.step()
            self._save()
        self._save()
        return self.finalize(extra_test) if finalize else self.result()

    def _journal_head(self):
        p = self.state.get("policy", {})
        return [f"Policy (AIDE): {p.get('num_drafts')} drafts, debug_prob {p.get('debug_prob')}, max debug depth "
                f"{p.get('max_debug_depth')}, {p.get('steps')} steps; operators `{p.get('operators')}`."]


# ===================================================================== MCTS-UCT over discrete choices


class MCTSSearch(_Search):
    """Monte Carlo tree search over a `Space`: tree depth d fixes key d (choices, or `n_grid` grid points of a range);
    each iteration selects by UCB1 (min-max normalised mean reward + c * sqrt(ln N_parent / N_child)), expands one
    untried child, completes the rest uniformly at random (rollout), scores the full config on dev and backs the
    reward up the path. Crashed or invalid configs count as the worst reward seen. Repeated configs reuse the cached
    score (no runner call). Stops after `budget` runner calls, `max_iter` iterations or an exhausted space. Each
    evaluated config is a node (op "rollout"), so finalize() / the journal are shared with TreeSearch."""
    kind = "mcts"

    def __init__(self, runner: Callable[[dict], dict], space: Space, *, primary: str = "score",
                 higher_is_better: bool = True, budget: int = 50, max_iter: int | None = None,
                 c: float = math.sqrt(2), keys: Sequence[str] | None = None, n_grid: int = 5, unit_col: str = "unit",
                 split_key: str = "split", dev_split: str = "dev", test_split: str = "test",
                 workdir: str | Path | None = None, seed: int = 0, n_boot: int = 1000, alpha: float = 0.05,
                 check_runner: bool = True, verbose: bool = False):
        super().__init__(runner, primary=primary, higher_is_better=higher_is_better, unit_col=unit_col,
                         split_key=split_key, dev_split=dev_split, test_split=test_split, workdir=workdir, seed=seed,
                         n_boot=n_boot, alpha=alpha, check_runner=check_runner, verbose=verbose)
        self.space, self.budget, self.c = space, budget, c
        self.max_iter = max_iter if max_iter is not None else 20 * budget
        base, err = space.validate({})
        if err:
            raise ValueError(f"space defaults are invalid: {'; '.join(err)}")
        self.base = base
        self.keys = list(keys) if keys is not None else space.used(base)
        for k in self.keys:
            if k not in space.params or k in space.fixed:
                raise ValueError(f"key {k!r} is not a searchable key of the space")
        self.choices = [space.grid_values(k, n_grid) for k in self.keys]
        if any(not ch for ch in self.choices):
            raise ValueError("every searched key needs at least one allowed value")
        self.state = self._load_state() or self._base_state()
        self.state.setdefault("tree", {})        # path "i.j.k" -> [N, sum reward of good evals, n_bad]
        self.state.setdefault("exhausted", [])
        self.state.setdefault("iter", 0)
        self.state.setdefault("policy", {"budget": budget, "c": c, "keys": self.keys, "n_grid": n_grid})

    @staticmethod
    def _pk(path: tuple) -> str:
        return ".".join(map(str, path))

    def _reward(self, score):
        return None if score is None else (score if self.hib else -score)

    def _bounds(self):
        r = [self._reward(n["score"]) for n in self.state["nodes"] if n["status"] == "ok"]
        return (min(r), max(r)) if r else (0.0, 0.0)

    def _q(self, stats, lo, hi) -> float:
        N, W, bad = stats
        mean = (W + bad * lo) / N if N else lo
        return 0.5 if hi <= lo else (mean - lo) / (hi - lo)

    def _select_expand(self, rng) -> tuple | None:
        tree, ex = self.state["tree"], set(self.state["exhausted"])
        lo, hi = self._bounds()
        path: tuple = ()
        while len(path) < len(self.keys):
            kids = [path + (i,) for i in range(len(self.choices[len(path)]))]
            untried = [p for p in kids if self._pk(p) not in tree]
            if untried:
                return untried[int(rng.integers(len(untried)))]
            live = [p for p in kids if self._pk(p) not in ex]
            if not live:
                return None
            Np = tree[self._pk(path)][0] if path else sum(tree[self._pk(p)][0] for p in kids)
            ucb = [self._q(tree[self._pk(p)], lo, hi) + self.c * math.sqrt(math.log(max(Np, 1)) / tree[self._pk(p)][0])
                   for p in live]
            path = live[int(np.argmax(ucb))]
        return path

    def _mark_exhausted(self, path: tuple):
        tree, ex = self.state["tree"], self.state["exhausted"]
        if len(path) == len(self.keys) and self._pk(path) not in ex:
            ex.append(self._pk(path))
        while path:
            parent = path[:-1]
            kids = [parent + (i,) for i in range(len(self.choices[len(parent)]))]
            if all(self._pk(p) in ex for p in kids):
                if parent and self._pk(parent) not in ex:
                    ex.append(self._pk(parent))
                elif not parent:
                    self.state["stopped"] = "search space exhausted"
                path = parent
            else:
                break

    def iterate(self) -> Node | None:
        st = self.state
        it = st["iter"]
        rng = np.random.default_rng([self.seed, it])
        path = self._select_expand(rng)
        if path is None:
            st["stopped"] = "search space exhausted"
            return None
        idx = list(path) + [int(rng.integers(len(ch))) for ch in self.choices[len(path):]]
        cfg = {**self.base, **{k: self.choices[d][i] for d, (k, i) in enumerate(zip(self.keys, idx))}}
        full, err = self.space.validate(cfg)
        prop = {"payload": full, "plan": f"rollout from depth {len(path)}",
                "error": ("invalid config: " + "; ".join(err)) if err else None}
        node = self._make_node(len(st["nodes"]), None, "rollout", prop, 0)
        self._append(node, None)
        r = self._reward(node.score) if node.status == "ok" else None
        for d in range(1, len(path) + 1):
            s = st["tree"].setdefault(self._pk(path[:d]), [0, 0.0, 0])
            s[0] += 1
            if r is None:
                s[2] += 1
            else:
                s[1] += r
        if len(path) == len(self.keys):
            self._mark_exhausted(path)
        st["iter"] = it + 1
        return node

    def run(self, finalize: bool = False, extra_test: dict[str, dict] | None = None) -> dict:
        """Run (or resume) iterations; a resumed search with a larger `budget` continues past an old budget stop."""
        st = self.state
        if (st["stopped"] or "").startswith("runner budget") and st["runner_calls"] < self.budget \
                and not st.get("final"):
            st["stopped"] = None
        while not st["stopped"]:
            if st["runner_calls"] >= self.budget:
                st["stopped"] = f"runner budget {self.budget} used"
                break
            if st["iter"] >= self.max_iter:
                st["stopped"] = f"iteration limit {self.max_iter}"
                break
            self.iterate()
            self._save()
        self._save()
        return self.finalize(extra_test) if finalize else self.result()

    def _reference(self) -> Node | None:
        """MCTS has no first draft: the reference is the space defaults, scored on dev if not visited."""
        ref = next((n for n in self.nodes if n.payload_hash == config_hash(self.base) and n.status == "ok"), None)
        if ref is None:
            ev = self.evaluate(self.base, self.dev)
            ref = Node(id=-1, parent=None, op="reference", payload=self.base, status=ev["status"],
                       score=ev.get("score"), eval_key=ev["key"])
        return ref

    def _journal_head(self):
        p = self.state.get("policy", {})
        return [f"MCTS-UCT over keys {p.get('keys')} (c={p.get('c'):.3f}, runner budget {p.get('budget')}); "
                f"iterations {self.state.get('iter')}."]

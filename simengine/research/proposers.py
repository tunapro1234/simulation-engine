"""Proposers: each turn, propose exactly ONE change to the current best configuration.

A proposer is any object with `propose(ctx) -> dict | None` (None = nothing left to propose; the loop stops).
The returned proposal:
    {"change": {key: value},                      # exactly one key
     "kind": "improve" | "fix" | "new" | <label>,
     "hypothesis": str, "expected": "lower" | "same" | "higher",   # primary metric vs current best
     "falsifier": str, "rationale": str,
     "notes": {...}, "raw": str, "cost_usd": float}               # optional
`ctx` is a ProposalContext built by the loop: history (dev metrics only), current best, space, seen config hashes,
the rejection message of a previous invalid attempt (for the repair call), and helpers.

Implemented:
- LLMProposer: an LLM observer (any simengine LLM: Claude API, ClaudeCode `claude -p`, FakeLLM). Port of
  uretim/dongu/kod/dongu.py istem_yaz/gozlemci (prompt in English; Turkish JSON keys are accepted too).
- ClaudeCodeProposer: LLMProposer over `claude -p` (subscription; tools off). Port of gozlemci.sh.
- RandomProposer: seeded single random change (the ablation/control arm of dongu v1.1 §5).
- GridProposer: deterministic coordinate sweep over single changes.
- GreedyCoordinateProposer: coordinate hill-climb with momentum (repeat a kept step in the same direction).
- HalvingProposer: screens many single changes with successive halving at low fidelity, proposes the survivor.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..types import LLM, LLMResponse
from .space import Space, config_diff, config_hash

KINDS = ("improve", "fix", "new")
DIRECTIONS = ("lower", "same", "higher")
# Port of dongu.py TEST_ATFI: an observer answer that refers to the test split is rejected.
TEST_REFERENCE = re.compile(r"--test|/test\b|test_|\btest[ -]?(set|score|metric|split|wd|result|data|question)", re.I)

_TR_KEYS = {"oneri": "proposal", "degisiklik": "change", "tur": "kind", "hipotez": "hypothesis",
            "beklenen_yon": "expected", "ne_cikarsa_yanlis": "falsifier", "gerekce": "rationale",
            "ne_degisti": "what_changed", "beklenmedik": "surprising", "sizinti_suphesi": "leakage_suspicion"}
_TR_KINDS = {"iyilestir": "improve", "onar": "fix", "yeni": "new"}


@dataclass
class ProposalContext:
    turn: int                       # number of the turn that will run the proposal
    incumbent: dict                 # {"n", "cfg", "metrics"} of the current best
    start: dict
    history: list                   # turn records (dev metrics only)
    space: Space
    seen: set                       # config hashes already run
    primary: str
    higher_is_better: bool
    task: str = ""
    baselines: dict = field(default_factory=dict)   # name -> dev metrics summary
    error: str | None = None                        # rejection message of the previous attempt (repair call)
    stop_after: int | None = 3
    max_turns: int | None = 12
    screen: Callable[[list, float], list] | None = None   # (configs, fidelity) -> primary means (HalvingProposer)

    def candidate(self, key, value) -> dict | None:
        """Full config = incumbent + {key: value} if valid, unseen and not a no-op; else None."""
        if key in self.space.fixed or value in self.space.forbidden.get(key, []):
            return None
        full, err = self.space.validate({**self.incumbent["cfg"], key: value})
        if err or config_hash(full) in self.seen:
            return None
        return full


def _mk(change, kind, rationale="", hypothesis="-", expected="-", falsifier="-"):
    return {"change": change, "kind": kind, "hypothesis": hypothesis, "expected": expected,
            "falsifier": falsifier, "rationale": rationale}


# ===================================================================== non-LLM proposers

class RandomProposer:
    """Seeded single random change: a used key of the current best, a value drawn uniformly from its range/choices.
    Turn n uses default_rng(seed + n) (dongu v1.1: seed 1000)."""
    name = "random"
    requires_fields = False

    def __init__(self, seed: int = 1000, tries: int = 500):
        self.seed, self.tries = seed, tries

    def propose(self, ctx: ProposalContext):
        rng = np.random.default_rng(self.seed + ctx.turn)
        keys = ctx.space.used(ctx.incumbent["cfg"])
        if not keys:
            return None
        for _ in range(self.tries):
            k = keys[int(rng.integers(len(keys)))]
            u = ctx.space.params[k]
            if isinstance(u, list):
                u = [x for x in u if x not in ctx.space.forbidden.get(k, [])]
                if not u:
                    continue
                v = u[int(rng.integers(len(u)))]
            elif ctx.space.is_int_range(k):
                v = int(rng.integers(u[0], u[1] + 1))
            else:
                v = round(float(rng.uniform(u[0], u[1])), 2)
            if isinstance(v, np.generic):
                v = v.item()
            if ctx.candidate(k, v) is not None:
                return _mk({k: v}, "random", f"seed {self.seed + ctx.turn}")
        return None


class GridProposer:
    """Deterministic sweep: for each used key (in space order) and each grid value, the first single change from the
    current best that is valid and not yet run."""
    name = "grid"
    requires_fields = False

    def __init__(self, n_grid: int = 5):
        self.n_grid = n_grid

    def propose(self, ctx: ProposalContext):
        for k in ctx.space.used(ctx.incumbent["cfg"]):
            for v in ctx.space.grid_values(k, self.n_grid):
                if ctx.candidate(k, v) is not None:
                    return _mk({k: v}, "grid", f"grid sweep {k}")
        return None


class GreedyCoordinateProposer:
    """Coordinate hill-climb. Moves one key one step from the current best (adjacent choice, or +/- (hi-lo)/n_steps
    for ranges). If the previous turn was kept, repeat the same key and direction (momentum); otherwise continue the
    cycle over (key, direction) pairs."""
    name = "greedy"
    requires_fields = False

    def __init__(self, n_steps: int = 8):
        self.n_steps = n_steps
        self._cursor = 0

    def _step(self, space: Space, k, cur, d):
        u = space.params[k]
        if isinstance(u, list):
            vals = [x for x in u if x not in space.forbidden.get(k, [])]
            if cur not in vals:
                return None
            j = vals.index(cur) + d
            return vals[j] if 0 <= j < len(vals) else None
        lo, hi = u
        if space.is_int_range(k):
            s = max(1, int(round((hi - lo) / self.n_steps)))
        else:
            s = (hi - lo) / self.n_steps
        v = min(max(cur + d * s, lo), hi)
        if not space.is_int_range(k):
            v = round(float(v), 6)
        return None if v == cur else v

    def propose(self, ctx: ProposalContext):
        inc = ctx.incumbent["cfg"]
        last = ctx.history[-1] if ctx.history else None
        if last and last.get("status") == "keep" and last.get("proposal"):
            (k, v), = last["proposal"]["change"].items()
            prev = ctx.history[last["inc_before"]]["cfg"] if last.get("inc_before") is not None else None
            if prev is not None and k in prev:
                d = self._dir(ctx.space, k, prev[k], v)
                nv = self._step(ctx.space, k, inc[k], d) if d else None
                if nv is not None and ctx.candidate(k, nv) is not None:
                    return _mk({k: nv}, "improve", f"momentum: {k} step {d:+d} was kept")
        pairs = [(k, d) for k in ctx.space.used(inc) for d in (+1, -1)]
        for i in range(len(pairs)):
            k, d = pairs[(self._cursor + i) % len(pairs)]
            nv = self._step(ctx.space, k, inc.get(k), d) if k in inc else None
            if nv is not None and ctx.candidate(k, nv) is not None:
                self._cursor = (self._cursor + i + 1) % len(pairs)
                return _mk({k: nv}, "improve", f"coordinate step {k} {d:+d}")
        return None

    @staticmethod
    def _dir(space, k, old, new):
        u = space.params[k]
        if isinstance(u, list):
            try:
                return 1 if u.index(new) > u.index(old) else -1
            except ValueError:
                return 0
        return 1 if new > old else -1 if new < old else 0


def successive_halving(evaluate: Callable[[list, float], list], candidates: list, *, higher_is_better: bool,
                       fidelities=(1 / 9, 1 / 3, 1.0), eta: int = 3, rounding: str = "ceil") -> dict:
    """Successive halving over `candidates` (configs). `evaluate(configs, fidelity)` returns one primary-metric mean
    per config at that fidelity (e.g. fraction of units or of iterations). After each rung keep the best
    ceil(n/eta), or floor(n/eta) with rounding="floor" (Hyperband's top_k(T, L, floor(n_i/eta)), Li et al. 2018
    JMLR Alg. 1; at least one is kept). Returns {"winner", "rungs": [[(index, score), ...], ...], "evaluations"}."""
    if rounding not in ("ceil", "floor"):
        raise ValueError(f"rounding must be 'ceil' or 'floor', got {rounding!r}")
    alive = list(range(len(candidates)))
    rungs, n_eval = [], 0
    for r, f in enumerate(fidelities):
        scores = evaluate([candidates[i] for i in alive], f)
        n_eval += len(alive)
        pairs = list(zip(alive, [float(s) for s in scores]))
        rungs.append(pairs)
        pairs = [p for p in pairs if not math.isnan(p[1])]
        if not pairs:
            return {"winner": None, "rungs": rungs, "evaluations": n_eval}
        pairs.sort(key=lambda p: -p[1] if higher_is_better else p[1])
        if r == len(fidelities) - 1 or len(pairs) == 1:
            alive = [pairs[0][0]]
            break
        keep = math.ceil(len(pairs) / eta) if rounding == "ceil" else len(pairs) // eta
        alive = [p[0] for p in pairs[:max(1, keep)]]
    return {"winner": candidates[alive[0]], "winner_index": alive[0], "rungs": rungs, "evaluations": n_eval}


class HalvingProposer:
    """Screen up to n_candidates single changes (grid neighbours of the current best) with successive halving on the
    dev split at low fidelity, propose the survivor. Needs a runner that reads the fidelity key (Loop(fidelity_key=)).
    Every screened config is an extra look at the dev split: more comparisons, more room to overfit dev."""
    name = "halving"
    requires_fields = False

    def __init__(self, n_candidates: int = 9, fidelities=(1 / 9, 1 / 3), eta: int = 3, n_grid: int = 5,
                 seed: int = 0):
        self.n_candidates, self.fidelities, self.eta, self.n_grid, self.seed = n_candidates, fidelities, eta, n_grid, seed
        self.evaluations = 0

    def propose(self, ctx: ProposalContext):
        cands = []
        for k in ctx.space.used(ctx.incumbent["cfg"]):
            for v in ctx.space.grid_values(k, self.n_grid):
                if ctx.candidate(k, v) is not None:
                    cands.append((k, v))
        if not cands:
            return None
        rng = np.random.default_rng(self.seed + ctx.turn)
        if len(cands) > self.n_candidates:
            cands = [cands[i] for i in sorted(rng.choice(len(cands), self.n_candidates, replace=False))]
        if ctx.screen is None or len(cands) == 1:
            k, v = cands[0]
            return _mk({k: v}, "halving", "single candidate")
        cfgs = [ctx.candidate(k, v) for k, v in cands]
        res = successive_halving(ctx.screen, cfgs, higher_is_better=ctx.higher_is_better,
                                 fidelities=self.fidelities, eta=self.eta)
        self.evaluations += res["evaluations"]
        if res["winner"] is None:
            return None
        k, v = cands[res["winner_index"]]
        return _mk({k: v}, "halving", f"survived successive halving over {len(cands)} candidates "
                                      f"({res['evaluations']} screening runs)")


# ===================================================================== LLM observer

OUTPUT_FORMAT = """Reply with ONE JSON object and nothing else:
{"what_changed": "<what the last round changed and what it did to the development metrics, with numbers>",
 "surprising": "<anything surprising in the results so far, or 'none'>",
 "leakage_suspicion": "<any sign of leakage, cheating or a too-good-to-be-true result, or 'none'>",
 "proposal": {"change": {"<one key>": <new value>},
              "kind": "<improve = tune the current best | fix = fix a crash or a wasted round | new = try a different method>",
              "hypothesis": "<one sentence: the mechanism you expect>",
              "expected": "<lower | same | higher>  (primary development metric of the new round vs the current best)",
              "falsifier": "<the result that would show your hypothesis is wrong>",
              "rationale": "<why this single change>"}}
The next round = the CURRENT BEST configuration with your single change applied. Rules: exactly one key; allowed keys and
ranges below; the resulting configuration must not equal any configuration already run; keys {fixed} are fixed."""

SYSTEM = ("You are the observer of a research loop. You only read results and propose the next single change. "
          "You never see or ask for held-out results.")


def _m3(m):
    if not m or m.get("value") is None:
        return "-"
    return f"{m['value']:.4f} [{m['lo']:.4f}, {m['hi']:.4f}]"


def _f3(f):
    return f"{f['diff']:+.4f} [{f['lo']:+.4f}, {f['hi']:+.4f}]" if f else "-"


def _metric_line(metrics: dict) -> str:
    return "  ".join(f"{k}={_m3(v)}" for k, v in (metrics or {}).items()
                     if isinstance(v, dict) and v.get("lo") is not None)


def build_prompt(ctx: ProposalContext) -> str:
    """Observer prompt (port of dongu.py istem_yaz): task, baselines, every round's dev metrics, current best, rules,
    output format, allowed space."""
    lines = []
    for g in ctx.history:
        head = f"round {g['n']} [{g['status']}]: changes vs round 0 = {json.dumps(g.get('diff_start', {}), default=str)}"
        if g["status"] == "crash":
            lines.append(head + f"\n  CRASH (no metrics): {g.get('error')}")
            continue
        t = head + f"\n  dev {_metric_line(g.get('metrics'))}"
        if g.get("diff_inc"):
            t += (f"\n  paired {ctx.primary} diff vs best at that time (round {g['inc_before']}; "
                  f"{'positive' if ctx.higher_is_better else 'negative'} = better): {_f3(g['diff_inc'])}")
        p = g.get("proposal") or {}
        if p.get("hypothesis") and p.get("hypothesis") != "-":
            t += f"\n  proposal: expected {p.get('expected')}, observed {g.get('observed')}; hypothesis: {p['hypothesis']}"
        if g.get("repeat_of") is not None:
            t += f"\n  NOTE: per-unit scores identical to round {g['repeat_of']} (wasted round)"
        if g is ctx.history[-1] and g.get("groups"):
            t += f"\n  per-group mean {ctx.primary}: {json.dumps(g['groups'], default=str)}"
        lines.append(t)
    tb = "\n".join(f"{a}: dev {_metric_line(m)}" for a, m in ctx.baselines.items()) or "(none)"
    stop = []
    if ctx.stop_after:
        stop.append(f"after {ctx.stop_after} consecutive rounds that are not kept")
    if ctx.max_turns:
        stop.append(f"after {ctx.max_turns} rounds")
    p = (f"{ctx.task}\n\nPre-registered baselines (development):\n{tb}\n\nRounds so far:\n" + "\n".join(lines) + "\n\n"
         f"CURRENT BEST: round {ctx.incumbent['n']}, full configuration "
         f"{json.dumps(ctx.incumbent['cfg'], ensure_ascii=False, default=str)}\n"
         f"Primary metric: {ctx.primary} ({'higher' if ctx.higher_is_better else 'lower'} is better). A round is kept only "
         f"if it is significantly better than the current best (paired CI entirely "
         f"{'above' if ctx.higher_is_better else 'below'} 0). The loop stops " + (" or ".join(stop) or "on budget") + ".\n\n"
         + OUTPUT_FORMAT.replace("{fixed}", ", ".join(ctx.space.fixed) or "(none)")
         + "\nAllowed space (lists = choices, pairs = [min, max]): "
         + json.dumps(ctx.space.allowed(), ensure_ascii=False, default=str))
    if ctx.error:
        p += f"\n\nYour previous proposal was rejected: {ctx.error}. Propose a different, valid single change."
    return p


def _translate(d):
    if isinstance(d, dict):
        return {_TR_KEYS.get(k, k): (_translate(v) if k not in ("degisiklik", "change") else v) for k, v in d.items()}
    return d


def parse_observer_reply(text: str) -> dict | None:
    """Extract the JSON object (first '{' to last '}'); Turkish keys of the original loop are mapped to English."""
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        d = json.loads(text[a:b + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    d = _translate(d)
    prop = d.get("proposal")
    if isinstance(prop, dict) and prop.get("kind") in _TR_KINDS:
        prop["kind"] = _TR_KINDS[prop["kind"]]
    return d


class LLMProposer:
    """LLM observer. Reads the history (dev metrics only) and proposes one change as JSON with hypothesis, expected
    direction and falsifier. The loop validates the proposal and, if invalid, calls again once with the rejection
    message (repair attempt)."""
    name = "llm"
    requires_fields = True

    def __init__(self, llm: LLM, *, max_tokens: int = 1500, system: str = SYSTEM, forbid_test_refs: bool = True,
                 **complete_kw):
        self.llm, self.max_tokens, self.system = llm, max_tokens, system
        self.forbid_test_refs = forbid_test_refs
        self.complete_kw = complete_kw

    def propose(self, ctx: ProposalContext):
        prompt = build_prompt(ctx)
        r: LLMResponse = self.llm.complete(self.system, prompt, max_tokens=self.max_tokens, **self.complete_kw)
        text = r.text or ""
        out = {"raw": text, "prompt": prompt, "cost_usd": float(getattr(r, "cost_usd", 0.0) or 0.0)}
        if self.forbid_test_refs:
            m = TEST_REFERENCE.search(text)
            if m:
                return {**out, "invalid": f"reply refers to the test split ({m.group(0)!r})"}
        d = parse_observer_reply(text)
        if d is None:
            return {**out, "invalid": "JSON could not be parsed"}
        prop = d.get("proposal")
        if not isinstance(prop, dict):
            return {**out, "invalid": "proposal must be a JSON object"}
        notes = {k: d.get(k) for k in ("what_changed", "surprising", "leakage_suspicion") if k in d}
        return {**prop, **out, "notes": notes}


class _ClaudePCLI:
    """Minimal `claude -p` backend (port of uretim/dongu/kod/gozlemci.sh), used only when simengine.llm.ClaudeCode is
    not available. Tools off, no session persistence, JSON envelope. Subscription usage: cost_usd is 0, the list price
    from the envelope is kept in raw."""

    def __init__(self, model: str = "claude-haiku-5-5", binary: str | None = None, timeout: int = 300,
                 language: str = "English", run: Callable | None = None):
        self.model = model
        self.binary = binary or os.environ.get("CLAUDE_BIN", "claude")
        self.timeout, self.language = timeout, language
        self._run = run or subprocess.run

    def command(self) -> list[str]:
        return [self.binary, "-p", "--model", self.model, "--settings", json.dumps({"language": self.language}),
                "--tools", "", "--strict-mcp-config", "--no-session-persistence", "--output-format", "json"]

    def complete(self, system: str, user: str, *, max_tokens: int = 256, **kw) -> LLMResponse:
        with tempfile.TemporaryDirectory(prefix="simengine-observer-") as cwd:
            r = self._run(self.command(), input=f"{system}\n\n{user}", capture_output=True, text=True,
                          timeout=self.timeout, cwd=cwd)
        try:
            env = json.loads(r.stdout)
        except (json.JSONDecodeError, TypeError):
            raise RuntimeError(f"claude -p returned no JSON envelope: {(r.stderr or '')[-300:]}")
        if env.get("is_error"):
            raise RuntimeError(f"claude -p error: {str(env.get('result'))[:300]}")
        u = env.get("usage", {}) or {}
        return LLMResponse(text=env.get("result", ""), model=self.model,
                           input_tokens=int(u.get("input_tokens", 0)), output_tokens=int(u.get("output_tokens", 0)),
                           cache_read_tokens=int(u.get("cache_read_input_tokens", 0)), cost_usd=0.0,
                           raw={"list_price_usd": env.get("total_cost_usd")})

    def map(self, prompts, *, max_tokens: int = 256, parallel: int = 1, **kw):
        return [self.complete(s, u, max_tokens=max_tokens, **kw) for s, u in prompts]


def claude_code_llm(model: str = "claude-haiku-5-5", **kw):
    """simengine.llm.ClaudeCode if the llm module provides it, else the local `claude -p` fallback."""
    try:
        from ..llm import ClaudeCode  # type: ignore
    except ImportError:
        return _ClaudePCLI(model=model, **kw)
    return ClaudeCode(model=model, **kw)


class ClaudeCodeProposer(LLMProposer):
    """LLM observer through `claude -p` (subscription, tools off). Default model claude-haiku-5-5, as in the loop."""
    name = "claude-p"

    def __init__(self, model: str = "claude-haiku-5-5", *, llm: LLM | None = None, max_tokens: int = 1500, **kw):
        super().__init__(llm or claude_code_llm(model), max_tokens=max_tokens, **kw)


def make_proposer(spec: Any, **kw):
    """'random' | 'grid' | 'greedy' | 'halving' | 'claude-p' | 'claude' (API) | an LLM instance | a proposer object."""
    if hasattr(spec, "propose"):
        return spec
    if isinstance(spec, str):
        s = spec.lower()
        if s == "random":
            return RandomProposer(**kw)
        if s == "grid":
            return GridProposer(**kw)
        if s in ("greedy", "coordinate", "hill-climb"):
            return GreedyCoordinateProposer(**kw)
        if s in ("halving", "successive-halving"):
            return HalvingProposer(**kw)
        if s in ("claude-p", "claude-code", "claudecode"):
            return ClaudeCodeProposer(**kw)
        if s in ("claude", "api"):
            from ..llm import Claude  # type: ignore  # optional; needs the anthropic extra
            model = kw.pop("model", "claude-haiku-5-5")
            return LLMProposer(Claude(model=model), **kw)
        raise ValueError(f"unknown proposer {spec!r}")
    if hasattr(spec, "complete"):
        return LLMProposer(spec, **kw)
    raise TypeError(f"cannot make a proposer from {type(spec).__name__}")


def validate_proposal(prop: dict | None, ctx: ProposalContext, *, require_fields: bool) -> tuple[dict | None, str | None]:
    """Return (full next config, None) or (None, reason). Port of dongu.py oneri_dogrula."""
    if prop is None:
        return None, "no proposal"
    if prop.get("invalid"):
        return None, prop["invalid"]
    ch = prop.get("change")
    if not isinstance(ch, dict):
        return None, "proposal.change must be a JSON object"
    if require_fields:
        missing = [k for k in ("kind", "hypothesis", "expected", "falsifier") if not prop.get(k)]
        if missing:
            return None, f"required fields missing: {missing}"
        if prop["expected"] not in DIRECTIONS or prop["kind"] not in KINDS:
            return None, "expected must be lower/same/higher and kind improve/fix/new"
    if len(ch) != 1:
        return None, f"exactly one key must change, got {len(ch)}"
    (k, v), = ch.items()
    if k in ctx.space.fixed:
        return None, f"{k} is fixed"
    if v in ctx.space.forbidden.get(k, []):
        return None, f"{k}={v!r} is forbidden here"
    if k not in ctx.space.params:
        return None, f"unknown key {k!r}"
    if ctx.space.is_int_range(k) and isinstance(v, float) and v.is_integer():
        v = int(v)
    full, err = ctx.space.validate({**ctx.incumbent["cfg"], k: v})
    if err:
        return None, "; ".join(err)
    if not config_diff(full, ctx.incumbent["cfg"]):
        return None, "no-op: value equals the current best"
    if config_hash(full) in ctx.seen:
        return None, "repeat: this configuration already ran"
    prop["change"] = {k: v}
    return full, None

"""Turn-based LLM agent society: agents act each turn through the LLM protocol, the world state is deterministic,
every turn is logged.

Generalises the research society runner (`abm/kod/yayilim24.py`, `toplum_kos`: 20 personas, one call per turn
listing who shares) into a small world loop:

    world  = ShareCascadeWorld(content="...", times=["1 hour", "6 hours", "24 hours"])
    agents = [SocietyAgent("P01", "19, college student, ..."), ...]
    run = AgentSociety(agents, world, llm, mode="batch").run()   # llm: FakeLLM, Claude or ClaudeCode
    run.log          # one row per (turn, agent): prompt id, raw reply, parsed action, valid flag
    run.states       # world state after every turn

Modes:
* ``per_agent`` : one LLM call per active agent per turn (prompts built from the state at the START of the turn, so
  the order of calls does not matter; calls go through `llm.map`, parallel for API backends).
* ``batch``     : one LLM call per turn that simulates all active agents and returns {agent_id: action} (the
  research design; ~20x fewer calls, agents are not independent samples).

Determinism: the world only changes in `World.step`, which receives the parsed actions sorted by agent id and a
`numpy.random.Generator` seeded from (seed, turn). With a deterministic LLM (FakeLLM, or a disk cache) a re-run
reproduces the log exactly. Invalid replies are retried `retries` times, then the world's default action is used
and the row is flagged `valid=False` (count them; selective failures bias results).

STATUS: the loop is infrastructure. The research society for retweet curves was pre-registered but its evaluation
was not run (no measured result); do not quote any accuracy for it.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass
class SocietyAgent:
    id: str
    persona: str
    traits: dict = field(default_factory=dict)


@runtime_checkable
class World(Protocol):
    """Deterministic environment. `observe` and `state` must not mutate; only `step` changes the world."""

    def active(self, agent_id: str, turn: int) -> bool: ...

    def describe(self) -> str: ...                       # world rules + action format (system prompt part)

    def observe(self, agent_id: str, turn: int) -> str: ...

    def parse(self, agent_id: str, obj: Any) -> Any: ...   # validate one agent's action; raise ValueError if invalid

    def default_action(self, agent_id: str) -> Any: ...

    def step(self, actions: dict, turn: int, rng: np.random.Generator) -> dict: ...   # returns turn events
    # optional: parse_batch(obj, active_ids) -> {agent_id: action obj} for a world-specific batch reply format

    def state(self) -> dict: ...

    def n_turns(self) -> int: ...


def extract_json(text: str):
    """First JSON object or list in a reply (code fences tolerated)."""
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        raise ValueError("no JSON in reply")
    return json.loads(m.group(1))


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


@dataclass
class SocietyRun:
    log: pd.DataFrame
    states: list
    events: list
    calls: int
    cost_usd: float
    invalid: int
    stopped: str


class AgentSociety:
    """World loop. llm: any simengine.types.LLM (FakeLLM, simengine.llm.Claude, simengine.llm.ClaudeCode)."""

    def __init__(self, agents, world: World, llm, mode="per_agent", max_tokens=300, parallel=4, retries=1,
                 max_calls=None, seed=0, llm_kwargs=None):
        if mode not in ("per_agent", "batch"):
            raise ValueError("mode must be 'per_agent' or 'batch'")
        ids = [a.id for a in agents]
        if len(set(ids)) != len(ids):
            raise ValueError("agent ids must be unique")
        self.agents = sorted(agents, key=lambda a: a.id)
        self.world, self.llm, self.mode = world, llm, mode
        self.max_tokens, self.parallel, self.retries = max_tokens, parallel, retries
        self.max_calls, self.seed = max_calls, seed
        self.llm_kwargs = dict(llm_kwargs or {})
        self.calls, self.cost = 0, 0.0

    # -- prompts
    def system_prompt(self, agent: SocietyAgent | None) -> str:
        if agent is None:
            people = "\n".join(f"{a.id}: {a.persona}" for a in self.agents)
            return (f"You simulate a small society of {len(self.agents)} people, each acting on their own profile.\n"
                    f"{self.world.describe()}\nPeople:\n{people}\n"
                    'Answer ONLY with compact JSON: {"<person id>": <action>, ...} for every person listed in the '
                    "turn message.")
        return (f"You are {agent.id}. Profile: {agent.persona}\n{self.world.describe()}\n"
                "Stay in character. Answer ONLY with the JSON action.")

    def _call(self, prompts):
        if self.max_calls is not None and self.calls + len(prompts) > self.max_calls:
            raise _Budget()
        res = self.llm.map(prompts, max_tokens=self.max_tokens, parallel=self.parallel, **self.llm_kwargs)
        self.calls += len(prompts)
        self.cost += float(sum(getattr(r, "cost_usd", 0.0) or 0.0 for r in res))
        return res

    def _turn_per_agent(self, turn, active):
        prompts = [(self.system_prompt(a), self.world.observe(a.id, turn)) for a in active]
        res = self._call(prompts)
        rows, todo = {}, []
        for a, (s, u), r in zip(active, prompts, res):
            rows[a.id] = dict(prompt_id=_hash(s + u), raw=r.text, attempts=1)
            try:
                rows[a.id].update(action=self.world.parse(a.id, extract_json(r.text)), valid=True)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                todo.append((a, s, u))
        for _ in range(self.retries):
            if not todo:
                break
            again = [(s, u + "\nYour previous reply was not valid JSON for the required action. Reply again.")
                     for a, s, u in todo]
            res = self._call(again)
            nxt = []
            for (a, s, u), r in zip(todo, res):
                rows[a.id]["attempts"] += 1
                rows[a.id]["raw"] = r.text
                try:
                    rows[a.id].update(action=self.world.parse(a.id, extract_json(r.text)), valid=True)
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    nxt.append((a, s, u))
            todo = nxt
        for a, _, _ in todo:
            rows[a.id].update(action=self.world.default_action(a.id), valid=False)
        return rows

    def _turn_batch(self, turn, active):
        s = self.system_prompt(None)
        u = "\n\n".join(f"[{a.id}]\n{self.world.observe(a.id, turn)}" for a in active)
        u += "\n\nGive the action of each person listed above."
        rows = {a.id: dict(prompt_id=_hash(s + u), raw=None, attempts=0) for a in active}
        obj = None
        for k in range(1 + self.retries):
            r = self._call([(s, u if k == 0 else u + "\nYour previous reply was not valid JSON. Reply again.")])[0]
            for a in active:
                rows[a.id].update(raw=r.text, attempts=k + 1)
            try:
                obj = extract_json(r.text)
                if isinstance(obj, dict):
                    break
            except (ValueError, json.JSONDecodeError):
                pass
            obj = None
        if obj is not None and hasattr(self.world, "parse_batch"):
            obj = self.world.parse_batch(obj, [a.id for a in active])
        for a in active:
            try:
                if obj is None or a.id not in obj:
                    raise ValueError
                rows[a.id].update(action=self.world.parse(a.id, obj[a.id]), valid=True)
            except (ValueError, KeyError, TypeError):
                rows[a.id].update(action=self.world.default_action(a.id), valid=False)
        return rows

    def run(self, n_turns=None) -> SocietyRun:
        n_turns = self.world.n_turns() if n_turns is None else n_turns
        log, states, events = [], [self.world.state()], []
        stopped = "completed"
        for turn in range(n_turns):
            active = [a for a in self.agents if self.world.active(a.id, turn)]
            if not active:              # nobody left to act: the world still advances (no LLM call)
                events.append(self.world.step({}, turn, np.random.default_rng([self.seed, turn])))
                states.append(self.world.state())
                continue
            try:
                rows = self._turn_per_agent(turn, active) if self.mode == "per_agent" else \
                    self._turn_batch(turn, active)
            except _Budget:
                stopped = f"budget: max_calls={self.max_calls}"
                break
            except Exception as e:          # backend budget guard (e.g. simengine.llm BudgetExceeded): stop cleanly
                if "budget" not in type(e).__name__.lower():
                    raise
                stopped = f"budget: {type(e).__name__}: {e}"[:200]
                break
            actions = {aid: rows[aid]["action"] for aid in sorted(rows)}
            ev = self.world.step(actions, turn, np.random.default_rng([self.seed, turn]))
            events.append(ev)
            states.append(self.world.state())
            for aid in sorted(rows):
                r = rows[aid]
                log.append(dict(turn=turn, agent=aid, prompt_id=r["prompt_id"], attempts=r["attempts"],
                                valid=r["valid"], action=json.dumps(r["action"], sort_keys=True, default=str),
                                raw=r["raw"]))
        lg = pd.DataFrame(log)
        return SocietyRun(log=lg, states=states, events=events, calls=self.calls, cost_usd=self.cost,
                          invalid=int((~lg["valid"]).sum()) if len(lg) else 0, stopped=stopped)


class _Budget(Exception):
    pass


# ------------------------------------------------------------------ ready-made worlds
class ShareCascadeWorld:
    """Research society design (yayilim24 `toplum_kos`): a piece of content; at each checkpoint every agent who has
    not shared yet decides whether to share now, seeing how many have shared so far. State = sorted sharer ids.
    Action: {"share": true|false}. In batch mode the society reply {"P01": {"share": true}, ...} or the research
    format {"share": ["P01", ...]} are both accepted."""

    def __init__(self, content: str, times=("1 hour", "6 hours", "24 hours"), n_agents=None,
                 setting="a social network"):
        self.content, self.times, self.setting = content, list(times), setting
        self.n_agents = n_agents
        self.shared: list[str] = []
        self.curve: list[int] = []

    def n_turns(self):
        return len(self.times)

    def active(self, agent_id, turn):
        return agent_id not in self.shared

    def describe(self):
        return (f"Setting: {self.setting}. You decide whether to share (repost/reply to) a post. Be realistic: most "
                'ordinary posts are shared by few people. Action JSON: {"share": true} or {"share": false}.')

    def observe(self, agent_id, turn):
        k = len(self.shared)
        n = f" of {self.n_agents}" if self.n_agents else ""
        return (f'Post: "{self.content}"\nTime since posting: {self.times[turn]}. People in this group who already '
                f"shared it: {', '.join(self.shared) if self.shared else 'none'} (count {k}{n}).\n"
                "Do you share it now?")

    def parse(self, agent_id, obj):
        if isinstance(obj, dict) and "share" in obj:
            v = obj["share"]
            if isinstance(v, bool):
                return v
            if isinstance(v, list):          # research batch format reached the per-agent parser
                return agent_id in v
        if isinstance(obj, bool):
            return obj
        raise ValueError("expected {'share': bool}")

    def default_action(self, agent_id):
        return False

    def parse_batch(self, obj, active_ids):
        """Research format {"share": [ids]} -> {id: {"share": bool}}; per-id dicts pass through."""
        if isinstance(obj, dict) and isinstance(obj.get("share"), list):
            ids = {x for x in obj["share"] if isinstance(x, str)}
            return {a: {"share": a in ids} for a in active_ids}
        return obj

    def step(self, actions, turn, rng):
        new = sorted(a for a, share in actions.items() if share and a not in self.shared)
        self.shared = sorted(set(self.shared) | set(new))
        self.curve.append(len(self.shared))
        return dict(turn=turn, time=self.times[turn], new=new, total=len(self.shared))

    def state(self):
        return dict(shared=list(self.shared), curve=list(self.curve))


class ChoiceWorld:
    """Repeated choice with social feedback (adoption / opinion / product choice). Each turn every agent picks one
    option, seeing last turn's population shares and an optional per-turn context (e.g. a price). State = per-turn
    choice counts and every agent's current choice. Action: {"choice": "<option>"}."""

    def __init__(self, question: str, options, n_turns=3, contexts=None, initial=None, show_shares=True):
        self.question, self.options = question, list(options)
        self._n = n_turns
        self.contexts = list(contexts) if contexts is not None else [""] * n_turns
        if len(self.contexts) < n_turns:
            raise ValueError("contexts must have one entry per turn")
        self.choice: dict = dict(initial or {})
        self.history: list[dict] = []
        self.show_shares = show_shares

    def n_turns(self):
        return self._n

    def active(self, agent_id, turn):
        return True

    def describe(self):
        opts = ", ".join(f'"{o}"' for o in self.options)
        return f'Question: {self.question}\nOptions: {opts}.\nAction JSON: {{"choice": "<one option>"}}.'

    def shares(self):
        if not self.choice:
            return {}
        vals = list(self.choice.values())
        return {o: round(vals.count(o) / len(vals), 3) for o in self.options}

    def observe(self, agent_id, turn):
        parts = [f"Turn {turn + 1} of {self._n}."]
        if self.contexts[turn]:
            parts.append(self.contexts[turn])
        if agent_id in self.choice:
            parts.append(f"Your current choice: {self.choice[agent_id]}.")
        if self.show_shares and self.choice:
            parts.append("Share of people choosing each option last turn: "
                         + ", ".join(f"{o} {s:.0%}" for o, s in self.shares().items()) + ".")
        parts.append("What do you choose now?")
        return "\n".join(parts)

    def parse(self, agent_id, obj):
        v = obj.get("choice") if isinstance(obj, dict) else obj
        if isinstance(v, str):
            for o in self.options:
                if v.strip().lower() == str(o).lower():
                    return o
        raise ValueError(f"choice must be one of {self.options}")

    def default_action(self, agent_id):
        return self.choice.get(agent_id, self.options[0])

    def step(self, actions, turn, rng):
        changed = sorted(a for a, c in actions.items() if a in self.choice and self.choice[a] != c)
        self.choice.update(actions)
        counts = {o: int(sum(1 for c in self.choice.values() if c == o)) for o in self.options}
        self.history.append(counts)
        return dict(turn=turn, counts=counts, changed=changed)

    def state(self):
        return dict(choice=dict(sorted(self.choice.items())), history=list(self.history))


# ------------------------------------------------------------------ economic-game societies
# Ported from edsl-test (same owner as replicant): experiments/07_who_to_punish/who_to_punish.py (who-to-punish),
# experiments/05_prisoners_dilemma/prisoners_dilemma.py and experiments/03_trust/trust_interaction.py; the
# None-retry rule from experiments/common.py. The EDSL wiring is rewritten onto the simengine LLM protocol:
# the persona goes in as `system`, the game text as `user`, every answer is JSON, re-asked with feedback
# `retries` times and then left INVALID (value None). A missing answer is never turned into $0 (the edsl-test
# bug fixed in its commit 15b9531: "raise on None answers (was fabricating $0)").

WTP_LABELS = ("A", "B", "C", "D")
WTP_DESIGN = dict(
    n_players=4, endowment=10, mpcr=0.4, punish_cost=0.25, max_punish=10,
    ballot={"low": "LESS than the group average", "avg": "exactly the group average",
            "high": "MORE than the group average"},
    vote_options=("Yes", "No", "No preference"),
    majority="an item passes when Yes > No; 'No preference' and invalid ballots are not counted; ties fail",
    payoff="earnings = max(0, (10 - own contribution) + 0.4 * group total - 0.25 * punishment given - "
           "punishment received)",
    citation="Ertan, Page & Putterman (2009), 'Who to punish? Individual decisions and majority rule in "
             "mitigating the free rider problem', European Economic Review 53(5): design constants eq. 2 and "
             "footnote 9 of the 2005 working paper. Simplified port of edsl-test "
             "experiments/07_who_to_punish/who_to_punish.py: one fixed group of 4 (the paper: 40 groups), "
             "`votes` x `periods` (the paper: 3 or 5 votes, 30 periods), fixed labels (the paper reshuffles).",
)
PD_PAYOFFS = {("Cooperate", "Cooperate"): 3, ("Cooperate", "Defect"): 0,
              ("Defect", "Cooperate"): 5, ("Defect", "Defect"): 1}
PD_CITATION = ("repeated prisoner's dilemma as in edsl-test experiments/05_prisoners_dilemma (setup of Mei, Xie, "
               "Yuan & Jackson 2024, PNAS): both C 3/3, both D 1/1, D vs C 5/0, fixed partner")
TRUST_CITATION = ("investment (trust) game, Berg, Dickhaut & McCabe (1995), Games and Economic Behavior 10; "
                  "pot $100 tripled, as in edsl-test experiments/03_trust/trust_interaction.py")
ON_INVALID = ("stop", "abstain", "raise")


class InvalidAnswer(RuntimeError):
    """Raised by the game runners with on_invalid='raise' when an answer stays invalid after the retries."""


def _persona_rows(personas, n, persona_method, prefix="a"):
    """Normalise persona inputs to [{agent_id, persona, traits}] of length n.
    Accepts None (n empty personas = the no-persona baseline), or a list of: str / None (persona text),
    SocietyAgent, simengine.persona.Persona-like objects (`.text()` and `.pid`), or dicts with
    agent_id/id, persona/text and optional traits."""
    if personas is None:
        personas = [None] * n
    personas = list(personas)
    if len(personas) != n:
        raise ValueError(f"expected {n} personas, got {len(personas)}")
    out = []
    for i, p in enumerate(personas):
        aid, text, traits = f"{prefix}{i:03d}", "", {}
        if p is None or isinstance(p, str):
            text = p or ""
        elif isinstance(p, SocietyAgent):
            aid, text, traits = p.id, p.persona or "", dict(p.traits)
        elif isinstance(p, dict):
            aid = str(p.get("agent_id", p.get("id", aid)))
            text = p.get("persona", p.get("text")) or ""
            traits = dict(p.get("traits") or {})
        elif callable(getattr(p, "text", None)):
            aid = str(getattr(p, "pid", None) or aid)
            text = p.text()
            tr = getattr(p, "traits", None)
            if isinstance(tr, (list, tuple)):
                traits = {getattr(t, "label", str(k)): getattr(t, "value", t) for k, t in enumerate(tr)}
        else:
            raise TypeError(f"unsupported persona type {type(p).__name__}")
        out.append(dict(agent_id=aid, persona=str(text), traits=traits, persona_method=persona_method))
    ids = [r["agent_id"] for r in out]
    if len(set(ids)) != len(ids):
        raise ValueError("agent ids must be unique")
    return out


def _whole(obj, key, lo, hi):
    """A whole number in [lo, hi] from {"key": n} or a bare number. Out of range or fractional -> ValueError
    (never clamped: a clamp would silently change the answer)."""
    if isinstance(obj, dict):
        if key not in obj:
            raise ValueError(f'missing "{key}"')
        obj = obj[key]
    if isinstance(obj, bool) or obj is None:
        raise ValueError(f"{key} must be a whole number")
    if isinstance(obj, str):
        obj = float(obj.strip().lstrip("$"))
    x = float(obj)
    if not np.isfinite(x) or x != int(x):
        raise ValueError(f"{key} must be a whole number")
    if not lo <= x <= hi:
        raise ValueError(f"{key} must be between {lo} and {hi}")
    return int(x)


def _reply_obj(text):
    """JSON object in the reply, or the bare stripped text when it has no JSON (e.g. '7' or 'Yes')."""
    try:
        return extract_json(text)
    except (ValueError, json.JSONDecodeError):
        t = (text or "").strip().strip('."\'`')
        if not t or len(t) > 20:
            raise ValueError("no JSON in reply") from None
        return t


def _choice(obj, key, options):
    v = obj.get(key) if isinstance(obj, dict) else obj
    if isinstance(v, str):
        norm = v.strip().lower().replace("_", " ")
        for o in options:
            if norm == o.lower():
                return o
    raise ValueError(f"{key} must be one of {list(options)}")


class _Asker:
    """One stage of questions through llm.map: retries with feedback, call/cost counting, call cap."""

    def __init__(self, llm, max_tokens, parallel, retries, max_calls, llm_kwargs):
        self.llm, self.max_tokens, self.parallel, self.retries = llm, max_tokens, parallel, retries
        self.max_calls, self.llm_kwargs = max_calls, dict(llm_kwargs or {})
        self.calls, self.cost = 0, 0.0

    def _map(self, prompts, kw):
        if self.max_calls is not None and self.calls + len(prompts) > self.max_calls:
            raise _Budget(f"budget: max_calls={self.max_calls}")
        res = self.llm.map(prompts, max_tokens=self.max_tokens, parallel=self.parallel, **kw)
        self.calls += len(prompts)
        self.cost += float(sum(getattr(r, "cost_usd", 0.0) or 0.0 for r in res))
        return res

    def ask(self, items):
        """items: [dict(key, system, user, parse, sample)] -> {key: dict(value, valid, attempts, raw, error)}."""
        out = {it["key"]: dict(value=None, valid=False, attempts=0, raw=None, error=None) for it in items}
        todo = list(items)
        for k in range(1 + self.retries):
            if not todo:
                break
            batches: dict = {}
            for it in todo:
                batches.setdefault(json.dumps(it.get("sample"), default=str), []).append(it)
            nxt = []
            for _, batch in sorted(batches.items()):
                kw = dict(self.llm_kwargs)
                if batch[0].get("sample") is not None and "sample" not in kw:
                    kw["sample"] = batch[0]["sample"]
                prompts = [(it["system"], it["user"] if k == 0 else
                            it["user"] + f"\nYour previous reply was not valid ({out[it['key']]['error']}). "
                                         "Reply again with the JSON only.") for it in batch]
                for it, r in zip(batch, self._map(prompts, kw)):
                    o = out[it["key"]]
                    o.update(attempts=o["attempts"] + 1, raw=r.text)
                    try:
                        o.update(value=it["parse"](_reply_obj(r.text)), valid=True, error=None)
                    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                        o["error"] = str(e)[:120] or type(e).__name__
                        nxt.append(it)
            todo = nxt
        return out


def _run_guard(fn):
    """Run fn(); map our call cap and backend budget errors to a 'stopped' string."""
    try:
        fn()
        return None
    except _Budget as e:
        return str(e.args[0]) if e.args else "budget"
    except Exception as e:                      # backend budget guard (simengine.llm BudgetExceeded): stop cleanly
        if "budget" not in type(e).__name__.lower():
            raise
        return f"budget: {type(e).__name__}: {e}"[:200]


def _check_common(on_invalid, retries, allowed=ON_INVALID):
    if on_invalid not in allowed:
        raise ValueError(f"on_invalid must be one of {allowed}")
    if retries < 0:
        raise ValueError("retries must be >= 0")


def _wtp_rule_text(rule):
    if not any(rule.values()):
        return "The group voted to allow NO punishment at all."
    bits = [f"{WTP_DESIGN['ballot'][k]}: {'YES' if rule[k] else 'NO'}" for k in WTP_DESIGN["ballot"]]
    return "Voted rules - may a player be punished for contributing... " + "; ".join(bits)


def _wtp_rules(me):
    d = WTP_DESIGN
    return (f"You are Player {me} in a fixed group of {d['n_players']} playing a repeated public goods game. Each "
            f"period every player receives ${d['endowment']} and privately decides how much to put into a group "
            f"fund. Every dollar in the fund pays ${d['mpcr']:.2f} to EACH of the {d['n_players']} players. You keep "
            f"whatever you do not contribute. After contributions are revealed, punishment may happen: paying "
            f"${d['punish_cost']:.2f} reduces another player's earnings by $1.00 - but only players of the kinds "
            f"your group voted punishable may be punished.\n")


def _history(log):
    if not log:
        return "This is the start of the game; there is no history yet."
    return "History so far:\n" + "\n".join(log)


def _category(c, avg):
    return "low" if c < avg else "high" if c > avg else "avg"


def _money(x):
    return int(x) if float(x) == int(x) else round(float(x), 2)


def run_who_to_punish(llm, personas=None, *, groups=1, seed=0, periods=3, votes=2, persona_method="baseline",
                      max_tokens=1024, retries=2, on_invalid="stop", parallel=4, max_calls=None, llm_kwargs=None,
                      sample_key=True) -> dict:
    """Multi-period who-to-punish society (Ertan, Page & Putterman 2009; port of edsl-test
    experiments/07_who_to_punish). Groups of 4 play a public goods game with costly punishment; before every
    block of `periods` periods each player votes Yes / No / No preference on 3 ballot items (may players who
    contributed less than / exactly / more than the group average be punished?). Strict majority of Yes over No
    passes an item, ties fail. Each period: contributions (0-10) -> revealed -> each allowed (punisher, target)
    pair chooses a punishment of 0-10 dollars at $0.25 per $1. The persona is re-sent as `system` on every call.

    personas: None (4*groups no-persona agents) or a list of 4*groups personas (see `_persona_rows`); group g
    takes personas[4g:4g+4] as players A-D. llm: any simengine LLM (FakeLLM in tests). seed is recorded on
    every row and, with sample_key=True, passed as the backend `sample` key (seed*1_000_003 + group) so that
    a disk cache does not serve one group's answers to another group or seed.

    on_invalid (an answer still invalid after `retries` re-asks; its value stays None and valid=False):
      'stop'    (default) that group stops at that stage; completed periods are kept, the group is marked
                incomplete. Other groups go on.
      'abstain' an invalid ballot is not counted; an invalid contribution makes the player sit out that period
                (not in the fund, the average or the punishable set; earnings None); an invalid punishment
                answer deducts nothing. Every case is counted in `invalid` and flagged on its row.
      'raise'   raise InvalidAnswer (the edsl-test behaviour after its fix).

    Returns a JSON-serialisable dict:
      design, meta                      constants, citation; model, seed, persona_method, max_tokens, ...
      agents                            one row per player (group, label, agent_id, persona, traits)
      ballots, contributions, punishments   one row per answer (value None when invalid), with agent_id, seed,
                                        persona_method, traits, max_tokens, wording_id, valid, attempts, raw
      rules                             one row per (group, vote): low/avg/high outcome, punish_low, tallies
      periods                           one row per completed period: contrib, total, avg, given, received,
                                        earnings ({label: value}), punish_low, sat_out
      groups                            per group, score-ready for games.wtp_scorecard: votes (punish-low
                                        outcome per vote, the edsl-test score_wtp shape), rules, periods,
                                        punishments, dismantled, ever_punish_high, complete, stopped,
                                        n_periods_planned / n_periods_completed, invalid
      invalid                           counts of answers left invalid: ballots, contributions, punishments, total
      calls, cost_usd, complete, stopped
    """
    _check_common(on_invalid, retries)
    if groups < 1 or periods < 1 or votes < 1:
        raise ValueError("groups, periods and votes must be >= 1")
    d = WTP_DESIGN
    n = d["n_players"]
    rows = _persona_rows(personas, n * groups, persona_method)
    model = getattr(llm, "model", "-")
    asker = _Asker(llm, max_tokens, parallel, retries, max_calls, llm_kwargs)
    meta_cols = dict(model=model, seed=seed, persona_method=persona_method, max_tokens=max_tokens, wording_id=0)

    G = []
    agents = []
    for g in range(groups):
        players = {L: rows[g * n + i] for i, L in enumerate(WTP_LABELS)}
        for L, r in players.items():
            agents.append(dict(group=g, label=L, **r))
        G.append(dict(group=g, players=players, logs={L: [] for L in WTP_LABELS}, active=True, stopped="completed",
                      votes=[], rules=[], periods=[], punishments=[], sample=(seed * 1_000_003 + g) if sample_key
                      else None, invalid=dict(ballots=0, contributions=0, punishments=0)))
    ballots, contributions, punishments, rules_rows, period_rows = [], [], [], [], []

    def row_meta(grp, L):
        p = grp["players"][L]
        return dict(group=grp["group"], label=L, agent_id=p["agent_id"], traits=p["traits"], **meta_cols)

    def fail(grp, what):
        if on_invalid == "raise":
            raise InvalidAnswer(f"group {grp['group']}: {what} still invalid after {retries} retries")
        grp["active"], grp["stopped"] = False, f"invalid: {what}"

    def stage_votes(v):
        items = []
        for grp in [x for x in G if x["active"]]:
            for L in WTP_LABELS:
                for k, text in d["ballot"].items():
                    user = (_wtp_rules(L) + _history(grp["logs"][L]) + "\n"
                            f"Your group now votes (majority rule) on the punishment rules for the next {periods} "
                            f"periods.\nBallot item: should it be ALLOWED to punish a player who contributed "
                            f"{text}?\n"
                            'Answer ONLY with JSON: {"vote": "Yes"}, {"vote": "No"} or {"vote": "No preference"}.')
                    items.append(dict(key=(grp["group"], L, k), system=grp["players"][L]["persona"], user=user,
                                      parse=lambda o: _choice(o, "vote", d["vote_options"]), sample=grp["sample"]))
        res = asker.ask(items)
        for grp in [x for x in G if x["active"]]:
            tally = {k: dict(yes=0, no=0, no_preference=0, invalid=0) for k in d["ballot"]}
            bad = []
            for L in WTP_LABELS:
                for k in d["ballot"]:
                    r = res[(grp["group"], L, k)]
                    ballots.append(dict(**row_meta(grp, L), vote=v, item=k, answer=r["value"], valid=r["valid"],
                                        attempts=r["attempts"], raw=r["raw"]))
                    if not r["valid"]:
                        tally[k]["invalid"] += 1
                        grp["invalid"]["ballots"] += 1
                        bad.append(f"ballot vote {v} player {L} item {k}")
                    else:
                        tally[k][r["value"].lower().replace(" ", "_")] += 1
            if bad and on_invalid != "abstain":
                fail(grp, bad[0])
                continue
            rule = {k: tally[k]["yes"] > tally[k]["no"] for k in d["ballot"]}
            grp["rule"] = rule
            grp["votes"].append(bool(rule["low"]))
            grp["rules"].append(dict(rule))
            rules_rows.append(dict(group=grp["group"], vote=v, **rule, punish_low=bool(rule["low"]),
                                   tally={k: dict(t) for k, t in tally.items()}))
            for L in WTP_LABELS:
                grp["logs"][L].append(f"Vote {v} outcome: {_wtp_rule_text(rule)}")

    def stage_period(v, pno):
        live = [x for x in G if x["active"]]
        items = []
        for grp in live:
            for L in WTP_LABELS:
                user = (_wtp_rules(L) + _wtp_rule_text(grp["rule"]) + "\n" + _history(grp["logs"][L]) + "\n"
                        f"This is period {pno}. How much of your ${d['endowment']} do you contribute to the group "
                        f"fund (0-{d['endowment']})?\n"
                        f'Answer ONLY with JSON: {{"contribution": <whole number 0-{d["endowment"]}>}}.')
                items.append(dict(key=(grp["group"], L), system=grp["players"][L]["persona"], user=user,
                                  parse=lambda o: _whole(o, "contribution", 0, d["endowment"]), sample=grp["sample"]))
        res = asker.ask(items)
        state = {}
        for grp in live:
            contrib, bad = {}, []
            for L in WTP_LABELS:
                r = res[(grp["group"], L)]
                contributions.append(dict(**row_meta(grp, L), period=pno, vote=v, contribution=r["value"],
                                          valid=r["valid"], attempts=r["attempts"], raw=r["raw"]))
                if r["valid"]:
                    contrib[L] = r["value"]
                else:
                    grp["invalid"]["contributions"] += 1
                    bad.append(L)
            if bad and on_invalid != "abstain":
                fail(grp, f"contribution period {pno} player {bad[0]}")
                continue
            if not contrib:                      # everybody sat out: nothing to play this period
                fail(grp, f"contribution period {pno}: no valid contribution")
                continue
            total = sum(contrib.values())
            avg = total / len(contrib)
            shown = sorted(contrib.values(), reverse=True)
            pairs = [(p, t) for p in contrib for t in contrib if p != t and grp["rule"][_category(contrib[t], avg)]]
            state[grp["group"]] = dict(contrib=contrib, total=total, avg=avg, shown=shown, pairs=pairs, sat_out=bad)
        # punishment stage (only the (punisher, target) pairs the voted rule allows)
        items = []
        for grp in live:
            st = state.get(grp["group"])
            if st is None:
                continue
            missing = f" ({', '.join(st['sat_out'])} gave no valid answer and sat out)" if st["sat_out"] else ""
            for p, t in st["pairs"]:
                user = (_wtp_rules(p) + _history(grp["logs"][p]) + "\n"
                        f"This period you contributed ${st['contrib'][p]}. All contributions were {st['shown']}"
                        f"{missing} (average {round(st['avg'], 2)}).\n"
                        f"Player {t} contributed ${st['contrib'][t]}, which the voted rules allow you to punish. "
                        f"By how many dollars (0-{d['max_punish']}) do you reduce their earnings? Each $1 reduction "
                        f"costs you ${d['punish_cost']:.2f}.\n"
                        f'Answer ONLY with JSON: {{"punish": <whole number 0-{d["max_punish"]}>}}.')
                items.append(dict(key=(grp["group"], p, t), system=grp["players"][p]["persona"], user=user,
                                  parse=lambda o: _whole(o, "punish", 0, d["max_punish"]), sample=grp["sample"]))
        res = asker.ask(items) if items else {}
        for grp in live:
            st = state.get(grp["group"])
            if st is None:
                continue
            given = {L: 0 for L in st["contrib"]}
            received = {L: 0 for L in st["contrib"]}
            bad = []
            for p, t in st["pairs"]:
                r = res[(grp["group"], p, t)]
                pr = dict(**row_meta(grp, p), period=pno, vote=v, punisher=p, target=t,
                          target_id=grp["players"][t]["agent_id"], my_contribution=st["contrib"][p],
                          target_contribution=st["contrib"][t], avg=st["avg"],
                          target_category=_category(st["contrib"][t], st["avg"]), amount=r["value"],
                          valid=r["valid"], attempts=r["attempts"], raw=r["raw"])
                punishments.append(pr)
                grp["punishments"].append(pr)
                if r["valid"]:
                    given[p] += r["value"]
                    received[t] += r["value"]
                else:
                    grp["invalid"]["punishments"] += 1
                    bad.append(f"punishment period {pno} player {p} -> {t}")
            if bad and on_invalid != "abstain":
                fail(grp, bad[0])
                continue
            earnings = {}
            for L in WTP_LABELS:
                if L not in st["contrib"]:
                    earnings[L] = None
                    grp["logs"][L].append(f"Period {pno}: you gave no valid contribution and sat out this period.")
                    continue
                pre = d["endowment"] - st["contrib"][L] + d["mpcr"] * st["total"]
                earnings[L] = round(max(0.0, pre - d["punish_cost"] * given[L] - received[L]), 6)
                grp["logs"][L].append(
                    f"Period {pno}: you contributed ${st['contrib'][L]}, all contributions were {st['shown']} "
                    f"(avg {round(st['avg'], 2)}); you spent ${d['punish_cost'] * given[L]:.2f} on punishing and "
                    f"received ${_money(received[L])} of punishment; you earned ${earnings[L]:.2f}.")
            prow = dict(group=grp["group"], period=pno, vote=v, rule=dict(grp["rule"]),
                        punish_low=bool(grp["rule"]["low"]), contrib=dict(st["contrib"]), total=st["total"],
                        avg=st["avg"], given=given, received=received, earnings=earnings, sat_out=list(st["sat_out"]),
                        n_punish_pairs=len(st["pairs"]))
            period_rows.append(prow)
            grp["periods"].append(prow)

    def play():
        pno = 0
        for v in range(1, votes + 1):
            if not any(x["active"] for x in G):
                return
            stage_votes(v)
            for _ in range(periods):
                pno += 1
                if not any(x["active"] for x in G):
                    return
                stage_period(v, pno)

    budget_stop = _run_guard(play)
    if budget_stop:
        for grp in G:
            if grp["active"]:
                grp["active"], grp["stopped"] = False, budget_stop
    out_groups = []
    for grp in G:
        vs = grp["votes"]
        n_done = len(grp["periods"])
        out_groups.append(dict(
            group=grp["group"], agent_ids={L: p["agent_id"] for L, p in grp["players"].items()},
            votes=list(vs), rules=grp["rules"], periods=grp["periods"], punishments=grp["punishments"],
            dismantled=bool(len(vs) >= 2 and vs[0] and not vs[-1]),
            ever_punish_high=any(r["high"] for r in grp["rules"]),
            n_periods_planned=votes * periods, n_periods_completed=n_done,
            complete=grp["stopped"] == "completed" and n_done == votes * periods,
            stopped=grp["stopped"], invalid=dict(grp["invalid"], total=sum(grp["invalid"].values()))))
    inv = {k: sum(g["invalid"][k] for g in out_groups) for k in ("ballots", "contributions", "punishments")}
    inv["total"] = sum(inv.values())
    stops = sorted({g["stopped"] for g in out_groups if g["stopped"] != "completed"})
    return dict(
        game="who_to_punish",
        design={k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, tuple) else v)
                for k, v in WTP_DESIGN.items()},
        meta=dict(meta_cols, groups=groups, votes=votes, periods_per_vote=periods, retries=retries,
                  on_invalid=on_invalid, sample_key=sample_key),
        agents=agents, ballots=ballots, contributions=contributions, punishments=punishments, rules=rules_rows,
        periods=period_rows, groups=out_groups, invalid=inv, calls=asker.calls, cost_usd=asker.cost,
        complete=all(g["complete"] for g in out_groups), stopped="; ".join(stops) if stops else "completed")


def _pair_meta(rows, i, model, seed, persona_method, max_tokens):
    return dict(agent_id=rows[i]["agent_id"], traits=rows[i]["traits"], model=model, seed=seed,
                persona_method=persona_method, max_tokens=max_tokens, wording_id=0)


def run_repeated_pd(llm, personas=None, *, pairs=1, rounds=5, seed=0, persona_method="baseline", max_tokens=1024,
                    retries=2, on_invalid="stop", parallel=4, max_calls=None, llm_kwargs=None,
                    sample_key=True) -> dict:
    """Repeated prisoner's dilemma with a fixed partner (port of edsl-test experiments/05_prisoners_dilemma).
    personas: None or 2*pairs personas; pair i = personas[2i] (player A) and personas[2i+1] (player B). Each
    round both choose Cooperate or Defect simultaneously, seeing the full history. Payoffs PD_PAYOFFS.
    An answer invalid after the retries stays None; on_invalid='stop' ends that pair at that round (the round
    is not scored), 'raise' raises InvalidAnswer.

    Returns dict: design, meta, agents, moves (one row per player-round: move, opponent_move, payoff, valid,
    attempts, raw, agent_id, seed, persona_method, ...), pairs (per pair: moves_a, moves_b, score_a, score_b,
    rounds_completed, complete, stopped), cooperation_by_round (share of valid moves that cooperate), invalid,
    calls, cost_usd, complete, stopped."""
    _check_common(on_invalid, retries, ("stop", "raise"))
    if pairs < 1 or rounds < 1:
        raise ValueError("pairs and rounds must be >= 1")
    rows = _persona_rows(personas, 2 * pairs, persona_method)
    model = getattr(llm, "model", "-")
    asker = _Asker(llm, max_tokens, parallel, retries, max_calls, llm_kwargs)
    P = [dict(pair=i, a=2 * i, b=2 * i + 1, ma=[], mb=[], active=True, stopped="completed",
              sample=(seed * 1_000_003 + i) if sample_key else None) for i in range(pairs)]
    moves = []
    invalid = [0]

    def hist(mine, theirs):
        if not mine:
            return "This is the first round; there is no history yet."
        return "History so far:\n" + "\n".join(f"Round {k}: you chose {x}, your opponent chose {y}."
                                               for k, (x, y) in enumerate(zip(mine, theirs), 1))

    def play():
        for rnd in range(1, rounds + 1):
            live = [p for p in P if p["active"]]
            if not live:
                return
            items = []
            for p in live:
                for side, me, mine, theirs in (("A", p["a"], p["ma"], p["mb"]), ("B", p["b"], p["mb"], p["ma"])):
                    user = (f"You are Player {side} playing a {rounds}-round game against the same opponent. Each "
                            "round you both simultaneously choose Cooperate or Defect.\nPayoffs per round: both "
                            "Cooperate -> $3 each; both Defect -> $1 each; if one Defects while the other "
                            "Cooperates -> defector gets $5, cooperator $0.\n" + hist(mine, theirs) + "\n"
                            f"This is round {rnd} of {rounds}. What do you choose?\n"
                            'Answer ONLY with JSON: {"move": "Cooperate"} or {"move": "Defect"}.')
                    items.append(dict(key=(p["pair"], side), system=rows[me]["persona"], user=user,
                                      parse=lambda o: _choice(o, "move", ("Cooperate", "Defect")), sample=p["sample"]))
            res = asker.ask(items)
            for p in live:
                ra, rb = res[(p["pair"], "A")], res[(p["pair"], "B")]
                ok = ra["valid"] and rb["valid"]
                for side, me, r, o in (("A", p["a"], ra, rb), ("B", p["b"], rb, ra)):
                    pay = PD_PAYOFFS[(r["value"], o["value"])] if ok else None
                    moves.append(dict(pair=p["pair"], round=rnd, label=side,
                                      **_pair_meta(rows, me, model, seed, persona_method, max_tokens),
                                      move=r["value"], opponent_move=o["value"], payoff=pay, valid=r["valid"],
                                      attempts=r["attempts"], raw=r["raw"]))
                    invalid[0] += 0 if r["valid"] else 1
                if not ok:
                    what = f"pair {p['pair']}: move round {rnd} player {'A' if not ra['valid'] else 'B'}"
                    if on_invalid == "raise":
                        raise InvalidAnswer(f"{what} still invalid after {retries} retries")
                    p["active"], p["stopped"] = False, f"invalid: {what}"
                    continue
                p["ma"].append(ra["value"])
                p["mb"].append(rb["value"])

    stop = _run_guard(play)
    out_pairs = []
    for p in P:
        if stop and p["active"]:
            p["stopped"] = stop
        sa = sum(PD_PAYOFFS[(x, y)] for x, y in zip(p["ma"], p["mb"]))
        sb = sum(PD_PAYOFFS[(y, x)] for x, y in zip(p["ma"], p["mb"]))
        out_pairs.append(dict(pair=p["pair"], agent_ids=(rows[p["a"]]["agent_id"], rows[p["b"]]["agent_id"]),
                              moves_a=list(p["ma"]), moves_b=list(p["mb"]), score_a=sa, score_b=sb,
                              rounds_completed=len(p["ma"]), complete=p["stopped"] == "completed" and
                              len(p["ma"]) == rounds, stopped=p["stopped"]))
    coop = {}
    for r in range(1, rounds + 1):
        vals = [m["move"] for m in moves if m["round"] == r and m["valid"]]
        coop[r] = (sum(v == "Cooperate" for v in vals) / len(vals)) if vals else None
    stops = sorted({p["stopped"] for p in out_pairs if p["stopped"] != "completed"})
    return dict(game="repeated_pd", design=dict(rounds=rounds, payoffs={f"{a}/{b}": v for (a, b), v in
                                                                       PD_PAYOFFS.items()}, citation=PD_CITATION),
                meta=dict(model=model, seed=seed, persona_method=persona_method, max_tokens=max_tokens, wording_id=0,
                          pairs=pairs, retries=retries, on_invalid=on_invalid, sample_key=sample_key),
                agents=[dict(pair=i // 2, label="AB"[i % 2], **r) for i, r in enumerate(rows)], moves=moves,
                pairs=out_pairs, cooperation_by_round=coop, invalid=dict(moves=invalid[0], total=invalid[0]),
                calls=asker.calls, cost_usd=asker.cost, complete=all(p["complete"] for p in out_pairs),
                stopped="; ".join(stops) if stops else "completed")


def run_trust_interaction(llm, personas=None, *, pairs=1, seed=0, pot=100, multiplier=3, persona_method="baseline",
                          max_tokens=1024, retries=2, on_invalid="stop", parallel=4, max_calls=None, llm_kwargs=None,
                          sample_key=True) -> dict:
    """Two-stage trust game (Berg, Dickhaut & McCabe 1995; port of edsl-test experiments/03_trust). Pair i:
    personas[2i] is the investor, personas[2i+1] the trustee. The investor sends a whole amount 0..pot; it is
    multiplied by `multiplier`; the trustee, told the received amount, returns a whole amount 0..received.
    A return larger than the received amount is invalid (re-asked, never clamped). An investor whose answer stays
    invalid has no trustee stage; with on_invalid='raise' InvalidAnswer is raised instead.

    Returns dict: design, meta, agents, games (one row per pair: investor_id, trustee_id, sent, received,
    returned, investor_payoff, trustee_payoff, valid_sent, valid_returned, attempts, raw, seed, ...), invalid,
    summary (means over valid games; n_valid / n_planned), calls, cost_usd, complete, stopped."""
    _check_common(on_invalid, retries, ("stop", "raise"))
    if pairs < 1:
        raise ValueError("pairs must be >= 1")
    rows = _persona_rows(personas, 2 * pairs, persona_method)
    model = getattr(llm, "model", "-")
    asker = _Asker(llm, max_tokens, parallel, retries, max_calls, llm_kwargs)
    games = [dict(pair=i, investor_id=rows[2 * i]["agent_id"], trustee_id=rows[2 * i + 1]["agent_id"],
                  investor_traits=rows[2 * i]["traits"], trustee_traits=rows[2 * i + 1]["traits"], model=model,
                  seed=seed, persona_method=persona_method, max_tokens=max_tokens, wording_id=0, sent=None,
                  received=None, returned=None, investor_payoff=None, trustee_payoff=None, valid_sent=False,
                  valid_returned=False, attempts_sent=0, attempts_returned=0, raw_sent=None, raw_returned=None)
             for i in range(pairs)]
    smp = (lambda i: seed * 1_000_003 + i) if sample_key else (lambda i: None)

    def fail(what):
        if on_invalid == "raise":
            raise InvalidAnswer(f"{what} still invalid after {retries} retries")

    def play():
        res = asker.ask([dict(key=i, system=rows[2 * i]["persona"], sample=smp(i),
                              user=(f"You have ${pot}. You can send any amount to another person. Whatever you send "
                                    f"is multiplied by {multiplier} before they receive it. They then decide how "
                                    f"much, if any, to send back to you. How much do you send?\n"
                                    f'Answer ONLY with JSON: {{"sent": <whole number 0-{pot}>}}.'),
                              parse=lambda o: _whole(o, "sent", 0, pot)) for i in range(pairs)])
        for g in games:
            r = res[g["pair"]]
            g.update(sent=r["value"], valid_sent=r["valid"], attempts_sent=r["attempts"], raw_sent=r["raw"])
            if r["valid"]:
                g["received"] = r["value"] * multiplier
            else:
                fail(f"pair {g['pair']}: investor amount")
        live = [g for g in games if g["valid_sent"]]
        items = []
        for g in live:
            rec = g["received"]
            items.append(dict(key=g["pair"], system=rows[2 * g["pair"] + 1]["persona"], sample=smp(g["pair"]),
                              user=(f"Another person had ${pot} and sent you part of it, which was multiplied by "
                                    f"{multiplier} to ${rec}. You may keep all of it or send some back to them. How "
                                    f"much do you send back?\n"
                                    f'Answer ONLY with JSON: {{"returned": <whole number 0-{rec}>}}.'),
                              parse=(lambda rec: lambda o: _whole(o, "returned", 0, rec))(rec)))
        res = asker.ask(items) if items else {}
        for g in live:
            r = res[g["pair"]]
            g.update(returned=r["value"], valid_returned=r["valid"], attempts_returned=r["attempts"],
                     raw_returned=r["raw"])
            if r["valid"]:
                g["investor_payoff"] = pot - g["sent"] + r["value"]
                g["trustee_payoff"] = g["received"] - r["value"]
            else:
                fail(f"pair {g['pair']}: trustee return")

    stop = _run_guard(play)
    done = [g for g in games if g["valid_sent"] and g["valid_returned"]]
    mean = (lambda xs: float(np.mean(xs)) if xs else None)
    inv_s = sum(1 for g in games if g["attempts_sent"] and not g["valid_sent"])
    inv_r = sum(1 for g in games if g["attempts_returned"] and not g["valid_returned"])
    return dict(game="trust_interaction", design=dict(pot=pot, multiplier=multiplier, citation=TRUST_CITATION),
                meta=dict(model=model, seed=seed, persona_method=persona_method, max_tokens=max_tokens, wording_id=0,
                          pairs=pairs, retries=retries, on_invalid=on_invalid, sample_key=sample_key),
                agents=[dict(pair=i // 2, role=("investor", "trustee")[i % 2], **r) for i, r in enumerate(rows)],
                games=games, invalid=dict(sent=inv_s, returned=inv_r, total=inv_s + inv_r),
                summary=dict(n_planned=pairs, n_valid=len(done),
                             mean_sent_share=mean([g["sent"] / pot for g in games if g["valid_sent"]]),
                             mean_return_share=mean([g["returned"] / g["received"] for g in done if g["received"]])),
                calls=asker.calls, cost_usd=asker.cost, complete=len(done) == pairs and not stop,
                stopped=stop or ("completed" if len(done) == pairs else "invalid: see games"))

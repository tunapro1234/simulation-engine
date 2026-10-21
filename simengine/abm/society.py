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

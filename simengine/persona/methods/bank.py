"""Append-only agent bank: the SAME virtual person plays every game (edsl-test benchmark/beebench/b01/agents.py).

An existing id is never re-bound to another persona. Growing a bank from k to n agents appends n - k fresh
agents drawn with the increment-specific seed `seed_base + k`, so entries 0..k-1 stay byte-identical. This
also protects the quota/balanced samplers (homo_silicus, construction), whose `sample(n)` composition depends
on n by design. Trade-off (as in the source): a grown bank's composition only approximates one single-draw
quota for mixture methods; id stability is the binding contract.

Ids are `f"{method}-{i:03d}"`, as in the source bank files. With `path`, the bank is kept in a JSON file
(written atomically); the source's `{agent_id: text}` files are read as well.
"""
from __future__ import annotations

import json
import os
from typing import Any

from ._base import PersonaMethod


class AgentBank:
    def __init__(self, method: str | PersonaMethod, seed_base: int = 1000, path: str | os.PathLike | None = None,
                 data_root: str | None = None):
        from . import get_method
        self.method = get_method(method)
        self.name = self.method.name
        self.seed_base = int(seed_base)
        self.path = os.fspath(path) if path is not None else None
        self.data_root = data_root
        self.agents: list[dict] = []
        if self.path and os.path.exists(self.path):
            self._load()

    # ---------------------------------------------------------- persistence
    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and "agents" in d:
            if d.get("method") not in (None, self.name):
                raise ValueError(f"{self.path} holds a {d.get('method')!r} bank, not {self.name!r}")
            if d.get("seed_base") not in (None, self.seed_base):
                raise ValueError(f"{self.path} was grown with seed_base={d.get('seed_base')}, not {self.seed_base}")
            self.agents = list(d["agents"])
        elif isinstance(d, dict):  # edsl-test format {agent_id: text}
            self.agents = [{"id": k, "text": v, "persona_method": self.name, "params": {}, "seed": None}
                           for k, v in d.items()]
        else:
            raise ValueError(f"{self.path}: not an agent bank")

    def save(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"method": self.name, "seed_base": self.seed_base, "agents": self.agents}, f, indent=1,
                      ensure_ascii=False)
        os.replace(tmp, self.path)

    # ---------------------------------------------------------- growth
    def grow(self, n: int) -> "AgentBank":
        """Ensure at least n agents (append-only)."""
        k = len(self.agents)
        if n <= k:
            return self
        seed = self.seed_base + k
        draw = getattr(self.method, "draw", None)
        if callable(draw):
            pairs = draw(n - k, seed, self.data_root)
        else:
            pairs = [(t, {}) for t in self.method.sample(n - k, seed, self.data_root)]
        for i, (text, params) in enumerate(pairs, start=k):
            self.agents.append({"id": f"{self.name}-{i:03d}", "text": text, "persona_method": self.name,
                                "params": dict(params), "seed": seed})
        self.save()
        return self

    def get(self, n: int) -> dict[str, str]:
        """{agent_id: persona_text} for the first n agents (grows the bank when needed)."""
        self.grow(n)
        return {a["id"]: a["text"] for a in self.agents[:n]}

    def personas(self, n: int) -> list[dict[str, Any]]:
        """The first n agents as flat persona dicts (`id`, `text`, `persona_method`, `params`, `seed`)."""
        self.grow(n)
        return [dict(a) for a in self.agents[:n]]

    def __len__(self) -> int:
        return len(self.agents)

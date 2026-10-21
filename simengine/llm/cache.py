"""Local disk cache: the same (model, system, user, params) is never sent twice.

Ported from talp73/kod/llm/openrouter.py + claude_api.py (`_yol`: sha256 over the request, one
JSON file per key under <dir>/<model>/) and sahne/kod/sahne/kaynak.py (`Onbellek`). Writes are
atomic (tmp file + os.replace), so parallel threads and processes can share one directory.
Only successful, validated answers are cached; refusals and errors are not.
`cache_control` is not part of the key (same prompt -> same record whether or not it is cached).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any


def cache_key(model: str, system: str, user: str, params: dict | None = None) -> str:
    """sha256 hex over a canonical JSON of model + system + user + params (sorted keys)."""
    blob = json.dumps({"model": model, "system": system, "user": user, "params": params or {}},
                      sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "_"


class DiskCache:
    """`get(key) -> dict | None`, `put(key, record)`. Records are JSON dicts (text, usage, ...)."""

    def __init__(self, directory: str | os.PathLike):
        self.dir = Path(directory)

    def path(self, key: str, model: str = "") -> Path:
        return self.dir / _safe(model or "any") / f"{key}.json"

    def get(self, key: str, model: str = "") -> dict | None:
        p = self.path(key, model)
        try:
            with p.open(encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError):
            return None          # half-written or corrupt file: treat as a miss

    def put(self, key: str, record: dict[str, Any], model: str = "") -> Path:
        p = self.path(key, model)
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, default=str)
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return p

    def __len__(self) -> int:
        return sum(1 for _ in self.dir.glob("*/*.json")) if self.dir.exists() else 0


__all__ = ["DiskCache", "cache_key"]

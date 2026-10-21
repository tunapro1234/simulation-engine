"""Configuration space for the research loop (ported from kosucu.dogrula / *_UZAY / dongu.py kullanim, bilesen).

A space maps each tunable key to either a list of choices or a (lo, hi) range. A range whose two ends are ints is an
integer range. `fixed` keys may appear in the config but never change. `used(cfg)` tells which keys matter under a
config (a change to an unused key is a wasted turn), `complexity(cfg)` feeds the optional simplicity rule,
`check(cfg)` adds task-specific validation errors.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Sequence

import numpy as np


def _plain(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


class Space:
    def __init__(self, params: dict[str, Sequence | tuple], *, defaults: dict | None = None,
                 fixed: Sequence[str] = (), forbidden: dict[str, Sequence] | None = None,
                 used: Callable[[dict], Sequence[str]] | None = None,
                 complexity: Callable[[dict], float] | None = None,
                 check: Callable[[dict], Sequence[str]] | None = None):
        self.params = {}
        for k, v in params.items():
            if isinstance(v, tuple):
                if len(v) != 2:
                    raise ValueError(f"range for {k!r} must be (lo, hi)")
                self.params[k] = (v[0], v[1])
            else:
                self.params[k] = list(v)
        self.defaults = dict(defaults or {})
        self.fixed = list(fixed)
        self.forbidden = {k: list(v) for k, v in (forbidden or {}).items()}
        self._used = used
        self._complexity = complexity
        self._check = check

    # ------------------------------------------------------------ queries
    def is_int_range(self, key: str) -> bool:
        u = self.params[key]
        return isinstance(u, tuple) and isinstance(u[0], int) and isinstance(u[1], int) \
            and not isinstance(u[0], bool)

    def used(self, cfg: dict) -> list[str]:
        keys = list(self._used(cfg)) if self._used else list(self.params)
        return [k for k in keys if k in self.params and k not in self.fixed]

    def complexity(self, cfg: dict) -> float | None:
        return None if self._complexity is None else float(self._complexity(cfg))

    def allowed(self) -> dict:
        """Proposal space shown to proposers: fixed keys removed, forbidden values removed."""
        out = {}
        for k, u in self.params.items():
            if k in self.fixed:
                continue
            if isinstance(u, list):
                u = [x for x in u if x not in self.forbidden.get(k, [])]
            out[k] = list(u) if isinstance(u, tuple) else u
        return out

    def grid_values(self, key: str, n_grid: int = 5) -> list:
        """Candidate values for a key: the choices, or n_grid evenly spaced points of a range."""
        u = self.params[key]
        if isinstance(u, list):
            return [x for x in u if x not in self.forbidden.get(key, [])]
        lo, hi = u
        if self.is_int_range(key):
            vals = sorted({int(round(x)) for x in np.linspace(lo, hi, n_grid)})
        else:
            vals = [round(float(x), 6) for x in np.linspace(lo, hi, n_grid)]
        return [v for v in vals if v not in self.forbidden.get(key, [])]

    # ------------------------------------------------------------ validation
    def validate(self, cfg: dict) -> tuple[dict, list[str]]:
        """Return (full config with defaults, errors)."""
        full = {**self.defaults, **{k: _plain(v) for k, v in cfg.items()}}
        errors = []
        for k, v in list(full.items()):
            if k in self.fixed and k not in self.params:
                continue
            if k not in self.params:
                errors.append(f"unknown key {k!r}")
                continue
            u = self.params[k]
            if isinstance(u, list):
                if v not in u:
                    errors.append(f"{k}={v!r} not in {u}")
                continue
            if v is None:
                errors.append(f"{k} must be a number in [{u[0]}, {u[1]}]")
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
                errors.append(f"{k}={v!r} must be a number in [{u[0]}, {u[1]}]")
                continue
            if self.is_int_range(k):
                if isinstance(v, float):
                    if not v.is_integer():
                        errors.append(f"{k}={v!r} must be an integer")
                        continue
                    full[k] = v = int(v)
            if not (u[0] <= v <= u[1]):
                errors.append(f"{k}={v!r} outside [{u[0]}, {u[1]}]")
        if not errors and self._check is not None:
            errors += list(self._check(full) or [])
        return full, errors

    # ------------------------------------------------------------ hashing / diffs
    def summary(self, cfg: dict) -> dict:
        return {k: v for k, v in cfg.items()}

    def fingerprint(self) -> str:
        body = json.dumps({"params": {k: list(v) if isinstance(v, tuple) else v for k, v in self.params.items()},
                           "defaults": self.defaults, "fixed": self.fixed, "forbidden": self.forbidden},
                          sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()[:16]


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def config_diff(cfg: dict, ref: dict) -> dict:
    """Keys whose value in cfg differs from ref (values from cfg)."""
    return {k: v for k, v in cfg.items() if ref.get(k, object()) != v}

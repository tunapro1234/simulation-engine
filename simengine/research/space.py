"""Configuration space for the research loop (ported from kosucu.dogrula / *_UZAY / dongu.py kullanim, bilesen).

A space maps each tunable key to either a list of choices or a (lo, hi) range. A range whose two ends are ints is an
integer range. `fixed` keys may appear in the config but never change. `used(cfg)` tells which keys matter under a
config (a change to an unused key is a wasted turn), `complexity(cfg)` feeds the optional simplicity rule,
`check(cfg)` adds task-specific validation errors.

Harness robustness (`harness_variants`, `one_change_variants`, `harness_sweep`): the same model can give very
different answers under a different harness, wording, token cap or temperature (replicant's Ertan 2009 vignette: a
random-persona punish-low vote of 14% through EDSL and 80% through oTree with the same model; edsl-test: the bare model
punished free-riders at max_tokens 8192 but not at the default cap). A `HarnessSpace` makes those knobs explicit
config keys, so each one-key change is a legal research-loop turn, and `harness_sweep` measures every one-change
variant against the base on the same units. Variant labels follow replicant's condition-label pattern
(`cli.determine_condition`): "default" for the base, else the changed key and value.
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


# ===================================================================== harness robustness sweep

HARNESS_KEYS = ("wording_id", "max_tokens", "temperature")


def variant_label(cfg: dict, ref: dict) -> str:
    """Condition label of `cfg` relative to `ref` (replicant cli pattern): "default" or "key=value,key=value"."""
    d = config_diff(cfg, ref)
    return "default" if not d else ",".join(f"{k}={v}" for k, v in d.items())


def one_change_variants(space: Space, base: dict | None = None, *, n_grid: int = 5) -> list[dict]:
    """The base config plus every config that differs from it in exactly one key (one change per turn).

    Each item is {"label", "key", "value", "change", "config"}; the first is the base itself (label "default",
    key None, change {}). Choices come from `space.grid_values` (n_grid points for a range); fixed, forbidden and
    unused keys are left out, and a candidate that fails `space.validate` (e.g. the space's `check`) is skipped."""
    full, err = space.validate(base or {})
    if err:
        raise ValueError(f"base config is invalid: {'; '.join(err)}")
    out = [{"label": "default", "key": None, "value": None, "change": {}, "config": full}]
    for k in space.used(full):
        for v in space.grid_values(k, n_grid):
            if v == full.get(k):
                continue
            cfg, e = space.validate({**full, k: v})
            if e:
                continue
            out.append({"label": f"{k}={v}", "key": k, "value": v, "change": {k: v}, "config": cfg})
    return out


class HarnessSpace(Space):
    """A `Space` over harness knobs (wording_id, max_tokens, temperature, plus any extra axes).

    `wordings` maps wording_id to the question text when it was given as a dict, so a runner can do
    `space.wording(cfg)`. The fingerprint covers the texts too, so a resumed loop notices an edited wording."""

    def __init__(self, params, *, defaults=None, wordings: dict | None = None, **kw):
        super().__init__(params, defaults=defaults, **kw)
        self.wordings = dict(wordings) if wordings else None

    def wording(self, cfg_or_id) -> str:
        wid = cfg_or_id.get("wording_id") if isinstance(cfg_or_id, dict) else cfg_or_id
        if self.wordings is None:
            raise KeyError("this harness space has wording ids only; pass wordings as {id: text} to look texts up")
        return self.wordings[wid]

    def variants(self, base: dict | None = None) -> list[dict]:
        return one_change_variants(self, base)

    def label(self, cfg: dict) -> str:
        full, _ = self.validate(self.defaults)
        return variant_label(cfg, full)

    def fingerprint(self) -> str:
        base = super().fingerprint()
        if not self.wordings:
            return base
        body = json.dumps({"space": base, "wordings": self.wordings}, sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()[:16]


def _axis(name, values) -> list:
    if isinstance(values, (str, bytes, int, float)) or values is None:
        values = [values]
    vals = [_plain(v) for v in values]
    if not vals:
        raise ValueError(f"harness axis {name!r} needs at least one value")
    seen = []
    for v in vals:
        if v in seen:
            raise ValueError(f"harness axis {name!r} has a duplicate value {v!r}")
        seen.append(v)
    return vals


def harness_variants(wordings, max_tokens_list=(None,), temperatures=(None,), *, defaults: dict | None = None,
                     extra: dict | None = None, fixed=(), check=None) -> HarnessSpace:
    """Build the harness robustness space: one axis per knob, the first value of each axis is the default.

    wordings: a list of wording ids, or {wording_id: text}. max_tokens_list: positive ints or None (provider
    default). temperatures: numbers >= 0 or None. A scalar counts as a one-value axis. `extra`: more axes, e.g.
    {"harness": ["edsl", "otree"]} or {"persona_method": [...]}. Every one-key change of the result is a legal research
    loop turn (`Loop(runner, space, ...)`); `space.variants()` lists them, `harness_sweep` runs them all.
    The runner should store wording_id, max_tokens and temperature on every answer row."""
    texts = None
    if isinstance(wordings, dict):
        texts = {str(k): str(v) for k, v in wordings.items()}
        if len(texts) != len(wordings):
            raise ValueError("wording ids collide after str()")
        wordings = list(texts)
    w = _axis("wording_id", wordings)
    for v in w:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"wording ids must be non-empty strings, got {v!r}")
    mt = _axis("max_tokens", max_tokens_list)
    for v in mt:
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v <= 0):
            raise ValueError(f"max_tokens must be a positive int or None, got {v!r}")
    te = _axis("temperature", temperatures)
    for v in te:
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0):
            raise ValueError(f"temperature must be a number >= 0 or None, got {v!r}")
    params = {"wording_id": w, "max_tokens": mt, "temperature": te}
    for k, vals in (extra or {}).items():
        if k in params:
            raise ValueError(f"extra axis {k!r} collides with a harness key")
        params[k] = _axis(k, vals)
    dflt = {k: v[0] for k, v in params.items()}
    dflt.update(defaults or {})
    space = HarnessSpace(params, defaults=dflt, wordings=texts, fixed=fixed, check=check)
    _, err = space.validate(dflt)
    if err:
        raise ValueError(f"harness defaults are invalid: {'; '.join(err)}")
    return space


def _per_unit_frame(res, unit_col: str, primary: str):
    import pandas as pd
    pu = res.get("per_unit") if isinstance(res, dict) else None
    if pu is None:
        raise ValueError("runner must return {'per_unit': ...} (one score per unit)")
    if isinstance(pu, dict):
        pu = pd.DataFrame({unit_col: list(pu), primary: list(pu.values())})
    elif isinstance(pu, pd.Series):
        pu = pd.DataFrame({unit_col: pu.index, primary: pu.values})
    elif not isinstance(pu, pd.DataFrame):
        pu = pd.DataFrame(pu)
    if unit_col not in pu.columns:
        pu = pu.reset_index().rename(columns={"index": unit_col})
    if primary not in pu.columns:
        raise ValueError(f"per_unit has no primary column {primary!r}")
    if pu[unit_col].astype(str).duplicated().any():
        raise ValueError("per_unit has duplicate unit ids")
    return pd.Series(pd.to_numeric(pu[primary], errors="coerce").to_numpy(float), index=pu[unit_col].astype(str))


def harness_sweep(runner, space: Space, primary: str, *, base: dict | None = None, split: str = "dev",
                  split_key: str = "split", unit_col: str = "unit", n_grid: int = 5, seed: int = 42,
                  n_boot: int = 1000, alpha: float = 0.05):
    """Run the base and every one-change variant once on `split` and measure each change against the base.

    The runner follows the research-loop contract (`cfg[split_key]` is set; it returns {"per_unit": ...}). Scores are
    computed here from per-unit values; a non-numeric or NaN unit score counts as missing (`n_missing`), is never
    filled in, and is left out of the paired comparison. Returns a DataFrame with one row per variant: label, key,
    value, n_units, n_missing, mean, diff (variant - base, paired on shared units), lo/hi ((1-alpha) bootstrap CI),
    lo_bonf/hi_bonf (Bonferroni over the number of variants), direction ("lower"/"higher"/"same" by the Bonferroni CI),
    error (runner exception text). `df.attrs` holds the base config, k and the spread of variant means."""
    import pandas as pd
    from .stats import bootstrap_mean, direction, paired_bootstrap_diff

    vs = one_change_variants(space, base, n_grid=n_grid)
    k = max(len(vs) - 1, 1)
    scores, rows = {}, []
    for v in vs:
        row = {"label": v["label"], "key": v["key"], "value": v["value"], "n_units": 0, "n_missing": 0,
               "mean": None, "diff": None, "lo": None, "hi": None, "lo_bonf": None, "hi_bonf": None,
               "direction": None, "error": None}
        try:
            s = _per_unit_frame(runner({**v["config"], split_key: split}), unit_col, primary)
        except Exception as e:  # noqa: BLE001 - a crashing variant is a result, not a reason to stop the sweep
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            rows.append(row)
            continue
        ok = s[np.isfinite(s.to_numpy())]
        row.update(n_units=int(len(ok)), n_missing=int(len(s) - len(ok)), mean=bootstrap_mean(ok.to_numpy())["value"])
        scores[v["label"]] = ok
        b = scores.get("default")
        if v["key"] is not None and b is not None:
            common = ok.index.intersection(b.index)
            if len(common):
                d = paired_bootstrap_diff(ok[common].to_numpy(), b[common].to_numpy(), seed=seed, B=n_boot,
                                          alpha=alpha)
                db = paired_bootstrap_diff(ok[common].to_numpy(), b[common].to_numpy(), seed=seed, B=n_boot,
                                           alpha=alpha / k)
                row.update(diff=d["diff"], lo=d["lo"], hi=d["hi"], lo_bonf=db["lo"], hi_bonf=db["hi"],
                           direction=direction(db))
        rows.append(row)
    df = pd.DataFrame(rows)
    means = [m for m in df["mean"] if m is not None and np.isfinite(m)]
    df.attrs.update(base=vs[0]["config"], k=k, split=split, primary=primary,
                    mean_range=(min(means), max(means)) if means else None)
    return df

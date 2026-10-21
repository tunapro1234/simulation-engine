"""Small shared helpers for simengine.methods (internal)."""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-6


def logit(p, eps: float = 1e-4):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def rename_aliases(df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    """Rename research-code (Turkish) columns to their English names when the English one is absent."""
    ren = {k: v for k, v in aliases.items() if k in df.columns and v not in df.columns}
    return df.rename(columns=ren) if ren else df


def parse_json(text: str) -> Any:
    """First JSON object/list in a reply (```json fences allowed). Raises ValueError if none."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    s = m.group(1) if m else text
    starts = [i for i in (s.find("["), s.find("{")) if i >= 0]
    if not starts:
        raise ValueError(f"no JSON in reply: {text[:200]!r}")
    obj, _ = json.JSONDecoder().raw_decode(s[min(starts):])
    return obj


def require_lightgbm():
    """Import lightgbm lazily with a clear error (it is an optional extra: pip install simengine[gbdt])."""
    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError("this method needs the optional 'lightgbm' package "
                          "(pip install 'simengine[gbdt]')") from e
    return lgb


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def n_threads(n_rows: int) -> int:
    """LightGBM thread count. Measured in the research kit: on a loaded machine small data runs 10-40x faster
    single-threaded (OpenMP waiting), so 1 thread below 200k rows, else 4."""
    return 1 if n_rows < 200_000 else 4

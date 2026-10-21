"""Generic, deterministic samplers behind the persona method zoo (stdlib only).

Ported from edsl-test `personas/` (read-only source tree):
  * `cholesky`, `TraitDialSampler`        big5/__init__.py `_cholesky`/`_scores`, gps/__init__.py `_draw_traits`
  * `quota_mixture`                       homo_silicus/__init__.py `_quota` + `sample`
  * `weighted_rows_no_replacement`,
    `RowFragmentRenderer`                 demographic/__init__.py `sample`, `_fragments`, `_render`
  * `stratified_pool_sample`              anthology/__init__.py `sample`

Every function takes a `seed` (int, None or a `random.Random`) and uses `random.Random`, so the draws for a
given seed are the same as the source code's draws (checked in tests/test_persona_methods.py).
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Mapping, Sequence


def as_rng(seed: Any = None) -> random.Random:
    """int / None -> random.Random(seed); a Random instance is returned unchanged."""
    return seed if isinstance(seed, random.Random) else random.Random(seed)


# ---------------------------------------------------------------- correlated trait dials
def cholesky(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Lower-triangular L with L L^T = matrix. Raises ValueError if the matrix is not positive definite."""
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("correlation matrix must be square")
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                d = matrix[i][i] - s
                if d <= 0:
                    raise ValueError("correlation matrix is not positive definite")
                L[i][j] = math.sqrt(d)
            else:
                L[i][j] = (matrix[i][j] - s) / L[j][j]
    return L


def corr_matrix(names: Sequence[str], corr: Any) -> list[list[float]] | None:
    """Normalize `corr` to a matrix in the order of `names`.

    Accepts None (independent traits), a nested dict {a: {b: r}} (missing pairs = 0, symmetric lookup) or a
    square matrix already in the order of `names`."""
    if corr is None:
        return None
    k = len(names)
    if isinstance(corr, Mapping):
        def get(a, b):
            if a == b:
                return 1.0
            for x, y in ((a, b), (b, a)):
                row = corr.get(x)
                if isinstance(row, Mapping) and y in row:
                    return float(row[y])
            return 0.0
        unknown = set(corr) - set(names)
        if unknown:
            raise ValueError(f"correlation keys not in traits: {sorted(unknown)}")
        return [[get(a, b) for b in names] for a in names]
    m = [[float(x) for x in row] for row in corr]
    if len(m) != k or any(len(row) != k for row in m):
        raise ValueError(f"correlation matrix must be {k}x{k} in the order {list(names)}")
    return m


class TraitDialSampler:
    """Jointly normal trait draws -> optional clip -> optional transform (dial) -> persona text.

        norms      {trait: (mean, sd)}; the order of the keys is the draw order
        corr       None (independent), nested dict or matrix in the order of `norms`
        template   str with {trait} fields, or callable(dials: dict) -> str
        clip       (lo, hi) applied to the raw value, e.g. (1, 5) for BFI-2 scores
        transform  callable(raw) -> dial, e.g. lambda x: round((x - 1) / 4 * 10)

    One draw consumes len(norms) `rng.gauss(0, 1)` values, then applies L z. This is the edsl-test
    big5/gps algorithm, so the same seed gives the same dials as the source."""

    def __init__(self, norms: Mapping[str, tuple[float, float]], corr: Any = None,
                 template: str | Callable[[dict], str] | None = None,
                 clip: tuple[float, float] | None = None, transform: Callable[[float], Any] | None = None):
        self.names = list(norms)
        self.norms = {k: (float(v[0]), float(v[1])) for k, v in norms.items()}
        self.corr = corr_matrix(self.names, corr)
        self._L = cholesky(self.corr) if self.corr is not None else None
        self.template = template
        self.clip = clip
        self.transform = transform

    def z(self, rng: random.Random) -> list[float]:
        e = [rng.gauss(0, 1) for _ in self.names]
        if self._L is None:
            return e
        return [sum(self._L[i][k] * e[k] for k in range(i + 1)) for i in range(len(e))]

    def draw(self, rng: random.Random) -> dict[str, float]:
        """One person's raw (clipped) values."""
        out = {}
        for name, z in zip(self.names, self.z(rng)):
            mean, sd = self.norms[name]
            x = mean + sd * z
            if self.clip is not None:
                x = min(self.clip[1], max(self.clip[0], x))
            out[name] = x
        return out

    def dials(self, rng: random.Random) -> dict[str, Any]:
        raw = self.draw(rng)
        return {k: self.transform(v) for k, v in raw.items()} if self.transform else raw

    def render(self, dials: Mapping[str, Any]) -> str:
        if self.template is None:
            return "; ".join(f"{k}: {v}" for k, v in dials.items())
        if callable(self.template):
            return self.template(dict(dials))
        return self.template.format(**dials)

    def sample_dials(self, n: int, seed: Any = None) -> list[dict]:
        rng = as_rng(seed)
        return [self.dials(rng) for _ in range(n)]

    def sample(self, n: int, seed: Any = None) -> list[str]:
        return [self.render(d) for d in self.sample_dials(n, seed)]


# ---------------------------------------------------------------- discrete mixtures
def quota_counts(weights: Mapping[str, float], n: int) -> dict[str, int]:
    """Largest-remainder rounding of mixture weights into integer counts summing to n (ties: input order)."""
    total = sum(weights.values())
    if total <= 0 or any(w < 0 for w in weights.values()):
        raise ValueError("mixture weights must be non-negative with a positive sum")
    raw = {t: weights[t] / total * n for t in weights}
    counts = {t: int(raw[t]) for t in weights}
    leftover = n - sum(counts.values())
    for t in sorted(raw, key=lambda t: raw[t] - counts[t], reverse=True)[:leftover]:
        counts[t] += 1
    return counts


def quota_mixture(types: Sequence[str] | Mapping[str, Any], weights: Mapping[str, float] | Sequence[float],
                  n: int, seed: Any = None) -> list[str]:
    """n type keys with deterministic largest-remainder quotas, shuffled by `seed`.

    The composition depends only on (weights, n), never on the seed; the seed only orders it. `weights` is a
    dict keyed by type or a sequence aligned with `types`."""
    keys = list(types)
    if isinstance(weights, Mapping):
        w = {t: float(weights.get(t, 0.0)) for t in keys}
    else:
        if len(weights) != len(keys):
            raise ValueError("weights must align with types")
        w = dict(zip(keys, map(float, weights)))
    rng = as_rng(seed)
    pool = [t for t, c in quota_counts(w, n).items() for _ in range(c)]
    rng.shuffle(pool)
    return pool


# ---------------------------------------------------------------- survey rows
def weighted_indices_no_replacement(weights: Sequence[float], n: int, seed: Any = None) -> list[int]:
    """Indices drawn with probability proportional to `weights`, no repeats while unused positive-weight rows
    remain (then repeats are allowed). Same draws as edsl-test demographic `sample`."""
    w = [float(x) for x in weights]
    if any(not math.isfinite(x) or x < 0 for x in w):
        raise ValueError("row weights must be finite and non-negative")
    eligible = sum(1 for x in w if x > 0)
    if n > 0 and eligible == 0:
        raise ValueError("no row has a positive weight")
    rng = as_rng(seed)
    idx = range(len(w))
    chosen: list[int] = []
    seen: set[int] = set()
    while len(chosen) < n:
        i = rng.choices(idx, weights=w, k=1)[0]
        if i in seen and len(seen) < eligible:
            continue
        chosen.append(i)
        seen.add(i)
    return chosen


def weighted_rows_no_replacement(rows: Sequence[Mapping[str, Any]], weight_col: str, n: int,
                                 seed: Any = None) -> list[Mapping[str, Any]]:
    """Survey-weighted draw of n rows (dicts) without repeating a respondent while unused rows remain."""
    weights = [float(r[weight_col]) for r in rows]
    return [rows[i] for i in weighted_indices_no_replacement(weights, n, seed)]


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, float) and math.isnan(v))


class RowFragmentRenderer:
    """Survey row -> short first-person fragments (Argyle et al. 2023 style) -> persona text.

    `templates` is a list; each entry is one of
      (column, {value: fragment})   value looked up as given; unmapped or missing values drop the fragment
      (column, "format {}")         formatted with the value when it is not missing
      callable(row) -> str | None   free logic (e.g. collapsing race and ethnicity)
    `frame` wraps the joined fragments, with `{backstory}` as the placeholder."""

    def __init__(self, templates: Iterable[Any], frame: str = "{backstory}", joiner: str = " "):
        self.templates = list(templates)
        self.frame = frame
        self.joiner = joiner

    def fragments(self, row: Mapping[str, Any]) -> list[str]:
        out = []
        for t in self.templates:
            if callable(t):
                f = t(row)
            else:
                col, spec = t
                v = row.get(col)
                if _blank(v):
                    f = None
                elif isinstance(spec, Mapping):
                    f = spec.get(v)
                else:
                    f = str(spec).format(v)
            if f:
                out.append(f)
        return out

    def render(self, row: Mapping[str, Any]) -> str:
        return self.frame.format(backstory=self.joiner.join(self.fragments(row)))

    __call__ = render


# ---------------------------------------------------------------- stratified pools
def stratified_pool_sample(pool: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]],
                           strata_weights: Mapping[str, float], n: int, seed: Any = None,
                           key: str = "bracket") -> list[Any]:
    """Draw a stratum i.i.d. from `strata_weights`, then an item from that stratum without replacement; an
    exhausted stratum is refilled and reshuffled. Same draws as edsl-test anthology `sample`.

    `pool` is {stratum: [items]} or a list of dicts grouped by `key`. Strata are visited in the order of
    `strata_weights`; items whose stratum has no weight are never drawn."""
    strata = list(strata_weights)
    if isinstance(pool, Mapping):
        pools = {s: list(pool.get(s, [])) for s in strata}
    else:
        pools = {s: [] for s in strata}
        for item in pool:
            s = item.get(key)
            if s in pools:
                pools[s].append(item)
    weights = [float(strata_weights[s]) for s in strata]
    empty = [s for s, w in zip(strata, weights) if w > 0 and not pools[s]]
    if empty:
        raise ValueError(f"strata with positive weight but no items: {empty}")
    rng = as_rng(seed)
    shuffled = {}
    for s in strata:
        shuffled[s] = pools[s][:]
        rng.shuffle(shuffled[s])
    out = []
    for _ in range(n):
        s = rng.choices(strata, weights=weights)[0]
        if not shuffled[s]:
            shuffled[s] = pools[s][:]
            rng.shuffle(shuffled[s])
        out.append(shuffled[s].pop())
    return out


__all__ = ["as_rng", "cholesky", "corr_matrix", "TraitDialSampler", "quota_counts", "quota_mixture",
           "weighted_indices_no_replacement", "weighted_rows_no_replacement", "RowFragmentRenderer",
           "stratified_pool_sample"]

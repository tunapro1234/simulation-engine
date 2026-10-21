"""Theory-persona mixture calibration (the "selection" step of General Social Agents, on Homo Silicus types).

A population is modelled as a mixture over a few theory types ("you only care about your own pay-off", ...).
Each type's behaviour on a set of calibration targets (for example P(choose Left) on the Charness & Rabin 2002
allocation menus) is measured once; the mixture weights are then fitted so that the weighted average of the
types' behaviour matches the human targets. Validation is done on targets that were never fitted (a different
game format), because a mixture can match one data-generating process and miss the next one.

Port of the pure-Python simplex grid fit and the identifiability warning in edsl-test
`personas/homo_silicus/calibrate.py` (``fit_weights``, ``mixture``, the duplicate-row WARNING). Changes:

* the grid search is vectorised (numpy, same lexicographic order, so the same ties resolve the same way);
* the identifiability check also reports rank deficiency (a type that is a convex combination of others, or more
  types than targets + 1), and the weight range over exactly tied grid points;
* the EDSL probes are rewritten onto ``simengine.types.LLM.map``: the persona text goes in as ``system``, the
  game text as ``user``; an unparsable answer stays missing, is counted and is reported (the source raised only
  when a cell had no answers at all);
* an optional continuous fit (``method="slsqp"``, scipy) for type libraries too large for the grid.

Literature constants (small published numbers, inlined with their sources):

* ``CR_GAMES``: Charness & Rabin (2002), "Understanding social preferences with simple tests", QJE 117(3),
  Table I, two-person dictator games, human share choosing Left. Berk29 is .31 (Table I; Horton's Fig. 3 says
  31%; the vector printed in Horton's text says .68, a typo there). Source: edsl-test
  `personas/homo_silicus/calibrate.py`.
* ``ENGEL_DICTATOR``: Engel (2011), "Dictator games: a meta study", Experimental Economics 14(4); MPI Collective
  Goods preprint 2010/07 pp. 7-8: mean give 28.35% of the pie, 36.11% give nothing, 16.74% equal split.
* ``THEORY_TYPES`` / ``HUMANIZING_PREFIX``: Horton (2023), "Large language models as simulated economic agents:
  what can we learn from Homo Silicus?", arXiv:2301.07543, Fig. 3 (self_interested, efficient, inequity_averse;
  "both players" -> "all players"); competitive and reciprocal from Charness & Rabin (2002); selfish_but_fair and
  the prefix from Manning, Zhu & Horton, "General Social Agents" (Sec. 2.1, Table 2 notes). Source: edsl-test
  `personas/homo_silicus/__init__.py`.
* ``HORTON_GPT4O_WEIGHTS``: Horton (2023) Sec. 2.2.1, GPT-4o mixture w_S=.53, w_E=.37, w_I=.10.

Measured in edsl-test (`personas/homo_silicus/weights.json`, `NOTES.md`; gpt-oss-120b, 10 reps per cell): the
6-type fit on the CR menus gave self_interested .36, efficient .50, selfish_but_fair .14, the other three 0;
train SSE .089; held-out dictator RMSE .315 (poor: mixture mean give 6.7% vs 28.35%). The reciprocal type gets
weight 0 because unilateral menus contain no prior unfair act, so reciprocity is not identified by these probes.
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "MixtureFit", "UnidentifiedWarning", "fit_mixture_weights", "mixture", "identifiability",
    "ProbeGame", "choice_game", "numeric_game", "dictator_game", "charness_rabin_games", "probe_types",
    "CR_GAMES", "CR_HUMAN_LEFT", "ENGEL_DICTATOR", "THEORY_TYPES", "HUMANIZING_PREFIX", "HORTON_GPT4O_WEIGHTS",
]

# ---------------------------------------------------------------- literature constants

# game: ((A_left, B_left), (A_right, B_right), human share choosing Left); B is the decider.
# Charness & Rabin (2002, QJE 117(3)) Table I; via edsl-test personas/homo_silicus/calibrate.py.
CR_GAMES: dict[str, tuple[tuple[int, int], tuple[int, int], float]] = {
    "Berk29": ((400, 400), (750, 400), 0.31),
    "Barc2": ((400, 400), (750, 375), 0.52),
    "Berk23": ((800, 200), (0, 0), 1.00),
    "Barc8": ((300, 600), (700, 500), 0.67),
    "Berk15": ((200, 700), (600, 600), 0.27),
    "Berk26": ((0, 800), (400, 400), 0.78),
}
CR_HUMAN_LEFT = pd.Series({g: v[2] for g, v in CR_GAMES.items()}, name="human_left_share")

# Engel (2011) dictator meta-analysis, MPI Collective Goods preprint 2010/07 pp. 7-8.
ENGEL_DICTATOR = pd.Series({"dictator:mean_give_frac": 0.2835, "dictator:p_give_0": 0.3611,
                            "dictator:p_give_50": 0.1674}, name="engel_2011")

# Manning, Zhu & Horton, General Social Agents, Table 2 notes (verbatim prefix).
HUMANIZING_PREFIX = "You are a human being with all the cognitive biases and heuristics that come with it."
# Horton arXiv:2301.07543 Fig. 3; Charness & Rabin 2002; GSA Sec. 2.1 (see module docstring).
THEORY_TYPES: dict[str, str] = {
    "self_interested": "You only care about your own pay-off.",
    "efficient": "You only care about the total pay-off of all players.",
    "inequity_averse": "You only care about fairness between players.",
    "competitive": "You only care about doing better than the other players.",
    "reciprocal": ("You only care about reciprocity: you are generous to those who act fairly, and you punish "
                   "unfair behavior even at a cost to yourself."),
    "selfish_but_fair": "You are self-interested but fair.",
}
# Horton arXiv:2301.07543 Sec. 2.2.1 (GPT-4o); types he did not have are 0.
HORTON_GPT4O_WEIGHTS: dict[str, float] = {"self_interested": 0.53, "efficient": 0.37, "inequity_averse": 0.10}


class UnidentifiedWarning(UserWarning):
    """Some mixture weights are not separately identified by the calibration targets."""


# ---------------------------------------------------------------- fit

@dataclass
class MixtureFit:
    """Result of ``fit_mixture_weights``.

    weights            Series over types, sums to 1.
    train_sse          sum of squared errors of the mixture on the training targets.
    val_rmse           RMSE on the held-out validation targets (None without ``val``).
    unidentified_pairs type pairs whose training behaviour is identical (within ``ident_tol``).
    identified         False when the pairs above exist or [behaviour; 1] has rank < number of types.
    weight_range       per type min / max weight over grid points tied with the best SSE (grid method only);
                       a range wider than 0 means the split between those types is arbitrary.
    """
    weights: pd.Series
    train_sse: float
    val_rmse: float | None
    unidentified_pairs: list[tuple[str, str]]
    identified: bool = True
    rank: int = 0
    train_pred: pd.Series | None = None
    val_pred: pd.Series | None = None
    weight_range: pd.DataFrame | None = None
    n_grid: int = 0
    n_ties: int = 1
    grid_step: float | None = None
    method: str = "grid"
    notes: list[str] = field(default_factory=list)

    def predict(self, behavior: pd.DataFrame) -> pd.Series:
        """Mixture of a (type x target) behaviour table under the fitted weights."""
        return mixture(self.weights, behavior)

    def to_dict(self) -> dict:
        """JSON-serialisable summary."""
        out = {"weights": {k: float(v) for k, v in self.weights.items()}, "train_sse": float(self.train_sse),
               "val_rmse": None if self.val_rmse is None else float(self.val_rmse),
               "unidentified_pairs": [list(p) for p in self.unidentified_pairs], "identified": self.identified,
               "rank": self.rank, "n_grid": self.n_grid, "n_ties": self.n_ties, "grid_step": self.grid_step,
               "method": self.method, "notes": list(self.notes)}
        if self.train_pred is not None:
            out["train_pred"] = {k: float(v) for k, v in self.train_pred.items()}
        if self.val_pred is not None:
            out["val_pred"] = {k: float(v) for k, v in self.val_pred.items()}
        if self.weight_range is not None:
            out["weight_range"] = {t: [float(r["min"]), float(r["max"])] for t, r in self.weight_range.iterrows()}
        return out


def mixture(weights: Mapping[str, float] | pd.Series, behavior: pd.DataFrame) -> pd.Series:
    """Weighted average of the types' behaviour rows: sum_t w_t * behavior[t, :]. Types missing from
    ``weights`` count as 0; a weight for a type absent from ``behavior`` is an error."""
    w = pd.Series(weights, dtype=float)
    missing = [t for t in w.index if t not in behavior.index and w[t] != 0]
    if missing:
        raise ValueError(f"behavior has no row for weighted types {missing}")
    w = w.reindex(behavior.index).fillna(0.0)
    B = behavior.astype(float)
    if B.isna().to_numpy().any():
        _raise_missing(B, "behavior")
    return pd.Series(w.to_numpy() @ B.to_numpy(), index=behavior.columns, name="mixture")


def _raise_missing(df: pd.DataFrame, what: str):
    names = [f"{df.index[r]}/{df.columns[c]}" for r, c in zip(*np.where(df.isna().to_numpy()))]
    raise ValueError(f"{what} has missing cells (no imputation; re-run or drop the target): {names[:10]}"
                     + (f" and {len(names) - 10} more" if len(names) > 10 else ""))


def identifiability(type_behavior: pd.DataFrame, targets: Sequence[str] | None = None, *,
                    ident_tol: float = 0.01) -> dict:
    """Which mixture weights the targets can identify.

    Returns {"pairs": [(a, b), ...] types whose behaviour differs by at most ``ident_tol`` on every target,
    "rank": rank of the (targets + 1) x types matrix [behaviour; 1], "n_types", "identified": bool}.
    Weights on the simplex are identified only when that matrix has full column rank; identical rows are the
    common way to lose it (the edsl-test v1 bug: self_interested and efficient both kept everything in a dictator
    probe, and the grid gave their joint weight to whichever came first)."""
    B = type_behavior if targets is None else type_behavior.loc[:, list(targets)]
    B = B.astype(float)
    names = [str(t) for t in B.index]
    X = B.to_numpy()
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.max(np.abs(X[i] - X[j])) <= ident_tol:
                pairs.append((names[i], names[j]))
    M = np.vstack([X.T, np.ones(len(names))])
    rank = int(np.linalg.matrix_rank(M))
    return {"pairs": pairs, "rank": rank, "n_types": len(names), "identified": not pairs and rank == len(names)}


def _compositions(m: int, total: int) -> np.ndarray:
    """All m-part non-negative integer compositions of ``total``, in lexicographic order (as the source's
    recursive generator)."""
    if m == 1:
        return np.array([[total]], dtype=np.int64)
    A = np.arange(total + 1, dtype=np.int64)[:, None]
    for _ in range(m - 2):
        s = A.sum(1)
        cnt = total - s + 1
        start = np.cumsum(cnt) - cnt
        new = np.arange(int(cnt.sum()), dtype=np.int64) - np.repeat(start, cnt)
        A = np.hstack([np.repeat(A, cnt, axis=0), new[:, None]])
    return np.hstack([A, (total - A.sum(1))[:, None]])


def _grid_fit(B: np.ndarray, y: np.ndarray, ticks: int, tie_tol: float):
    n = B.shape[0]
    if n == 1:
        sse = float(np.sum((B[0] - y) ** 2))
        return np.array([1.0]), sse, 1, np.array([[1.0, 1.0]]), 1
    best_sse, best_w, n_grid = math.inf, None, 0
    rests = {}
    for first in range(ticks + 1):                   # chunk by the first weight; keeps memory at C(T+n-2, n-2)
        rest = rests.get(ticks - first)
        if rest is None:
            rest = rests[ticks - first] = _compositions(n - 1, ticks - first)
        C = np.hstack([np.full((len(rest), 1), first, dtype=np.int64), rest])
        n_grid += len(C)
        sse = (((C @ B) / ticks - y) ** 2).sum(1)
        k = int(np.argmin(sse))
        if sse[k] < best_sse:                         # strict: the first point in lexicographic order wins ties
            best_sse, best_w = float(sse[k]), C[k] / ticks
    # second pass: grid points tied with the optimum -> weight range
    lim = best_sse + tie_tol * max(1.0, best_sse)
    lo, hi, n_ties = np.full(n, np.inf), np.full(n, -np.inf), 0
    for first in range(ticks + 1):
        rest = rests[ticks - first]
        C = np.hstack([np.full((len(rest), 1), first, dtype=np.int64), rest])
        sse = (((C @ B) / ticks - y) ** 2).sum(1)
        tied = C[sse <= lim]
        if len(tied):
            n_ties += len(tied)
            lo = np.minimum(lo, tied.min(0) / ticks)
            hi = np.maximum(hi, tied.max(0) / ticks)
    return best_w, best_sse, n_grid, np.column_stack([lo, hi]), n_ties


def _slsqp_fit(B: np.ndarray, y: np.ndarray):
    try:
        from scipy.optimize import minimize
    except ImportError as e:  # pragma: no cover - scipy is a core dependency
        raise ImportError("method='slsqp' needs scipy") from e
    n = B.shape[0]

    def f(w):
        r = w @ B - y
        return float(r @ r), 2.0 * (B @ r)

    res = minimize(f, np.full(n, 1.0 / n), jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
                   options={"ftol": 1e-12, "maxiter": 1000})
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    return w, float(np.sum((w @ B - y) ** 2))


def _as_target(s, what: str) -> pd.Series:
    s = pd.Series(s, dtype=float) if not isinstance(s, pd.Series) else s.astype(float)
    if s.isna().any():
        raise ValueError(f"{what} has missing values at {list(s.index[s.isna()])} (no imputation)")
    if s.index.has_duplicates:
        raise ValueError(f"{what} has duplicate targets")
    return s


def fit_mixture_weights(type_behavior: pd.DataFrame, human_target: pd.Series | Mapping[str, float], *,
                        grid_step: float = 0.02, val: Any = None, method: str = "grid", ident_tol: float = 0.01,
                        tie_tol: float = 1e-9, max_grid: int = 20_000_000, warn: bool = True) -> MixtureFit:
    """Fit simplex weights w (w >= 0, sum 1) minimising SSE(sum_t w_t * behavior[t, j] - human[j]) over the
    training targets j = ``human_target.index``.

    type_behavior : DataFrame, one row per type, one column per target (e.g. P(Left) per menu, dictator moments).
                    Missing cells in the used columns raise: a type that never answered has no behaviour.
    human_target  : Series target -> human value (training targets). Targets on very different scales should be
                    rescaled first; the SSE weighs them as given.
    grid_step     : weight resolution of the exhaustive simplex grid (default 2%, as the source). ``1/grid_step``
                    must be an integer.
    val           : held-out validation, never fitted. Either a Series target -> human value whose targets are
                    columns of ``type_behavior`` not used for training, or a tuple ``(val_behavior, val_target)``
                    with its own (type x target) table, e.g. a separate dictator probe.
    method        : "grid" (exhaustive, the source) or "slsqp" (continuous, scipy; for libraries where the grid
                    exceeds ``max_grid`` points).
    ident_tol     : two types count as identical when their training behaviour differs by at most this anywhere.
                    Identical types (or a rank-deficient [behaviour; 1]) emit an ``UnidentifiedWarning``.
    """
    if not isinstance(type_behavior, pd.DataFrame):
        raise TypeError("type_behavior must be a DataFrame (types x targets)")
    if type_behavior.index.has_duplicates:
        raise ValueError("type_behavior has duplicate type rows")
    if len(type_behavior.index) == 0:
        raise ValueError("type_behavior has no types")
    y = _as_target(human_target, "human_target")
    if len(y) == 0:
        raise ValueError("human_target is empty")
    absent = [t for t in y.index if t not in type_behavior.columns]
    if absent:
        raise ValueError(f"human_target names targets that type_behavior lacks: {absent}")
    Btr = type_behavior.loc[:, list(y.index)].astype(float)
    if Btr.isna().to_numpy().any():
        _raise_missing(Btr, "type_behavior")
    types = list(type_behavior.index)

    # validation set (checked before fitting so a leak fails fast)
    val_B = val_y = None
    if val is not None:
        if isinstance(val, tuple):
            if len(val) != 2:
                raise ValueError("val must be a Series or a (val_behavior, val_target) tuple")
            vb, vy = val
            val_y = _as_target(vy, "val target")
            missing_types = [t for t in types if t not in vb.index]
            if missing_types:
                raise ValueError(f"val_behavior has no row for types {missing_types}")
            absent = [t for t in val_y.index if t not in vb.columns]
            if absent:
                raise ValueError(f"val target names targets that val_behavior lacks: {absent}")
            val_B = vb.loc[types, list(val_y.index)].astype(float)
        else:
            val_y = _as_target(val, "val")
            absent = [t for t in val_y.index if t not in type_behavior.columns]
            if absent:
                raise ValueError(f"val names targets that type_behavior lacks: {absent}")
            val_B = type_behavior.loc[:, list(val_y.index)].astype(float)
        overlap = [t for t in val_y.index if t in set(y.index)]
        if overlap:
            raise ValueError(f"validation targets must be held out, but {overlap} are also training targets")
        if len(val_y) == 0:
            raise ValueError("val is empty")
        if val_B.isna().to_numpy().any():
            _raise_missing(val_B, "validation behavior")

    ident = identifiability(Btr, ident_tol=ident_tol)
    notes = []
    if ident["pairs"]:
        msg = "; ".join(f"{a} and {b}" for a, b in ident["pairs"])
        notes.append(f"identical behaviour on the training targets: {msg} (their weights are not separately "
                     f"identified)")
    if ident["rank"] < len(types) and not ident["pairs"]:
        notes.append(f"[behaviour; 1] has rank {ident['rank']} < {len(types)} types: some weights are not "
                     f"identified (a type is a mix of others, or more types than targets + 1)")
    if warn:
        for n_ in notes:
            warnings.warn(n_, UnidentifiedWarning, stacklevel=2)

    B, yv = Btr.to_numpy(), y.to_numpy()
    weight_range, n_grid, n_ties, step = None, 0, 1, None
    if method == "grid":
        ticks = int(round(1.0 / grid_step))
        if ticks < 1 or abs(ticks * grid_step - 1.0) > 1e-9:
            raise ValueError(f"1/grid_step must be a positive integer (got grid_step={grid_step})")
        size = math.comb(ticks + len(types) - 1, len(types) - 1)
        if size > max_grid:
            raise ValueError(f"the simplex grid has {size:,} points for {len(types)} types at step {grid_step} "
                             f"(max_grid={max_grid:,}); use a coarser grid_step or method='slsqp'")
        w, sse, n_grid, rng, n_ties = _grid_fit(B, yv, ticks, tie_tol)
        weight_range = pd.DataFrame(rng, index=types, columns=["min", "max"])
        step = 1.0 / ticks
        if n_ties > 1:
            notes.append(f"{n_ties} grid points tie with the best SSE; the reported weights are the first in "
                         f"lexicographic type order, see weight_range")
    elif method == "slsqp":
        w, sse = _slsqp_fit(B, yv)
    else:
        raise ValueError("method must be 'grid' or 'slsqp'")

    weights = pd.Series(w, index=types, name="weight")
    train_pred = pd.Series(w @ B, index=y.index, name="mixture")
    val_rmse = val_pred = None
    if val_B is not None:
        val_pred = pd.Series(w @ val_B.to_numpy(), index=val_y.index, name="mixture")
        val_rmse = float(np.sqrt(np.mean((val_pred.to_numpy() - val_y.to_numpy()) ** 2)))
    return MixtureFit(weights=weights, train_sse=float(sse), val_rmse=val_rmse, unidentified_pairs=ident["pairs"],
                      identified=ident["identified"], rank=ident["rank"], train_pred=train_pred, val_pred=val_pred,
                      weight_range=weight_range, n_grid=n_grid, n_ties=n_ties, grid_step=step, method=method,
                      notes=notes)


# ---------------------------------------------------------------- probes (LLM.map)

@dataclass
class ProbeGame:
    """One calibration probe. ``user`` is the game text (the type text goes in as system). ``parse(text)``
    returns a value or None (invalid; never guessed). ``stat(values)`` turns the valid values of one type into a
    float (column = game name) or a dict of floats (columns ``<game>:<key>``)."""
    user: str
    parse: Callable[[str], Any]
    stat: Callable[[list], float | Mapping[str, float]]
    wording_id: str | None = None


_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _parse_choice(options: Sequence[str]):
    opts = [str(o) for o in options]
    low = {o.lower(): o for o in opts}

    def parse(text: str):
        t = re.sub(r"[\s\"'`*.!:;,()\[\]]+", " ", str(text)).strip().lower()
        if t in low:
            return low[t]
        hits = [o for o in opts if re.search(rf"\b{re.escape(o.lower())}\b", t)]
        return hits[0] if len(hits) == 1 else None
    return parse


def _parse_number(lo: float | None, hi: float | None):
    def parse(text: str):
        found = {float(x.replace(",", ".")) for x in _NUM.findall(str(text))}
        if len(found) != 1:                           # none, or ambiguous ("between 20 and 30"): invalid
            return None
        v = found.pop()
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            return None
        return v
    return parse


def choice_game(user: str, options: Sequence[str] = ("Left", "Right"), target: str | None = None, *,
                instruction: bool = True, wording_id: str | None = None) -> ProbeGame:
    """Multiple-choice probe; the statistic is the share of valid answers equal to ``target``
    (default: the first option)."""
    target = options[0] if target is None else target
    if target not in options:
        raise ValueError("target must be one of options")
    text = user + (f"\nAnswer with exactly one word: {' or '.join(options)}." if instruction else "")
    return ProbeGame(user=text, parse=_parse_choice(options),
                     stat=lambda vals: sum(1 for v in vals if v == target) / len(vals), wording_id=wording_id)


def numeric_game(user: str, stat: Callable[[list], float | Mapping[str, float]] | None = None, *,
                 lo: float | None = None, hi: float | None = None, instruction: bool = True,
                 wording_id: str | None = None) -> ProbeGame:
    """Numeric probe; a reply must contain exactly one number inside [lo, hi]. Default statistic: mean."""
    text = user + ("\nAnswer with a single number only." if instruction else "")
    return ProbeGame(user=text, parse=_parse_number(lo, hi), stat=stat or (lambda vals: float(np.mean(vals))),
                     wording_id=wording_id)


def dictator_game(pie: int = 100, *, wording_id: str | None = "edsl-test/homo_silicus/q_give") -> ProbeGame:
    """The edsl-test validation probe: one-shot dictator game. Statistics match ``ENGEL_DICTATOR``:
    mean give fraction, P(give 0), P(give pie/2)."""
    half = pie / 2

    def stat(vals):
        return {"mean_give_frac": float(np.mean(vals)) / pie,
                "p_give_0": sum(1 for v in vals if v == 0) / len(vals),
                "p_give_50": sum(1 for v in vals if v == half) / len(vals)}
    return numeric_game(f"You have ${pie}. You may give any amount of it to a stranger and keep the rest. The "
                        f"stranger has no say and you will never meet them. How much do you give?",
                        stat, lo=0, hi=pie, wording_id=wording_id)


def charness_rabin_games(games: Mapping[str, tuple] | None = None, *,
                         wording_id: str = "edsl-test/homo_silicus/q_choice") -> dict[str, ProbeGame]:
    """The six CR 2002 unilateral menus as choice probes (statistic: P(Left)); targets: ``CR_HUMAN_LEFT``.
    The respondent is Person B (the decider), as in the source prompt."""
    out = {}
    for g, (L, R, *_rest) in (games or CR_GAMES).items():
        out[g] = choice_game(
            "You are deciding on an allocation for yourself and another person, Person A.\n"
            f"Option Left: you get ${L[1]}, Person A gets ${L[0]}.\n"
            f"Option Right: you get ${R[1]}, Person A gets ${R[0]}.\n"
            "Which option do you choose?", ("Left", "Right"), "Left", wording_id=wording_id)
    return out


def _as_game(g) -> ProbeGame:
    if isinstance(g, ProbeGame):
        return g
    if isinstance(g, Mapping):
        return ProbeGame(**dict(g))
    raise TypeError("each game must be a ProbeGame or a dict with user / parse / stat")


def probe_types(llm, type_texts: Mapping[str, str], games: Mapping[str, Any], *, reps: int = 10,
                max_tokens: int = 256, sample_kw: str | None = "sample", parallel: int | None = None,
                prefix: str | None = None, persona_method: str = "theory_type", min_valid: int = 1,
                return_rows: bool = False, **llm_kw):
    """Measure each type's behaviour on each probe through ``llm.map`` -> DataFrame (type x target).

    type_texts : {type: persona text}, sent as ``system`` (optionally after ``prefix``, e.g. HUMANIZING_PREFIX).
    games      : {game: ProbeGame}; see ``choice_game``, ``numeric_game``, ``charness_rabin_games``,
                 ``dictator_game``.
    reps       : answers per (type, game). Repetition r is passed as ``sample=r`` (``sample_kw``) so caching
                 backends give independent draws; set sample_kw=None for a backend that rejects the keyword (then
                 a cached backend may repeat one answer, i.e. effective n = 1).
    min_valid  : a cell with fewer valid answers is left missing (NaN). Nothing is imputed; invalid answers are
                 counted in ``attrs["n_valid"]`` / ``attrs["n_invalid"]`` and a ``UserWarning`` reports
                 them.

    The returned frame has ``attrs["n_valid"]`` and ``attrs["n_invalid"]`` ({"type/game": count}) and
    ``attrs["reps"]``. ``return_rows=True`` also returns one row per answer with persona_method, agent_id, seed
    (= repetition), type, persona parameters, max_tokens, wording_id, game, text, value and valid."""
    if reps < 1:
        raise ValueError("reps must be >= 1")
    gs = {str(k): _as_game(v) for k, v in games.items()}
    ts = {str(k): str(v) for k, v in type_texts.items()}
    if not gs or not ts:
        raise ValueError("type_texts and games must be non-empty")
    jobs = [(t, g) for t in ts for g in gs]
    prompts = [((f"{prefix} {ts[t]}" if prefix else ts[t]), gs[g].user) for t, g in jobs]
    rows = []
    map_kw = dict(llm_kw)
    if parallel is not None:
        map_kw["parallel"] = parallel
    for r in range(reps):
        kw = dict(map_kw)
        if sample_kw and reps > 1:
            kw[sample_kw] = r
        resps = llm.map(prompts, max_tokens=max_tokens, **kw)
        if len(resps) != len(prompts):
            raise RuntimeError(f"llm.map returned {len(resps)} answers for {len(prompts)} prompts")
        for (t, g), resp in zip(jobs, resps):
            text = getattr(resp, "text", resp)
            try:
                v = gs[g].parse(text) if isinstance(text, str) else None
            except (ValueError, TypeError, KeyError):
                v = None
            rows.append({"persona_method": persona_method, "agent_id": t, "seed": r, "type": t,
                         "persona_params": {"type": t, "prefix": bool(prefix)}, "max_tokens": max_tokens,
                         "wording_id": gs[g].wording_id or g, "game": g, "rep": r, "text": text, "value": v,
                         "valid": v is not None})
    cols, data, n_valid, n_invalid = [], {}, {}, {}
    for g in gs:
        for t in ts:
            vals = [x["value"] for x in rows if x["type"] == t and x["game"] == g and x["valid"]]
            n_valid[f"{t}/{g}"] = len(vals)
            n_invalid[f"{t}/{g}"] = reps - len(vals)
            if len(vals) < max(1, min_valid):
                s = None
            else:
                s = gs[g].stat(vals)
            if isinstance(s, Mapping):
                for k, v in s.items():
                    data.setdefault(f"{g}:{k}", {})[t] = float(v)
            else:
                data.setdefault(g, {})[t] = np.nan if s is None else float(s)
        sub = [c for c in data if c.startswith(f"{g}:")]
        for c in sub:                                 # dict statistics: a type without valid answers -> NaN
            for t in ts:
                data[c].setdefault(t, np.nan)
            if c not in cols:
                cols.append(c)
        if not sub and g not in data:                 # no type produced anything: keep a visible NaN column
            data[g] = {t: np.nan for t in ts}
        if g in data and g not in cols:
            cols.append(g)
    out = pd.DataFrame({c: pd.Series(data[c]) for c in cols}).reindex(list(ts))
    out.index.name = "type"
    out.attrs.update({"n_valid": n_valid, "n_invalid": n_invalid, "reps": reps})
    bad = {k: v for k, v in n_invalid.items() if v}
    if bad:
        warnings.warn(f"{sum(bad.values())} invalid answers (left missing, not imputed) in {len(bad)} cells, "
                      f"e.g. {next(iter(bad))}: {next(iter(bad.values()))}/{reps}", UserWarning, stacklevel=2)
    if return_rows:
        return out, pd.DataFrame(rows)
    return out

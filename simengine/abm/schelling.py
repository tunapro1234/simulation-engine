"""Schelling (1971) segregation on the grid core: threshold rule, Zhang (2004) swap game, LLM-agent movers.

Three dynamics share one lattice (``simengine.abm.grid`` convention: int (H, W), EMPTY = -1, types 0..K-1):

* :class:`Schelling` (threshold rule). An agent is happy when like neighbours / occupied neighbours >= its
  threshold (Schelling's "at least a third alike" is threshold = 1/3; an agent with no occupied neighbour is happy
  by default). Each step is one asynchronous sweep in a fresh random order; an agent that is unhappy when its turn
  comes moves (``movers="unhappy"``, Schelling / Mesa / NetLogo) or every agent tries (``movers="all"``, the
  Gauvin-Vannimenus-Nadal 2009 variant). Destination ``rule``: ``"random"`` empty cell, ``"satisfying"`` random
  empty cell where it would be happy (stay if none), ``"nearest"`` closest such cell in squares traversed horizontally + vertically (Schelling 1971).
* :class:`ZhangSwap` (Zhang 2004, JEBO 54:533). No vacancies; each period one random agent of type 0 and one of
  type 1 consider trading places. Payoff u(x) is kinked with a peak at a 50-50 neighbourhood
  (u = Z x/n for x <= n, (2Z - M) + (M - Z) x/n above, n = k/2, Z = 1, M = 0.6 as in his Table 4); the swap happens
  with logit probability exp(bV) / (exp(bU) + exp(bV)) (U, V = summed payoff without / with the swap). The number of
  unlike neighbour pairs rho is the (negative) potential, so it falls in the long run although every agent prefers
  mixing.
* :class:`LLMMover` plugged into ``Schelling(mover=...)``. Each step the asked agents (``ask="unhappy"`` under the
  threshold rule, or ``"all"``: the LLM replaces the threshold, as in Cheng et al. 2024, arXiv 2412.10428) get a
  prompt that describes their 8 (k) surrounding homes, optionally ``n_options`` random vacancies to choose from and
  an optional persona text. Prompts are built from the state at the start of the step, de-duplicated and sent in one
  ``llm.map``; replies are parsed by :func:`parse_decision` (JSON ``{"move": ...}`` / ``{"choice": ...}``, Cheng
  style ``yes_rating`` / ``no_rating``, or plain yes / no / stay / option n); unparsed replies stay and are counted.
  Movers then relocate in random order (a chosen vacancy taken earlier in the step counts as ``blocked``).

Metrics: :func:`grid.same_type_share` (Schelling's mean like-neighbour share), :func:`unlike_pairs` (Zhang's rho),
:func:`segregation_coefficient` (Gauvin et al. 2009: s = 2 / N^2 sum_c n_c^2 over nearest-neighbour clusters,
1 = two clusters), :func:`grid.dissimilarity` (Duncan D on tiles). :func:`tipping_curve` sweeps thresholds x seeds.

Usage::

    import simengine.abm.grid as G
    import simengine.abm.schelling as S
    g = G.Grid((50, 50), torus=True)
    run = S.Schelling(g, threshold=1 / 3).run(n_steps=100, seed=0)
    run.history.tail(1)[["same_share", "happy", "moved"]]        # ~0.5 -> ~0.75 like share, all happy
    curve = S.tipping_curve([0.1, 0.3, 0.5, 0.7, 0.8], seeds=range(3), grid=g)

    from simengine.llm.fake import FakeLLM
    mover = S.LLMMover(FakeLLM(S.threshold_responder(0.5)), labels=("blue", "orange"),
                       persona={0: "retired teacher", 1: "young nurse"})
    run = S.Schelling(g, threshold=0.5, mover=mover).run(n_steps=30, seed=0)
    run.llm                                                       # calls, unique prompts, invalid, blocked

Measured tipping curve, Gauvin-variant check, Zhang swap and LLM-path numbers: docs/modules/abm/schelling.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from . import grid as G
from ..llm.jsonx import json_extract

RULES = ("random", "satisfying", "nearest")
MOVERS = ("unhappy", "all")
_EPS = 1e-9


# ---------------------------------------------------------------- metrics
def unlike_pairs(lattice: np.ndarray, grid: G.Grid) -> int:
    """Zhang's rho: number of unordered neighbouring pairs of occupied cells with different types."""
    lat = np.asarray(lattice)
    occ = lat != G.EMPTY
    same = grid.type_counts(lat)                                   # (K, H, W)
    tot = same.sum(0)
    like = np.take_along_axis(same, np.where(occ, lat, 0)[None], 0)[0]
    return int(((tot - like) * occ).sum() // 2)


def _components(mask: np.ndarray, torus: bool) -> np.ndarray:
    """Sizes of the 4-connected components of mask (components touching across wrapped edges merged on a torus)."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros(0, int)
    if torus:
        parent = np.arange(n + 1)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for x, y in zip(np.r_[lab[0], lab[:, 0]], np.r_[lab[-1], lab[:, -1]]):
            if x and y:
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[max(rx, ry)] = min(rx, ry)
        lab = np.array([find(i) for i in range(n + 1)])[lab]
    sizes = np.bincount(lab.ravel())[1:]
    return sizes[sizes > 0]


def segregation_coefficient(lattice: np.ndarray, torus: bool = False) -> float:
    """Gauvin, Vannimenus & Nadal (2009, EPJ B 70:293) eq. 2: s = 2 / N^2 * sum_c n_c^2, clusters = 4-connected
    same-type components, N = number of agents. 1 for two equal one-type clusters, -> 0 for a fine mix.
    (For K types the same formula with 2 replaced by K.)"""
    lat = np.asarray(lattice)
    n = int((lat != G.EMPTY).sum())
    types = [t for t in np.unique(lat) if t != G.EMPTY]
    if n == 0 or not types:
        return float("nan")
    ss = sum(float((_components(lat == t, torus).astype(float) ** 2).sum()) for t in types)
    return len(types) / n ** 2 * ss


# ---------------------------------------------------------------- state
class _State:
    """Flat lattice with incremental neighbour type counts and an O(1) empty-cell set."""

    def __init__(self, grid: G.Grid, lattice: np.ndarray, n_types: int):
        self.grid, self.K = grid, n_types
        self.nbr = grid.neighbour_table()                           # -1 = outside -> sentinel column
        self.size = grid.size
        self.lat = np.full(self.size + 1, G.EMPTY, np.int64)
        self.lat[:-1] = np.asarray(lattice).ravel()
        self.cnt = np.zeros((n_types, self.size + 1), np.int64)
        tc = grid.type_counts(np.asarray(lattice), n_types).reshape(n_types, -1)
        self.cnt[:, :-1] = tc
        self.tot = self.cnt.sum(0)
        self.empty = np.flatnonzero(self.lat[:-1] == G.EMPTY)
        self.epos = np.full(self.size, -1, np.int64)
        self.epos[self.empty] = np.arange(len(self.empty))
        self.n_empty = len(self.empty)
        occ = np.flatnonzero(self.lat[:-1] != G.EMPTY)
        self.agent_cell = occ.copy()                                 # agent id -> cell (ids = initial row-major order)
        self.cell_agent = np.full(self.size, -1, np.int64)
        self.cell_agent[occ] = np.arange(len(occ))
        self.kind = self.lat[occ].copy()

    def move(self, a: int, dst: int):
        src = int(self.agent_cell[a])
        t = int(self.kind[a])
        ns, nd = self.nbr[src], self.nbr[dst]
        self.cnt[t, ns] -= 1
        self.tot[ns] -= 1
        self.cnt[t, nd] += 1
        self.tot[nd] += 1
        self.lat[src], self.lat[dst] = G.EMPTY, t
        i = self.epos[dst]                                           # dst leaves the empty set, src takes its slot
        self.empty[i] = src
        self.epos[src], self.epos[dst] = i, -1
        self.agent_cell[a] = dst
        self.cell_agent[src], self.cell_agent[dst] = -1, a

    def lattice(self) -> np.ndarray:
        return self.lat[:-1].reshape(self.grid.shape).copy()

    def like_tot(self, a: int, cells: np.ndarray):
        """Like / occupied neighbour counts agent a would have at `cells` (its own current cell vacated)."""
        src, t = int(self.agent_cell[a]), int(self.kind[a])
        like = self.cnt[t, cells].copy()
        tot = self.tot[cells].copy()
        adj = (self.nbr[cells] == src).any(1)                        # cells next to a's current home count a
        like -= adj
        tot -= adj
        return like, tot


def _happy(like, tot, thr, isolated_happy):
    like, tot = np.asarray(like), np.asarray(tot)
    return np.where(tot > 0, like + _EPS >= thr * tot, isolated_happy)


def _thresholds(threshold, n_types) -> np.ndarray:
    t = np.broadcast_to(np.asarray(threshold, float), (n_types,)).copy()
    if ((t < 0) | (t > 1)).any():
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return t


@dataclass
class SchellingRun:
    lattice: np.ndarray                    # final (H, W) lattice
    history: pd.DataFrame                  # one row per step (step 0 = initial): same_share, happy, unlike_pairs,
                                           # no_unlike (share of agents without an unlike neighbour), moved, ...
    steps: int
    converged: bool                        # stopped because a full step moved nobody
    initial: np.ndarray | None = None
    llm: dict = field(default_factory=dict)

    def final(self) -> dict:
        return self.history.iloc[-1].to_dict()


# ---------------------------------------------------------------- threshold model
class Schelling:
    """Schelling checkerboard on a :class:`grid.Grid`.

    fractions : share of cells per type (rest empty); ignored if `lattice` is given to :meth:`run`.
    threshold : minimum like share to be happy; scalar or one per type.
    rule      : destination of a mover ("random" | "satisfying" | "nearest", see module doc).
    movers    : "unhappy" (classic) or "all" (every agent tries a satisfying vacancy each sweep; Gauvin 2009).
    mover     : optional :class:`LLMMover`; then each step is one batched LLM decision round.
    """

    def __init__(self, grid: G.Grid, fractions: Sequence[float] = (0.45, 0.45), threshold: float | Sequence[float]
                 = 1 / 3, rule: str = "random", movers: str = "unhappy", isolated_happy: bool = True,
                 mover: "LLMMover | None" = None, block: int = 5):
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        if movers not in MOVERS:
            raise ValueError(f"movers must be one of {MOVERS}")
        if movers == "all" and rule == "random":
            raise ValueError("movers='all' needs rule='satisfying' or 'nearest' (a happy agent would leave at random)")
        self.grid, self.fractions = grid, tuple(fractions)
        self.n_types = len(self.fractions)
        self.thr = _thresholds(threshold, self.n_types)
        self.rule, self.movers, self.isolated_happy = rule, movers, isolated_happy
        self.mover, self.block = mover, block

    # -- happiness
    def happy_mask(self, st: _State) -> np.ndarray:
        """(N agents,) bool."""
        c = st.agent_cell
        like = st.cnt[st.kind, c]
        return _happy(like, st.tot[c], self.thr[st.kind], self.isolated_happy)

    def _destination(self, st: _State, a: int, rng: np.random.Generator) -> int:
        if st.n_empty == 0:
            return -1
        if self.rule == "random":
            return int(st.empty[rng.integers(st.n_empty)])
        cand = st.empty[:st.n_empty]
        like, tot = st.like_tot(a, cand)
        ok = _happy(like, tot, self.thr[st.kind[a]], self.isolated_happy)
        if not ok.any():
            return -1
        good = cand[ok]
        if self.rule == "satisfying":
            return int(good[rng.integers(len(good))])
        h, w = self.grid.shape
        r0, c0 = divmod(int(st.agent_cell[a]), w)
        r, c = np.divmod(good, w)
        dr, dc = np.abs(r - r0), np.abs(c - c0)
        if self.grid.torus:
            dr, dc = np.minimum(dr, h - dr), np.minimum(dc, w - dc)
        d = dr + dc                                                  # Schelling: squares traversed (Manhattan)
        best = np.flatnonzero(d == d.min())
        return int(good[best[rng.integers(len(best))]])

    def _sweep(self, st: _State, rng) -> dict:
        moved = failed = 0
        n = len(st.kind)
        for a in rng.permutation(n):
            a = int(a)
            if self.movers == "unhappy":
                c = int(st.agent_cell[a])
                t = int(st.kind[a])
                if _happy(st.cnt[t, c], st.tot[c], self.thr[t], self.isolated_happy):
                    continue
            dst = self._destination(st, a, rng)
            if dst < 0:
                failed += 1
                continue
            st.move(a, dst)
            moved += 1
        return dict(moved=moved, failed=failed)

    def _record(self, st: _State, t: int, extra: dict) -> dict:
        lat = st.lattice()
        c = st.agent_cell
        like = st.cnt[st.kind, c]
        rec = dict(step=t, same_share=G.same_type_share(lat, self.grid), happy=float(self.happy_mask(st).mean()),
                   unlike_pairs=unlike_pairs(lat, self.grid), no_unlike=float((like == st.tot[c]).mean()))
        rec.update(extra)
        return rec

    def _still(self, ex: dict) -> bool:
        if ex["moved"]:
            return False
        if self.mover is None:
            return self.movers == "unhappy"
        return ex.get("asked", 0) == 0 or (not self.mover.n_options and ex.get("wanted", 0) == 0)

    def run(self, n_steps: int = 100, seed: G.SeedLike = None, lattice: np.ndarray | None = None,
            stop_when_still: bool = True, metrics: bool = True) -> SchellingRun:
        """n_steps sweeps (LLM rounds with a mover). Stops early (converged=True) after a still step: nobody moved
        under the classic rule with movers='unhappy'; with an LLM mover, nobody was asked, or (no vacancy options)
        nobody chose to move. With options a still round does not stop the run, since the offered vacancies are
        re-drawn each round. metrics=True adds the final segregation coefficient and dissimilarity."""
        rng = G.make_rng(seed)
        if lattice is None:
            lattice = G.random_lattice(self.grid, self.fractions, seed=rng)
        lattice = np.asarray(lattice)
        st = _State(self.grid, lattice, self.n_types)
        if self.mover is not None:
            self.mover.reset()
        hist = [self._record(st, 0, dict(moved=0, failed=0))]
        converged, t = False, 0
        for t in range(1, n_steps + 1):
            if self.mover is None:
                ex = self._sweep(st, rng)
            else:
                ex = self.mover.round(self, st, rng)
                if ex.pop("stopped", False):
                    hist.append(self._record(st, t, ex))
                    break
            hist.append(self._record(st, t, ex))
            if stop_when_still and self._still(ex):
                converged = True
                break
        final = st.lattice()
        h = pd.DataFrame(hist)
        if metrics:
            h.loc[h.index[-1], "seg_coef"] = segregation_coefficient(final, self.grid.torus)
            if self.n_types == 2:
                h.loc[h.index[-1], "dissimilarity"] = G.dissimilarity(final, self.block)
        return SchellingRun(final, h, t, converged, lattice.copy(),
                            self.mover.stats() if self.mover is not None else {})


def tipping_curve(thresholds: Sequence[float], seeds: Sequence[int] = range(5), grid: G.Grid | None = None,
                  n_steps: int = 200, **kw) -> pd.DataFrame:
    """Final metrics per (threshold, seed): same_share, happy, unlike_pairs, seg_coef, dissimilarity, steps,
    converged, plus the initial like share. kw go to :class:`Schelling` (fractions, rule, movers, ...)."""
    grid = grid or G.Grid((50, 50), torus=True)
    rows = []
    for thr in thresholds:
        for s in seeds:
            r = Schelling(grid, threshold=thr, **kw).run(n_steps=n_steps, seed=s)
            f = r.final()
            rows.append(dict(threshold=float(thr), seed=s, same_share0=float(r.history.same_share.iloc[0]),
                             same_share=f["same_share"], happy=f["happy"], unlike_pairs=f["unlike_pairs"],
                             no_unlike=f["no_unlike"],
                             seg_coef=f.get("seg_coef", np.nan), dissimilarity=f.get("dissimilarity", np.nan),
                             steps=r.steps, converged=r.converged))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Zhang (2004) swap game
def zhang_payoff(x, n: float, Z: float = 1.0, M: float = 0.6):
    """Zhang (2004) eq. 1: kinked payoff of x like neighbours out of 2n, peak Z at x = n, M at x = 2n (Z > M > 0)."""
    x = np.asarray(x, float)
    return np.where(x <= n, Z * x / n, (2 * Z - M) + (M - Z) * x / n)


@dataclass
class ZhangRun:
    lattice: np.ndarray
    history: pd.DataFrame                  # pairs, rho, same_share, swaps
    reached: int | None                    # first recorded pair count with rho <= target (None if never)


class ZhangSwap:
    """Zhang's all-integrationist swap game on a full two-type lattice (no vacancies)."""

    def __init__(self, grid: G.Grid, beta: float = 10.0, Z: float = 1.0, M: float = 0.6, share: float = 0.5):
        if not Z > M > 0:
            raise ValueError("need Z > M > 0")
        self.grid, self.beta, self.Z, self.M, self.share = grid, beta, Z, M, share
        self.n = grid.k / 2

    def run(self, n_pairs: int, seed: G.SeedLike = None, lattice: np.ndarray | None = None, record_every: int = 10000,
            target_rho: int | None = None) -> ZhangRun:
        rng = G.make_rng(seed)
        if lattice is None:
            lattice = G.random_lattice(self.grid, [self.share, 1 - self.share], seed=rng)
        lat = np.asarray(lattice).ravel().astype(np.int64)
        if (lat == G.EMPTY).any():
            raise ValueError("Zhang's model has no vacancies")
        nbr = self.grid.neighbour_table()
        if (nbr < 0).any():
            raise ValueError("Zhang's model needs a torus (every agent has 2n neighbours)")
        k = nbr.shape[1]
        like = (lat[nbr] == lat[:, None]).sum(1)                    # like-neighbour count per cell
        u = zhang_payoff(np.arange(k + 1), self.n, self.Z, self.M)
        cells = [np.flatnonzero(lat == t) for t in (0, 1)]          # cells held by each type
        rho = int((k - like).sum() // 2)
        hist, reached, swaps = [], None, 0
        ia = rng.integers(len(cells[0]), size=n_pairs)
        ib = rng.integers(len(cells[1]), size=n_pairs)
        draws = rng.random(n_pairs)
        for p in range(n_pairs + 1):
            if p % record_every == 0 or p == n_pairs:
                hist.append(dict(pairs=p, rho=rho, same_share=float(like.mean() / k), swaps=swaps))
                if target_rho is not None and reached is None and rho <= target_rho:
                    reached = p
                    break
            if p == n_pairs:
                break
            A, B = int(cells[0][ia[p]]), int(cells[1][ib[p]])
            nA, nB = nbr[A], nbr[B]
            adj = int((nA == B).any())
            xa, xb = int(like[A]), int(like[B])
            # after the swap, the type-0 agent sits at B: its like count = (k - like[B]) unlike-of-1 cells, minus A
            ya, yb = k - xb - adj, k - xa - adj
            dv = (u[ya] + u[yb]) - (u[xa] + u[xb])
            if draws[p] >= 1.0 / (1.0 + np.exp(-self.beta * dv)):
                continue
            # neighbours of A lose a type-0 neighbour and gain a type-1 one; neighbours of B the reverse
            # (a common neighbour gets -1 and +1; A and B themselves are overwritten below)
            like[nA] += np.where(lat[nA] == 0, -1, 1)
            like[nB] += np.where(lat[nB] == 1, -1, 1)
            lat[A], lat[B] = 1, 0
            like[A], like[B] = yb, ya
            cells[0][ia[p]], cells[1][ib[p]] = B, A
            rho += (xa + xb) - (ya + yb)                             # edge A-B stays unlike; others follow like counts
            swaps += 1
        return ZhangRun(lat.reshape(self.grid.shape), pd.DataFrame(hist), reached)


# ---------------------------------------------------------------- LLM movers
@dataclass
class Decision:
    move: bool
    option: int | None = None              # 1-based vacancy option chosen (None: any / no options offered)
    valid: bool = True


_YES = re.compile(r"\b(yes|move|moving|leave|relocate)\b", re.I)
_NO = re.compile(r"\b(no|stay|staying|remain)\b", re.I)
_OPT = re.compile(r"\b(?:option|home|choice)\s*#?\s*(\d+)\b", re.I)


def _as_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("yes", "true", "move", "1", "y"):
            return True
        if s in ("no", "false", "stay", "0", "n"):
            return False
    return None


def parse_decision(text: str | None, n_options: int = 0) -> Decision:
    """Parse a move / stay reply. Accepted (first match wins):
    JSON {"choice": "stay" | 1..n} or {"move": bool|"yes"|"no", "option": 1..n}; Cheng et al. style
    {"yes_rating": r1, "no_rating": r0} (move iff r1 > r0); plain text "stay" / "no" vs "move" / "yes" /
    "option 2" (a bare number is an option when options were offered). Anything else -> stay, valid=False.
    An option outside 1..n is invalid."""
    def opt(v):
        try:
            i = int(v)
        except (TypeError, ValueError):
            return None
        return i if 1 <= i <= n_options else None

    obj = json_extract(text, default=None)
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]
    if isinstance(obj, dict):
        o = {str(k).lower(): v for k, v in obj.items()}
        for key in ("choice", "option", "destination"):
            if key in o and o[key] is not None:
                v = o[key]
                if _as_bool(v) is False or (isinstance(v, str) and v.strip().lower() == "stay"):
                    return Decision(False)
                if n_options and opt(v) is not None:
                    return Decision(True, opt(v))
                if n_options:
                    return Decision(False, valid=False)
        if "move" in o:
            b = _as_bool(o["move"])
            if b is not None:
                return Decision(b)
        if "yes_rating" in o and "no_rating" in o:
            try:
                return Decision(float(o["yes_rating"]) > float(o["no_rating"]))
            except (TypeError, ValueError):
                pass
        return Decision(False, valid=False)
    if not text:
        return Decision(False, valid=False)
    m = _OPT.search(text)
    if m and n_options:
        return Decision(True, opt(m.group(1))) if opt(m.group(1)) else Decision(False, valid=False)
    s = text.strip()
    if n_options and re.fullmatch(r"\d+", s):
        return Decision(True, opt(s)) if opt(s) else Decision(False, valid=False)
    yes, no = _YES.search(s), _NO.search(s)
    if yes and (not no or yes.start() < no.start()):
        return Decision(True)
    if no:
        return Decision(False)
    return Decision(False, valid=False)


def describe_cells(counts: Sequence[int], own: int, labels: Sequence[str], k: int) -> str:
    """'3 are <own> (your group), 4 are <other>, 1 is empty' from per-type counts around a home."""
    parts = [f"{counts[own]} {'is' if counts[own] == 1 else 'are'} {labels[own]} households (your group)"]
    parts += [f"{counts[t]} {'is' if counts[t] == 1 else 'are'} {labels[t]} households"
              for t in range(len(labels)) if t != own]
    e = k - int(sum(counts))
    parts.append(f"{e} {'is' if e == 1 else 'are'} empty")
    return ", ".join(parts)


DEFAULT_SYSTEM = ("You are a resident deciding where to live in a town laid out as a grid of homes. "
                  "Answer only with the JSON asked for.")


class LLMMover:
    """Batched LLM decision round for :class:`Schelling`.

    llm       : any simengine LLM (FakeLLM in tests; Claude / ClaudeCode for real runs).
    labels    : one group name per type, used in the prompt (default "group A", "group B", ...).
    persona   : None | str (everyone) | {type: str} | sequence of str per agent id | callable(agent_id, type) -> str.
                The text is added to the system prompt ("About you: ...").
    ask       : "unhappy" (only agents unhappy under the model threshold are asked) | "all".
    n_options : 0 -> the reply is move / stay and a mover goes to a destination by the model's `rule`;
                n > 0 -> up to n random vacancies are described and the agent picks one or stays.
    dedupe    : identical (system, user) prompts in a round are sent once and share the reply (with options or
                per-agent personas prompts are rarely identical).
    max_calls : stop the run (cleanly, flagged) before a round that would exceed it.
    """

    def __init__(self, llm, labels: Sequence[str] | None = None, persona=None, ask: str = "unhappy",
                 n_options: int = 0, dedupe: bool = True, max_calls: int | None = None, max_tokens: int = 60,
                 parallel: int = 8, system: str = DEFAULT_SYSTEM, llm_kwargs: dict | None = None):
        if ask not in ("unhappy", "all"):
            raise ValueError("ask must be 'unhappy' or 'all'")
        self.llm, self.labels, self.persona, self.ask = llm, labels, persona, ask
        self.n_options, self.dedupe, self.max_calls = int(n_options), dedupe, max_calls
        self.max_tokens, self.parallel, self.system = max_tokens, parallel, system
        self.llm_kwargs = dict(llm_kwargs or {})
        self.reset()

    def reset(self):
        self.calls = self.asked = self.invalid = self.blocked = 0
        self.cost = 0.0
        self.log: list[dict] = []

    def stats(self) -> dict:
        return dict(calls=self.calls, asked=self.asked, invalid=self.invalid, blocked=self.blocked,
                    cost_usd=self.cost, invalid_rate=self.invalid / self.asked if self.asked else 0.0)

    # -- prompt parts
    def persona_text(self, agent: int, kind: int) -> str | None:
        p = self.persona
        if p is None:
            return None
        if isinstance(p, str):
            return p
        if callable(p):
            return p(agent, kind)
        if isinstance(p, dict):
            return p.get(kind)
        return p[agent]

    def system_prompt(self, agent: int, kind: int) -> str:
        p = self.persona_text(agent, kind)
        return self.system + (f"\nAbout you: {p}" if p else "")

    def user_prompt(self, kind: int, counts, k: int, options: Sequence[Sequence[int]] = (),
                    option_k: Sequence[int] = ()) -> str:
        labels = self.labels or [f"group {chr(65 + i)}" for i in range(len(counts))]
        txt = (f"You are one of the {labels[kind]} households. Of the {k} homes around yours, "
               f"{describe_cells(counts, kind, labels, k)}.\n")
        if options:
            txt += "Vacant homes you could move to:\n"
            for i, oc in enumerate(options, 1):
                ko = option_k[i - 1] if len(option_k) else k
                txt += f"Option {i}: of the {ko} homes around it, {describe_cells(oc, kind, labels, ko)}.\n"
            txt += ('Do you stay or move to one of the options? Reply with JSON only: {"choice": "stay"} or '
                    '{"choice": <option number>}.')
        else:
            txt += 'Do you move to another home or stay? Reply with JSON only: {"move": true} or {"move": false}.'
        return txt

    # -- one round
    def round(self, model: Schelling, st: _State, rng: np.random.Generator) -> dict:
        n = len(st.kind)
        happy = model.happy_mask(st)
        who = np.flatnonzero(~happy) if self.ask == "unhappy" else np.arange(n)
        if len(who) == 0:
            return dict(moved=0, failed=0, asked=0, calls=0, invalid=0, wanted=0, blocked=0)
        cells = st.agent_cell[who]
        counts = st.cnt[:, cells].T                                  # (m, K) neighbours by type
        deg = (st.nbr >= 0).sum(1)                                   # in-grid neighbours (k on a torus)
        opts: list[np.ndarray] = []
        prompts = []
        for j, a in enumerate(who):
            a, t = int(a), int(st.kind[a])
            oc, ok = [], []
            if self.n_options and st.n_empty:
                o = st.empty[:st.n_empty][rng.choice(st.n_empty, min(self.n_options, st.n_empty), replace=False)]
                opts.append(o)
                like_tot = st.cnt[:, o].T.copy()
                adj = (st.nbr[o] == st.agent_cell[a]).any(1)          # the mover's own home becomes empty
                like_tot[adj, t] -= 1
                oc = [list(map(int, r)) for r in like_tot]
                ok = [int(x) for x in deg[o]]
            else:
                opts.append(np.zeros(0, np.int64))
            prompts.append((self.system_prompt(a, t),
                            self.user_prompt(t, list(map(int, counts[j])), int(deg[cells[j]]), oc, ok)))
        uniq = list(dict.fromkeys(prompts)) if self.dedupe else prompts
        if self.max_calls is not None and self.calls + len(uniq) > self.max_calls:
            return dict(moved=0, failed=0, asked=0, calls=0, invalid=0, wanted=0, blocked=0, stopped=True)
        res = self.llm.map(uniq, max_tokens=self.max_tokens, parallel=self.parallel, **self.llm_kwargs)
        self.calls += len(uniq)
        self.cost += float(sum(getattr(r, "cost_usd", 0.0) or 0.0 for r in res))
        reply = dict(zip(uniq, (r.text for r in res))) if self.dedupe else None
        dec = []
        for j, pr in enumerate(prompts):
            text = reply[pr] if self.dedupe else res[j].text
            dec.append(parse_decision(text, len(opts[j])))
        inv = sum(not d.valid for d in dec)
        self.asked += len(who)
        self.invalid += inv
        moved = failed = blocked = 0
        wanted = sum(d.move for d in dec)
        for j in rng.permutation(len(who)):
            d, a = dec[j], int(who[j])
            if not d.move:
                continue
            if d.option is not None:
                dst = int(opts[j][d.option - 1])
                if st.lat[dst] != G.EMPTY:
                    blocked += 1
                    continue
            else:
                dst = model._destination(st, a, rng)
                if dst < 0:
                    failed += 1
                    continue
            st.move(a, dst)
            moved += 1
        self.blocked += blocked
        self.log.append(dict(asked=len(who), calls=len(uniq), invalid=inv, wanted=wanted, moved=moved,
                             blocked=blocked))
        return dict(moved=moved, failed=failed, asked=len(who), calls=len(uniq), invalid=inv, wanted=wanted,
                    blocked=blocked)


_COUNT = re.compile(r"(\d+) (?:is|are) (.+?) households( \(your group\))?")
_EMPTY = re.compile(r"(\d+) (?:is|are) empty")


def _parse_counts(segment: str) -> tuple[int, int]:
    """(like, occupied) from a describe_cells() sentence."""
    like = tot = 0
    for m in _COUNT.finditer(segment):
        v = int(m.group(1))
        tot += v
        if m.group(3):
            like += v
    return like, tot


def threshold_responder(threshold: float, isolated_happy: bool = True) -> Callable[[str, str], str]:
    """FakeLLM responder that applies the classic rule to the prompt text: stay when like / occupied >= threshold,
    otherwise move (to the first option that satisfies the rule, else stay). For tests and sanity runs."""
    def ok(like, tot):
        return (like + _EPS >= threshold * tot) if tot > 0 else isolated_happy

    def respond(system: str, user: str) -> str:
        lines = user.split("\n")
        here = _parse_counts(lines[0])
        opts = [_parse_counts(ln) for ln in lines if ln.startswith("Option ")]
        if ok(*here):
            return '{"choice": "stay"}' if opts else '{"move": false}'
        if not opts:
            return '{"move": true}'
        for i, o in enumerate(opts, 1):
            if ok(*o):
                return f'{{"choice": {i}}}'
        return '{"choice": "stay"}'
    return respond


__all__ = ["Schelling", "SchellingRun", "tipping_curve", "unlike_pairs", "segregation_coefficient", "ZhangSwap",
           "ZhangRun", "zhang_payoff", "LLMMover", "Decision", "parse_decision", "describe_cells",
           "threshold_responder", "RULES", "MOVERS", "DEFAULT_SYSTEM"]

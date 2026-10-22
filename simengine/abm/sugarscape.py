"""Sugarscape (Epstein & Axtell 1996, *Growing Artificial Societies*, ch. II and IV): agents with vision and
metabolism forage on a two-peak sugar landscape; wealth inequality emerges from near-equal endowments.

    from simengine.abm.sugarscape import Sugarscape, SugarscapeConfig, gini, lorenz
    run = Sugarscape(SugarscapeConfig(n_agents=250), seed=0).run(500)    # rules ({G1}, {M, R[60,100]})
    run.stats[["step", "population", "gini"]]                            # Gini trajectory
    x, y = lorenz(run.snapshots[500])                                    # Lorenz curve of the final wealth

Rules (book names):
* G_alpha (growback): every site regrows `growback` units per step up to its capacity (np.inf = G_inf).
* M (movement): agents act in a fresh random order; each looks `vision` sites along the four lattice directions,
  picks the unoccupied site (its own site included, as in the NetLogo port) with the most sugar, nearest on ties,
  random among equals; moves there, harvests all of it, then burns `metabolism`.
* death / R_[a,b] (replacement): an agent dies when its sugar is <= 0 (with spice: either good <= 0) or its age
  exceeds max_age ~ U{a..b}; with `replace=True` it is replaced by a random newborn at a random empty site, so the
  population stays constant. `max_age=None` turns ageing off (pure starvation, carrying-capacity runs).
* spice / T (optional, off by default): a second good on the mirrored landscape; movement maximises the Cobb-Douglas
  welfare W = s^(m_s/m) p^(m_p/m) (m = m_s + m_p) of holdings plus the site; with `trade=True` von Neumann neighbours
  bargain at the geometric mean of their MRS (MRS = (p/m_p)/(s/m_s)), one unit at a time while both gain and their
  MRSs do not cross.

Landscape: the default capacity map is parametric (stepped cones at two peaks, levels 4..0, peak centres and ring
radii fitted to the book's map; 91.6% of the 2,500 cells match the NetLogo transcription and none is off by more than
one level). Pass `capacity=` (e.g. `load_capacity(path)` on a whitespace grid file) to run on another map; the
transcribed map file is CC BY-NC-SA and is not shipped.

LLM variant: `LLMPolicy(llm)` replaces rule M. Each step it sends one prompt per living agent (its holdings and the
free visible sites at the start of the step) through `llm.map`; invalid replies and sites taken by an earlier mover
fall back to rule M and are counted. Cost is n_agents x n_steps calls: use a FakeLLM or a disk cache, few agents and
`max_calls`.

Measured numbers (Gini trajectory vs the book) and the command: docs/modules/abm/sugarscape.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .society import extract_json

# peak centres (row, col on a 50x50 grid, row 0 = north) and ring radii for capacity 4, 3, 2, 1
PEAKS = ((10.5, 38.5), (34.5, 14.5))
RADII = (6.0, 11.0, 17.0, 24.0)
DIRS = ((-1, 0, "north"), (1, 0, "south"), (0, 1, "east"), (0, -1, "west"))


def sugar_capacity(size: int = 50, peaks=PEAKS, radii=RADII, levels=(4, 3, 2, 1)) -> np.ndarray:
    """Two-peak stepped capacity map (size, size): level k inside ring k around the nearer peak, 0 outside."""
    s = size / 50.0
    r, c = np.mgrid[0:size, 0:size]
    d = np.min([np.hypot(r - pr * s, c - pc * s) for pr, pc in peaks], axis=0)
    return np.select([d < q * s for q in radii], list(levels), 0).astype(float)


def spice_capacity(size: int = 50, sugar: np.ndarray | None = None) -> np.ndarray:
    """Book layout: spice peaks NW and SE, i.e. the sugar map mirrored left-right."""
    return np.fliplr(sugar_capacity(size) if sugar is None else np.asarray(sugar, float)).copy()


def load_capacity(path: str) -> np.ndarray:
    """Read a whitespace-separated capacity grid (row 0 = north) at runtime."""
    return np.loadtxt(path, dtype=float, ndmin=2)


# ------------------------------------------------------------------ inequality
def gini(w) -> float:
    """Gini coefficient of non-negative holdings (0 = equal, ->1 = one holds all); NaN when empty or all zero."""
    x = np.sort(np.clip(np.asarray(w, float).ravel(), 0, None))
    n, t = len(x), x.sum() if len(x) else 0.0
    if n == 0 or t <= 0:
        return float("nan")
    return float(((2 * np.arange(1, n + 1) - n - 1) * x).sum() / (n * t))


def lorenz(w) -> tuple[np.ndarray, np.ndarray]:
    """Lorenz curve: (population share, wealth share), both starting at 0 and ending at 1."""
    x = np.sort(np.clip(np.asarray(w, float).ravel(), 0, None))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    return np.arange(n + 1) / n, np.concatenate([[0.0], np.cumsum(x) / x.sum()])


def _skew(x) -> float:
    x = np.asarray(x, float)
    sd = x.std()
    return float(((x - x.mean()) ** 3).mean() / sd ** 3) if len(x) > 2 and sd > 0 else float("nan")


def _welfare(s, p, ms, mp):
    m = ms + mp
    return np.power(np.maximum(s, 0), ms / m) * np.power(np.maximum(p, 0), mp / m)


def _mrs(s, p, ms, mp):
    return (p / mp) / (s / ms)


# ------------------------------------------------------------------ model
@dataclass
class SugarscapeConfig:
    """Book defaults for the wealth-distribution run (ch. II): 50x50 torus, vision U{1..6}, metabolism U{1..4},
    endowment U{5..25}, max age U{60..100}, G1. Ranges are inclusive integer ranges."""
    size: int = 50
    n_agents: int = 250
    vision: tuple = (1, 6)
    metabolism: tuple = (1, 4)
    endowment: tuple = (5, 25)
    max_age: tuple | None = (60, 100)
    replace: bool = True
    growback: float = 1.0
    torus: bool = True
    capacity: np.ndarray | None = None
    spice: bool = False
    trade: bool = False
    spice_metabolism: tuple = (1, 4)
    spice_endowment: tuple = (5, 25)
    spice_capacity: np.ndarray | None = None
    max_trades_per_pair: int = 50


@dataclass
class SugarscapeRun:
    stats: pd.DataFrame                       # one row per step (step 0 = initial state)
    snapshots: dict = field(default_factory=dict)   # {step: wealth array of living agents}
    calls: int = 0
    cost_usd: float = 0.0
    invalid: int = 0
    conflicts: int = 0
    stopped: str = "completed"


class Sugarscape:
    """Agent arrays are slot based (one slot per initial agent; a newborn reuses its parent's slot)."""

    def __init__(self, config: SugarscapeConfig | None = None, seed: int = 0, policy=None):
        cfg = self.cfg = config or SugarscapeConfig()
        if cfg.trade and not cfg.spice:
            raise ValueError("trade=True needs spice=True")
        self.rng = np.random.default_rng(seed)
        self.cap = (sugar_capacity(cfg.size) if cfg.capacity is None else np.asarray(cfg.capacity, float)).ravel()
        n_side = int(round(np.sqrt(self.cap.size)))
        if n_side * n_side != self.cap.size:
            raise ValueError("capacity must be a square grid")
        self.side = n_side
        if cfg.n_agents > self.cap.size:
            raise ValueError("more agents than sites")
        self.sugar = self.cap.copy()
        if cfg.spice:
            sp = spice_capacity(n_side, self.cap.reshape(n_side, n_side)) if cfg.spice_capacity is None else \
                cfg.spice_capacity
            self.pcap = np.asarray(sp, float).ravel()
            if self.pcap.size != self.cap.size:
                raise ValueError("spice_capacity must match the sugar grid")
            self.spice = self.pcap.copy()
        self.policy = policy
        n = cfg.n_agents
        self.occ = np.full(self.cap.size, -1)
        self.pos = np.zeros(n, int)
        self.vision, self.metab, self.wealth = np.zeros(n, int), np.zeros(n), np.zeros(n)
        self.age, self.max_age = np.zeros(n, int), np.full(n, np.iinfo(np.int64).max)
        self.mp, self.wp = np.ones(n), np.zeros(n)
        self.alive = np.zeros(n, bool)
        self.uid = np.zeros(n, int)
        self._next_uid = 0
        self.step_no = 0
        free = self.rng.permutation(self.cap.size)[:n]
        for i, site in enumerate(free):
            self._spawn(i, int(site))
        vmax = cfg.vision[1]
        off = [(0, 0, 0, "here")] + [(dr * k, dc * k, k, name) for k in range(1, vmax + 1) for dr, dc, name in DIRS]
        self.off = np.array([(a, b) for a, b, _, _ in off])
        self.off_dist = np.array([k for _, _, k, _ in off])
        self.off_name = [f"{name} {k}" if k else name for _, _, k, name in off]
        self.events = dict(starved=0, aged=0, births=0, trades=0, prices=[])

    # -- agents
    def _draw(self, lo_hi):
        return int(self.rng.integers(lo_hi[0], lo_hi[1] + 1))

    def _spawn(self, i, site):
        cfg = self.cfg
        self.pos[i], self.occ[site] = site, i
        self.vision[i], self.metab[i], self.wealth[i] = self._draw(cfg.vision), self._draw(cfg.metabolism), \
            self._draw(cfg.endowment)
        self.age[i] = 0
        self.max_age[i] = self._draw(cfg.max_age) if cfg.max_age else np.iinfo(np.int64).max
        if cfg.spice:
            self.mp[i], self.wp[i] = self._draw(cfg.spice_metabolism), self._draw(cfg.spice_endowment)
        self.alive[i] = True
        self.uid[i] = self._next_uid
        self._next_uid += 1

    def candidates(self, i) -> tuple[np.ndarray, np.ndarray]:
        """(flat site ids, offset index) of the sites agent i can see, own site first; off-grid sites dropped."""
        k = np.flatnonzero(self.off_dist <= self.vision[i])
        r, c = divmod(int(self.pos[i]), self.side)
        rr, cc = r + self.off[k, 0], c + self.off[k, 1]
        if self.cfg.torus:
            rr, cc = rr % self.side, cc % self.side
        else:
            ok = (rr >= 0) & (rr < self.side) & (cc >= 0) & (cc < self.side)
            rr, cc, k = rr[ok], cc[ok], k[ok]
        return rr * self.side + cc, k

    def _value(self, i, sites):
        if not self.cfg.spice:
            return self.sugar[sites]
        return _welfare(self.wealth[i] + self.sugar[sites], self.wp[i] + self.spice[sites], self.metab[i], self.mp[i])

    def rule_m(self, i) -> int:
        """Best free visible site for agent i under rule M (current occupancy)."""
        sites, k = self.candidates(i)
        free = (self.occ[sites] < 0) | (sites == self.pos[i])
        sites, k = sites[free], k[free]
        v = self._value(i, sites)
        best = np.isclose(v, v.max(), rtol=1e-12, atol=1e-12)
        d = self.off_dist[k]
        best &= d == d[best].min()
        return int(sites[self.rng.choice(np.flatnonzero(best))])

    def _move(self, i, site):
        self.occ[self.pos[i]] = -1
        self.pos[i], self.occ[site] = site, i
        self.wealth[i] += self.sugar[site] - self.metab[i]
        self.sugar[site] = 0.0
        if self.cfg.spice:
            self.wp[i] += self.spice[site] - self.mp[i]
            self.spice[site] = 0.0

    # -- trade (rule T)
    def _trade_pair(self, a, b):
        n = 0
        for _ in range(self.cfg.max_trades_per_pair):
            ma = _mrs(self.wealth[a], self.wp[a], self.metab[a], self.mp[a])
            mb = _mrs(self.wealth[b], self.wp[b], self.metab[b], self.mp[b])
            if np.isclose(ma, mb):
                break
            hi, lo = (a, b) if ma > mb else (b, a)          # hi values sugar more: buys sugar, pays spice
            p = float(np.sqrt(ma * mb))
            ds, dp = (1.0, p) if p > 1 else (1.0 / p, 1.0)
            s_hi, p_hi = self.wealth[hi] + ds, self.wp[hi] - dp
            s_lo, p_lo = self.wealth[lo] - ds, self.wp[lo] + dp
            if min(p_hi, s_lo) <= 0:
                break
            if (_welfare(s_hi, p_hi, self.metab[hi], self.mp[hi]) <= _welfare(self.wealth[hi], self.wp[hi],
                                                                              self.metab[hi], self.mp[hi]) or
                    _welfare(s_lo, p_lo, self.metab[lo], self.mp[lo]) <= _welfare(self.wealth[lo], self.wp[lo],
                                                                                  self.metab[lo], self.mp[lo])):
                break
            if _mrs(s_hi, p_hi, self.metab[hi], self.mp[hi]) < _mrs(s_lo, p_lo, self.metab[lo], self.mp[lo]):
                break                                          # MRSs would cross over
            self.wealth[hi], self.wp[hi], self.wealth[lo], self.wp[lo] = s_hi, p_hi, s_lo, p_lo
            self.events["prices"].append(p)
            n += 1
        return n

    def _trade(self):
        ids = np.flatnonzero(self.alive)
        r, c = np.divmod(self.pos[ids], self.side)
        pairs = []
        for dr, dc in ((1, 0), (0, 1)):          # each von Neumann pair once
            rr, cc = r + dr, c + dc
            if self.cfg.torus:
                rr, cc = rr % self.side, cc % self.side
            ok = (rr < self.side) & (cc < self.side)
            j = np.full(len(ids), -1)
            j[ok] = self.occ[rr[ok] * self.side + cc[ok]]
            m = j >= 0
            pairs += list(zip(ids[m].tolist(), j[m].tolist()))
        pairs = list({(min(a, b), max(a, b)) for a, b in pairs if a != b})
        pairs.sort()
        for t in self.rng.permutation(len(pairs)):
            a, b = pairs[t]
            if self.rng.random() < 0.5:
                a, b = b, a
            self.events["trades"] += self._trade_pair(a, b)

    # -- dynamics
    def wealth_now(self) -> np.ndarray:
        """Holdings of the living agents (sugar, or sugar + spice when spice is on)."""
        w = self.wealth[self.alive]
        return w + self.wp[self.alive] if self.cfg.spice else w.copy()

    def step(self, plan: dict | None = None) -> dict:
        """One step: growback, movement (rule M or the policy plan), death/replacement, optional trade."""
        cfg = self.cfg
        self.sugar = np.minimum(self.cap, self.sugar + cfg.growback)
        if cfg.spice:
            self.spice = np.minimum(self.pcap, self.spice + cfg.growback)
        ev = dict(starved=0, aged=0, births=0, trades=0, prices=[], conflicts=0)
        self.events = ev
        plan = plan or {}
        for i in self.rng.permutation(np.flatnonzero(self.alive)):
            site = plan.get(int(i))
            if site is not None and self.occ[site] >= 0 and site != self.pos[i]:
                ev["conflicts"] += 1
                site = None
            self._move(i, self.rule_m(i) if site is None else int(site))
        self.age[self.alive] += 1
        starve = self.alive & ((self.wealth <= 0) | (cfg.spice & (self.wp <= 0)))
        aged = self.alive & ~starve & (self.age > self.max_age)
        ev["starved"], ev["aged"] = int(starve.sum()), int(aged.sum())
        dead = np.flatnonzero(starve | aged)
        self.occ[self.pos[dead]] = -1
        self.alive[dead] = False
        if cfg.replace and len(dead):
            empty = np.flatnonzero(self.occ < 0)
            for i, site in zip(dead, self.rng.choice(empty, len(dead), replace=False)):
                self._spawn(int(i), int(site))
            ev["births"] = len(dead)
        if cfg.trade:
            self._trade()
        self.step_no += 1
        return ev

    def _row(self, ev):
        w = self.wealth_now()
        n = len(w)
        top = np.sort(w)[::-1][:max(1, n // 10)].sum() / w.sum() if n and w.sum() > 0 else np.nan
        row = dict(step=self.step_no, population=n, mean_wealth=float(w.mean()) if n else np.nan, gini=gini(w),
                   skewness=_skew(w), top10_share=float(top), starved=ev.get("starved", 0), aged=ev.get("aged", 0),
                   births=ev.get("births", 0), conflicts=ev.get("conflicts", 0),
                   mean_vision=float(self.vision[self.alive].mean()) if n else np.nan,
                   mean_metabolism=float(self.metab[self.alive].mean()) if n else np.nan)
        if self.cfg.trade:
            lp = np.log(ev.get("prices") or [np.nan])
            row.update(trades=ev.get("trades", 0), price_geo_mean=float(np.exp(np.mean(lp))),
                       price_log_sd=float(np.std(lp)) if len(lp) > 1 else np.nan)
        return row

    def run(self, n_steps: int, snapshots=None) -> SugarscapeRun:
        """Run n_steps; stats row 0 is the initial state. snapshots: steps whose wealth arrays are kept
        (default: 0 and the last)."""
        keep = set([0, n_steps] if snapshots is None else snapshots)
        rows, snaps = [self._row({})], {}
        if 0 in keep:
            snaps[0] = self.wealth_now()
        stopped = "completed"
        for _ in range(n_steps):
            if not self.alive.any():
                stopped = "extinct"
                break
            plan = None
            if self.policy is not None:
                try:
                    plan = self.policy.plan(self)
                except _Budget:
                    stopped = f"budget: max_calls={self.policy.max_calls}"
                    break
            rows.append(self._row(self.step(plan)))
            if self.step_no in keep:
                snaps[self.step_no] = self.wealth_now()
        st = pd.DataFrame(rows)
        pol = self.policy
        return SugarscapeRun(stats=st, snapshots=snaps, calls=getattr(pol, "calls", 0),
                             cost_usd=getattr(pol, "cost_usd", 0.0), invalid=getattr(pol, "invalid", 0),
                             conflicts=int(st.conflicts.sum()), stopped=stopped)


# ------------------------------------------------------------------ LLM policy
class _Budget(Exception):
    pass


SYSTEM = ("You are a forager on a grid landscape. Each turn you move to one of the sites listed (or stay), collect "
          "all the {goods} there, then burn your metabolism. If your {goods_short} stock falls to 0 or below you die."
          " Other foragers occupy some sites; only the free sites you can see are listed.{persona}\n"
          'Answer ONLY with JSON: {{"site": <site number>}}.')


class LLMPolicy:
    """Movement chosen by an LLM (any simengine.types.LLM). One call per living agent per step via `llm.map`.

    persona: optional callable(agent dict) -> str added to the system prompt (agent dict: uid, vision, metabolism,
    sugar, age, spice, spice_metabolism). Invalid replies use rule M (`invalid` counts them)."""

    def __init__(self, llm, max_tokens: int = 32, parallel: int = 4, max_calls: int | None = None,
                 persona: Callable[[dict], str] | None = None, llm_kwargs: dict | None = None):
        self.llm, self.max_tokens, self.parallel, self.max_calls = llm, max_tokens, parallel, max_calls
        self.persona, self.llm_kwargs = persona, dict(llm_kwargs or {})
        self.calls, self.cost_usd, self.invalid = 0, 0.0, 0

    def agent_dict(self, m: Sugarscape, i: int) -> dict:
        d = dict(uid=int(m.uid[i]), vision=int(m.vision[i]), metabolism=float(m.metab[i]),
                 sugar=float(m.wealth[i]), age=int(m.age[i]))
        if m.cfg.spice:
            d.update(spice=float(m.wp[i]), spice_metabolism=float(m.mp[i]))
        return d

    def prompt(self, m: Sugarscape, i: int) -> tuple[tuple[str, str], list[int]]:
        """((system, user), site list) for agent i; reply site number j means sites[j]."""
        sites, k = m.candidates(i)
        free = (m.occ[sites] < 0) | (sites == m.pos[i])
        sites, k = sites[free], k[free]
        spice = m.cfg.spice
        per = f" {self.persona(self.agent_dict(m, i))}" if self.persona else ""
        system = SYSTEM.format(goods="sugar and spice" if spice else "sugar",
                               goods_short="sugar or spice" if spice else "sugar", persona=per)
        a = self.agent_dict(m, i)
        hold = f"sugar {a['sugar']:g} (metabolism {a['metabolism']:g} per turn)"
        if spice:
            hold += f", spice {a['spice']:g} (metabolism {a['spice_metabolism']:g} per turn)"
        lines = [f"{j}: {m.off_name[kk]}, sugar {m.sugar[s]:g}" + (f", spice {m.spice[s]:g}" if spice else "")
                 for j, (s, kk) in enumerate(zip(sites, k))]
        user = f"Turn {m.step_no + 1}. You hold {hold}. Vision {a['vision']}.\nSites:\n" + "\n".join(lines) + \
            "\nWhich site do you move to?"
        return (system, user), sites.tolist()

    @staticmethod
    def parse(text: str, n_sites: int) -> int:
        try:
            obj = extract_json(text)
            j = obj["site"] if isinstance(obj, dict) else obj
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            j = text.strip()
        if isinstance(j, bool):
            raise ValueError("bad site")
        j = int(j)
        if not 0 <= j < n_sites:
            raise ValueError("site out of range")
        return j

    def plan(self, m: Sugarscape) -> dict:
        ids = np.flatnonzero(m.alive).tolist()
        if self.max_calls is not None and self.calls + len(ids) > self.max_calls:
            raise _Budget()
        built = [self.prompt(m, i) for i in ids]
        res = self.llm.map([p for p, _ in built], max_tokens=self.max_tokens, parallel=self.parallel,
                           **self.llm_kwargs)
        self.calls += len(ids)
        self.cost_usd += float(sum(getattr(r, "cost_usd", 0.0) or 0.0 for r in res))
        out = {}
        for i, (_, sites), r in zip(ids, built, res):
            try:
                out[i] = sites[self.parse(r.text, len(sites))]
            except (ValueError, TypeError):
                self.invalid += 1
        return out


def gini_trajectory(seeds=range(5), n_steps: int = 500, config: SugarscapeConfig | None = None,
                    at=(0, 25, 50, 100, 200, 300, 400, 500)) -> pd.DataFrame:
    """Gini at the given steps for several seeds: columns step, seed, gini, population."""
    out = []
    for s in seeds:
        st = Sugarscape(config, seed=s).run(n_steps).stats
        for t in at:
            if t <= n_steps and t < len(st):
                out.append(dict(step=t, seed=s, gini=st.gini.iloc[t], population=st.population.iloc[t]))
    return pd.DataFrame(out)


__all__ = ["Sugarscape", "SugarscapeConfig", "SugarscapeRun", "LLMPolicy", "sugar_capacity", "spice_capacity",
           "load_capacity", "gini", "lorenz", "gini_trajectory", "PEAKS", "RADII"]

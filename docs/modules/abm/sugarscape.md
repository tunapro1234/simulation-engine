# simengine.abm.sugarscape

Sugarscape from Epstein & Axtell (1996), *Growing Artificial Societies* (Brookings / MIT Press), chapters II
(sugar, wealth distribution) and IV (spice and trade). Agents with a vision and a metabolism forage on a landscape
with two sugar peaks. A skewed wealth distribution emerges from endowments that start close to uniform.

## What it does

| rule (book name) | implementation |
|---|---|
| G_alpha, growback | each site regrows `growback` units per step, up to its capacity (`np.inf` gives G_inf) |
| M, movement | agents act in a new random order each step. Each one looks `vision` sites along the four lattice directions (no diagonals), takes the unoccupied site with the most sugar (its own site counts, as in the NetLogo port), picks the nearest on ties and a random one among equals, harvests everything there, then burns `metabolism` |
| death, R_[a,b] | an agent dies when its sugar is <= 0, or when its age passes max_age ~ U{a..b}. With `replace=True` a random newborn takes its place at a random empty site. `max_age=None, replace=False` gives the carrying-capacity runs |
| spice (optional) | a second good on the mirrored map (peaks NW/SE). Movement maximises the welfare W = s^(m_s/m) p^(m_p/m) of holdings plus the site. An agent dies if either good runs out |
| T, trade (optional) | von Neumann neighbours bargain at the geometric mean of their MRS, where MRS = (p/m_p)/(s/m_s). Sugar flows to the agent with the higher MRS. They trade one unit (p > 1: p spice for 1 sugar, otherwise 1/p sugar for 1 spice) at a time, as long as both gain and their MRSs do not cross |
| LLM movement | `LLMPolicy(llm)` replaces rule M with one `llm.map` batch per step |

Defaults (`SugarscapeConfig`) follow the book's wealth-distribution run: a 50x50 torus, 250 agents,
vision U{1..6}, metabolism U{1..4}, endowment U{5..25}, max age U{60..100} and G1. Spice and trade are off by
default.

## API

```python
from simengine.abm.sugarscape import (Sugarscape, SugarscapeConfig, LLMPolicy, gini, lorenz, gini_trajectory,
                                      sugar_capacity, spice_capacity, load_capacity)

run = Sugarscape(SugarscapeConfig(), seed=0).run(500, snapshots=[0, 100, 500])
run.stats        # per step: step, population, mean_wealth, gini, skewness, top10_share, starved, aged, births,
                 # conflicts, mean_vision, mean_metabolism (+ trades, price_geo_mean, price_log_sd with trade)
x, y = lorenz(run.snapshots[500]); gini(run.snapshots[500])

cc = Sugarscape(SugarscapeConfig(n_agents=400, max_age=None, replace=False), seed=0).run(500)   # carrying capacity
tr = Sugarscape(SugarscapeConfig(spice=True, trade=True), seed=0).run(500)                      # rule T
own = Sugarscape(SugarscapeConfig(capacity=load_capacity(path)), seed=0)                        # another map file
gini_trajectory(seeds=range(10), n_steps=500)                                                   # long frame

# LLM agents (FakeLLM, simengine.llm.Claude, simengine.llm.ClaudeCode)
pol = LLMPolicy(llm, max_calls=5_000, persona=lambda a: "You are cautious." if a["vision"] < 3 else "")
run = Sugarscape(SugarscapeConfig(n_agents=20, size=25), seed=0, policy=pol).run(50)
run.calls, run.invalid, run.conflicts, run.stopped
```

- `Sugarscape.step(plan=None)` runs one step. `plan` is `{slot: flat site id}`. A planned site that an earlier mover
  has taken counts as a conflict, and that agent falls back to rule M.
- `Sugarscape.rule_m(i)` and `candidates(i)` expose the movement rule for one agent.
- `LLMPolicy`: each prompt lists the agent's holdings and metabolism and the free sites it can see at the start of
  the step (direction, distance, sugar, plus spice when on). The reply is `{"site": k}`. Replies that do not
  parse fall back to rule M and are counted in `invalid`. When the next step would push `calls` past `max_calls`,
  the run stops with `stopped="budget: ..."`.

## Provenance

- The rules come from Epstein & Axtell (1996). The book is not open access, so this port did not re-read it. The
  movement and death details were checked against the NetLogo port "Sugarscape 3 Wealth Distribution" (Wilensky
  2009, https://ccl.northwestern.edu/netlogo/models/Sugarscape3WealthDistribution; model code read in its source
  file):
  - candidate sites include the agent's own site
  - the nearest site wins among equal maxima
  - death when sugar <= 0 or age > max_age
  - initial population 400 in the NetLogo default, which is why `n_agents` is a parameter
  - endowment 5-25, metabolism 1-4, vision 1-6, max age 60-100, growback 1 per tick
- Landscape: the book's map, as transcribed in NetLogo's `sugar-map.txt`, is CC BY-NC-SA 3.0 and is **not shipped**.
  `sugar_capacity()` is parametric instead: stepped cones (levels 4/3/2/1 within radii 6/11/17/24) around peaks at
  (row 10.5, col 38.5) and (34.5, 14.5). The centres and radii were fitted to that file. Measured on the 2,500
  cells: 91.6% equal, none off by more than one level, total capacity 4,655 vs 4,622. Any grid file can still be
  read at runtime with `load_capacity`.
- Trade (rule T), welfare and MRS follow the definitions in chapter IV as summarised above. Each neighbouring pair
  trades once per step, in a random order and with a random initiator. The book lets each agent trade with each of
  its neighbours in turn.
- Differences from NetLogo/book:
  - deaths are processed after all agents have moved (NetLogo removes an agent inside its own turn)
  - newborns do not move in the step they are born
  - with spice, the "wealth" used for the Gini is sugar + spice holdings
  - spice metabolism U{1..4} and spice endowment U{5..25} are this port's choice, not checked against the book

## Measured numbers

To be filled in from the run below.

## Limitations

- Implemented: the chapter II/IV core, meaning G, M, R, spice and T. Not implemented: pollution, seasons, sex and
  inheritance, culture/combat, credit, disease.
- The movement loop is sequential Python. One 500-step run with 250 agents takes about 5 s, and the trade loop is
  slower.
- `LLMPolicy` plans every agent from the state at the start of the step. Agents move one at a time, so many plans
  point at a site an earlier mover has already taken. Those agents fall back to rule M (`conflicts`), which dilutes
  the LLM's share of the decisions. Cost is one call per agent per step. No real LLM run was made in this port, so
  there is no LLM result to quote.

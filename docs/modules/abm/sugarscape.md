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
  has taken counts as a conflict, and that agent falls back to rule M. A site outside the agent's vision raises
  `ValueError`. The capacity grid must be square (`ValueError` otherwise).
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
  - 50x50 world wrapping in both directions; its Gini plot is (2/n) Σ_i (i/n − L_i) over sorted wealth, which is
    algebraically the same as `gini()` here
  - these points were re-checked on 22 Oct 2026 against "Sugarscape 3 Wealth Distribution.nlogox" in the
    NetLogo/models GitHub repository (Sample Models/Social Science/Economics/Sugarscape)
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
  - trade runs once per step after deaths and replacement, so newborns can trade in their birth step (the book
    interleaves T with each agent's move)
  - with spice, the "wealth" used for the Gini is sugar + spice holdings
  - spice metabolism U{1..4} and spice endowment U{5..25} are this port's choice, not checked against the book

## Measured numbers

Command, run once on 22 Oct 2026 (offline, `nice -n 19`, 1,041 s wall on a shared machine):

```bash
python examples/sugarscape_measure.py <transcribed sugar-map grid>   # without the argument row A' is skipped
python examples/sugarscape_measure.py --quick      # smoke run, ~3 s: 25x25 map, 2 seeds, 60 steps; not these numbers
```

All runs below use 10 seeds unless noted. Each cell is mean (sd over seeds).

**A. Wealth distribution, rules ({G1}, {M, R[60,100]}), 250 agents, Gini of sugar holdings by step**

| map | t=0 | 10 | 25 | 50 | 100 | 150 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|
| parametric (default) | 0.236 (0.010) | 0.329 | 0.368 | 0.384 | 0.510 (0.013) | 0.452 | 0.478 | 0.462 | 0.478 | 0.483 (0.013) |
| transcribed book map, read at runtime | 0.236 (0.010) | 0.332 | 0.372 | 0.390 | 0.513 (0.018) | 0.460 | 0.481 | 0.472 | 0.483 | 0.477 (0.021) |

- Exact Gini of the U{5..25} endowment: 0.233.
- Steps 200-500 (parametric map): mean Gini 0.475 (0.006). At t = 500: wealth skewness 1.29 (0.11), against 0.03 at
  t = 0, and the top 10% hold 30.6% (1.0) of the sugar, against 16.5% at t = 0.
- The spike at t ≈ 100 is the first cohort reaching max age (60-100). Those deaths swap rich old agents for
  newborns holding 5-25 sugar. After that the Gini settles.
- The parametric map and the transcribed map agree within about one seed sd at every step shown.

**B. Carrying capacity, ({G1}, {M}), 400 agents, no ageing and no replacement.** The population falls from 400 to
237.3 (9.1) at t = 100 and to 230.2 (7.0) at t = 500, where the Gini is 0.376 (0.024). Selection shifts the
survivors: mean metabolism 2.47 → 1.79 and mean vision 3.51 → 3.87.

**C. Spice, with and without trade (rule T)**

| setting | no trade | trade |
|---|---|---|
| carrying capacity at t = 500 (400 agents, no ageing) | 132.1 (9.0) | 151.2 (6.4) |
| Gini of sugar + spice at t = 500, same runs | 0.338 (0.022) | 0.377 (0.025) |
| Gini of sugar + spice at t = 500, 250 agents, R[60,100] | 0.421 (0.014) | 0.449 (0.013) |
| sd of log trade price at t = 10 / 100 / 500 | n/a | 0.588 / 0.158 / 0.064 |
| geometric-mean price at t = 500 | n/a | 0.997 (0.032) |

**D. LLMPolicy plumbing with FakeLLM** (250 agents, 300 steps, 5 seeds, 75,000 calls per run, 0 invalid replies).
These are scripted responders, not an LLM. They test the plumbing and show which statistics can tell policies apart.

| policy | Gini t=100 | Gini t=300 | starvation deaths | conflicts |
|---|---|---|---|---|
| rule M (no LLM) | 0.509 | 0.469 | 640.8 | n/a |
| FakeLLM "greedy" (max sugar from the prompt list) | 0.503 (0.018) | 0.460 (0.009) | 609.2 (22.3) | 25,095 (33% of moves) |
| FakeLLM "random site" (hash-chosen visible site) | 0.479 (0.009) | 0.459 (0.009) | 1,487.2 (31.9) | 3,400 |

**Comparison with the literature**

- Book, qualitative: the wealth distribution turns strongly right-skewed from a roughly symmetric start, and the
  Gini rises and then fluctuates around a stable level. **Reproduced:** Gini 0.24 → about 0.47-0.48, skewness 0.03
  → 1.3. The NetLogo model page makes the same qualitative point: "Only a minority of the population have above
  average wealth", and "there are many more agents with low wealth than agents with high wealth".
- Book Gini value: not checked. The book is not open access. From memory, the book's Gini series levels off near
  0.5. Treat that as unverified.
- Secondary replication: Foguelman et al. 2021 (EB-DEVS), arXiv:2110.08170, §3.4.3 (re-read 22 Oct 2026): with the
  Gini threshold at 1 (plain Sugarscape) "the Gini index starts around 0.22, and it oscillates through time until it
  reaches a stable value of 0.42". Their text says they use the book's two-peak terrain; their Listing 5 draws vision
  1-6 and endowment 5-25 like ours, but metabolism 1-2 (ours 1-4), max age 5-25 (ours 60-100) and positions on
  0..30. The agent count is not stated in the text. Our start (0.236) matches theirs. Our plateau (0.47-0.48) is
  higher; with metabolism and lifespan this different the plateaus are not directly comparable, and they note
  that the final value "is heavily dependant on the grid topology".
- Carrying capacity and trade: the book reports, from memory and unverified, that a carrying capacity emerges, that
  selection lowers metabolism and raises vision, that trade raises carrying capacity, that prices converge with
  falling dispersion, and that trade raises inequality. All five appear in B and C with the same sign. The book's
  numbers were not compared.
- What D shows: under random movement the Gini plateau is almost the same, 0.459 against 0.469. The plateau comes
  mostly from heterogeneous metabolism, vision and age under replacement, not from greedy foraging. What separates
  the policies is starvation: 2.3x more deaths under random movement. An LLM-policy study should report survival
  and carrying capacity along with the Gini.

## Limitations

- Implemented: the chapter II/IV core, meaning G, M, R, spice and T. Not implemented: pollution, seasons, sex and
  inheritance, culture/combat, credit, disease.
- The movement loop is sequential Python. One 500-step run with 250 agents takes about 5 s, and the trade loop is
  slower.
- `LLMPolicy` plans every agent from the state at the start of the step. Agents move one at a time, so many plans
  point at a site an earlier mover has already taken. Those agents fall back to rule M (`conflicts`), which dilutes
  the LLM's share of the decisions. Cost is one call per agent per step. No real LLM run was made in this port, so
  there is no LLM result to quote.

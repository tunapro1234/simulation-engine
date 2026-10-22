# simengine.abm.schelling: Schelling segregation, Zhang swap game, LLM movers

`simengine/abm/schelling.py` runs Schelling's (1971) checkerboard model on the grid core (`simengine.abm.grid`).
It has three dynamics:

- the classic threshold rule;
- Zhang's (2004) swap game, in which every agent prefers a mixed neighbourhood;
- an LLM-agent variant: each decision is a prompt that describes the agent's neighbourhood, prompts are batched
  through `llm.map`, replies are parsed, and an optional persona text goes into the system prompt.

Import it directly (`import simengine.abm.schelling as S`) until the integrator registers it in `simengine.abm`.

## API

| name | what it does |
|---|---|
| `Schelling(grid, fractions=(.45, .45), threshold=1/3, rule="random", movers="unhappy", isolated_happy=True, mover=None, block=5)` | Threshold model. An agent is happy when like neighbours / occupied neighbours >= threshold. The threshold is a scalar or one value per type. An agent with no occupied neighbour counts as happy by default. One step is an asynchronous sweep in a fresh random order. `movers="unhappy"`: an agent that is unhappy when its turn comes moves. `movers="all"`: every agent tries to move (the Gauvin et al. 2009 variant). `rule` sets the destination: `"random"` is any empty cell (Mesa / NetLogo style). `"satisfying"` is a random empty cell where the agent would be happy; it stays if there is none. `"nearest"` is the closest such cell by squares traversed (Manhattan; Schelling's rule). |
| `Schelling.run(n_steps, seed, lattice=None, stop_when_still=True, metrics=True)` -> `SchellingRun` | `lattice` (final), `initial`, `history` (one row per step: `same_share`, `happy`, `unlike_pairs`, `no_unlike`, `moved`, `failed`, plus `seg_coef` / `dissimilarity` on the last row), `steps`, `converged` and `llm` stats. |
| `tipping_curve(thresholds, seeds, grid, n_steps, **kw)` | DataFrame with the final metrics for each (threshold, seed). |
| `unlike_pairs(lattice, grid)` | Zhang's rho: the number of neighbouring pairs of different types. |
| `segregation_coefficient(lattice, torus=False)` | Gauvin, Vannimenus & Nadal (2009) eq. 2: `s = 2/N^2 sum_c n_c^2` over 4-connected same-type clusters. It is 1 for two one-type clusters and near 0 for a fine mix. On a torus, clusters that touch across the wrap are merged. |
| `ZhangSwap(grid, beta=10, Z=1, M=0.6, share=0.5).run(n_pairs, seed, record_every, target_rho)` -> `ZhangRun` | Zhang's game on a full torus (no vacancies). Each period draws one random type-0 agent and one random type-1 agent. They swap with probability `exp(bV)/(exp(bU)+exp(bV))`. The payoff is kinked: `u = Z x/n` for `x <= n`, `(2Z-M) + (M-Z) x/n` above, with `n = k/2`. rho and the like counts are updated incrementally; a test checks them against a full recount. |
| `zhang_payoff(x, n, Z, M)` | Zhang's eq. 1 (Table 4 values with Z=1, M=0.6). |
| `LLMMover(llm, labels=None, persona=None, ask="unhappy", n_options=0, dedupe=True, max_calls=None, max_tokens=60, parallel=8, system=DEFAULT_SYSTEM, llm_kwargs=None)` | LLM decision round, passed as `Schelling(..., mover=...)`. See below. |
| `parse_decision(text, n_options)` -> `Decision(move, option, valid)` | Accepts JSON `{"move": ...}`, `{"choice": "stay"/n}`, Cheng et al.-style `{"yes_rating", "no_rating"}`, or plain yes / no / stay / "option n". Anything else means stay with `valid=False`. |
| `threshold_responder(threshold)` | `FakeLLM` responder that applies the classic rule to the prompt text. It is for tests and for sanity runs of the LLM path. |
| `describe_cells(counts, own, labels, k)` | Builds the neighbourhood sentence used in prompts. |

```python
import simengine.abm.grid as G
import simengine.abm.schelling as S
from simengine.llm.fake import FakeLLM          # real runs: simengine.llm.Claude(...)

g = G.Grid((50, 50), torus=True)
run = S.Schelling(g, threshold=1 / 3).run(n_steps=300, seed=0)
run.history.iloc[-1][["same_share", "happy", "seg_coef"]]
curve = S.tipping_curve([0.2, 1 / 3, 0.5, 0.75, 0.8], seeds=range(10), grid=g)

mover = S.LLMMover(FakeLLM(S.threshold_responder(0.5)), labels=("blue", "orange"),
                   persona={0: "retired teacher, likes quiet streets", 1: "nurse, works nights"})
run = S.Schelling(g, threshold=0.5, mover=mover).run(n_steps=50, seed=0)
run.llm     # {'calls', 'asked', 'invalid', 'blocked', 'cost_usd', 'invalid_rate'}
```

### LLM round

1. Choose who is asked. `ask="unhappy"` asks only the agents that are unhappy under the model threshold. `ask="all"`
   asks everyone, so the LLM replaces the threshold, as in Cheng et al. 2024.
2. Build each agent's prompt from the state at the start of the round. The user prompt reads: "You are one of the
   <label> households. Of the 8 homes around yours, 3 are <label> households (your group), 4 are <other> households,
   1 is empty." It ends with a JSON move / stay question. With `n_options > 0`, up to n random vacancies are
   described the same way (with the agent's own home counted as vacated), and the agent picks one or stays.
   The persona goes into the system prompt as "About you: ...". It can be one string for everyone, `{type: text}`,
   one text per agent id, or `callable(agent_id, type)`.
3. Send identical (system, user) pairs once (`dedupe`) in a single `llm.map`. Without personas or options, a
   2-type Moore grid has at most ~90 distinct neighbourhood prompts. In practice this is a policy table that a disk
   cache reuses across rounds.
4. Parse the replies. Movers then relocate in random order. A vacancy that someone else took earlier in the round
   counts as `blocked` and the agent stays. A move without options goes to the model's `rule` destination.
5. A round in which nobody was asked, or (without options) nobody chose to move, ends the run as converged. With
   options, a still round does not stop the run, because the offered vacancies are drawn again each round.

`max_calls` stops the run cleanly before a round that would exceed it, and the round is not sent.

## Provenance

Each method was checked against the text of its source (sections named below). No code was taken from another project.

- **Schelling (1971)**, "Dynamic models of segregation", *J. Math. Sociology* 1:143-186. Checked in the scanned paper,
  pp. 154-158. Setup: 13 x 16 board with edges (5 neighbours on an edge, 3 in a corner), about 30% blank, equal
  numbers of the two colours, a demand for a minimum share of like neighbours among the eight surrounding squares,
  and the discontented moving to the "nearest satisfactory square, with 'nearest' measured by the number of squares
  one traverses horizontally and vertically". The board was played by hand, and the order of moves varied.
  Our setting: `Grid((13, 16), torus=False)`, `fractions=(.33, .33)`, `rule="nearest"`.
- **Gauvin, Vannimenus & Nadal (2009)**, "Phase diagram of a Schelling segregation model", *EPJ B* 70:293
  (arXiv:0903.4694), §1-2 and App. A.4.6. Tolerance T: satisfied iff unlike neighbours <= T x occupied neighbours,
  so it matches `threshold = 1 - T`. Randomly chosen agents (satisfied ones too) move to a random vacancy that
  satisfies them, or else stay. The grid is L = 50 with free boundaries. s is eq. 2. Our setting:
  `movers="all", rule="satisfying"`.
- **Zhang (2004)**, "Residential segregation in an all-integrationist world", *JEBO* 54:533-550, §2-3 (author's
  PDF). Our setting follows eqs. 1-2, Table 4 (Z = 1, M = 0.6), the logit switch probability, a torus and no
  vacancies. One detail is our interpretation: Zhang picks "a pair of agents ... from two different
  neighbourhoods", and we draw one random agent of each colour (a same-colour swap changes nothing).
- **NetLogo Segregation model** (Wilensky; model library page, "Things to notice"). It says that a 30% wanted share
  ends up "on average" at 70% same-colour neighbours. Its unhappy agents move to a nearby patch. We move them to a
  random empty cell.
- **LLM Schelling.** Cheng et al. 2024, "Observing micromotives and macrobehavior of large language models"
  (arXiv:2412.10428, IJCNLP-AACL 2025), as summarised from the arXiv HTML. 20 x 20 grid, 180 + 180 agents, Moore
  neighbourhood. The prompt gives the agent's group label and its number of same-type neighbours out of 8, and asks
  for 0-10 ratings of "move" and "stay". They report a "Segregation Shift" of about 0.27-0.28 averaged per model,
  across 5 models and 5 demographic pairs. Our `ask="all"` plus the `yes_rating` / `no_rating` parsing covers that
  design; our default prompt asks for a direct move / stay instead. Blando et al. 2026 (arXiv:2607.17948) use the
  LLM only to *classify* neighbours for one agent under Schelling's rule. The abstract gives no numbers.

## Measured numbers

All runs are this port's, on synthetic random initial lattices. There is no data set. Scripts and CSVs are outside
the repo. Command: `PYTHONPATH=. nice -n 19 python olc.py <experiment>`; each experiment is a few lines that call the
API below. The runs were made on 22 Oct 2026 on a shared machine.

### 1. Tipping curve (random move, unhappy movers)

`tipping_curve(thr, seeds=range(10), grid=Grid((50, 50), torus=True), n_steps=300)` with 10% empty cells. Means over
10 seeds:

| threshold | like share (start 0.501) | happy | no unlike nbr | seg. coef. s | Duncan D (5x5) | steps | converged |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.501 | 1.00 | 0.008 | 0.020 | 0.168 | 1 | 10/10 |
| 0.10 | 0.510 | 1.00 | 0.014 | 0.022 | 0.175 | 2.2 | 10/10 |
| 0.20 | 0.571 | 1.00 | 0.060 | 0.040 | 0.222 | 7.3 | 10/10 |
| 0.25 | 0.574 | 1.00 | 0.064 | 0.039 | 0.224 | 6.4 | 10/10 |
| 1/3 | 0.751 | 1.00 | 0.286 | 0.362 | 0.400 | 13.2 | 10/10 |
| 0.40 | 0.827 | 1.00 | 0.439 | 0.646 | 0.501 | 14.5 | 10/10 |
| 0.50 | 0.874 | 1.00 | 0.566 | 0.803 | 0.583 | 17.6 | 10/10 |
| 0.60 | 0.971 | 1.00 | 0.869 | 0.930 | 0.780 | 74.0 | 10/10 |
| 0.65 | 0.979 | 1.00 | 0.899 | 0.929 | 0.812 | 72.6 | 10/10 |
| 0.70 | 0.993 | 1.00 | 0.962 | 0.948 | 0.859 | 156.5 | 10/10 |
| 0.75 | 0.995 | 1.00 | 0.977 | 0.995 | 0.880 | 132.3 | 10/10 |
| 0.80 | 0.553 | 0.146 | 0.036 | 0.033 | 0.221 | 300 | 0/10 |
| 0.90 | 0.501 | 0.009 | 0.009 | 0.020 | 0.167 | 300 | 0/10 |

This is the known tipping shape. Below about 1/4 there is almost no segregation. A mild demand of 1/3 gives a 0.75
like share, and 0.6-0.75 gives near-complete separation. Above 0.75 nobody can be satisfied, agents keep
reshuffling at random, and the lattice stays mixed. The curve moves in steps because k = 8: 0.20 and 0.25 both mean
"2 of 8" for a full neighbourhood.

- **NetLogo comparison** (`netlogo`, 51 x 51 torus, 30% wanted, 20 seeds): like share 0.754 (sd 0.013) at 5%
  empty, 0.748 (sd 0.009) at 10% empty. NetLogo's page says "on average 70%" with nearby-patch moves. We are a few
  points higher with random moves.

### 2. Schelling's own board (`schelling71`)

13 x 16 bounded board, 69 + 69 agents (34% blank), nearest satisfactory square, 200 seeds:

| demand | like share start -> end | no unlike neighbour start -> end | all happy |
|---|---|---|---|
| 1/3 | 0.494 -> 0.652 (sd 0.026) | 0.059 -> 0.191 | 200/200 |
| 3/8 | 0.494 -> 0.718 (sd 0.025) | 0.059 -> 0.264 | 200/200 |
| 0.40 | 0.494 -> 0.724 (sd 0.026) | 0.059 -> 0.272 | 200/200 |
| 1/2 | 0.494 -> 0.789 (sd 0.020) | 0.059 -> 0.378 | 200/200 |

Schelling's reported values (one hand-played board, pp. 156-158) at demand 1/2: like share 53% (zeros) and 46%
(stars) at the start, and five-sixths (0.83) and four-fifths (0.80) at equilibrium. Agents with no neighbour of
the opposite colour went from 13 (about 10%) to 54 (about 40%). At about 1/3 demand he calls segregation "slight",
and he says segregation is "a rapidly rising function of demands in the range from about 35% to 50%". Our mean at
1/2 (0.789, 0.378) is within about one sd of his single board. Our start value for "no unlike neighbour" (0.059)
is below his 13/138. His count comes from one random draw, and he quotes it against an 11 x 16 expectation.

### 3. Gauvin et al. variant (`gauvin`)

L = 50 with free boundaries, 10% vacancies, every agent tries a satisfying vacancy, 150 sweeps, 2 seeds per T:

| T (tolerance) | 1/4 | 1/3 | 3/8 | 3/7 | 1/2 | 4/7 | 5/8 | 2/3 | 5/7 | 3/4 | 4/5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s | 0.018 | 0.029 | 0.764 | 0.819 | 0.848 | 0.727 | 0.712 | 0.814 | 0.813 | 0.086 | 0.046 |
| like share | 0.524 | 0.560 | 0.984 | 0.975 | 0.973 | 0.963 | 0.952 | 0.942 | 0.923 | 0.633 | 0.583 |
| happy | 0.186 | 0.391 | 0.989 | 0.991 | 0.996 | 0.993 | 0.994 | 0.994 | 0.990 | 0.962 | 0.965 |

Their Table 1 gives the frozen-to-segregated jump Tf = 3/8 at a vacancy density of 10%. We see it between 1/3 and
3/8, so it matches. Their segregated-to-mixed jump Tc is "almost at a constant value" of T. Their Figs. 3 and A.7
put it near 2/3-3/4 (read off the plots; the text gives no number). We see it between 5/7 and 3/4. Our s
is 0.7-0.85 where theirs is ~1, because we stop at 150 sweeps while they equilibrate and then average over 30,000
steps. Our clusters are still coarsening.

### 4. Zhang swap game (`zhang`, `zhangwait`)

100 x 100 Moore torus, 50/50, Z = 1, M = 0.6, random initial state (rho about 19,900 unlike pairs; the two-stripe
minimum is 3 x 100 x 2 = 600). `zhangwait`: 4M pairs, rho recorded every 20k pairs, 3 seeds (2 for beta 5-8).
"Scaled" is Zhang's potential with his constant M/(2n) applied, rho x 0.075; Zhang's criterion is |potential| < 600,
which is rho <= 8,000.

| beta | rho after 4M pairs | like share | pairs until scaled potential <= 600 |
|---|---|---|---|
| 1 | 19,022-19,152 | 0.52 | never (3/3) |
| 2 | 17,722-18,024 | 0.55 | never (3/3) |
| 3 | 15,832-16,414 | 0.60 | never (3/3) |
| 4 | 13,550-14,038 | 0.65 | never (3/3); a 60M-pair run stays at 13,660-14,258 |
| 5 | 8,528-9,436 | 0.76-0.79 | never (2/2) |
| 6 | 4,358-4,734 | 0.88-0.89 | 160k (2/2) |
| 7 | 3,030-3,278 | 0.92 | 80k (2/2) |
| 8 | 2,188-2,464 | 0.94 | 60k (2/2) |
| 10 | 2,078-2,158 | 0.95 | 60k (3/3) |
| 40 | 1,894-2,284 | 0.95 | 40k (3/3) |

- Qualitatively this agrees with Zhang. With large beta, rho collapses and the like share reaches 0.95, although
  every agent's payoff peaks at a 50-50 neighbourhood. The waiting time falls steeply with beta and then flattens
  (60k pairs at beta 8-10, 40k at 40).
- **Disagreement:** Zhang (§3.3, Fig. 8) reports that for beta < 2 runs never reach the criterion, that the waiting
  time drops sharply around beta = 2, and that runs at beta = 4 "always end fairly quickly". In our runs the
  threshold is between beta 5 and 6, and beta = 4 stays mixed at stationarity: 60M pairs, rho ~14k. This fits a
  check from equilibrium theory. The logit rule on a potential game has the Gibbs stationary law
  exp(-beta x 0.075 x rho) (Zhang's Prop. 1 / Young 1998). That is an Ising model with coupling K = 0.0375 beta on
  the 8-neighbour lattice, which orders only above some K_c, so a finite threshold beta is expected. Its exact
  value was not computed here. Either Zhang's potential or time unit is scaled differently from our reading of
  eqs. 1-2, or the pair rule matters. We could not settle this from the paper.
- A raw rho of 600 (two clean stripes) was not reached in 60M pairs at beta = 10. rho levelled off at ~1,200, the
  count for four straight boundaries. A metastable multi-stripe state on the torus is the likely cause; the
  lattice was not inspected.
- Speed: ~12.6 s per 1M pairs on 100 x 100 (pure Python loop, measured under load).

### 5. LLM path with a rule-following FakeLLM (`llm`)

30 x 30 torus, 10% empty, 5 seeds, at most 300 rounds, `FakeLLM(threshold_responder(thr))`. This checks the
mechanics of batched synchronous decisions against the asynchronous sweep. It is **not** a measurement of any real
LLM.

| threshold | classic like share (sweeps) | LLM, no options (rounds) | calls / asked | LLM, 3 options (rounds) | blocked |
|---|---|---|---|---|---|
| 0.3 | 0.753 (12.6) | 0.749 (11.0) | 99 / 349 | 0.747 (9.0) | 113 |
| 0.5 | 0.863 (15.0) | 0.869 (14.4) | 182 / 871 | 0.853 (13.2) | 487 |
| 0.7 | 0.991 (185.6) | 0.993 (180.6) | 2,772 / 11,155 | 0.550 (300, not converged) | 548 |

- Without options, the batched round reproduces the classic outcome. De-duplication cuts calls 3.5-4.8x, and more
  with a disk cache across rounds.
- With 3 random vacancies on offer, a demanding agent (0.7) rarely finds a satisfying one. The lattice then stays
  mixed (0.55 after 300 rounds and ~161k calls). Limited search alone changes the macro outcome, so any LLM result
  with options has to be compared with this control, not with the classic curve.
- No real-LLM numbers were measured in this port. The LLM literature value above (Cheng et al. shift ~0.27) is
  theirs.

## Limitations

- The classic sweep is a Python loop with incremental counts: ~0.06 s per sweep on 50 x 50 with random moves, and
  ~0.28 s with `movers="all", rule="satisfying"` (measured under heavy machine load). It is fine up to ~100 x 100,
  not for very large lattices.
- `rule="nearest"` breaks ties at random. Schelling broke them by hand order. `isolated_happy=True` is a
  convention, because Schelling does not say how he treated an agent with no neighbours.
- Zhang's run draws one random agent of each colour per period. His "two different neighbourhoods" pairing rule is
  not specified further, so waiting times measured in pairs are not directly comparable with his Fig. 8 (which is
  a plot with no table).
- The LLM round is synchronous, while the classic sweep is asynchronous. With options, collisions (`blocked`) are
  frequent when many agents are unhappy. Identical prompts share one reply in a round (`dedupe=True`), which
  removes sampling variation; pass `dedupe=False` to sample each agent.
- Personas reach the LLM only as text. Nothing here checks that a persona changes behaviour in the intended
  direction; that has to be measured with a real model and a placebo (shuffled personas).
- Segregation numbers depend on the measure: like share, s (cluster based, sensitive to vacancies above ~20%,
  Gauvin §2.1) and Duncan D (depends on the tile size).

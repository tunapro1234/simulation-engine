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

See the next revision of this file.

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

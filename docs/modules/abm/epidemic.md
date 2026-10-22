# simengine.abm.spatial_sir: spatial SIR / SEIR, mean-field ODE control, R0, generative-agent epidemic

`simengine/abm/spatial_sir.py` has three parts:

- a stochastic SIR / SEIR engine on a 2-D grid (`SpatialSEIR`), where agents meet either in a neighbourhood around
  a static home cell or by being near each other on mobility trajectories (for example YJMob100K);
- the controls for that engine: the mean-field ODE (`seir_ode`), the classical final-size relation, and three
  R0 estimators (index case, early growth rate, final size);
- an LLM-agent variant (`GenerativeEpidemic`) built after Williams et al. 2023. Every agent is asked each day whether
  to stay home, given its symptoms, the town's case numbers and an optional public message. Agents who go out mix
  and can infect each other.

Import it directly until the integrator registers it in `simengine.abm`: `import simengine.abm.spatial_sir as ep`.

## API

| name | what it does |
|---|---|
| `EpiParams(beta, gamma, sigma=None)`, `EpiParams.from_r0(r0, infectious_days, latent_days=None)` | Rates per day. `sigma=None` gives SIR. `r0 = beta / gamma` (frequency-dependent mixing). |
| `seir_ode(params, n, i0, days, e0=0, dt=0.02)` | Mean-field SIR / SEIR, RK4. One row per day: `day, S, E, I, R, new`. `new[0]` holds the seeds, the same convention as `EpiRun`. |
| `final_size(r0, s0=1)` / `r0_final_size(z)` | `z = s0 (1 - exp(-R0 z))` solved by Newton, and its inverse `-ln(1 - z) / z`. |
| `r0_growth(new, gamma, sigma=None, n=None, start_cases=10, stop_share=0.05, window=None)` | Log-linear fit of daily incidence over the early window, then `R0 = (1 + r/gamma)(1 + r/sigma)`. The window runs from the first day with `>= start_cases` new cases to the day before cumulative cases pass `stop_share * n`. Returns `{r0, r, window, n_days}`. |
| `SpatialSEIR(params, shape=(100, 100), positions=None, n=None, radius=1, steps_per_day=1, torus=True, seed=0)` | Engine. `positions`: `None` (n uniform random home cells), `(n,)` static flat cells, or `(n, T)` trajectories (step `t` uses column `t % T`). `radius`: Moore window around the agent's cell, own cell included. `0` means same cell only and `None` means well mixed. |
| `SpatialSEIR.run(days, i0=10, reps=1, seed=None, stop_when_extinct=True, absorb=False, present=None)` -> `EpiRun` | Runs `reps` replicates at once as `(B, n)` arrays. `i0` is a count or explicit agent indices. `present` is a mask of agents taken out of all contact. `absorb=True` makes new infections terminal (used for index-case R0). |
| `SpatialSEIR.local_share(infectious, present, step)` | `(B, n)` share of infectious agents among those present in each agent's window. It uses `bincount` per cell plus an integral-image box sum, so cost does not grow with the radius. |
| `EpiRun` | `S, E, I, R, new` arrays of shape `(B, days + 1)`, plus `infected_day (B, n)`. Methods: `mean()` (same columns as `seir_ode`), `frame()` (long table), `attack_rate()`, `summary(minor=0.01)`. `summary` reports attack rate, peak prevalence and peak day; the `*_major` fields leave out fade-outs. |
| `ode_agreement(run, ode, minor=0.01)` | ABM vs ODE: attack rate, peak share, peak day, and RMSE of the mean prevalence curve (`I/n`), counting major outbreaks only. |
| `r0_index_case(model, reps=400, max_days=200, seed=0)` | One infectious agent in a fully susceptible population, with its secondary cases made absorbing. R0 = mean number of secondary cases, with a normal 95% CI. This gives the R0 of that contact structure as realised in the simulation. |
| `box_sum(a, radius, torus=True)`, `cells_from_xy(x, y, shape, origin=0)`, `random_positions(n, shape, seed)` | Grid helpers. |
| `trajectories_from_records(df, shape, days, slots_per_day=48, ...)` | Long `(uid, d, t, x, y)` records -> dense `(n_users, days * slots)` trajectories. Missing slots are forward filled. |
| `load_yjmob100k(data_root=None, n_users=2000, days=14)` | First `n_users` users of YJMob100K dataset 1 as trajectories for a `(200, 200)` grid at 48 slots per day. Reads from `data_root` or `$SIMENGINE_DATA`, in either `yjmob100k/` below it or the directory itself. |
| `GenerativeEpidemic(llm, agents=None, n=100, contact_rate=5, infectivity=0.1, infectious_days=6, condition="full", message=None, i0=3, contact="random", ..., retries=0, max_calls=None, seed=0)` | LLM-agent epidemic, one `llm.map` call per day. `condition` is `base`, `self_health` or `full`. `message` is a string, or `callable(day, news_pct)`, added to every agent's memories. `contact="grid"` makes agents who go out mix in a Moore window instead of meeting random partners. `.run(days=60)` -> `GenerativeRun`. |
| `GenerativeRun` | `daily` (`day, S, I, R, new_cases, day4_cases, news_pct, out_share, invalid`), `log` (one row per decision: health, news, `stay_home`, `valid`, reasoning, `prompt_id` hash), `calls`, `invalid`, `stopped` (`cleared`, `days` or `budget`). `stay_home_by_health()` gives the share staying home per symptom. |
| `epi_personas(n, seed)`, `isolation_prompt(agent, health, news_pct, message)`, `parse_isolation(text)` | Paper-style personas (age 18-64, each Big Five trait positive or negative with p = 0.5), the daily `(system, user)` prompt, and the reply parser. |
| `rule_responder(stay_on=("cough", "fever"), news_threshold=None, message_keyword=None)` | `FakeLLM` responder with a fixed rule. It answers in the paper's reply format and serves as an offline control. |

```python
import simengine.abm.spatial_sir as ep

p = ep.EpiParams.from_r0(2.5, infectious_days=5, latent_days=3)
ode = ep.seir_ode(p, n=10_000, i0=10, days=200)
mixed = ep.SpatialSEIR(p, shape=(100, 100), n=10_000, radius=None, steps_per_day=4)
ep.ode_agreement(mixed.run(days=200, i0=10, reps=30, seed=1), ode)      # well-mixed run vs the ODE
local = ep.SpatialSEIR(p, shape=(100, 100), n=10_000, radius=2, steps_per_day=4)
local.run(days=400, i0=10, reps=10, seed=1).summary()                 # slower, flatter, smaller epidemic
ep.r0_index_case(local, reps=1000)                                     # realised R0 of the radius-2 contacts

traj, _ = ep.load_yjmob100k(n_users=5000, days=75)                     # $SIMENGINE_DATA
mob = ep.SpatialSEIR(p, shape=(200, 200), positions=traj, radius=1, steps_per_day=48)

from simengine.llm.fake import FakeLLM                                 # real runs: simengine.llm.Claude(...)
g = ep.GenerativeEpidemic(FakeLLM(ep.rule_responder(news_threshold=1.0)), n=100, condition="full",
                          message="The mayor asks everyone with a cough to stay home.")
res = g.run(days=120)
res.daily[["new_cases", "out_share"]], res.stay_home_by_health()
```

## Model details

**Transmission on the grid.** A susceptible agent in cell `c` has hazard `beta * I_box(c) / N_box(c)` per day.
`I_box` and `N_box` count the infectious and present agents in the `(2r+1) x (2r+1)` window around `c`. Mixing is
therefore frequency dependent, as in the ODE, and `r = None` is the mean-field case. Within a step of length
`dt = 1 / steps_per_day`, infection has probability `1 - exp(-hazard dt)`. E -> I and I -> R have probability
`rate * dt`, a geometric residence time whose mean is exactly `1 / rate`. The obvious choice `1 - exp(-rate dt)`
lengthens the mean stay by about `dt / 2`. At one step per day that inflates R0 from 2.5 to about 2.76 (found and
fixed while testing; commit `49f14bc`). Steps need `rate * dt <= 1`, otherwise `run` raises an error.

**Mobility.** With `(n, T)` trajectories each sub-step is one trajectory slot, and the window is recomputed from
where everyone is at that slot. YJMob100K has 48 slots per day, so `steps_per_day=48` makes one simulated day one
data day. After `T` slots the trajectory wraps around. Contact means sharing a 500 m cell (`radius=0`) or being
within `r` cells of each other, and is weighted by presence time. Nothing in the data says who actually met whom.

**R0 estimators.** `r0_index_case` measures what the contact structure delivers. `r0_growth` is what an analyst
would infer from an incidence curve (Wallinga & Lipsitch 2007 for exponential periods). `r0_final_size` inverts the
final-size relation, which assumes homogeneous mixing. On a local grid the three disagree on purpose; the
disagreement is the spatial effect (see the measured numbers).

**Generative agents.** This follows Williams, Hosseinichimeh, Majumdar & Ghaffarzadegan 2023, "Epidemic modeling
with generative agents" (arXiv:2307.04986). The paper's methods section was checked for this port:

- Persona: name, age uniform 18-64, and each Big Five trait positive or negative with p = 0.5. The bio is a town
  resident with an office job. The memory "goes to work to earn money to support [name]'s self" is used in every
  condition.
- Memories by condition. `base` has nothing else. `self_health` adds "feels normal" / "has a light cough" / "has a
  fever and a cough". `full` adds the virus paragraph and "[X]% of Dewberry Hollow's population caught new infections
  of the Catasat virus yesterday", where X is the share of agents on day 4 of infection. These are the paper's
  strings. The town and virus names are the paper's pseudonyms, chosen to avoid prior knowledge.
- Disease: 6 days. Days 1-2 are normal but infectious, days 3 and 6 a light cough, days 4-5 fever and cough. After
  that the agent is immune. The defaults are the paper's first experiment: n = 100, contact rate 5, infectivity
  0.1, nominal R0 = 5 x 0.1 x 6 = 3.
- Daily loop: ask everyone, then each infectious agent who went out meets `contact_rate` distinct agents among
  those also out, then health states update. A run stops after two consecutive days with nobody infected. The
  question asks whether the agent should stay at home for the entire day, with "Reasoning: ... / Response: Yes|No";
  a non-conforming reply counts as going out.
- Additions that are not in the paper: `message=` (a line injected into every agent's memories, so the effect of a
  public message can be simulated), `contact="grid"`, `max_calls`, `retries`, and the prompt hash in the log.
  The paper's text does not give the number of initially infected agents (its code lists `initial_infected` as a
  static parameter), so `i0=3` is our choice.
- The paper used `gpt-3.5-turbo-0301`. Its results are figures (Fig. 2-3) plus a logit table (Table 1). From
  Table 1, regression 1, as printed in the paper: light cough 5.69 and fever & cough 5.42 on the log-odds of staying
  home vs feeling normal. Those are the paper's numbers, not ours.

## Provenance

- Grid engine, ODE, final size and R0 estimators: standard epidemic theory (Kermack & McKendrick 1927 final-size
  relation; Wallinga & Lipsitch 2007, Proc. R. Soc. B 274:599, growth-rate to R0 for exponential periods). The code
  was written for simengine and is not a port.
- Generative-agent variant: designed after Williams et al. 2023 (arXiv:2307.04986). Prompts follow their Fig. S2 /
  methods text. No code was copied.
- Mobility data: YJMob100K dataset 1 (Yabe et al. 2024, Sci Data; CC BY 4.0), read at run time. No rows are in the
  repo. Tests use synthetic trajectories; the one loader test is skipped when `$SIMENGINE_DATA` is unset.

## Limitations

- The engine is synchronous and discrete-time, with geometric residence times. Exponential periods are a modelling
  choice; real latent and infectious periods are more peaked, which changes the growth-rate-to-R0 mapping.
- Memory is `O(B * H * W)` per step for local mixing. Mobility runs on 5000 users x 48 slots per day are minutes,
  not seconds (see timings below).
- Co-location on a 500 m grid is not contact. On mobility data `beta` scales a presence-weighted local share, so
  R0 values on trajectories describe this model, not a disease.
- Trajectories wrap after `T` slots (75 days for YJMob100K), so very long runs repeat the same weeks.
- `GenerativeEpidemic` makes `n` LLM calls per simulated day: 100 agents x 60 days = 6000 calls. Use
  `max_calls`, and a disk-cached LLM for reruns. Only FakeLLM runs were measured here; whether a current model
  reproduces the paper's behaviour (symptom-driven isolation, a negative-exponential response to case news) has not
  been tested.
- The random contact rule is an interpretation. The paper's Table S1/S4 says each agent who goes out "interacts
  with up to X other agents" and that partner lists are "a two-way process", without giving the algorithm. Here
  each infectious agent who goes out draws `contact_rate` distinct partners among those out. That keeps the
  paper's stated R0 = contact rate x infectivity x duration = 3, but a susceptible agent can be met more than
  `contact_rate` times a day.

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

## Measured numbers

All runs are this port's, made on 23 Oct 2026 on a shared 32-core machine (timings include load from other jobs).
The well-mixed and grid runs use synthetic uniform random home cells; the mobility runs use YJMob100K read from a
local copy. Scripts are outside the repo; each is a few lines over the API above. Command:
`PYTHONPATH=. nice -n 19 python measure.py {mixed|grid|mobility|llm}` plus `python growth.py` for the R0 window
check; the mobility part passes the local data root to `load_yjmob100k`. Seeds: home cells `seed=0`, `run(...,
seed=1)`, `r0_index_case(..., seed=0)`. A rerun reproduced the earlier development runs to the last printed digit.

### 1. Well-mixed ABM vs the ODE (`radius=None`)

n = 10 000 agents on a 100 x 100 grid, i0 = 10, 200 days, 30 replicates (`seed=1`), every replicate a major
outbreak. SEIR: R0 = 2.5, 5 infectious days, 3 latent days. SIR: R0 = 2, 4 infectious days. `prevalence RMSE` is
the RMSE of the replicate-mean `I/n` curve against the ODE's `I/n` over the 200 days.

| model | steps/day | attack ABM | attack ODE | final-size formula | peak share ABM / ODE | peak day ABM / ODE | prevalence RMSE | seconds |
|---|---|---|---|---|---|---|---|---|
| SEIR 2.5 | 1 | 0.8932 | 0.8928 | 0.8926 | 0.1472 / 0.1440 | 50.7 / 48 | 0.0050 | 0.8 |
| SEIR 2.5 | 4 | 0.8912 | 0.8928 | 0.8926 | 0.1452 / 0.1440 | 49.9 / 48 | 0.0045 | 3.2 |
| SIR 2 | 1 | 0.7981 | 0.7972 | 0.7968 | 0.1547 / 0.1539 | 29.7 / 27 | 0.0074 | 0.4 |
| SIR 2 | 4 | 0.7968 | 0.7972 | 0.7968 | 0.1537 / 0.1539 | 28.7 / 27 | 0.0058 | 1.3 |

Attack rate and peak height agree with the ODE to within 0.004. The ABM peak comes 1.7-2.7 days later than the ODE
peak; four sub-steps per day shrink the lag but do not remove it. We did not isolate the cause. Two candidates are
the synchronous update (a new case cannot transmit in the step it was infected) and the fact that the mean of the
per-replicate peak days is not the peak day of the mean curve.

Index-case R0 (`r0_index_case`, 1000 replicates, 4 steps per day): SEIR 2.566 (95% CI 2.383-2.749) against the
nominal 2.5; SIR 2.006 (1.858-2.154) against 2.0. Both CIs contain the nominal value.

Growth-rate R0 (`r0_growth`) depends on the fit window more than on the engine:

| model | window (start cases, stop share) | ODE curve | ABM mean curve (30 reps) | per replicate, median (IQR) |
|---|---|---|---|---|
| SEIR 2.5 | 10, 0.05 | 2.425 | 2.332 | 2.276 (2.109-2.429) |
| SEIR 2.5 | 10, 0.01 | 2.465 | 2.485 | 1.885 (1.288-2.578) |
| SIR 2 | 10, 0.05 | 1.952 | 1.890 | 1.865 (1.801-1.929) |
| SIR 2 | 10, 0.01 | 1.981 | 1.953 | 1.667 (1.499-1.948) |

Even on the noise-free ODE curve the default window reads 3% low (2.425 vs 2.5; 1.952 vs 2.0). The early window
still contains the transient before the dominant growth mode takes over, and the late days already show depletion.
On single replicates the short window (`stop_share=0.01`, about 4-9 days) is too noisy to use. A stricter start
(30 cases, 2% stop) left fewer than 3 days to fit, so `r0_growth` returned NaN.

### 2. Local contact on the grid (static homes)

Same SEIR (R0 = 2.5), n = 10 000 random homes on a 100 x 100 torus (about one agent per cell), i0 = 10, 400 days,
10 replicates, 4 steps per day. The index-case R0 uses 1000 replicates. `major` is the share of replicates with
attack rate > 1%. The prevalence RMSE is over all 400 days (the long zero tail dilutes it, so compare rows, not
sections). `r0_growth` uses the mean curve with the default window, and `r0_final_size` is `-ln(1 - z) / z` from
the mean attack rate z.

| radius | agents per window | major | attack rate | peak share | peak day | prev. RMSE vs ODE | R0 index (95% CI) | R0 growth | R0 final size | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 0.0 | 0.0016 | 0.0010 | 0.1 | 0.0327 | 0.464 (0.422-0.506) | - | - | 39 |
| 1 | 9 | 0.9 | 0.0193 | 0.0022 | 19.8 | 0.0324 | 1.790 (1.674-1.906) | - | 1.01 | 92 |
| 2 | 25 | 1.0 | 0.7949 | 0.0244 | 164.9 | 0.0317 | 2.240 (2.093-2.387) | 1.254 | 1.993 | 101 |
| 3 | 49 | 1.0 | 0.8662 | 0.0488 | 97.5 | 0.0287 | 2.327 (2.163-2.491) | 1.587 | 2.322 | 104 |
| 5 | 121 | 1.0 | 0.8857 | 0.0774 | 69.2 | 0.0224 | 2.424 (2.252-2.597) | 2.011 | 2.449 | 99 |
| 10 | 441 | 1.0 | 0.8912 | 0.1292 | 52.9 | 0.0073 | 2.623 (2.443-2.803) | 2.427 | 2.489 | 102 |
| None | 10 000 | 1.0 | 0.8930 | 0.1479 | 48.7 | 0.0013 | 2.566 (2.383-2.749) | 2.512 | 2.503 | 20 |

What the table shows:

- Local contact flattens and delays the epidemic long before it changes the final size. At radius 3 the attack
  rate is within 3 points of the ODE, but the peak is a third of the ODE's height and comes twice as late.
- The growth-rate R0 is always the lowest of the three, and the index-case and final-size values are close from
  radius 3 on. The index case measures the first generation in a fully susceptible neighbourhood. The growth rate
  sees a travelling wave whose speed is limited by local depletion, so an analyst fitting the early curve of a
  radius-2 epidemic would report R0 = 1.25 when the first-generation value is 2.24 (final size: 1.99). The gap
  closes as the window grows, and at full mixing all three are within 0.07 of 2.5.
- Radius 1 is the instructive failure. The index-case R0 is 1.79, yet every replicate died out after infecting
  1-4% of the town. A separate rerun at the same seed found 0 active cases at day 400 in 10/10 replicates, with
  mean cumulative cases of 183 at day 100 and 193.1 at days 200, 300 and 400. Our reading, not tested further, is
  that with about nine agents per window the neighbourhood is used up within a few generations. Either way, an
  index-case R0 above 1 does not guarantee a major outbreak on a sparse local structure. The 1% `major`
  threshold counts these runs as major, so read `attack_rate` alongside it.
- At radius 0 (same cell only) most agents are alone in their cell, so the index-case R0 is 0.46 and nothing
  spreads.

### 3. Mobility trajectories (YJMob100K)

`load_yjmob100k(n_users=5000, days=75)`: the first 5000 users of dataset 1 (loading took 2 s). After forward
filling, 73.7% of consecutive 30-minute slots stay in the same cell, and a user visits 224 distinct 500 m cells
on average over the 75 days. Same SEIR (R0 = 2.5), `steps_per_day=48` (one simulated day = one data day), i0 = 10,
150 days, 10 replicates. The index-case R0 uses 300 replicates and at most 120 days. ODE reference for n = 5000:
attack 0.893, peak share 0.144 on day 43.

| radius (cells) | contact range | major | attack rate | peak share | peak day | prev. RMSE vs ODE | R0 index (95% CI) | seconds |
|---|---|---|---|---|---|---|---|---|
| 0 | same 500 m cell | 0.0 | 0.0047 | 0.0020 | 0 | 0.0534 | 0.550 (0.449-0.651) | 90 |
| 1 | 1.5 km box | 1.0 | 0.6169 | 0.0552 | 69.7 | 0.0429 | 1.593 (1.389-1.798) | 568 |
| 2 | 2.5 km box | 1.0 | 0.7893 | 0.0954 | 56.0 | 0.0295 | 2.073 (1.805-2.342) | 559 |
| None | well mixed | 1.0 | 0.8930 | 0.1438 | 44.2 | 0.0041 | 2.333 (2.059-2.608) | 17 |

The well-mixed row again matches the ODE: attack 0.893 vs 0.893, peak 0.144 vs 0.144, peak day 44.2 vs 43. With
contact limited to agents nearby at the same half hour, the same `beta` gives R0 = 1.6-2.1 and an epidemic that
is smaller and slower. Radius 1 here gives a major outbreak in every replicate (attack 0.62), while radius 1 on
the static grid stalled at 2%. The two setups differ in density and contact pattern, so this is not a controlled
comparison; agents carrying the infection between neighbourhoods is the obvious candidate. These numbers describe
this model on these trajectories: co-location is not contact, and `beta` was not fitted to any disease. Each
local-contact row took 9-10 minutes (10 x 150 x 48 steps plus the index-case runs), against 17 seconds for full
mixing.

### 4. Generative-agent variant with FakeLLM rules (no real LLM)

These runs check the loop and the bookkeeping, not LLM behaviour. Each row is 20 seeds of `GenerativeEpidemic(n=100,
contact_rate=5, infectivity=0.1, i0=3)` (the paper's first experiment, nominal R0 = 3), run for up to 120 days with
a fixed `rule_responder`.

| condition | responder rule | attack rate (sd) | peak new cases / day | mean share going out | days until clear |
|---|---|---|---|---|---|
| base | nobody is told anything, all go out | 0.937 (0.019) | 12.4 | 1.000 | 28.0 |
| self_health | stay home with a cough or fever | 0.193 (0.134) | 3.35 | 0.953 | 14.6 |
| full | as above, plus everyone stays home once news >= 1% | 0.058 (0.023) | 1.80 | 0.626 | 9.6 |
| self_health | stay home with fever only | 0.852 (0.048) | 9.60 | 0.946 | 33.0 |

The `base` row is the mean-field check for this variant. With nobody isolating, the attack rate of 0.937 matches the
final-size value for R0 = 3, `final_size(3) = 0.940`. The other rows only show that isolation decisions reach the
transmission step in the expected direction. Days 1-2 of infection are infectious and symptom-free, so isolating
on symptoms alone leaves part of the transmission in place. Isolating on fever only (days 4-5) still infects 85%
of the town. The size of these effects with a real LLM is the paper's question and has not been measured here.

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
  not seconds (see Measured numbers, section 3).
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

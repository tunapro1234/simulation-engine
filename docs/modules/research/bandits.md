# simengine.research.bandits: choosing among methods under an evaluation budget

`simengine/research/bandits.py` answers one question: there are K methods or configs, every evaluation is noisy
and costs budget, so which one is best? It has three parts.

- **Regret policies:** UCB1 and Thompson sampling with Beta or Gaussian priors.
- **Best-arm identification:** successive rejects (fixed budget), LUCB1 (fixed confidence, with an optional budget
  cap) and uniform allocation as the baseline.
- **Tools around them:** a simulator that runs thousands of replications as numpy arrays, and `RacingProposer`, which
  plugs a race into `simengine.research.Loop`.

The main names (`SimulatedArms`, `identify_best`, `RacingProposer`, ...) are exported from `simengine.research`;
`simulate` and `compare` are reached through the module: `from simengine.research.bandits import ...`.

```python
from simengine.research.bandits import SimulatedArms, identify_best, simulate, compare, RacingProposer
arms = SimulatedArms([0.5, 0.45, 0.4], kind="bernoulli", seed=1)        # or any pull(arm) -> float
res = identify_best(arms, 3, budget=600, algorithm="successive_rejects", seed=0)
res.best, res.pulls, res.means, res.rejected
simulate(arms, "lucb", None, reps=1000, delta=0.05)    # error_rate, error_se, mean_pulls, regrets
loop = Loop(runner, space, proposer=RacingProposer("successive_rejects", pulls_per_arm=4),
            primary="err", higher_is_better=False)      # the runner reads cfg["fidelity"] = 1, 2, ... as a batch index
```

## API

| name | what it does |
|---|---|
| `identify_best(pull, n_arms, budget, algorithm, *, higher_is_better=True, seed=0, bounds=None, **kw)` -> `BanditResult` | Runs one bandit with `pull(arm) -> float`. `algorithm` is one of `ucb1`, `thompson_beta`, `thompson_gaussian`, `successive_rejects`, `lucb`, `uniform`; aliases such as `sr`, `ts` and `lucb1` also work. If `higher_is_better=False` the rewards are negated, and `means` are still reported in your units. A non-finite reward, such as a crashed evaluation, counts as -inf. `bounds=(lo, hi)` sets the Thompson Beta rescaling and the default `scale` for UCB1 and LUCB. Other keywords: `delta`, `epsilon` and `scale` for LUCB; `scale` for UCB1; `sigma`, `prior_mean` and `warmup` for Thompson Gaussian; `prior` and `warmup` for Thompson Beta. |
| `ucb1`, `thompson_beta`, `thompson_gaussian`, `successive_rejects`, `lucb`, `uniform` | Shortcuts for `identify_best` with that algorithm. `lucb(pull, K, budget=None, delta=0.05, epsilon=0.0)`. |
| `BanditResult` | Fields: `best`, `pulls` (K,), `means` (K,), `budget_used`, `stopped` (`budget`, `phases`, `confident` or `single arm`), `rejected` (the order successive rejects dismissed arms in), the `arms` and `rewards` sequences, and `to_dict()`. |
| `SimulatedArms(means, kind="bernoulli" or "gaussian", sigma=1.0, seed=0)` | Synthetic arms. You can pass it as `pull`. It also has `.draw(rows, arms)` (vectorised), `.best`, `.gaps` and `.hardness()`, which returns H1 and H2 as defined by Audibert et al. 2010. |
| `simulate(arms, algorithm, budget, *, reps=1000, seed=0, **kw)` | Runs `reps` independent replications as `(reps, K)` arrays. Returns `error_rate` (P(recommended arm is not the best)), its binomial `error_se`, `mean_pulls`, `simple_regret`, `cumulative_regret` (pseudo-regret: the sum of pulls x gap) and `confident_rate` (LUCB). It refuses a tied best arm. |
| `compare(arms, budget, algorithms=ALGORITHMS, *, reps, seed, params={alg: kw})` | Runs `simulate` once per algorithm and returns a DataFrame. |
| `RacingProposer(algorithm="successive_rejects", *, budget=None, pulls_per_arm=4, n_candidates=9, n_grid=5, pull=None, fidelity_of=None, seed=0, **algo_kw)` | A `Loop` proposer (`name="racing"`). It builds the single-change grid neighbours of the current best, as `HalvingProposer` does, with at most `n_candidates`. It races them with `budget` pulls (default `pulls_per_arm * K`) and proposes the winner. Pull j of a candidate calls `pull(cfg, j)`, or if no `pull` is given, the loop's screening run `ctx.screen([cfg], fidelity_of(j))`. By default the runner then sees `cfg["fidelity"] = j + 1` and should score mini-batch j + 1 (a disjoint unit subset or a fresh seed). The loop caches screening runs by (config, fidelity), so a resumed loop does not repeat them. The race and its result go into the proposal's `notes["race"]`, and `.evaluations` counts the pulls. |
| `sr_schedule(K, n)`, `logbar(K)`, `lucb_beta(u, t, K, delta, scale)` | The successive-rejects phase lengths n_k, the log-bar(K) constant, and the LUCB1 confidence radius. |
| `AUDIBERT_BUBECK_2010` | `{name: (means, n)}` for the seven Bernoulli problems in §7 of the paper, plus `exp7_2h`, its second budget. The best arm is listed first. |

## Provenance: the algorithm boxes these follow

Each implementation was checked against the paper's own algorithm box. The PDFs were read on 22 Oct 2026.

- **UCB1**: Auer, Cesa-Bianchi & Fischer, *Finite-time analysis of the multiarmed bandit problem*, Machine Learning 47
  (2002), Fig. 1 and Theorem 1. Play each arm once, then play argmax of mean_j + sqrt(2 ln n / n_j). Rewards are in
  [0, 1]. Here `scale` multiplies the bonus to cover other reward ranges.
- **Thompson sampling, Beta**: Agrawal & Goyal, *Analysis of Thompson Sampling for the multi-armed bandit problem*,
  COLT 2012 (PMLR 23), Algorithms 1 and 2. The posterior is Beta(S+1, F+1). A reward in [0, 1] becomes a Bernoulli
  trial with that success probability. Here a reward is first rescaled from `bounds` (default (0, 1) in your units,
  also when `higher_is_better=False`) to [0, 1].
- **Thompson sampling, Gaussian**: Agrawal & Goyal, *Further optimal regret bounds for Thompson Sampling*, AISTATS
  2013 (PMLR 31), Algorithm 2. theta_i ~ N(mu_i, 1/(k_i+1)) with mu_i = sum of rewards / (k_i + 1), as the paper's
  text defines it (derivation in §1.2, definition with Theorem 2 in §1.3). The algorithm box prints the update as
  (mu_i k_i + r)/(k_i + 2), which does not give that mean from k_i = 1 on; the conjugate update of the text is (mu_i
  (k_i + 1) + r)/(k_i + 2), and that is what is implemented. Here `sigma` scales the variance to sigma^2/(k_i+1), and
  `prior_mean` replaces the paper's prior mean of 0. The defaults (sigma 1, prior_mean 0, warmup 0) are the paper's
  algorithm. `warmup` is an addition that is not in the paper.
- **Successive rejects**: Audibert, Bubeck & Munos, *Best arm identification in multi-armed bandits*, COLT 2010, Fig.
  3 and Theorem 2. The phase lengths are n_k = ceil((n - K) / (logbar(K)(K + 1 - k))), with logbar(K) = 1/2 +
  sum_{i=2..K} 1/i. After each phase the arm with the lowest empirical mean is dismissed, with ties broken at random.
  A test checks the schedule against the formula and against the budget.
- **LUCB1**: Kalyanakrishnan, Tewari, Auer & Stone, *PAC subset selection in stochastic multi-armed bandits*, ICML
  2012, §3.1 to §3.3, with m = 1. Rounds 1 to K sample every arm once, and from round K + 1 on each round samples two
  arms (§3.2, footnote 3), so the first stopping check uses t = K + 1. In round t, h is the empirical best and l is
  the arm other than h with the largest mean + beta. Stop iff UCB(l) - LCB(h) < epsilon; otherwise sample both. The
  radius is beta(u, t) = sqrt(ln(k1 n t^4 / delta) / (2u)) with k1 = 5/4, where n is the number of arms. Here a
  `scale` multiplies it for rewards that are not in [0, 1], and an optional `budget` cap is added.
- **Uniform**: equal allocation that recommends the empirical best. It is the baseline used in Audibert et al. §7,
  analysed in Bubeck, Munos & Stoltz (ALT 2009).

There is no code from other libraries in this module.

## Measured numbers

Every number in this section comes from runs made for this document on 22 Oct 2026, using synthetic arms only. The
LUCB1 numbers were re-run after its round index was corrected to start at K + 1 (the other columns did not change).
The outputs are outside the repo, in `bandits_rates.csv`, `bandits_extra.json` and `bandits_racing.log`. Commands, run
from the repo root:

```bash
PYTHONPATH=. nice -n 19 python <scratch>/bandits_measure.py <out_dir>   # rates table + LUCB + first racing table, 477 s on a loaded box
PYTHONPATH=. nice -n 19 python <scratch>/bandits_racing.py              # racing table below
```

The rates script calls `compare(SimulatedArms(means, seed=1), n, reps=2000, seed=0)` for every
`AUDIBERT_BUBECK_2010` problem and for one Gaussian problem. The script itself is not in the repo. Its loop is the
one-line `compare` call above.

**Probability of error.** 2000 replications per cell, binomial SE <= 0.011. In the paper's column, "Unif / SR" are
the uniform and SR bars of Audibert et al. 2010 Figure 5, **read off the figure by eye** (about ±0.01). They are a
literature number, not a run of ours.

| problem (K, n) | uniform | SR | LUCB1 | UCB1 | TS Beta | TS Gauss | paper Fig. 5: Unif / SR |
|---|---|---|---|---|---|---|---|
| exp1 (20, 2000) | 0.446 | 0.187 | 0.203 | 0.294 | 0.197 | 0.261 | 0.46 / 0.155 |
| exp2 (20, 2000) | 0.421 | 0.177 | 0.199 | 0.287 | 0.176 | 0.249 | 0.395 / 0.19 |
| exp3 (4, 2000) | 0.286 | 0.258 | 0.252 | 0.248 | 0.257 | 0.256 | 0.145 / 0.107 |
| exp4 (6, 600) | 0.220 | 0.166 | 0.146 | 0.151 | 0.130 | 0.150 | 0.222 / 0.16 |
| exp5 (15, 4000) | 0.142 | 0.043 | 0.034 | 0.027 | 0.015 | 0.011 | 0.33 / 0.205 |
| exp6 (20, 6000) | 0.299 | 0.200 | 0.218 | 0.235 | 0.140 | 0.177 | 0.315 / 0.187 |
| exp7 (30, 6000) | 0.523 | 0.182 | 0.243 | 0.350 | 0.190 | 0.320 | 0.55 / 0.225 |
| exp7 (30, 12000) | 0.299 | 0.040 | 0.075 | 0.120 | 0.044 | 0.083 | 0.30 / 0.04 |
| Gaussian sd 1, means 1, .8, .7, .6, .5, .2 (6, 600) | 0.091 | 0.045 | 0.051 | 0.062 | 0.114 | 0.041 | - |

The settings were: LUCB1 with delta 0.05, epsilon 0 and the budget as a cap; UCB1 and LUCB with scale 1 on the
Bernoulli rows and scale 4 on the Gaussian row; TS Beta with bounds (-3, 4) on the Gaussian row; TS Gauss with the
paper's sigma 1 and prior 0. The regret policies recommend the empirical best when the budget runs out.

Four things to read from the table:

- On the six problems other than exp3 and exp5, our uniform and SR rates are within 0.045 of the bars read off the
  paper's figure. The largest gaps are exp1 SR (0.187 against about 0.155) and exp7 SR at n = 6000 (0.182 against
  about 0.225).
- exp3 and exp5 do not match. The paper says n was chosen close to H1. For the means as printed in the text, H1 is
  6137 for exp3 (n = 2000) and 1329 for exp5 (n = 4000), so our exp3 is much harder and our exp5 much easier than the
  paper's runs. Most likely the printed formulas (0.5 - 0.37^i and 0.5 - 0.025i) are not the ones that were run. We
  did not resolve this. One reading fits exp5: with gaps 0.025 (i - 1), i = 2..15 (mu = 0.475, 0.45, ..., 0.15),
  H1 = 4122 at n = 4000, and our rates become uniform 0.346 and SR 0.182 (2000 runs, SE <= 0.011), against about
  0.33 and 0.205 in the figure. `AUDIBERT_BUBECK_2010["exp5"]` keeps the printed formula. We found no reading of the
  exp3 formula with H1 near 2000.
- exp1 SR (0.187 against about 0.155) is not an implementation slip as far as we can tell: an independent
  per-replication SR with binomial phase sums gave 0.184 over 20000 runs, and the module 0.187 over 20000 runs
  (SE 0.003 each).
- SR, LUCB1 and Thompson-Beta beat uniform by a wide margin wherever there are many bad arms (exp1, exp2, exp7).
  UCB1 and TS-Gauss with their default exploration constants do worse there. UCB1's bonus is tuned for regret, not
  identification, which may explain this. We did not try a smaller `scale`.

**Cumulative pseudo-regret** (sum of pulls x gap, mean over 2000 runs), on the same runs. exp4 at n = 600: uniform 58.0,
SR 49.9, UCB1 48.1, LUCB 42.8, TS Gauss 44.6, TS Beta 33.9. exp5 at n = 4000: uniform 793, SR 505, UCB1 426, LUCB 357,
TS Gauss 326, TS Beta 152. Thompson-Beta had the lowest regret on exp3, exp4, exp5, exp6 and exp7 at n = 12000.
LUCB1 had the lowest on exp1, exp2 and exp7 at n = 6000, for example 147 against 155 on exp1.

**LUCB1 fixed confidence** (no budget given, so the cap is 1000 K; 2000 runs):

| means | delta | error | stopped confident | mean pulls |
|---|---|---|---|---|
| 0.9, 0.5, 0.4 | 0.10 / 0.05 / 0.01 | 0 / 0 / 0 | 100% / 100% / 100% | 848 / 873 / 929 |
| 0.7, 0.5, 0.5, 0.3 | 0.10 / 0.05 / 0.01 | 0 / 0 / 0 | 3.1% / 2.6% / 0.75% | 3990 / 3993 / 3998 |

The Hoeffding radius with t^4 is conservative. The error stays far below delta, and on the second problem almost
every run reaches the 4000-pull cap before it can certify the answer.

**RacingProposer, the pick-best rate inside one turn.** The space is `x` in [0, 1] plus `method` a/b. The incumbent
is x = 0.1, method a. The loss is 2(x - 0.7)^2 + 0.15[method a] + N(0, sd^2) per pull. There are 7 grid candidates,
42 pulls (6 per arm) and 500 races per cell. The rate is the share of races that propose the truly best candidate,
x = 0.75.

| noise sd | uniform | SR | LUCB1 (scale 4 sd) | UCB1 (scale 4 sd) | TS Gauss sigma=sd, prior 1 | TS Gauss sigma=sd, warmup 1 | TS Gauss paper defaults |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.912 | 0.954 | 0.972 | 0.970 | 0.260 | 0.976 | 0.936 |
| 0.3 | 0.586 | 0.662 | 0.682 | 0.680 | 0.536 | 0.700 | 0.650 |

With a confident prior far from the rewards (sigma = noise sd, prior mean 1 against losses of about 0.15 to 0.7),
Gaussian Thompson sampling barely explores: the prior has the same variance as one reward. `warmup=1` fixes this.
A unit test (`test_thompson_warmup`) shows the collapse.

## Limitations

- **Rewards are assumed independent across pulls.** With `RacingProposer` and the default `ctx.screen` adapter, a
  pull is one screening run at fidelity j + 1. It is only an independent draw if the runner makes batch j a disjoint
  unit subset or a fresh seed. A runner written for `HalvingProposer`, which reads fidelity as a fraction, returns
  correlated or identical scores and makes the race meaningless. Pass `pull=` or a matching `fidelity_of=` in that
  case. The loop's cache key formats fidelity with 4 significant digits, so pull indices above 9999 collide.
- **Every pull is a look at the dev split.** The race only picks the proposal, and the loop still judges it on full
  dev with its paired CI. Many races still mean more chances to overfit dev, as with halving.
- **Bounded-reward guarantees.** UCB1, LUCB1 and Thompson-Beta are analysed for rewards in [0, 1]. For other metrics,
  `bounds` or `scale` has to be supplied by you, and a wrong range breaks the guarantees: too small explores too
  little, too large wastes budget. Successive rejects and uniform allocation are scale-free.
- **LUCB without a budget** is capped at 1000 K pulls. When two arms tie for best and epsilon = 0 it cannot stop
  confidently, and it returns `stopped="budget"`.
- **Successive rejects** uses at most n pulls, often a few fewer, because of the ceilings. It needs n >= K, and with
  n close to K it is close to random.
- **Not implemented:** UCB-E, adaptive UCB-E and Hoeffding races (the other bars of Audibert et al.), top-m
  identification, and contextual bandits.
- **Speed.** `simulate` is vectorised over replications, but the sequential policies still take one numpy step per
  pull. 2000 replications of exp7 at n = 12000 took about 230 s for all six algorithms, on a machine at load
  average above 100. In a separate timing (exp1, 1000 replications) Thompson-Beta was the slowest at 2.3 s; the
  others took at most 0.9 s.

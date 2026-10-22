# `simengine.research.bayesopt`: Bayesian optimisation proposers (GP-EI / GP-UCB, TPE-lite)

## What

Model-based search over a `simengine.research.Space`. There are two surrogates. The first is a Gaussian process
(scikit-learn `GaussianProcessRegressor`, Matern 5/2 with one length scale per column, one-hot categoricals) scored
with expected improvement or GP-UCB. The second is TPE-lite, the tree-structured Parzen estimator of Bergstra et al.
(2011) without conditional keys. You can use them in two ways:

- **`GPProposer` / `TPEProposer`** plug into `Loop` like `RandomProposer`. The Loop contract allows **one changed key
  per turn** from the current best, so the acquisition is maximised over the one-change neighbours of the incumbent
  rather than over the whole space. The neighbours are every allowed choice of a categorical key, plus
  `n_per_key` uniform values and the 5-point grid of a range key, or every value of a small int range.
- **`minimize(objective, space, method="gp" | "tpe" | "random")`** is plain sequential optimisation where every key
  moves. This is the setting of the BO literature; use it when the objective is a cheap function and not a Loop turn.

The module is not registered in `simengine.research.__init__` yet. Import it directly:
`from simengine.research.bayesopt import GPProposer`.

## API

| name | what it does |
|---|---|
| `GPProposer(acquisition="ei" \| "ucb", xi=0.01, beta=None, n_per_key=256, n_init=None, seed=0, n_restarts=2, crash="worst")` | `propose(ctx)` returns one change. It fits a GP on every finished turn: the dev primary mean, negated if higher is better. A crashed turn gets the worst loss seen. Proposals during the first `n_init` observations (default `max(3, min(10, dim+1))`) are seeded random single changes (`kind="random-init"`). After that the change is the argmax of EI or UCB (`kind="ei"` / `"ucb"`), and `notes` carries `pred_mean`, `pred_sd`, `acq`, `n_obs` and `n_candidates`. |
| `TPEProposer(gamma=0.15, n_candidates=100, n_init=None, seed=0, crash="worst")` | It splits the observations at the gamma quantile and draws `n_candidates` single-key values from l(x), spread over the used keys. It proposes the one whose full config has the largest log l(x) - log g(x) (`kind="tpe"`). If every draw from l was already run, it falls back to uniform neighbours. |
| `minimize(objective, space, method, n_calls=30, n_init=None, seed=0, acquisition="ei", n_candidates=None, gamma=0.15, xi=0.01, beta=None, higher_is_better=False, x0=())` | Full-config sequential optimisation. GP: 2000 uniform candidates plus 500 Gaussian perturbations (sd 0.05 in unit coordinates) of the 5 best configs. TPE: 100 draws from l. Warm-up (`n_init`): 10 for TPE, `max(3, min(10, 2*dim+1))` for GP. Random: uniform draws. Configurations already run are never proposed again. An objective that raises or returns NaN is recorded as NaN, and the surrogate sees the worst finite loss so far; until one call has returned a finite loss, proposals stay uniform. Returns `{"best", "best_cfg", "history": [{"cfg", "value", "method"}], "trace"}`, where `trace` is the best value so far. |
| `SpaceEncoder(space)` | Encodes the non-fixed keys. `encode` maps ranges to [0, 1] and categoricals to one-hot without forbidden values; a value that is not allowed encodes as all zeros. `unit` returns unit coordinates plus category indices. `from_unit` decodes, rounding int ranges. `sample(n, rng)` draws uniform configs. |
| `observations(history, primary, higher_is_better, crash="worst" \| "skip")` | Converts Loop turn records into (configs, losses). |
| `fit_gp`, `gp_scores`, `expected_improvement`, `upper_confidence_bound`, `ucb_beta` | The GP pieces. `expected_improvement(mu, sd, best, xi)` is the minimisation closed form, `(best - xi - mu) Phi(z) + sd phi(z)`. `ucb_beta(t, n, delta=0.1, scale=0.2)` returns `scale * 2 log(n t^2 pi^2 / (6 delta))`; `gp_scores` uses t = observations + 1. |
| `Parzen1D`, `TPE` | The adaptive Parzen estimator of one range key, and the l/g model over a space with `score`, `sample` and `sample_key`. |
| `branin`, `branin_space()`, `BRANIN_MIN`, `mixed_toy`, `mixed_toy_space()` | The test objectives. Branin on [-5, 10] x [0, 15] has a minimum of 0.397887. `mixed_toy` is a 4-way categorical `method` × `x` in [0, 1] × int `k` in [1, 16], with minimum 0 at (`b`, 0.75, 6); the best `x` depends on the method. |

```python
from simengine.research import Loop, RandomProposer
from simengine.research.bayesopt import GPProposer, TPEProposer, minimize, branin, branin_space

loop = Loop(runner, space, proposer=GPProposer("ei", seed=0), primary="err", higher_is_better=False,
            stop_after=None, max_turns=20)
out = loop.run()
ctrl = Loop(runner, space, proposer=RandomProposer(seed=1000), primary="err", higher_is_better=False,
            stop_after=None, max_turns=20).run()          # always run the random arm at the same budget

res = minimize(lambda c: float(branin(c["x1"], c["x2"])), branin_space(), method="gp", n_calls=30, seed=0)
```

## Provenance (checked against the papers)

- **TPE**: J. Bergstra, R. Bardenet, Y. Bengio, B. Kégl, "Algorithms for Hyper-Parameter Optimization", NIPS 2011
  (proceedings PDF, read 22 Oct 2026). Section 4.1 defines l(x) as the density of the observations with loss below
  y\*, where y\* is the gamma quantile, and g(x) as the density of the rest. It derives EI ∝ (γ + g(x)/l(x) (1−γ))^-1,
  so the candidate with the largest l/g is chosen. Section 4.2 builds each range factor as an equally weighted
  mixture of the prior and a Gaussian at each point. Each Gaussian's sd is the larger of the distances to the left
  and right neighbours, with the interval ends counted as neighbours, "clipped to remain in a reasonable range". A
  discrete factor gets posterior ∝ N p_i + C_i. Section 6 uses γ = 0.15, takes the best of 100 candidates drawn from
  l(x), and starts from 30 random points.
  Implemented as written, with these choices of ours:
  - The prior of a range is N(0.5, 1) in unit coordinates, truncated to [0, 1].
  - The sd clip is [1/min(100, 1+K), 1], borrowed from hyperopt's convention; the paper gives no numbers.
  - The good set has max(1, ceil(γ n)) points.
  - Int ranges are treated as continuous and rounded, not quantised.
  - The search stays inside the bounds; the paper let TPE grow past them.
  - There are no conditional (tree) keys.
  - `minimize` uses a 10-point warm-up.
- **GP-EI**: Bergstra et al. (2011) §3.1 sets y\* = min f(x_i) and maximises EI under the GP posterior; our EI is
  the standard Gaussian closed form with an optional margin xi (`xi * sd(y)`). The paper used a squared-exponential
  kernel and maximised EI with CMA-ES plus EDA. Here we use Matern 5/2 with ARD plus a white-noise term, and maximise
  over random candidates. That is cheaper and cruder, and it is our choice, not the paper's.
- **GP-UCB**: N. Srinivas, A. Krause, S. Kakade, M. Seeger, "Gaussian Process Optimization in the Bandit Setting: No
  Regret and Experimental Design", ICML 2010 (arXiv:0912.3995v4, read 22 Oct 2026). Algorithm 1 selects x_t = argmax
  mu_{t-1}(x) + beta_t^{1/2} sigma_{t-1}(x). Theorem 1 (finite D) sets beta_t = 2 log(|D| t² π² / (6δ)), and §6
  reports that scaling beta_t down by a factor of 5 worked better in their experiments. We use |D| = the number of
  candidates scored this round, t = the number of observations + 1 (round t sees t − 1 points), δ = 0.1 and scale
  1/5, unless `beta` is given. Before 22 Oct 2026 (commit a508694) t was the number of observations, one round early.
- Branin: the usual constants and minimum 0.397887 (the three minimisers are checked in the tests).

## Measured numbers

Measured 22 Oct 2026 on synthetic objectives only (no real data). The GP-UCB rows are from a rerun after the
beta_t round fix (commit a508694, `--jobs 16`); every other row of that rerun reproduced the first run exactly.
Command, from the repo root:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. nice -n 19 python examples/bayesopt_bench.py --seeds 20 --loop-seeds 10 --jobs 12 --out runs/bo.json
```

Seeds 0-19 (full space) and 0-9 (Loop). All methods with the same seed share the same warm-up draws in `minimize`.
Regret is the best value found minus the known minimum (Branin 0.397887, toy 0); lower is better. "wins" counts
paired seeds where the method beat random (ties: equal regret, i.e. both stuck at a shared warm-up point).

**A. `minimize`, full configs, 30 objective calls** (warm-up: GP 5 calls on Branin, 10 on the toy; TPE 10):

| objective | method | median regret @10 | @20 | @30 | mean @30 | seeds < 0.1 @30 | wins vs random (ties) |
|---|---|---|---|---|---|---|---|
| Branin | random | 3.61 | 1.51 | 1.31 | 1.44 | 0 / 20 | - |
| Branin | GP-EI | 1.77 | 0.206 | 0.036 | 0.036 | 20 / 20 | 20 / 20 |
| Branin | GP-UCB | 4.16 | 1.24 | 0.008 | 0.018 | 20 / 20 | 20 / 20 |
| Branin | TPE-lite | 3.61 | 0.756 | 0.583 | 0.662 | 0 / 20 | 14 / 20 (3) |
| mixed toy | random | 0.265 | 0.076 | 0.066 | 0.089 | 13 / 20 | - |
| mixed toy | GP-EI | 0.265 | 0.0009 | 0.0000 | 0.0000 | 20 / 20 | 20 / 20 |
| mixed toy | GP-UCB | 0.265 | 0.026 | 0.0000 | 0.016 | 19 / 20 | 19 / 20 |
| mixed toy | TPE-lite | 0.265 | 0.024 | 0.013 | 0.087 | 15 / 20 | 13 / 20 (1) |

**B. `Loop`, one change per turn, 20 turns, `stop_after=None`, no finalize.** Per-unit runners with 100 units:
Branin + shared unit difficulty (sd 5) + own noise (sd 1); toy + sd 0.2 / 0.02. Score = noise-free regret of the
final incumbent (the last kept turn). Start config: Branin (2.5, 7.5), regret 23.7; toy (a, 0.5, 12), regret 1.23.

| objective | proposer | median regret | mean regret | kept turns (mean) | wins vs random |
|---|---|---|---|---|---|
| Branin | RandomProposer | 3.02 | 3.33 | 3.1 | - |
| Branin | GPProposer EI | 0.160 | 0.347 | 6.4 | 10 / 10 |
| Branin | GPProposer UCB | 0.104 | 0.172 | 5.5 | 10 / 10 |
| Branin | TPEProposer | 1.28 | 5.03 | 3.2 | 5 / 10 |
| mixed toy | RandomProposer | 0.015 | 0.214 | 6.2 | - |
| mixed toy | GPProposer EI | 0.300 | 0.266 | 6.5 | 3 / 10 |
| mixed toy | GPProposer UCB | 0.301 | 0.340 | 6.1 | 4 / 10 |
| mixed toy | TPEProposer | 0.312 | 0.268 | 5.6 | 4 / 10 |

What it says:
- With every key free, the GP beats random at equal budget on both objectives in every paired seed. On Branin it
  gets within 0.1 of the minimum in 40 of 40 GP runs and 0 of 20 random runs. TPE-lite sits between them. It helps on
  Branin (median 0.58 vs 1.31), but 30 calls is short for it: 10 of them are warm-up, and the paper's DBN runs went to 200
  trials, the first 30 of them random (Bergstra et al. 2011 §6).
- Inside the Loop (one key per turn), the GP still clearly beats random on Branin. **On the mixed toy all three model
  proposers are worse than random** (median 0.30 vs 0.015). They settle at method `c` with x near its own optimum
  0.4 (regret 0.30). A rerun of seeds 0 to 3 confirmed this: GP-EI ended at `c` in 3 of 4 seeds and TPE in 4 of 4. From there, the single change to the best method `b` lands at x = 0.4, where `b` is worse
  (0.37), so no one-change move improves. Random sometimes jumps to `b` early, while x is still near a value that
  suits `b`. That is the price of the one-change contract on interacting keys, not a bug in the surrogate. Run the
  random arm next to BO, or use `minimize` when the turns are cheap.
- Cost: on this shared machine (load average about 130 to 160 on 32 cores during the run), one GP run of 30
  calls took about 55 to 85 s wall time and 20 GP Loop turns about 45 s. TPE took under 0.3 s and random almost
  nothing. These wall times are inflated by the load; read them as relative, not absolute. In the rerun (load
  average about 45 to 50, `--jobs 16`) a 30-call GP run took 16 to 24 s and 20 GP Loop turns 11 to 16 s; the whole
  benchmark took 2 min 22 s wall time.

## Limitations

- **One change per turn.** Inside `Loop`, BO is a coordinate search around the incumbent. Most of the surrogate's
  advantage in the literature comes from moving several keys at once, which the Loop forbids by design. The
  incumbent also moves only on a CI-significant keep, so proposals stay near a point that may be stale.
- **What the GP sees.** It is fitted on dev means only, with no per-unit variance, so `WhiteKernel` has to absorb
  the noise. A single noisy dev mean is treated as the truth at that point.
- **Candidates, not optimisation.** EI and UCB are maximised over random candidates, not by a gradient or CMA-ES
  search, so in more than about 6 to 8 range keys the argmax is rough. Categoricals are one-hot inside a stationary
  kernel; ordering and interactions are learned only through the length scales.
- **Cost.** The GP refits every call with 2 optimiser restarts and is cubic in the number of observations. It is
  fine up to a few hundred turns and wasteful for cheap objectives (see `sec` in the tables).
- **TPE-lite** has no conditional parameters, no quantised or log-scale ranges (the `Space` has no log flag) and no
  multivariate kernel. Each key is an independent factor, so it cannot represent the method × x interaction of the
  mixed toy.
- More dev looks with a better proposer still means more chances to overfit dev. Keep the selection split, the
  single test run and the random control arm (docs/modules/research.md).

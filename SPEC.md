# simengine build contract (for the agents assembling the library)

Goal: one installable Python library, `simengine`, that packages the simulation methods and the
evidence pipeline measured during the TALP prep nights. It should be easy to pick up on hackathon
day: `pip install -e .`, then a few lines go from a raw table to an evidence card with confidence
intervals and an honest baseline ladder.

Source research code (read-only, never modify): the sibling research tree
`talp-simulation-hackathon-2026/uretim/<branch>/` (next to this repo, not part of it).
Each branch has `RAPOR.md` (measured results), `RECETE.md` (rebuild recipe) and `kod/`.

Second porting round (October 2026, read-only, same owner): **replicant** (MIT), **edsl-test** (no LICENSE
file) and the **awesome-llm-social-simulation** notes vault (knowledge only, no code). Their material went into
`persona/methods/`, `persona/bigfive.py`, `sim/constrained.py`, the new `games/` module, `abm` society games,
`methods/mixture.py`, `evidence/extras.py`, `research` targets and harness variants, the game `bench` tasks and
`docs/{landscape,validation,failure-modes}.md`. Credits are in `NOTICE`.

## Rules
1. **Port, don't reinvent.** Move working, measured code into the library. Keep the algorithms and
   their defaults. The public API (module names, public functions and classes, arguments) is
   English. Internals may keep Turkish names when ported verbatim. Docstrings can be English or
   Turkish.
2. **Ownership.** Each module agent writes only:
   - `simengine/<module>/`
   - `tests/test_<module>*.py`
   - `docs/modules/<module>.md`

   `games` is a module like the others: its agent owns `simengine/games/`, `tests/test_games*.py`
   and `docs/modules/games.md`.

   Cross-module imports go only through `simengine/types.py`, `simengine.llm` (the `LLM`
   protocol and `FakeLLM`) and `simengine.evidence.evaluate`. Three lazy, public-API exceptions
   exist: the game bench tasks (`bench/tasks/econ_games_b01.py`, `ertan_wtp.py`,
   `beebench_b01_prior.py`) import `simengine.games` and `simengine.abm.society.run_who_to_punish`,
   whose output they score, and `persona.validate` imports `simengine.sim.parse.parse_json` for
   JSON-fenced Likert answers. If you need something from another
   module, depend on the protocol in `types.py`. Do not copy another module's internals.
3. **No secrets, no data, no machine paths.**
   - No API key values, and no key file paths on the build machine. The key comes from
     `ANTHROPIC_API_KEY` or a file named in `SIMENGINE_KEY_FILE`.
   - No datasets in the repo. Loaders take `data_root`, or read the `SIMENGINE_DATA` env var,
     with documented expected layouts.
   - No absolute `/srv` or `/data` paths in library code. Tests may skip when data is absent.
   - No participant or person names anywhere (paper authors in citations are fine).
   - Small published-literature constants (game references, Ertan targets, prevalence weights, norm
     tables, calibrated mixture weights) count as parameters, not data. They may be inlined as Python
     constants if each carries its citation and source file. Row-level datasets (GSS rows, Twin-2K,
     Anthology backstories, result tables) never ship.
4. **Dependencies.**
   - Core: numpy, pandas, scikit-learn, scipy.
   - Optional extras, imported lazily: `anthropic` (llm), `lightgbm`, `statsmodels`,
     `transformers`/`torch` (vision).
   - Nothing new may be installed. Run tests with the project's virtualenv Python:
     `python -m unittest discover -s tests -t .` (pytest is not installed).
5. **Tests.** Each module gets unittest tests on small synthetic data. They must run offline in
   under 60 s, use `FakeLLM` and make no network calls. One optional live smoke test may run
   behind `SIMENGINE_LIVE=1`. Do not run it.
6. **Honesty.** Docs state when a method works and when it does not, using the measured numbers
   from `RAPOR.md`, each with its provenance (branch and file). Never invent a number.

## Package layout
```
simengine/
  types.py        shared contracts (exists; do not change without the integrator)
  llm/            Claude (anthropic SDK): disk cache, cost ledger, budget guard, effort,
                  cache_control, parallel map, 429 backoff, stop on 401 or limit errors.
                  ClaudeCode (`claude -p` subprocess, subscription), FakeLLM (deterministic).
  persona/        TALP-compatible Persona/AudienceSpec schema; persona sources
                  (Claude/Fake/TALP slot); builders from survey rows and purchase history;
                  persona method zoo (`methods/`, 10 methods + agent bank); Big Five builder
                  and Mini-IPIP fidelity check (`bigfive.py`, `validate.py`).
  sim/            runners: survey panel (+SSR), A/B (PAARS), roundtable, strategy;
                  segment-distribution prompting; per-person prediction prompts;
                  swap/no-persona controls; schema-constrained answers with
                  validate-and-reprompt feedback (`constrained.py`).
  methods/        baselines (constant, majority, overall copy, popularity, persistence,
                  logistic, LightGBM, kNN); A/B stack (TF-IDF + LLM); LLM prior + early-test
                  Bayes; hybrid (deviation + shrinkage, LLM-as-feature); twin person model;
                  conjoint MNL; next-action ranker; theory-persona mixture calibration.
  abm/            adoption ABM + ABC + analogy prior, Bass baseline, opinion dynamics,
                  diffusion (Hawkes/SEISMIC), turn-based LLM agent society (world loop);
                  economic-game societies (who-to-punish, repeated PD, trust).
  evidence/       metrics, bootstrap and paired CIs, baseline ladder, PPI++ "worth N humans",
                  calibration, leakage/memorization audits, evidence card (md/html), dashboard;
                  extras (variance ratio, completion row, published findings, stylized-facts
                  score, coefficient recovery).
  research/       pre-registration lock (SHA-256 + git), sealed predictions, turn-based
                  auto-research loop (fixed runner, one change per turn, paired-CI acceptance,
                  dev/test split, stop rule), proposers (LLM observer via API or `claude -p`,
                  random, grid, greedy/halving), journal; declared targets in the prereg,
                  one-change harness-variant sweeps.
  bench/          Task registry + adapters for local datasets with temporal/wave splits
                  (opinionqa groups, twin2k500, open_ecommerce purchases, upworthy A/B,
                  simbench, gss, bank_marketing); game tasks (econ_games_b01, ertan_wtp,
                  beebench_b01_prior); leaderboard runner.
  games/          one-shot economic game library (15 GAME dicts: 10 calibration, 5 holdout),
                  runner (`play`, cost gate, resume, retry-with-feedback, no imputation),
                  moments + cited human references + distance_to_human (agent bootstrap,
                  W1/KS), coherence audit, who-to-punish 9-criterion scorecard.
  cli.py          `simengine evaluate|card|bench|loop|demo` (integrator).
```

## Public API sketch (integrator re-exports in `simengine/__init__.py`)
```python
import simengine as se
llm = se.llm.Claude(model="claude-haiku-5-5", effort="low", cache_dir=".cache", budget_usd=5)
panel = se.persona.from_table(df, id_col="user_id")            # personas from data
sim = se.sim.survey(panel, question, options, llm=llm)         # simulated answers
frame = se.bench.get("opinionqa_groups").run(methods=[...])    # EvalFrame
report = se.evidence.evaluate(frame, ours="hybrid", raw="llm") # ladder, CIs, worth-N-humans
se.evidence.card(report, "card.md"); se.evidence.dashboard([report], "pano.html")
se.research.Loop(runner, space, proposer="claude-p", stop_after=3).run()
```

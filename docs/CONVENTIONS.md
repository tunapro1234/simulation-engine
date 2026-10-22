# API conventions

How the public API names things today, read off the signatures. Some names differ between modules because the
code was ported from several research repos; this page lists the differences so a script gets them right the
first time. It describes the current code and promises no renames. When a signature here disagrees with the code,
the code wins (`help(se.sim.survey)`), and `tests/test_docs_conventions.py` should fail.

## Where `llm` goes

| form | functions |
|---|---|
| keyword-only `llm=` | `sim.survey`, `sim.ab_test`, `sim.ab_many`, `sim.strategy`, `sim.roundtable`, `simengine.sim.roundtable.roundtable_strategy`, `simengine.sim.interview.interview`, `simengine.sim.funnel.funnel` |
| first positional argument | `sim.segment_distribution`, `sim.persona_distribution`, `sim.predict_people`, `sim.ask_constrained`, `sim.experiment`, `sim.ask_panel`, `sim.ask_many` |

```python
r = se.sim.survey(panel, questions, llm=llm)                     # keyword-only: survey(llm, ...) is a TypeError
d = se.sim.segment_distribution(llm, segments, questions)        # llm first
p = se.sim.predict_people(llm, people, items, level="L1")
```

The `llm`-first form works in every version of these functions, so use it there. Methods classes take the LLM as
their first constructor argument (`methods.LLMDirectAB(llm, reverse=True)`).

## Panel, personas, variants, scenarios

- **`personas` in signatures, "panel" in prose.** Every runner's first argument is `personas` (a list of persona
  dicts, `simengine.persona.Persona` objects or flat dicts with `id` / `text` / `segment`). The docs call that
  list a panel. They are the same thing.
- **`variants` (ab_test) vs `scenarios` (strategy).** `sim.ab_test(personas, variants, llm=...)` compares texts on a
  1-10 score; `sim.strategy(personas, scenarios, llm=...)` ranks 2+ scenarios on a 0-100 intent.
  `strategy(..., variants=...)` is a keyword-only alias of `scenarios`; pass one of the two, not both.
  `roundtable_strategy` takes `scenarios`.
- **`controls` means persona-control arms, not a control creative.** `controls=` on `survey`, `ab_test`, `strategy`
  selects the arms that check the persona is doing the work: `no_persona` (a neutral "typical adult consumer" in
  place of each persona) and `persona_swap` (each persona keeps its id and name but gets the next persona's
  content, a cyclic shift). Both are on by default. A control
  *creative* (the current page or ad) is an ordinary variant: put it in `variants` and compare against it
  (`examples/hackathon_day.md`, "N creatives against a control"). `roundtable_strategy(control="Control")` and
  `experiment(control=...)` do mean the control condition.

## Bootstrap and repeat counts

| name | meaning | where |
|---|---|---|
| `B` | bootstrap replicates (per-unit metrics, or the runner's own CIs) | `sim.*` runners (default 1000), `evidence.evaluate`, `evidence.paired_ci` |
| `B_aggregate` | bootstrap replicates for aggregate metrics (AUC, Spearman over items), default 200 | `evidence.evaluate`, `evidence.paired_ci`, CLI `--B-aggregate` |
| `n_boot` | bootstrap replicates | `methods.bayes_prior.simulate_saving_calibrated` (500), `research` loop / tree search / prompt evolution (1000) |
| `R` | Monte Carlo repeats of a subsample draw, not bootstrap replicates | `evidence.worth_humans` (400), `evidence.sample_multiplier` (40) |

## Seeds

Public functions take `seed`. The Turkish internal modules of `simengine.evidence` (`cekirdek`, `analiz`, `bedel`,
`merdiven`, `bootstrap`, `ezber`) take `tohum`, the same thing. Call the English entry points (`evaluate`,
`worth_humans`, ...) and pass `seed`.

## Default batch sizes (personas per call)

| runner | parameter | default |
|---|---|---|
| `sim.survey` | `batch` (personas), `question_group` (questions per call) | 8, 6 |
| `sim.ab_test` | `batch` | 10 |
| `sim.strategy` | `batch` | 8 |
| `sim.roundtable` | `batch` | 10 |
| `sim.persona_distribution` | `batch` | 12 |
| `sim.ab_many` | `tests_per_call` (all personas in every call) | 5 |
| `simengine.sim.interview.interview` | `batch` | 8 |
| `simengine.sim.funnel.funnel` | `batch` (each step is one `survey`) | 8 |
| `sim.ask_panel` | `batch` | 10 |
| `methods.LLMDirectAB` | `batch` (pairs per call) | 15 |

`sim.segment_distribution` makes one call per segment x question x framing (no batch).

## Person and id columns

| name | where | meaning |
|---|---|---|
| `group` | EvalFrame | cluster key for the bootstrap and the folds. `sim.survey_frame(level="person")` puts the persona id here; there is no `person` column |
| `person` | `evidence.extras.say_do(..., cluster="person")`, bench task tables | person id; `say_do` falls back to `group` when the frame has no `person` column |
| `uid` | `abm.mobility` (`uid, d, t, x, y`), flat persona dicts (one of `id` / `pid` / `persona_id` / `uid`) | user id |
| `id_col` | `persona.from_table(df, id_col=...)`, `methods.forecast` (`id_col="id"`) | the name of the id column in your table |

## Segments

- **Input `segment=`** (singular): which segment each person is in. `persona.from_table(df, segment=<column>)`, or
  a `segment` key on a persona dict. Personas without one fall in `all`.
- **Input `segments=`** (plural) on `segment_distribution` (`{key: description}` or a list) and
  `persona_distribution` (`{scheme: ...}`) names the groups to predict.
- **Output.** Runner results carry the per-segment breakdown under `segments`; `ab_test` also has
  `segment_summary`. In an EvalFrame, the `segment` column and every `seg_*` column get the per-segment error
  analysis in the evidence report.

## `unit` ids

| frame | `unit` |
|---|---|
| `sim.survey_frame(level="person")` | `<pid>|<qid>` |
| `sim.survey_frame(level="item")` | `<qid>|<segment>` |
| `sim.ab_frame` | `<test_id>|<variant>`, so `ab|<variant>` with the default `test_id="ab"` |

## Turkish internal names

The research code was Turkish. Input columns are accepted in either language (`simengine.types.COLUMN_ALIASES`),
and report keys keep the Turkish names; `se.evidence.summary(report)` gives an English flat view.

| Turkish | English |
|---|---|
| `tahmin_<method>` | `pred_<method>` (prediction column prefix) |
| `gercek`, `grup`, `madde`, `bolum` (`egitim` / `test`), `zaman`, `birim`, `tekrar` | `truth`, `group`, `item`, `split` (`train` / `test`), `time`, `unit`, `retest` |
| `taban_<name>` | baseline arm (`taban_marjinal` = marginal, `taban_rastgele` = random) |
| `kontrol_<name>` | control arm (`kontrol_karistir_<method>` = persona shuffle) |
| `manset` | headline (report key) |
| `merdiven` | baseline ladder (report key) |
| `denetim` | audits (report key; each row has `kontrol`, `durum`, `deger`, `ayrinti`) |
| `kanit_kart` | evidence card (report key) |
| `tamam` / `uyari` / `atlandi` | ok / warning / skipped (audit `durum`) |
| `tohum` | seed |
| `.onkayit/` + `zincir.jsonl` | seal registry + its hash chain (`se.seal`, = `se.evidence.seal`) |

`se.research.seal` keeps its own registry: `.prereg/` with `chain.jsonl`. `evaluate(prereg=...)` reads both chains
(a `zincir.jsonl` or `chain.jsonl` path stands for its directory).

## Module docstrings: `simengine.sim.strategy` is the function

`simengine/sim/__init__.py` re-exports the runner functions under their module names, so the attribute
`simengine.sim.strategy` is the function, and `import simengine.sim.strategy as m` binds the function too.
For the module (its docstring, its constants) go through the module table:

```python
import importlib, sys
mod = importlib.import_module("simengine.sim.strategy")     # or sys.modules["simengine.sim.strategy"]
print(mod.__doc__)
```

## Test environment

The tests need numpy, pandas, scikit-learn and scipy. A system `python3` without them fails at import with
`ModuleNotFoundError: No module named 'numpy'`. Use a virtual environment:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -t .
```

# `simengine.research`: pre-registration, sealed predictions, turn-based auto-research loop

Ported from the TALP prep code: the loop driver `uretim/dongu/kod/dongu.py` (rules `uretim/dongu/onkayit/dongu.md`
v1.0 to v1.2), the `claude -p` observer `uretim/dongu/kod/gozlemci.sh`, the bootstrap of `uretim/dongu/kod/kosucu.py`, and
the seal chain `uretim/kanit/kod/kanit/onkayit.py`. Design ideas (three splits, smoke gate, random control arm,
ablation, mini pre-registered proposals, repeat rejection) come from `arastirma/tahmin/notlar/p22-otoarastirma.md` §4.
That note looked at AIDE, AI Scientist v1/v2, Agent Laboratory, autoresearch, AIRA-dojo, OpenEvolve and others. No code
was taken from them; the note's licence check rules AI Scientist code out.

## Components

| file | what it does |
|---|---|
| `prereg.py` | `Prereg` (question, data, split, metrics, baselines, expected, falsification, stop rule) written as markdown or JSON. `lock_prereg` stores its SHA-256 and a timestamp in `<file>.lock.json` and can also commit both to git (`git_commit=True`). `verify_prereg` fails after any edit; `check_git=True` also reports the first commit time and whether the committed copy is identical. Sealed predictions: `seal` hashes a prediction table in canonical form (row and column order do not matter, floats fixed to 10 significant digits). Dicts and other files work too. Each record goes onto a hash chain with a gzip archive copy. Also `verify_seal`, `verify_chain` (detects edited records), and `grade_sealed`, which joins predictions with the truth only if they match a seal. |
| `space.py` | `Space(params, defaults, fixed, forbidden, used, complexity, check)`. Each param is a list of choices or a `(lo, hi)` range; int ends make an int range. `validate` returns the full config and a list of errors. |
| `loop.py` | `Loop` runs one change per turn on the dev split. A change is kept only if the paired bootstrap CI of its improvement over the current best excludes 0. Other statuses are discard, crash, and repeat (per-unit scores identical to an earlier turn). The loop stops after `stop_after` consecutive non-kept turns, `max_turns` turns, the `budget_usd` proposer budget, an exhausted proposer, or two invalid proposals in one turn. Finalisation has three steps. An optional `select_split` re-ranks the top 3 dev configs. An optional ablation reverts each changed key. Then the winner, the start config and the baselines run once on the test split, with 95% and Bonferroni CIs (k = number of proposal turns). The state is saved to `state.json` and can be resumed. The workdir also gets per-unit CSVs, `JOURNAL.md`, `results.tsv`, and the observer's raw prompt and reply. `control_arm(loop)` builds a random arm with the same number of turns. |
| `proposers.py` | `LLMProposer(llm)` is the observer. It sees dev metrics only and must reply with JSON: one change plus `kind`, `hypothesis`, `expected` (lower, same or higher) and `falsifier`. The loop validates the reply. An invalid reply gets one repair call that includes the rejection reason. A reply that mentions the test split is rejected. Turkish keys from the original loop are accepted. `ClaudeCodeProposer` does the same through `claude -p` (tools off, no session persistence), using `simengine.llm.ClaudeCode` if available, else a local fallback. Search proposers: `RandomProposer` (the control arm), `GridProposer`, `GreedyCoordinateProposer` (hill climbing with momentum), and `HalvingProposer`, which uses `successive_halving` to screen single changes at low fidelity on dev. `make_proposer("random"/"grid"/"greedy"/"halving"/"claude-p"/"claude"/llm)`. |
| `prereg.py`, `space.py` (additions) | Declared targets and harness robustness sweeps: see the next section. |
| `stats.py` | `paired_bootstrap_diff`, `bootstrap_mean` (seed 42, B=1000, percentile CI: the same defaults as the measured loops). |

**Runner contract** (`simengine.types.Runner`): the runner receives `cfg` with `cfg["split"]` set to `"dev"`, `"select"`
or `"test"` (key names can be changed). With `HalvingProposer` it also gets `cfg["fidelity"]`. It returns
`{"per_unit": DataFrame[unit, <primary>, ...], "metrics": {...}}`. The loop computes every number itself from the
per-unit scores; it never takes a metric the runner reports about itself. Every per-unit numeric column gets a
bootstrap CI in the journal, and the paired comparison uses the primary column on the units both runs share.

## Declared targets and harness robustness

Ported from replicant (`papers/TEMPLATE/config.py` `PAPER_FINDINGS`, declared before the run, and the condition label
of `src/replicant/cli.py` `determine_condition`). No replicant code is copied; the oTree/EDSL calls are not ported.

| function | what it does |
|---|---|
| `declare_targets(findings, prereg=None, overwrite=False)` | Normalises the findings a replication is compared against. Accepted forms: `{"metric": (value, "description")}` (the `PAPER_FINDINGS` form), `{"metric": value}`, `{"metric": {"value", "description", "tolerance", "interval", "source", "unit", "notes"}}`, or a list of dicts with `"metric"`. NaN values, unknown fields, `lo > hi` and a tolerance without a numeric value are refused. With a `Prereg` it sets `prereg.targets`; with a prereg file path it rewrites the file, and refuses if that file is already locked (`overwrite=True` rewrites it, after which the old lock fails until you re-lock). |
| `targets_sha256(targets)` | SHA-256 of the canonical targets (key order and float noise past 10 significant digits do not matter). |
| `lock_prereg` / `verify_prereg` | Unchanged API. The targets are part of the prereg file, so they are covered by the file hash. When the file is a `Prereg` with targets, the lock record also stores `targets_sha256` and `n_targets`, and `verify_prereg` returns `targets_sha256`. A prereg without targets is written byte-for-byte as before. |
| `compare_targets(targets, observed)` | One row per declared metric: `pass`/`fail` (inside `interval`, or within `tolerance`; equality for a non-numeric value), `reported` (numeric value with no tolerance or interval, so only the difference is shown), `missing` (absent, None or NaN, never filled in). An observed metric that was not declared gets its own row with status `undeclared`, so post-hoc metrics show up as post-hoc. |
| `harness_variants(wordings, max_tokens_list, temperatures, *, defaults, extra, fixed, check)` | A `HarnessSpace` (a `Space`) with keys `wording_id`, `max_tokens`, `temperature` plus any `extra` axes, e.g. `{"harness": ["edsl", "otree"]}`. The first value of each axis is the default. `wordings` can be a list of ids or `{id: text}` (`space.wording(cfg)` looks the text up, and the fingerprint covers the texts, so a resumed loop notices an edited wording). `max_tokens` takes positive ints or `None` (provider default); `temperature` takes numbers ≥ 0 or `None`. Every one-key change is a legal `Loop` turn. |
| `space.variants(base=None)` / `one_change_variants(space, base)` | The base plus every config that differs from it in exactly one key, labelled `"default"` or `"key=value"`. Works for any `Space` (ranges give `n_grid` points). Candidates that fail `space.validate` are skipped. |
| `harness_sweep(runner, space, primary, split="dev")` | Runs the base and each one-change variant once (same runner contract as `Loop`) and reports, per variant, `n_units`, `n_missing` (NaN or non-numeric unit scores, left out and never imputed), `mean`, the paired difference from the base with a 95% CI and a Bonferroni CI over the number of variants, `direction` from the Bonferroni CI, and `error` for a variant whose runner raised. `df.attrs["mean_range"]` gives the spread of the variant means. |

Why: in the replicant Ertan, Page & Putterman (2009) vignette, the same model and temperature gave a random-persona
punish-low vote of 14% through EDSL and 80% through oTree (`papers/ertan2009/results/big5_random_summary.json`,
`otree_random_summary.json`; single runs). In edsl-test the bare model punished free-riders at max_tokens 8192 but not
at the default cap (single run). A result that holds for one wording, one cap and one harness is a single draw from
this space. Lock the targets and the harness space in the prereg, then show the sweep next to the headline number.
The tests check only synthetic data (`tests/test_research_extras.py`); no new measurement is reported here.

```python
from simengine.research import Prereg, declare_targets, lock_prereg, harness_variants, harness_sweep, compare_targets

p = Prereg(title="ertan vignette", question="...", falsification="...")
declare_targets({"vote_punish_low_pct": (0.85, "late rounds"), "vote_punish_high_pct": (0.0, "never")}, p)
lock_prereg(p.write("prereg.md"))                       # lock record carries targets_sha256

space = harness_variants({"w1": "...", "w2": "..."}, [None, 8192], [0.0, 1.0], extra={"harness": ["a", "b"]})
sweep = harness_sweep(runner, space, primary="punish_low")   # runner reads cfg["wording_id"], ["max_tokens"], ...
table = compare_targets("prereg.md", {"vote_punish_low_pct": 0.80, "vote_punish_high_pct": 0.0})
```

## When it works and when it does not (measured)

Source loops: three real tasks, a Claude Haiku observer through `claude -p`, rules pre-registered before turn 0
(`uretim/dongu/onkayit/dongu.md`, `GUNLUK.md`, `turlar/<loop>/test/OZET.md`, `turlar/<loop>/<arm>/sonuclar.tsv`).
In the tables, "dev" is the split the observer climbs on, "selection" is a held-out part of dev, and "test" was run once.

**OpinionQA subgroup distributions, setting A** (`turlar/oqa_grup_A/test/OZET.md`; WD, lower is better):
- On dev, the loop found a significant gain over the pre-registered position-prior baseline. Switching to kNN gave a
  paired WD of -0.0106 [-0.0153, -0.0056]. The loop then stopped after 3 non-kept turns (5 turns in all, no invalid proposals).
- **On test the gain did not hold.** The winner scored 0.1669 [0.1590, 0.1750] against 0.1645 for the baseline. Winner
  minus baseline was +0.0025 [-0.0018, +0.0070]. The pre-registered falsification line (a) fired.
- **The random control arm was enough on selection.** LLM winner minus random winner was -0.0030 [-0.0096, +0.0032].
  On test, the LLM winner beat the random winner: -0.0062 [-0.0109, -0.0017].
- The observer's expected direction matched the outcome in 1 of 4 proposals (`turlar/oqa_grup_A/llm/sonuclar.tsv`).

**OpinionQA setting B** (`turlar/oqa_grup_B/test/OZET.md`):
- No change beat copying the known Overall distribution (dev WD 0.0530). Every move the observer tried was
  significantly worse; for example knn_sapma came out +0.0014 [+0.0006, +0.0021].
- The loop stopped at turn 3. The LLM arm, the random arm and the baseline gave the same test WD, 0.0649.
- 1 of 4 observer proposals was invalid (unparseable JSON) and was repaired.

**Twin-2K-500 kit** (`turlar/twin_kit/test/OZET.md`, loop v1.1; mean Fisher z, higher is better; the test summary
was added by the integrator after this module was written):
- LLM arm: 5 turns, 2 kept, winner `{"svd_r": 60, "knn_k": 100}`. Random arm: 1 kept, winner `{"svd_r": 63}`.
- Test z: LLM winner 0.3738 [0.3389, 0.4112], random winner 0.3720, fair baseline T-adil 0.3679, T-kit 0.2181
  (n = 122 columns).
- LLM winner − T-adil: +0.0058 [+0.0021, +0.0094] (Bonferroni k=4: [+0.0014, +0.0105]), a small real gain.
- **LLM − random winner: +0.0018 [−0.0011, +0.0046]. Random search was enough** (also on selection: +0.0017
  [−0.0010, +0.0048]).
- 3 of 7 observer proposals were invalid (unparseable JSON), all repaired; invalid+repeat rate 0.43 (falsification
  threshold 0.5).

**What this means:**
- The loop is honest machinery, not a source of gains. It finds real but small improvements when the space contains
  them.
- With about 700 dev units, repeated looks at dev can produce a gain that does not survive on test (setting A).
  That is the reason for the selection split, the single test run and the Bonferroni line.
- On these tasks the LLM observer was not clearly better than seeded random search.
- Plan for frequent invalid JSON from a small observer model; the repair attempt handles it.

**Sealed predictions in use** (`uretim/deney/RAPOR.md`, lines 144 and 259):
- In the Upworthy ek600 sample, the expected ranges were sealed before computation: 1.35x [1.15, 1.60] and
  2.45x [2.20, 2.80]. The results, 1.34x and 2.63x, fell inside them.
- In a third independent sample, 5 of 5 sealed expectations fell inside their ranges.
- A seal is the author's own timestamp. `arastirma/tahmin/notlar/karsit-p22.md` (via p22 §3) suggests also posting the
  hash somewhere external.

**Synthetic checks** (this module's tests and a scratch measurement, 22 Oct 2026). The objective is the one in
`tests/test_research_loop.py`: per-unit difficulty is shared by all configs, so the paired comparison is tight.
- **False keeps under the null.** 1000 independent one-turn loops where no config has any effect kept a change 37 times
  (3.7% per turn). The nominal one-sided rate is 2.5%; the percentile bootstrap at n=240 is slightly anti-conservative.
  Across 12 turns that adds up to roughly a 1 in 3 chance of at least one false keep, which is why the final test uses
  Bonferroni.
- **Proposers on the noisy quadratic.** The start config has a true excess loss of 0.906; lower is better.

  | proposer | runs | true excess loss of the winner | proposal turns | runner calls per run | beat the start on test |
  |---|---|---|---|---|---|
  | random | 10 seeds | mean 0.144 (range 0.043 to 0.553) | 8.3 (mean) | 13.3 (mean) | 10 of 10 |
  | grid, 5 points | 1 | 0.255 (stuck on its grid) | 10 | 14 | yes |
  | greedy coordinate | 1 | 0.004 | 12 | 17 | yes |
  | halving | 10 seeds | 0.005 | 6.1 (mean) | 52 (mean) | 10 of 10 |

  Halving spends its extra runner calls on dev screening, so it takes more looks at dev.
  On a smooth objective, greedy works best. Grid is limited by its resolution. Random is cheap but varies a lot.

## Limits
- A turn changes only the configuration. A runner whose source changes between sessions is refused on resume (its
  source fingerprint is checked). Changing the space or the start config also needs a new workdir.
- The paired CI needs the same unit ids across configs. Units missing from one run are dropped from that comparison.
  With fewer than about 50 units, CIs get wide and nearly every turn is discarded.
- The simplicity rule is on by default (`simpler_ties=True`), as in the original loop (`dongu/kod/dongu.py`): a
  candidate whose paired CI against the incumbent contains 0 and that has fewer components is kept. It only acts
  when the space defines `Space(complexity=...)`; pass `simpler_ties=False` to keep strictly-better changes only.
- The test-reference filter is a regex. It is a guard against the observer talking about test results, not proof that
  no leak happened. The loop never shows test numbers to any proposer.
- `ClaudeCodeProposer` and the `"claude"` proposer call a real model. The tests use only `FakeLLM` and a stubbed
  subprocess.

## Minimal usage
```python
import simengine as se
from simengine.research import Loop, Space, Prereg, lock_prereg, seal

Prereg(title="knn vs prior", question="Does kNN beat the position prior?", data="survey questions, wave split",
       split="dev = old waves, test = new waves", metrics=["wd (lower better)"], baselines={"prior": "method=prior"},
       expected="winner wd below prior on test", falsification="winner does not beat prior on test (Bonferroni CI)"
       ).write("prereg.md")
lock_prereg("prereg.md", git_commit=True)

def runner(cfg):                      # fixed code; reads cfg["split"]
    df = score_units(cfg)             # your task: DataFrame[unit, wd]
    return {"per_unit": df}

space = Space({"method": ["prior", "knn"], "k": (1, 200), "gamma": (0.0, 20.0)},
              defaults={"method": "prior", "k": 20, "gamma": 1.0})
loop = Loop(runner, space, proposer="claude-p", primary="wd", higher_is_better=False, stop_after=3, max_turns=12,
            baselines={"prior": {"method": "prior"}}, select_split="select", prereg="prereg.md",
            task="Predict subgroup answer distributions ...", workdir="runs/knn")
out = loop.run()                      # resume: run the same lines again
print(out["final"]["test_diff"], out["final"]["verdict"])
ctrl = se.research.control_arm(loop, "random", workdir="runs/knn-random").run(finalize=False)

seal(predictions_df, "forecast before truth", registry_dir="prereg")   # then grade_sealed(...) once truth exists
```

## Seal: research vs evidence

The top-level `se.seal` is `simengine.evidence.seal`, which writes `.onkayit/zincir.jsonl`.
`simengine.research.seal` is unchanged: it has its own schema and writes `.prereg/chain.jsonl` with English keys.
`verify_seal` and `grade_sealed` also read an evidence-format `zincir.jsonl` (written by `se.seal`) in the same
directory, or in the sibling default directory when `.prereg` holds no chain. Those records get English aliases
(`seq`, `time_utc`, `rows`, `description`, ...) and `"chain": "zincir.jsonl"`. `verify_chain` still checks only
this module's `chain.jsonl`. Example: `se.seal(df, "x"); se.research.verify_seal(df)["matched"]` is True.
`simengine.evidence.evaluate(prereg=dir)` and `evidence.chain(dir)` read both formats, so a research seal is attached
to the evidence report too:
```python
from simengine.research import seal
import simengine.evidence as ev
seal(pred_df, "forecast before truth")                          # .prereg/chain.jsonl
rep = ev.evaluate(df, ours="llm", prereg=".prereg")             # seals listed on the card
```

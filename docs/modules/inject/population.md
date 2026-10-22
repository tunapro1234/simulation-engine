# `persona.inject` family: `population`

These injectors ask about a demographic *cell* (the person's `keys` fields), not about one person. Three of them ask
for the cell's answer distribution, and one asks for a random member of the cell. A few real labels then correct
the simulated estimate. Module: `simengine/persona/inject/population.py`.

```python
import simengine.persona.inject as pi
from simengine.persona.inject import population as pop

res = pop.simulate_population(llm, people, question, "cell_verbalized", keys=["age_band", "region"])
res.dist["age_band=30-39|region=north"]          # {option: share}; one LLM call per cell
truth = pop.empirical_distribution(people, question, keys=["age_band", "region"])
pop.tvd(res.overall, truth.overall)              # Meister et al.'s metric

lab, unl = people[:50], people[50:]              # few real labels, many simulated
pop.ppi_distribution([p["answers"][question["id"]] for p in lab],
                     [res.per_person[p["id"]] for p in lab],
                     [res.per_person[p["id"]] for p in unl], question["options"])   # PPI++ estimate + CI per option

arms = [("cell_verbalized", {"keys": ["region"]}), ("cell_sample", {"keys": ["region"], "seed": 1})]
pi.compare_injectors(llm, people, questions, "answers", arms, parse=pop.parse_modal, dry_run=True)
```

All four injectors need `fields`, and every `keys` entry must be present in it. A missing field or key raises
`MissingPersonField`, and the message names both the field and the cell keys. With `keys=None` the cell is made of
all of the person's fields. The distribution injectors only accept choice questions with at least 2 options;
`cell_sample` also takes numeric questions. Members of a cell get the same distribution prompt, so a cell costs
one call; `simulate_population` sends each distinct prompt only once. The reply JSON keeps an `"answer"` field
(the modal option). As a result, `compare_injectors` can score per person with the default parser, and
`parse_modal` reads the distribution directly.

| injector | what goes in the prompt | params |
|---|---|---|
| `cell_verbalized` | Group-steering system prompt ("simulate a group of people with: ..."). The user turn asks for the share of each option as JSON percentages | `keys` |
| `cell_sequence` | The same group prompt. The user turn asks for `n` random members' answers as a JSON list. `parse_distribution` counts them | `n`=30, `keys` |
| `cell_sample` | First person: "you are respondent #j in a random sample of people with ...". j = sha256(seed, person id) mod `draws` + 1, so cell members get distinct prompts and cache entries | `draws`=30, `seed`=0, `keys` |
| `cell_fewshot` | `cell_verbalized`, plus k example questions with the cell's observed answer shares | `people`, `questions`, `k`=5, `min_n`=3, `keys` |

Helpers in the module:

- `simulate_population` returns a `PopulationResult` with these fields: `dist` (per cell), `overall`
  (people-weighted), `n`, `parsed`, `per_person`, `calls`, `prompts` and `replies`. A single-answer reply counts as
  one-hot, which is how `cell_sample` and other families' individual injectors pool.
- `empirical_distribution` gives the real shares per cell. It is the scoring target, never a prompt input for the
  same question.
- `tvd` is the total variation distance, Meister et al.'s metric.
- `ensemble` is a weighted linear opinion pool. It works over flat distributions or per-cell ones, from several
  injectors or models.
- `calibrate` maps simulated binary probabilities onto real labels with Platt or isotonic regression (scikit-learn,
  imported lazily). Fit it on people whose labels are not reused for the final estimate.
- `ppi_mean` and `ppi_distribution` are prediction-powered estimates with normal CIs. `lam=1` is PPI, `lam=0` uses
  the labels only, and `lam="auto"` is PPI++ power tuning.

Leakage: the cell prompts show no answers at all. `cell_fewshot` drops the target and its `exclude` ids from the
example pool, and it reads every member's answers through `known_answers` for the target question.

## Literature

**`cell_verbalized`, `cell_sequence`, `cell_fewshot`: Meister, Guestrin & Hashimoto 2024, *Benchmarking
Distributional Alignment of Large Language Models* (arXiv:2411.05403).**
- *Method:* the benchmark crosses three things:
  - question domains: political opinions (OpinionQA) and non-political ones such as NYT book preferences (three
    datasets in all)
  - steering: a persona description of the group, or few-shot examples of the group's real distributions on
    other questions
  - how the distribution is expressed: model log-probabilities, a sequence of 30 sampled tokens, or verbalized
    JSON
- *Few-shot selection* (their A.8): the 10 questions most similar to the target by SimCSE embedding, then the 5
  whose distributions differ most. `cell_fewshot` follows this as 2k by word overlap, then k by max-min TVD.
  Word overlap stands in for embeddings, which is a deviation.
- *Published effect* (Table 1a, mean TV over datasets, steering methods and groups; lower is better):

  | model | verbalized | sequence (30) | log-p |
  |---|---|---|---|
  | Opus | 0.226 | 0.325 | n/a |
  | GPT-4 | 0.229 | 0.278 | 0.550 (0.273 temperature-scaled) |
  | Llama 3 70B | 0.244 | 0.328 | 0.495 |

  - The uniform baseline is 0.363. Verbalizing was best for every model.
  - The "knowledge-to-simulation gap" (Table 1b): sequence sampling had 9% (GPT-3.5) to 44% (Opus) higher TV than
    verbalizing.
  - Few-shot steering beat persona steering for humans and for all models except GPT-3.5 (Fig. 4; no table
    value).
  - The human baseline on OpinionQA + NYT is 0.250 TV. GPT-4 verbalized scored 0.204 on the same subset (Table 2).

**`cell_sample`: Argyle et al. 2023, *Out of One, Many* (arXiv:2209.06899); seeded draws over Meister et al.**
- *Method:* silicon sampling conditions GPT-3 on each real participant's first-person demographic backstory from
  US surveys (ANES), then compares the sample's answer distribution with the human one.
- *How this injector differs:* it uses the *cell's* attributes plus a respondent index instead of each person's
  own full backstory. Members of one cell are therefore exchangeable draws, which matches the cell-level target.
  For per-person backstories use `plain_fields` or the narrative families. Use temperature > 0 through `llm_kw`
  for draw diversity.
- *Published effect:* Argyle et al. report high "algorithmic fidelity", e.g. vote-choice agreement with ANES. We
  quote no number here, because the effect depends on the survey and the measure.

**Estimators: Angelopoulos, Bates, Fannjiang, Jordan & Zrnic 2023, *Prediction-Powered Inference*
(arXiv:2301.09633); Angelopoulos, Duchi & Zrnic 2023, *PPI++* (arXiv:2311.01453).**
- *PPI:* the estimate is mean(y) + mean(f_unlabelled) − mean(f_labelled). It is unbiased whatever the simulator's
  bias. Its CI is narrower than labels alone when the simulated answers correlate with the real ones.
- *PPI++:* it scales the correction by λ = Cov(y, f) / ((1 + n/N) Var f). The paper guarantees intervals that are
  asymptotically never wider than the classical ones.
- *Published effect:* the interval-width gains are task-specific and are not quoted here. `ppi_mean` returns
  `width_ratio` so that each run reports its own.

**Related, not implemented here: Suh et al. 2025, SubPOP (arXiv:2502.16761).** They fine-tune LLMs on 3,362
survey questions and 70K subpopulation-response pairs to predict response distributions directly. They report up
to a 46% smaller LLM-human gap than prompt-steering baselines, which is the role `cell_verbalized` plays here.
Fine-tuning belongs in the `finetune` family. The distribution parser and `tvd` here can score such a model.

## Measured in simengine

Not yet measured in simengine. The tests run on `FakeLLM` only. They check rendering, the needs and keys errors,
seeded determinism, one call per cell, leakage, parsing, and the estimators' maths, including PPI CI coverage on
synthetic data. They are not an effect size.

## Caveats

- **Cell size:** the persona is the cell, so individual variation inside a cell is invisible to the
  distribution injectors. Per-person accuracy from `compare_injectors` is the cell's modal answer and is capped
  by the cell's majority share. Use TVD on `simulate_population` for the population claim.
- **Overlapping labels:** `calibrate` and PPI need labelled people that are not reused elsewhere. The PPI CI
  assumes the labelled people are a random subset of the population.
- **`ppi_distribution` CIs:** they are marginal per option, not simultaneous, and the point shares need not sum
  to 1.
- **Public data:** public panels such as OpinionQA may be in a model's training data. Memorisation is not tested.

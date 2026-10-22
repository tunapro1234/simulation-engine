# `simengine.persona.inject.values`

Value and culture profiles rendered as text: Schwartz basic values, Moral Foundations, Hofstede country
dimensions, and cultural prompting with optional Inglehart-Welzel map coordinates.

All scores are inputs. The module ships no country table: Hofstede Insights and WVS/IVS scores carry their own
terms of use. Callers bring the scores and cite the source.

```python
import simengine.persona.inject as pi

person = {"id": "u1", "country": "Chile",
          "schwartz": {"benevolence": 5.2, "power": 2.1, "self_direction": 4.8},   # PVQ means, 1-6
          "mfq": {"care": 4.1, "fairness": 3.9, "loyalty": 2.0, "authority": 2.2, "sanctity": 1.5},
          "hofstede": {"pdi": 63, "idv": 23},                                      # caller-supplied
          "ivs_map": {"traditional_secular": -0.4, "survival_self_expression": 0.6}}
arms = ["schwartz_scores", pi.get_injector("schwartz_anchor", k=2), "moral_foundations",
        pi.get_injector("hofstede_country", scores=True), "cultural_prompting"]
plan = pi.compare_injectors(llm, people, questions, truth, arms, dry_run=True)
```

## Person fields

Keys inside the score dicts are case-insensitive, and `_` or a space counts as `-` (`self_direction` =
`self-direction`). Common aliases are accepted (`purity` -> `sanctity`, `ingroup` -> `loyalty`,
`individualism` -> `idv`, `y`/`x` for the two map axes).

| field | type | read by |
|---|---|---|
| `schwartz` | `{value: score}`, or a value code / list of codes (`schwartz_anchor` only) | `schwartz_scores`, `schwartz_anchor` |
| `mfq` | `{foundation: score}` (care, fairness, loyalty, authority, sanctity, liberty) | `moral_foundations` |
| `country` | str; optional `birth_country` | `hofstede_country`, `cultural_prompting` |
| `hofstede` | `{pdi, idv, mas, uai, lto, ivr: score}` | `hofstede_country(scores=True)` |
| `ivs_map` | `{traditional-secular, survival-self-expression: coordinate}` | `cultural_prompting(map=True)` |

Errors:
- A missing field raises `MissingPersonField`, and the message lists the arm's `needs`. A score dict whose
  values are all `None` raises it too.
- An unknown key, a non-number (including bool or NaN), or a score outside `scale` raises `ValueError` naming
  the person and the field.

Leakage: none of these injectors shows the person's answers. Each reads only the fields above, so
`known_answers` is not needed.

Determinism: ties in a ranking, `order="shuffle"` and `variant="random"` use a hash of (`seed`, person id), so a
rerun renders the same text.

## Injectors

| name | params (default) | paper |
|---|---|---|
| `schwartz_scores` | `definitions=True`, `scale=(1, 6)`, `order="paper"` (`rank`, `shuffle`), `seed=0` | Kang et al. 2023, arXiv:2310.17857 |
| `schwartz_anchor` | `k=1`, `seed=0` | Rozen et al. 2024, arXiv:2407.12878 |
| `moral_foundations` | `style="statements"` (`scores`), `k=1`, `definitions=True`, `scale=(0, 5)`, `seed=0` | Abdulhai et al. 2023, arXiv:2310.15337 (MFQ: Graham et al. 2011) |
| `hofstede_country` | `scores=False`, `advice=False` | Kharchenko et al. 2024, arXiv:2406.14805 (dimensions: Hofstede 2010) |
| `cultural_prompting` | `variant=0` (0-9 or `"random"`), `seed=0`, `map=False` | Tao et al. 2024, arXiv:2311.14096 (map: Inglehart & Welzel 2005) |

### `schwartz_scores`

What the paper did: Kang et al.'s prompt baselines (LLaMA_Short / LLaMA_Long / ChatGPT_Long; Tables 22-23)
asked the model to roleplay a person described by a score on each of the ten Schwartz values on the PVQ 1-6
scale. The Long variant adds each value's definition. Their main method, VIM, fine-tunes LLaMA on argument
generation and question answering. The fine-tuning is not part of this injector; see the `finetune` family.

What this injector renders: the system prompt is "Let's roleplay..." with the scale, optional definitions
(`definitions=False` gives the Short variant), "- Value: score" lines, and "As this person, please answer". The
user turn is the question block. `order="rank"` or `"shuffle"` changes the order of the score lines, which is
an order-sensitivity check.

Published effect: these prompt baselines are the paper's comparison arms, not its method. Value survey (PVQ),
average NMSE over 128 target profiles, where lower is better: LLaMA_Short 0.196, LLaMA_Long 0.189, ChatGPT_Long
0.146, against fine-tuned ViLLaMA 0.034 (Table 2). ESS opinion prediction, average NMSE: LLaMA_Long 0.137,
ChatGPT_Long 0.169, ViLLaMA 0.099 (Table 4). ChatGPT_Long refused about 18% of ESS questions (Table 11), so its
ESS score is not comparable. Source: arXiv:2310.17857.

### `schwartz_anchor`

What the paper did: Rozen et al. compared five ways to prompt an LLM for a PVQ-RR answer. In the "Value Anchor"
prompt, the system says "Answer as a person that is [value]." and uses the paper's Appendix E text for one of
the 19 refined values.

What this injector renders: the person's top-`k` values, highest score first, joined into one anchor sentence.
`person["schwartz"]` may also be a code or a list of codes. The 19 refined values use the paper's texts (from
`simengine.persona.methods.value_anchor.ANCHORS`). Four of the ten basic values (stimulation, hedonism,
achievement, tradition) are not split in the refined model, so they also keep the paper's texts. The other six
basic values use simengine wording in the same template. `k > 1` is a simengine extension: the paper anchors
one value per persona.

Published effect: the paper measured value structure, not individual prediction. As the sum of squared
differences between the LLM's and the humans' MDS value circle, where lower is better, Value Anchor scored
0.10-0.23 against 0.42-0.92 for the basic prompt. It was best in 11 of 12 model x temperature rows (GPT-4,
Gemini Pro, Llama 3.1 8B/70B, Gemma 2 9B/27B; Table 1). No confidence intervals were reported. Source:
arXiv:2407.12878.

### `moral_foundations`

What the paper did: Abdulhai et al. searched for prompts that maximize each MFQ foundation in GPT-3 DaVinci2
and reported the winning sentence per foundation (Table 2). The paper has no liberty sentence. Its sanctity
sentence reads like hierarchy, and the paper itself calls this counterintuitive.

What this injector renders:
- `style="statements"` (default): the paper's sentence, verbatim, for the person's top-`k` foundations.
  Foundations without a sentence (liberty) are skipped. If only liberty is present, it raises `ValueError` and
  suggests `style="scores"`.
- `style="scores"`: every MFQ relevance score, on a 0-5 scale by default, in Kang et al.'s roleplay template.
  This is a simengine adaptation; neither paper ran it.

Published effect: in the paper's downstream donation task (GPT-3, 20 runs per prompt), the mean donation was
92.66 with no prompt. It was 144.87 with the loyalty sentence, 112.45 with sanctity, 108.07 with fairness, 97.71
with authority and 88.09 with care (Table 2). This shows the sentences move behaviour. It is not a measure of
how well they predict a person. Source: arXiv:2310.15337.

### `hofstede_country`

What the paper did: Kharchenko et al. asked advice questions built on five Hofstede dimensions. The system
prompt was "Answer the following prompt by imagining that you are proudly and thoroughly from the country of
<country>, providing appropriate advice." The paper covered 36 countries and compared this persona approach with
asking in the country's language.

What this injector renders: the same sentence, with the ", providing appropriate advice" tail only when
`advice=True`, since most simengine questions are survey items, not advice requests. `scores=True` adds the
caller's Hofstede scores with each dimension's poles and then also needs `hofstede`. This is a simengine
extension: the paper names the country only.

Published effect: the correlation between a country's Hofstede score and the share of advice favouring that
side was 0.39 (individualism), 0.19 (masculinity) and 0.39 (uncertainty avoidance) with GPT-4 personas. For
power distance it was negative (-0.49), and for long-term orientation it was near zero. Personas gave a
significant positive correlation in 17 of 25 model x dimension cells, against 4 of 25 for language-only
prompting (Table 2). The paper's conclusion is that models know the values but often do not act on them. Source:
arXiv:2406.14805.

### `cultural_prompting`

What the paper did: Tao et al. set the system prompt to "You are an average human being born in [country] and
living in [country] responding to the following survey question." and tested ten respondent descriptors
(Table 1). Replies to IVS items were placed on the Inglehart-Welzel map and scored by distance to the
country's position.

What this injector renders: the same sentence. `variant` picks the descriptor; `"random"` picks one per person
from the seed. `birth_country`, when given, fills the "born in" slot. `map=True` adds the caller's
Inglehart-Welzel coordinates and a one-paragraph explanation of the two axes, and then also needs `ivs_map`.
This is a simengine extension: the paper used the map only to score replies, never in the prompt.

Published effect: for GPT-4, GPT-4-turbo and GPT-4o, cultural prompting improved cultural alignment for 71-81%
of countries and territories (abstract). Source: arXiv:2311.14096.

## Measured in simengine

Not yet measured in simengine. The tests use `FakeLLM` only and check the paper templates, missing-field errors,
score validation, seeded ties and variants, no target-answer leakage, the swap control and the compare plumbing.
They are not an effect size.

## Caveats

- **Group-level papers:** Rozen, Abdulhai, Kharchenko and Tao measure group-level structure or shift, not
  individual prediction. Only Kang et al. score individual-profile prediction, and their best numbers come from
  fine-tuning, not from the prompt this family renders.
- **Score sources:** Hofstede and IVS country scores are not shipped. Check the licence of the table you use,
  and cite it in your run notes.
- **Scales:** `schwartz_scores` checks against `scale`. Centred PVQ scores (negative) need
  `scale=(-5, 5)` or similar. Hofstede scores are accepted from 0 to 120, because some published scores exceed 100.

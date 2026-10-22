# bench tasks `subpop_opinionqa` and `global_opinionqa`: group answer distributions

`simengine/bench/tasks/subpop.py` scores predicted answer distributions of population groups against survey data, one
row per (question, group):

- **`subpop_opinionqa`**: the OpinionQA test set as shipped by SubPOP (Suh et al. 2025, arXiv:2502.16761): Pew American
  Trends Panel questions (waves W26-W92) with weighted answer shares for 22 US groups (region, education, sex,
  ideology, income, party, race, religion), or for all OpinionQA groups with `groups="61"`.
- **`global_opinionqa`**: GlobalOpinionQA (Durmus et al. 2023, arXiv:2306.16388): Pew Global Attitudes and World
  Values Survey questions with answer shares per country.

Both follow the `BenchTask` conventions in `simengine/bench/base.py` (`layout`, `load`, `baselines`, `frame`,
`metrics`). They are not in the package registry yet: import the module directly. Importing it registers both names.

`opinionqa_groups` (`opinionqa.py`) is the related wave-out task on the larger OpinionQA pool with `position` and
TF-IDF `knn` baselines. This module adds the SubPOP metric exactly as SubPOP computes it, GlobalOpinionQA, a
group-held-out setting and the `nearest_group` baseline. It reads only SubPOP's `opinionqa.csv`, not the OpinionQA
pool files.

## API

```python
from simengine.bench.tasks import subpop as sp

task = sp.SubPOPOpinionQA()                 # kind distribution; metrics wd (primary), tvd, js_sim
data = task.load("<catalogue>")             # or $SIMENGINE_DATA; the directory holding subpop-opinionqa/
frame = task.run(data=data)                 # EvalFrame with uniform, overall, nearest_group (+ methods={...})
task.score(frame, "nearest_group")          # {'metric': 'wd', 'value', 'lo', 'hi', 'n': questions}
task.score_all(frame)
task.others(data, item, group)              # the rows a setting-B method may read for (item, group)

g = sp.GlobalOpinionQA()                    # metrics tvd (primary), js_sim; directory globalopinionqa/
```

| option | values (default) | tasks |
|---|---|---|
| `split` | `"test"` \| `"dev"` (`"test"`) | both |
| `setting` | `"A"` \| `"B"` (`"B"`) | both |
| `test_share` | float in (0, 1) (0.5) | both |
| `min_shared` | int (5): shared train questions needed before a group can be someone's nearest group | both |
| `groups` | `"22"` \| `"61"` (`"22"`) | subpop_opinionqa |

An unknown option key raises `ValueError`, as in every bench task.

Other public names: `ordinal_wd`, `tvd`, `js_distance`, `pairwise` (distance matrix among the groups of one question),
`load_subpop_opinionqa`, `load_global_opinionqa`, `split_items`, `group_distances`, `loo_marginal`, `nearest_group`,
`OpinionDistTask` (shared base), and the metrics `WD`, `TVD` and `JS_SIM`.

### Files

```
<root>/subpop-opinionqa/data/opinionqa/processed/opinionqa.csv            qkey, attribute, group, responses, ordinal, question, options
<root>/subpop-opinionqa/data/subpopulation_metadata/demographics_22.csv  attribute, group (list)   (demographics_61.csv for groups="61")
<root>/globalopinionqa/data/global_opinions.csv                          question, selections, options, source
```

The SubPOP files are the `data/` folder of github.com/JosephJeesungSuh/subpop. The gated HF SubPOP-Train/Eval sets
are not needed. GlobalOpinionQA is the HF dataset `Anthropic/llm_global_opinions`. Its `selections` column is a
`defaultdict(...)` repr, which is parsed as such.

Loading rules:

- **SubPOP:** `responses` already exclude "Refused", so `Refused` is dropped from `options`. A qkey asked with several
  question texts (in the real file, `IDIMPORT_W43` with 5 texts) becomes items `<qkey>#<k>`. In that case all
  variants share one Overall row, as the file gives it. Questions whose ordinal values are all equal are dropped,
  because SubPOP scores them NaN (8 of 494 qkeys).
- **GlobalOpinionQA:** the item id is `<GAS|WVS>:<sha256(question)[:12]>`, and group = country (`attribute =
  "country"`). Rows whose shares sum to 0 are dropped (Durmus: only countries with n_c|q > 0). Other rows are
  renormalised; most sum to 1 within 0.01, but the range is 0.011 to 1.072 (5 positive rows are more than 0.01 off). Numeric option labels are turned into strings. All options,
  including DK/Refused, are kept as categories.
- **Both:** only questions answered by at least two groups are used (on the real GlobalOpinionQA, 1,707 of 2,556).

### Split and settings

Questions are hashed with sha256 (`opinion_dist-v1:<item>`) into test (`test_share`, default 50%) and train, so all
groups of a question fall on one side. `split="dev"` removes the test questions and hashes the rest again with
another salt, so dev never reads a test question.

- **Setting A:** a method may use the train questions (all groups) and the target question's text and options.
  `overall` is `None`, and the only baseline is `uniform`. This is the SubPOP evaluation setting.
- **Setting B (default, group held out):** for target (q, g), a method may also read q's population marginal
  (`data.test["overall"]`) and the other groups' rows on q (`task.others(data, q, g)`), but never g's own row.
  `data.test` holds the truth column `dist`, so this rule is a contract, as in `opinionqa_groups`. The marginal is
  Pew's weighted Overall row for SubPOP (it includes g's respondents). For GlobalOpinionQA it is the unweighted mean
  of the other countries on q, since the file has no world row.

### Metrics

- **`wd`** (SubPOP primary, lower is better): SubPOP's `subpop.utils.survey_utils.ordinal_emd`. It is the
  Wasserstein-1 distance between the distributions placed on the hand-made OpinionQA ordinal values (sorted, ties
  allowed, e.g. `[1, 2, 1.5]`), divided by max - min. Options with a negative ordinal are dropped and the rest
  renormalised. If all ordinals are equal the value is NaN and the row is skipped. Rows are averaged over all
  (question, group) pairs, as in SubPOP Table 1 ("averaging over all questions and subpopulations"). The tests check
  the function against the two examples in the `ordinal_emd` docstring. In example 2 the code renormalises, which
  gives 0.1/0.7, not the docstring's 0.1; this module follows the code.
- **`tvd`** (GlobalOpinionQA primary, lower is better): total variation distance 0.5·Σ|p − q|.
- **`js_sim`** (higher is better): 1 − Jensen-Shannon distance, the similarity used by Durmus et al. 2023 (sec. 2.3).
  The paper does not state the log base. Base 2 is used here, so the value lies in [0, 1]. The tests check it against
  `scipy.spatial.distance.jensenshannon(base=2)`. Durmus average per country over questions. This module averages
  over all (question, country) rows; per-country scores come from grouping the frame by `seg_group`.

The 95% CIs come from a question-cluster bootstrap (`cluster_bootstrap`, B = 1000, seed 42), and `n` is the number
of questions. EvalFrame columns: `unit` (`item|attribute:group`), `item`, `group` (= item, the bootstrap cluster),
`segment` (attribute), `seg_group`, `seg_source` (ATP wave or GAS/WVS), `truth`, `ordinal`, `split`.

### Baselines

- `uniform`: 1/k over the options. SubPOP calls this the "upper bound".
- `overall` (setting B): the question's population marginal, as defined above.
- `nearest_group` (setting B): predict (q, g) with q's distribution of the group h ≠ g that is closest to g on the
  train questions. Closeness is the mean `wd` (SubPOP) or `tvd` (GlobalOpinionQA) over the train questions both
  groups answered, with at least `min_shared` of them; ties are broken alphabetically. If no such h answered q, the
  prediction falls back to `overall`. The fallback count is stored in `data.meta["nearest_group_fallbacks"]`.

## Limitations

- **Setting B contract:** setting B is a contract, not enforced. `data.test` carries every group's truth, so a method
  must read only `others()` and `overall`. For SubPOP, the Overall row includes the target group's own respondents,
  so `overall` leaks a little of the target, by design of the setting.
- **Split:** the hashed 50/50 question split is this module's own choice. SubPOP uses all of OpinionQA for evaluation
  and trains on other ATP waves (SubPOP-Train), which is not available locally (gated). Numbers on half the questions
  are not the paper's full-set numbers.
- **IDIMPORT_W43:** the five question texts share one Overall row in the file while their group rows differ.
  `opinionqa_groups` keys by qkey alone and keeps only the last text's group rows for this qkey; this module keeps
  all five as separate items.
- **JS log base:** Durmus et al. do not state the base. Base e would give distances × sqrt(ln 2) ≈ 0.833 of these.
- **GlobalOpinionQA marginal:** the country labels include "(Non-national sample)" variants, which are treated as
  separate countries. The marginal is an unweighted country mean, not population-weighted.
- **Licences:** SubPOP data come from Pew ATP terms (research use, attribution, no re-identification). GlobalOpinionQA
  is CC-BY-NC-SA-4.0. No data rows are in the repository.

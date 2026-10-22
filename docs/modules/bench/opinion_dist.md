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
TF-IDF `knn` baselines. This module adds the SubPOP metric as SubPOP computes it, GlobalOpinionQA, a
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
task.others(data, item, group, attribute)   # the rows a setting-B method may read for (item, group)

g = sp.GlobalOpinionQA()                    # metrics tvd (primary), js_sim; directory globalopinionqa/
```

| option | values (default) | tasks |
|---|---|---|
| `split` | `"test"` \| `"dev"` (`"test"`) | both |
| `setting` | `"A"` \| `"B"` (`"B"`) | both |
| `test_share` | float in (0, 1) (0.5) | both |
| `min_shared` | int (5): shared train questions needed before a group can be someone's nearest group | both |
| `peers` | `"blocks"` \| `"all"` (`"blocks"`): which other groups setting B may read (see below) | both |
| `groups` | `"22"` \| `"61"` (`"22"`) | subpop_opinionqa |

An unknown option key raises `ValueError`, as in every bench task.

Other public names: `ordinal_wd`, `tvd`, `js_distance`, `pairwise` (distance matrix among the groups of one question),
`load_subpop_opinionqa`, `load_global_opinionqa`, `split_items`, `group_distances`, `loo_marginal`, `nearest_group`,
`OpinionDistTask` (shared base), the metrics `WD`, `TVD` and `JS_SIM`, and `COLUMNS` (the long table, with a
`block` column: the attribute for SubPOP, the country without "(Non-national sample)" for GlobalOpinionQA).

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
  (`data.test["overall"]`) and the rows on q of the groups outside g's block (`task.others(data, q, g, attribute)`),
  never g's own row nor its block mates. `data.test` holds the truth column `dist`, so this rule is a contract, as in
  `opinionqa_groups`. The marginal is Pew's weighted Overall row for SubPOP (it includes g's respondents). For
  GlobalOpinionQA it is the unweighted mean of the other blocks' rows on q, since the file has no world row.
- **Why blocks:** Pew's Overall row is the weighted sum of the groups of any complete attribute (the weights can be
  fitted on train questions), so with g's block mates readable, g follows by subtraction. A "complement" method
  (non-negative least-squares weights per attribute on train, then (Overall − Σ mates·w)/w_g) measured on the test
  split under the old all-groups view: CREGION wd 0.0042 vs `nearest_group` 0.0284, AGE 0.0152 vs 0.0402, EDUCATION
  0.0158 vs 0.0369 with `groups="61"`; CREGION 0.0231 vs 0.0318 even with the 2 of 4 regions in the 22 groups. In
  GlobalOpinionQA, 26 countries also have a "(Non-national sample)" row, both on up to 91 questions, so a country was
  predicted from its own second sample. `peers="all"` restores the all-groups view for comparison.

### Metrics

- **`wd`** (SubPOP primary, lower is better): SubPOP's `subpop.utils.survey_utils.ordinal_emd`. It is the
  Wasserstein-1 distance between the distributions placed on the hand-made OpinionQA ordinal values (sorted, ties
  allowed, e.g. `[1, 2, 1.5]`), divided by max - min. Options with a negative ordinal are dropped and the rest
  renormalised. If all ordinals are equal the value is NaN and the row is skipped. Rows are averaged over all
  (question, group) pairs, as in SubPOP Table 1 ("averaging over all questions and subpopulations"). The tests check
  the function against the two examples in the `ordinal_emd` docstring and against a line-by-line port of its code
  on random cases with tied and negative ordinals. In example 2 the code renormalises, which gives 0.1/0.7, not the
  docstring's 0.1; this module follows the code. One deviation: a prediction with no mass on the ordered options (or
  a NaN) is scored as uniform here. SubPOP's code raises inside `list_normalize`, catches it and returns NaN, which
  would drop the row; that would let a method skip hard rows. A prediction of the wrong length raises `ValueError`.
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

## Measured numbers

These are from runs on the local copies (22 Oct 2026, `nice -n 19`, after the review that made `peers="blocks"`
the default). The script ran the five configurations below, each with the default `peers="blocks"` and with
`peers="all"`, in 61.2 s wall time with 188 MB peak RSS. Every value is shown as `value [95% CI]`. Command, from the
repository root:

```python
from simengine.bench.tasks import subpop as sp
cfg = [(sp.SubPOPOpinionQA, {}), (sp.SubPOPOpinionQA, {"groups": "61"}), (sp.SubPOPOpinionQA, {"split": "dev"}),
       (sp.GlobalOpinionQA, {}), (sp.GlobalOpinionQA, {"split": "dev"})]
for T, opts in cfg + [(T, {**o, "peers": "all"}) for T, o in cfg]:
    task = T(**opts); data = task.load("<catalogue>")
    print(task.score_all(task.run(data=data)))
```

**subpop_opinionqa, test split, 22 groups, setting B** (237 test questions, 253 train questions, 5,211 rows,
0 fallbacks; identical with `peers="all"`, no nearest group was a block mate):

| baseline | wd | tvd | js_sim |
|---|---|---|---|
| uniform | 0.1815 [0.1722, 0.1907] | 0.2718 [0.2608, 0.2822] | 0.7198 [0.7090, 0.7309] |
| overall | 0.0646 [0.0617, 0.0680] | 0.1044 [0.1009, 0.1081] | 0.8899 [0.8863, 0.8933] |
| nearest_group | **0.0506** [0.0480, 0.0531] | **0.0868** [0.0836, 0.0901] | **0.9064** [0.9033, 0.9097] |

- **`groups="61"`** (56 of the listed groups occur in the file; 13,247 rows): uniform 0.1784, overall 0.0588
  [0.0563, 0.0613], nearest_group 0.0487 [0.0465, 0.0510] (wd); `peers="all"`: nearest_group 0.0484
  [0.0462, 0.0507].
- **`split="dev"`** (130 dev and 123 train questions): uniform 0.1823, overall 0.0634, nearest_group 0.0487
  [0.0455, 0.0523] (wd); `peers="all"`: nearest_group 0.0496 [0.0464, 0.0532].
- **Uniform check:** uniform `wd` over all 490 items × 22 groups (10,775 rows, no split) is 0.1775. SubPOP Table 1
  reports 0.178 for the uniform upper bound on OpinionQA. The `TestRealData` test checks this agreement when
  `$SIMENGINE_DATA` is set.

**global_opinionqa, test split, setting B** (860 test questions, 847 train questions, 22,487 rows, 138 country labels,
7 fallbacks):

| baseline | tvd | js_sim |
|---|---|---|
| uniform | 0.4031 [0.3888, 0.4164] | 0.5819 [0.5669, 0.5975] |
| overall | 0.1787 [0.1738, 0.1840] | 0.8080 [0.8023, 0.8134] |
| nearest_group | **0.1602** [0.1564, 0.1639] | **0.8262** [0.8224, 0.8301] |

With `peers="all"` (a country may read its own non-national sample; 3 fallbacks): overall 0.1786 [0.1737, 0.1839],
nearest_group 0.1591 [0.1553, 0.1627] tvd, 0.8273 js_sim.

`split="dev"` (426 questions): uniform 0.3995, overall 0.1787, nearest_group 0.1621 [0.1566, 0.1673] (tvd);
`peers="all"`: overall 0.1786, nearest_group 0.1610 [0.1555, 0.1661].

The leak check under **Why blocks** (complement method, per-attribute wd on the test split) is a separate local run
of the same date, not part of the script above.

Literature numbers for comparison (not measured here). From SubPOP Table 1 on OpinionQA, 22 groups, setting A:
- human sampling lower bound 0.031;
- zero-shot prompting 0.138 to 0.209;
- SubPOP fine-tuned models 0.094 to 0.106 `wd`.

Setting B baselines read the target question's other groups, so they are not comparable with those setting-A
numbers. Under setting A, uniform (0.18) is the only baseline here; see `opinionqa_groups` for train-only baselines.

## Limitations

- **Setting B contract:** setting B is a contract, not enforced. `data.test` carries every group's truth, so a method
  must read only `others()` and `overall`. For SubPOP, the Overall row includes the target group's own respondents,
  so `overall` leaks a little of the target, by design of the setting.
- **Split:** the hashed 50/50 question split is this module's own choice. SubPOP uses all of OpinionQA for evaluation
  and trains on other ATP waves (SubPOP-Train), which is not available locally (gated). Numbers on half the questions
  are not the paper's full-set numbers.
- **Sibling questions cross the split:** items of one Pew battery (e.g. `RACESURV47a`..`f`, same wave) are hashed
  one by one, so a setting-A method can copy a train sibling. Measured (local run, 22 Oct): with batteries taken as
  the qkey minus trailing lowercase letters, 85 of the 237 test questions have a train sibling; copying the group's
  mean sibling distribution gives wd 0.130 on those 1,870 rows vs 0.172 for uniform. Uppercase-suffix batteries
  (`REASONGUNA`..`E`) are not caught by that rule, so the true overlap is larger. `opinionqa_groups` holds out whole
  waves if this matters.
- **IDIMPORT_W43:** the five question texts share one Overall row in the file while their group rows differ.
  `opinionqa_groups` keys by qkey alone and keeps only the last text's group rows for this qkey; this module keeps
  all five as separate items.
- **JS log base:** Durmus et al. do not state the base. Base e would give distances × sqrt(ln 2) ≈ 0.833 of these.
- **GlobalOpinionQA marginal:** the country labels include "(Non-national sample)" variants, which are scored as
  separate rows but share their country's block. The marginal is an unweighted mean over the other blocks' rows (a
  country with two samples counts twice), not population-weighted.
- **Licences:** SubPOP data come from Pew ATP terms (research use, attribution, no re-identification). GlobalOpinionQA
  is CC-BY-NC-SA-4.0. No data rows are in the repository.

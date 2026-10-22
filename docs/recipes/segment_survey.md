# Recipe: survey answer distributions by segment

Script: [`examples/recipes/segment_survey.py`](../../examples/recipes/segment_survey.py) (under 200 lines, numbered
steps 0-10). Test: `tests/test_recipe_segment_survey.py`.

```bash
python3 examples/recipes/segment_survey.py                                   # offline: FakeLLM + synthetic survey, ~3 s
python3 examples/recipes/segment_survey.py --data data/survey.csv --out runs/survey
python3 examples/recipes/segment_survey.py --live --max-usd 1 --data data/survey.csv   # Claude Haiku, key from $ANTHROPIC_API_KEY
python3 examples/recipes/segment_survey.py --live --key-file <path> --max-usd 1        # or the key from a file
```

Outputs go to `--out` (default `runs/segment_survey`, git-ignored): `card.md`, `dashboard.html`, the seal registry
`.onkayit/` and, with `--live`, the reply cache and cost ledger under `.cache/`.

## When to use

- The brief asks "how would segment X answer these questions?" or "what share of the population agrees?", with
  closed-ended questions (Likert, yes/no/maybe, pick one) **and** some real answers (even a few hundred respondents).
  The real answers are split by respondent: a small *train* survey gives the baselines and the PPI++ labels, the
  *holdout* is the truth the card grades against.
- The target is a **population or segment distribution**, not one person's answer. This is where an LLM is
  measurably useful: on CES 2020 segments a direct segment distribution closed 0.818 of the gap between the general
  marginal and a real 37-person survey of the segment (W1 0.077); a persona panel closed only 0.55
  (`docs/RESULTS.md` §1).
- Not for: predicting individual respondents (per-person accuracy is capped by the cell's majority share; there the
  LLM lost to models without an LLM in our talp73 runs), open-ended text answers, or effect sizes of an intervention
  (use the A/B recipes; LLM effects are inflated).

## Inputs

`--data` is a CSV with **one row per (respondent, question answered)**.

| column | type | meaning |
|---|---|---|
| `respondent_id` | str | person id; the bootstrap cluster; one respondent is never split across train and holdout |
| `segment` | str | segment key (`young_urban`, `party x age`, ...) |
| `segment_desc` | str | segment description given to the LLM; the first value per segment is used; **no answer information** |
| `question_id` | str | `q1`, `q2`, ... |
| `question_text` | str | the question as asked; first value per question |
| `options` | str | answer options in their natural order, `|`-separated (`Strongly disagree|...|Strongly agree`); first value per question. The order matters for W1 |
| `answer` | str | the option chosen; must be one of `options` (the script stops and counts the bad rows otherwise) |
| `split` | str, optional | `train` / `holdout`; without it 30 % of respondents (stable crc32 hash of the id) are train |

Without `--data` the script builds a synthetic survey: 4 segments of very different sizes (300 / 220 / 120 / 60), 5
Likert questions, segment-specific tilts. The fake LLM knows four of the tilts at 70 % strength and misses one
question entirely (the budget worry, q5), so the card shows an LLM that is useful but not complete.

## Steps (what each one does, what to change on the day, what backs it)

| step | in the script | change on the day | backed by |
|---|---|---|---|
| 0 | `POPULATION`, `SEGMENTS`, `QUESTIONS`, `TRAIN_SHARE`, `K_PRIOR` | the population line from the brief ("adults in Turkey who shop online"); `K_PRIOR` before you look at the holdout | literature rank #4 (Meister et al., arXiv:2411.05403): ask for the group's distribution, not for persona role play |
| 1 | load, respondent-disjoint split, option check | a split the brief fixes (e.g. a later wave as holdout) | person-disjoint split = the leakage guard of every result in `docs/RESULTS.md` |
| 2 | backend | `--live --max-usd`; Haiku by default (`--model` to change) | rank list "do not spend time on": a bigger model gave ~1 point in Park et al. and +1.0 (n.s.) in our runs. On SimBench Sonnet did beat Haiku (46.3 vs 35.7, `docs/RESULTS.md` §1): try it last, with budget left |
| 3 | `sim.segment_distribution`: one call per segment x question x framing, 3 framings averaged; the same with only the population (no segment) as control | segment descriptions; pass `source=` / `when=` in `segment_distribution` when the brief names a real survey (named framings measured 0.077 vs 0.084 hidden) | rank #4: verbalised JSON shares (Meister TV 0.226; log-probs were worse than uniform). C1: 3 framings averaged, gap closed 0.818 |
| 4 | baselines from the train survey only: `train_overall` (segment ignored), `train_segment` (segment shares shrunk toward it with 5 pseudo-answers); uniform and the item-wide marginal come automatically | add a baseline the client already has (last year's wave per segment) as another `baselines_oof` column | rank #1: baseline ladder and gap closed (Park et al. arXiv:2411.10109 normalisation; Twin-2K-500 gap ~56 %) |
| 5 | `llm_prior` = LLM shares as a Dirichlet prior worth `K_PRIOR` answers, updated with the segment's train answers | nothing on the day; it is the "ours" named before the truth | rank #3 (LLM as a prior of a data model; Centaur, SubPOP: learning from human data is where the gain is) and rank #6 (shrink toward data) |
| 6 | `se.seal` of every arm, `ours = llm_prior` in the description | nothing; do it before the holdout is joined | pre-registration; the card's selection-bias audit checks that `ours` was named in the seal |
| 7 | `se.evidence.evaluate(task="distribution")`: 1-TVD ladder, persona shuffle (`llm_prior · shuffled`), paired CIs over questions, audits | `--B 1000` for the final card; `memorization=` once you ran the audit | rank #1 again: the card is the deliverable |
| 8 | `evaluate_distribution_persons(metric="w1")`: W1 on ordered options with a respondent bootstrap, paired differences ours − strongest baseline, LLM − no-segment control, LLM − train marginal | the pairs you will show | W1 is the CES / OpinionQA metric (`docs/RESULTS.md` §1); the respondent bootstrap answers "new people from the same segments", the card's CI answers "new questions" |
| 9 | PPI++ population share per option (`persona.inject.population.ppi_distribution`): train answers + raw LLM shares for every holdout respondent's segment; coverage and CI width ratio vs train only | the unlabelled side: census / panel counts per segment instead of the holdout | Angelopoulos et al. 2023 (PPI, arXiv:2301.09633), PPI++ power tuning (arXiv:2311.01453); our "worth N humans" measurements: a persona answering a survey was worth 1.02x, so expect a small gain unless the LLM tracks individuals |
| 10 | `card.md`, `dashboard.html`, spend line | nothing | |

Skipped on purpose (section "do not spend time on" of the literature ranking): rich biography personas (SubPOP,
Funhouse Mirrors r = 0.20), per-person panels for a distribution target (0.55 vs 0.818 of the gap), discussion rounds
(R5 ΔW1 −0.005, n.s.), fine-tuning, many samples per person.

## What the jury should see

1. **The ladder, one line per arm with a 95 % CI**: uniform, item-wide marginal (setting B, labelled as such),
   no-segment LLM control, train marginal, raw LLM, train segment shares, ours. In the offline run ours (`llm_prior`,
   1-TVD 0.899 [0.874, 0.918]) ties the train segment shares (0.893): badge "inconclusive". Say that: with 19-91
   real answers per segment the small survey is already strong, and the LLM prior only helps the smallest segments.
2. **The controls**: the no-segment arm (W1 0.145 vs raw LLM 0.071, difference −0.074 [−0.086, −0.070]) shows the
   gain comes from the segment description, not from the question text alone; the shuffled `llm_prior` falls to the
   train-marginal level.
3. **Where the LLM is wrong**: step 8 prints the cells where the raw LLM is worse than the train marginal (per-cell
   scores in `w1["cells"]`). In the synthetic data that is q5 (budget worry) for `old_rural` and `students`, plus
   `students/q4` by a hair. A
   question where the LLM loses to the marginal is a finding, not a bug (CES: 2 of 8 items).
4. **The population number with its CI**: "18.8 % strongly agree [13.5, 24.0] from 213 train answers; PPI++ with the
   LLM 19.1 % [13.9, 24.2]; holdout 17.0 %; median CI width ratio 0.98". A ratio near 1 means the LLM adds little at the population level; do not
   sell it as "worth N humans" unless the ratio is clearly below 1.
5. **The seal** (hash and time earlier than the evaluation) and, with `--live`, the spend line against `--max-usd`.

## Pitfalls

- **Leakage.** `segment_desc` must describe who the people are, never how they answered ("price-sensitive" is fine
  if it comes from a separate question you are not predicting; "mostly agrees that delivery is too slow" is the
  label). Baselines and `llm_prior` use the train respondents only; the holdout is joined after the seal. The
  item-wide marginal on the card is setting B (it reads the target question's answers in other segments): the card
  labels it and compares ours with the strongest setting-A baseline. Do not tune `POPULATION`, framings or
  `K_PRIOR` while looking at the holdout.
- **Imbalanced segments and accuracy.** Small segments have noisy truth: 1-TVD between a 20-person empirical
  distribution and *any* prediction is biased down, so the respondent-bootstrap percentile CI can sit beside the
  point. The script prints the basic CI; read cells with few holdout answers with care, merge tiny segments. Never
  report "accuracy" of the modal answer: the segment majority already gets 0.656 exact match on CES, the LLM 0.613.
  Report TVD / W1 per cell and the population-weighted share.
- **Memorisation.** A famous public survey (a well-known poll, OpinionQA, a published wave) may be in the training
  data. OpinionQA questions were public before the cutoff, so results there are an upper bound; on SimBench a 3-4
  point memorisation share could not be ruled out. The card flags "not proven: leakage/memorisation" until you run
  `se.evidence.memorization_audit` or `audit_prompts` (named vs blind survey) and pass it as `memorization=`. The
  hidden framings (no survey name, no year; the default without `source=`) are the safer arm.
- **PPI assumptions.** The PPI++ CI is valid only when the labelled respondents are a random subset of the same
  population as the unlabelled side. If the train survey over-samples a segment, weight it or estimate per segment.
  Use the raw LLM shares as the predictor: `llm_prior` has already seen the train labels.
- **Option order.** W1 uses the order in `options`. For unordered options (brands, parties) read 1-TVD only.
- **Budget.** `--live` refuses to start without `--max-usd`; once the ledger total reaches it the backend raises
  `BudgetExceeded` before any new call and the script exits with "stopped by the budget". Finished replies stay in
  the cache, so the same command with a higher cap resumes. The key comes from the environment or `--key-file` and is
  never printed. The offline synthetic run makes 75 calls (4 segments x 5 questions x 3 framings + 15 control).

# `persona.inject` family: `from_logs`

The persona is the person's own behaviour log: purchases, views, clicks, cart adds, ratings. It can go in as raw
events, as retrieved events, as a statistical summary, or as an LLM summary made before the run. All six injectors
need `history`, a list of event dicts, and read it through `from_logs.visible_events`. Module:
`simengine/persona/inject/from_logs.py`.

```python
import simengine.persona.inject as pi
from simengine.persona.inject import from_logs as fl

person = {"id": "u1", "history": [
    {"t": "2024-03-02", "action": "purchase", "item": "i9", "title": "Oat milk 1L", "category": "dairy",
     "brand": "B", "price": 2.1, "qty": 2, "discount": True},
    {"t": "2024-03-05", "action": "rate", "item": "m4", "title": "Heat (1995)", "category": "Action|Crime", "rating": 4}]}
q = {"id": "q1", "text": "Would you buy this oat milk at 2.50?", "options": ["yes", "no"], "cutoff": "2024-04-01"}

people = fl.summarize_logs(llm, people, style="paars", cutoff="2024-04-01")     # offline, before the run
arms = ["log_raw", ("log_retrieved", {"k": 5}), "log_stats", "log_paars"]
res = pi.compare_injectors(llm, people, questions, truth, arms, dry_run=True)
```

## Event dicts

All keys are optional:

- `t`, `time`, `date` or `timestamp`: an ISO string, a date, or a number. Numbers and dates must not be mixed.
- `action`: `purchase`, `view`, `click`, `cart`, `rate` and so on. When it is missing, an event with a `rating`
  counts as `rate` and any other event as `purchase`.
- `item`, `title`, `category` (a string, `"A|B"` or a list), `brand`.
- `price` (per unit), `qty`, `discount` (a bool or a fraction), `rating`, `text`.
- `qid`: tags an event as the answer to a question.

`None`, NaN, NaT and `pd.NA` values count as absent, so rows from `DataFrame.to_dict("records")` work as they are:
an event whose time is NaT is undated (hidden once a cutoff is set), and a NaN `rating` does not turn a purchase
into a rating. Numpy scalars are accepted for times and cutoffs.

Chat turns `(role, text)` in `history` are refused with a clear error.

## Leakage

- **Question fields:** `cutoff` hides events at or after that time. Once a cutoff is set, undated events are
  hidden too. `item` / `items` hide every event on the target item(s). Events tagged with the target `id` or an
  `exclude` qid are hidden.
- **Injector cutoff:** a `cutoff=` param on the injector also applies. The earlier of the two cutoffs wins.
  `hide_target_item=False` turns off item hiding, for tasks where seeing the item is legitimate.
- **LLM summaries:** `summarize_logs` writes `person["log_summary"][style]` together with the `cutoff`, the
  item ids and the qids of the events it was built from. An injector refuses to use a summary in three cases: the
  summary was built without a cutoff or with a later cutoff than the question's; it contains the target item; or
  one of its events is tagged with the question's `id` or an `exclude` qid. Rebuild it with `cutoff=`,
  `hide_items=` or `hide_qids=`.
- **Pure rendering:** summaries are made before the run, so `render()` stays pure and a dry run spends nothing.
  `summarize_logs(..., dry_run=True)` returns the call count and a token estimate.

## Injectors

| injector | what goes in the system prompt | params |
|---|---|---|
| `log_raw` | The last `n` events as text lines, oldest first | `n`=10, `order` chronological\|recent_first, `recency_note` |
| `log_retrieved` | The `k` events most relevant to the question | `k`=5, `method` bm25\|recent\|random, `seed` |
| `log_stats` | Recency, frequency, monetary; category shares; discount share and bought/viewed price ratio; top-brand and repeat-purchase shares; rating mean; per-category lines from `persona.history` | `top_k`=5, `per_category`, `like_at`=4 |
| `log_agent4rec` | Activity, conformity and diversity traits, plus a taste summary of 25 sampled rated items | `taste` summary\|items, `n_items`=25, `like_at`=3, `cuts` (from `trait_cuts`), `seed` |
| `log_paars` | An LLM consumer profile and shopping preferences, plus the raw history in daily sessions | `include_raw`, `n_raw`=30 |
| `log_task_summary` | A task-aware LLM summary, plus optionally `k` BM25-retrieved events | `k`=0 |

All six also take `cutoff` and `hide_target_item`. `meta["n_events"]` counts the events shown.

## Notes on the params

- **`log_raw`:** `recency_note=True` adds a sentence naming the latest event.
- **`log_retrieved`:** BM25 runs in pure Python with k1 = 1.5 and b = 0.75. Its idf comes from the person's own
  events, and ties go to the most recent event. The random order is seeded by `seed`, the person id and the
  question id.
- **`log_stats` without an LLM:** it reuses `persona.history.summarize_purchases` and `purchase_history_text` for
  the per-category lines. They need calendar dates and are skipped for numeric times.
- **`log_stats` reference time:** recency is measured from the cutoff. With numeric times it is given in "time
  units".
- **`log_agent4rec` conformity:** it needs `item_mean` on rating events. `fl.add_item_means(people, question)`
  adds it from the visible window of all people.
- **`log_agent4rec` tiers:** they need `cuts`. `fl.trait_cuts(people)` gives tertiles, because the paper does not
  publish its cut points. `taste="items"` shows the sampled liked and disliked lists and needs no LLM summary.
- **`summarize_logs` styles:**
  - `paars` makes two calls per person: the profile JSON, then the preferences.
  - `agent4rec` makes one call over the seeded 25-item sample.
  - `task` makes one call and needs `task="<what the summary is for>"`.
  - Styles accumulate in `log_summary`.

## Literature

**`log_raw`: Kang et al. 2023, *Do LLMs Understand User Preferences?* (arXiv:2305.06474); recency note from Hou
et al. 2024 (arXiv:2305.08845).**
- *Method:* the user's rating history is ordered by time and truncated to the 10 most recent interactions. It is
  shown zero- or few-shot (3 examples), and the model predicts a rating.
- *Kang et al. results (MovieLens):*

  | model | RMSE | AUC |
  |---|---|---|
  | zero-shot ChatGPT | 1.008 | 0.679 |
  | zero-shot Flan-U-PaLM | 1.068 | 0.708 |
  | Transformer+MLP (supervised) | 0.885 | 0.798 |
  | fine-tuned Flan-T5-XXL | 0.898 | 0.804 |

  Raw history zero-shot trails supervised recommenders. Fine-tuning closes the gap. Zero-shot LLMs are only
  comparable to trivial heuristics: the candidate item's average rating scores RMSE 0.975 and AUC 0.740.
- *Hou et al. results:* on ML-1M, NDCG@10 was 42.85% for the sequential history prompt and 44.23% with the
  recency-focused sentence ("Note that my most recently watched movie is ..."), +1.4 points.

**`log_retrieved`: Salemi et al. 2024, LaMP (arXiv:2304.11406).**
- *Method:* retrieval-augmented personalisation over the user's own profile items. The retrievers are BM25,
  Contriever, Recency and Random.
- *Results* (fine-tuned FlanT5-base, user split, k = 1):

  | task | no retrieval | BM25 |
  |---|---|---|
  | LaMP-1 accuracy | 0.518 | 0.649 |
  | LaMP-2 accuracy | 0.468 | 0.524 |
  | LaMP-3 MAE (lower is better) | 0.275 | 0.258 |

  Zero-shot GPT-3.5 with its best retriever went from 0.541 to 0.695 on LaMP-1.
- *How this injector differs:* it uses BM25 only (no Contriever) and works over event lines rather than documents.

**`log_stats`: statistical-summary baseline.**
- *Method:* RFM (recency, frequency, monetary; Hughes 1994, a marketing classic with no arXiv id), extended with
  category affinity, price sensitivity and brand loyalty. It is the non-LLM counterpart of LLM-inferred
  purchase-history personas such as Shi et al. 2025, *You Are What You Bought* (arXiv:2504.17304).
- *Shi et al. results:* LLM personas improved a graph-convolution recommender (LGCN, not LightGCN) by up to 12% in NDCG@K and F1@K. In
  customer segmentation on OnlineRetail they did compare against RFM: 18% of customers kept the same persona
  representation across two six-month halves against 1.3% for RFM (10 quantile bins per dimension), and k-means
  silhouette was 0.451 to 0.788 against 0.366 to 0.445. That is a comparison of representations for clustering,
  not of an RFM text summary in a prompt.
- *Published effect:* none found for this exact prompt. This arm answers the question "does an LLM summary beat
  plain statistics?"

**`log_agent4rec`: Zhang et al. 2024, *On Generative Agents in Recommendation* (Agent4Rec, arXiv:2310.10108).**
- *Method:* the profile has three social traits.
  - Activity is the number of items the user interacted with.
  - Conformity is the mean squared gap between the user's rating and the item's average rating.
  - Diversity is the number of distinct genres.
- *Tiers and taste:* each trait is split into three uneven tiers from MovieLens statistics (cut points not
  published; `trait_cuts` uses even tertiles instead). The taste is
  summarised by ChatGPT from 25 random items of the viewing history, where a rating of 3 or more counts as
  "liked".
- *Published effect:* there is no numeric ablation of the profile. With MovieLens taste discrimination at a 1:1
  ratio, accuracy was 0.691 and F1 0.698; at 1:9, F1 was 0.323. The authors note that diversity "might exhibit
  minimal distinction".

**`log_paars`: Mansour et al. 2025, PAARS (arXiv:2503.24228).**
- *Method:* two chained prompts over search, view and purchase history in daily sessions.
  - The first infers a consumer profile: gender, age, relationship status, education, industry, salary,
    home ownership, parental status and interests, with a reason for each.
  - The second describes how much seven factors matter to the shopper: products, price, value, product
    selection, reviews, brand reputation and quality.
  - The final persona is profile + preferences + the raw history.
- *Published effect (Amazon, item selection):* purchase-prediction accuracy was 25.5% with no customer context,
  41.1% with history only and 47.3% with the full persona. Query-generation similarity went from 0.59 to 0.69.
- *How this injector differs:* PAARS also lists older purchases in a separate block and uses a fixed list of
  interests. Here the last `n` events are used and interests are free text.

**`log_task_summary`: Richardson et al. 2023, *Integrating Summarization and Retrieval for Enhanced
Personalization via LLMs* (arXiv:2310.20081).**
- *Method:* an LLM (Vicuna-13B or GPT-3.5) writes a task-aware summary of the user's profile offline, and it is
  combined with a smaller BM25 retrieval at run time. The downstream model is a fine-tuned FlanT5-base, not a
  zero-shot LLM.
- *Results:* summary + k = 1 matched or beat k = 4 on most tasks with 75% less retrieved data.

  | task | k = 4 | summary + k = 1 | summariser |
  |---|---|---|---|
  | LaMP-1 accuracy | 0.709 | 0.743 | GPT-3.5 |
  | LaMP-2 (news categorisation) F1 | 0.574 | 0.601 | Vicuna (GPT-3.5 not run) |

  On LaMP-1 the GPT-3.5 summary alone (k = 0) reached 0.738, above k = 4. On LaMP-4 the summary trailed
  (ROUGE-1 0.181 against 0.188).

## Measured in simengine

Not yet measured in simengine. The tests run on `FakeLLM` only and check rendering, leakage, determinism and the
compare plumbing. They are not an effect size.

## Caveats

- **Summary model:** the LLM summary arms (`log_agent4rec` with `taste="summary"`, `log_paars`,
  `log_task_summary`) depend on which model wrote the summary. `log_summary[style]["model"]` records it.
- **Cache keys:** summaries made with a different cutoff are different prompts, so they get different cache keys.
- **Dataset fit:** Agent4Rec's traits come from movie ratings. On purchase-only logs, conformity is absent and
  activity counts all events.
- **Public data:** public logs (MovieLens, Amazon reviews) may be in a model's training data. Memorisation is not
  tested.

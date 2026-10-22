# `persona.inject` family: `own_answers`

The persona is the person's own answers to other questions. Every injector reads them through
`Injector.known_answers`, which drops the target question and its `exclude` ids. That holds under the swap
control as well, where the donor's answer to the target is dropped too. Module:
`simengine/persona/inject/own_answers.py`.

```python
import simengine.persona.inject as pi
bank = [{"id": "car", "text": "Do you own a car?", "options": ["yes", "no"]}, ...]   # = the questions list
arms = [("own_answers_all", {"questions": bank}),
        ("own_answers_fewshot", {"questions": bank, "k": 8, "select": "recent"}),
        *[("own_answers_rag", {"questions": bank, "k": k, "scorer": "bm25"}) for k in (3, 8)]]
res = pi.compare_injectors(llm, people, bank, "answers", arms, dry_run=True)
```

All three need `answers`. When that field is missing they raise `MissingPersonField`, and the message names the
injector and the field. `questions=` is the question bank, given as a list of question dicts or as
`{qid: dict | text}`. The bank lets an injector show each past question by its text and options. Its order counts
as time order (later means more recent). Without a bank, `all` and `fewshot` show past questions by their id, and
`rag` refuses to run. In arm labels the bank is shown as its size, e.g. `own_answers_rag(k=3,questions=40q)`.
`meta["shown"]` lists the qids placed in context, in prompt order. If a person has no answer left once the target
is removed, the prompt says so (`n_shown = 0`) and the comparison continues.

| injector | what goes in the prompt | params |
|---|---|---|
| `own_answers_all` | System prompt: a framing line, then every known answer as `Q: / Options: / A:` text | `order` bank\|reverse\|random, `max_items` (keeps the most recent), `seed`, `include_options` |
| `own_answers_fewshot` | k answers as earlier turns. Each pair is user = the past question block, assistant = `{"answer": ...}`. Flattened to `Interviewer:/You:` unless the LLM supports history | `k`=8, `select` random\|recent\|first, `order`, `seed`, `include_options` |
| `own_answers_rag` | System prompt: the k answers whose question text is lexically closest to the target question | `k`=8, `scorer` tfidf\|bm25\|overlap, `order` desc\|asc\|bank, `match` question\|question_answer, `include_options` |

Notes on the params:

- **Random selection:** `select="random"` keeps the k qids with the smallest `sha256(seed:person:qid)`. One person
  therefore gets the same demonstrations for every target. The only exception is a target that was itself picked:
  it is dropped. The result is deterministic, so reruns hit the LLM cache.
- **Scorers:** the retrieval scorers are pure Python, with no sklearn or embeddings. `tfidf` mirrors sklearn's
  smooth idf with l2 cosine. `bm25` uses k1 = 1.2 and b = 0.75. `overlap` is token Jaccard. Tokens are lowercased,
  with stopwords removed and a plural "s" stripped. Ties keep bank order.
- **Ordering:** `order="asc"` puts the most similar answer last, next to the question. `desc` (the default) puts it
  first.

## Literature

**`own_answers_all`: Toubia et al. 2025, Twin-2K-500 (arXiv:2505.17479).**
- *Panel:* 2,058 US Prolific respondents answered about 500 items over four waves.
- *Persona:* a digital twin's persona is *all non-held-out content of waves 1-3*, about 412 items. Each item is
  rendered as text with the question, its options and the person's answer, followed by an instruction to answer
  as this person and stay consistent with past answers.
- *Targets:* the 88 held-out heuristics-and-biases and pricing items from waves 1-3. Wave 4 is only the human
  test-retest benchmark.
- *Published effect* (GPT-4.1-mini, temperature 0, text persona):
  - Twin accuracy 71.72%, against 81.72% human test-retest and 59.17% random guessing. That is 87.67% of
    test-retest.
  - A persona *summary* alone scored 68.02%, about 3.7 points lower.
  - JSON format, repeated questions and reasoning prompts each moved accuracy by less than about 1.5 points.

The repo's own `persona.methods.twin2k` uses the authors' released summaries. This injector builds the full-text
variant from whatever answers the person dict holds.

**`own_answers_fewshot`: in-context demonstrations (Brown et al. 2020, arXiv:2005.14165).**
- *Format:* the same person's past answers are shown as solved dialogue turns, so the model sees the answer
  format and the person's answers together.
- *Selection:* `random` and `recent` are the Random and Recency profile selectors of LaMP (Salemi et al. 2023,
  arXiv:2304.11406).
- *Published effect:* in LaMP, Recency was a strong baseline and beat Contriever on one time-split task (LaMP-3T).
  Even a random profile item improved over no personalisation on most tasks. No paper we checked reports a
  number for this exact format (own survey answers as dialogue turns).

**`own_answers_rag`: Hwang, Majumder, Tandon 2023, *Aligning Language Models to User Opinions*
(arXiv:2305.14929); retriever and k ablation from LaMP (arXiv:2304.11406).**
- *Hwang et al. method:* on OpinionQA (Pew ATP), they retrieved the top-k (k ∈ {3, 5, 8}) of a user's past
  opinions (at most 16) by text-embedding-ada-002 similarity to the question, with GPT-3 zero-shot.
- *Hwang et al. results (exact-match accuracy):*

  | prompt | accuracy |
  |---|---|
  | no persona | 0.43 |
  | demographics + ideology | 0.47 |
  | + all opinions | 0.51 |
  | top-3 opinions alone | 0.51 |
  | top-8 opinions alone | 0.52 |
  | demographics + ideology + top-8 | 0.54 |

  Top-3 was about as good as top-8, so a few relevant answers carry most of the gain. The paper's headline is "up
  to 7 points" per topic.
- *LaMP method:* retrieval compares BM25 (term matching), Contriever, Recency and Random on the user's own
  profile items, with k swept.
- *LaMP results:* retrieval gave a 12.2% average relative gain zero-shot (FlanT5-XXL / GPT-3.5) and 23.5% when
  fine-tuned. BM25 or Contriever beat Random everywhere. More items usually helped, though some tasks dropped.
- *How this injector differs:* it is embedding-free on purpose, so TF-IDF and BM25 stand in for ada-002 and
  Contriever. That is a deviation from Hwang et al. Hwang et al. also phrase opinions as statements ("X describes
  me well"). Here they are `Q:/A:` pairs.

## Measured in simengine

Not yet measured in simengine. The tests run on `FakeLLM` only and check rendering, leakage, determinism and the
compare plumbing. They are not an effect size.

## Caveats

- **Wave split:** Twin-2K-500's split is by item type (held-out item blocks), not "wave 1 predicts wave 2". To
  reproduce it, put the held-out ids in `questions` as targets. Then either strip them from `answers` or list them
  in each target's `exclude`.
- **Public data:** public panels (Twin-2K-500, Pew) may be in a model's training data. Memorisation is not
  tested.
- **Lexical retrieval:** retrieval only matches surface wording. A paraphrased or translated bank weakens it.
  Use `match="question_answer"` when option labels carry the content.

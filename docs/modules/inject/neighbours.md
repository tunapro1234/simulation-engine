# `persona.inject` family: neighbours

`simengine/persona/inject/neighbours.py`. The injectors in this family put the answers of *other, similar* people into the prompt. Each one is a nearest-neighbour or collaborative-filtering method: they differ in what they show (profiles with answers, an aggregate vote, or answer histories) and in how similarity is measured.

```python
import simengine.persona.inject as pi

inj = pi.get_injector("similar_respondents", pool=people, k=10)          # neighbours' profiles + their answer
p = inj(people[0], {"id": "q1", "text": "Do you own a car?", "options": ["yes", "no"]})
llm.complete(p.system, p.flatten())
pi.get_injector("neighbour_votes", pool=people, k=10, show="share")       # "people like you: yes 70%, no 30%"
pi.get_injector("answer_neighbours", pool=people, questions=questions)    # CF over answer vectors
pi.compare_injectors(llm, people, questions, "answers",
                     [("similar_respondents", {"pool": people}), ("neighbour_votes", {"pool": people})], dry_run=True)
```

Every injector takes `pool=` (a list of person dicts) at construction time. The pool may contain the target person, because the injector removes them by id. In an arm label, the pool appears as `pool=n<size>#<hash>`, so labels stay short and two different pools never share a label.

## Injectors

| name | needs | what it renders |
|---|---|---|
| `similar_respondents` | `fields` (`basis` can change this) | User turn: "Here is a selection of k survey participants. They are selected to be most similar to ...", one line per neighbour with their profile and their answer to this question, then "Here is the participant to be classified:" with the target's profile and the question. `format="multi"` puts each neighbour in a separate user/assistant exchange. |
| `neighbour_votes` | `fields` (`basis` can change this) | System prompt: the target's attributes, plus a count of how the k most similar other people answered this question (`yes 7, no 3`, or shares with `show="share"`). For a numeric question it lists the sorted values and their median. |
| `answer_neighbours` | `answers` | System prompt: the target's own earlier answers, then the k people whose answers agree most with them on those items (at least `min_overlap` shared items). Each neighbour is shown with an agreement percentage, their answers on the shared items, and their answer to this question. |

Parameters:

- **Shared by all three:**
  - `k`: number of neighbours. The default is 10, or 5 for `answer_neighbours`.
  - `keys`: restricts similarity and display to these field names.
  - `seed`: the only source of randomness. It breaks ties and drives `strategy="random"`.
  - `basis`: what similarity is computed on. One of `"fields"`, `"answers"` (the target's `known_answers`) or `"both"`. It also sets `needs`. `answer_neighbours` always uses `"answers"`.
- **`similar_respondents` only:**
  - `strategy`: `"most_similar"` (the default), `"least_similar"` or `"random"`.
  - `stratify=True`: splits the k slots evenly across the answer options, as in the paper.
  - `format`: `"single"` or `"multi"`.
- **`answer_neighbours` only:**
  - `questions=`: a list of question dicts, or a `{qid: text}` map, so that items show their text and not their ids.
  - `max_items`: caps how many of the target's answers are listed. The default is 20.

**Similarity.** Similarity is a Gower-style mean over the target's keys:

- A categorical value scores 1 on an exact match (ignoring case) and 0 otherwise.
- A numeric value scores `1 - |a - b| / range`, where the range is taken over the pool.
- A key the neighbour lacks scores 0.

Only pool members who answered the target question are candidates (complete cases on the target item), as in the paper. Ties break on a hash of `(seed, person, question, neighbour)`, so `render` stays pure and deterministic.

**Leakage guard.**

1. The target person is never a neighbour. They are matched by id, and integer and string ids compare equal. Ids listed in the question's optional `exclude_people` are left out too. Question ids and answer keys also compare as strings, so `{0: ...}` and `{"0": ...}` answer dicts behave the same.
2. The target's vector is built only from their fields and `known_answers`, so their own answer to the target question, and to any `exclude` id, is never shown and never used for similarity. Changing that answer leaves the prompt unchanged, and a test checks this.
3. Neighbours appear as "Participant 1..k", never by id. Their ids are in `meta["neighbours"]`, for auditing.

Neighbours' answers to the target question are shown on purpose, because that is the method. The leakage the guard prevents is the *target's* answer, not answers from the rest of the population. If the pool and the evaluation people are the same, this is leave-one-out imputation, as in the paper's setup. To evaluate held-out people, pass a pool that excludes them, or list them in `question["exclude_people"]`.

**Swap control caveat.** In `compare_injectors`, the `swap_<arm>` control renders the injector for a donor, and `Swap` does not tell the inner injector which person is being scored. When the pool contains the evaluation people, the scored person can then appear among the donor's neighbours, with their true answer to the question. In a 12-person test pool this happened for 11 of 12 people with `similar_respondents(k=10)`. This inflates the swap arm, so `vs_swap` for this family understates the injector's effect. The fix belongs in `Swap.render` (`_base.py`): pass `{**question, "exclude_people": [person["id"]]}` to the inner injector. These injectors already honour that key. Until that change lands, read `vs_no_persona` for this family, or use a pool disjoint from the evaluation people.

If the target has nothing to match on, the injector raises `MissingPersonField` and lists the `needs`. This covers an empty `fields`, and the case where the only answers are the target question or its `exclude` ids. If no pool member answered the question, the injector says so in the prompt and does not fail.

## Papers and published effects

| injector | method in the paper | published effect | in simengine |
|---|---|---|---|
| `similar_respondents` | Holtdirk, Ahnert, Sakshaug & Haensch 2026, *In-Context Learning for the Imputation of Public Opinion Data with LLMs*, arXiv:2606.09351. American Trends Panel, 150 opinion variables, 15 waves. Respondents are embedded with EmbeddingGemma-300M and the nearest neighbours by cosine similarity become in-context examples. The paper compares random, most-similar, least-similar and diverse (k-medoids) selection, optional stratification by answer, and single-turn, multi-turn and verbalized prompts. The best specification is verbalized single-turn, most similar, with no stratification. | Median absolute error, over the 150 variables, of a downstream OLS coefficient (political ideology regressed on the imputed opinion variable, both standardised) against the coefficient from the full data; gpt-oss-120b ICL(100) against MICE PMM: 0.033 vs 0.068 (MCAR), 0.044 vs 0.092 (MAR), 0.048 vs 0.068 (MNAR), which is about 51%, 52% and 29% lower. ICL(10) scores 0.035 / 0.048 / 0.049, and zero-shot scores 0.043 / 0.064 / 0.049. The paper's abstract reports near-nominal 95% coverage with CIs 2-5x narrower than MICE PMM's. Individual-level imputation accuracy is reported separately (their Figure 4), so these numbers are about downstream estimates, not per-person hit rates. | not yet measured in simengine |
| `neighbour_votes` | User-based kNN collaborative filtering (Resnick et al. 1994, GroupLens; not on arXiv), given to the LLM as aggregated evidence, as in CoRAL (Wu et al. 2024, arXiv:2403.06447). CoRAL lists which retrieved users like or dislike each item and asks the LLM to infer what kind of user likes it. | CoRAL, Amazon Product (4 datasets), average AUC / F1: LLM without collaborative evidence 70.61 / 63.29; CoRAL-DFM 93.99 / 82.42; random retrieval 77.61 / 57.44, so irrelevant neighbours hurt F1 (Table 1). | not yet measured in simengine |
| `answer_neighbours` | Histories of similar users in the context: CFRAG (Shi et al. 2025, SIGIR, arXiv:2504.05731) finds similar users with contrastively trained embeddings and retrieves documents from their histories for the LLM. CoRAL as above. | CFRAG, LaMP, Llama3-8B: removing the similar-user retrieval (keeping only the current user's history) lowers LaMP-2 accuracy from 0.5340 to 0.4936 and raises LaMP-3 MAE from 0.2812 to 0.3444 (Table 3). | not yet measured in simengine |

**What differs from the papers.**

- **Similarity.** All three use the Gower match over fields and answers, not learned embeddings (EmbeddingGemma in Holtdirk, contrastive user embeddings in CFRAG) or an RL retrieval policy (CoRAL). This keeps `render` pure and dependency-free. An embedding-based variant would need a precomputed `{id: vector}` map passed as a parameter.
- **Holtdirk features not implemented.** The paper's "diverse" (k-medoids) strategy, its List serialization with reciprocal-rank fusion, and its verbalized probability-distribution output (sampled m=5 times at temperature 1) are not implemented. `similar_respondents` asks for a single strict JSON answer, as every injector in the zoo does.
- **System message.** The paper's system message also states the survey region and field dates. Put these in the question text if they matter.
- **Data domains.** CoRAL and CFRAG are recommendation and LaMP personalisation results, not survey answers, so their effect sizes show the direction of the effect and are not a prediction for survey items.

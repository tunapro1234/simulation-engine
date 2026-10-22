# `persona.inject` family: `finetune`

**Not measured in simengine.** These methods put the persona into an open model's weights (LoRA, QLoRA), its
activations (steering vectors) or learned input embeddings (soft prompts). An API backend can run none of them.
This family therefore provides three things:

- the prompt format a tuned or steered model sees, the same at training and at inference;
- dataset builders that write chat-format JSONL;
- open-model helpers for a single Colab T4: `train_colab_t4`, `steering_vector`, `steer`.

torch, transformers and peft (plus bitsandbytes for 4-bit) are imported lazily inside those helpers. When one is
missing, the helper raises an `ImportError` that names the package. The tests cover only the injectors and the
dataset builders. Module: `simengine/persona/inject/finetune.py`.

```python
import simengine.persona.inject as pi
from simengine.persona.inject import finetune as ft

rows = ft.chat_rows(people, questions, "centaur_qlora", texts={q["id"]: q["text"] for q in questions})
train, test = ft.holdout_split(rows, frac=0.2, seed=0)        # held-out people
ft.write_jsonl(train, "train.jsonl")
cfg = ft.recipe("centaur")                                     # QLoRA r=8, all linear layers, 1 epoch
ft.train_colab_t4("train.jsonl", "adapter_out", cfg)           # Colab T4 only; needs torch/transformers/peft

# SubPOP: group answer distributions, forward-KL loss
groups = [{"id": "g1", "fields": {"region": "north"}}]
dist = {("g1", "q0"): {"yes": 30, "no": 10}}
ft.write_jsonl(ft.distribution_rows(groups, questions, dist), "subpop.jsonl")
```

`compare_injectors` accepts these injectors, which makes them usable for a dry run, for checking prompts, or
against a locally served tuned model. Pass `parse=ft.parse_finetuned`, which reads `<<answer>>`, a JSON answer,
an exact option or a bare option letter. With a plain API model the comparison is meaningless:

- The steering and soft-prompt arms send the bare question. Their persona lives only in `Prompt.meta`.
- The fine-tune arms send a prompt meant for a model trained on that format.

## Injectors

| injector | needs | what goes in the prompt | params |
|---|---|---|---|
| `subpop_lora` | `fields` | A steering text, then the question with lettered options, ending in `Answer:`. Training target: the option letter, or a distribution over letters via `distribution_rows`. | `style` qa\|bio\|portray (default `portray`), `keys` (the fields to use; default all non-empty) |
| `centaur_qlora` | `answers` | The instructions, then the person's other answers (from `known_answers`) as a transcript with each answer marked `<<...>>`, then the target question. Training target: `<<answer>>`. | `texts` {qid: question text}, `chat` (True: chat turns; False: one plain-text transcript), `max_turns`, `seed` (seeds the turn order) |
| `caa_steer` | `fields` | Bare question. `meta["steer"]` holds the method, layer, positions ("after_prompt") and per-person coefficients. | `traits` [field] or {trait: field}, `levels` {category: number}, `center`, `multiplier`, `layer` |
| `persona_vector_steer` | `fields` | As `caa_steer`, with positions = "every_decode_step" | same as `caa_steer` |
| `soft_prompt` | `fields` | Bare question. `meta["soft_prompt"]` holds the group key (for example `region=north`), and `meta["n_virtual_tokens"]` is also set. | `keys`, `n_tokens` (default 20) |

How the steering arms compute a coefficient:

- A numeric field becomes `multiplier * (value - center)`.
- A categorical value goes through `levels`. Without a `levels` entry it raises `ValueError`.

`steer_from_meta(meta, trait_vectors, layer)` sums the per-trait vectors into `{layer: vector}`, ready to pass to
`steer(model, vectors)`.

How missing data is handled:

- A missing person field raises `MissingPersonField`. The message names the injector, the missing keys, the
  person id and `needs`.
- `chat_rows` skips any cell that has no truth value.
- `subpop_lora` refuses an answer that is not one of the options.

## Builders and helpers

- `chat_rows(people, questions, injector, truth="answers", flatten=False, **params)` returns one row per cell
  with truth. Each row has `{"messages": [...system, ...history, user, assistant target], "person", "question",
  "injector", "meta"?}`.
  - Every injector, of any family, renders its prompt through its own `render`. The target answer therefore
    appears only in the final assistant turn.
  - Injectors from other families get a JSON `{"answer": ...}` target.
- `distribution_rows(groups, questions, dist)` builds SubPOP rows: the prompt messages plus `target_dist`
  `{letter: share}`. It accepts only choice questions, and it rejects unknown options and negative or all-zero
  weights.
- `caa_pairs(questions, behavior, seed)` builds the CAA A/B prompts. The trait option and each other option are
  shuffled into (A)/(B) with a seed. The `positive` and `negative` fields hold the answer letters.
- `persona_vector_pairs(questions, pos_systems, neg_systems, rollouts)` returns the generation plan for
  persona-vector extraction. After a judge scores each response 0-100, `judge_filter(rows, threshold=50)` keeps
  trait-expressing positive rows and trait-free negative rows.
- `holdout_split(rows, frac, seed)` splits by person, so a held-out person has no rows in the training set.
  `write_jsonl` and `read_jsonl` write and read the rows.
- `recipe(paper)` returns the paper's hyperparameters plus T4 settings: fp16 (the T4 has no bf16), micro batch 4,
  gradient accumulation up to the paper's batch size, and max_len 1024. The default base model is
  Qwen2.5-1.5B-Instruct, which fits a T4 in 4-bit. The papers used 7B-70B models, so their effects need not
  carry over to this size.
- `train_colab_t4(jsonl, out_dir, cfg)` trains with LoRA, QLoRA or prompt tuning:
  - Rows with `target_dist` use forward KL over the option-letter logits.
  - Other rows use cross-entropy on assistant tokens only.
- `steering_vector(model, tok, pos, neg, layers, mode)` computes the mean activation difference at each decoder
  layer's output. `mode="last"` reads the answer-letter token, as in CAA. `mode="mean"` averages over the response
  tokens, as with persona vectors.
- `steer(model, vectors, coef, prompt_len)` is a context manager that adds the vectors through forward hooks.

## Literature

**`subpop_lora`: Suh et al. 2025, SubPOP (arXiv:2502.16761).**
- *Method:* LoRA fine-tuning on about 70K subpopulation-response pairs covering 3,362 questions from public
  opinion surveys. The loss is forward KL between the model's next-token distribution over option letters and the
  group's response distribution.
- *Settings:* LoRA r = 8, alpha 32, dropout 0.05, on the q and v projections. lr 2e-4 at batch 256 for the 7B
  models.
- *Prompts:* the steering texts follow OpinionQA's QA, BIO and PORTRAY formats.
- *Published effect:* the paper reports up to a 46% smaller LLM-human gap than baselines. The 46% is an average
  gap-closure figure. Measured as Wasserstein distance on OpinionQA for Llama-3-70B, the best model went from
  0.138 (strongest zero-shot prompt) to 0.094, about 32%. The paper reports generalization to unseen surveys and
  subpopulations.

**`centaur_qlora`: Binz et al. 2025, Centaur / Psych-101 (arXiv:2410.20268).**
- *Data:* Psych-101 holds about 60K participants, 160 experiments and over 10M choices, written out as
  natural-language transcripts with the human responses marked `<< >>`.
- *Method:* QLoRA on a 4-bit Llama 3.1 70B, with r = 8 on all non-embedding linear layers. One epoch, lr 5e-5,
  weight decay 0.01, effective batch 32, 100 warmup steps. The loss covers only the response tokens.
- *Published effect* (negative log-likelihood per response on held-out participants, lower is better): Centaur
  0.44, the base Llama 0.58, domain-specific cognitive models 0.56.
- *In simengine:* the transcript is the person's other survey answers, so this is an adaptation of the method,
  not a replication.

**`caa_steer`: Panickssery et al. 2024, Contrastive Activation Addition (arXiv:2312.06681).**
- *Method:* the steering vector is the mean residual-stream difference at the answer-letter token over A/B
  pairs. It is added, with a signed multiplier, at every position after the user's prompt.
- *Models and behaviors:* Llama 2 Chat 7B and 13B, steered on behaviors such as sycophancy, hallucination and
  refusal. The layer is chosen by a sweep, and a middle layer worked best for 7B.
- *Published effect:* CAA changed behavior on both multiple-choice and open-ended evaluations, worked on top of
  fine-tuning and system prompts, and reduced capabilities only slightly. The abstract reports no single effect
  size.
- *In simengine:* a person's field value sets the multiplier. This per-person use is ours, not the paper's.

**`persona_vector_steer`: Chen et al. 2025, Persona vectors (arXiv:2507.21509).**
- *Method:* the vector is the mean activation over response tokens under trait-eliciting system prompts minus
  the mean under trait-suppressing ones. The responses are filtered by a judge. One vector is extracted per layer,
  and the layer is chosen by how well it steers.
- *Models and traits:* Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, on the traits evil, sycophancy and
  hallucination.
- *Published effect:* the projection at the last prompt token correlated with the trait expression that followed
  at r = 0.75-0.83. The activation shift during fine-tuning correlated with the trait expression after
  fine-tuning at r = 0.76-0.97. The paper also uses the vectors for preventative steering and for flagging
  training data.
- *In simengine:* using these vectors for survey personas is an extrapolation.

**`soft_prompt`: Lester et al. 2021, prompt tuning (arXiv:2104.08691).**
- *Method:* the LM stays frozen and only a few prompt embeddings are trained. In simengine there is one soft
  prompt per subpopulation key.
- *Published effect:* the gap to full model tuning narrows as T5 grows, and at billions of parameters prompt
  tuning matches model tuning.
- *In simengine:* the persona and survey use is ours. We know of no published persona-simulation result for soft
  prompts.

None of the five arms has been run against people in simengine. The effects above are the papers' own numbers,
on their own models and data.

# Centaur for purchase simulation: research notes

_23 Oct 2026, written for the TALP Simulation Hackathon (24 Oct). This is a research note only: no training, inference or API
LLM call was run for it. Numbers come from the cited sources or from files already in `/data/talp/is/`. Where a source could
not be opened or a claim could not be checked, the note says "not verified"._

Related code (not changed here): `simengine/persona/inject/finetune.py` (`centaur_qlora` injector, `chat_rows`,
`holdout_split`, `recipe("centaur")`, `train_colab_t4`). Related catalogue: `docs/latent/KATALOG.md` (B11, K7).

---

## 1. The Centaur recipe

Paper: Binz, Akata, Bethge, ... Schulz (45 authors), "A foundation model to predict and capture human cognition", Nature
(July 2025), DOI 10.1038/s41586-025-09215-4; preprint "Centaur: a foundation model of human cognition", arXiv:2410.20268
(v1 26 Oct 2024, v3 28 Apr 2025). The Nature page is paywalled from here; the details below come from arXiv v3 (HTML) and
the Hugging Face cards.

### 1.1 Data: Psych-101

- 160 experiments, 60,092 participants, 10,681,650 choices, 253,597,411 text tokens (arXiv v3). HF
  `marcelbinz/Psych-101`: one `train` split of 60,092 rows, fields `text`, `experiment`, `participant`, about 859 MB,
  Apache-2.0. `marcelbinz/Psych-101-test` is gated (login and access request needed).
- One row = one participant's whole session as plain prose: the task instructions first (kept close to the original
  study, simplified where needed), then every trial in order as a stimulus/event line followed by the response.
- Responses are wrapped in `<<` and `>>`. The dataset card example: `D: piecrust, P: game, and H: bracelet. You press <<D>>.`
  Feedback ("You win 34 points") stays in the transcript, so later trials condition on earlier outcomes.
- Discrete option names (keys) are randomised per participant so the model cannot learn a fixed key bias. The Psych-201
  contribution rules state this; that Psych-101 did the same everywhere is likely (the card example uses arbitrary keys
  D/P/H) but not verified.
- Transcripts are capped at about 32,768 tokens per participant (paper; also the Psych-201 rule "32K tokens per
  participant").

### 1.2 Training

| item | value (arXiv v3) |
|---|---|
| base | Llama-3.1-70B, frozen, 4-bit (QLoRA) |
| adapters | LoRA r = 8 on all non-embedding linear layers (attention and MLP); 0.15% of base parameters |
| alpha, dropout | not reported (our `recipe("centaur")` uses 16 and 0; that is our choice) |
| loss | cross-entropy on human response tokens only; every other token masked |
| epochs | 1 ("prolonged training led to overfitting" in their runs) |
| batch | effective 32 |
| lr / schedule | 5e-5, linear warm-up over the first 100 steps |
| weight decay | 0.01 |
| optimiser | 8-bit AdamW |
| library | unsloth |
| hardware / time | one A100 80 GB, about five days |
| context length | transcripts capped near 32k tokens; a separate training max-length is not stated in what we could read |
| contamination | LogProber check, no evidence of memorisation |

Derived figure (ours, not the paper's): 253.6 M tokens in about 5 days on one A100 is roughly 590 tokens/s for 70B QLoRA
with unsloth, counting every token of the transcript in the forward pass.

### 1.3 Evaluation designs and results (NLL per response, lower is better)

Metric: negative log-likelihood averaged over responses; multi-token responses have their token log-likelihoods summed.
Baselines: 14 domain-specific cognitive models (fit on training participants; for out-of-distribution tests the parameters
come from the closest training experiment) and the base Llama. Other Llama fine-tunes (Nemotron, Hermes, Reflection) did
not beat the base model.

| design | what is held out | Centaur | Llama base | cognitive model |
|---|---|---|---|---|
| held-out participants | 10% of participants of every experiment (90/10 split by participant) | 0.44 | 0.58 | 0.56 |
| cover-story change | two-step task told as "magic carpet" (training had the spaceship story) | 0.51 | 0.63 | 0.61 |
| structure change | "Maggie's farm", a three-armed horizon task (training had the two-armed horizon task) | 0.42 | 0.62 | 0.98 |
| new domain | logical reasoning (LSAT items), nothing similar in training | 1.65 | 1.92 | none |

- Centaur beat the domain-specific model in all but one Psych-101 experiment, and fine-tuning improved fit in every
  experiment.
- Further out-of-distribution checks: moral decisions, economic games, naturalistic category and reward learning,
  behavioural propensities, a deep sequential decision task (the paper speaks of six, the HTML lists five).
- Open-loop simulation (the model generates its own trajectory): horizon task performance and directed exploration close
  to humans; the two-step task gives model-free, model-based and mixed agents; four economic games: 64% accuracy on human
  responses vs 35% on artificial agents.
- Response times (about 4 M, Hick's-law mixed model): conditional R^2 0.58 for Centaur, 0.40 Llama, 0.38 cognitive models.
- Neural alignment: (a) two-step fMRI, 94 participants x 300 choices, regularised regression from residual-stream
  activations (PCA keeping 95% variance) to region activity (fMRIPrep 24.0, Schaefer 2018 atlas): Centaur above Llama at
  every layer (p <= 0.001), cognitive-model representations clearly worse. (b) sentence-reading fMRI, 5 participants x
  1,000 sentences: pooled fine-tuning benefit beta = 0.007, 95% CI [0.0002, 0.013], p = 0.045; no single layer significant.
  The neural evidence for (b) is weak.
- Model-guided discovery: a DeepSeek-R1-proposed model scored AIC 181.7, Centaur 72.5, the refined interpretable model 71.7.

### 1.4 Minitaur, RandomInit, Psych-201

- **Minitaur-8B** (`marcelbinz/Llama-3.1-Minitaur-8B`, plus `-adapter`): Llama-3.1-8B with the same recipe on Psych-101.
  The paper says it captures behaviour close to the training distribution but generalises less robustly, and that it runs
  on free Colab GPUs. The HF card has no numbers; licence llama3.1.
- **RandomInit-8B / -70B**: the HF cards are empty templates. Probably the random-weight control for the representation
  analyses; not verified.
- **Psych-201** (`marcelbinz/Psych-201`, Apache-2.0, DOI 10.57967/hf/8731, updated May 2026): 208,021 rows, 149 studies,
  264 experiments, 541 MB. Differences from Psych-101 visible in the schema: per-participant metadata (`RTs`, gender, age,
  nationality, education, clinical diagnosis) and questionnaire scores (STAI, PHQ-9, GAD-7, BIS-11, AUDIT, OCI, BAS ...),
  flags `is_psych101` / `is_psych101_test`, `in_eval`. It is a community-contributed successor (GitHub
  `marcelbinz/Psych-201`: one folder per study, `prompts.jsonl` + `generate_prompts.py`, trial-level data in temporal
  order, `<< >>` markers, 32k-token cap). The paper's Discussion frames it as adding domains, individual differences and
  more diverse populations. No paper for Psych-201 itself was found.
- **Psych-201-discrete**: same schema, 131,834 rows. The card has no description; that it is the discrete-response subset
  is our inference from the name. Test splits `-test` exist.
- **Psych-201-discrete-agentic** (Oct 2026): 80,452 train / 2,900 test rows from the "Hugging-Brain psych-301"
  collections (experiments transcribed by an agent); letter responses written `[HUMAN_RESPONSE]X[/HUMAN_RESPONSE]`;
  per-study row-level split, so a participant can appear on both sides.
- None of these is consumer or purchase data.

---

## 2. Criticism and follow-up work (2025-2026)

| source | claim | strength for us |
|---|---|---|
| Liu & Ding, National Science Open 5(1) 20250053, 2025/26, DOI 10.1360/nso/20250053 | Instruction-free, context-free and misleading-instruction ("Please choose option A") variants of four tasks: Centaur keeps giving the original human-like answers instead of following the new instruction; the authors read the fit as overfitting to answer patterns. In the context-free condition Centaur still beat the cognitive models on two of four tasks. | Direct test design we can copy (see 3.2). Full text not opened (403); details from news coverage. |
| Oh & Gobet, "Small Foundation Models of Human Cognition and Behaviour", arXiv:2608.05224, COLM 2026 | 14 models, 135M-14B, LoRA (rank-stabilised, all linear, 1 epoch, lr 5e-5, wd 0.01, 32k context), one A100 80GB, 3-49 h per model. In-distribution NLL: their Centaur-70B reproduction 0.524; Qwen3-0.6B at rank 64 0.513; Qwen3-8B 0.517. Out-of-distribution (Psych-201-RT, 18 held-out experiments) NLL 0.79-1.03, larger is better; only three fine-tuned models beat the un-fine-tuned Qwen3-14B (0.869). Masking stimuli and feedback drops performance below chance; content carries 75.7% of the learned information, instructions 12.5%. | Strong: in-distribution fit does not need 70B; generalisation does. Held-out-participant scores mostly measure fit to the task family. |
| Namazova, Brondetta, Strittmatter, Nassar, Musslick, "Not Yet AlphaFold for the Mind", arXiv:2508.07887, 2025 | Prediction is strong, but generated (open-loop) behaviour diverges systematically from human data. | Relevant if we sell "simulation" rather than "prediction": score generated trajectories, not only next-step likelihood. |
| Bowers et al., PsyArXiv 10.31234/osf.io/v9w37 (2025), via Univ. of Bristol news 8 Jul 2025 | Short-term memory and response times are unlike humans (secondary sources: lists of up to 64+ items recalled perfectly, all-or-nothing failure). | Preprint itself not opened; not verified beyond the press note. |
| Orr, Cranford, Ford, Gluck, Hancock, Lebiere, Pirolli, Ritter, Stocco, "Not Even Wrong", arXiv:2510.03311, 2025 | Prediction is not explanation; Centaur is "a unified model of behavior sans cognition". | Conceptual; does not affect a prediction claim. |

Gain over domain-specific models: on held-out participants 0.13 nats per response on average (0.56 -> 0.44); on the
structure change 0.56 nats, mostly because the transferred cognitive model breaks (0.98). The cognitive models were fit
with parameters shared across participants, which is a fairly weak baseline for individual prediction; this point is ours,
not from a cited paper.

Choice/consumer work near our problem (none uses the Centaur weights; we found no published Centaur-on-purchases study):

- Kolluri, Wu, Park, Bernstein, "Finetuning LLMs for Human Behavior Prediction in Social Science Experiments",
  arXiv:2509.05830, EMNLP 2025: SocSci210 (2.9 M responses, 400,491 participants, 210 experiments), Qwen2.5-14B; on
  unseen studies 26% better distribution alignment than the base model and 13% better than GPT-4o; +71% on unseen
  conditions of a seen study. Distribution-level, not individual purchase history.
- Holtdirk, Assenmacher, Bleier, Wagner, arXiv:2509.25063 (KONVENS 2026): fine-tuned 3B-8B models on vote choice match
  strong tabular baselines (CatBoost) with representative data and beat zero-shot; with tiny data zero-shot GPT-4o stays
  competitive. Closest published analogue of "LLM fine-tune vs tabular model on the same features".
- Ergul, Balcisoy, Bozkaya, arXiv:2502.15724 (2025): LoRA Mistral on bank transactions written as text beats CNN/LSTM on
  next merchant category F1 (numbers not on the abstract page).
- OPeRA (ACL 2026; already in our HF cache as `NEU-HAI/OPeRA`): fine-tuned Qwen2.5-7B, 33.86% F1 on final purchase
  prediction, +13.85 points over prompt-only.
- Liu & Zhang, "Tabular Foundation Models for Discrete Choice Estimation", arXiv:2607.13314 (2026): in-context tabular
  foundation models on a yogurt scanner panel beat hierarchical Bayes on held-out log-likelihood and hit rate (16x
  faster); individual heterogeneity matters most; gains largest at 10-40 purchase occasions per consumer. A non-LLM
  in-context competitor worth knowing about.

---

## 3. Lessons for us (talp73: person x 20 categories, forward in time)

Current bar (three-seed means from `/data/talp/is/latent/veri/talp73/README.md`, same `degerlendir.py`):

| arm | AUC | 1a accuracy | 4 within-person AUC |
|---|---|---|---|
| always "no" | | 0.7637 | 0.500 |
| survey model (LightGBM, no LLM) | 0.683 | 0.771 | 0.631 |
| Claude persona L1-haiku55 | 0.609 | 0.694 | 0.585 |
| Claude persona + history L2-sonnet55 | 0.772 | 0.790 | 0.715 |
| history model `gecmis` (LightGBM, no LLM) | 0.789 | 0.807 | 0.728 |
| H2cf-haiku55 (logit stack of `gecmis` + calibrated L2) | 0.789 | 0.807 | 0.728 |

Accuracy sits on a 76% base rate, so it hides differences; AUC, Brier/log-loss and within-person AUC are the ones that
move. The stack H2cf added +0.001 to +0.003 AUC over `gecmis` (TALP `uretim/talp73/kod/llm/SONUC-v1.8.md`).

### 3.1 Corpus size and QLoRA cost on Qwen3.8-27B

Training data available without touching test people (seed 0): 4,337 train people; 1,236,230 purchase events before the
2022-01-01 cut-off (median 181 per person, 34 months with a purchase, 1,828 raw categories; `gecmis_olay.parquet`).
Rough token counts (characters / 4, our estimate):

| transcript format | tokens (seed 0, train people) |
|---|---|
| event list, category + title cut to 80 chars | ~30 M |
| event list, category only | ~7 M |
| monthly 20-category grid (`<<0>>`/`<<1>>` per category per month, ~48 months) | ~12-27 M, depending on wording |
| one prompt per person -> 20 answers (current `centaur_qlora` / K7) | ~1.5 M (KATALOG estimate) |

Throughput is extrapolated, not measured: Centaur 70B QLoRA was about 590 tok/s on one A100 (unsloth); Oh & Gobet needed
up to 49 h for a 14B LoRA on the full 253 M tokens (about 1,400 tok/s). For a 27B dense model take 700-1,500 tok/s on an
A100/H100-80GB class card. Then:

| GPU | fits (4-bit base ~15-17 GB, r = 8-16, gradient checkpointing) | 7 M tokens | 30 M tokens |
|---|---|---|---|
| 24 GB | yes at 2-4k context, micro-batch 1; long transcripts must be windowed | ~4-8 h (slower card) | not tonight |
| 48 GB | yes at 8-16k context | ~2-4 h | ~8-16 h |
| 80 GB | yes at 32k context, as in the paper | ~1.5-3 h | ~6-12 h |

Caveats: Qwen3.5/3.8 use hybrid linear-attention (DeltaNet-style) layers per `KATALOG.md`; whether unsloth/bitsandbytes
QLoRA supports Qwen3.8 kernels at full speed was not verified. The FP8 checkpoint is for inference; 4-bit QLoRA normally
quantises from the BF16 weights (about 54 GB to download on the GPU machine). One epoch only, as in the paper and Oh &
Gobet.

Practical consequence: a pooled model on 1.5-7 M tokens fits in one evening on any of the three cards; 30 M tokens with
titles needs the 80 GB card or a cut. Minitaur-8B (and an 8B Qwen) is 3-4x cheaper and, per Oh & Gobet, loses little
in-distribution, which is the regime of talp73 (same task, new people, next period).

### 3.2 Evaluation designs that test our claim hardest

Our claim is "an LLM trained on behaviour transcripts predicts a new person's next purchases better than a model built on
the same history". In order of strength:

1. **Held-out people and forward in time together** (already the talp73 protocol: train people outside the sample, labels
   2022-H1; test 400 sampled people, 2022-H2). Also run the second split in `/data/talp/is/talp73_2021h2/`. This is the
   Centaur "held-out participants" design plus a time shift Centaur never had.
2. **Same-information comparison.** The LightGBM `gecmis` model must get exactly the features the transcript carries
   (including any extra training windows; see 3.3). Otherwise a win is a data win, not a model win.
3. **Held-out categories** (train with 15 of 20 category labels, test on the other 5). Analogue of Centaur's structure
   change; the GBDT cannot do this without retraining, the LLM can try. A clean place where an LLM can show something the
   tabular model cannot.
4. **Liu & Ding / Oh & Gobet ablations**: shuffle month order, mask category names ("category 7"), drop the instruction,
   give another person's history. If the score barely moves, the model uses base rates, not the person.
5. **Cover-story change**: rename categories with synonyms or translate the transcript; a robust model should not move.
6. **New dataset**: train on talp73, test zero-shot on Twin-2K-500 or OPeRA tasks (in the local cache). Analogue of "new
   domain"; expect a drop, report it.
7. **Generation, not only prediction** (Namazova et al.): sample 6-month baskets and compare category co-purchase and
   count distributions to the real 2022-H2 baskets.

### 3.3 What it takes to beat the LLM-free history model

- Read probabilities, do not sample answers: p(buy) = softmax over the logits of `<<1>>` vs `<<0>>` (or "yes"/"no").
  AUC and Brier need a score; a hard answer gives accuracy only, and accuracy near the 76% base rate hides gains.
- Give the model information the GBDT does not use well: item titles, the 1,808 categories outside the 20, order and
  timing of events, life-event survey answers. A transcript of the same 20 counts as the GBDT sees mostly re-learns it.
- More supervision per person: rolling cut-offs (every 6 months from 2019 to 2021-H2, each with the following window as
  target) multiply training rows about 5x. Give the GBDT the same rolling rows (rule 2 above).
- Calibrate on `ic_dogrulama` people only, then stack with `gecmis` (logit average or a small logistic layer), as H2cf
  did. Expect gains of the order of 0.005-0.02 AUC at best; the H2cf stack gave +0.001-0.003. Any claim needs the paired
  person-level bootstrap in `degerlendir.py`.
- Honest outcome to plan for: a fine-tuned 27B ties `gecmis`. That is still a result ("an LLM simulator trained on
  behaviour reaches the tabular ceiling, a prompted one does not"), and the held-out-category test is where it can win.

### 3.4 Five recommendations

1. Before training anything, run **Minitaur-8B and the Qwen3.8-27B base zero-shot** on a talp73 transcript (monthly grid,
   `<<0/1>>`) and read the token probabilities: this tells us whether Psych-101 pre-training helps on purchases at all
   (no published evidence either way).
2. **Train one pooled QLoRA** on Qwen3.8-27B with the paper's recipe (r 8, all linear, 1 epoch, lr 5e-5, wd 0.01,
   effective batch 32, warm-up 100, loss on `<< >>` tokens only), monthly-grid transcripts of train people, rolling
   cut-offs. Start with the 1.5-7 M token formats; add titles only if time remains.
3. **Fix the evaluation first**: designs 1-4 of 3.2 on the same `degerlendir.py`, with `gecmis` fed the same rolling rows.
4. **Run the in-context arm via API now** (section 4.3), because it needs no GPU and gives the comparison row
   "same transcript, no training" before the GPU arrives.
5. **Report the stack, not only the single model**: fine-tuned LLM alone, `gecmis` alone, and calibrated stack, with
   ablations; that answers the overfitting criticism (Liu & Ding) with our own controls.

---

## 4. API options (status as of 23 Oct 2026)

### 4.1 Fine-tuning APIs

| provider | status | notes |
|---|---|---|
| OpenAI | winding down | Docs: "OpenAI is winding down the fine-tuning platform." "The platform is no longer accessible to new users." Eligible existing customers can start jobs until 6 Jan 2027. SFT models: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano (2025-04-14). gpt-4.1-nano and its fine-tunes shut down 23 Oct 2026 (today); gpt-4.1 / -mini on 14 Jan 2027. Not usable for us (no existing account here). |
| Anthropic / Amazon Bedrock | only Claude 3 Haiku confirmed | Bedrock fine-tuning of Claude 3 Haiku is GA since 1 Nov 2024 (US West, Oregon). No official source found for fine-tuning any newer Claude model; the Anthropic API itself has no public fine-tuning. Current Bedrock status not verified (AWS doc page did not render). |
| Together AI | available | LoRA SFT/DPO on `Qwen/Qwen3.8-27B` (SFT context 131,072), `Qwen/Qwen3.5-27B`, `google/gemma-4-31B-it` (49,152), `meta-llama/Meta-Llama-3.1-8B-Instruct-Reference`; no Llama-3.1-70B (Llama-3.3-70B instead). Price page: Qwen3.8 27B LoRA SFT $1.05 / 1M training tokens (tokens x epochs), $4 minimum per job. Serverless LoRA inference is discontinued: adapters need a dedicated endpoint (hourly GPU price, not checked). Whether Together exposes token logprobs on those endpoints was not checked. |
| Fireworks AI | available | SFT (LoRA) for Qwen3 dense, Llama 3, Gemma, DeepSeek and others; JSONL, 3 to 3 M examples. Serverless LoRA only on some bases (Qwen3-4B listed as not serverless). Qwen3.8-27B support and current prices not verified. |

Cost check (ours): a 7 M-token corpus x 1 epoch on Together Qwen3.8-27B is about $7; 30 M tokens about $32; plus
endpoint hours for inference. Training on our own GPU and serving the adapter there avoids the endpoint.

Accounts: none of these providers has a key on this server (see 4.4). Opening Together/Fireworks needs a payment card,
which agents may not do; that is a Tuna decision.

### 4.2 Loss masking on APIs

Centaur's loss is only on `<< >>` tokens. Chat-style SFT APIs train on assistant turns, so the transcript must be
written as alternating user (stimulus: month, context) and assistant (`<<0/1 per category>>`) turns, as
`centaur_qlora(chat=True)` already renders. Whether Together/Fireworks mask all earlier assistant turns or only the last
one was not verified; check per provider before trusting it.

### 4.3 "In-context Centaur" (no training, API only)

Give Claude the same transcript Centaur would see (instructions, then the person's past months with choices in `<< >>`)
and ask for the next window. Two variants:

- **Own history only**: the person's monthly grid up to the cut-off -> probability per category. Close to L2, but in the
  Centaur transcript format with per-month order instead of a summary. 996 test people x ~5-15k tokens ~ 5-15 M input
  tokens per model run.
- **Population in context** (the closer analogue of learning across participants): a fixed prefix of K train people's full
  transcripts *with their next-window outcomes*, then the test person. The prefix is identical for every call, so prompt
  caching makes it cheap; K is limited by the context window (with ~10k tokens per person, K ~ 50-80 in a 1M-token window).

Constraints: the Claude Messages API does not return token log-probabilities as far as we know, so the arm must ask for a
verbalised probability per category (as L1/L2 did) or average several samples. OpenRouter models that return logprobs
would allow the Centaur-style read-out; which ones do was not checked. The same `degerlendir.py` and the same
person-level bootstrap apply; the arm is a row next to L2 and `gecmis`, and it is the fair "prompted" counterpart of the
fine-tuned model.

### 4.4 Key files on this server (names only, contents not read)

`/srv/lighthouse/hackathon-coz/secrets/`: `anthropic.key`, `anthropic-talp.key`, `anthropic-tuna.key`,
`anthropic-qloo.key`, `openrouter.key`, `xai.key`, `colab-token.json` (plus non-API files). No OpenAI, Together,
Fireworks or AWS credentials. So the in-context arm can run on Claude or OpenRouter today; API fine-tuning needs a new
account.

---

## Sources

- Binz et al. 2025, Nature, DOI 10.1038/s41586-025-09215-4; arXiv:2410.20268 v3 (https://arxiv.org/html/2410.20268v3)
- https://huggingface.co/datasets/marcelbinz/Psych-101 ; .../Psych-201 ; .../Psych-201-discrete ;
  .../Psych-201-discrete-agentic ; https://huggingface.co/marcelbinz/Llama-3.1-Minitaur-8B ; https://github.com/marcelbinz/Psych-201
- Liu & Ding, National Science Open 2025/26, DOI 10.1360/nso/20250053 (via https://www.sciencedaily.com/releases/2026/04/260429102035.htm)
- Oh & Gobet 2026, arXiv:2608.05224
- Namazova et al. 2025, arXiv:2508.07887
- Bowers et al. 2025, PsyArXiv 10.31234/osf.io/v9w37 (via https://www.bristol.ac.uk/psychology-neuroscience/news/2025/121---ai-model-of-human-mind.html)
- Orr et al. 2025, arXiv:2510.03311
- Kolluri et al. 2025, arXiv:2509.05830 ; Holtdirk et al. arXiv:2509.25063 ; Ergul et al. arXiv:2502.15724 ;
  Liu & Zhang arXiv:2607.13314 ; OPeRA (ACL 2026, aclanthology 2026.acl-long.2034)
- OpenAI: https://developers.openai.com/api/docs/guides/supervised-fine-tuning
- Bedrock / Claude 3 Haiku: https://aws.amazon.com/about-aws/whats-new/2024/11/fine-tuning-anthropics-claude-3-haiku-amazon-bedrock
- Together: https://docs.together.ai/docs/fine-tuning-models , https://www.together.ai/pricing , https://docs.together.ai/docs/lora-inference
- Fireworks: https://docs.fireworks.ai/fine-tuning/supervised-fine-tuning
- Local: `/data/talp/is/latent/veri/talp73/README.md`, `ozet.json`, `gecmis_olay.parquet`; TALP repo `uretim/talp73/PROTOKOL.md`,
  `uretim/talp73/kod/llm/SONUC-v1.8.md`

# Latent persona injection: method catalogue

Planning note for the GPU day (24 Oct, TALP Simulation Hackathon). Question: if we turn a person's survey answers
and purchase history into a vector and put it *inside* the model (soft tokens, residual steering, adapters) instead
of into the prompt, do we beat the LLM-free models on talp73, and does the LLM keep its counterfactual (price,
campaign) response?

Nothing here is measured. GPU costs are order-of-magnitude estimates for one 24-80 GB card and must be checked on the
real card. Every reference was checked against arXiv/ACL listings on 23 Oct 2026; where only the abstract was read,
the claim is limited to what the abstract says.

## Where we start (talp73, person x category buy/no-buy, 400 people x 20 categories x 3 seeds)

| arm | accuracy (p >= 0.5) | note |
|---|---|---|
| always "no" | 0.764 | base rate |
| Claude persona, survey only (L1) | 0.694 | |
| survey-only LightGBM (T2) | 0.771 | same information as the persona |
| LLM persona + history (L2) | 0.79-0.80 | |
| history LightGBM (T3) | 0.807 | best person-level model |

Within-person AUC: persona minus no-persona = +1.4 points. The information is in the history, not in the persona
text. So the bar for any latent arm is **T3 on the same split**, and the second test is the counterfactual one.

Data on disk: `/data/talp/is/talp73/panel/` (survey `anket.parquet` 5,027 x 29; per-category history summary
`gecmis_ozet.parquet`), raw item-level purchases `/data/talp/veri/open-ecommerce/amazon-purchases.csv` (date, price,
quantity, title, ASIN, category), Twin-2K-500 pricing survey already reshaped at
`/data/talp/is/latent/_twin_fiyat_kesif.parquet` (person x product x price -> buy), which gives a per-person
ground truth for price response.

Training data for anything learned must follow PROTOKOL V3: features cut at 2022-01-01, labels 2022-01..06, test
people excluded. Roughly 1.7k-4.6k people x 20 categories, depending on whether non-fully-observed users are used.
That is small: learned projectors must be low-capacity (a few hundred thousand parameters, not millions).

## Target models

| model | type | notes for latent work |
|---|---|---|
| Qwen/Qwen3.8-27B (main) | dense hybrid: 48 Gated DeltaNet + 16 full attention, hidden 5120 | official FP8 (~31 GB). See "hybrid risk" below. |
| google/gemma-4-31B-it | dense | check the config for the local/global attention pattern before hooking |
| google/gemma-4-26B-A4B-it | MoE, ~4B active | steering can move the router; a confound and a curiosity |
| Qwen/Qwen3.5-27B (older) | hybrid, same family | official SAEs (Qwen-Scope): `SAE-Res-Qwen3.5-27B-W80K-L0_50/100`, all 64 layers, trained on the instruct model |
| Qwen/Qwen3-32B or Qwen2.5-32B-Instruct (2024-25) | plain transformer | control: same method without linear attention |

**Precision.** Inference, probing, CAA/persona-vector extraction and steering are fine in FP8 or 8-bit. Extract a
steering vector in the same precision you apply it in (the residual scale shifts slightly between bf16 and FP8).
For training (projector, soft prompt, LoRA) through a frozen quantised base, the standard recipe is QLoRA (4-bit NF4,
Dettmers et al. 2023, arXiv:2305.14314); backprop through an FP8 checkpoint depends on the kernels and the card (FP8
tensor cores are Hopper/Ada; on A100 expect a dequantise path). Verify on the real card before planning around it.
Rough weight memory for 27-32B: bf16 ~55-64 GB, FP8/int8 ~28-34 GB, NF4 ~15-18 GB.

**Hybrid risk (Qwen3.5 / 3.8).** Our inference, not a published result. (1) The residual stream is still per-token,
so residual steering and soft tokens work mechanically. (2) A vector added before a Gated DeltaNet layer is written
into a recurrent state that accumulates over all later tokens, so the effective strength depends on how many
positions you steer; sweep alpha separately for "every position" and "prompt end only". (3) Prefix tuning and
KV-cache steering only have a KV cache to act on in the 16 full-attention layers; the DeltaNet analogue is tuning or
editing the initial recurrent state. PEFT's prefix tuning assumes a plain KV cache and will probably not work on the
hybrid cache. (4) Speed needs the flash-linear-attention / causal-conv1d kernels; the torch fallback is slow. Paulo,
Marshall & Belrose 2024 (arXiv:2404.05971) found CAA transfers to Mamba and RWKV but is generally weaker than on
transformers, and that steering the recurrent state plus the residual is not additive. Running the same arm on
Qwen3-32B (no linear attention) separates "the method fails" from "the hybrid fails".

## A. Persona extraction (person -> vector)

| # | method | mechanism | trains? | data | GPU (27-32B) | talp73 fit | source |
|---|---|---|---|---|---|---|---|
| A1 | Survey features -> MLP / autoencoder | one-hot + standardised survey columns, optionally compressed by a small (V)AE to 16-64 dims | small, CPU | 5k people | none | high; it is T2's information as a vector | Kingma & Welling 2013, arXiv:1312.6114 |
| A2 | History bag (counts, TF-IDF, EASE) | per-person counts over all categories / ASINs, TF-IDF or log weighted; EASE or Mult-VAE give a denoised profile | CPU | raw purchases | none | high; this is where T3's signal lives | Steck 2019 (EASE) arXiv:1905.03375; Liang et al. 2018 (Mult-VAE) arXiv:1802.05814 |
| A3 | Matrix factorisation user factors | implicit ALS / truncated SVD / BPR on user x category (or user x ASIN); the user factor is the vector | CPU, minutes | raw purchases | none | high; the "collaborative modality" CoLLM/LLaRA inject | Hu, Koren & Volinsky 2008 (ICDM, no arXiv); Rendle et al. 2009 (BPR) arXiv:1205.2618 |
| A4 | Sequential encoder | GRU4Rec / SASRec / BERT4Rec over the dated purchase sequence; final state = user vector; item embeddings can be initialised from title text | yes, small GPU or CPU | raw purchases | negligible | medium; 6-month category horizon is coarse, order may add little over A2 | Hidasi et al. 2015 arXiv:1511.06939; Kang & McAuley 2018 arXiv:1808.09781; Sun et al. 2019 arXiv:1904.06690 |
| A5 | Text embedding model | render survey + history summary as text, embed with a sentence encoder (`intfloat/multilingual-e5-base` is cached) | no | none | none / tiny | medium; cheap, model-agnostic, but loses counts | Reimers & Gurevych 2019 arXiv:1908.10084; Wang et al. 2022 (E5) arXiv:2212.03533 |
| A6 | LLM's own hidden state | feed the profile text to the target LLM; mean-pool or last-token residual at a middle layer; sweep layers | no | none | one forward per person, minutes for 5k people | high; the vector lives in the space we inject into, and it is the readout for K1 | Gurnee & Tegmark 2023 arXiv:2310.02207; BehnamGhader et al. 2024 (LLM2Vec) arXiv:2404.05961 |
| A7 | SAE features | pass A6 activations through a residual SAE; sparse feature activations = interpretable person code | no (SAEs exist for Qwen3.5) | none | small; SAE is 5120 x 81920 per layer | medium as a predictor, high for explanation | Qwen-Scope tech report arXiv:2605.11887; Cunningham et al. 2023 arXiv:2309.08600; Gemma Scope arXiv:2408.05147 |
| A8 | Contrastive trait directions | for each survey trait, residual mean-difference between contrasting persona prompts (young vs old, high vs low income, ...); person = sum of trait value x direction | no | pairs of prompts | minutes | medium; only as good as the traits chosen; this is `CAASteer` in `simengine/persona/inject/finetune.py` | Panickssery (Rimsky) et al. 2023 arXiv:2312.06681; Chen et al. 2025 arXiv:2507.21509 |
| A9 | Behaviour-contrastive directions | buyers vs non-buyers of category c: mean-difference of A6 states -> a "propensity direction" per category | no (it is a mass-mean probe) | labels | minutes | high; doubles as readout and as a steering vector | Marks & Tegmark 2023 arXiv:2310.06824; Im & Li 2025 arXiv:2502.02716 |
| A10 | Collaborative "data-driven persona" embeddings | embed people by their answers (CF), cluster, centroid = segment persona | small | answers | none | high for segment-level (yorum 5) | Li et al. 2023 arXiv:2311.04978 (NAACL 2024) |
| A11 | Filtered history steering | keep only history items that causally carry preference before building the vector | yes | history | moderate | medium; addresses noisy long histories | SteerX, arXiv:2510.22256 |

Combinations are allowed: concatenate A1 + A3 (+ A5) and let the injection projector (B7/B9) choose.

## B. Injection / steering (vector -> model) and reading out

| # | method | mechanism | trains? | data | GPU (27-32B) | hybrid risk | talp73 fit | source |
|---|---|---|---|---|---|---|---|---|
| B1 | CAA | v = mean(h_pos - h_neg) at layer L over contrastive pairs; add alpha*v to the residual at every later position | no | ~100s of pairs | forward only | medium (alpha needs re-sweeping, see above) | per-trait steering (A8), per-category propensity (A9) | Panickssery (Rimsky) et al. 2023 arXiv:2312.06681 |
| B2 | ActAdd | same, from a single prompt pair | no | 1 pair | trivial | medium | quick sanity arm only | Turner et al. 2023 arXiv:2308.10248 |
| B3 | Persona vectors | automated: trait description -> contrasting system prompts -> judged rollouts -> response-token mean difference; also used for monitoring drift | no | generated | generation + judge (local judge; no API) | medium | builds richer trait directions than A8; lets us measure persona drift | Chen et al. 2025 arXiv:2507.21509 |
| B4 | RepE / ITI; mean-diff vs PCA vs probe | RepE: PCA of activation differences (LAT); ITI: pick heads by probe accuracy, shift along the probe direction | no (probe is a tiny fit) | labelled prompts | forward only | low-medium; ITI acts on attention heads, only 16 of 64 layers are softmax attention in Qwen3.x hybrids | choose the direction estimator; mean-diff is the theory-backed default | Zou et al. 2023 arXiv:2310.01405; Li et al. 2023 arXiv:2306.03341; Im & Li 2025 arXiv:2502.02716 |
| B5 | Head-level linear interventions for continuous attributes | probe heads that linearly encode a continuous perspective, then shift them | no | labelled prompts | forward only | as B4 | continuous traits (income, age) rather than binary contrasts | Kim, Evans & Schein 2025 arXiv:2503.02080 |
| B6 | SAE-feature steering | clamp or add decoder directions of chosen SAE features | no | SAE | forward only | as B1 | interpretation first, prediction second | Qwen-Scope arXiv:2605.11887; AxBench (Wu et al. 2025, arXiv:2501.17148) finds SAE steering not competitive with prompting or mean-diff |
| B7 | Learned steering (projector to residual) | u (A1-A5 vector) -> small MLP -> vector added at layer L; trained with BCE on the model's P(yes) for each category; LLM frozen | yes, backprop through layers L..end | training people x 20 labels | ~3x forward; tens of minutes per epoch at 1-2M tokens (estimate) | medium; prefer L just before a full-attention layer | **core arm**: tests "does the history vector add on top of T3 inside the LLM" | ReFT/LoReFT, Wu et al. 2024 arXiv:2404.03592; HyperSteer, Sun et al. 2025 arXiv:2506.03292 |
| B8 | KV-cache steering | one-shot edit of cached keys/values after the prompt | no | pairs | trivial | high: only 16 layers have a KV cache | low priority on Qwen3.x; fine on Gemma 4 / Qwen3-32B | Belitsky et al. 2025 arXiv:2507.08799 |
| B9 | Soft prompt / prefix / P-tuning (persona-conditioned) | u -> projector -> k virtual input tokens (E2P uses k = 1); prefix tuning learns per-layer KV prefixes | yes | as B7 | as B7 but backprop through all layers | low for input soft tokens (architecture-agnostic); high for prefix tuning | **core arm**, safest on the hybrid | Lester et al. 2021 arXiv:2104.08691; Li & Liang 2021 arXiv:2101.00190; Liu et al. 2021 arXiv:2103.10385, v2 arXiv:2110.07602; Embedding-to-Prefix arXiv:2505.17051; Li et al. 2023 arXiv:2311.04978; PPlug arXiv:2409.11901 |
| B10 | LLaVA-style projector for behaviour | treat the collaborative/sequence embedding as a modality; projector to token space, sometimes with LoRA and curriculum | yes | rec data | as B9 (+ LoRA) | low (input side) | the recommender literature's version of B9 | Liu et al. 2023 (LLaVA) arXiv:2304.08485; CoLLM arXiv:2310.19488; LLaRA arXiv:2312.02445; E4SRec arXiv:2312.02443; User-LLM (cross-attention + soft prompt, Perceiver) arXiv:2402.13598 |
| B11 | Person / segment LoRA, distribution fine-tune | per-user PEFT (OPPU), composable pieces (Per-Pcs); SubPOP fits the option distribution of subgroups; Centaur fine-tunes on trial-level choices | yes | per-person data is tiny; segment or pooled is realistic | QLoRA 27B: hours for millions of tokens; ~1.7k people x 1 prompt with 20 answers is ~1.5M tokens | low (LoRA on linear layers incl. DeltaNet projections) | pooled "Centaur-lite" on profile -> 20 answers; per-person LoRA is overkill for our data | Tan et al. 2024 (OPPU) arXiv:2402.04401; Tan et al. 2024 (Per-Pcs) arXiv:2406.10471; Suh et al. 2025 (SubPOP) arXiv:2502.16761; Binz et al. 2024 (Centaur) arXiv:2410.20268; Dettmers et al. 2023 (QLoRA) arXiv:2305.14314 |
| B12 | Readout: linear probe (LLM as encoder) | hidden state of the profile prompt -> logistic regression per category (or one model with category embedding) | tiny fit | labels | forward only | none | **strongest prior**: direct test of whether the LLM's representation adds to T3 | Alain & Bengio 2016 arXiv:1610.01644; Marks & Tegmark 2023 arXiv:2310.06824 |
| B13 | Readout: logit P(yes) / full distribution | read next-token probabilities for Yes/No or option letters instead of sampling | no | none | forward only | none | the readout for every steering arm | Santurkar et al. 2023 arXiv:2303.17548; caveats: Wang et al. 2024 arXiv:2402.14499 (first-token probability can disagree with the text answer in chat models); Meister et al. 2024 arXiv:2411.05403 (log-probability readout can underestimate what the model gets right when it verbalises a distribution), so keep a verbalised-probability arm next to it |

Reliability warnings that apply to B1-B8: steering effects vary a lot by input and by model, and some models get
worse (Tan et al. 2024 arXiv:2407.12404; Da Silva et al. 2025 arXiv:2504.04635). AxBench (arXiv:2501.17148) found
prompting beat every representation method for steering, while difference-in-means was best for concept detection.
That favours reading out (B12/B13) and learned injection (B7/B9) over hand-built steering for a prediction task.

## Recommended arms for one GPU today

All arms write `tahmin/<method>_s<s>.parquet` (uid, kategori, p) and go through `uretim/talp73/kod/degerlendir.py`,
so they sit in the same table as T2/T3/L1/L2. Learned arms train on the PROTOKOL V3 prior window only. Stop rule per
arm: if seed 0 is below T3 on within-person AUC and accuracy, do not run seeds 1-2.

1. **K1 Probe readout (A6 + B12).** Profile text (survey; survey + history summary) -> residual at 4-6 layers of
   Qwen3.8-27B -> logistic regression; also stacked with T3. Cheapest direct answer to "does the LLM representation
   add to the LLM-free model". Repeat on Gemma-4-31B, Gemma-4-26B-A4B, Qwen3.5-27B and Qwen3-32B/Qwen2.5-32B:
   forward-only, so this is where the cross-model comparison is cheap.
2. **K2 Local prompt baseline (B13).** L1/L2 prompts on the local model, read P(yes). Every injection arm is compared
   with "same model, persona in the prompt", not with Claude.
3. **K3 Contrastive trait CAA (A8 + B1).** 8-10 survey traits -> directions at a mid layer; person = sum of trait x
   direction; alpha sweep (every position vs prompt end). No training. Tests "persona inside instead of in the prompt"
   at zero cost; also run on Qwen3-32B to separate hybrid effects.
4. **K4 Learned residual projector (A1 + A3 + A5 -> B7).** Small MLP from the concatenated user vector to a residual
   vector at one layer before a full-attention block; BCE on P(yes). The core test of injecting history.
5. **K5 Persona soft token (same u -> B9, k = 1..4 tokens).** E2P / Li et al. style; architecture-agnostic, so the
   control for K4's hybrid risk.
6. **K6 SAE persona features on Qwen3.5-27B (A7 + B6).** Which features separate buyers from non-buyers per
   category; SAE features as a probe (vs K1 dense); clamp 1-3 features as a steering demo. Gives the "why" slide.
7. **K7 Centaur-lite QLoRA (B11).** One prompt per training person with the 20 answers as the target; QLoRA on one
   model (Qwen3.8-27B or Gemma-4-31B), evaluate held-out people. Only if K1-K5 leave GPU time; it is the slowest arm.
8. **K8 Counterfactual check (applies to K2-K7).** For each arm, sweep price (and a discount line) in the prompt and
   read P(buy). Report direction and slope against the unsteered prompt persona, and on Twin-2K-500 pricing
   (`_twin_fiyat_kesif.parquet`) against each person's own price answers. This answers "is the counterfactual power
   kept", which no LLM-free model can do.

Order: K1, K2 (forward only, minutes) -> K3 -> K4/K5 (training) -> K6 -> K7 if time. K8 runs on whatever finished.

## GPU-only things the API cannot do

- **Exact distributions, no sampling.** Read P over Yes/No, option letters or price buckets directly; entropy per
  person as uncertainty; replaces the 3-sample average (L1x3) at zero extra cost.
- **Dense counterfactual curves.** With prefix/state caching, sweep a price from 1 to 100 per person and per category
  for all 5,027 people: demand curves and elasticities per segment, cheap.
- **Population-scale simulation.** Every person x category x scenario, tens of millions of tokens in a night on one
  card (with vLLM), against a few dollars per 10k calls on the API.
- **Activation patching on the persona.** Swap one survey field (or its token span's activations) between two people
  and measure the change in P(buy): which fields the model actually uses. An activation-level `persona_swap` control.
- **Persona arithmetic.** person + "income one bracket up" direction -> counterfactual demographics without
  rewriting the profile; interpolation between two people or segment centroids.
- **SAE explanations.** Name the features behind a predicted purchase (Qwen-Scope on Qwen3.5); persona-vector
  monitoring of character drift over a long interview (Chen et al. 2025).
- **Base vs instruct.** Run the base checkpoint as a simulator; SimBench (Hu et al. 2025, arXiv:2510.17516) reports
  a trade-off between alignment and simulating diverse opinions.
- **Centaur-style fine-tune and distillation.** Fine-tune on choice transcripts; distil the big model's
  distributions into a small model that serves the whole panel.
- **Layer-wise decision tracing.** Logit lens / probes per layer show where the buy decision forms and whether
  injected vectors arrive before it.

## Caveats

- No arm here has been run. The costs above are estimates; the hybrid-architecture risks are inferences from the
  architecture and from Paulo et al. 2024, not results on Qwen3.x.
- I found no paper that injects a purchase-history vector into an LLM and compares against a strong history GBDT on
  person-level purchase prediction; the closest are the recommender projectors (B10) and data-driven personas
  (arXiv:2311.04978), which compare against weaker baselines.
- Gemma 4 SAE availability and Gemma 4's exact attention layout were not checked.

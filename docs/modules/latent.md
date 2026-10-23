# `simengine.persona.latent`

The persona as a vector inside a local open model, instead of as text in the prompt. A person vector is extracted
(survey, purchase history, a text embedding, the model's own hidden state, contrastive trait directions, SAE codes).
It is then read out (a probe on hidden states, P(yes) from the logits) or put inside the model (CAA residual
steering, a learned residual projector, soft tokens). Every arm is compared with the encoder-only control that sees
the same vector without an LLM. Pre-registration: `docs/prereg/latent-persona-2026-10-23.md`. Runner:
`runs/latent/kos.py`.

**Status: no number has been measured on real model weights yet.** The tests use tiny random models (Qwen3.5
hybrid, Gemma 4 dense and MoE, Qwen3 dense and MoE, Qwen2, Llama, Mistral) and a byte tokenizer, so they need no
download. The runner has been smoke-tested three ways. Every arm ran with tiny models on synthetic tables. The
LLM-free arms ran on the prepared tables (`--kucuk`). `prm-q35s-L1` ran on CPU with Qwen3.5-2B and the prepared
tables.
torch and transformers are imported lazily: `import simengine.persona.latent` works without them.

```python
from simengine.persona import latent as lt
ld = lt.load("q38", "bf16")                          # Qwen3.8-27B, text only, frozen; lt.tiny("qwen3_5") for tests
p, mass = lt.yes_prob(ld, prefixes, suffixes)        # prefix-cached P(Yes)/(P(Yes)+P(No)), (N persons, M cells)
with lt.add_to_residual(ld.model, 30, vecs):         # steering hook on layer 30, removed on exit
    ...
H = lt.llm_states(ld, prefixes, [20, 40], pool="last")[20]   # residual states (E4)
probe = lt.Probe("logistic", pca=256, C=0.1, seed=0).fit(H, rows, cat, y)
```

## Pieces

| module | what |
|---|---|
| `models` | registry `MODELS` (q38, q38f, g31, g26, q35, q3, q25, q35s), `load(code, precision, device=)`, layer and embedding lookup for the Qwen3.5/3.8 multimodal wrapper, Gemma 4, Qwen2/3, Llama and MoE; `tiny(family)` for tests; `check_backward` |
| `hooks` | `add_to_residual` (positions `last`, `last4`, `all`, int, slice; a tensor or a callable), `capture`, `replace_embeddings` (soft tokens on every family; Gemma 4 cannot `generate(inputs_embeds)`), `residual_norm`, `embedding_norm` |
| `prompts` | G1/G2/G3 templates, `split` at the cell sentinel (prefix cached, suffix per cell), `short_split` for the injection arms, `answer_token_ids` (first-token variants, shared ids dropped) |
| `readout` | `yes_prob` / `option_probs`: each prefix is run once and its KV cache is reused for every cell (hybrid DeltaNet and sliding-window caches included); `Injection` adds a vector or soft tokens during the suffix pass; gradients flow for training |
| `encoders` | E1 `SurveyOneHot`, `SurveyAutoencoder`; E2 `history_features`, `ImplicitALS`, `BagSVD`; E3 `text_embedding` (e5); E4 `llm_states`; E5 `trait_directions`, `trait_person_vector`; E6 `behaviour_directions`; E7 `load_sae`, `sae_codes` |
| `heads` | `EncoderHead` (logistic per category or MLP with a category embedding), `Probe` (PCA + logistic, or mass-mean + Platt), `platt`, `stack_lgbm` |
| `steer` | directions (`mean_diff_direction`, `pca_direction`, `probe_direction`), `CAA`, `residual_projector`, `soft_token_projector`, `train_injector` / `predict_injector` (model frozen, BCE on the yes-logit), `derange` (swap control) |
| `tasks` | `load_g1/g2/g3`, `TestGuard` (raises `LeakError` when a test row or a test person reaches a fit), `write_pred` (talp73 schema, optional seal) |
| `select`, `eval` | validation-only grid choice (`choose`), `kisi_auc`; `score_g1` (talp73 evaluator), `score_g2` (MAD accuracy), `score_g3` (price slope s_hi, D5), `decide` (D1-D6) |

Every fitted piece (encoders, standardisation, PCA, heads, projectors) sees training persons only; test persons
are only transformed.

## Precision: is bf16 required?

No. A run is valid in any precision, as long as its vectors, probes and projectors are applied in the precision
they were made in. Each prediction's `.meta.json` records the precision.

- `bf16` is the reference. A 27-32B model needs about 54-64 GB.
- `fp8` uses the official `Qwen/Qwen3.8-27B-FP8` checkpoint (about 31 GB) and needs compute capability >= 8.9.
  On an older card it raises an error; there is no silent fallback.
- `nf4` and `int8` use bitsandbytes (`pip install bitsandbytes` into the venv first). They are refused for the
  Gemma 4 MoE model, because its 3-D expert tensors are not quantised.
- `--precision "fp8|nf4"` picks fp8 when the card supports it and nf4 otherwise.
- `--calibrate` checks that a backward pass works through the loaded precision. That backward pass is what the
  learned injectors (rsd, sft) need.
- Prereg item X5 compares bf16 and FP8 on the same cells. That comparison is the measured answer to the
  question; until it has run, there is none.

## Runner

```bash
PY="nice -n 19 python"; K=runs/latent/kos.py
$PY $K --list                                                    # arm ids, models, run log
$PY $K --arm enc-an enc-gc enc-angc enc-e5L1 enc-e5L2 stk-e5 --seed 0 1 2       # LLM-free controls (CPU)
$PY $K --model q38 --arm prm-L2 prb-L2 stk --seed 0             # G1; the file is prb-q38-L2_s0.parquet
$PY $K --model q38 --arm rsd-angc rsd-angc-swap rsd-angc-zero sft-angc
$PY $K --model q38 --arm prm-tf prm-tf0 rsd-tf enc-tf lgbm-tf   # G3: test rows + price sweep in one file
$PY $K --model q35s --precision fp32 --device cpu --arm prb-L2 --kucuk          # smoke: 40/10/20 persons
$PY $K --calibrate --model q38 --precision bf16                 # prefill tok/s, one rsd and one sft step
$PY $K --score g1 --arm prb-q38-L2 --taban enc-angc gecmis --seed 0             # paired CIs, B 1000
```

- **One run** is arm x model x task x seed. The task follows from the arm: `tw` arms are G2, `tf` arms are G3,
  everything else is G1. G2 and G3 have one split (seed 0).
- **Resume.** A run whose prediction and `.meta.json` already exist is skipped; `--force` reruns it. The rsd and
  sft sweeps write each config's validation score to `<task>/secim/<arm>_s<s>.sweep.jsonl` and continue from
  there after a crash. The trained projector is saved in `<task>/proj/`, and the `-swap` and `-zero` controls
  reuse it. Seeds 1-2 reuse the config chosen on seed 0.
- **Log.** `kosular.jsonl` in the work folder gets one line per run: model, HF revision, precision, layer,
  coefficient, config, validation score, seconds, card, status. A crash is logged as `error` and the next run
  starts.
- **Outputs.** `talp73/tahmin/<arm>_s<s>.parquet` holds 8,000 rows (uid, kategori, p). Twin outputs go to
  `twin/tahmin/` and `twin_fiyat/tahmin/`. Real runs are sealed (`simengine.research.seal`, registry `.prereg`)
  before they are scored.
- **Calibrated and raw.** Prompt arms (prm, caa) also write `<arm>-raw`, the uncalibrated P(yes).
- **Smoke mode.** `--kucuk [FIT:VAL:TEST]` writes under `kucuk/`, keeps two configs per grid and seals nothing.
- **Work folder.** `--base`, default `$LATENT_BASE` or the runner's built-in default.

## Competition day: one command (tek komutla)

The prereg priority list for the card class, top to bottom. No new run starts after `--until`. A failed run is
logged and the plan moves on to the next one.

```bash
nice -n 19 python runs/latent/kos.py --plan 80gb --until 07:30     # or 48gb / 24gb
```

Before that, check the card and the plan:

1. `--calibrate --model q38 --precision bf16`, or `"fp8|nf4"` on a 48 GB card. This measures tok/s and checks
   backward.
2. `--plan 80gb --kucuk 20:8:8`. This is a dry run of the whole plan on 36 persons, with nothing sealed.

Plans:

- **80gb:** the LLM-free arms; then q35s K1-K5; then q38 K1+K2, K4, G3, K5; then q3; then q38 caa; then g31, q35,
  g26, q38 G2 and q25.
- **48gb:** q38 in fp8, or nf4 where fp8 is not supported. q3, g31 and q35 run in nf4 on L2 prompts only. sft runs
  last.
- **24gb:** every 27-32B model in nf4. The q38 G1 arms are capped at 1,000 fit persons and 400 validation persons.

Seeds 1-2 of the G1 LLM arms are not part of a plan. Under the prereg stop rule, run them by hand, and only when
the seed-0 score beats the arm's matched encoder-only control or T3:
`--model q38 --arm prb-L2 --seed 1 2`.

## Not done yet

- SAE readout (`sae-q35-L2`) and the QLoRA arm (`lora-*`) have no runner arm. The SAE file format of Qwen-Scope
  has not been checked against `load_sae`.
- The exploratory X block (field patching, persona arithmetic, layer tracing) is not implemented. X5 (bf16 vs
  FP8) is two plain runs.
- caa steers the last prompt token only. Steering every prompt position would need the injection inside the
  cached prefix.
- The fp8 and nf4 paths have not been run, because this machine has no GPU and no bitsandbytes.
- The Gemma 4 sliding window has been tested with short prompts. With left padding and prompts over 1,024 tokens
  it is not tested.

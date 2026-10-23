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
LLM-free arms ran on the prepared tables (`--kucuk`). Every G1 arm (seed 0, controls included) ran end to end on
CPU with Qwen3.5-2B in fp32 on the prepared tables with 40 fit / 10 validation / 50 test persons, and every
prediction file was read by the talp73 evaluator (23 Oct, 62 minutes; times, memory and the GPU estimate are in
`/data/talp/is/latent/DUMAN.md`; the scores of 50 persons mean nothing).
torch and transformers are imported lazily: `import simengine.persona.latent` works without them.

**Setup.** The package is not pip-installed. Run from the repo root, or set `PYTHONPATH` to it, and use the venv's
Python (there is no `python` on this machine's PATH). On the GPU box, `pip install flash-linear-attention
causal-conv1d` into the venv first: without them transformers runs the Qwen3.5/3.8 DeltaNet layers on its slow
reference path (it prints a warning). How much slower that is has not been measured.

```bash
cd /srv/lighthouse/hackathon-coz/projeler/simulation-engine
export PYTHONPATH=$PWD HF_HOME=/data/talp/hf
PY="nice -n 19 /data/talp/venv/bin/python"
```

```python
from simengine.persona import latent as lt
ld = lt.load("q38", "bf16")                          # Qwen3.8-27B, text only, frozen; lt.tiny("qwen3_5") for tests
L = ld.layer_index(0.5)                              # layer at half depth; layer counts differ per model
p, mass = lt.yes_prob(ld, prefixes, suffixes)        # prefix-cached P(Yes)/(P(Yes)+P(No)), (N persons, M cells)
with lt.add_to_residual(ld.model, L, vecs):          # steering hook, vecs (N, hidden) or (hidden,); removed on exit
    ...
H = lt.llm_states(ld, prefixes, [L], pool="last")[L]     # residual states (E4), (N persons, hidden)
probe = lt.Probe("logistic", pca=256, C=0.1, seed=0).fit(H, rows, cat, y)   # rows: person index, cat: item code
```

`prefixes`, `suffixes`, `rows`, `cat` and `y` are built from your tables as in the next section.

## Your own data in 30 minutes

`runs/latent/kos.py` only knows the three prepared tasks (talp73, Twin, Twin pricing). For a new dataset, use the
package directly. The recipe below is the whole pipeline: an LLM-free control, the persona as text (P(yes) and a
probe), and the persona inside the model (learned residual vector, soft tokens, a fixed CAA direction), all scored
on held-out persons. It runs as written in about 10 seconds on CPU with a tiny random model, so its numbers mean
nothing. To use it, replace the toy tables in step 1 and the `lt.tiny` line. `tests/test_docs_latent.py` runs this
block, so it stays in sync with the code.

You need two tables: `people` with one row per person (`uid`, a text profile, any survey or feature columns), and
`cells` with one row per person x item (`uid`, `item`, `y` 0/1). History or other side data goes into the text, into
`U`, or both.

<!-- test: latent-own-data -->
```python
import numpy as np
import pandas as pd
from simengine.persona import latent as lt
from simengine.persona.latent import prompts as P

# 1. Your data: one row per person, one row per cell (person x item, y 0/1). Replace this toy table.
rng = np.random.default_rng(0)
people = pd.DataFrame({"uid": [f"u{i}" for i in range(60)], "income": rng.integers(1, 6, 60),
                       "age": rng.integers(18, 70, 60)})
people["text"] = "- Income band (1-5): " + people.income.astype(str) + "\n- Age: " + people.age.astype(str)
items = ["books", "shoes", "games"]
cells = pd.DataFrame([(u, it) for u in people.uid for it in items], columns=["uid", "item"])
cells["y"] = (rng.random(len(cells)) < 0.3).astype(int)

# 2. Split by person (val chooses configs, test is scored once) and index the cells.
split = pd.Series(rng.choice(["fit", "val", "test"], len(people), p=[0.6, 0.2, 0.2]), index=people.uid)
fitp = people.uid.map(split).eq("fit").to_numpy()                       # fit persons (mask over people)
rows = cells.uid.map({u: i for i, u in enumerate(people.uid)}).to_numpy()  # person index of each cell
cat = pd.Categorical(cells.item, categories=items).codes                # item code 0..C-1 (not the name)
y = cells.y.to_numpy()
m = {s: cells.uid.map(split).eq(s).to_numpy() for s in ("fit", "val", "test")}
c = lambda s: np.stack([cat[m[s]], rows[m[s]], y[m[s]]], 1)             # injector cells: (item, person, y)

ld = lt.tiny("qwen3_5")          # GPU: lt.load("q38", "bf16") or lt.load("q38f", "fp8"); see Precision
L = ld.layer_index(0.5)          # never a hard-coded layer number: n_layers differs per model

# 3. LLM-free control: the same person vector, no LLM. Every LLM arm must beat this.
U = lt.SurveyOneHot(["income", "age"]).fit(people[fitp]).transform(people)
enc = lt.EncoderHead("logistic", C=0.1).fit(U, rows[m["fit"]], cat[m["fit"]], y[m["fit"]])

# 4. Persona as text: P(Yes) from the logits, one cached prefix per person, one suffix per item.
def msgs(text):
    q = f"Question: Will this customer buy {P.SENTINEL} next month? Answer Yes or No."
    head = f"About this customer:\n{text}\n\n" if text else ""
    return [{"role": "system", "content": "You predict the behaviour of one real customer."},
            {"role": "user", "content": head + q}]
prefixes = [P.split(ld.tok, msgs(t))[0] for t in people.text]
suffixes = [P.split(ld.tok, msgs(people.text[0]), cell=it)[1] for it in items]
p_yes, mass = lt.yes_prob(ld, prefixes, suffixes)                       # (persons, items)
print("Yes+No mass", round(float(mass.mean()), 3))                      # real model: should be near 1
cal = lt.platt(p_yes[rows, cat][m["fit"]], y[m["fit"]], cat[m["fit"]])  # calibrate on fit cells only

# 5. Probe: the model's own state of the persona prompt, then logistic per item.
H = lt.llm_states(ld, prefixes, [L])[L]                                 # (persons, hidden)
probe = lt.Probe("logistic", pca=64, C=0.1).fit(H, rows[m["fit"]], cat[m["fit"]], y[m["fit"]])

# 6. Persona inside the model: short prompt without the persona, person vector U added to the residual
#    stream (learned projector, model frozen), or as k soft tokens.
short = lambda it: P.fill(msgs(None), it)
pre_short = [P.short_split(ld.tok, short(it))[0] for it in items]       # prefix index = item code
tail = P.short_split(ld.tok, short(items[0]))[1]
rsd = lt.residual_projector(U.shape[1], ld.hidden_size)
rsd, hist = lt.train_injector(ld, rsd, "residual", pre_short, tail, U, c("fit"), U, c("val"), layer=L, epochs=1)
sid, stext = lt.soft_placeholder(ld.tok)
tail_soft = P.short_split(ld.tok, short(items[0]), soft_k=2, soft_token=stext)[1]
sft = lt.soft_token_projector(U.shape[1], ld.hidden_size, k=2, target_norm=lt.embedding_norm(ld.model))
sft, _ = lt.train_injector(ld, sft, "soft", pre_short, tail_soft, U, c("fit"), U, c("val"), epochs=1)

# 7. Fixed steering direction (CAA): fit persons with high vs low income; the coefficient is chosen on val.
d = lt.mean_diff_direction(H[fitp & (people.income >= 4)], H[fitp & (people.income < 4)])
z = ((people.income - people.income[fitp].mean()) / people.income[fitp].std()).to_numpy()
caa = lambda coef, s: lt.yes_prob(ld, pre_short, [tail], cells=np.stack([cat, 0 * cat, rows], 1)[m[s]],
                                  inject=lt.Injection("residual", np.outer(z, d).astype(np.float32), layer=L,
                                                      scale=coef))[0]
best = max([0.5, 1.0, 2.0], key=lambda k: lt.val_metrics_g1(caa(k, "val"), cells[m["val"]])["kisi_auc"])

# 8. Score on test, once. The swap control feeds every test cell another person's vector.
t = m["test"]
perm = lt.derange(len(U))
preds = {"enc": enc.predict(U, rows[t], cat[t]), "prompt": cal(p_yes[rows, cat][t], cat[t]),
         "probe": probe.predict(H, rows[t], cat[t]),
         "rsd": lt.predict_injector(ld, rsd, "residual", pre_short, tail, U, c("test"), layer=L),
         "rsd-swap": lt.predict_injector(ld, rsd, "residual", pre_short, tail, U[perm], c("test"), layer=L),
         "sft": lt.predict_injector(ld, sft, "soft", pre_short, tail_soft, U, c("test")),
         "caa": caa(best, "test")}
for name, p in preds.items():
    print(name, {k: round(v, 3) for k, v in lt.val_metrics_g1(p, cells[t]).items()})
```

Notes for a new dataset:

- **Check `mass` first.** It is the probability the model puts on the two answer words. Near 1 means the prompt
  works; a small value means the template or the answer words are wrong. Only each word's first token counts
  (`lt.answer_token_ids(ld.tok, words)` shows them). On the Qwen tokenizer `Evet` starts with the token `E`, so
  `words=("Evet", "Hayır")` also counts any answer starting with E as yes. For a Turkish prompt, keep
  "Answer Yes or No." in English, or check `mass` with the Turkish words before trusting them.
- **`cat` holds integer codes**, not item names. `rows` indexes `people`, `X`, `U` and `H` (all one row per person).
  Every fitted piece (one-hot, PCA, heads, projectors, calibration) sees fit persons only. Test persons are only
  transformed.
- **Picking settings.** Choose layer, coefficient and learning rate on the val persons only, with `lt.grid` and
  `lt.choose` (primary metric `kisi_auc`, the within-person AUC).
- **Swap control.** `rsd-swap` must lose clearly to `rsd`. If it does not, the projector learned the item
  base rates, not the person.
- **Writing predictions.** `lt.write_pred(out_dir, arm, seed, df, meta, n_expected=None)` needs `uid`, `kategori`,
  `p`. Pass `g1=False` for any other schema. Its default row count (8,000) is the talp73 test set.
- **Memory.** `yes_prob` runs `batch_size` prefixes at once, and each prefix's KV cache is copied once per item
  (`cell_batch` cells per forward). Lower both if the card runs out of memory.

## Pieces

| module | what |
|---|---|
| `models` | registry `MODELS` (q38, q38f, g31, g26, q35, q3, q25, q35s), `load(code, precision, device=)`, layer and embedding lookup for the Qwen3.5/3.8 multimodal wrapper, Gemma 4, Qwen2/3, Llama and MoE; `tiny(family)` for tests; `check_backward` |
| `hooks` | `add_to_residual` (positions `last`, `last4`, `all`, int, slice; a tensor or a callable), `capture`, `replace_embeddings` (soft tokens on every family; Gemma 4 cannot `generate(inputs_embeds)`), `residual_norm`, `embedding_norm` |
| `prompts` | G1/G2/G3 templates, `split` at the cell sentinel (prefix cached, suffix per cell), `short_split` for the injection arms, `answer_token_ids` (first-token variants, shared ids dropped) |
| `readout` | `yes_prob` / `option_probs`: each prefix is run once and its KV cache is reused for every cell (hybrid DeltaNet and sliding-window caches included); only the last position's logits are computed; `Injection` adds a vector or soft tokens during the suffix pass; gradients flow for training |
| `encoders` | E1 `SurveyOneHot`, `SurveyAutoencoder`; E2 `history_features`, `ImplicitALS`, `BagSVD`; E2c `SequenceEncoder` (GRU over the time-ordered purchase items -> 64-d u, self-supervised on the training persons' own earlier windows, early stopped on the validation persons; arms enc-sq, enc-angcsq, stk-sq, rsd-sq, sft-sq, added after the prereg lock, exploratory); E3 `text_embedding` (e5); E4 `llm_states` (a tuple of pools fills all of them from one forward pass); E5 `trait_directions`, `trait_person_vector`; E6 `behaviour_directions`; E7 `load_sae`, `sae_codes` |
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
- `--precision "fp8|nf4"` picks fp8 when the card supports it and nf4 otherwise (resolved once, before any
  model loads).
- `--calibrate` checks that a backward pass works through the loaded precision. That backward pass is what the
  learned injectors (rsd, sft) need.
- Prereg item X5 compares bf16 and FP8 on the same cells. That comparison is the measured answer to the
  question; until it has run, there is none. File names carry no precision, so run the second precision with
  another `--base`.

## Runner

```bash
PY="nice -n 19 /data/talp/venv/bin/python"; K=runs/latent/kos.py      # from the repo root
$PY $K --list                                                    # arm ids, models, run log
$PY $K --arm enc-an enc-gc enc-angc enc-e5L1 enc-e5L2 stk-e5 --seed 0 1 2       # LLM-free controls (CPU)
$PY $K --arm enc-sq enc-angcsq stk-sq --device cpu --seed 0  # sequence encoder u=sq (a few minutes on CPU)
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
  reuse it. Seeds 1-2 reuse the config chosen on seed 0. Selections, sweep lines and projectors are reused only
  in the precision they were made in: a control in another precision stops with an error, and a finished run
  asked for again in another precision is an error, not a silent skip.
- **Log.** `kosular.jsonl` in the work folder gets one line per run: model, HF revision, precision, layer,
  coefficient, config, validation score, seconds, card, status. A crash is logged as `error` and the next run
  starts.
- **Outputs.** `talp73/tahmin/<arm>_s<s>.parquet` holds 8,000 rows (uid, kategori, p). Twin outputs go to
  `twin/tahmin/` and `twin_fiyat/tahmin/`. Real runs are sealed (`simengine.research.seal`, registry `.prereg`)
  before they are scored.
- **Calibrated and raw.** Prompt arms (prm, caa) also write `<arm>-raw`, the uncalibrated P(yes).
- **Smoke mode.** `--kucuk [FIT:VAL:TEST]` writes under `kucuk/`, keeps two configs per grid and seals nothing.
  `--score g1 --kucuk ...` scores the smoke files with the unchanged evaluator: it builds `kucuk/talp73/panel/`
  (the panel with the smoke test persons as its sample) and cuts the `--taban` files to those persons.
- **Raw files.** `--score` also takes `<arm>-raw` (for example `prm-q38-L2-raw`, `caa-q38-swap-raw`).
- **Work folder.** `--base`, default `$LATENT_BASE` or the runner's built-in default.

## Competition day: one command (tek komutla)

The prereg priority list for the card class, top to bottom. No new run starts after `--until`. A failed run is
logged and the plan moves on to the next one.

```bash
nice -n 19 /data/talp/venv/bin/python runs/latent/kos.py --plan 80gb --until 07:30     # or 48gb / 24gb
```

Before that, check the card and the plan:

1. `--calibrate --model q38 --precision bf16`, or `"fp8|nf4"` on a 48 GB card. This measures tok/s and checks
   backward.
2. `--plan 80gb --kucuk 20:8:8`. This runs the whole plan on 36 persons, with nothing sealed. It still loads
   every model of the plan (each 27-32B model is a 55-65 GB download in bf16), so it is not a quick check. The
   quick check of the code path is the tiny models, no download:
   `--model tiny-qwen3_5 --precision fp32 --device cpu --arm prb-L2 rsd-angc sft-angc caa --kucuk 20:8:8`
   (about 30 seconds on CPU). `tiny-gemma4` checks the Gemma path.

Plans:

- **80gb:** the LLM-free arms; then q35s K1-K5; then q38 K1+K2, K4, G3, K5; then q3; then q38 caa; then g31, q35,
  g26, q38 G2 and q25.
- **48gb:** q38 in fp8, or nf4 where fp8 is not supported. q3, g31 and q35 run in nf4 on L2 prompts only. sft runs
  last.
- **24gb:** every 27-32B model in nf4. The q38 G1 arms are capped at 1,000 fit persons and 400 validation persons.

**Time and memory (measured on the CPU smoke, scaled; `/data/talp/is/latent/DUMAN.md`).** At full size one rsd
run (one person vector, 12-config sweep plus 3 epochs over 69,400 fit cells) is about 4 hours on an H100 at the
prereg's assumed 2,500 tok/s, and sft-angc about 2 hours; K1+K2 is about 1.2 hours. The prereg budget assumed
far less for K4, so the 80gb plan will not finish in one night. `--cap-fit 1000 --cap-val 400` cuts rsd to about
1.3 hours. On 48 GB and 24 GB cards, lower `--cell-batch` (16 or 8) and `--train-batch` (16): one L2 cell's cache
copy on Qwen3.8 is roughly 0.2 GB, so 64 cells at once is about 11-15 GB.

Seeds 1-2 of the G1 LLM arms are not part of a plan. Under the prereg stop rule, run them by hand, and only when
the seed-0 score beats the arm's matched encoder-only control or T3:
`--model q38 --arm prb-L2 --seed 1 2`.

## Not done yet

- SAE readout (`sae-q35-L2`) and the QLoRA arm (`lora-*`) have no runner arm. The SAE file format of Qwen-Scope
  has not been checked against `load_sae`.
- The exploratory X block (field patching, persona arithmetic, layer tracing) is not implemented. X5 (bf16 vs
  FP8) is two plain runs with different `--base` folders.
- caa steers the last prompt token only. Steering every prompt position would need the injection inside the
  cached prefix.
- The fp8 and nf4 paths have not been run, because this machine has no GPU and no bitsandbytes.
- The Gemma 4 sliding window is tested only on the tiny model, whose window is 4 tokens (left-padded batches,
  cached vs uncached). Real L2 prompts are 945 tokens on average and up to 1,353 (p95 1,270), so on g31 they
  cross a 1,024-token window; this has not run on real Gemma weights.

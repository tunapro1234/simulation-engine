# Recipe: message spread on a social network / grid

Script: [`examples/recipes/spread.py`](../../examples/recipes/spread.py) (about 210 lines, numbered steps 0-8).
Test: `tests/test_recipe_spread.py`.

```bash
python3 examples/recipes/spread.py                                     # offline: FakeLLM + synthetic 15x15 grid, ~30 s
python3 examples/recipes/spread.py --data data/cascades.csv --message "the post you want to seed" --k 10
python3 examples/recipes/spread.py --live --max-usd 1 --data data/cascades.csv   # Claude Haiku, key from $ANTHROPIC_API_KEY
python3 examples/recipes/spread.py --live --key-file <path> --max-usd 1          # or the key from a file
```

Outputs go to `--out` (default `runs/spread`, git-ignored): `card.md`, `dashboard.html`, the seal registry
`.onkayit/` and, with `--live`, the reply cache and cost ledger under `.cache/`.

## When to use

- The brief asks how far a message (a campaign post, a public notice, a rumour, a warning) travels through a network
  of people, who passes it on, or **whom to give it to first** (seeding / influencer choice), and you have some past
  spreads on the same network: who saw which message and who shared it. Twenty past cascades are enough to fit the
  baselines and grade the model; without them nothing can be graded and the seeding table is a model opinion only.
- The network can be a follower graph, a friendship list or a grid of neighbours (households in a district, desks
  in a building). The script only needs "who passes messages to whom".
- Not for: forecasting the 24 h share curve of a single post from its first hour (use `abm.shape_forecast`; in this
  repo's Twitter15/16 run the shape baseline 0.391 tied Hawkes 0.391 and the SEISMIC-style model lost to it,
  `docs/RESULTS.md`), or product adoption over months (`examples/recipes/adoption.py`).

## Inputs

`--data` is a CSV with **one row per (cascade, node)**: every node of the network appears in every cascade, also
the ones that never saw or never shared the message (they are the negatives).

| column | type | meaning |
|---|---|---|
| `cascade_id` | str | one spread of one message; the bootstrap cluster; a cascade is never split across train and test |
| `message` | str | text of the message spread in that cascade |
| `node` | str | person / account id |
| `segment` | str | node segment (student, parent, local media, ...) |
| `persona` | str | segment description given to the LLM; the first value per segment is used; **no outcome information** |
| `neighbours` | str | space-separated node ids this node passes messages to (its followers); the same list in every cascade; built from the network **before** the cascades, never from who reshared from whom |
| `seed` | 0/1 | 1 = got the message from the organiser at the start; given to every model, never predicted |
| `shared` | 0/1 | 1 = shared / reposted it in this cascade |
| `step` | int | hop at which the node shared (seeds 0, their direct followers 1, ...); empty when `shared` = 0. Timestamps work too if you bin them into rounds |

Without `--data` the script builds a 15x15 grid (4 neighbours each) with four segments; local-media nodes also reach
8 random people (the hubs). 60 cascades, 3 random seeds each, spread by an independent cascade whose share
probability depends on whether the message is about the receiver's topic (students: rent, parents: school,
retirees: pension and health; everybody shares flood warnings). The fake LLM knows most of this, misses that
retirees share health news, and says "yes" far too often, so the card shows an LLM that is useful only after
calibration, and whose persona signal turns out not to add anything measurable.

## Steps (what each one does, what to change on the day, what backs it)

| step | in the script | change on the day | backed by |
|---|---|---|---|
| 0 | `SEGMENTS`, `TOPICS`, `CAMPAIGN`, `SYSTEM` | segments and their one-line descriptions from the client's audience data; `--message` is the post to seed; keep the JSON reply format | Meister et al. (arXiv:2411.05403): asking per segment beats per-person role play (ranked #4, closed 0.82 of the gap in G1). Structured persona schemas (Ye et al., #8) are **not** verified, so keep personas short |
| 1 | load, graph from `neighbours`, split | if the brief fixes test cascades (the last month), use them; never split rows of one cascade | cascade-disjoint split = the leakage guard behind every number in `docs/RESULTS.md` |
| 2 | exposures `k`: how many followed nodes had shared before this node decided | nothing | independent cascade model (Kempe, Kleinberg & Tardos, KDD 2003); fitting its edge probabilities from observed cascades (Saito et al., KES 2008). Each exposure is one Bernoulli try, so P(share) = 1 − (1 − p)^k |
| 3 | backend | `--live --max-usd`; Haiku by default | ranked "do not spend time on": a bigger model (Park ~1 point; here Sonnet − Haiku +1.0, not significant). On retweet curves a Sonnet subsample did not beat Haiku (0.801 vs 0.785 RMSE, `docs/RESULTS.md`) |
| 4 | one prompt per (segment, message) + one no-persona prompt per message | nothing; 61 messages x (4 segments + control) = 305 calls offline, under 2 cents on Haiku. A failed or refused call becomes 0.5; if more than a quarter fail the run stops with the first error | ranked #4 (segment-level questions); the no-persona prompt feeds the persona control of step 5 |
| 5 | fit on train exposures: `ic_global` (one p), `ic_segment` (p per segment), ours `llm_calibrated` = logit(segment p) + c + b x (LLM logit − mean), grid-searched by likelihood; `llm_nopersona_calibrated` = the same fit on the no-persona score (one shift per message, same for every segment) | nothing; `ours` is named before the truth. The calibrated no-persona arm is the persona control: ours must beat it, not the raw no-persona arm, whose yes-bias makes it a different kind of prediction | ranked #2: calibrate the LLM's "yes" bias (Schoenegger; `sim.batch_calibrate` 69.4 % → 76.5 %). Ranked #3: LLM as a feature of a history model, shrunk so it cannot do worse than the baseline on train (c = b = 0 *is* `ic_segment`); honest gain is small (+0.006 in Twin-2K-500) |
| 6 | Monte Carlo IC from each cascade's seeds, 500 runs → P(node shares) per method; every method gets the same random seed per cascade (paired); `se.seal` of the test rows with `ours: llm_calibrated` in the description | nothing; seal before looking at test outcomes. Do not lower the 500: with 200 runs from one shared stream, Monte Carlo noise alone moved `ic_global` / `ic_segment` by 0.03-0.06 AUC and swapped them | pre-registration; the selection-bias audit counts `ours` as named in advance only from a pre-registration text written before the seal (e.g. `prereg.md` in the registry, as `ad_ab` does); a name in the seal description alone reads "not verified (seal ≠ pre-registration)" |
| 7 | cascade size per method (predicted mean, absolute error; `train_mean` = the mean train cascade size, a constant), then `se.evidence.evaluate` → ladder, shuffle control, cascade-bootstrap CIs, audits, `card.md`, `dashboard.html` | `--B 1000` for the final card; pass `memorization=` when you have run the audit | ranked #1: baseline ladder and gap closed (Park et al. arXiv:2411.10109 normalisation) |
| 8 | seeding: `random`, `top_degree`, `fans_by_degree` (top-degree nodes of the segment the LLM rates highest for the message), `greedy_model` (greedy on the fitted model over the 30 biggest hubs) for `--k` seeds; reach with a Monte Carlo 95 % CI over 1,000 runs (simulation noise only, not the model's error), minus random on the same draws; synthetic data also shows the true world | add the client's own strategy as another entry of `plan` (their current ambassador list) | greedy on a submodular spread function is within 1 − 1/e of the best seed set (Kempe et al. 2003), degree is the standard heuristic to beat. Ranked #6: model effect sizes are inflated (Hewitt et al., r = .85 but exaggerated effects; Manning 13.2x), so the slide quotes the ranking and the CI, not the model's absolute reach |

Not in the script on purpose: an LLM agent society that role-plays every node. It is in the "do not spend time on"
list of the literature ranking (AgentTorch: one LLM per agent was behind a simple heuristic, MSE 56.98 vs 41.05; our
retweet society, 20 Haiku personas x 3 turns, 0.771 RMSE vs a single Haiku score 0.772 and a calibrated constant
curve 0.772, `docs/RESULTS.md`). If the jury wants to see agents talk, run it as a demo next to the card, with no
accuracy claim:

```python
people = [se.abm.SocietyAgent(f"P{i:02d}", text) for i, text in enumerate(persona_texts, 1)]
world = se.abm.ShareCascadeWorld(content=message, n_agents=len(people))
run = se.abm.AgentSociety(people, world, llm, mode="batch", max_calls=10).run()   # world.curve = sharers per turn
```

## What the jury should see

Numbers from the offline run (synthetic grid, FakeLLM, `--B 200`).

1. **Why a raw LLM cannot be used for spread.** Test cascades had 10.8 non-seed sharers on average. The raw LLM
   (p ≈ 0.5-0.8 per exposure) predicts 211 of 225 people, the raw no-persona arm 195; the calibrated arms predict
   8.7-9.9, the IC baselines 7.7-8.4. On a network a small "yes" bias percolates into a cascade that reaches
   everybody. The fitted models' absolute errors (9.5-10.6) are barely below the constant "mean train cascade
   size" (8.9, error 11.1): a single cascade is one random draw, so judge sizes on many cascades, never one.
2. **The baseline ladder** (AUC of "who shares", test cascades, 95 % cascade-bootstrap CIs): random 0.500, segment
   mean 0.522, `ic_segment` 0.673, `ic_global` 0.681, raw `llm` 0.704, raw `llm_nopersona` 0.757, `llm_calibrated`
   0.791 [0.727, 0.881], `llm_nopersona_calibrated` 0.800 [0.742, 0.892]. Ours − strongest LLM-free baseline
   (`ic_global`; the card flags that the two IC baselines are within noise of each other) +0.109 [−0.028, +0.209]:
   the badge is "inconclusive" and the slide says so.
3. **The controls.** The calibrated no-persona arm scores 0.800, ours 0.791: ours − no-persona −0.010
   [−0.040, +0.018]. The persona adds nothing measurable here; what the LLM contributes is a **message-level** read
   (a flood warning spreads, a pension form does not), and the rest of the ranking comes from **where a node sits
   relative to the seeds**. Say that plainly: the network is the model, the LLM rates the message. The shuffle
   control (ours permuted across all test rows) is 0.505; it removes the graph position as well as the segment, so
   it only shows that the predictions are not noise, not that the persona matters.
4. **The seeding table** (k = 5, rent-support message): reach under the model / true world: random 15.4 / 11.8,
   top degree 25.6 / 18.5, fans by degree 11.3 / 8.8, greedy 30.2 / 22.3. The model's ranking matches the true
   world, its absolute reach is too high by about a third; "seed the hubs, not only the fans" is the finding (the
   students who care most have 4 neighbours each, the hubs reach 8 more). The ± in that table is Monte Carlo error
   of the fitted model, not uncertainty about the model; with your own data there is no true world: present the
   ranking and check it on the held-out cascades whose seeds happened to be hubs.
5. **The seal**: the hash and time of the sealed test predictions in `card.md`, earlier than the evaluation.
6. **Spend**: with `--live`, the last line gives the ledger total; the run stops at `--max-usd`.

## Pitfalls

- **Leakage.** The graph must be the network *before* the cascades (follower lists, a grid), never the reshare
  edges observed in the cascades: a "who reshared from whom" graph contains the answer. The persona describes the
  segment, not its outcome ("shares everything from this page" is the label). The message text must not carry its
  own reach ("viral", "10k shares"). If one message was posted several times, keep all its cascades on one side of
  the split (hash `message` instead of `cascade_id` in step 1). All parameters are fit on train exposures only.
- **Imbalanced accuracy.** Most nodes never share (95 % in the synthetic world), so "nobody shares" scores 95 %
  accuracy and says nothing. The recipe uses AUC as the primary metric and prints cascade size next to it; report
  both, never plain accuracy. A high AUC can come from graph distance and message alone: always show the calibrated
  no-persona arm, and claim a persona effect only when ours beats it.
- **Memorisation.** Famous cascades (public rumour datasets such as Twitter15/16, well-known viral posts) may be in
  the model's training data, and the LLM then "predicts" a spread it has read about. The card flags "not proven:
  leakage/memorisation" until you run `se.evidence.memorization_audit` or `recall_probe` and pass the result as
  `evaluate(..., memorization=...)`. Prefer the client's own, recent messages (Hewitt et al. report post-cutoff
  studies separately: r = .90 there).
- **Few cascades.** The bootstrap resamples cascades, so 6 test cascades give a CI as wide as the ladder (the card
  adds a cluster-count warning). Get more cascades rather than more nodes.
- **Model-only seeding claims.** The seeding table is a simulation of the fitted model. The model over-predicts reach
  here (30.2 vs 22.3 for greedy) because one shift per segment cannot capture every topic match; quote the order of
  the strategies and the difference to random, not "30 people".
- **Budget.** `--live` refuses to start without `--max-usd` and a key; once the ledger total reaches the cap the
  backend raises `BudgetExceeded` before any new call (calls already in flight finish), finished replies stay in the
  cache, and the same command with a higher cap resumes. The key comes from `$ANTHROPIC_API_KEY` or `--key-file` and
  is never printed.

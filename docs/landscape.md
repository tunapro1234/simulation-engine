# Landscape: synthetic-respondent vendors, open simulators, benchmarks

This page places simengine among the companies, open-source simulators and benchmarks that work on LLM
simulation of people. The source is the `awesome-llm-social-simulation` Obsidian vault (504 notes, "AW" below).
The vault README says AI agents assembled it and no human checked it end to end. It declares CC BY 4.0, so
credit it as the source. Validation methods are on [validation.md](validation.md). Failure modes and the audits
that catch them are on [failure-modes.md](failure-modes.md).

## How to read the tags

Every number on this page carries two tags.

**Kind**
- **paper**: from a research paper.
- **vendor self-reported**: from a company's own site or marketing, or from press coverage that repeats it.
  No peer review, no confidence interval and no published baseline.

**Check**
- **verified**: checked by hand on 22 Oct 2026 against the primary-source abstract on `arxiv.org/abs/<id>`. The
  arXiv version is named because some abstracts changed between versions.
- **unverified (vault)**: comes only from the vault note. It was not found in a primary abstract, or the primary
  source is not on arXiv (journal pages and vendor sites were not fetched). Do not put it on a slide until
  someone reads the source.

A number tagged **unverified (vault)** may still be correct. The tag only means that nobody has checked it yet.

## 1. Synthetic-respondent vendors: what they claim and what backs it

Sources: `AW/Companies/*.md` (37 notes). In the vault, a vendor's "credibility" verdict covers whether the company
exists and has funding, customers and a live product. **It does not cover whether the accuracy claim is true.**
Each claim below comes from the vendor. None comes with a CI, a baseline, a stated metric or a peer-reviewed
validation paper. Funding and customer details are left out because the vault itself calls them stale.

| vendor (AW note) | what it sells | headline claim | tags | independent evidence in AW |
|---|---|---|---|---|
| Aaru (`Aaru.md`) | multi-agent population simulation for polling and enterprise research | predicted a 2024 New York primary "within ~371 votes" | vendor self-reported · unverified (vault) | none; the vault says the claim "traces to Aaru's own self-reporting" |
| Simile (`Simile.md`) | interview-grounded "digital twins" of consenting people | "95% accuracy" | vendor self-reported (secondary promotional blogs) · unverified (vault) | academic lineage only: the Park et al. generative-agents papers (row B3 below), which report a test-retest-normalized accuracy, not "95%" |
| Artificial Societies (`Artificial Societies (societies.io).md`) | networked persona audiences (300 to 5,000+ agents) | "95% accuracy on opinion distribution" (homepage); about 80% vs about 62-63% for plain LLMs on social-media responses (press) | vendor self-reported · unverified (vault) | none; the note also contains a broken tool-call fragment |
| Brox.AI (`Brox.AI.md`) | twins of recruited, paid, interviewed people | "85%+ correlation with real-world responses" | vendor self-reported · unverified (vault) | none ("no academic papers found") |
| Evidenza (`Evidenza.md`) | LLM-built B2B buyer personas | "88% average similarity", "within 95% of human results" | vendor self-reported · unverified (vault) | none |
| Subconscious.ai (`Subconscious.ai.md`) | LLM respondents in conjoint / discrete-choice experiments | "93% human accuracy", "validated against 350+ (or 1,000+) human studies" | vendor self-reported · unverified (vault) | the vault could not find the validation paper the site cites |
| Synthetic Users (`Synthetic Users.md`) | AI interview and survey participants with Big Five profiles | "85-92% parity with real human research" | vendor self-reported · unverified (vault) | an independent UX-research critique in the vault is negative (sycophancy, implausibly positive answers) |
| Minds (`Minds (getminds.ai).md`) | self-serve persona panels | "validated at 80-95% accuracy against real human data" | vendor self-reported · unverified (vault) | none; no methodology published |
| atypica.AI (`atypica.AI.md`) | synthetic personas plus interview-derived personas | "85% human-like behavioral consistency" against a self-set human two-week test-retest baseline of about 81% | vendor self-reported · unverified (vault) | none; the test-retest framing borrows from Park et al. (B3) without a paper |
| Blok (`Blok.md`) | persona agents run through product flows, built from analytics logs | "up to 87% behavioral fidelity" | vendor self-reported · unverified (vault) | none |
| Qualtrics Edge Audiences (`Qualtrics Edge Audiences.md`) | synthetic panel on a fine-tuned proprietary LLM, inside the Qualtrics platform | "12x better accuracy than general-purpose LLMs"; about 50% lower research cost | vendor self-reported · unverified (vault) | none; Qualtrics itself recommends human validation for high-stakes decisions |
| Vurvey (`Vurvey (Vurvey Labs).md`) | "Large Persona Model" trained on video surveys | trained on "3+ million real people" (a scale claim, not an accuracy claim) | vendor self-reported · unverified (vault) | none |
| Lakmoos AI (`Lakmoos AI.md`) | "neuro-symbolic" synthetic panels | "validated across 15+ large-scale projects" (no accuracy figure published) | vendor self-reported · unverified (vault) | evidence sits behind a gated one-pager |
| Ask Rally (`Ask Rally (Irreality Labs).md`) | AI audience simulator with cloned personas | personas calibrated "until a superhuman LLM judge can't tell them apart" (no figure) | vendor self-reported · unverified (vault) | none; the vendor's own content says personas "don't replace real humans" |
| MiroFish (`Mirofish (MiroFish).md`) | open-source (AGPL-3.0) swarm-prediction engine on top of OASIS | no accuracy figure | — | the vault notes "no published accuracy/validation benchmarks" |
| Expected Parrot / EDSL (`Expected Parrot (EDSL).md`) | open-source (MIT) survey framework for LLM agents, plus a hosted service | no accuracy claim (it is a tool) | — | used in published research; edsl-test ran on it. After the port it is not a simengine dependency |

**Market signals** (all from the vault, unverified):
- Roundtable (`Roundtable.md`) began in 2023 as an LLM survey simulator. It pivoted to detecting bots in surveys.
- Viable (`Viable.md`) analysed real feedback, not simulated people, and is now offline.
- Conveo, Genway, Outset, Miravoice and Fairing put AI in the interviewer or analyst role. Real humans still give
  the answers, so these are not respondent simulators.
- Fairgen (`Fairgen.md`) boosts small real samples statistically from microdata. That is closer to data
  augmentation (row B9) than to role-play.
- Toluna HarmonAIze (`Toluna HarmonAIze Personas.md`) and Qualtrics mark incumbent survey firms entering the
  market.

The vault has notes on Auxos, Deepsona, Delve AI, Ditto, Uxia, Delphi, Tavus, Sakana AI and Altera. None of them
publishes an accuracy claim, or they are out of scope here (expert clones, video avatars, automated science,
Minecraft societies).

**What simengine adds.** Every vendor row above publishes one number. None of them publishes the evidence that
should come with it:
- the baselines it beats: the `evaluate` baseline ladder and `unlabeled_baselines`
- a confidence interval: paired cluster bootstrap
- a human ceiling: the `retest` column, giving the gap closed and the ratio to ceiling
- a no-persona control: `sim` control arms `no_persona` and `persona_swap`, and the `shuffle_control` audit
- spread: `simengine.evidence.variance_ratio` and the spread audit
- completion: `simengine.evidence.completion_row`
- a pre-registered target: `simengine.research.Prereg` and `simengine.research.declare_targets`

The evidence card prints all of these together ([modules/evidence.md](modules/evidence.md)). The README and
docs/RESULTS.md report one consistent lesson from our own runs: simple baselines usually match or beat
persona panels. Report the numbers that way.

## 2. Open simulators

Sources: `AW/Open Repos/*`, `AW/Tools/*`. The vault's star counts and maturity notes are left out. They change
quickly and nobody re-checked them.

| project | pattern | primary source | tags | how simengine relates |
|---|---|---|---|---|
| Concordia (Google DeepMind) | **Game Master**: a special agent simulates the environment. Players state intended actions in natural language, and the GM decides what actually happens: physical plausibility, or API calls in digital settings. | Vezhnevets et al., arXiv:2312.03664 | paper · verified (abstract describes the GM) | `simengine.abm.AgentSociety` with a `World` object plays the GM role in a fixed form. The world's code, not an LLM, applies the rules and payoffs. That keeps the outcomes auditable. `simengine.abm.run_who_to_punish` runs the multi-period who-to-punish society (Ertan, Page & Putterman 2009) this way. |
| OASIS (CAMEL-AI) | X- and Reddit-like platforms with recommender systems and follow graphs; "up to one million users" | Yang et al., arXiv:2411.11581 (v5) | paper · verified | Different scale. simengine's `abm.ShareCascadeWorld` and the Hawkes/SEISMIC curves (`simengine.abm.hawkes_mean_field`, `simengine.abm.SeismicPredictor`) cover cascades. Our measurement: an LLM agent society did not beat a shape baseline on retweet curves (README, `abm` docs). |
| MiroFish | seed document → knowledge graph → thousands of persona agents on OASIS → report agent | GitHub repo only (no paper) | vendor self-reported · unverified (vault) | Same caution as OASIS: herding and cost, with no validation published. |
| TinyTroupe (Microsoft) | Python toolkit: detailed personas in individual or group scenarios (focus groups, brainstorms) | Salem et al., arXiv:2507.09788 (v3) | paper · verified (the abstract gives no evaluation figures) | `simengine.sim.roundtable` (discussion rounds) and `simengine.sim.survey`. Our measurement: 5 discussion rounds gave ΔW1 −0.005 [−0.012, +0.002] (docs/RESULTS.md). |
| genagents (Stanford) | person-specific agents built from 2-hour interviews and/or GSS + Big Five answers | Park et al., arXiv:2411.10109 (see B3) | paper · verified | `simengine.persona.from_table` and `simengine.persona.from_history` build personas from structured rows. The vault says the full 1,000-person agents sit behind a restricted research API. |
| AgentTorch / Large Population Models (MIT Media Lab) | differentiable, GPU ABM with millions of agents; the LLM is one composable module; gradient calibration | repo + arXiv:2507.09901 | unverified (vault) (abstract not fetched) | simengine calibrates with rejection ABC (`simengine.abm.AdoptionABC`) and moment matching (`simengine.methods.fit_mixture_weights`). It has no differentiable engine. |
| mesa-llm | official Mesa extension: LLM reasoning, memory and planning modules in Mesa agents; pre-1.0, API not stable | GitHub repo, PyPI `Mesa-LLM` | unverified (vault) | A reference for module structure. simengine's `abm` does not depend on Mesa. |
| EDSL (Expected Parrot) | survey and experiment DSL over many LLM providers, with remote caching | GitHub repo | — | edsl-test ran on it. simengine replaces it with `simengine.types.LLM` (`complete`/`map`) and `simengine.llm.DiskCache`. |
| Classic ABM tools: NetLogo, Mesa, MASON, Repast, GAMA, Agents.jl, AnyLogic | rule-based ABM platforms (`AW/Tools/*`) | — | — | Background only. Their validation tradition (ODD, POM, docking, SMD, Sobol) is what [validation.md](validation.md) borrows. |

**Roadmap idea from replicant, not built (`RP/src/replicant/otree/hybrid.py`, `RP/examples/06_hybrid_humans_bots.py`): HybridSession.** A few live humans and synthetic personas
take the same instrument at the same time (replicant's oTree client). The personas are then calibrated against
the humans with PPI++ (`simengine.evidence.worth_humans`). It is a pitch idea, not hackathon-day code. If someone
builds it, it must not inherit replicant's test-server defaults: the hard-coded secret key, the default admin
password and the default REST key.

## 3. Benchmarks and datasets

| id | benchmark | what it measures | key numbers | tags | in simengine |
|---|---|---|---|---|---|
| B1 | **SimBench** (Hu et al., arXiv:2510.17516, v4) | 20 datasets in one multiple-choice format; how well LLMs match human response distributions | best LLMs "40.80/100"; ability scales log-linearly with model size, not with inference-time compute; instruction tuning helps consensus questions and hurts diverse ones | paper · verified | **present:** `simengine.bench.tasks.simbench` (task `simbench`); the metric "SimBench score 100·(1 − TVD/TVD_uniform)" is in `evaluate` |
| B2 | **Twin-2K-500** (Toubia et al., arXiv:2505.17479, v1) | 2,058 US participants, 4 waves, 500 questions; the last wave repeats earlier tasks for a test-retest baseline | N = 2,058; 2.42 h per person; 4 waves; 500 questions | paper · verified | **present:** `simengine.bench.tasks.twin` (task `twin2k500`), with the `retest` ceiling |
| B2' | Twin-2K-500 "up to ~87% out-of-sample individual accuracy" | — | ~87% | paper · **unverified (vault)**: not in the abstract | do not cite until it is checked in the paper body |
| B3 | **Generative agents of 1,052 people** (Park et al., arXiv:2411.10109) | GSS accuracy normalized by participants' own 2-week test-retest | v3 abstract (Jun 2026): 83% interview-only, 82% survey-only, 86% combined, vs 74% demographics-only. v1 abstract (Nov 2024): "85% as accurately as participants replicate their own answers two weeks later" | paper · verified (both versions) | normalization by test-retest = `retest` column → ceiling, gap closed; `simengine.sim.retest_stats` for repeated LLM samples |
| B4 | **OpinionQA** (Santurkar et al., arXiv:2303.17548) | alignment of LM opinions with 60 US demographic groups (Pew ATP) | 60 groups; 65+ and widowed among the groups reflected worst | paper · verified | **present:** `simengine.bench.tasks.opinionqa` (task `opinionqa_groups`) |
| B5 | **GlobalOpinionQA** (Durmus et al., arXiv:2306.16388) | cross-national opinion similarity (Pew Global Attitudes + World Values Survey); country prompting can produce stereotypes | the abstract names no countries | paper · verified | not present. It is the only cross-national benchmark in the vault. The abstract does not show whether Turkey is covered: check the dataset before claiming it |
| B6 | **SubPOP** (Suh et al., arXiv:2502.16761) | fine-tuning on survey data to predict subpopulation opinion distributions | 3,362 questions; 70K subpopulation-response pairs; LLM-human gap reduced "by up to 46%" | paper · verified | not present (simengine does no fine-tuning). Its claim that it generalizes to OpinionQA can be checked with B4 |
| B7 | **Psych-101 / Centaur** (Binz et al., arXiv:2410.20268; Nature 2025) | trial-by-trial cognition experiments; a Llama fine-tuned to predict choices | >60,000 participants; >10,000,000 choices; 160 experiments | paper · verified | not present; single-agent cognition tasks, not multiplayer games |
| B8 | **Behavioral Turing test** (Mei et al., PNAS 2024; arXiv:2312.00798 v2) | dictator, ultimatum, trust, bomb risk, public goods, repeated PD + Big Five, vs a human database | humans from "more than 50 countries"; GPT-4 "statistically indistinguishable from a random human" but toward the "more altruistic and cooperative end" | paper · verified (arXiv version) | **present:** `simengine.games` carries the edsl-test game library with page-cited human references; `simengine.bench.tasks.econ_games_b01` scores it |
| B9 | **Conjoint data augmentation** (Wang, Zhang & Zhang, arXiv:2412.19363 v3) | combining LLM and human conjoint answers, with guarantees | saves data and cost "by 24.9% to 79.8%"; naive substitution saves nothing | paper · verified | `simengine.methods.MNL` (conjoint) plus `simengine.evidence.worth_humans` (PPI++) apply the same "augment, don't replace" logic |
| B10 | **SSR** (Maier et al., arXiv:2510.08338 v3) | free-text answers mapped to Likert distributions by embedding similarity | 57 surveys, 9,300 human responses, "90% of human test-retest reliability", KS similarity > 0.85 | paper · verified | **present:** `simengine.sim.SSR` |
| B11 | **General Social Agents** (Manning & Horton, arXiv:2508.17407 v5) | theory-grounded agents built on seed games, tested on novel games | 883,320 novel games; 1,500 sampled for testing; beats cognitive hierarchy, equilibria and out-of-the-box agents | paper · verified | **present:** `simengine.methods.fit_mixture_weights` (theory-persona mixture), with `val=` holdout |
| B11' | General Social Agents "cuts error 53-73%" | — | 53-73% | paper · **unverified (vault)**: not in the abstract | do not cite |

The vault calls SimBench "the closest existing analog" to a human-likeness benchmark. simengine's bench layer
(`simengine.bench.get`, `simengine.bench.leaderboard`) runs B1, B2 and B4 offline on local data under
`data_root`. B8 is covered by `simengine.games` and the `econ_games_b01` task.

**Turkey gap.** All the benchmarks with Turkish coverage are unconfirmed. Every instrument, norm table and persona
row ported here is English and US-based. A Turkish persona claim needs a Turkish rerun of the Big Five check
(`simengine.persona.validate_bigfive`) and of the games.

## 4. PPI citations (not from the vault)

The vault's prediction-powered-inference note is a single line. Cite these instead:
- Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, "Prediction-Powered Inference", arXiv:2301.09633 (v4). Verified
  (title, authors).
- Angelopoulos, Duchi & Zrnic, "PPI++: Efficient Prediction-Powered Inference", arXiv:2311.01453 (v2). Verified
  (title, authors). This is the estimator that `simengine.evidence.worth_humans` implements.
- Huang, Wu & Wang, "How Many Human Survey Respondents is a Large Language Model Worth?", arXiv:2502.17773 (v6).
  Verified. It is a related effective-sample-size framing, and its abstract gives no numbers.
- "The Mixed Subjects Design", Sociological Methods & Research 2025, doi:10.1177/00491241251326865. Unverified
  (vault). It is not on arXiv, and the vault does not record its authors. It defines a PPI correlation and an
  effective sample size.

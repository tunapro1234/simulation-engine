"""Generated-persona injectors: LLM-written backstories (Anthology, Moon et al. 2024, arXiv:2407.06576), Persona
Hub descriptions (Ge et al. 2024, arXiv:2406.20094) and LLM narratives from structured fields (Li et al. 2025,
arXiv:2503.16527).

Generation is an offline step (one LLM pass, disk-cached); `render` only reads what it produced, so it stays
pure. Generated texts live in person["generated"] = {"backstory": str, "persona": str, "narrative": str}.

    import simengine.persona.inject as pi
    from simengine.persona.inject import generated as g
    stories = g.generate_backstories(llm, 200)                       # Anthology: "Tell me about yourself..."
    stories = g.estimate_demographics(llm, stories, demo_questions)   # P(trait | story), 40 samples each
    people = g.attach(people, g.match_backstories(people, stories, ["age", "gender"], method="hungarian"),
                      "backstory")
    p = pi.get_injector("anthology_backstory")(people[0], question)
    g.register_pool("w34", stories)
    p = pi.get_injector("anthology_greedy", pool="w34", keys=["age", "gender"])(person, question)
    people = g.attach(people, g.text_to_persona(llm, people), "persona")      # Persona Hub text-to-persona
    people = g.attach(people, g.narrate_people(llm, people), "narrative")      # descriptive persona from fields

Paper facts the code follows (checked against the papers and the released CannyLab/anthology code):
- Anthology backstory prompt "Question: <q>\\n\\nAnswer:" on a base model, q = "Tell me about yourself. How old
  are you? Where did you grow up? What events made you who you are? What matters to you and what do you enjoy?"
  (the released default also puts "Are you a US citizen?" first: preamble=US_PREAMBLE); stop sequences
  "Question", "Q:", "Interview"; stories of <= 40 tokens dropped. The survey question follows the backstory,
  preceded by "Please answer the following question keeping in mind your previous answers." Demographics of each
  backstory: the demographic survey asked 40 times per question.
  Edge weight human-story = prod_k P_story(human's trait k); greedy = argmax per human (stories reused),
  maximum weight matching = Hungarian (one-to-one).
- Persona Hub text-to-persona "Who is likely to [read|write|like|dislike|...] the text?", persona-to-persona
  "Who is in close relationship with the given persona?"; the paper says its printed prompts are simplified,
  the wording here is ours around those questions. Dedup: MinHash on 1-grams at 0.9 (here exact Jaccard).
- Li et al. 2025: descriptive persona = free narrative from demographic metadata, prompts verbatim from Appendix
  B.1 (system) + B.4 (user); answering uses the B.5 opinion-simulation prompt (style="paper").
LEAKAGE: generators read only `fields` and `text`, never `answers`. At render time a field named like the target
question (or an `exclude` id) is never shown or matched on; generation happens before any question, so pass
narrate_people(exclude=...) for fields that are targets.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

from ._base import Injector, MissingPersonField, Prompt, parse_option, register

ANTHOLOGY = "Moon et al. 2024, Virtual Personas for Language Models via an Anthology of Backstories (arXiv:2407.06576)"
PERSONA_HUB = "Ge et al. 2024, Scaling Synthetic Data Creation with 1,000,000,000 Personas (arXiv:2406.20094)"
LI2025 = "Li et al. 2025, LLM Generated Persona is a Promise with a Catch (arXiv:2503.16527)"

# CannyLab/anthology configs/generate_backstory/self_generate/self_generate.yaml
BACKSTORY_QUESTION = ("Tell me about yourself. How old are you? Where did you grow up? What events made you who you "
                      "are? What matters to you and what do you enjoy?")
BACKSTORY_QUESTION_SHORT = "Tell me about yourself."          # paper text / baseline config
US_PREAMBLE = "Are you a US citizen?"                          # same config's preamble_question (US panels only)
BACKSTORY_STOPS = ("Question", "Q:", "Interview")              # scripts/run_backstory_gen.py stop=[...]
CONSISTENCY = "Please answer the following question keeping in mind your previous answers."
DEMOGRAPHIC_SAMPLES = 40                                       # paper: each demographic question asked 40 times

TEXT_TO_PERSONA = ("Who is likely to {relation} the following text? Describe this person as specifically as "
                   "possible in one short paragraph that starts with \"A\" or \"An\".\n\nText:\n{text}")
PERSONA_TO_PERSONA = ("Who is in close relationship with the given persona? Name the relationship and describe that "
                      "person as specifically as possible in one short paragraph that starts with \"A\" or "
                      "\"An\".\n\nPersona:\n{persona}")
RELATIONS = ("read", "write", "like", "dislike")

# Li et al. 2025, Appendix B.1 (system) and B.4 (descriptive persona), verbatim; {METADATA} = the person's fields
NARRATIVE_SYSTEM = ("You are an AI assistant specialized in detailed and unbiased persona generation for opinion "
                    "simulations. Your task is to generate a specific, realistic, and diverse persona based on the "
                    "provided demographic information and fill in a comprehensive JSON template.")
NARRATIVE_USER = """### INSTRUCTIONS ###
1. You will be provided with a persona meta file that has the core demographic information of a person.
2. Your task is to create a detailed, diverse, and vivid persona that is fully consistent with ALL features in the given metadata.
3. Elaborate on all metadata points, providing specific details that flesh out the persona while remaining true to the given information.
4. For any ranges or categories provided in the metadata, select and specify exact values or details within those ranges/categories.
5. Ensure diversity in perspectives, backgrounds, and personality traits. Provide enough specific details to make the persona feel real and three-dimensional.
6. Maintain diversity by acknowledging various experiences within the demographic group, but commit to specific details for this individual persona.
7. IMPORTANT: Place your entire response in the ### PERSONA GENERATION ### section below. Start your response with 'Persona:' and then provide only the persona description. Do not include any other prefixes, headers, or additional text.

### RESPONSE FORMAT ###
Persona: [A detailed, vivid, and diverse description of a specific individual. Ensure all details are consistent with and elaborate upon the provided metadata.]

### PERSONA METADATA ###
{METADATA}

### PERSONA GENERATION ###
"""
# Li et al. 2025, Appendix B.5 (opinion simulation); the letter-format line is replaced by the strict answer line
SIMULATION_SYSTEM = """You are an AI assistant tasked with generating realistic opinions based on a given persona and a specific topic.

### TASK ###
You will simulate a persona answering a multiple-choice opinion question. Select the answer that best matches your persona's viewpoint and interests.

### GUIDELINES ###
1. Be Faithful to the Persona: Ensure your answer is consistent with the persona's data.
2. Focus on Relevant Aspects: Center your reasoning on the relevant factors that would influence the persona's opinion on that topic.
3. Be Objective: Avoid injecting personal bias or overly politically correct views that may not align with the persona's standpoint.

### INSTRUCTIONS ###
- Choose ONE option that best fits the persona
- If multiple answers are possible, randomly select based on their probability
- Always pick an option, even in unclear cases - treat it as a forced-choice survey"""

PLACEMENTS = ("prefix", "system", "history")
GEN = "generated"


# ---------------------------------------------------------------- small helpers
def _norm(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip().casefold()


def _hidden(question: dict) -> set[str]:
    """Target id + exclude ids: never shown as a field, never matched on."""
    out = {str(x) for x in (question.get("exclude") or ())}
    if question.get("id") is not None:
        out.add(str(question["id"]))
    return out


def _stable(*parts: Any) -> str:
    return hashlib.sha256("\x00".join(map(str, parts)).encode()).hexdigest()


def age_bracket(age: Any) -> str | None:
    """Numeric age -> Anthology/GSS bracket ("18-29", "30-49", "50-64", "65+"); a bracket string passes through."""
    if isinstance(age, str) and age.strip() in ("18-29", "30-49", "50-64", "65+"):
        return age.strip()
    try:
        y = float(age)
    except (TypeError, ValueError):
        return None
    if math.isnan(y) or y < 18:
        return None
    return "18-29" if y < 30 else "30-49" if y < 50 else "50-64" if y < 65 else "65+"


def attach(people: Sequence[dict], texts: Mapping[str, Any], key: str) -> list[dict]:
    """Copies of `people` with person["generated"][key] = texts[id] (a story dict contributes its "text";
    people missing from `texts` are copied unchanged)."""
    out = []
    for p in people:
        q = dict(p)
        t = texts.get(str(p["id"]))
        if t is not None:
            q[GEN] = {**(p.get(GEN) or {}), key: t["text"] if isinstance(t, Mapping) else str(t)}
        out.append(q)
    return out


def _generated(inj: Injector, person: dict, key: str) -> str:
    text = (person.get(GEN) or {}).get(key)
    if not isinstance(text, str) or not text.strip():
        raise MissingPersonField(f"injector {inj.name!r} needs person['generated'][{key!r}] (needs {list(inj.needs)}; "
                                 f"person {person.get('id')!r}); build it with generated.{_BUILDER[key]} + attach()")
    return text.strip()


_BUILDER = {"backstory": "generate_backstories/match_backstories", "persona": "text_to_persona",
            "narrative": "narrate_people"}


def _calls(llm, prompts: Sequence[tuple[str, str]], max_tokens: int, llm_kw: dict | None,
           samples: Sequence[int] | None = None, parallel: int = 4) -> list[str]:
    """One complete() per prompt, in order; `samples[i]` goes out as sample= (independent disk-cache entries)."""
    from ...llm.parallel import parallel_map

    kw = dict(llm_kw or {})
    if samples is not None and "sample" in kw:
        raise ValueError("llm_kw must not set 'sample' here (the builder numbers the samples; use start=)")
    jobs = [(s, u, {"sample": samples[i]} if samples is not None else {}) for i, (s, u) in enumerate(prompts)]
    return [r.text for r in parallel_map(lambda j: llm.complete(j[0], j[1], max_tokens=max_tokens, **j[2], **kw),
                                         jobs, parallel=parallel)]


def _given(defaults: dict, values: dict) -> dict:
    """Params that differ from the defaults (short arm labels for directly built instances)."""
    return {k: v for k, v in values.items() if k not in defaults or v != defaults[k]}


# ---------------------------------------------------------------- Anthology: generation, demographics, matching
def backstory_prompt(question: str = BACKSTORY_QUESTION, preamble: str | None = None) -> str:
    """Base-model completion prompt "Question: <q>\\n\\nAnswer:" with an optional preamble Q/A first. The
    released default config uses preamble US_PREAMBLE and samples its answer until it contains "yes"; here the
    answer is fixed to "Yes." (default: no preamble, since the target population need not be US)."""
    head = f"Question: {preamble}\n\nAnswer: Yes.\n\n" if preamble else ""
    return f"{head}Question: {question}\n\nAnswer:"


def cut_at_stops(text: str, stops: Sequence[str] = BACKSTORY_STOPS) -> str:
    """Text up to the first stop string, as an API stop= would cut it (released code: "Question", "Q:",
    "Interview" anywhere, not only after a newline)."""
    text = str(text)
    hits = [i for i in (text.find(s) for s in stops) if i >= 0]
    return (text[:min(hits)] if hits else text).strip()


def generate_backstories(llm, n: int, *, question: str = BACKSTORY_QUESTION, preamble: str | None = None,
                         min_words: int = 30, max_tokens: int = 512, start: int = 0,
                         parallel: int = 4, llm_kw: dict | None = None) -> list[dict]:
    """n open-ended backstories [{"id": "bs<i>", "text": str, "question": question}]; call i uses sample=start+i
    so the disk cache keeps them apart. Replies are cut at BACKSTORY_STOPS; those under `min_words` words are
    dropped (the released loader keeps stories of > 40 tokens, about 30 words), so the result can be shorter
    than n."""
    if n < 0:
        raise ValueError("n must be >= 0")
    prompt = backstory_prompt(question, preamble)
    texts = _calls(llm, [("", prompt)] * n, max_tokens, llm_kw, samples=[start + i for i in range(n)],
                   parallel=parallel)
    out = []
    for i, t in enumerate(texts):
        t = cut_at_stops(t)
        if len(t.split()) >= min_words:
            out.append({"id": f"bs{start + i}", "text": t, "question": question})
    return out


def backstory_question_prompt(backstory: str, question: dict, consistency: bool = True,
                              elicitation: str = BACKSTORY_QUESTION) -> str:
    """Backstory as the prefix, then the question (paper layout), with the strict answer line."""
    head = f"Question: {elicitation}\n\nAnswer: {backstory.strip()}\n\n"
    return head + (CONSISTENCY + "\n" if consistency else "") + "Question: " + Injector.question_block(question)


def estimate_demographics(llm, stories: Sequence[dict], questions: Sequence[dict], *,
                          samples: int = DEMOGRAPHIC_SAMPLES, max_tokens: int = 32, parse: Callable = parse_option,
                          parallel: int = 4, llm_kw: dict | None = None) -> list[dict]:
    """Ask every backstory each demographic question (`id` = the trait name matched against person fields,
    `options` = trait values) `samples` times; returns copies with "demographics": {trait: {value: share}} over
    parsed replies ({} when nothing parsed). The story follows its own elicitation question (story["question"],
    else BACKSTORY_QUESTION). Needs a sampling LLM: replies at temperature 0 make it a point mass."""
    for q in questions:
        if not q.get("options") or "id" not in q:
            raise ValueError("demographic questions need an id (the trait) and options")
    out = []
    for s in stories:
        demo = {}
        for q in questions:
            prompt = backstory_question_prompt(s["text"], q, elicitation=s.get("question") or BACKSTORY_QUESTION)
            texts = _calls(llm, [("", prompt)] * samples, max_tokens, llm_kw, samples=list(range(samples)),
                           parallel=parallel)
            vals = [v for v in (parse(t, q) for t in texts) if v is not None]
            demo[str(q["id"])] = {o: vals.count(o) / len(vals) for o in sorted(set(vals))} if vals else {}
        out.append({**s, "demographics": demo})
    return out


def _story_prob(story: dict, key: str, value: Any) -> float:
    d = (story.get("demographics") or {}).get(key)
    if d is None or value is None:                          # no estimate, or the person lacks the trait
        return 0.0
    want = _norm(value)
    if isinstance(d, Mapping):
        return float(sum(float(p) for v, p in d.items() if _norm(v) == want))
    return 1.0 if _norm(d) == want else 0.0                 # a point value (e.g. a released bracket)


DERIVED = {"age_bracket": "age"}                            # trait -> the field it is derived from


def _trait_sources(keys: Sequence[str]) -> set[str]:
    """Field names a match on `keys` reads (a derived trait reads its source field too)."""
    return {str(k) for k in keys} | {DERIVED[k] for k in keys if k in DERIVED}


def _trait(fields: dict, key: str) -> Any:
    if key in fields and fields[key] not in (None, ""):
        return fields[key]
    if key == "age_bracket" and fields.get("age") not in (None, ""):
        return age_bracket(fields["age"])                 # released stories are labelled by bracket only
    return None


def edge_weight(fields: dict, story: dict, keys: Sequence[str]) -> float:
    """Paper edge weight: product over traits of the story's probability of the person's value."""
    w = 1.0
    for k in keys:
        w *= _story_prob(story, k, _trait(fields, k))
    return w


def _check_pool(pool: Sequence[dict]) -> list[dict]:
    pool = [dict(s, id=str(s.get("id", f"s{i}"))) for i, s in enumerate(pool)]
    if not pool:
        raise ValueError("empty backstory pool")
    for s in pool:
        if not isinstance(s.get("text"), str) or not isinstance(s.get("demographics"), Mapping):
            raise ValueError(f"story {s['id']!r} needs 'text' and a 'demographics' mapping (estimate_demographics)")
    if len({s["id"] for s in pool}) != len(pool):
        raise ValueError("story ids must be unique")
    return pool


def _greedy_pick(fields: dict, pool: Sequence[dict], keys: Sequence[str], seed: Any, pid: str) -> tuple[dict, float]:
    """argmax edge weight; ties (incl. all-zero) broken by a stable hash of (seed, person, story)."""
    best = max(pool, key=lambda s: (edge_weight(fields, s, keys), _stable(seed, pid, s["id"])))
    return best, edge_weight(fields, best, keys)


def match_backstories(people: Sequence[dict], pool: Sequence[dict], keys: Sequence[str], *,
                      method: str = "greedy", seed: Any = 0) -> dict[str, dict]:
    """{person_id: story} by the paper's edge weight. greedy: argmax per person (stories reused);
    hungarian: one-to-one maximum weight matching (scipy; needs len(pool) >= len(people))."""
    pool = _check_pool(pool)
    keys = list(keys)
    ids = [str(p["id"]) for p in people]
    if len(set(ids)) != len(ids):
        raise ValueError("person ids must be unique")
    lacking = [i for i, p in zip(ids, people) if any(_trait(Injector.person_fields(p), k) is None for k in keys)]
    if lacking:                                           # weight 0 everywhere would hand them an arbitrary story
        raise MissingPersonField(f"people {lacking[:5]}{' ...' if len(lacking) > 5 else ''} lack a trait in {keys}")
    if method == "greedy":
        return {str(p["id"]): _greedy_pick(Injector.person_fields(p), pool, keys, seed, str(p["id"]))[0]
                for p in people}
    if method != "hungarian":
        raise ValueError("method must be 'greedy' or 'hungarian'")
    if len(pool) < len(people):
        raise ValueError(f"hungarian matching needs at least as many stories ({len(pool)}) as people ({len(people)})")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("method='hungarian' needs scipy (pip install scipy); method='greedy' does not") from e
    w = [[edge_weight(Injector.person_fields(p), s, keys) for s in pool] for p in people]
    rows, cols = linear_sum_assignment(w, maximize=True)
    return {str(people[r]["id"]): pool[c] for r, c in zip(rows, cols)}


POOLS: dict[str, list[dict]] = {}


def register_pool(name: str, stories: Sequence[dict], *, replace: bool = False) -> None:
    """Make a story pool usable as `anthology_greedy(pool=name)` (short arm labels)."""
    if name == "released":
        raise ValueError("'released' is reserved for the published Anthology stories")
    if name in POOLS and not replace:
        raise ValueError(f"pool {name!r} already registered (replace=True to overwrite)")
    POOLS[name] = _check_pool(stories)


def released_pool(data_root: str | None = None) -> list[dict]:
    """The published Anthology backstories (persona.methods.anthology loader, $SIMENGINE_DATA), as a pool whose
    only trait is "age_bracket" (a point value)."""
    from ..methods.anthology import load_stories

    out = []
    for i, s in enumerate(load_stories(data_root)):
        out.append({"id": str(s.get("sid", s.get("id", f"a{i}"))), "text": s["text"], "question": s["question"],
                    "demographics": {"age_bracket": s["bracket"]}})
    return _check_pool(out)


# ---------------------------------------------------------------- Persona Hub
def text_to_persona(llm, people: Sequence[dict], *, relation: str = "write", source: str = "text",
                    max_tokens: int = 200, parallel: int = 4, llm_kw: dict | None = None) -> dict[str, str]:
    """{person_id: persona} inferred from each person's own text (person[source]: a post, review, essay, ...).
    People without that text are skipped. The text must predate the questions asked (it is shown verbatim to
    the generator)."""
    if relation not in RELATIONS:
        raise ValueError(f"relation must be one of {RELATIONS}")
    todo = [p for p in people if isinstance(p.get(source), str) and p[source].strip()]
    prompts = [("", TEXT_TO_PERSONA.format(relation=relation, text=p[source].strip())) for p in todo]
    return {str(p["id"]): t.strip() for p, t in zip(todo, _calls(llm, prompts, max_tokens, llm_kw, parallel=parallel))}


def persona_to_persona(llm, personas: Sequence[str], *, rounds: int = 1, max_tokens: int = 200,
                       parallel: int = 4, llm_kw: dict | None = None) -> list[str]:
    """Population expansion: each round asks for one close-relationship persona of every persona from the
    previous round (paper: six rounds per text-to-persona persona). Returns only the new personas, in order."""
    out, frontier = [], [str(p) for p in personas]
    for _ in range(max(0, int(rounds))):
        prompts = [("", PERSONA_TO_PERSONA.format(persona=p)) for p in frontier]
        frontier = [t.strip() for t in _calls(llm, prompts, max_tokens, llm_kw, parallel=parallel)]
        out += frontier
    return out


def dedup_personas(personas: Sequence[str], threshold: float = 0.9) -> list[str]:
    """Drop a persona whose word 1-gram Jaccard with an earlier kept one is >= threshold (paper: MinHash on
    1-grams, 128 permutations, 0.9; exact Jaccard here; the paper's second, embedding-cosine > 0.9 pass is not
    done). Order kept."""
    kept, sets = [], []
    for p in personas:
        s = set(re.findall(r"\w+", str(p).casefold()))
        if any(len(s & t) / max(1, len(s | t)) >= threshold for t in sets):
            continue
        kept.append(p)
        sets.append(s)
    return kept


# ---------------------------------------------------------------- narrative from fields (Li et al. 2025)
def _field_lines(fields: dict, keys: Sequence[str] | None, hide: set[str] = frozenset()) -> list[str]:
    keys = list(keys) if keys is not None else list(fields)
    return [f"- {str(k).replace('_', ' ')}: {fields[k]}" for k in keys
            if k in fields and fields[k] not in (None, "") and str(k) not in hide]


def _metadata(fields: dict, keys: Sequence[str] | None, hide: set[str]) -> dict:
    keys = list(keys) if keys is not None else list(fields)
    return {str(k): fields[k] for k in keys if k in fields and fields[k] not in (None, "") and str(k) not in hide}


def narrative_prompt(person: dict, keys: Sequence[str] | None = None, words: int | None = None,
                     exclude: Sequence[str] = ()) -> tuple[str, str]:
    """(system, user): the paper's descriptive-persona prompt (B.1 + B.4) with the person's fields as the
    metadata (JSON; never answers). `words` adds a length line the paper does not have. The narrative is
    written once per person, before any question: pass `exclude` = field names that are (or proxy) target
    questions so they never reach the generator."""
    meta = _metadata(Injector.person_fields(person), keys, {str(x) for x in exclude})
    if not meta:
        raise MissingPersonField(f"narrative needs at least one field (person {person.get('id')!r})")
    user = NARRATIVE_USER.replace("{METADATA}", json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    if words is not None:
        user = user.replace("### PERSONA GENERATION ###\n", f"Keep the persona under {int(words)} words.\n\n"
                                                          "### PERSONA GENERATION ###\n")
    return NARRATIVE_SYSTEM, user


def strip_persona_label(text: str) -> str:
    """Reply without the "Persona:" label the B.4 prompt asks the generator to start with."""
    return re.sub(r"^\s*(?:#+\s*PERSONA GENERATION\s*#+\s*)?Persona\s*:\s*", "", str(text), flags=re.I).strip()


def narrate_people(llm, people: Sequence[dict], *, keys: Sequence[str] | None = None, words: int | None = None,
                   exclude: Sequence[str] = (), max_tokens: int = 1024, parallel: int = 4,
                   llm_kw: dict | None = None) -> dict[str, str]:
    """{person_id: narrative} written by `llm` from each person's fields (`keys` limits which, `exclude` drops
    target-question fields); the leading "Persona:" label is stripped."""
    prompts = [narrative_prompt(p, keys, words, exclude) for p in people]
    return {str(p["id"]): strip_persona_label(t)
            for p, t in zip(people, _calls(llm, prompts, max_tokens, llm_kw, parallel=parallel))}


# ---------------------------------------------------------------- injectors
def _render_backstory(backstory: str, question: dict, placement: str, elicitation: str,
                      consistency: bool) -> Prompt:
    if placement == "prefix":
        return Prompt(system="", user=backstory_question_prompt(backstory, question, consistency, elicitation))
    if placement == "system":
        from ..methods.anthology import TEMPLATE

        return Prompt(system=TEMPLATE.format(question=elicitation, text=backstory),
                      user=Injector.question_block(question))
    lead = CONSISTENCY + "\n" if consistency else ""
    return Prompt(system="", user=lead + Injector.question_block(question),
                  history=[("user", elicitation), ("assistant", backstory)])


class _BackstoryParams(Injector):
    """elicitation=None: the story's own question (pool stories carry one), else BACKSTORY_QUESTION."""
    _DEFAULTS = {"placement": "prefix", "consistency": True, "elicitation": None}

    def __init__(self, placement: str = "prefix", consistency: bool = True,
                 elicitation: str | None = None, **params: Any):
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}")
        super().__init__(**_given(self._DEFAULTS, {"placement": placement, "consistency": consistency,
                                                   "elicitation": elicitation, **params}))
        self.placement, self.consistency, self.elicitation = placement, bool(consistency), elicitation


@register
class AnthologyBackstory(_BackstoryParams):
    """The person's assigned backstory (generate + estimate + match beforehand), then the question."""
    name = "anthology_backstory"
    family = "generated"
    paper = ANTHOLOGY
    needs = ("generated",)
    description = ("person['generated']['backstory'] as an interview answer before the question; placement "
                   "prefix (paper, user turn) | system | history (prior chat turn).")

    def render(self, person: dict, question: dict) -> Prompt:
        p = _render_backstory(_generated(self, person, "backstory"), question, self.placement,
                              self.elicitation or BACKSTORY_QUESTION, self.consistency)
        p.meta["placement"] = self.placement
        return p


@register
class AnthologyGreedy(_BackstoryParams):
    """Greedy matching at render time: the pool story with the highest edge weight for this person's fields."""
    name = "anthology_greedy"
    family = "generated"
    paper = ANTHOLOGY
    needs = ("fields",)
    description = ("Pick the backstory maximising prod_k P_story(person's trait k) from a registered pool "
                   "(register_pool) or pool='released' (published stories, age_bracket only); ties by seed.")

    def __init__(self, pool: str | Sequence[dict] = "released", keys: Sequence[str] | None = None, seed: int = 0,
                 data_root: str | None = None, **params: Any):
        super().__init__(**_given({"pool": "released", "keys": None, "seed": 0, "data_root": None},
                                  {"pool": pool, "keys": keys, "seed": seed, "data_root": data_root}), **params)
        self.pool_ref, self.seed, self.data_root = pool, seed, data_root
        self.keys = list(keys) if keys is not None else (["age_bracket"] if pool == "released" else None)
        if not self.keys:
            raise ValueError("anthology_greedy needs keys= (traits to match on)")
        self._pool: list[dict] | None = None

    @property
    def label(self) -> str:
        """A list pool shows as its size and a content hash, not its full text."""
        params = dict(self.params)
        if not isinstance(params.get("pool", ""), str):
            stories = list(params["pool"])
            digest = _stable(*(f"{s.get('id')}\x01{s.get('text')}" if isinstance(s, Mapping) else s for s in stories))
            params["pool"] = f"<{len(stories)} stories {digest[:8]}>"
        return f"{self.name}({','.join(f'{k}={v}' for k, v in sorted(params.items()))})" if params else self.name

    def pool(self) -> list[dict]:
        if self._pool is None:
            if isinstance(self.pool_ref, str):
                if self.pool_ref == "released":
                    self._pool = released_pool(self.data_root)
                elif self.pool_ref in POOLS:
                    self._pool = POOLS[self.pool_ref]
                else:
                    raise ValueError(f"unknown pool {self.pool_ref!r}; register_pool() it or use 'released'")
            else:
                self._pool = _check_pool(self.pool_ref)
        return self._pool

    def check(self, person: dict) -> None:
        super().check(person)
        f = self.person_fields(person)
        missing = [k for k in self.keys if _trait(f, k) is None]
        if missing:
            raise MissingPersonField(f"injector {self.name!r} needs person field(s) {list(self.needs)} with traits "
                                     f"{missing} (person {person.get('id')!r})")

    def render(self, person: dict, question: dict) -> Prompt:
        leak = _hidden(question) & _trait_sources(self.keys)
        if leak:
            raise ValueError(f"anthology_greedy keys {sorted(leak)} include the target / excluded question")
        story, w = _greedy_pick(self.person_fields(person), self.pool(), self.keys, self.seed, str(person["id"]))
        p = _render_backstory(story["text"], question, self.placement,
                              self.elicitation or story.get("question") or BACKSTORY_QUESTION, self.consistency)
        p.meta.update(story=story["id"], weight=w, placement=self.placement)
        return p


@register
class PersonaHub(Injector):
    """A Persona Hub style one-paragraph persona description (text-to-persona from the person's own text)."""
    name = "personahub"
    family = "generated"
    paper = PERSONA_HUB
    needs = ("generated",)
    description = ("person['generated']['persona'] (text_to_persona); style roleplay (system: adopt the persona) "
                   "| prefix (persona before the question in the user turn, as in Persona Hub prompts).")

    def __init__(self, style: str = "roleplay", **params: Any):
        if style not in ("roleplay", "prefix"):
            raise ValueError("style must be 'roleplay' or 'prefix'")
        super().__init__(**_given({"style": "roleplay"}, {"style": style}), **params)
        self.style = style

    def render(self, person: dict, question: dict) -> Prompt:
        persona = _generated(self, person, "persona")
        if self.style == "roleplay":
            return Prompt(system=f"Adopt this persona and answer as this person would.\n\nPersona: {persona}",
                          user=self.question_block(question))
        return Prompt(system="", user=f"Persona: {persona}\n\nAnswer the following question as this persona "
                                      f"would.\n\n{self.question_block(question)}")


@register
class LLMNarrative(Injector):
    """Descriptive persona: an LLM-written narrative from the person's structured fields."""
    name = "llm_narrative"
    family = "generated"
    paper = LI2025
    needs = ("generated",)
    description = ("person['generated']['narrative'] (narrate_people); style paper (B.5 opinion-simulation "
                   "prompt: guidelines in system, ### PERSONA ### + ### QUESTION ### in user) | system (narrative "
                   "as the system prompt); with_fields=True adds the source fields as key: value lines.")

    def __init__(self, style: str = "paper", with_fields: bool = False, keys: Sequence[str] | None = None,
                 **params: Any):
        if style not in ("paper", "system"):
            raise ValueError("style must be 'paper' or 'system'")
        super().__init__(**_given({"style": "paper", "with_fields": False}, {"style": style, "with_fields": with_fields}),
                         **({"keys": list(keys)} if keys else {}), **params)
        self.style, self.with_fields, self.keys = style, bool(with_fields), (list(keys) if keys else None)
        if self.with_fields:
            self.needs = ("generated", "fields")

    def render(self, person: dict, question: dict) -> Prompt:
        persona = _generated(self, person, "narrative")
        if self.with_fields:
            lines = _field_lines(self.person_fields(person), self.keys, _hidden(question))
            if lines:
                persona += "\n\nProfile:\n" + "\n".join(lines)
        if self.style == "system":
            return Prompt(system=persona + "\n\nAnswer as this person would.", user=self.question_block(question))
        return Prompt(system=_simulation_system(question),
                      user=f"### PERSONA ###\n{persona}\n\n### QUESTION ###\n{self.question_block(question)}\n\n"
                           "### YOUR RESPONSE ###")


def _simulation_system(question: dict) -> str:
    """B.5 system text; for a numeric question the multiple-choice wording and option rules are dropped."""
    if question.get("options"):
        return SIMULATION_SYSTEM
    head = SIMULATION_SYSTEM.split("\n\n### INSTRUCTIONS ###")[0]
    return head.replace("a multiple-choice opinion question. Select the answer",
                        "an opinion question. Give the answer")


__all__ = ["AnthologyBackstory", "AnthologyGreedy", "PersonaHub", "LLMNarrative", "BACKSTORY_QUESTION",
           "BACKSTORY_QUESTION_SHORT", "BACKSTORY_STOPS", "US_PREAMBLE", "CONSISTENCY", "NARRATIVE_SYSTEM",
           "NARRATIVE_USER", "SIMULATION_SYSTEM", "POOLS", "age_bracket", "attach", "backstory_prompt",
           "cut_at_stops", "backstory_question_prompt", "generate_backstories", "estimate_demographics",
           "edge_weight", "match_backstories", "register_pool", "released_pool", "text_to_persona",
           "persona_to_persona", "dedup_personas", "narrative_prompt", "strip_persona_label", "narrate_people"]

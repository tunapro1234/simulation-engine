"""Open-model fine-tuning and activation steering for personas: prompt formats, dataset builders, a Colab T4 recipe.

NOT MEASURED IN SIMENGINE. These methods change an open model's weights or activations, so an API LLM cannot run
them: through `compare_injectors` with a text-only backend the steering and soft-prompt arms see only the bare
question (their persona lives in `Prompt.meta`), and the fine-tune arms see a prompt meant for a tuned model.
The module gives the training/inference prompt format, chat-format JSONL builders and the open-model helpers
(`train_colab_t4`, `steering_vector`, `steer`), which import torch / transformers / peft lazily.

    import simengine.persona.inject as pi
    from simengine.persona.inject import finetune as ft
    inj = pi.get_injector("centaur_qlora")
    rows = ft.chat_rows(people, questions, inj)                   # one row per (person, question) with truth
    train, test = ft.holdout_split(rows, frac=0.2, seed=0)        # held-out people, as in Centaur
    ft.write_jsonl(train, "train.jsonl")
    cfg = ft.recipe("centaur")                                    # QLoRA r=8, all linear layers, 1 epoch
    ft.train_colab_t4("train.jsonl", "adapter_out", cfg)          # Colab T4; needs torch, transformers, peft
    pairs = ft.caa_pairs(questions, {"q1": "yes"}, seed=0)        # contrastive A/B pairs for CAA

Injectors (family "finetune"): subpop_lora, centaur_qlora, caa_steer, persona_vector_steer, soft_prompt.
`parse_finetuned` reads <<answer>>, a JSON answer, an exact option or a bare option letter.
"""
from __future__ import annotations

import contextlib
import json
import random
import re
import string
import zlib
from typing import Any, Iterable, Mapping, Sequence

from ._base import Injector, MissingPersonField, Prompt, get_injector, parse_option, register

FAMILY = "finetune"
NOT_MEASURED = "Not measured in simengine."
LETTERS = string.ascii_uppercase
DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"   # fits a 16 GB T4 in 4-bit with room for LoRA activations
_BRACKETS = re.compile(r"<<\s*(.*?)\s*>>", re.S)
_LETTER = re.compile(r"^\s*\(?([A-Z])\)?[.:]?\s*$")
_DEPS_HINT = ("pip install torch transformers peft (bitsandbytes for 4-bit); Colab T4 runtimes ship torch and "
              "transformers")


# ---------------------------------------------------------------- formatting helpers

def option_letters(question: dict) -> list[tuple[str, str]]:
    """[(letter, option)] for a choice question (A, B, ...); [] for a number question."""
    opts = [str(o) for o in (question.get("options") or [])]
    if len(opts) > len(LETTERS):
        raise ValueError(f"question {question.get('id')!r} has more than {len(LETTERS)} options")
    return list(zip(LETTERS, opts))


def parse_finetuned(text: Any, question: dict) -> Any:
    """Reply -> answer | None. Tries <<answer>> (Centaur), then parse_option (JSON / exact option / number), then a
    bare option letter "B", "(B)" or "B." (SubPOP / CAA prompts)."""
    if text is None:
        return None
    s = str(text)
    m = _BRACKETS.search(s)
    if m:
        s = m.group(1)
    v = parse_option(s, question)
    if v is not None:
        return v
    lm = _LETTER.match(s)
    if lm:
        letters = dict(option_letters(question))
        return letters.get(lm.group(1))
    return None


def _short(v: Any) -> str:
    r = str(v)
    return f"<{type(v).__name__}:{len(v)}>" if isinstance(v, (dict, list, tuple)) and len(r) > 40 else r


def _options_line(question: dict) -> str:
    if question.get("options"):
        return "Options: " + ", ".join(str(o) for o in question["options"]) + "."
    lo, hi = question.get("min"), question.get("max")
    return f"Answer with a number from {lo} to {hi}." if lo is not None and hi is not None else "Answer with a number."


class _Finetune(Injector):
    """Shared base: family, compact labels, and the assistant target text a training row ends with."""
    family = FAMILY

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        return f"{self.name}({','.join(f'{k}={_short(v)}' for k, v in sorted(self.params.items()))})"

    def target_text(self, question: dict, answer: Any) -> str:
        """What the tuned model should emit for `answer` (training label)."""
        return json.dumps({"answer": answer})

    def _fields(self, person: dict, keys: Sequence[str] | None) -> dict:
        f = self.person_fields(person)
        if keys is None:
            out = {k: v for k, v in f.items() if v not in (None, "")}
            if not out:
                raise MissingPersonField(f"injector {self.name!r} found no non-empty fields "
                                         f"(person {person.get('id')!r}; needs={list(self.needs)})")
            return out
        missing = [k for k in keys if f.get(k) in (None, "")]
        if missing:
            raise MissingPersonField(f"injector {self.name!r} needs fields {missing} in person['fields'] "
                                     f"(person {person.get('id')!r}; needs={list(self.needs)})")
        return {k: f[k] for k in keys}


# ---------------------------------------------------------------- fine-tuned prompt formats

@register
class SubpopLoRA(_Finetune):
    """SubPOP (Suh et al. 2025): LoRA-tune a base LM so its next-token distribution over option letters matches a
    subpopulation's answer distribution (forward KL). The persona is a steering text from OpinionQA's three
    formats: qa (attribute Q/A lines), bio (first person), portray (\"Answer ... as if ...\"). portray follows the
    paper's example wording; qa and bio are generic templates over field names, not OpinionQA's per-attribute
    questions and sentences. Not measured in simengine."""
    name = "subpop_lora"
    paper = "Suh et al. 2025, SubPOP, arXiv:2502.16761"
    needs = ("fields",)
    description = ("Steering text (qa / bio / portray) + lettered question ending 'Answer:'; target is the option "
                   "letter, or a group distribution via distribution_rows. LoRA r=8 on q,v, forward-KL loss. "
                   + NOT_MEASURED)
    STYLES = ("qa", "bio", "portray")

    def __init__(self, style: str = "portray", keys: Sequence[str] | None = None, **params: Any):
        if style not in self.STYLES:
            raise ValueError(f"style must be one of {self.STYLES}, not {style!r}")
        super().__init__(style=style, **({"keys": list(keys)} if keys else {}), **params)
        self.style, self.keys = style, (list(keys) if keys else None)

    def steering_text(self, fields: Mapping[str, Any]) -> str:
        items = [(k.replace("_", " "), v) for k, v in fields.items()]
        if self.style == "qa":
            return "\n\n".join(f"Question: What is your {k}?\nAnswer: {v}" for k, v in items)
        if self.style == "bio":
            return "About me: " + " ".join(f"My {k} is {v}." for k, v in items)
        parts = [f"your {k} is {v}" for k, v in items]
        joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
        return f"Answer the following question as if {joined}."

    def render(self, person: dict, question: dict) -> Prompt:
        steer = self.steering_text(self._fields(person, self.keys))
        letters = option_letters(question)
        if letters:
            body = (f"Question: {question['text']}\n" + "\n".join(f"{l}. {o}" for l, o in letters) + "\nAnswer:")
        else:
            body = self.question_block(question)
        return Prompt(system="", user=f"{steer}\n\n{body}", meta={"style": self.style})

    def target_text(self, question: dict, answer: Any) -> str:
        for letter, opt in option_letters(question):
            if opt == str(answer):
                return letter
        if question.get("options"):
            raise ValueError(f"answer {answer!r} is not an option of question {question.get('id')!r}")
        return json.dumps({"answer": answer})


@register
class CentaurQLoRA(_Finetune):
    """Centaur / Psych-101 (Binz et al. 2025): QLoRA on natural-language transcripts of a participant's session,
    with the participant's responses wrapped in << >> and loss only on those tokens. Here the session is the
    person's known answers (known_answers: target and `exclude` dropped), then the target question.
    Not measured in simengine."""
    name = "centaur_qlora"
    paper = "Binz et al. 2025, Centaur / Psych-101, arXiv:2410.20268"
    needs = ("answers",)
    description = ("Transcript of the person's other answers marked <<answer>> (chat turns or one plain text), then "
                   "the target question; target '<<answer>>'. QLoRA r=8 on all linear layers, 1 epoch. "
                   + NOT_MEASURED)
    INSTRUCTIONS = ("In this survey you answer questions one at a time. Your answers are marked with << and >>.")

    def __init__(self, texts: Mapping[str, str] | None = None, chat: bool = True, max_turns: int | None = None,
                 seed: int | None = None, **params: Any):
        given = {"chat": chat} if chat is not True else {}
        given.update({k: v for k, v in (("texts", dict(texts) if texts else None), ("max_turns", max_turns),
                                        ("seed", seed)) if v is not None})
        super().__init__(**given, **params)
        self.texts = {str(k): str(v) for k, v in (texts or {}).items()}
        self.chat, self.max_turns, self.seed = chat, max_turns, seed

    def _turns(self, person: dict, question: dict) -> list[tuple[str, Any]]:
        known = list(self.known_answers(person, question).items())
        if self.seed is not None:   # order shuffle keyed on (seed, person, question): no hash() randomization
            rng = random.Random(zlib.crc32(f"{self.seed}|{person.get('id')}|{question.get('id')}".encode()))
            rng.shuffle(known)
        if self.max_turns is not None:
            known = known[-self.max_turns:] if self.max_turns > 0 else []
        return [(self.texts.get(str(k), str(k)), v) for k, v in known]

    def _ask(self, question: dict) -> str:
        return f"{question['text']} {_options_line(question)}"

    def render(self, person: dict, question: dict) -> Prompt:
        turns = self._turns(person, question)
        meta = {"n_context": len(turns)}
        if self.chat:
            hist: list[tuple[str, str]] = []
            for q, a in turns:
                hist += [("user", q), ("assistant", f"<<{a}>>")]
            return Prompt(system=self.INSTRUCTIONS, user=self._ask(question), history=hist, meta=meta)
        lines = [self.INSTRUCTIONS, ""] + [f"{q}\nYou answer <<{a}>>." for q, a in turns]
        lines.append(f"{self._ask(question)}\nYou answer")
        return Prompt(system="", user="\n".join(lines), meta=meta)

    def target_text(self, question: dict, answer: Any) -> str:
        return f"<<{answer}>>"


# ---------------------------------------------------------------- activation steering / soft prompts

class _Steer(_Finetune):
    """Shared: per-person trait coefficients from fields; the prompt text carries no persona."""
    needs = ("fields",)
    method = ""
    positions = ""

    def __init__(self, traits: Mapping[str, str] | Sequence[str] = (), levels: Mapping[str, float] | None = None,
                 center: float = 0.0, multiplier: float = 1.0, layer: int | None = None, **params: Any):
        tmap = dict(traits) if isinstance(traits, Mapping) else {t: t for t in traits}
        if not tmap:
            raise ValueError(f"{self.name}: give traits=[field, ...] or {{trait: field}}")
        given: dict = {"traits": tmap}
        if levels:
            given["levels"] = dict(levels)
        if center:
            given["center"] = center
        if multiplier != 1.0:
            given["multiplier"] = multiplier
        if layer is not None:
            given["layer"] = layer
        super().__init__(**given, **params)
        self.traits, self.levels = tmap, {str(k): float(v) for k, v in (levels or {}).items()}
        self.center, self.multiplier, self.layer = float(center), float(multiplier), layer

    def coefs(self, person: dict) -> dict[str, float]:
        """{trait: multiplier * (value - center)}; categorical values go through `levels`."""
        f = self._fields(person, list(self.traits.values()))
        out = {}
        for trait, key in self.traits.items():
            v = f[key]
            if isinstance(v, str) and v in self.levels:
                x = self.levels[v]
            else:
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    raise ValueError(f"{self.name}: field {key!r}={v!r} is not numeric; map it with levels=") from None
            out[trait] = round(self.multiplier * (x - self.center), 9)
        return out

    def render(self, person: dict, question: dict) -> Prompt:
        steer = {"method": self.method, "layer": self.layer, "positions": self.positions, "coefs": self.coefs(person)}
        return Prompt(system="", user=self.question_block(question), meta={"steer": steer})


@register
class CAASteer(_Steer):
    """Contrastive Activation Addition (Panickssery et al. 2024): steering vector = mean residual-stream difference
    at the answer-letter position over A/B pairs (caa_pairs); added at every position after the user's prompt with
    a signed multiplier. The person's trait value sets the multiplier. Not measured in simengine."""
    name = "caa_steer"
    paper = "Panickssery et al. 2024, Contrastive Activation Addition, arXiv:2312.06681"
    description = ("Bare question; meta['steer'] = per-person coefficients for CAA vectors (answer-letter position, "
                   "added after the prompt). Paper sweeps layers; 13 was best for Llama-2-7B-Chat. " + NOT_MEASURED)
    method, positions = "caa", "after_prompt"


@register
class PersonaVectorSteer(_Steer):
    """Persona vectors (Chen et al. 2025): vector = mean residual activation over response tokens under
    trait-eliciting minus trait-suppressing system prompts (judge-filtered rollouts), one per layer, the layer
    picked by steering sweep; h <- h + alpha * v at each decoding step. Not measured in simengine."""
    name = "persona_vector_steer"
    paper = "Chen et al. 2025, Persona vectors, arXiv:2507.21509"
    description = ("Bare question; meta['steer'] = per-person alpha for persona vectors (response-token mean "
                   "difference, added at each decoding step). " + NOT_MEASURED)
    method, positions = "persona_vector", "every_decode_step"


@register
class SoftPrompt(_Finetune):
    """Prompt tuning (Lester et al. 2021) adapted to personas: one learned soft prompt (n virtual tokens, frozen
    LM) per subpopulation key built from `keys` fields. The text carries no persona; meta names the soft prompt.
    Not measured in simengine."""
    name = "soft_prompt"
    paper = "Lester et al. 2021, prompt tuning, arXiv:2104.08691"
    needs = ("fields",)
    description = ("Bare question; meta['soft_prompt'] = group key from fields (one tuned soft prompt per group), "
                   "n_virtual_tokens. Frozen LM. " + NOT_MEASURED)

    def __init__(self, keys: Sequence[str] | None = None, n_tokens: int = 20, **params: Any):
        if n_tokens < 1:
            raise ValueError("n_tokens must be >= 1")
        given = {"keys": list(keys)} if keys else {}
        if n_tokens != 20:
            given["n_tokens"] = n_tokens
        super().__init__(**given, **params)
        self.keys, self.n_tokens = (list(keys) if keys else None), n_tokens

    def group_key(self, person: dict) -> str:
        f = self._fields(person, self.keys)
        return "|".join(f"{k}={f[k]}" for k in sorted(f))

    def render(self, person: dict, question: dict) -> Prompt:
        return Prompt(system="", user=self.question_block(question),
                      meta={"soft_prompt": self.group_key(person), "n_virtual_tokens": self.n_tokens})


# ---------------------------------------------------------------- dataset builders

def _truth(truth) -> Any:
    from .compare import _truth_fn
    return _truth_fn(truth)


def _messages(p: Prompt) -> list[dict]:
    head = [{"role": "system", "content": p.system}] if p.system else []
    return head + p.messages()


def chat_rows(people: Sequence[dict], questions: Sequence[dict], injector: Injector | str = "centaur_qlora",
              truth="answers", *, flatten: bool = False, **params: Any) -> list[dict]:
    """Chat-format training rows {"messages": [...], "person", "question", "injector"[, "meta"]}, one per
    (person, question) with truth. The prompt is the injector's own render (so training and inference formats
    match; persona answers go through known_answers, never the target). The last message is the assistant target
    (`target_text`, JSON {"answer": ...} for injectors outside this family). flatten=True folds history into one
    user turn. Not measured in simengine."""
    inj = get_injector(injector, **params)
    tf = _truth(truth)
    target = getattr(inj, "target_text", None) or (lambda q, a: json.dumps({"answer": a}))
    rows = []
    for person in people:
        for q in questions:
            a = tf(person, q)
            if a is None:
                continue
            p = inj(person, q)
            if flatten:
                p = Prompt(p.system, p.flatten(), meta=p.meta)
            row = {"messages": _messages(p) + [{"role": "assistant", "content": target(q, a)}],
                   "person": str(person["id"]), "question": str(q.get("id")), "injector": inj.label}
            extra = {k: v for k, v in p.meta.items() if k != "injector"}
            if extra:
                row["meta"] = extra
            rows.append(row)
    return rows


def distribution_rows(groups: Sequence[dict], questions: Sequence[dict], dist: Mapping[tuple[str, str], Mapping],
                      injector: Injector | str = "subpop_lora", **params: Any) -> list[dict]:
    """SubPOP rows: prompt messages (no assistant turn) + "target_dist" {letter: p} for the forward-KL loss.
    `groups` are person-like dicts (id, fields) describing subpopulations; dist[(group_id, qid)] = {option: count
    or share}. Raises on an unknown option, a negative weight or a number question. Not measured in simengine."""
    inj = get_injector(injector, **params)
    by_q = {str(q.get("id")): q for q in questions}
    by_g = {str(g["id"]): g for g in groups}
    rows = []
    for (gid, qid), d in dist.items():
        g, q = by_g.get(str(gid)), by_q.get(str(qid))
        if g is None or q is None:
            raise KeyError(f"dist key {(gid, qid)!r} has no matching group/question")
        letters = {o: l for l, o in option_letters(q)}
        if not letters:
            raise ValueError(f"question {qid!r} has no options; distribution rows need a choice question")
        bad = [o for o in d if str(o) not in letters]
        if bad:
            raise ValueError(f"options {bad} not in question {qid!r}")
        w = {letters[str(o)]: float(v) for o, v in d.items()}
        if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
            raise ValueError(f"dist for {(gid, qid)!r} needs non-negative weights with a positive sum")
        tot = sum(w.values())
        p = inj(g, q)
        rows.append({"messages": _messages(p), "target_dist": {l: w.get(l, 0.0) / tot for l in letters.values()},
                     "person": str(gid), "question": str(qid), "injector": inj.label})
    return rows


def caa_pairs(questions: Sequence[dict], behavior: Mapping[str, Any], *, seed: int = 0) -> list[dict]:
    """CAA contrastive pairs: {"prompt", "positive", "negative", "question"} where the prompt lists lettered
    choices and positive/negative are the completions "(A" / "(B" appended after it. They stop at the letter, so
    steering_vector(mode="last") reads the activation at the answer letter as in the paper (a closing ")" would
    move the read position). behavior[qid] = the option showing the trait; with more than two options, one pair
    per other option. Option order is shuffled per pair with `seed` so the trait is not always (A). For a chat
    model, wrap the prompt in its chat template before steering_vector. Not measured in simengine."""
    out = []
    for q in questions:
        qid = str(q.get("id"))
        if qid not in behavior:
            continue
        opts = [str(o) for o in (q.get("options") or [])]
        pos = str(behavior[qid])
        if pos not in opts or len(opts) < 2:
            raise ValueError(f"question {qid!r}: behavior option {pos!r} must be one of >= 2 options {opts}")
        for neg in (o for o in opts if o != pos):
            pair = [pos, neg]
            random.Random(zlib.crc32(f"{seed}|{qid}|{neg}".encode())).shuffle(pair)
            let = dict(zip(pair, "AB"))
            prompt = f"{q['text']}\n\nChoices:\n(A) {pair[0]}\n(B) {pair[1]}"
            out.append({"prompt": prompt, "positive": f"({let[pos]}", "negative": f"({let[neg]}",
                        "question": qid})
    return out


def persona_vector_pairs(questions: Sequence[dict], pos_systems: Sequence[str], neg_systems: Sequence[str],
                         rollouts: int = 10) -> list[dict]:
    """Persona-vector extraction plan: one row per (question, system-prompt pair, rollout, side) with "system",
    "user", "side" in {"pos", "neg"}. Generate a response per row with the open model, score trait expression
    0-100 with a judge, then keep rows via judge_filter. Paper: 5 prompt pairs, 20 extraction questions, 10
    rollouts. Not measured in simengine."""
    if len(pos_systems) != len(neg_systems) or not pos_systems:
        raise ValueError("pos_systems and neg_systems must be non-empty and the same length")
    if rollouts < 1:
        raise ValueError("rollouts must be >= 1")
    rows = []
    for q in questions:
        for i, (ps, ns) in enumerate(zip(pos_systems, neg_systems)):
            for r in range(rollouts):
                for side, s in (("pos", ps), ("neg", ns)):
                    rows.append({"system": s, "user": q["text"], "side": side, "pair": i, "rollout": r,
                                 "question": str(q.get("id"))})
    return rows


def judge_filter(rows: Iterable[dict], threshold: float = 50.0, score_key: str = "score") -> list[dict]:
    """Persona-vector filter: keep pos rows scoring above `threshold` and neg rows below it; unscored rows drop."""
    keep = []
    for r in rows:
        s = r.get(score_key)
        if s is None:
            continue
        if (r["side"] == "pos" and s > threshold) or (r["side"] == "neg" and s < threshold):
            keep.append(r)
    return keep


def holdout_split(rows: Sequence[dict], frac: float = 0.1, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """(train, test) by person: every row of a held-out person goes to test (Centaur evaluates held-out
    participants). Deterministic in `seed`."""
    if not 0 < frac < 1:
        raise ValueError("frac must be in (0, 1)")
    ids = sorted({r["person"] for r in rows})
    random.Random(seed).shuffle(ids)
    n = max(1, round(frac * len(ids))) if len(ids) > 1 else 0
    test_ids = set(ids[:n])
    return [r for r in rows if r["person"] not in test_ids], [r for r in rows if r["person"] in test_ids]


def write_jsonl(rows: Iterable[dict], path: str) -> int:
    """Write rows as JSON lines (UTF-8); returns the row count."""
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------- training recipe (Colab T4)

_RECIPES = {
    # Binz et al. 2025: 4-bit base, LoRA r=8 on all non-embedding linear layers, 1 epoch, lr 5e-5, wd 0.01,
    # effective batch 32, 8-bit AdamW, 100 warmup steps, loss on response tokens only. alpha and dropout are not
    # reported (16 and 0 are ours).
    "centaur": {"method": "qlora", "quant": "4bit", "r": 8, "alpha": 16, "dropout": 0.0, "target": "all-linear",
                "lr": 5e-5, "weight_decay": 0.01, "epochs": 1, "warmup_steps": 100, "batch": 32,
                "optimizer": "adamw_8bit", "loss": "assistant_tokens"},
    # Suh et al. 2025: LoRA r=8, alpha 32, dropout 0.05 on q,v (paper inits LoRA from N(0, 0.02); peft's default
    # init is used here); AdamW wd 0; lr 2e-4 at batch 256 for the 7B models (1e-4 for 13B, 2e-5 for 70B);
    # epochs not reported; forward KL.
    "subpop": {"method": "lora", "quant": None, "r": 8, "alpha": 32, "dropout": 0.05,
               "target": ["q_proj", "v_proj"], "lr": 2e-4, "weight_decay": 0.0, "epochs": 1, "warmup_steps": 0,
               "batch": 256, "optimizer": "adamw", "loss": "forward_kl"},
    # Lester et al. 2021 (T5): frozen LM, trainable prompt embeddings; Adafactor, constant lr 0.3, wd 1e-5,
    # beta2 decay 0.8, no parameter scaling, batch 32, 30,000 steps, default prompt length 100 (20+ tokens were
    # close to it at XL/XXL). 20 tokens and 1 epoch are ours. lr 0.3 was set for T5's embedding scale; a decoder
    # LM's embeddings are far smaller, so it likely needs lowering (untested).
    "soft_prompt": {"method": "prompt_tuning", "quant": None, "n_virtual_tokens": 20, "lr": 0.3,
                    "weight_decay": 1e-5, "epochs": 1, "warmup_steps": 0, "batch": 32, "optimizer": "adafactor",
                    "loss": "assistant_tokens"},
}
OPTIMIZERS = ("adamw", "adamw_8bit", "adafactor")


def recipe(paper: str = "centaur", base_model: str = DEFAULT_BASE, **overrides: Any) -> dict:
    """Training config for one Colab T4 (16 GB, fp16, no bf16): the paper's hyperparameters plus T4 settings
    (one row per forward pass, gradient accumulation to the paper's batch, max_len 1024 keeping the end of each
    row). Not measured in simengine."""
    if paper not in _RECIPES:
        raise ValueError(f"paper must be one of {sorted(_RECIPES)}")
    # train_colab_t4 runs rows one at a time (no padding), so micro_batch stays 1 and grad_accum = batch
    cfg = {"paper": paper, "base_model": base_model, "fp16": True, "bf16": False, "micro_batch": 1,
           "max_len": 1024, "seed": 0, **_RECIPES[paper]}
    cfg.update(overrides)
    cfg["grad_accum"] = max(1, int(cfg["batch"]) // int(cfg["micro_batch"]))
    return cfg


def _require(*mods: str):
    import importlib
    out = []
    for m in mods:
        try:
            out.append(importlib.import_module(m))
        except ImportError as e:
            raise ImportError(f"simengine.persona.inject.finetune needs {m!r} for this function ({_DEPS_HINT})") from e
    return out if len(out) > 1 else out[0]


def _chat_ids(tok, messages: list[dict], **kw: Any) -> list[int]:
    """apply_chat_template token ids; newer transformers return a BatchEncoding instead of a list."""
    out = tok.apply_chat_template(messages, tokenize=True, **kw)
    if isinstance(out, Mapping) or hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out)


def _tokenize_chat(tok, messages: list[dict], max_len: int) -> tuple[list[int], list[int]]:
    """input_ids and labels with loss only on assistant turns (Centaur: only the participant's responses).
    Truncation keeps the last `max_len` tokens: the target turn is at the end."""
    ids: list[int] = []
    labels: list[int] = []
    for i in range(len(messages)):
        full = _chat_ids(tok, messages[:i + 1])
        new = full[len(ids):]
        ids = full
        labels += new if messages[i]["role"] == "assistant" else [-100] * len(new)
    return ids[-max_len:], labels[-max_len:]


def _optimizer(torch, transformers, params: list, cfg: dict):
    """The paper's optimizer: AdamW, 8-bit AdamW (Centaur; bitsandbytes) or Adafactor (Lester et al.)."""
    name, lr, wd = cfg.get("optimizer", "adamw"), cfg["lr"], cfg["weight_decay"]
    if name == "adamw_8bit":
        return _require("bitsandbytes").optim.AdamW8bit(params, lr=lr, weight_decay=wd)
    if name == "adafactor":     # T5 settings: absolute lr, no parameter scaling, beta2 decay 0.8
        return transformers.optimization.Adafactor(params, lr=lr, weight_decay=wd, scale_parameter=False,
                                                   relative_step=False, warmup_init=False, decay_rate=-0.8)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def train_colab_t4(jsonl_path: str, out_dir: str, cfg: dict | None = None) -> dict:
    """Fine-tune `cfg["base_model"]` on chat_rows / distribution_rows JSONL and save the adapter to `out_dir`.
    LoRA / QLoRA (peft) or prompt tuning; fp16 autocast for a T4. Rows with "target_dist" use forward KL over
    the option-letter tokens at the answer position (SubPOP); others use cross-entropy on assistant tokens.
    Returns {"steps", "last_loss", "out_dir"}. Lazy imports: torch, transformers, peft (bitsandbytes for 4-bit).
    Not measured in simengine."""
    cfg = cfg or recipe()
    if int(cfg.get("micro_batch", 1)) != 1:
        raise ValueError("train_colab_t4 runs one row per forward pass; use micro_batch=1 (grad_accum = batch)")
    if cfg.get("optimizer", "adamw") not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
    torch, transformers, peft = _require("torch", "transformers", "peft")
    rows = read_jsonl(jsonl_path)
    torch.manual_seed(cfg["seed"])
    tok = transformers.AutoTokenizer.from_pretrained(cfg["base_model"])
    kw: dict = {"torch_dtype": torch.float16}
    if cfg.get("quant") == "4bit":
        _require("bitsandbytes")
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = transformers.AutoModelForCausalLM.from_pretrained(cfg["base_model"], device_map="auto", **kw)
    if cfg.get("quant") == "4bit":
        model = peft.prepare_model_for_kbit_training(model)
    if cfg["method"] == "prompt_tuning":
        pc = peft.PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=cfg["n_virtual_tokens"])
    else:
        pc = peft.LoraConfig(task_type="CAUSAL_LM", r=cfg["r"], lora_alpha=cfg["alpha"],
                             lora_dropout=cfg["dropout"], target_modules=cfg["target"])
    model = peft.get_peft_model(model, pc)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = _optimizer(torch, transformers, params, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["fp16"] and torch.cuda.is_available())
    dev = next(model.parameters()).device
    letter_ids = {l: tok.encode(l, add_special_tokens=False)[0] for l in LETTERS}
    model.train()
    step, last, accum = 0, float("nan"), cfg["grad_accum"]
    for epoch in range(cfg["epochs"]):
        order = list(range(len(rows)))
        random.Random(cfg["seed"] + epoch).shuffle(order)
        for i, j in enumerate(order):
            r = rows[j]
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=cfg["fp16"]):
                if "target_dist" in r:
                    ids = _chat_ids(tok, r["messages"], add_generation_prompt=True)
                    x = torch.tensor([ids[-cfg["max_len"]:]], device=dev)
                    logits = model(input_ids=x).logits[0, -1]
                    lets = sorted(r["target_dist"])
                    logq = torch.log_softmax(logits[[letter_ids[l] for l in lets]].float(), -1)
                    p = torch.tensor([r["target_dist"][l] for l in lets], device=dev)
                    loss = (p * (torch.log(p.clamp_min(1e-12)) - logq)).sum()     # KL(p_human || p_model)
                else:
                    ids, labels = _tokenize_chat(tok, r["messages"], cfg["max_len"])
                    out = model(input_ids=torch.tensor([ids], device=dev), labels=torch.tensor([labels], device=dev))
                    loss = out.loss
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == len(order):
                for g in opt.param_groups:   # linear warmup
                    g["lr"] = cfg["lr"] * min(1.0, (step + 1) / max(1, cfg["warmup_steps"]))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                step += 1
            last = float(loss.detach())
    model.save_pretrained(out_dir)
    return {"steps": step, "last_loss": last, "out_dir": out_dir}


# ---------------------------------------------------------------- steering helpers (open models)

def _layers(model):
    for path in ("model.layers", "base_model.model.model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("cannot find the decoder layer list on this model")


def steering_vector(model, tok, pos: Sequence[tuple[str, str]], neg: Sequence[tuple[str, str]],
                    layers: Sequence[int] | None = None, mode: str = "last") -> dict:
    """{layer: vector} = mean activation over `pos` minus over `neg`, each item (prompt, completion), read at the
    output of decoder layer l. mode "last": the last completion token (CAA: the answer letter, see caa_pairs);
    "mean": averaged over completion tokens (persona vectors: response tokens). Text is tokenized as given, so
    apply the chat template first for a chat model. In most HF decoders the last layer's hidden state is taken
    after the final norm, so that layer's vector is on a different scale. Lazy torch import. Not measured in
    simengine."""
    torch = _require("torch")
    if mode not in ("last", "mean"):
        raise ValueError("mode must be 'last' or 'mean'")
    dev = next(model.parameters()).device

    def mean_acts(items):
        sums, n = None, 0
        for prompt, completion in items:
            p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
            ids = tok(prompt + completion, add_special_tokens=True)["input_ids"]
            with torch.no_grad():
                hs = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states
            sel = slice(len(ids) - 1, len(ids)) if mode == "last" else slice(min(len(p_ids), len(ids) - 1), len(ids))
            a = torch.stack([h[0, sel].float().mean(0) for h in hs[1:]])     # hs[l+1] = output of layer l
            sums = a if sums is None else sums + a
            n += 1
        if not n:
            raise ValueError("steering_vector needs at least one pos and one neg item")
        return sums / n

    diff = mean_acts(pos) - mean_acts(neg)
    idx = range(diff.shape[0]) if layers is None else layers
    return {int(l): diff[l] for l in idx}


@contextlib.contextmanager
def steer(model, vectors: Mapping[int, Any], coef: float = 1.0, prompt_len: int = 0):
    """Add coef * vectors[l] to the output of decoder layer l while inside the block. In a forward over a full
    sequence only positions >= prompt_len are changed (CAA: after the user's prompt; 0 = every position);
    cached single-token decoding steps are always changed (persona vectors). Lazy torch import."""
    _require("torch")
    layer_list = _layers(model)
    handles = []

    def hook_for(v):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            vec = (coef * v).to(device=h.device, dtype=h.dtype)
            if h.shape[1] > 1 and prompt_len > 0:
                h = h.clone()                       # no in-place edit: keeps autograd valid (preventative steering)
                h[:, prompt_len:] = h[:, prompt_len:] + vec
            else:
                h = h + vec
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    try:
        for l, v in vectors.items():
            handles.append(layer_list[int(l)].register_forward_hook(hook_for(v)))
        yield model
    finally:
        for hd in handles:
            hd.remove()


def steer_from_meta(meta: Mapping, trait_vectors: Mapping[str, Any], layer: int | None = None) -> dict:
    """Combine per-trait vectors with a steering injector's meta['steer']['coefs'] into {layer: vector} for
    `steer(..., coef=1)`. trait_vectors[trait] is one vector (already at its chosen layer)."""
    s = meta.get("steer") if "steer" in meta else meta
    lay = layer if layer is not None else s.get("layer")
    if lay is None:
        raise ValueError("no layer: pass layer= or set layer on the injector (the papers pick it by sweep)")
    missing = [t for t in s["coefs"] if t not in trait_vectors]
    if missing:
        raise KeyError(f"no steering vector for traits {missing}")
    total = None
    for t, c in s["coefs"].items():
        part = trait_vectors[t] * c
        total = part if total is None else total + part
    return {int(lay): total} if total is not None else {}


__all__ = ["FAMILY", "NOT_MEASURED", "DEFAULT_BASE", "option_letters", "parse_finetuned", "SubpopLoRA",
           "CentaurQLoRA", "CAASteer", "PersonaVectorSteer", "SoftPrompt", "chat_rows", "distribution_rows",
           "caa_pairs", "persona_vector_pairs", "judge_filter", "holdout_split", "write_jsonl", "read_jsonl",
           "recipe", "train_colab_t4", "steering_vector", "steer", "steer_from_meta"]

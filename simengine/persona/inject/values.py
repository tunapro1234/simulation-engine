"""Value and culture injectors: Schwartz basic values, Moral Foundations, Hofstede country dimensions and
cultural prompting (with optional Inglehart-Welzel map coordinates).

Every score is taken as input on the person dict; no published country table is shipped (Hofstede Insights and
WVS/IVS scores carry their own terms), so callers bring their own and cite them.

    import simengine.persona.inject as pi
    person = {"id": "u1", "country": "Chile",
              "schwartz": {"benevolence": 5.2, "power": 2.1, "self_direction": 4.8},   # PVQ means, 1-6
              "mfq": {"care": 4.1, "fairness": 3.9, "loyalty": 2.0, "authority": 2.2, "sanctity": 1.5},
              "hofstede": {"pdi": 63, "idv": 23},                                      # 0-100, caller-supplied
              "ivs_map": {"traditional_secular": -0.4, "survival_self_expression": 0.6}}
    q = {"id": "q1", "text": "Is divorce justifiable?", "options": ["never", "sometimes", "always"]}
    pi.get_injector("schwartz_scores")(person, q)              # Kang et al. 2023 roleplay template
    pi.get_injector("schwartz_anchor", k=1)(person, q)         # Rozen et al. 2024 "Answer as a person that is ..."
    pi.get_injector("moral_foundations", style="statements")(person, q)
    pi.get_injector("hofstede_country", scores=True)(person, q)
    pi.get_injector("cultural_prompting", variant="random", seed=3)(person, q)

Person keys read: `schwartz` (dict code -> score, or for schwartz_anchor also a code / list of codes), `mfq`
(dict foundation -> score), `country` (+ optional `birth_country`), `hofstede` (dict dim -> score), `ivs_map`
(dict axis -> coordinate). Keys are case-insensitive; "_" and " " count as "-".
None of these injectors shows the person's own answers.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

from ._base import Injector, MissingPersonField, Prompt, register

# ---------------------------------------------------------------- vocabularies

# Schwartz (1992, 2012) ten basic values, in the order of Kang et al. 2023 (alphabetical); definitions as in
# their Table 23 ("values" + Schwartz's definition).
SCHWARTZ_10: dict[str, str] = {
    "achievement": "values personal success through demonstrating competence according to social standards",
    "benevolence": "values preserving and enhancing the welfare of those with whom one is in frequent personal "
                   "contact (the in-group)",
    "conformity": "values restraint of actions, inclinations and impulses likely to upset or harm others and "
                  "violate social expectations or norms",
    "hedonism": "values pleasure or sensuous gratification for oneself",
    "power": "values social status and prestige, control or dominance over people and resources",
    "security": "values safety, harmony, and stability of society, of relationships, and of self",
    "self-direction": "values independent thought and action: choosing, creating, exploring",
    "stimulation": "values excitement, novelty, and challenge in life",
    "tradition": "values respect, commitment, and acceptance of the customs and ideas that one's culture or "
                 "religion provides",
    "universalism": "values understanding, appreciation, tolerance, and protection for the welfare of all people "
                    "and for nature",
}

# simengine wording, in Rozen et al.'s anchor template, for the 6 basic values that Appendix E splits into
# refined ones; stimulation, hedonism, achievement and tradition keep the paper's own texts
# (simengine.persona.methods.value_anchor.ANCHORS)
BASIC_ANCHORS: dict[str, str] = {
    "self-direction": "thinking and acting independently, choosing your own goals",
    "power": "seeking social status, prestige and control over people and resources",
    "security": "seeking safety, harmony and stability for society, your relationships and yourself",
    "conformity": "holding back from actions that could upset others or break social norms",
    "benevolence": "caring for the welfare of the people close to you",
    "universalism": "seeking understanding, tolerance and protection for the welfare of all people and of nature",
}

# Moral Foundations (Graham et al. 2011 MFQ; liberty from Haidt 2012), simengine definitions
MFQ_FOUNDATIONS: dict[str, str] = {
    "care": "concern for the suffering of others; cherishing and protecting the vulnerable",
    "fairness": "justice, rights and reciprocity; not cheating or being cheated",
    "loyalty": "standing with your group, family and nation; not betraying them",
    "authority": "respect for tradition, legitimate authority and social order",
    "sanctity": "purity; avoiding what is degrading or disgusting, living in an elevated way",
    "liberty": "resentment of domination; not being oppressed or restricted",
}
MFQ_ALIASES = {"harm": "care", "care-harm": "care", "fairness-cheating": "fairness", "ingroup": "loyalty",
               "in-group": "loyalty", "loyalty-betrayal": "loyalty", "authority-subversion": "authority",
               "respect": "authority", "purity": "sanctity", "sanctity-degradation": "sanctity",
               "liberty-oppression": "liberty"}
# Abdulhai et al. 2023 (arXiv:2310.15337) Table 2: the prompt found to maximize each foundation, verbatim.
# The sanctity/purity sentence is the paper's own (it reads like hierarchy, the paper notes it is counterintuitive).
# The paper has no liberty prompt.
MFQ_STATEMENTS: dict[str, str] = {
    "care": "You do not like to cause harm.",
    "fairness": "You believe the rich and poor should be treated with equity.",
    "authority": "You believe in traditional roles.",
    "sanctity": "You believe that some people are more important than others.",
    "loyalty": "You would sacrifice yourself for your country.",
}

# Hofstede 6-D model: (label, low pole, high pole)
HOFSTEDE_DIMS: dict[str, tuple[str, str, str]] = {
    "pdi": ("Power distance", "power should be spread equally", "people accept that power is distributed unequally"),
    "idv": ("Individualism", "people belong to tight in-groups that look after them",
            "people look after themselves and their immediate family"),
    "mas": ("Masculinity", "caring for others and quality of life matter most",
            "competition, achievement and success matter most"),
    "uai": ("Uncertainty avoidance", "ambiguity and the unknown are tolerated",
            "ambiguity and the unknown are threatening and avoided with rules"),
    "lto": ("Long-term orientation", "traditions are kept and change is viewed with suspicion",
            "thrift and adapting to prepare for the future are valued"),
    "ivr": ("Indulgence", "gratification of desires is restrained by strict social norms",
            "enjoying life and having fun is relatively free"),
}
HOFSTEDE_ALIASES = {"power-distance": "pdi", "individualism": "idv", "masculinity": "mas",
                    "uncertainty-avoidance": "uai", "long-term-orientation": "lto", "lto-ws": "lto",
                    "indulgence": "ivr", "motivation": "mas"}

# Tao et al. 2024 (arXiv:2311.14096) Table 1: the ten respondent descriptors (variant 0 is the main one)
CULTURAL_DESCRIPTORS = ("an average human being", "a typical human being", "a human being", "an average person",
                        "a typical person", "a person", "an average individual", "a typical individual",
                        "an individual", "a world citizen")
IVS_AXES = {"traditional-secular": "traditional (negative) to secular-rational (positive)",
            "survival-self-expression": "survival (negative) to self-expression (positive)"}
IVS_ALIASES = {"y": "traditional-secular", "tradsec": "traditional-secular", "x": "survival-self-expression",
               "survself": "survival-self-expression"}


# ---------------------------------------------------------------- helpers

def _key(k: Any) -> str:
    return str(k).strip().lower().replace("_", "-").replace(" ", "-")


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.2f}".rstrip("0")


def _hash(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def _scores(inj: Injector, person: dict, field: str, known: Sequence[str], aliases: dict | None = None,
            scale: tuple[float, float] | None = None) -> dict[str, float]:
    """person[field] as {canonical code: float} in `known` order; None values are dropped. Unknown codes,
    non-numbers and out-of-scale values raise ValueError; nothing usable raises MissingPersonField."""
    raw = person.get(field)
    pid = person.get("id")
    if not isinstance(raw, dict):
        raise ValueError(f"injector {inj.name!r}: person {pid!r} field {field!r} must be a dict of scores, "
                         f"got {type(raw).__name__}")
    out: dict[str, float] = {}
    for k, v in raw.items():
        if v is None:
            continue
        code = _key(k)
        code = (aliases or {}).get(code, code)
        if code not in known:
            raise ValueError(f"injector {inj.name!r}: unknown {field} key {k!r} (person {pid!r}); "
                             f"use {list(known)}")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v):
            raise ValueError(f"injector {inj.name!r}: {field}[{k!r}] must be a number, got {v!r} (person {pid!r})")
        if scale is not None and not scale[0] <= v <= scale[1]:
            raise ValueError(f"injector {inj.name!r}: {field}[{k!r}]={v} outside scale {tuple(scale)} "
                             f"(person {pid!r})")
        out[code] = float(v)
    if not out:
        raise MissingPersonField(f"injector {inj.name!r} needs person field(s) {list(inj.needs)} with at least "
                                 f"one score (person {pid!r})")
    return {c: out[c] for c in known if c in out}


def _ranked(scores: dict[str, float], seed: Any, pid: Any) -> list[str]:
    """Codes by score, highest first; ties broken by a hash of (seed, person id, code), so reruns agree."""
    return sorted(scores, key=lambda c: (-scores[c], _hash(seed, pid, c)))


def _ordered(scores: dict[str, float], order: str, seed: Any, pid: Any) -> list[str]:
    if order == "paper":
        return list(scores)
    if order == "rank":
        return _ranked(scores, seed, pid)
    return sorted(scores, key=lambda c: _hash(seed, pid, "shuffle", c))          # "shuffle"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + (", and " if len(items) > 2 else " and ") + items[-1]


def _choice_tail(question: dict) -> str:
    return " Please choose only one option, even if you're not sure." if question.get("options") else ""


_ORDERS = ("paper", "rank", "shuffle")


def _check_scale(scale: Sequence[float]) -> tuple[float, float]:
    lo, hi = (float(x) for x in scale)
    if not lo < hi:
        raise ValueError(f"scale must be (low, high) with low < high, got {tuple(scale)}")
    return lo, hi


# ---------------------------------------------------------------- injectors

@register
class SchwartzScores(Injector):
    """Kang et al. 2023 prompt baseline (LLaMA/ChatGPT_Long; Short with definitions=False): a roleplay system
    prompt that lists the ten Schwartz values with the person's score on each."""
    name = "schwartz_scores"
    family = "values"
    paper = "Kang et al. 2023, From Values to Opinions (arXiv:2310.17857), Tables 22-23 prompt baselines"
    needs = ("schwartz",)
    description = ("System: 'Let's roleplay' + optional Schwartz value definitions + '- Value: score' lines "
                   "(PVQ 1-6 by default) + 'As this person, please answer'.")

    def __init__(self, definitions: bool = True, scale: Sequence[float] = (1, 6), order: str = "paper",
                 seed: int = 0, **params: Any):
        super().__init__(definitions=definitions, scale=tuple(scale), order=order, seed=seed, **params)
        if order not in _ORDERS:
            raise ValueError(f"order must be one of {_ORDERS}, not {order!r}")
        self.definitions, self.scale, self.order, self.seed = bool(definitions), _check_scale(scale), order, seed

    def render(self, person: dict, question: dict) -> Prompt:
        s = _scores(self, person, "schwartz", list(SCHWARTZ_10), scale=self.scale)
        lo, hi = (_fmt(x) for x in self.scale)
        parts = [f"Let's roleplay. I will describe a person who has values between {lo} and {hi} for each value. "
                 f"{lo} means that the value is not important to the person and {hi} means that the value is very "
                 f"important to the person."]
        if self.definitions:
            parts.append("Value definitions:\n" + "\n".join(f"- {c.title()}: {SCHWARTZ_10[c]}" for c in s))
        codes = _ordered(s, self.order, self.seed, person.get("id"))
        parts.append("The person's values:\n" + "\n".join(f"- {c.title()}: {_fmt(s[c])}" for c in codes))
        parts.append("As this person, please answer the following question." + _choice_tail(question))
        return Prompt(system="\n\n".join(parts), user=self.question_block(question),
                      meta={"values": {c: s[c] for c in codes}})


@register
class SchwartzAnchor(Injector):
    """Rozen et al. 2024 value anchor: "Answer as a person that is [value]." with the person's top-k values
    (19 refined values with the paper's Appendix E texts, or the 10 basic values in simengine wording)."""
    name = "schwartz_anchor"
    family = "values"
    paper = "Rozen et al. 2024, Do LLMs have Consistent Values? (arXiv:2407.12878), value anchor prompt"
    needs = ("schwartz",)
    description = ("System: 'Answer as a person that is <top value>.' from the person's highest Schwartz score(s), "
                   "or from a given value code / list of codes; ties broken by seed.")

    def __init__(self, k: int = 1, seed: int = 0, **params: Any):
        super().__init__(k=k, seed=seed, **params)
        if int(k) < 1:
            raise ValueError(f"k must be >= 1, got {k!r}")
        self.k, self.seed = int(k), seed

    @staticmethod
    def phrases() -> dict[str, str]:
        from ..methods.value_anchor import ANCHORS     # lazy: the method zoo is not needed to import this family
        return {**BASIC_ANCHORS, **ANCHORS}

    def render(self, person: dict, question: dict) -> Prompt:
        phrases = self.phrases()
        raw = person.get("schwartz")
        if isinstance(raw, dict):
            codes = _ranked(_scores(self, person, "schwartz", list(phrases)), self.seed, person.get("id"))
        else:
            if not isinstance(raw, (str, list, tuple)):
                raise ValueError(f"injector {self.name!r}: person {person.get('id')!r} field 'schwartz' must be a "
                                 f"dict of scores, a value code or a list of codes, got {type(raw).__name__}")
            codes = [_key(c) for c in ([raw] if isinstance(raw, str) else raw)]
            bad = [c for c in codes if c not in phrases]
            if bad:
                raise ValueError(f"injector {self.name!r}: unknown Schwartz value code(s) {bad} "
                                 f"(person {person.get('id')!r}); use {list(phrases)}")
        top = codes[:self.k]
        system = f"Answer as a person that is {_join([phrases[c] for c in top])}."
        return Prompt(system=system, user=self.question_block(question), meta={"anchors": top})


@register
class MoralFoundations(Injector):
    """Moral Foundations profile. style="statements": Abdulhai et al.'s foundation-maximizing sentence for the
    person's top-k foundations (verbatim; none exists for liberty). style="scores": every MFQ relevance score in
    the Kang et al. roleplay template (simengine adaptation of that template to MFT)."""
    name = "moral_foundations"
    family = "values"
    paper = ("Abdulhai et al. 2023, Moral Foundations of Large Language Models (arXiv:2310.15337), Table 2 "
             "prompts; scores style adapts Kang et al. 2023 (arXiv:2310.17857); MFQ: Graham et al. 2011")
    needs = ("mfq",)
    description = ("style=statements: top-k foundation sentences from Abdulhai et al.; style=scores: '- Foundation: "
                   "score' lines on the MFQ 0-5 relevance scale with optional definitions.")

    def __init__(self, style: str = "statements", k: int = 1, definitions: bool = True,
                 scale: Sequence[float] = (0, 5), seed: int = 0, **params: Any):
        super().__init__(style=style, k=k, definitions=definitions, scale=tuple(scale), seed=seed, **params)
        if style not in ("statements", "scores"):
            raise ValueError(f"style must be 'statements' or 'scores', not {style!r}")
        if int(k) < 1:
            raise ValueError(f"k must be >= 1, got {k!r}")
        self.style, self.k, self.definitions, self.seed = style, int(k), bool(definitions), seed
        self.scale = _check_scale(scale)

    def render(self, person: dict, question: dict) -> Prompt:
        s = _scores(self, person, "mfq", list(MFQ_FOUNDATIONS), MFQ_ALIASES, self.scale)
        if self.style == "statements":
            top = [c for c in _ranked(s, self.seed, person.get("id")) if c in MFQ_STATEMENTS][:self.k]
            if not top:
                raise ValueError(f"injector {self.name!r}: style='statements' has no sentence for {list(s)} "
                                 f"(person {person.get('id')!r}); use style='scores'")
            return Prompt(system=" ".join(MFQ_STATEMENTS[c] for c in top), user=self.question_block(question),
                          meta={"foundations": top})
        lo, hi = (_fmt(x) for x in self.scale)
        parts = [f"Let's roleplay. I will describe a person by how relevant each moral foundation is to their "
                 f"judgements of right and wrong, from {lo} (not at all relevant) to {hi} (extremely relevant)."]
        if self.definitions:
            parts.append("Foundation definitions:\n" + "\n".join(f"- {c.title()}: {MFQ_FOUNDATIONS[c]}" for c in s))
        parts.append("The person's moral foundations:\n" + "\n".join(f"- {c.title()}: {_fmt(v)}" for c, v in s.items()))
        parts.append("As this person, please answer the following question." + _choice_tail(question))
        return Prompt(system="\n\n".join(parts), user=self.question_block(question), meta={"foundations": list(s)})


@register
class HofstedeCountry(Injector):
    """Kharchenko et al. 2024 country persona ("imagining that you are proudly and thoroughly from the country
    of ..."); scores=True adds the country's Hofstede dimension scores from person['hofstede'] with each
    dimension's poles (simengine extension: the paper names the country only)."""
    name = "hofstede_country"
    family = "values"
    paper = ("Kharchenko et al. 2024, How Well Do LLMs Represent Values Across Cultures? (arXiv:2406.14805), "
             "persona prompt; dimensions: Hofstede 2010 (scores supplied by the caller)")
    needs = ("country",)
    description = ("System: 'Answer ... imagining that you are proudly and thoroughly from the country of <country>' "
                   "(+ ', providing appropriate advice' with advice=True); scores=True adds caller-supplied "
                   "Hofstede scores (needs 'hofstede').")

    def __init__(self, scores: bool = False, advice: bool = False, **params: Any):
        super().__init__(scores=scores, advice=advice, **params)
        self.scores, self.advice = bool(scores), bool(advice)
        self.needs = ("country", "hofstede") if self.scores else ("country",)

    def render(self, person: dict, question: dict) -> Prompt:
        tail = ", providing appropriate advice." if self.advice else "."
        system = ("Answer the following prompt by imagining that you are proudly and thoroughly from the country of "
                  f"{person['country']}{tail}")
        meta: dict = {"country": person["country"]}
        if self.scores:
            s = _scores(self, person, "hofstede", list(HOFSTEDE_DIMS), HOFSTEDE_ALIASES, (0, 120))
            lines = [f"- {HOFSTEDE_DIMS[d][0]}: {_fmt(v)} (low: {HOFSTEDE_DIMS[d][1]}; high: {HOFSTEDE_DIMS[d][2]})"
                     for d, v in s.items()]
            system += ("\n\nYour country's scores on Hofstede's cultural dimensions (roughly 0 to 100):\n"
                       + "\n".join(lines))
            meta["hofstede"] = s
        return Prompt(system=system, user=self.question_block(question), meta=meta)


@register
class CulturalPrompting(Injector):
    """Tao et al. 2024 cultural prompting: system "You are an average human being born in X and living in X
    responding to the following survey question." (variant 0-9 = their ten descriptors, "random" = one per
    person by seed). map=True adds the Inglehart-Welzel map coordinates from person['ivs_map'] (simengine
    extension; the paper used the map only to score replies)."""
    name = "cultural_prompting"
    family = "values"
    paper = ("Tao et al. 2024, Cultural bias and cultural alignment of large language models (arXiv:2311.14096), "
             "Table 1 descriptors; map: Inglehart & Welzel 2005 (coordinates supplied by the caller)")
    needs = ("country",)
    description = ("System: 'You are <descriptor> born in <birth_country|country> and living in <country> responding "
                   "to the following survey question.'; map=True adds caller-supplied Inglehart-Welzel "
                   "coordinates (needs 'ivs_map').")

    def __init__(self, variant: int | str = 0, seed: int = 0, map: bool = False, **params: Any):  # noqa: A002
        super().__init__(variant=variant, seed=seed, map=map, **params)
        if variant != "random" and not (isinstance(variant, int) and 0 <= variant < len(CULTURAL_DESCRIPTORS)):
            raise ValueError(f"variant must be 0-{len(CULTURAL_DESCRIPTORS) - 1} or 'random', not {variant!r}")
        self.variant, self.seed, self.map = variant, seed, bool(map)
        self.needs = ("country", "ivs_map") if self.map else ("country",)

    def render(self, person: dict, question: dict) -> Prompt:
        v = (_hash(self.seed, person.get("id"), "variant") % len(CULTURAL_DESCRIPTORS)
             if self.variant == "random" else self.variant)
        country = person["country"]
        born = person.get("birth_country") or country
        system = (f"You are {CULTURAL_DESCRIPTORS[v]} born in {born} and living in {country} "
                  "responding to the following survey question.")
        meta: dict = {"country": country, "variant": v}
        if self.map:
            s = _scores(self, person, "ivs_map", list(IVS_AXES), IVS_ALIASES)
            lines = [f"- {IVS_AXES[a]}: {_fmt(x)}" for a, x in s.items()]
            system += ("\n\nOn the Inglehart-Welzel cultural map, the society you live in "
                       "scores:\n" + "\n".join(lines) + "\nTraditional values stress religion, family ties, deference "
                       "to authority and national pride; secular-rational values weigh these less. Survival values "
                       "stress economic and physical security; self-expression values stress tolerance, trust, "
                       "well-being and political participation.")
            meta["ivs_map"] = s
        return Prompt(system=system, user=self.question_block(question), meta=meta)


__all__ = ["SchwartzScores", "SchwartzAnchor", "MoralFoundations", "HofstedeCountry", "CulturalPrompting",
           "SCHWARTZ_10", "BASIC_ANCHORS", "MFQ_FOUNDATIONS", "MFQ_STATEMENTS", "HOFSTEDE_DIMS",
           "CULTURAL_DESCRIPTORS", "IVS_AXES"]

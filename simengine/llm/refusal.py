"""Conservative detector for *soft* refusals: plain-text replies with stop_reason "end_turn" in which
the model declines the task ("I can't produce this as a survey dataset...") instead of answering.

The API's hard refusals (stop_reason == "refusal") raise RefusalError; soft ones do not, and in the
D2 rehearsal 59 of 150 Haiku 5.5 survey batches came back like that (SIMENGINE-EKSIK.md #6).

Design (false negatives are preferred over false positives):
- Only the opening of the reply is read (the first REFUSAL_HEAD characters); a phrase must start
  within the first REFUSAL_START characters. A long answer that mentions "I can't ..." later is not a refusal.
- A reply that starts with JSON ('{' or '[', or a ```json fence) is an answer, never a refusal.
- Quoted spans ("...", '...', curly quotes) are removed first, so a quoted survey option such as
  "I can't afford it" does not count.
- English: first-person negation + a task verb ("I can't produce this", "I cannot create responses",
  "I'm not able to role-play", "I won't fabricate", "I'm not comfortable generating fake responses"); verbs
  that also occur in ordinary answers (produce, create, provide, write, complete, fill out, ...) only count with
  a task object: a task noun (responses, data, survey, request, ...), a demonstrative followed by a task noun
  ("this survey"), or a bare demonstrative at the end of the clause ("I can't produce this."), so "I will not
  complete this purchase" and "I can't fill out these forms every week" are answers. "pretend" needs a person
  object ("pretend to be these people"; "I won't pretend I love it" is an answer). Also: "I must decline",
  "I'm going to decline to role-play ...", "people who don't exist", "invent answers for ...", an opening
  "As an AI", "I don't think it's appropriate for me to fabricate ...", "Rather than fabricate data, ...",
  and an apology + "I can't do that." at the end of the clause.
- Turkish: a refusal verb (yapamam, üretemem, uyduramam, ...) only counts with a refusal context:
  an AI/fiction/request word ("yapay zeka", "var olmayan", "kurgusal", "bu isteği"); a demonstrative + task
  noun right before the verb ("bu yanıtları üretemem"); or a demonstrative pronoun right before the verb
  ("bunu", "bunları", "bu konuda") with a fabrication verb or an apology ("maalesef bunu yapamam").
  "Maalesef bu fiyata alışveriş yapamam" and "Bu tür bir ürünü üretemem ama satın alırım" (persona answers)
  are not refusals.
"""
from __future__ import annotations

import re

REFUSAL_HEAD = 400    # characters of the reply that are inspected
REFUSAL_START = 200   # a refusal phrase must start before this offset (after quote removal)

_A = r"['’]"
_NEG = (rf"(?:can(?:no|{_A})t|can not|cannot|won{_A}t|will not|shouldn{_A}t|"
        rf"(?:am|{_A}m)\s+(?:not\s+able|unable|not\s+going)\s+to|"
        rf"(?:am|{_A}m)\s+not\s+(?:really\s+)?comfortable(?:\s+with)?|"
        rf"(?:do\s+not|don{_A}t)\s+(?:feel\s+comfortable(?:\s+with)?|think\s+I\s+(?:can|should))|"
        rf"(?:must|have\s+to)\s+decline\s+to|(?:would\s+not|wouldn{_A}t)\s+(?:feel\s+comfortable(?:\s+with)?|be\s+able\s+to))")
# nouns that make "this / that / these ..." a task object ("complete this survey", not "complete this purchase")
_TASK_NOUN = (r"(?:request|task|survey|questionnaire|exercise|simulation|role[- ]?play|data(?:set)?|responses?|answers?|"
              r"content|information|opinions?|views|personas?|profiles?|people|individuals|respondents|kind|type|sort)\b")
# a bare demonstrative only counts at the end of the clause ("I can't produce this.", "... this as a dataset")
_DEMO_END = r"(?=\s*(?:[.,;:!?)]|$|\s+(?:for\s+you|as\s|because|since|without|but\b|[-–—])))"
_OBJ = (rf"(?:(?:this|that|these|those|such)(?:\s+(?:kind|type|sort)\s+of)?(?:\s+[\w-]+)?\s+{_TASK_NOUN}|"
        rf"(?:this|that|these|those)\b{_DEMO_END}|"
        r"(?:fake|fabricated|fictional|fictitious|synthetic|simulated|made[- ]up|invented|"
        r"hypothetical|survey|responses?|answers?|data(?:set)?|opinions?|views|personal|individual|"
        r"an?\s+(?:survey|dataset|response|answer|persona|fictional|fake|realistic|simulated|synthetic))\b)")
_VERB_OBJ = (r"(?:produc(?:e|ing)|creat(?:e|ing)|generat(?:e|ing)|provid(?:e|ing)|writ(?:e|ing)|complet(?:e|ing)|"
             r"fill(?:ing)?\s+(?:in|out)|help(?:ing)?\s+with|assist(?:ing)?\s+with)")
# "pretend" only with a person object: "I won't pretend I love it" is an answer, "pretend to be these people" is not
_PRETEND = (r"pretend(?:ing)?\s+(?:to\s+be|that\s+I\s*(?:am|['’]m)|I\s*(?:am|['’]m)|(?:these|those|other|real)\s+|"
            r"(?:someone|people|individuals|a\s+(?:person|real|specific)))")
_VERB_ALONE = (r"(?:fabricat(?:e|ing)|invent(?:ing)?|mak(?:e|ing)\s+up(?!\s+(?:my|our|his|her|their|one{_A}s)\s+mind)|"
               r"simulat(?:e|ing)|role[- ]?play(?:ing)?|impersonat(?:e|ing)|" + _PRETEND + r"|comply|"
               r"speak\s+(?:for|on\s+behalf)|"
               r"answer\s+(?:as\s+(?:if|though|a|an|someone|this)|on\s+behalf|for\s+(?:people|individuals|someone|real)))"
               ).replace("{_A}", _A)
_ADV = r"(?:\s+(?:really|honestly|simply|just|actually|unfortunately))?"
_EN_FIRST_PERSON = re.compile(
    rf"\bI{_ADV}\s*{_NEG}\s+(?:[\w'’-]+\s+){{0,3}}?(?:{_VERB_ALONE}|{_VERB_OBJ}\s+(?:[\w'’-]+\s+){{0,1}}?{_OBJ})",
    re.I)
# "I decline the offer" is a valid game move, so a bare "I decline" does not count
_DECLINE_VERBS = (r"(?:produce|create|generate|fabricate|invent|simulate|role[- ]?play|pretend|impersonate|fill|"
                  r"complete|provide|answer|speak|take\s+on)\b")
_EN_DECLINE = re.compile(
    r"\bI\s+(?:(?:must|have\s+to|need\s+to|will\s+have\s+to)\s+(?:respectfully\s+|politely\s+)?decline\b|"
    r"(?:respectfully\s+|politely\s+)?decline\s+(?:to\s+" + _DECLINE_VERBS + r"|(?:this|that|your)\s+(?:request|task)\b))|"
    r"\bI(?:\s*['’]m\s+going\s+to|\s+am\s+going\s+to|\s+will|\s*['’]ll|\s+would\s+like\s+to|\s*['’]d\s+like\s+to|"
    r"\s*['’]d\s+rather|\s+would\s+rather)\s+(?:respectfully\s+|politely\s+)?decline\s+(?:to\s+" + _DECLINE_VERBS +
    r"|(?:this|that|your)\s+(?:request|task)\b)", re.I)
_FAB_ING = r"(?:fabricat|invent|simulat|role[- ]?play|impersonat|pretend|mak(?:e|ing)\s+up|generat|creat|produc)\w*"
_EN_PHRASES = re.compile(
    r"(?:\bpeople\s+who\s+(?:don['’]t|do\s+not)\s+(?:actually\s+|really\s+)?exist\b|"
    r"\b(?:invent|fabricat|mak(?:e|ing)\s+up)\w*\s+(?:[\w'’-]+\s+){0,3}?(?:answers|responses|data|opinions)\s+for\b|"
    r"^\W*as\s+an\s+ai(?:\s+(?:language\s+model|assistant|model))?\s*,|"
    r"\bI\s*(?:am|['’]m)\s+an?\s+ai\s+(?:language\s+model|assistant|model)\b|"
    # "I don't think it's appropriate for me to fabricate ...", "it wouldn't be appropriate to simulate ..."
    r"\b(?:I\s+(?:do\s+not|don['’]t)\s+think\s+it(?:['’]s|\s+is|\s+would\s+be)|it\s+(?:would\s+not|wouldn['’]t)\s+be|"
    r"it\s+(?:is\s+not|isn['’]t)|it['’]s\s+not)\s+(?:appropriate|ethical|right)\s+(?:for\s+me\s+)?to\s+"
    + _FAB_ING + r"|"
    # "Rather than fabricate data, I'd suggest ..."
    r"\b(?:rather|instead)\s+(?:than|of)\s+" + _FAB_ING + r"\s+(?:[\w'’-]+\s+){0,2}?"
    r"(?:data|answers|responses|opinions|individual|people|personas?|survey)\b)",
    re.I)
# "Sorry, but I can't do that." (an apology + a bare 'do this / that' at the end of the clause)
_EN_DO_THAT = re.compile(
    rf"\b(?:sorry|I\s+apologi[sz]e|unfortunately)\b[^.?!\n]{{0,40}}?\bI{_ADV}\s*{_NEG}\s+(?:help\s+with|do)\s+"
    r"(?:this|that)\s*(?:[.!]|$)", re.I)

_TR_FAB = r"(?:üretemem|uyduramam|oluşturamam)"
_TR_HELP = r"(?:yardımcı\s+olamam|yerine\s+getiremem)"
_TR_VERB = (rf"(?:{_TR_FAB}|{_TR_HELP}|yapamam|veremem|sağlayamam|cevaplayamam|yanıtlayamam|canlandıramam|"
            r"taklit\s+edemem)")
_TR_VERB_RE = re.compile(rf"(?<!\w){_TR_VERB}(?!\w)", re.I)
# AI / fiction context: enough on its own next to a refusal verb
_TR_AI = re.compile(r"(?<!\w)(?:yapay\s+zek[aâ]|dil\s+modeli|gerçek\s+olmayan|var\s+olmayan|kurgusal|"
                    r"bu\s+(?:isteği|isteğinizi|talebi|talebinizi|görevi))(?!\w)", re.I)
_TR_APOLOGY = re.compile(r"(?<!\w)(?:maalesef|üzgünüm|özür\s+dilerim|kusura\s+bakma\w*)(?!\w)", re.I)
# a demonstrative PRONOUN right before the verb ("bunu yapamam"), not an adjective ("bu fiyata alışveriş yapamam")
_TR_PRON = r"(?:bunu|bunları|böyle\s+bir\s+şey|bu\s+konuda|bu\s+şekilde)"
# task nouns that make "bu / bu tür ... <verb>" a refusal ("bu yanıtları üretemem", not "bu tür bir ürünü üretemem")
_TR_TASK = (r"(?:yanıt|cevap|veri|görüş|fikir|kişi|insan|profil|anket|içerik|senaryo|persona|rol|istek|talep|"
            r"simülasyon)\w*")
_TR_DEMO = r"(?:bunları|bunu|bu|böyle|bu\s+tür|bu\s+şekilde|bu\s+konuda|şu)"

_QUOTED = re.compile(r"\"[^\"\n]{0,300}\"|“[^”\n]{0,300}”|‘[^’\n]{0,300}’|«[^»\n]{0,300}»|„[^“”\n]{0,300}[“”]|"
                     r"(?<![\w'])'[^'\n]{1,300}'(?![\w])")
_LEAD = re.compile(r"^[\s>*#_`~-]+")


def _clean_head(text: str) -> str:
    head = _LEAD.sub("", text[:REFUSAL_HEAD])
    return _QUOTED.sub(" ", head)


def looks_like_refusal(text: str | None) -> bool:
    """True when the opening of `text` reads like a soft refusal (see the module docstring).
    Conservative on purpose: it is meant to keep refusals out of the cache and count them, not to
    judge answers. Empty text and JSON replies return False."""
    if not text:
        return False
    s = text.lstrip()
    if not s or s[0] in "{[" or s.lower().startswith("```json"):
        return False
    head = _clean_head(s)
    for rx in (_EN_FIRST_PERSON, _EN_DECLINE, _EN_PHRASES, _EN_DO_THAT):
        m = rx.search(head)
        if m and m.start() < REFUSAL_START:
            return True
    m = _TR_VERB_RE.search(head)
    if m and m.start() < REFUSAL_START:
        verb = m.group(0).lower()
        if _TR_AI.search(head[:REFUSAL_START + 100]):
            return True
        before = head[max(0, m.start() - 60):m.start()]
        pron = re.search(rf"(?<!\w){_TR_PRON}\s+$", before, re.I) is not None
        task = re.search(rf"(?<!\w){_TR_DEMO}\s+(?:\w+\s+){{0,2}}?{_TR_TASK}\s+(?:\w+\s+){{0,1}}$", before,
                         re.I) is not None
        apology = _TR_APOLOGY.search(head[:60]) is not None
        if task:
            return True          # "bu yanıtları üretemem", "bu tür cevaplar veremem"
        if pron and (re.fullmatch(_TR_FAB, verb) or apology):
            return True          # "bunu üretemem", "maalesef bunu yapamam"
        if apology and re.fullmatch(_TR_HELP, verb, re.I):
            return True          # "maalesef yardımcı olamam"
    return False


__all__ = ["looks_like_refusal", "REFUSAL_HEAD", "REFUSAL_START"]

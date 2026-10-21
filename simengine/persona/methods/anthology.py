"""Anthology narrative backstories (edsl-test personas/anthology; Moon et al. 2024, arXiv:2407.06576).

Persona = a first-person life story in the paper's "Question: ... Answer: ..." form plus one bridging line for
instruction-tuned models. The stories are the paper's released anthology (HF SuhongMoon/anthology_backstory).
Population targeting is approximated by AGE-stratified sampling: draw an age bracket from GSS 2024 weighted
shares, then a story from that bracket without replacement (refilled when exhausted). The paper's full
estimate-then-match on 5 demographics is not implemented.

Data (not shipped): `<root>/persona_methods/anthology_backstories.json`, a list of
{"bracket": "18-29"|"30-49"|"50-64"|"65+", "question": str, "text": str, ...}.
"""
from __future__ import annotations

import json

from ._base import FunctionMethod, cached_load, data_path
from .helpers import stratified_pool_sample

DATA_FILE = "anthology_backstories.json"

# GSS 2024 age-bracket shares weighted by wtssps (edsl-test personas/anthology/__init__.py, n = 3,208)
GSS_AGE_WEIGHTS = {"18-29": 0.201, "30-49": 0.338, "50-64": 0.237, "65+": 0.224}

TEMPLATE = (
    "Before this study you answered an interview question.\n\n"
    "Question: {question}\n\n"
    "Answer: {text}\n\n"
    "You are the person who gave this answer. In everything that follows, "
    "stay in character and decide exactly as this person would."
)


def _read(path):
    with open(path, encoding="utf-8") as f:
        stories = json.load(f)
    if isinstance(stories, dict):
        stories = stories.get("stories") or stories.get("backstories") or []
    for i, s in enumerate(stories):
        if not isinstance(s, dict) or not {"bracket", "question", "text"} <= set(s):
            raise ValueError(f"{path}: story {i} needs bracket, question and text")
    return stories


def load_stories(data_root=None):
    return cached_load(data_path(DATA_FILE, data_root), _read)


def _draw(n, seed=None, data_root=None):
    stories = load_stories(data_root)
    picks = stratified_pool_sample(stories, GSS_AGE_WEIGHTS, n, seed, key="bracket")
    out = []
    for s in picks:
        params = {"bracket": s["bracket"]}
        for k in ("sid", "id", "age"):
            if k in s:
                params[k] = s[k]
        out.append((TEMPLATE.format(question=s["question"], text=s["text"]), params))
    return out


METHOD = FunctionMethod(
    name="anthology", draw_fn=_draw, needs_data=True, data_file=DATA_FILE,
    description="Released Anthology backstory, age-stratified to GSS 2024 shares.",
    paper="Moon et al. 2024, Virtual Personas for Language Models via an Anthology of Backstories (arXiv:2407.06576)",
    source="edsl-test personas/anthology/__init__.py")

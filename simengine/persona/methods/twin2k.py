"""Real-person persona summaries from Twin-2K-500 (edsl-test personas/twin2k; Toubia et al. 2025,
arXiv:2505.17479).

Each persona is the verbatim `persona_summary` of one real US respondent (demographics, ~40 scale scores with
percentiles, the person's own open-ended answers), framed like the authors' simulation pipeline ("Persona
Profile" header + respond-as-this-person footer). The panel is Prolific, quota-sampled on age, sex and
ethnicity only (not a probability sample), with wave attrition 2,509 -> 2,058.

sample(n) is PREFIX-STABLE: the whole pool is shuffled once by seed and the first n are taken, so
sample(k)[:j] == sample(j). For n > pool size it appends draws with replacement.
Twin-2K-500 is public; models may have seen it (memorization not tested).

Data (not shipped): `<root>/persona_methods/twin2k_personas_sample.json`, {"personas": [{"pid": str,
"summary": str}, ...]} (the source's 500-person subsample; any size works).
"""
from __future__ import annotations

import json

from ._base import FunctionMethod, cached_load, data_path
from .helpers import as_rng

DATA_FILE = "twin2k_personas_sample.json"

HEADER = "## Persona Profile (one real person's survey responses, Twin-2K-500 study)\n"
FOOTER = (
    "\n## End of profile\n"
    "You are the person described above. Answer every question as this person "
    "would, staying consistent with their survey answers and stated "
    "characteristics."
)


def _read(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    pool = d["personas"] if isinstance(d, dict) else d
    for i, p in enumerate(pool):
        if not isinstance(p, dict) or "summary" not in p:
            raise ValueError(f"{path}: persona {i} needs a 'summary'")
    return pool


def load_pool(data_root=None):
    return cached_load(data_path(DATA_FILE, data_root), _read)


def render(summary):
    return HEADER + summary.strip() + FOOTER


def _draw(n, seed=None, data_root=None):
    rng = as_rng(seed)
    pool = list(load_pool(data_root))
    rng.shuffle(pool)
    picks = pool[:n] if n <= len(pool) else pool + rng.choices(pool, k=n - len(pool))
    return [(render(p["summary"]), {"pid": str(p.get("pid", ""))}) for p in picks]


METHOD = FunctionMethod(
    name="twin2k", draw_fn=_draw, needs_data=True, data_file=DATA_FILE,
    description="Verbatim Twin-2K-500 persona summary of one real respondent (prefix-stable draw).",
    paper="Toubia et al. 2025, Twin-2K-500 (arXiv:2505.17479)",
    source="edsl-test personas/twin2k/__init__.py")

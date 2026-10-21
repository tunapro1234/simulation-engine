"""Argyle silicon sampling from real GSS 2024 rows (edsl-test personas/demographic).

Each persona is a first-person backstory built from ONE real respondent row. Fragment style follows Argyle et
al. (2023), Political Analysis, Fig. 1 ("Racially, I am white. I am male. Financially, I am upper-class.") and
App. C.1: answers become short first-person declarations, age is inserted directly, and a missing answer drops
its fragment. Because every backstory is a real row, the joint distribution of traits is the real one.
Rows are drawn with the GSS survey weight (WTSSNRPS in the source preparation) without repeating a respondent
while unused rows remain. A two-line role frame wraps the fragments for chat models (source adaptation).

Data (not shipped): `<root>/persona_methods/gss_sample.csv`, root = data_root or $SIMENGINE_DATA. Columns:
id, age, sex, race, hispanic, born, degree, marital, childs, wrkstat, income16, class, relig, attend,
partyid, polviews, region, weight (GSS 2024 lower-case category labels; empty = missing). The source's
prepare.py builds it from the NORC GSS 2024 public-use file.
"""
from __future__ import annotations

import csv

from ._base import FunctionMethod, cached_load, data_path
from .helpers import RowFragmentRenderer, weighted_indices_no_replacement

DATA_FILE = "gss_sample.csv"
WEIGHT_COL = "weight"
COLUMNS = ("id", "age", "sex", "race", "hispanic", "born", "degree", "marital", "childs", "wrkstat",
           "income16", "class", "relig", "attend", "partyid", "polviews", "region", "weight")

RACE = {"white": "Racially, I am white.", "black": "Racially, I am Black.",
        "other": "Racially, I am of another race."}
SEX = {"male": "I am male.", "female": "I am female."}
REGION = {
    "northeast": "I live in the Northeast of the United States.",
    "midwest": "I live in the Midwest of the United States.",
    "south": "I live in the South of the United States.",
    "west": "I live in the West of the United States.",
}
MARITAL = {"married": "I am married.", "widowed": "I am widowed.", "divorced": "I am divorced.",
           "separated": "I am separated.", "never married": "I have never been married."}
DEGREE = {
    "less than high school": "I did not finish high school.",
    "high school": "I have a high school diploma.",
    "associate/junior college": "I have an associate degree.",
    "bachelor's": "I have a bachelor's degree.",
    "graduate": "I have a graduate degree.",
}
WRKSTAT = {
    "working full time": "I work full time.",
    "working part time": "I work part time.",
    "with a job, but not at work because of temporary illness, vacation, strike":
        "I have a job but am temporarily not at work.",
    "unemployed, laid off, looking for work": "I am unemployed and looking for work.",
    "retired": "I am retired.",
    "in school": "I am in school.",
    "keeping house": "I keep house.",
}
CLASS = {
    "lower class": "Financially, I belong to the lower class.",
    "working class": "Financially, I belong to the working class.",
    "middle class": "Financially, I belong to the middle class.",
    "upper class": "Financially, I belong to the upper class.",
}
RELIG = {
    "protestant": "Religiously, I am Protestant.",
    "catholic": "Religiously, I am Catholic.",
    "christian": "Religiously, I am Christian.",
    "orthodox-christian": "Religiously, I am Orthodox Christian.",
    "inter-nondenominational": "Religiously, I belong to an interdenominational church.",
    "jewish": "Religiously, I am Jewish.",
    "muslim/islam": "Religiously, I am Muslim.",
    "buddhism": "Religiously, I am Buddhist.",
    "hinduism": "Religiously, I am Hindu.",
    "other eastern religions": "Religiously, I follow an Eastern religion.",
    "native american": "Religiously, I follow a Native American religion.",
    "other": "Religiously, I belong to another religion.",
    "none": "I have no religion.",
}
ATTEND = {
    "never": "I never attend religious services.",
    "less than once a year": "I attend religious services less than once a year.",
    "about once or twice a year": "I attend religious services about once or twice a year.",
    "several times a year": "I attend religious services several times a year.",
    "about once a month": "I attend religious services about once a month.",
    "2-3 times a month": "I attend religious services two or three times a month.",
    "nearly every week": "I attend religious services nearly every week.",
    "every week": "I attend religious services every week.",
    "several times a week": "I attend religious services several times a week.",
}
PARTYID = {
    "strong democrat": "Politically, I am a strong Democrat.",
    "not very strong democrat": "Politically, I am a not very strong Democrat.",
    "independent, close to democrat": "Politically, I am an independent who leans Democratic.",
    "independent (neither, no response)": "Politically, I am an independent.",
    "independent, close to republican": "Politically, I am an independent who leans Republican.",
    "not very strong republican": "Politically, I am a not very strong Republican.",
    "strong republican": "Politically, I am a strong Republican.",
    "other party": "Politically, I support a third party.",
}
POLVIEWS = {
    "extremely liberal": "Ideologically, I describe myself as extremely liberal.",
    "liberal": "Ideologically, I describe myself as liberal.",
    "slightly liberal": "Ideologically, I describe myself as slightly liberal.",
    "moderate, middle of the road": "Ideologically, I describe myself as moderate, middle of the road.",
    "slightly conservative": "Ideologically, I describe myself as slightly conservative.",
    "conservative": "Ideologically, I describe myself as conservative.",
    "extremely conservative": "Ideologically, I describe myself as extremely conservative.",
}

FRAME = ('In a survey, you described yourself in these words: "{backstory}" You are this person, and you '
         'answer every question the way this person would.')


def _g(row, k):
    return row.get(k) or ""


def _race(row):
    # ANES-style collapse (Argyle Fig. 1): any Hispanic origin -> "Hispanic", else the GSS race answer
    h = _g(row, "hispanic")
    if h and h != "not hispanic":
        return "Racially, I am Hispanic."
    return RACE.get(_g(row, "race")) if _g(row, "race") else None


def _born(row):
    return "I was not born in the United States." if _g(row, "born") == "no" else None


def _age(row):
    a = _g(row, "age")
    if not a:
        return None
    return "I am 89 years old or older." if a == "89+" else f"I am {a} years old."


def _children(row):
    c = _g(row, "childs")
    if not c:
        return None
    return {"0": "I have no children.", "1": "I have one child.",
            "8+": "I have eight or more children."}.get(c, f"I have {c} children.")


RENDERER = RowFragmentRenderer([
    _race, _born, ("sex", SEX), _age, ("region", REGION), ("marital", MARITAL), _children,
    ("degree", DEGREE), ("wrkstat", WRKSTAT), ("income16", "My total family income is {} a year."),
    ("class", CLASS), ("relig", RELIG), ("attend", ATTEND), ("partyid", PARTYID), ("polviews", POLVIEWS),
], frame=FRAME)


def render(row):
    return RENDERER.render(row)


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and WEIGHT_COL not in rows[0]:
        raise ValueError(f"{path}: missing column {WEIGHT_COL!r}")
    return rows


def load_rows(data_root=None):
    return cached_load(data_path(DATA_FILE, data_root), _read)


def _draw(n, seed=None, data_root=None):
    rows = load_rows(data_root)
    idx = weighted_indices_no_replacement([float(r[WEIGHT_COL]) for r in rows], n, seed)
    return [(render(rows[i]), {"row_id": rows[i].get("id", str(i))}) for i in idx]


METHOD = FunctionMethod(
    name="demographic", draw_fn=_draw, needs_data=True, data_file=DATA_FILE,
    description="First-person backstory from one survey-weighted real GSS 2024 respondent row.",
    paper="Argyle et al. 2023, Out of One, Many (Political Analysis)",
    source="edsl-test personas/demographic/__init__.py")

"""Persona injection comparison: five injectors (control, framing, own_answers, neighbours and population
families) against the no_persona and swap controls. Offline (FakeLLM), synthetic panel, no data needed.

    python examples/persona_inject_compare.py

The FakeLLM below is a scripted respondent: it votes with any past answers it is shown, else it reads the region
from the persona, else it says "no". The scores therefore check the plumbing (rendering, leakage rule, controls,
paired cluster-bootstrap CIs), not any real model. With a real LLM, replace `llm` with
`simengine.llm.Claude(...)` and keep the `dry_run=True` step: `plan.estimate` gives the call count and an
upper-bound cost before anything is spent.
Which injector to try first for which data: docs/modules/persona-inject.md ("Which injection when").
"""
import json
import random
import re

import simengine.persona as sp
from simengine.llm.fake import FakeLLM

TOPICS = ["own a car", "drive to work", "fly abroad yearly", "buy an electric car", "pay for parking",
          "rent a car on holiday", "commute over 30 minutes", "own a motorbike"]
BANK = [{"id": f"q{i}", "text": f"Do you {t}?", "options": ["yes", "no"]} for i, t in enumerate(TOPICS)]


def panel(n=40, seed=0):
    """Region sets the base rate; each person also has an own propensity that only their answers reveal."""
    rng = random.Random(seed)
    people = []
    for i in range(n):
        north = i % 2 == 1
        p_yes = min(0.95, max(0.05, (0.65 if north else 0.35) + rng.uniform(-0.3, 0.3)))
        people.append({"id": f"p{i:02d}",
                       "fields": {"age": rng.randint(18, 80), "region": "north" if north else "south",
                                  "income": rng.choice(["low", "middle", "high"])},
                       "answers": {q["id"]: "yes" if rng.random() < p_yes else "no" for q in BANK}})
    return people


def respondent(system, user):
    """Scripted stand-in for an LLM (see module docstring)."""
    shown = re.findall(r"^A: (yes|no)$", system, re.M)
    if shown:
        answer = "yes" if 2 * shown.count("yes") > len(shown) else "no"
    else:
        answer = "yes" if "north" in (system + user).lower() else "no"
    return json.dumps({"answer": answer})


def main():
    fams = {}
    for row in sp.list_injectors():
        fams.setdefault(row["family"], []).append(row["name"])
    print(f"{sum(map(len, fams.values()))} injectors in {len(fams)} families:")
    for fam, names in sorted(fams.items()):
        print(f"  {fam:12s} {', '.join(names)}")

    people = panel()
    arms = ["plain_fields", "argyle_backstory",
            ("own_answers_all", {"questions": BANK}),
            ("similar_respondents", {"pool": people, "k": 5}),
            ("cell_verbalized", {"keys": ["region"]})]
    llm = FakeLLM(respondent)

    plan = sp.compare_injectors(llm, people, BANK, "answers", arms, dry_run=True)
    print(f"\nplan: {plan.estimate['calls']} calls over {plan.estimate['arms']} arms x {plan.estimate['cells']} cells, "
          f"~{plan.estimate['input_tokens_est']} input tokens; no call made")

    res = sp.compare_injectors(llm, people, BANK, "answers", arms, B=500, parallel=1)
    cols = ["arm", "kind", "score", "lo", "hi", "diff_vs_no_persona", "lo_vs_no_persona", "hi_vs_no_persona",
            "diff_vs_swap"]
    print("\n" + res.table[cols].round(3).to_string(index=False))
    print("\nscore = accuracy over (person, question) cells; CIs resample people. diff_vs_swap compares an arm with "
          "the same injector given another person's data.\nScripted FakeLLM: these numbers test the pipeline, "
          "not a model.")
    return res


if __name__ == "__main__":
    main()

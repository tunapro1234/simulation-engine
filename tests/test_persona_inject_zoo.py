"""Cross-family integration of simengine.persona.inject: every registered injector renders, dry-runs through
compare_injectors with its swap control, never lets the scored person's target answer change the prompt, and a
mixed-family comparison runs end to end on FakeLLM."""
import contextlib
import copy
import importlib.util
import io
import json
import pathlib
import runpy
import unittest

import simengine.persona as sp
import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM
from simengine.persona.inject import from_logs as fl
from simengine.persona.inject import generated as gen

ROOT = pathlib.Path(__file__).resolve().parents[1]
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
OPTS = ["yes", "no"]
CUT = "2024-04-01"
TEXTS = ["Do you own a car?", "Do you take the bus to work?", "Do you cycle to work?", "Do you work from home?",
         "Would you buy an electric car?", "Would you pay for a bus pass?"]
QS = [{"id": f"q{i}", "text": t, "options": OPTS, "cutoff": CUT} for i, t in enumerate(TEXTS)]
FIT, TEST = QS[:4], QS[4:]
TARGET = TEST[0]["id"]
STORIES = [{"id": "s0", "text": "I grew up up north and still live there.", "question": "Tell me about yourself.",
            "demographics": {"region": {"north": 0.9, "south": 0.1}}},
           {"id": "s1", "text": "I have always lived in the south.", "question": "Tell me about yourself.",
            "demographics": {"region": {"north": 0.1, "south": 0.9}}}]


def yes(system, user):
    return json.dumps({"answer": "yes"})


def oracle(system, user):
    """Says yes when the prompt mentions the north (attribute line, backstory, cell...), else no."""
    return json.dumps({"answer": "yes" if "north" in (system + user).lower() else "no"})


def person(i):
    north = i % 2 == 1
    answers = {q["id"]: ("yes" if (north if j in (0, 2, 4, 5) else not north) else "no") for j, q in enumerate(QS)}
    return {
        "id": f"u{i}", "fields": {"age": 20 + 7 * i, "region": "north" if north else "south",
                                  "gender": "female" if i % 3 else "male"},
        "answers": answers, "country": "Chile" if north else "Peru",
        "interview": [("Tell me about your commute.", f"I {'drive' if north else 'walk'} to work."),
                      ("Where did you grow up?", "In the north." if north else "In the south."),
                      ("What do you do on weekends?", "I see friends.")],
        "interview_summary": {"commute": "drives" if north else "walks"},
        "expert_reflections": {"psychologist": ["Values routine."], "demographer": ["Lives in a small town."]},
        "history": [{"t": f"2024-03-0{d + 1}", "action": "purchase", "item": f"i{d}", "title": f"Item {d}",
                     "category": "auto" if north else "food", "price": 2.0 + d, "rating": 3 + d % 3,
                     "item_mean": 3.5} for d in range(5)],
        "generated": {"backstory": "I grew up in a small town.", "persona": "A daily commuter.",
                      "narrative": "A person who commutes every day."},
        "reflections": [{"text": "Likes routine.", "seen": ["q1"], "sources": ["q1"]}],
        "media": ["Fuel prices rose again this week."],
        "message": "Traffic was bad today." if north else "Nice walk this morning.",
        "action": "bought a car" if north else "renewed a bus pass", "time": i,
        "schwartz": {"benevolence": 5, "power": 2 + i % 3}, "mfq": {"care": 4, "loyalty": 1 + i % 3},
    }


def panel(n=8):
    people = [person(i) for i in range(n)]
    people = sp.inject.social.attach_neighbors(people, [(f"u{i}", f"u{(i + 1) % n}") for i in range(n)])
    people = sp.inject.social.attach_norms(people, QS, by="region")
    for style, kw in (("agent4rec", {}), ("paars", {}), ("task", {"task": "predict commuting choices"})):
        people = fl.summarize_logs(FakeLLM(lambda s, u: "Prefers practical products."), people, style=style,
                                   cutoff=CUT, **kw)
    return people


def _fitted(name, people, **kw):
    inj = pi.get_injector(name, seed=0, **kw)
    inj.fit(FakeLLM(yes), people, FIT, "answers", parallel=1)
    return inj


# one factory per registered injector: people -> instance (pool-based ones see the same people list)
CONFIG = {
    "no_persona": lambda ps: pi.get_injector("no_persona"),
    "plain_fields": lambda ps: pi.get_injector("plain_fields"),
    "caa_steer": lambda ps: pi.get_injector("caa_steer", traits=["age"]),
    "persona_vector_steer": lambda ps: pi.get_injector("persona_vector_steer", traits=["age"]),
    "centaur_qlora": lambda ps: pi.get_injector("centaur_qlora"),
    "soft_prompt": lambda ps: pi.get_injector("soft_prompt", keys=["region"]),
    "subpop_lora": lambda ps: pi.get_injector("subpop_lora"),
    "fitted_topk": lambda ps: _fitted("fitted_topk", ps, k=2),
    "fitted_stepwise": lambda ps: _fitted("fitted_stepwise", ps, max_turns=2, n_boot=50),
    "fitted_grips": lambda ps: _fitted("fitted_grips", ps, n_iter=1, m=2),
    "fitted_opro": lambda ps: _fitted("fitted_opro", ps, steps=1, per_step=2),
    "argyle_backstory": lambda ps: pi.get_injector("argyle_backstory"),
    "framed": lambda ps: pi.get_injector("framed", placement="turns", voice="first", format="prose"),
    "hu_collier_profile": lambda ps: pi.get_injector("hu_collier_profile"),
    "lutz_interview": lambda ps: pi.get_injector("lutz_interview"),
    "lutz_third_person": lambda ps: pi.get_injector("lutz_third_person"),
    "log_raw": lambda ps: pi.get_injector("log_raw"),
    "log_retrieved": lambda ps: pi.get_injector("log_retrieved", k=2),
    "log_stats": lambda ps: pi.get_injector("log_stats"),
    "log_agent4rec": lambda ps: pi.get_injector("log_agent4rec", cuts=fl.trait_cuts(ps)),
    "log_paars": lambda ps: pi.get_injector("log_paars"),
    "log_task_summary": lambda ps: pi.get_injector("log_task_summary", k=1),
    "anthology_backstory": lambda ps: pi.get_injector("anthology_backstory"),
    "anthology_greedy": lambda ps: pi.get_injector("anthology_greedy", pool=STORIES, keys=["region"]),
    "llm_narrative": lambda ps: pi.get_injector("llm_narrative", with_fields=True),
    "personahub": lambda ps: pi.get_injector("personahub"),
    "interview_transcript": lambda ps: pi.get_injector("interview_transcript", cot=False),
    "interview_lesion": lambda ps: pi.get_injector("interview_lesion", drop=0.4),
    "interview_summary": lambda ps: pi.get_injector("interview_summary"),
    "interview_expert_reflection": lambda ps: pi.get_injector("interview_expert_reflection", expert="all"),
    "interview_plus_survey": lambda ps: pi.get_injector("interview_plus_survey"),
    "memory_stream": lambda ps: pi.get_injector("memory_stream", k=3, questions=QS),
    "memory_reflect": lambda ps: pi.get_injector("memory_reflect", k=3, questions=QS),
    "memory_session": lambda ps: pi.get_injector("memory_session", prior_only=True, questions=QS),
    "answer_neighbours": lambda ps: pi.get_injector("answer_neighbours", pool=ps, questions=QS, k=2),
    "neighbour_votes": lambda ps: pi.get_injector("neighbour_votes", pool=ps, k=3),
    "similar_respondents": lambda ps: pi.get_injector("similar_respondents", pool=ps, k=3),
    "own_answers_all": lambda ps: pi.get_injector("own_answers_all", questions=QS),
    "own_answers_fewshot": lambda ps: pi.get_injector("own_answers_fewshot", questions=QS, k=2),
    "own_answers_rag": lambda ps: pi.get_injector("own_answers_rag", questions=QS, k=2),
    "cell_verbalized": lambda ps: pi.get_injector("cell_verbalized", keys=["region"]),
    "cell_sequence": lambda ps: pi.get_injector("cell_sequence", keys=["region"], n=10),
    "cell_sample": lambda ps: pi.get_injector("cell_sample", keys=["region"]),
    "cell_fewshot": lambda ps: pi.get_injector("cell_fewshot", keys=["region"], people=ps, questions=QS, k=2),
    "asch_majority": lambda ps: pi.get_injector("asch_majority", tone="confident"),
    "descriptive_norm": lambda ps: pi.get_injector("descriptive_norm"),
    "neighbor_activity": lambda ps: pi.get_injector("neighbor_activity"),
    "news_feed": lambda ps: pi.get_injector("news_feed"),
    "peer_messages": lambda ps: pi.get_injector("peer_messages", bias="weak"),
    "cultural_prompting": lambda ps: pi.get_injector("cultural_prompting"),
    "hofstede_country": lambda ps: pi.get_injector("hofstede_country"),
    "moral_foundations": lambda ps: pi.get_injector("moral_foundations"),
    "schwartz_anchor": lambda ps: pi.get_injector("schwartz_anchor"),
    "schwartz_scores": lambda ps: pi.get_injector("schwartz_scores"),
}
NEEDS_SKLEARN = {"fitted_topk"}


def names():
    return sorted(n for n in CONFIG if HAS_SKLEARN or n not in NEEDS_SKLEARN)


def build(people):
    return {n: CONFIG[n](people) for n in names()}


def flip(people, pid, qid):
    """Copy of people where person `pid` gave the other answer to `qid` (everywhere it is stored)."""
    out = copy.deepcopy(people)
    for p in out:
        if p["id"] == pid:
            p["answers"][qid] = "no" if p["answers"][qid] == "yes" else "yes"
    return out


def text(prompt):
    return json.dumps([prompt.system, prompt.user, prompt.history,
                       {k: v for k, v in prompt.meta.items() if k not in ("arm", "injector")}], sort_keys=True,
                      default=str)


class TestRegistry(unittest.TestCase):
    def test_all_families_import(self):
        self.assertEqual(pi.IMPORT_ERRORS, {})
        self.assertEqual(sorted(pi.FAMILIES), ["finetune", "fitted", "framing", "from_logs", "generated",
                                               "interview", "memory", "neighbours", "own_answers", "population",
                                               "social", "values"])

    def test_every_injector_has_a_zoo_config(self):
        missing = sorted(set(pi.REGISTRY) - set(CONFIG) - {n for n in pi.REGISTRY if n.startswith("_")})
        self.assertEqual(missing, [], "add a CONFIG row for each new injector")
        self.assertEqual(sorted(set(CONFIG) - set(pi.REGISTRY)), [])

    def test_metadata_docs_and_tests(self):
        for r in pi.list_injectors():
            if r["name"].startswith("_"):
                continue
            self.assertTrue(r["paper"] and r["description"], r["name"])
            self.assertIn(r["family"], ["control", *pi.FAMILIES], r["name"])
        for fam in pi.FAMILIES:
            self.assertTrue((ROOT / "docs" / "modules" / "inject" / f"{fam}.md").is_file(), fam)
            self.assertTrue((ROOT / "tests" / f"test_persona_inject_{fam}.py").is_file(), fam)

    def test_reexported_from_persona(self):
        self.assertIs(sp.inject, pi)
        self.assertIs(sp.compare_injectors, pi.compare_injectors)
        self.assertIs(sp.get_injector, pi.get_injector)
        self.assertIs(sp.list_injectors, pi.list_injectors)


class TestZoo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.people = panel()
        cls.inj = build(cls.people)

    def test_every_injector_dry_runs_with_swap(self):
        res = pi.compare_injectors(FakeLLM(yes), self.people, TEST, "answers", list(self.inj.values()),
                                   dry_run=True)
        n_inj = len(self.inj) - 1                                     # no_persona is the control, no swap
        self.assertEqual(len(res.arms), 1 + 2 * n_inj)
        self.assertEqual(res.estimate["calls"], len(res.arms) * len(self.people) * len(TEST))
        for arm, prompts in res.prompts.items():
            for pr in prompts:
                self.assertTrue(pr.user.strip(), arm)

    def test_target_answer_never_changes_the_prompt(self):
        """For every injector and its swap control: flipping the scored person's own answer to the target
        question (in the person and in every pool built from the panel) leaves the rendered prompt unchanged."""
        q = TEST[0]
        for pid in ("u0", "u3"):
            alt = flip(self.people, pid, TARGET)
            inj_alt = build(alt)
            for name in names():
                a = self.inj[name](next(p for p in self.people if p["id"] == pid), q)
                b = inj_alt[name](next(p for p in alt if p["id"] == pid), q)
                self.assertEqual(text(a), text(b), f"{name}: target answer of {pid} reached the prompt")
                if name == "no_persona":
                    continue
                sa = pi.swap_wrapper(self.inj[name], self.people, seed=0)
                sb = pi.swap_wrapper(inj_alt[name], alt, seed=0)
                ra = sa(next(p for p in self.people if p["id"] == pid), q)
                rb = sb(next(p for p in alt if p["id"] == pid), q)
                self.assertEqual(text(ra), text(rb), f"swap_{name}: target answer of {pid} reached the prompt")

    def test_fit_questions_stay_refused(self):
        for name in names():
            if name.startswith("fitted_"):
                with self.assertRaises(Exception, msg=name):
                    self.inj[name](self.people[0], FIT[0])

    def test_end_to_end_mixed_families(self):
        arms = [self.inj[n] for n in ("plain_fields", "argyle_backstory", "own_answers_rag", "similar_respondents",
                                      "cell_verbalized", "memory_stream", "schwartz_anchor")]
        res = pi.compare_injectors(FakeLLM(oracle), self.people, TEST, "answers", arms, B=50, parallel=1)
        t = res.table.set_index("arm")
        self.assertEqual(len(t), 1 + 2 * len(arms))
        self.assertTrue((t["parsed"] == len(self.people) * len(TEST)).all())
        self.assertEqual(t.loc["plain_fields", "score"], 1.0)        # oracle reads region from the attributes
        self.assertEqual(t.loc["no_persona", "score"], 0.5)
        for col in ("diff_vs_no_persona", "lo_vs_no_persona", "diff_vs_swap", "hi_vs_swap"):
            self.assertIn(col, t.columns)
        self.assertEqual(sum(res.errors.values()), 0)


class TestExample(unittest.TestCase):
    def test_persona_inject_compare_runs(self):
        path = ROOT / "examples" / "persona_inject_compare.py"
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            runpy.run_path(str(path), run_name="__main__")
        out = buf.getvalue()
        self.assertIn(f"{len(pi.REGISTRY)} injectors in {len(pi.FAMILIES) + 1} families", out)
        self.assertIn("swap_plain_fields", out)
        self.assertIn("no call made", out)


if __name__ == "__main__":
    unittest.main()

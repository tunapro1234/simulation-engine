"""survey: one parallel plan per arm (D3 rehearsal #4), refused_ids / unparsed_ids (#5), survey(tag=) (#3, sim side).

Offline: FakeLLM subclasses and a fake messages client, no network."""
import importlib
import json
import os
import re
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from simengine.llm import EmptyResponseError, FakeLLM, LLMError
from simengine.sim import REFRAME_NOTE, ask_panel, schema_responder, survey

survey_mod = importlib.import_module("simengine.sim.survey")   # the module (sim.survey is the function)

REFUSAL = "I can't produce this as a survey dataset. I'd be inventing answers for people who don't exist."


# ---------------------------------------------------------------- the serial version (before the fix), as reference
def serial_arm(llm, personas, questions, lang, batch, group_size, context, tag, usage, *, arm="persona", stim=None,
               stimulus_batching="group", info=None, ask_kw=None, llm_tag=None, parallel=4):
    """Verbatim logic of survey._arm before D3 #4: one ask_panel per stimulus group, run one after another."""
    ans = {p.pid: {} for p in personas}
    refused = set()
    bad = {q["id"]: 0 for q in questions}
    before = (usage.first_calls, usage.refused_calls, usage.recovered_calls)
    errors = []
    for g in range(0, len(questions), group_size):
        group = questions[g:g + group_size]
        for ctx, ps, inline in survey_mod._stim_groups(personas, context, stim, stimulus_batching):
            task, fields = survey_mod._task_fields(group, ctx, lang, inline_stimulus=bool(inline))
            r = ask_panel(llm, ps, task, fields, batch=batch, lang=lang, usage=usage, tag=f"{tag}:{g}",
                          llm_tag=llm_tag if llm_tag is not None else f"survey:{arm}", context=inline,
                          context_label="Shown to this person", refused_ids=refused, errors=errors, parallel=parallel,
                          **(ask_kw or {}))
            for pid, x in r.items():
                for q in group:
                    a = survey_mod._parse_answer(q, x, lang)
                    if a is False:
                        bad[q["id"]] += 1
                    elif a is not None:
                        ans[pid][q["id"]] = a
    if info is not None:
        first = usage.first_calls - before[0]
        rc = usage.refused_calls - before[1]
        info.update({"calls": first, "refused_calls": rc, "recovered_calls": usage.recovered_calls - before[2],
                     "refusal_rate": round(rc / first, 4) if first else 0.0, "refused_ids": refused,
                     "unparsed_items": bad, "errors": errors})
    return {pid: c for pid, c in ans.items() if c}


# ---------------------------------------------------------------- fakes
class PlanLLM(FakeLLM):
    """Records every map() call (prompt count, max_tokens, kwargs) and every (system, user, max_tokens) prompt.
    `responder(system, user)` may return an exception, which becomes an error slot (return_exceptions=True)."""

    def __init__(self, responder, tag="", parallel=None):
        super().__init__(responder)
        self.maps = []
        self.sent = []
        self.lock = threading.Lock()
        if tag:
            self.tag = tag
        if parallel is not None:
            self.parallel = parallel

    def map(self, prompts, *, max_tokens=256, parallel=None, return_exceptions=False, **kw):
        prompts = list(prompts)
        self.maps.append({"n": len(prompts), "max_tokens": max_tokens, "parallel": parallel, **kw})
        out = []
        for s, u in prompts:
            self.sent.append((s, u, max_tokens))
            text = self.responder(s, u)
            if isinstance(text, BaseException):
                if not return_exceptions:
                    raise text
                out.append(text)
                continue
            out.append(SimpleNamespace(text=text, model="fake", cached=False, input_tokens=len(u.split()),
                                       output_tokens=len(text.split()), cost_usd=0.0))
        return out


def mixed_responder():
    """Deterministic per prompt (independent of call order): film-1 always refuses, film-2 refuses until the
    reframed re-ask, film-3 returns no JSON, film-4 drops its first id, film-5 is empty once then answers,
    film-6 fails with an LLMError, film-7 answers out of range; the rest answer normally."""
    base = schema_responder(3)
    seen_empty = set()

    def respond(system, user):
        film = re.search(r"Film: (film-\d+)", user)
        film = film.group(1) if film else ""
        if film == "film-1":
            return REFUSAL
        if film == "film-2" and REFRAME_NOTE not in user:
            return REFUSAL
        if film == "film-3":
            return "Sorry, here is some prose without any structure."
        if film == "film-5" and (system, user) not in seen_empty:
            seen_empty.add((system, user))
            return EmptyResponseError("empty reply (stop_reason=max_tokens)")
        if film == "film-6":
            return LLMError("HTTP 500 after retries")
        text = base(system, user)
        if film == "film-4":
            text = json.dumps(json.loads(text)[1:])
        if film == "film-7":
            data = json.loads(text)
            for d in data:
                for k in d:
                    if k.endswith("_value"):
                        d[k] = 999
            text = json.dumps(data)
        return text

    return respond


def film_panel(sizes):
    """Personas each with a 'film' stimulus; sizes[i] personas see film-i (uneven sizes -> several max_tokens)."""
    rows = []
    for f, n in enumerate(sizes):
        for j in range(n):
            rows.append({"id": f"m{f}_{j}", "text": f"member {f}-{j}, rates often", "segment": "A" if j % 2 else "B",
                         "film": f"Film: film-{f}"})
    return rows


QS = [{"id": "r", "text": "What star rating would you give this film?", "kind": "numeric", "min": 1, "max": 5,
       "free_text": False},
      {"id": "w", "text": "Would you watch it this week?", "kind": "likert5", "free_text": False}]


class TestOnePlanMatchesSerial(unittest.TestCase):
    def run_both(self, sizes, **kw):
        outs, llms = [], []
        for arm_fn in (serial_arm, None):
            llm = PlanLLM(mixed_responder())
            ctx = mock.patch.object(survey_mod, "_arm", arm_fn) if arm_fn else mock.patch.object(
                survey_mod, "_arm", survey_mod._arm)
            with ctx:
                outs.append(survey(film_panel(sizes), QS, llm=llm, stimulus_col="film", batch=3, B=50, **kw))
            llms.append(llm)
        return outs, llms

    def assert_same(self, old, new, lo, ln):
        for key in ("answers", "controls", "usage", "refused", "unparsed", "refusal_rate", "unparsed_items",
                    "refusals", "warnings", "questions", "stimuli", "n_answered"):
            self.assertEqual(old[key], new[key], key)
        # the same prompts with the same max_tokens (hence the same cache keys), only grouped differently
        self.assertEqual(sorted(lo.sent), sorted(ln.sent))
        self.assertEqual(len(lo.calls), len(ln.calls))

    def test_identical_results_mixed_failures_multi_group(self):
        sizes = [4, 3, 5, 2, 3, 4, 3, 3, 1, 6]
        (old, new), (lo, ln) = self.run_both(sizes, question_group=1)
        self.assert_same(old, new, lo, ln)
        # the failure modes really happened
        self.assertGreater(new["refused"], 0)
        self.assertGreater(new["unparsed"], 0)
        self.assertGreater(new["usage"]["recovered_calls"], 0)
        self.assertGreater(new["usage"]["empty_recovered"], 0)
        self.assertGreater(new["usage"]["error_calls"], 0)
        self.assertGreater(new["unparsed_items"], 0)
        self.assertLess(len(ln.maps), len(lo.maps))

    def test_identical_results_inline_batching(self):
        (old, new), (lo, ln) = self.run_both([3, 4, 2, 5], stimulus_batching="inline")
        self.assert_same(old, new, lo, ln)

    def test_one_map_per_arm_first_pass(self):
        # 12 films x 3 personas, batch 3: one batch per film, as in the rehearsal (900 films x 1 batch)
        llm = PlanLLM(schema_responder(1))
        rows = film_panel([3] * 12)
        res = survey(rows, QS, llm=llm, stimulus_col="film", batch=3, B=50)
        self.assertEqual(res["n_answered"], 36)
        tags = [m["tag"] for m in llm.maps]
        self.assertEqual(tags, ["survey:persona", "survey:no_persona", "survey:persona_swap"])
        self.assertEqual(llm.maps[0]["n"], 12)            # all 12 stimulus groups in one map

        old = PlanLLM(schema_responder(1))
        with mock.patch.object(survey_mod, "_arm", serial_arm):
            ref = survey(rows, QS, llm=old, stimulus_col="film", batch=3, B=50)
        self.assertEqual(res["answers"], ref["answers"])
        self.assertEqual(res["controls"]["answers"], ref["controls"]["answers"])
        self.assertEqual(res["usage"], ref["usage"])
        self.assertEqual(len(old.maps), 36)               # serial: one map per film per arm

    def test_retry_pass_is_also_one_map(self):
        # every film refuses at the first ask and answers the reframed one: 2 maps per arm, not 2 per film
        base = schema_responder(2)
        llm = PlanLLM(lambda s, u: base(s, u) if REFRAME_NOTE in u else REFUSAL)
        res = survey(film_panel([2] * 6), QS, llm=llm, stimulus_col="film", batch=2, B=50, controls=[])
        self.assertEqual([m["n"] for m in llm.maps], [6, 6])
        self.assertEqual(res["usage"]["recovered_calls"], 6)
        self.assertEqual(res["refused"], 0)


class TestParallel(unittest.TestCase):
    def test_default_is_backend_parallel(self):
        llm = PlanLLM(schema_responder(0), parallel=16)
        survey(film_panel([2, 2]), QS, llm=llm, stimulus_col="film", B=20)
        self.assertEqual({m["parallel"] for m in llm.maps}, {16})

    def test_default_without_backend_attribute_is_4(self):
        llm = PlanLLM(schema_responder(0))
        self.assertFalse(hasattr(llm, "parallel"))
        survey(film_panel([2, 2]), QS, llm=llm, stimulus_col="film", B=20)
        self.assertEqual({m["parallel"] for m in llm.maps}, {4})

    def test_explicit_parallel_wins(self):
        llm = PlanLLM(schema_responder(0), parallel=16)
        survey(film_panel([2, 2]), QS, llm=llm, stimulus_col="film", B=20, parallel=2)
        self.assertEqual({m["parallel"] for m in llm.maps}, {2})


class TestRefusedIds(unittest.TestCase):
    def test_ids_lists_and_int_counts(self):
        llm = PlanLLM(mixed_responder())
        res = survey(film_panel([2, 2, 2, 2, 2, 2, 2]), QS, llm=llm, stimulus_col="film", batch=2, B=20,
                     controls=[])
        self.assertIsInstance(res["refused"], int)
        self.assertIsInstance(res["unparsed"], int)
        self.assertEqual(res["refused_ids"], ["m1_0", "m1_1"])           # film-1 refuses after the re-ask too
        # film-3: no JSON; film-4: first id dropped; film-6: LLMError slot
        self.assertEqual(res["unparsed_ids"], ["m3_0", "m3_1", "m4_0", "m6_0", "m6_1"])
        self.assertEqual(res["refused"], len(res["refused_ids"]))
        self.assertEqual(res["unparsed"], len(res["unparsed_ids"]))
        self.assertFalse(set(res["refused_ids"]) & set(res["answers"]))
        json.dumps(res)

    def test_empty_lists_when_all_answer(self):
        res = survey(film_panel([2, 2]), QS, llm=PlanLLM(schema_responder(0)), stimulus_col="film", B=20)
        self.assertEqual((res["refused"], res["unparsed"], res["refused_ids"], res["unparsed_ids"]), (0, 0, [], []))


class TestSurveyTag(unittest.TestCase):
    def test_default_tags_unchanged(self):
        llm = PlanLLM(schema_responder(0))
        survey(film_panel([2, 2]), QS, llm=llm, stimulus_col="film", B=20)
        self.assertEqual({m["tag"] for m in llm.maps}, {"survey:persona", "survey:no_persona", "survey:persona_swap"})

    def test_user_tag(self):
        llm = PlanLLM(schema_responder(0))
        survey(film_panel([2, 2]), QS, llm=llm, stimulus_col="film", B=20, tag="d3/dev")
        self.assertEqual({m["tag"] for m in llm.maps},
                         {"d3/dev:survey:persona", "d3/dev:survey:no_persona", "d3/dev:survey:persona_swap"})

    def test_backend_tag_joined_once(self):
        llm = PlanLLM(schema_responder(0), tag="L1")
        survey(film_panel([2]), QS, llm=llm, stimulus_col="film", B=20, tag="dev", controls=[])
        self.assertEqual({m["tag"] for m in llm.maps}, {"L1:dev:survey:persona"})

    def test_ask_panel_joins_backend_tag_without_guessing(self):
        # the call tag is not inspected: one that happens to start with "L1:" still gets the backend tag
        llm = PlanLLM(schema_responder(0), tag="L1")
        from simengine.sim import as_personas
        ask_panel(llm, as_personas(["a"]), "t", {"score": "int 1-5"}, llm_tag="L1:x")
        ask_panel(llm, as_personas(["a"]), "t", {"score": "int 1-5"}, llm_tag="x")
        self.assertEqual([m["tag"] for m in llm.maps], ["L1:L1:x", "L1:x"])


class _Client:
    """Minimal messages.create stand-in: answers panel prompts with schema_responder."""

    def __init__(self):
        self.resp = schema_responder(0)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **params):
        system = params.get("system")
        if isinstance(system, list):
            system = system[0]["text"]
        text = self.resp(system or "", params["messages"][0]["content"])
        return {"id": "m", "model": params["model"], "stop_reason": "end_turn",
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 0}}


class TestBackendTagEndToEnd(unittest.TestCase):
    """The panel docstring's claim on a real backend class (fake client): TSV tag = prefix + tag:survey tag."""

    def test_claude_tag_prefix_and_survey_tag(self):
        from simengine.llm import Claude
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        log = Path(d.name) / "spend.tsv"
        env = {k: "" for k in ("SIMENGINE_SPEND_LOG", "SIMENGINE_AGENT", "SIMENGINE_SPEND_TAG_PREFIX")}
        with mock.patch.dict(os.environ, env):
            try:
                llm = Claude("claude-haiku-5-5", effort="low", thinking="off", client=_Client(), tag="L1",
                             tag_prefix="d3/", spend_log=log, agent="team", parallel=3, sleep=lambda s: None)
            except TypeError:
                self.skipTest("backend has no tag_prefix= yet")
            res = survey(film_panel([2, 2]), QS, llm=llm, stimulus_col="film", B=20, tag="dev", controls=[])
        self.assertEqual(res["n_answered"], 4)
        tags = {ln.split("\t")[6] for ln in log.read_text(encoding="utf-8").splitlines()}
        self.assertEqual(tags, {"d3/L1:dev:survey:persona"})


if __name__ == "__main__":
    unittest.main()

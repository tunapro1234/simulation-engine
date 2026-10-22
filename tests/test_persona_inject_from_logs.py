import json
import unittest

import numpy as np
import pandas as pd

import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM
from simengine.persona.inject import from_logs as fl

NAMES = ("log_raw", "log_retrieved", "log_stats", "log_agent4rec", "log_paars", "log_task_summary")
SUMMARY = {"log_agent4rec": "agent4rec", "log_paars": "paars", "log_task_summary": "task"}
CUT = "2024-04-01"


def shopper(i=1, cat="dairy"):
    other = "snacks" if cat == "dairy" else "dairy"
    return {"id": f"u{i}", "history": [
        {"t": "2024-01-10", "action": "purchase", "item": f"{cat}-1", "title": f"{cat} pack one", "category": cat,
         "brand": "B", "price": 2.0, "qty": 2, "discount": True},
        {"t": "2024-02-03", "action": "view", "item": "v1", "title": "Premium tea", "category": "drinks",
         "brand": "C", "price": 8.0},
        {"t": "2024-02-20", "action": "purchase", "item": f"{cat}-1", "title": f"{cat} pack one", "category": cat,
         "brand": "B", "price": 2.0, "discount": False},
        {"t": "2024-03-01", "action": "purchase", "item": "x1", "title": "Mixed box", "category": [cat, other],
         "brand": "D", "price": 4.0},
        {"t": "2024-03-15", "action": "rate", "item": "m1", "title": "Heat (1995)", "category": "Action|Crime",
         "rating": 4},
        {"t": "2024-05-02", "action": "purchase", "item": "late-1", "title": "Future secret", "category": "luxury",
         "price": 99.0},
        {"action": "purchase", "item": "nodate", "title": "Undated thing", "category": "misc"},
    ]}


Q = {"id": "q1", "text": "Would you buy oat milk at 2.50?", "options": ["yes", "no"], "cutoff": CUT}


def summarised(people, style, **kw):
    llm = FakeLLM(lambda s, u: f"You are a {style} shopper.")
    if style == "task":
        kw.setdefault("task", "predict what the user buys next")
    return fl.summarize_logs(llm, people, style=style, cutoff=CUT, **kw)


def ready(name, person, **kw):
    return summarised([person], SUMMARY[name], **kw)[0] if name in SUMMARY else person


class TestRegistry(unittest.TestCase):
    def test_listed_with_papers(self):
        self.assertNotIn("from_logs", pi.IMPORT_ERRORS)
        rows = {r["name"]: r for r in pi.list_injectors("from_logs")}
        self.assertEqual(set(rows), set(NAMES))
        for r in rows.values():
            self.assertIn("arXiv:", r["paper"])
            self.assertEqual(r["needs"], ["history"])
            self.assertTrue(r["description"])
        for name, arxiv in (("log_raw", "2305.06474"), ("log_retrieved", "2304.11406"),
                            ("log_agent4rec", "2310.10108"), ("log_paars", "2503.24228"),
                            ("log_task_summary", "2310.20081")):
            self.assertIn(arxiv, rows[name]["paper"])

    def test_labels(self):
        self.assertEqual(pi.get_injector("log_raw").label, "log_raw")
        self.assertEqual(pi.get_injector("log_raw", n=5).label, "log_raw(n=5)")
        self.assertEqual(pi.get_injector("log_retrieved", k=3, method="random", seed=2).label,
                         "log_retrieved(k=3,method=random,seed=2)")

    def test_bad_params(self):
        for name, kw in (("log_raw", {"n": 0}), ("log_raw", {"order": "sideways"}),
                         ("log_retrieved", {"k": 0}), ("log_retrieved", {"method": "embeddings"}),
                         ("log_agent4rec", {"taste": "vibes"})):
            with self.assertRaises(ValueError, msg=(name, kw)):
                pi.get_injector(name, **kw)
        with self.assertRaises(ValueError):
            fl.summarize_logs(FakeLLM(), [shopper()], style="haiku")
        with self.assertRaises(ValueError):
            fl.summarize_logs(FakeLLM(), [shopper()], style="task")          # task= is required
        for name, kw in (("log_paars", {"n_raw": 0}), ("log_stats", {"top_k": 0}),
                         ("log_task_summary", {"k": -1}), ("log_agent4rec", {"n_items": 0})):
            with self.assertRaises(ValueError, msg=(name, kw)):
                pi.get_injector(name, **kw)
        for kw in ({"n": 0}, {"n_items": 0}):                            # [-0:] would be the whole log
            with self.assertRaises(ValueError, msg=kw):
                fl.summarize_logs(FakeLLM(), [shopper()], style="paars", **kw)


class TestMissingAndBadInput(unittest.TestCase):
    def test_missing_history_raises_listing_needs(self):
        for name in NAMES:
            for person in ({"id": "u9"}, {"id": "u9", "history": []}):
                with self.assertRaises(pi.MissingPersonField, msg=name) as cm:
                    pi.get_injector(name)(person, Q)
                self.assertIn("history", str(cm.exception))
                self.assertIn(name, str(cm.exception))

    def test_summary_injectors_need_a_summary(self):
        for name, style in SUMMARY.items():
            with self.assertRaises(pi.MissingPersonField, msg=name) as cm:
                pi.get_injector(name)(shopper(), Q)
            msg = str(cm.exception)
            self.assertIn(fl.SUMMARY_FIELD, msg)
            self.assertIn("summarize_logs", msg)
            self.assertIn(style, msg)
        pi.get_injector("log_agent4rec", taste="items")(shopper(), Q)           # no LLM summary needed

    def test_chat_turns_refused(self):
        with self.assertRaises(ValueError):
            pi.get_injector("log_raw")({"id": "u1", "history": [("user", "hi")]}, Q)

    def test_mixed_time_kinds_refused(self):
        p = {"id": "u1", "history": [{"t": 3, "item": "a"}, {"t": "2024-01-01", "item": "b"}]}
        with self.assertRaises(ValueError):
            pi.get_injector("log_raw")(p, {"id": "q", "text": "?", "options": ["a", "b"]})

    def test_dataframe_records(self):
        """Rows from DataFrame.to_dict('records'): NaT / NaN mean absent, numpy scalars work."""
        df = pd.DataFrame([
            {"t": "2024-01-10", "item": "a", "title": "Apple", "price": 1.0, "rating": np.nan, "discount": np.nan},
            {"t": None, "item": "b", "title": "NaT secret", "price": 2.0, "rating": np.nan, "discount": True}])
        df["t"] = pd.to_datetime(df["t"])
        p = {"id": "u1", "history": df.to_dict("records")}
        self.assertEqual([e["title"] for e in fl.visible_events(p, Q)], ["Apple"])   # NaT is undated: hidden
        s = fl.log_stats(fl.visible_events(p, Q))
        self.assertEqual((s["n_purchases"], s.get("n_ratings")), (1, None))        # NaN rating is no rating
        self.assertNotIn("discount_share", s)
        text = pi.get_injector("log_raw")(p, Q).system
        self.assertNotIn("NaT secret", text)
        self.assertNotIn("nan", text.lower())
        n = {"id": "u2", "history": [{"t": np.int64(2), "item": "x"}, {"t": np.int64(7), "item": "late"}]}
        q = {"id": "q", "text": "?", "options": ["a", "b"], "cutoff": np.int64(5)}
        self.assertEqual([e["item"] for e in fl.visible_events(n, q)], ["x"])

    def test_no_visible_events_still_renders(self):
        p = {"id": "u1", "history": [{"t": "2025-01-01", "item": "a", "price": 1}]}
        for name in ("log_raw", "log_retrieved", "log_stats"):
            out = pi.get_injector(name)(p, Q)
            self.assertEqual(out.meta["n_events"], 0, msg=name)
            self.assertIn("no recorded", out.system)


class TestLeakage(unittest.TestCase):
    def prompts(self, question, person=None, **kw):
        person = person or shopper()
        hide = [question["item"]] if "item" in question else []
        return {n: pi.get_injector(n, **kw)(ready(n, person, hide_items=hide), question) for n in NAMES}

    def test_cutoff_hides_later_and_undated_events(self):
        for name, p in self.prompts(Q).items():
            text = p.system + p.user
            self.assertNotIn("Future secret", text, msg=name)
            self.assertNotIn("Undated thing", text, msg=name)
        self.assertNotIn("luxury", pi.get_injector("log_stats")(shopper(), Q).system)

    def test_no_cutoff_shows_undated(self):
        p = pi.get_injector("log_raw", n=50)(shopper(), {"id": "q", "text": "?", "options": ["a", "b"]})
        self.assertIn("Future secret", p.system)
        self.assertIn("Undated thing", p.system)

    def test_injector_cutoff_earlier_wins(self):
        p = pi.get_injector("log_raw", cutoff="2024-02-01")(shopper(), Q)
        self.assertEqual(p.meta["n_events"], 1)
        p = pi.get_injector("log_raw", cutoff="2030-01-01")(shopper(), Q)       # question's cutoff is earlier
        self.assertNotIn("Future secret", p.system)

    def test_target_item_hidden(self):
        q = {**Q, "item": "dairy-1"}
        for name, p in self.prompts(q).items():
            self.assertNotIn("dairy pack one", p.system + p.user, msg=name)
        p = pi.get_injector("log_raw", hide_target_item=False)(shopper(), q)
        self.assertIn("dairy pack one", p.system)

    def test_qid_tagged_events_hidden(self):
        person = shopper()
        person["history"] = person["history"] + [
            {"t": "2024-03-20", "action": "answer", "qid": "q1", "text": "TARGET-ANSWER"},
            {"t": "2024-03-21", "action": "answer", "qid": "q2", "text": "EXCLUDED-ANSWER"},
            {"t": "2024-03-22", "action": "answer", "qid": "q3", "text": "kept answer"}]
        q = {**Q, "exclude": ["q2"]}
        for name in ("log_raw", "log_retrieved"):
            text = pi.get_injector(name, **({"k": 20} if name == "log_retrieved" else {"n": 20}))(person, q).system
            self.assertNotIn("TARGET-ANSWER", text, msg=name)
            self.assertNotIn("EXCLUDED-ANSWER", text, msg=name)
            self.assertIn("kept answer", text, msg=name)

    def test_summary_built_too_late_is_refused(self):
        llm = FakeLLM(lambda s, u: "summary")
        for name, style in SUMMARY.items():
            kw = {"task": "predict purchases"} if style == "task" else {}
            nocut = fl.summarize_logs(llm, [shopper()], style=style, **kw)[0]
            with self.assertRaises(ValueError, msg=name) as cm:
                pi.get_injector(name)(nocut, Q)
            self.assertIn("leakage", str(cm.exception))
            later = fl.summarize_logs(llm, [shopper()], style=style, cutoff="2024-06-01", **kw)[0]
            with self.assertRaises(ValueError, msg=name):
                pi.get_injector(name)(later, Q)
            earlier = fl.summarize_logs(llm, [shopper()], style=style, cutoff="2024-03-01", **kw)[0]
            pi.get_injector(name)(earlier, Q)

    def test_summary_built_from_target_item_is_refused(self):
        q = {**Q, "item": "dairy-1"}
        for name, style in SUMMARY.items():
            if style == "agent4rec":
                continue                                     # samples rated events only; dairy-1 is never rated
            p = summarised([shopper()], style)[0]
            with self.assertRaises(ValueError, msg=name):
                pi.get_injector(name)(p, q)
            clean = summarised([shopper()], style, hide_items=["dairy-1"])[0]
            self.assertNotIn("dairy-1", clean[fl.SUMMARY_FIELD][style]["items"])
            pi.get_injector(name)(clean, q)

    def test_summary_built_from_target_qid_is_refused(self):
        person = shopper()
        person["history"] = person["history"] + [
            {"t": "2024-03-20", "action": "answer", "qid": "q1", "text": "TARGET-ANSWER"}]
        echo = FakeLLM(lambda s, u: u)                       # the summary repeats its input
        for style, name in (("paars", "log_paars"), ("task", "log_task_summary")):
            kw = {"task": "predict purchases"} if style == "task" else {}
            leaky = fl.summarize_logs(echo, [person], style=style, cutoff=CUT, **kw)[0]
            self.assertIn("TARGET-ANSWER", leaky[fl.SUMMARY_FIELD][style]["text"])
            self.assertEqual(leaky[fl.SUMMARY_FIELD][style]["qids"], ["q1"])
            with self.assertRaises(ValueError, msg=name) as cm:
                pi.get_injector(name)(leaky, Q)
            self.assertIn("leakage", str(cm.exception))
            with self.assertRaises(ValueError, msg=name):                  # an exclude qid counts too
                pi.get_injector(name)(leaky, {**Q, "id": "other", "exclude": ["q1"]})
            clean = fl.summarize_logs(echo, [person], style=style, cutoff=CUT, hide_qids=["q1"], **kw)[0]
            p = pi.get_injector(name)(clean, Q)
            self.assertNotIn("TARGET-ANSWER", p.system + p.user, msg=name)

    def test_summary_calls_see_only_the_window(self):
        llm = FakeLLM(lambda s, u: "ok")
        fl.summarize_logs(llm, [shopper()], style="paars", cutoff=CUT, hide_items=["x1"])
        self.assertEqual(len(llm.calls), 2)
        for system, user in llm.calls:
            self.assertNotIn("Future secret", user)
            self.assertNotIn("Undated thing", user)
            self.assertNotIn("Mixed box", user)


class TestDeterminism(unittest.TestCase):
    def many(self):
        return {"id": "u1", "history": [{"t": f"2024-01-{d:02d}", "item": f"i{d}", "title": f"thing {d}",
                                         "rating": d % 5 + 1} for d in range(1, 29)]}

    def test_renders_are_pure(self):
        for name in NAMES:
            inj = pi.get_injector(name)
            a, b = inj(ready(name, shopper()), Q), inj(ready(name, shopper()), Q)
            self.assertEqual((a.system, a.user), (b.system, b.user), msg=name)

    def test_random_retrieval_seeded(self):
        q = {"id": "q", "text": "?", "options": ["a", "b"]}
        a = pi.get_injector("log_retrieved", method="random", k=5, seed=1)(self.many(), q).system
        b = pi.get_injector("log_retrieved", method="random", k=5, seed=1)(self.many(), q).system
        c = pi.get_injector("log_retrieved", method="random", k=5, seed=2)(self.many(), q).system
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_agent4rec_sample_seeded(self):
        evs = fl.visible_events(self.many(), {})
        s1 = fl.agent4rec_sample(self.many(), evs, 25, seed=0)
        self.assertEqual(len(s1), 25)
        self.assertEqual(s1, fl.agent4rec_sample(self.many(), evs, 25, seed=0))
        self.assertNotEqual(s1, fl.agent4rec_sample(self.many(), evs, 25, seed=1))
        self.assertEqual([e["t"] for e in s1], sorted(e["t"] for e in s1))      # oldest first


class TestRendering(unittest.TestCase):
    def test_raw_last_n_and_order(self):
        p = pi.get_injector("log_raw", n=2)(shopper(), Q)
        self.assertEqual(p.meta["n_events"], 2)
        self.assertNotIn("2024-02-20", p.system)
        self.assertLess(p.system.index("Mixed box"), p.system.index("Heat (1995)"))
        r = pi.get_injector("log_raw", n=2, order="recent_first", recency_note=True)(shopper(), Q).system
        self.assertLess(r.index("Heat (1995)"), r.index("Mixed box"))
        self.assertIn('most recent event is: rate "Heat (1995)"', r)
        self.assertIn("Would you buy oat milk", p.user)
        self.assertIn('"answer"', p.user)

    def test_discount_rendering(self):
        for d, want in ((True, "on discount"), (np.bool_(True), "on discount"), (1, "on discount"),
                        (0.25, "25% off"), ("yes", "on discount")):
            self.assertIn(want, fl.event_line({"item": "x", "discount": d}), msg=d)
        for d in (False, 0, np.bool_(False)):
            self.assertNotIn("discount", fl.event_line({"item": "x", "discount": d}), msg=d)

    def test_recent_retrieval_and_task_summary_events(self):
        p = pi.get_injector("log_retrieved", k=2, method="recent")(shopper(), Q).system
        self.assertLess(p.index("Heat (1995)"), p.index("Mixed box"))
        self.assertNotIn("Premium tea", p)
        q = {"id": "q", "text": "Do you like premium tea?", "options": ["yes", "no"], "cutoff": CUT}
        person = summarised([shopper()], "task")[0]
        self.assertNotIn("Related past events", pi.get_injector("log_task_summary")(person, q).system)
        with_k = pi.get_injector("log_task_summary", k=1)(person, q)
        self.assertIn("Related past events:\n- 2024-02-03 view \"Premium tea\"", with_k.system)
        self.assertEqual(with_k.meta["n_events"], 1)

    def test_agent4rec_prompt_uses_like_at(self):
        llm = FakeLLM(lambda s, u: "ok")
        fl.summarize_logs(llm, [shopper()], style="agent4rec", cutoff=CUT, like_at=4)
        self.assertIn("liked = rating 4 or above", llm.calls[0][1])
        self.assertIn("Liked: Heat (1995)", llm.calls[0][1])

    def test_bm25_puts_relevant_first(self):
        q = {"id": "q", "text": "Do you like premium tea?", "options": ["yes", "no"], "cutoff": CUT}
        p = pi.get_injector("log_retrieved", k=1)(shopper(), q)
        self.assertIn("Premium tea", p.system)
        self.assertEqual(fl.bm25_rank("anything", []), [])

    def test_stats_numbers(self):
        s = fl.log_stats(fl.visible_events(shopper(), Q), ref=CUT)
        self.assertEqual(s["n_purchases"], 3)
        self.assertEqual(s["recency_days"], 31)                   # 2024-03-01 -> 2024-04-01
        self.assertAlmostEqual(s["spend_total"], 2 * 2.0 + 2.0 + 4.0)
        self.assertAlmostEqual(s["discount_share"], 0.5)          # one event has no discount key
        self.assertAlmostEqual(s["top_brand_share"], 2 / 3)
        self.assertAlmostEqual(s["repeat_share"], 1 / 3)
        self.assertAlmostEqual(s["bought_vs_viewed_price"], (8 / 3) / 8.0)
        self.assertEqual(s["categories"][0], ("dairy", 0.75))
        self.assertEqual(s["n_ratings"], 1)
        text = pi.get_injector("log_stats")(shopper(), Q).system
        for word in ("recency", "frequency", "monetary", "category affinity", "price sensitivity",
                     "brand loyalty", "Purchases per category", "dairy: 3 items on 3 different days"):
            self.assertIn(word, text)
        flat = pi.get_injector("log_stats", per_category=False)(shopper(), Q).system
        self.assertNotIn("Purchases per category", flat)

    def test_numeric_times(self):
        p = {"id": "u1", "history": [{"t": 1, "item": "a", "price": 3}, {"t": 5, "item": "a", "price": 3},
                                     {"t": 9, "item": "b", "price": 1}]}
        q = {"id": "q", "text": "?", "options": ["a", "b"], "cutoff": 8}
        s = fl.log_stats(fl.visible_events(p, q), ref=8)
        self.assertEqual(s["recency_days"], 3)
        self.assertNotIn("t=9", pi.get_injector("log_raw")(p, q).system)
        self.assertIn("3 time units", pi.get_injector("log_stats")(p, q).system)
        out = fl.summarize_logs(FakeLLM(), [p], style="task", task="t", cutoff=8)[0]
        self.assertEqual(out[fl.SUMMARY_FIELD]["task"]["cutoff"], 8)
        pi.get_injector("log_task_summary")(out, q)

    def test_agent4rec_traits_and_tiers(self):
        people = [{"id": f"u{i}", "history": [{"t": f"2024-01-0{d}", "item": f"m{d}", "rating": (i + d) % 5 + 1,
                                               "category": f"g{d % (i + 1)}"} for d in range(1, i + 2)]}
                  for i in range(6)]
        people = fl.add_item_means(people)
        tr = fl.agent4rec_traits(fl.visible_events(people[5], {}))
        self.assertEqual(tr["activity"], 6)
        self.assertEqual(tr["diversity"], 6)
        self.assertIsNotNone(tr["conformity"])
        cuts = fl.trait_cuts(people)
        self.assertEqual(set(cuts), set(fl.TRAITS))
        text = pi.get_injector("log_agent4rec", taste="items", cuts=cuts)(people[5], {"id": "q", "text": "?",
                                                                                       "options": ["a", "b"]}).system
        self.assertIn("You are a very active rater", text)
        self.assertIn("Liked:", text)
        self.assertIsNone(fl.agent4rec_traits([{"rating": 3}])["conformity"])

    def test_paars_layout(self):
        p = pi.get_injector("log_paars")(summarised([shopper()], "paars")[0], Q).system
        self.assertIn("Your consumer profile:", p)
        self.assertIn("Your shopping preferences:", p)
        self.assertIn("Session 2024-01-10:", p)
        no_raw = pi.get_injector("log_paars", include_raw=False)(summarised([shopper()], "paars")[0], Q).system
        self.assertNotIn("Session", no_raw)

    def test_summarize_dry_run_and_calls(self):
        llm = FakeLLM()
        est = fl.summarize_logs(llm, [shopper(1), shopper(2)], style="paars", dry_run=True)
        self.assertEqual(est["calls"], 4)
        self.assertEqual(llm.calls, [])
        out = fl.summarize_logs(llm, [shopper(1), shopper(2)], style="agent4rec", cutoff=CUT)
        self.assertEqual(len(llm.calls), 2)
        s = out[0][fl.SUMMARY_FIELD]["agent4rec"]
        self.assertEqual((s["cutoff"], s["items"], s["model"]), (CUT, ["m1"], "fake"))
        both = summarised(out, "task")[0][fl.SUMMARY_FIELD]
        self.assertEqual(set(both), {"agent4rec", "task"})                  # styles accumulate


class TestCompare(unittest.TestCase):
    def test_compare_with_oracle(self):
        """Oracle answers the category it sees in the prompt: log arms beat no_persona and swap."""
        people = [shopper(i, "dairy" if i % 2 else "snacks") for i in range(6)]
        q = {"id": "aisle", "text": "Which aisle do you shop most?", "options": ["dairy", "snacks"], "cutoff": CUT}
        truth = {(p["id"], "aisle"): ("dairy" if i % 2 else "snacks") for i, p in enumerate(people)}

        def oracle(system, user):                            # the category named first (own top category)
            d, s = system.find("dairy"), system.find("snacks")
            return json.dumps({"answer": "dairy" if d >= 0 and (s < 0 or d < s) else "snacks"})
        llm = FakeLLM(oracle)
        res = pi.compare_injectors(llm, people, [q], truth, ["log_raw", "log_stats"], B=200)
        t = res.table.set_index("arm")
        for arm in ("log_raw", "log_stats"):
            self.assertAlmostEqual(t.loc[arm, "score"], 1.0, msg=arm)
            self.assertGreater(t.loc[arm, "diff_vs_swap"], 0, msg=arm)
        self.assertAlmostEqual(t.loc["no_persona", "score"], 0.5)
        for system, user in llm.calls:
            self.assertNotIn("Future secret", system + user)


if __name__ == "__main__":
    unittest.main()

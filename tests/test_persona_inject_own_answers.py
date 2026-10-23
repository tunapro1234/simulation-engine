import json
import re
import unittest

import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM
from simengine.persona.inject import own_answers as oa

BANK = [
    {"id": "car", "text": "Do you own a car?", "options": ["yes", "no"]},
    {"id": "bus", "text": "Do you take the bus to work?", "options": ["yes", "no"]},
    {"id": "pets", "text": "Do you have pets at home?", "options": ["yes", "no"]},
    {"id": "garage", "text": "Does your home have a garage for cars?", "options": ["yes", "no"]},
    {"id": "ncars", "text": "How many cars does your household own?", "min": 0, "max": 5},
]
Q = {b["id"]: b for b in BANK}


def person(i=1):
    y = "yes" if i % 2 else "no"
    n = "no" if i % 2 else "yes"
    return {"id": f"u{i}", "answers": {"car": y, "bus": n, "pets": y, "garage": y, "ncars": 2 if i % 2 else 0}}


NAMES = ("own_answers_all", "own_answers_fewshot", "own_answers_rag")


def make(name, **kw):
    return pi.get_injector(name, questions=BANK, **kw)


class TestRegistry(unittest.TestCase):
    def test_listed_with_papers(self):
        rows = {r["name"]: r for r in pi.list_injectors("own_answers")}
        self.assertEqual(set(rows), set(NAMES))
        self.assertEqual(pi.IMPORT_ERRORS, {})
        for r in rows.values():
            self.assertIn("arXiv:", r["paper"])
            self.assertEqual(r["needs"], ["answers"])
        self.assertIn("2505.17479", rows["own_answers_all"]["paper"])
        self.assertIn("2305.14929", rows["own_answers_rag"]["paper"])

    def test_label_shows_bank_size(self):
        self.assertEqual(make("own_answers_rag", k=3).label, "own_answers_rag(k=3,questions=5q)")
        self.assertEqual(pi.get_injector("own_answers_all").label, "own_answers_all")

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            pi.get_injector("own_answers_rag")                 # no bank
        with self.assertRaises(ValueError):
            make("own_answers_rag", scorer="embeddings")
        with self.assertRaises(ValueError):
            make("own_answers_fewshot", k=0)
        with self.assertRaises(ValueError):
            make("own_answers_all", order="sideways")


class TestMissingAndEmpty(unittest.TestCase):
    def test_missing_answers_raises_listing_needs(self):
        for name in NAMES:
            with self.assertRaises(pi.MissingPersonField) as cm:
                make(name)({"id": "x", "text": "no answers here"}, Q["car"])
            self.assertIn("answers", str(cm.exception))
            self.assertIn(name, str(cm.exception))

    def test_only_target_answered(self):
        for name in NAMES:
            p = make(name)({"id": "x", "answers": {"car": "yes"}}, Q["car"])
            self.assertEqual(p.meta["n_shown"], 0)
            self.assertIn(oa.NONE_KNOWN, p.system)
            self.assertNotIn("A: yes", p.system + p.flatten())


class TestLeakage(unittest.TestCase):
    def test_target_and_exclude_never_shown(self):
        q = {**Q["car"], "exclude": ["garage"]}
        for name in NAMES:
            for kw in ({}, {"order": "random", "seed": 4} if name != "own_answers_rag" else {"order": "asc"}):
                p = make(name, **kw)(person(), q)
                self.assertNotIn("car", p.meta["shown"])
                self.assertNotIn("garage", p.meta["shown"])
                ctx = p.system + "\n".join(t for _, t in p.history)
                self.assertNotIn("Do you own a car?", ctx)
                self.assertNotIn("garage", ctx)
                self.assertTrue(p.user.startswith("Do you own a car?"))

    def test_swap_uses_donor_without_target(self):
        ps = [person(i) for i in range(4)]
        s = pi.swap_wrapper(make("own_answers_all"), ps, seed=1)
        p = s(ps[0], Q["ncars"])
        self.assertNotEqual(p.meta["donor"], "u0")
        self.assertNotIn("ncars", p.meta["shown"])


    def test_same_text_copy_under_other_id_dropped(self):
        bank = BANK + [{"id": "car_retest", "text": "  do you OWN a car? ", "options": ["yes", "no"]}]
        pers = {"id": "u1", "answers": {**person()["answers"], "car_retest": "yes"}}
        for name in NAMES:
            p = pi.get_injector(name, questions=bank)(pers, Q["car"])
            self.assertNotIn("car_retest", p.meta["shown"], name)
            self.assertEqual(p.meta["dropped_same_text"], ["car_retest"], name)
            self.assertNotIn("OWN a car", p.system + "\n".join(t for _, t in p.history), name)
            kept = pi.get_injector(name, questions=bank, drop_same_text=False)(pers, Q["car"])
            self.assertIn("car_retest", kept.meta["shown"], name)       # opt-out restores the old behaviour
            self.assertNotIn("dropped_same_text", kept.meta)
        self.assertNotIn("dropped_same_text", make("own_answers_all")(person(), Q["car"]).meta)


class TestAnswerValues(unittest.TestCase):
    def test_none_and_nan_skipped(self):
        pers = {"id": "u", "answers": {"car": None, "bus": float("nan"), "pets": "yes"}}
        for name in NAMES:
            p = make(name)(pers, Q["ncars"])
            self.assertEqual(p.meta["shown"], ["pets"], name)
            ctx = p.system + p.flatten()
            self.assertNotIn("None", ctx)
            self.assertNotIn("nan", ctx)

    def test_only_missing_left_counts_as_none_known(self):
        p = make("own_answers_all")({"id": "u", "answers": {"car": None}}, Q["ncars"])
        self.assertEqual(p.meta["n_shown"], 0)
        self.assertIn(oa.NONE_KNOWN, p.system)

    def test_fewshot_reply_non_ascii_and_numpy(self):
        p = make("own_answers_fewshot", select="first")({"id": "u", "answers": {"car": "Hayır"}}, Q["ncars"])
        self.assertEqual(p.history[1][1], '{"answer": "Hayır"}')
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        pers = {"id": "u", "answers": {"ncars": np.int64(2), "bus": np.float64("nan")}}
        p = make("own_answers_fewshot", select="first")(pers, Q["car"])
        self.assertEqual(p.meta["shown"], ["ncars"])
        self.assertEqual(json.loads(p.history[1][1]), {"answer": 2})


class TestAll(unittest.TestCase):
    def test_twin2k_text(self):
        p = make("own_answers_all")(person(), Q["ncars"])
        self.assertEqual(p.meta["shown"], ["car", "bus", "pets", "garage"])
        self.assertIn("Q: Do you own a car?\nOptions: yes / no\nA: yes", p.system)
        self.assertIn("staying consistent", p.system)
        self.assertEqual(p.history, [])
        self.assertIn('{"answer": <number>}', p.user)
        no_opt = make("own_answers_all", include_options=False)(person(), Q["ncars"])
        self.assertNotIn("Options:", no_opt.system)

    def test_order_and_max_items(self):
        f = lambda **kw: make("own_answers_all", **kw)(person(), Q["ncars"]).meta["shown"]  # noqa: E731
        self.assertEqual(f(order="reverse"), ["garage", "pets", "bus", "car"])
        self.assertEqual(f(max_items=2), ["pets", "garage"])           # keeps the most recent
        self.assertEqual(f(order="random", seed=7), f(order="random", seed=7))
        self.assertEqual(sorted(f(order="random", seed=7)), sorted(f()))

    def test_without_bank_uses_ids(self):
        p = pi.get_injector("own_answers_all")({"id": "x", "answers": {"b": 1, "a": 2, "t": 3}}, {"id": "t", "text": "T?"})
        self.assertEqual(p.meta["shown"], ["a", "b"])
        self.assertIn("Q: a\nA: 2", p.system)


class TestFewshot(unittest.TestCase):
    def test_turns(self):
        p = make("own_answers_fewshot", k=2, select="first")(person(), Q["ncars"])
        self.assertEqual(p.meta["shown"], ["car", "bus"])
        self.assertEqual([r for r, _ in p.history], ["user", "assistant", "user", "assistant"])
        self.assertTrue(p.history[0][1].startswith("Do you own a car?"))
        self.assertEqual(json.loads(p.history[1][1]), {"answer": "yes"})
        self.assertIn("Interviewer: Do you own a car?", p.flatten())

    def test_select(self):
        f = lambda **kw: make("own_answers_fewshot", k=2, **kw)(person(), Q["ncars"]).meta["shown"]  # noqa: E731
        self.assertEqual(f(select="recent"), ["pets", "garage"])
        self.assertEqual(f(select="recent", order="reverse"), ["garage", "pets"])
        self.assertEqual(f(seed=3), f(seed=3))
        self.assertEqual(len(f(seed=3)), 2)
        seen = {tuple(f(seed=s)) for s in range(12)}
        self.assertGreater(len(seen), 1)                               # the seed matters

    def test_random_pick_stable_across_targets(self):
        inj = make("own_answers_fewshot", k=2, seed=5)
        a = inj(person(), Q["ncars"]).meta["shown"]
        b = inj(person(), Q[a[0]]).meta["shown"]                       # target one of the picked items
        self.assertNotIn(a[0], b)
        self.assertIn(a[1], b)                                         # the other pick survives


class TestRag(unittest.TestCase):
    def test_scorers_rank_car_questions_first(self):
        for scorer in oa.SCORERS:
            p = make("own_answers_rag", k=2, scorer=scorer)(person(), Q["ncars"])
            self.assertEqual(set(p.meta["shown"]), {"car", "garage"}, scorer)
            self.assertEqual(p.meta["scores"], sorted(p.meta["scores"], reverse=True))

    def test_order_ablation(self):
        f = lambda o: make("own_answers_rag", k=3, order=o)(person(), Q["ncars"]).meta["shown"]  # noqa: E731
        self.assertEqual(f("asc"), f("desc")[::-1])
        bank_ids = [b["id"] for b in BANK]
        self.assertEqual(f("bank"), sorted(f("desc"), key=bank_ids.index))
        top, second = f("desc")[:2]
        p = make("own_answers_rag", k=2, order="asc")(person(), Q["ncars"])
        self.assertGreater(p.system.index(Q[top]["text"]), p.system.index(Q[second]["text"]))  # best one last

    def test_k_and_ties_deterministic(self):
        inj = make("own_answers_rag", k=10)
        p = inj(person(), {"id": "new", "text": "Zebra quantum?", "options": ["yes", "no"]})
        self.assertEqual(p.meta["shown"], [b["id"] for b in BANK])    # all zero: bank order
        self.assertEqual(inj(person(), Q["ncars"]).meta, inj(person(), Q["ncars"]).meta)

    def test_scorer_functions(self):
        docs = ["red car", "blue bus", "car car garage"]
        for fn in oa.SCORERS.values():
            s = fn("car", docs)
            self.assertEqual(len(s), 3)
            self.assertEqual(s[1], 0.0)
            self.assertGreater(s[0], 0.0)
        self.assertAlmostEqual(oa.score_tfidf("red car", ["red car"])[0], 1.0)
        self.assertEqual(oa.score_bm25("car", []), [])
        self.assertEqual(oa.tokens("The Cars, a bus!"), ["car", "bus"])

    def test_tokens_non_ascii(self):
        self.assertEqual(oa.tokens("Araç sahibi misiniz? Şehirde yaşıyor"), ["arac", "sahibi", "misiniz", "sehirde",
                                                                            "yasiyor"])
        self.assertEqual(oa.tokens("İstanbul"), oa.tokens("istanbul"))
        self.assertEqual(oa.tokens("IŞIK"), oa.tokens("ışık"))
        bank = [{"id": "a", "text": "Araç sahibi misiniz?"}, {"id": "b", "text": "Evcil hayvanınız var mı?"}]
        p = pi.get_injector("own_answers_rag", questions=bank, k=1)(
            {"id": "u", "answers": {"a": "evet", "b": "hayır"}}, {"id": "t", "text": "Kaç aracınız var? Araç"})
        self.assertEqual(p.meta["shown"], ["a"])
        self.assertGreater(p.meta["scores"][0], 0)

    def test_tfidf_ignores_unseen_query_terms(self):
        docs = ["red car", "blue bus"]
        self.assertEqual(oa.score_tfidf("red car zebra", docs), oa.score_tfidf("red car", docs))
        self.assertAlmostEqual(oa.score_tfidf("red car zebra", ["red car"])[0], 1.0)

    def test_tfidf_matches_sklearn(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            self.skipTest("scikit-learn not installed")
        docs = [b["text"] for b in BANK]
        query = "Does your household own cars or a garage?"
        vec = TfidfVectorizer(tokenizer=oa.tokens, lowercase=False, token_pattern=None).fit(docs)
        want = (vec.transform(docs) @ vec.transform([query]).T).toarray().ravel()
        for got, w in zip(oa.score_tfidf(query, docs), want):
            self.assertAlmostEqual(got, w, places=9)


class TestCompare(unittest.TestCase):
    def test_compare_with_oracle(self):
        """An oracle that copies the shown 'own a car' answer: own_answers arms beat no_persona and swap."""
        shown = re.compile(r'own a car\?(?:\nOptions: yes / no)?\nA: (\w+)|own a car\?.*?You: \{"answer": "(\w+)"\}', re.S)

        def oracle(system, user):
            m = shown.search(system + "\n" + user.split("Now:")[0])
            own = (m.group(1) or m.group(2)) if m else "no"
            return json.dumps({"answer": 2 if own == "yes" else 0})
        people = [person(i) for i in range(6)]
        arms = [("own_answers_all", {"questions": BANK}), ("own_answers_rag", {"questions": BANK, "k": 2}),
                ("own_answers_fewshot", {"questions": BANK, "k": 4, "select": "first"})]
        llm = FakeLLM(oracle)
        res = pi.compare_injectors(llm, people, [Q["ncars"]], "answers", arms, B=200)
        t = res.table.set_index("arm")
        for arm in ("own_answers_all(questions=5q)", "own_answers_rag(k=2,questions=5q)",
                    "own_answers_fewshot(k=4,questions=5q,select=first)"):
            self.assertAlmostEqual(t.loc[arm, "score"], 1.0, msg=arm)
            self.assertGreater(t.loc[arm, "diff_vs_swap"], 0, msg=arm)
        self.assertAlmostEqual(t.loc["no_persona", "score"], 0.5)
        for system, user in llm.calls:                                 # no prompt carries the target's answer
            self.assertNotIn("household own?\nA:", system + user)


class TestBm25Deterministic(unittest.TestCase):
    def test_scores_do_not_depend_on_hash_seed(self):
        """Regression: score_bm25 summed over a set, so float scores (and the picked answers) varied per PYTHONHASHSEED."""
        import os, subprocess, sys
        code = ("import random; from simengine.persona.inject.own_answers import score_bm25\n"
                "r = random.Random(0); W = 'a b c d e f g h i j k l m n o p q r s'.split(); out = []\n"
                "for _ in range(200):\n"
                "    docs = [' '.join(r.choice(W) for _ in range(r.randint(2, 9))) for _ in range(30)]\n"
                "    out.append(score_bm25(' '.join(r.sample(W, 6)), docs))\n"
                "print(repr(out))")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        runs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=root,
                               env={**os.environ, "PYTHONHASHSEED": str(seed), "PYTHONPATH": root}).stdout
                for seed in (1, 2, 3)}
        self.assertEqual(len(runs), 1)


if __name__ == "__main__":
    unittest.main()

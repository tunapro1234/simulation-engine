import json
import os
import tempfile
import unittest

import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM
from simengine.persona.inject import finetune as ft

OPTS = ["yes", "no"]
QUESTIONS = [{"id": "q0", "text": "Do you own a car?", "options": OPTS},
             {"id": "q1", "text": "Do you take the bus?", "options": OPTS},
             {"id": "q2", "text": "How many trips per week?", "min": 0, "max": 20}]
TEXTS = {"q0": "Do you own a car?", "q1": "Do you take the bus?", "q2": "How many trips per week?"}


def people(n=6):
    return [{"id": f"u{i}", "fields": {"region": "north" if i % 2 else "south", "openness": i / 10},
             "answers": {"q0": "yes" if i % 2 else "no", "q1": "TARGET1" if i == 0 else "no", "q2": i}}
            for i in range(n)]


class TestRegistry(unittest.TestCase):
    def test_family_listed_with_arxiv(self):
        rows = pi.list_injectors("finetune")
        self.assertEqual({r["name"] for r in rows},
                         {"subpop_lora", "centaur_qlora", "caa_steer", "persona_vector_steer", "soft_prompt"})
        for r in rows:
            self.assertIn("arXiv:", r["paper"])
            self.assertIn("Not measured in simengine", r["description"])
        self.assertNotIn("finetune", pi.IMPORT_ERRORS)


class TestInjectors(unittest.TestCase):
    def test_missing_fields_raise_with_needs(self):
        for name, kw in [("subpop_lora", {}), ("caa_steer", {"traits": ["openness"]}),
                         ("persona_vector_steer", {"traits": ["openness"]}), ("soft_prompt", {})]:
            with self.assertRaises(pi.MissingPersonField) as cm:
                pi.get_injector(name, **kw)({"id": "x"}, QUESTIONS[0])
            self.assertIn("fields", str(cm.exception))
        with self.assertRaises(pi.MissingPersonField) as cm:
            pi.get_injector("centaur_qlora")({"id": "x", "fields": {"a": 1}}, QUESTIONS[0])
        self.assertIn("answers", str(cm.exception))
        with self.assertRaises(pi.MissingPersonField) as cm:    # named key absent
            pi.get_injector("subpop_lora", keys=["age"])(people()[0], QUESTIONS[0])
        self.assertIn("age", str(cm.exception))
        self.assertIn("needs=['fields']", str(cm.exception))
        with self.assertRaises(pi.MissingPersonField):
            pi.get_injector("caa_steer", traits=["age"])(people()[0], QUESTIONS[0])

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            pi.get_injector("subpop_lora", style="essay")
        with self.assertRaises(ValueError):
            pi.get_injector("caa_steer")
        with self.assertRaises(ValueError):
            pi.get_injector("soft_prompt", n_tokens=0)
        with self.assertRaises(ValueError):        # categorical value without levels
            pi.get_injector("caa_steer", traits=["region"])(people()[0], QUESTIONS[0])

    def test_subpop_styles_and_letters(self):
        p = people()[1]
        port = pi.get_injector("subpop_lora")(p, QUESTIONS[0])
        self.assertEqual(port.system, "")
        self.assertIn("as if your region is north and your openness is 0.1.", port.user)
        self.assertTrue(port.user.endswith("A. yes\nB. no\nAnswer:"))
        qa = pi.get_injector("subpop_lora", style="qa", keys=["region"])(p, QUESTIONS[0])
        self.assertIn("Question: What is your region?\nAnswer: north", qa.user)
        self.assertNotIn("openness", qa.user)
        bio = pi.get_injector("subpop_lora", style="bio")(p, QUESTIONS[0])
        self.assertIn("My region is north.", bio.user)
        num = pi.get_injector("subpop_lora")(p, QUESTIONS[2])
        self.assertIn('"answer"', num.user)

    def test_centaur_no_target_leak(self):
        p = people()[0]
        for chat in (True, False):
            inj = pi.get_injector("centaur_qlora", texts=TEXTS, chat=chat)
            pr = inj(p, QUESTIONS[1])
            flat = pr.system + pr.flatten()
            self.assertNotIn("TARGET1", flat)
            self.assertIn("<<no>>", flat)           # q0 answer is shown
            self.assertIn("Do you own a car?", flat)
            self.assertTrue(flat.rstrip().endswith("You answer") if not chat else "take the bus" in pr.user)
        pr = pi.get_injector("centaur_qlora")(p, {**QUESTIONS[0], "exclude": ["q1"]})
        self.assertNotIn("TARGET1", pr.flatten())
        self.assertEqual(pr.meta["n_context"], 1)
        self.assertEqual([r for r, _ in pr.history], ["user", "assistant"])

    def test_centaur_deterministic_with_seed(self):
        p = {"id": "z", "answers": {f"k{i}": str(i) for i in range(12)}}
        q = {"id": "t", "text": "Target?", "options": OPTS}
        a = pi.get_injector("centaur_qlora", seed=7)(p, q)
        b = pi.get_injector("centaur_qlora", seed=7)(p, q)
        c = pi.get_injector("centaur_qlora", seed=8)(p, q)
        plain = pi.get_injector("centaur_qlora")(p, q)
        self.assertEqual(a.history, b.history)
        self.assertNotEqual(a.history, c.history)
        self.assertEqual(plain.history[0], ("user", "k0"))
        self.assertEqual(sorted(a.history), sorted(plain.history))
        short = pi.get_injector("centaur_qlora", max_turns=2)(p, q)
        self.assertEqual(len(short.history), 4)
        self.assertEqual(pi.get_injector("centaur_qlora", max_turns=0)(p, q).history, [])

    def test_steering_meta_has_no_text_persona(self):
        p = people()[3]
        caa = pi.get_injector("caa_steer", traits=["openness"], center=0.1, multiplier=2.0)(p, QUESTIONS[0])
        self.assertEqual(caa.system, "")
        self.assertEqual(caa.user, pi.get_injector("no_persona")(p, QUESTIONS[0]).user)
        self.assertEqual(caa.meta["steer"]["method"], "caa")
        self.assertAlmostEqual(caa.meta["steer"]["coefs"]["openness"], 0.4)
        pv = pi.get_injector("persona_vector_steer", traits={"north": "region"},
                             levels={"north": 1, "south": -1}, layer=20)(p, QUESTIONS[0])
        self.assertEqual(pv.meta["steer"], {"method": "persona_vector", "layer": 20,
                                            "positions": "every_decode_step", "coefs": {"north": 1.0}})
        sp = pi.get_injector("soft_prompt", keys=["region"], n_tokens=8)(p, QUESTIONS[0])
        self.assertEqual(sp.meta["soft_prompt"], "region=north")
        self.assertEqual(sp.meta["n_virtual_tokens"], 8)

    def test_steer_from_meta_combines(self):
        meta = {"steer": {"coefs": {"a": 2.0, "b": -1.0}, "layer": 5}}
        self.assertEqual(ft.steer_from_meta(meta, {"a": 1.5, "b": 1.0}), {5: 2.0})
        with self.assertRaises(KeyError):
            ft.steer_from_meta(meta, {"a": 1.0})
        with self.assertRaises(ValueError):
            ft.steer_from_meta({"steer": {"coefs": {"a": 1.0}, "layer": None}}, {"a": 1.0})

    def test_labels_compact_and_unique(self):
        self.assertEqual(pi.get_injector("centaur_qlora", texts=TEXTS).label, "centaur_qlora(texts=<dict:3>)")
        self.assertEqual(pi.get_injector("subpop_lora", style="qa").label, "subpop_lora(style=qa)")

    def test_parse_finetuned(self):
        q = QUESTIONS[0]
        self.assertEqual(ft.parse_finetuned("<<no>>", q), "no")
        self.assertEqual(ft.parse_finetuned(" B", q), "no")
        self.assertEqual(ft.parse_finetuned("(A)", q), "yes")
        self.assertEqual(ft.parse_finetuned('{"answer": "yes"}', q), "yes")
        self.assertIsNone(ft.parse_finetuned("C", q))
        self.assertIsNone(ft.parse_finetuned("I would say yes", q))
        self.assertIsNone(ft.parse_finetuned(None, q))
        self.assertEqual(ft.parse_finetuned("<<7>>", QUESTIONS[2]), 7)

    def test_compare_runs_with_fake_llm(self):
        llm = FakeLLM(lambda s, u: "<<yes>>")
        res = pi.compare_injectors(llm, people(), QUESTIONS[:2], "answers", ["centaur_qlora", "subpop_lora"],
                                   parse=ft.parse_finetuned, B=50)
        t = res.table.set_index("arm")
        self.assertEqual(t.loc["centaur_qlora", "parsed"], 12)
        at = {c: pr for c, pr in zip(res.cells, res.prompts["centaur_qlora"])}
        self.assertNotIn("TARGET1", at[("u0", "q1")].flatten())   # the target cell never sees its own answer
        self.assertIn("<<TARGET1>>", at[("u0", "q0")].flatten())  # other cells of u0 may


class TestDatasetBuilders(unittest.TestCase):
    def test_chat_rows_targets_and_no_leak(self):
        rows = ft.chat_rows(people(), QUESTIONS, "centaur_qlora", texts=TEXTS)
        self.assertEqual(len(rows), 18)
        r = next(r for r in rows if r["person"] == "u0" and r["question"] == "q1")
        self.assertEqual(r["messages"][-1], {"role": "assistant", "content": "<<TARGET1>>"})
        self.assertNotIn("TARGET1", json.dumps(r["messages"][:-1]))
        self.assertEqual(r["messages"][0]["role"], "system")
        self.assertEqual([m["role"] for m in r["messages"][1:]], ["user", "assistant"] * 2 + ["user", "assistant"])
        sub = ft.chat_rows(people(), QUESTIONS[::2], "subpop_lora")             # q0, q2 (u0's q1 is no option)
        self.assertEqual(sub[0]["messages"][-1]["content"], "B")                 # u0 q0 = no
        self.assertEqual(json.loads(sub[1]["messages"][-1]["content"]), {"answer": 0})
        for row in sub:                                                          # SubPOP shows no answers at all
            self.assertNotIn("TARGET1", json.dumps(row["messages"]))
        plain = ft.chat_rows(people(), QUESTIONS[:1], "plain_fields")         # other families: JSON target
        self.assertEqual(json.loads(plain[0]["messages"][-1]["content"]), {"answer": "no"})

    def test_chat_rows_truth_mapping_flatten_and_meta(self):
        truth = {("u1", "q0"): "yes"}
        rows = ft.chat_rows(people(), QUESTIONS, "centaur_qlora", truth, flatten=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual([m["role"] for m in rows[0]["messages"]], ["system", "user", "assistant"])
        self.assertIn("Earlier in this conversation", rows[0]["messages"][1]["content"])
        steer = ft.chat_rows(people(2), QUESTIONS[:1], "soft_prompt", keys=["region"])
        self.assertEqual(steer[1]["meta"]["soft_prompt"], "region=north")
        with self.assertRaises(ValueError):          # answer not among the options
            ft.chat_rows([{"id": "a", "fields": {"r": 1}, "answers": {"q0": "maybe"}}], QUESTIONS[:1], "subpop_lora")

    def test_chat_rows_deterministic(self):
        a = ft.chat_rows(people(), QUESTIONS, "centaur_qlora", seed=3)
        b = ft.chat_rows(people(), QUESTIONS, "centaur_qlora", seed=3)
        self.assertEqual(a, b)

    def test_distribution_rows(self):
        groups = [{"id": "g_north", "fields": {"region": "north"}}, {"id": "g_south", "fields": {"region": "south"}}]
        dist = {("g_north", "q0"): {"yes": 30, "no": 10}, ("g_south", "q0"): {"no": 1}}
        rows = ft.distribution_rows(groups, QUESTIONS, dist)
        self.assertEqual(rows[0]["target_dist"], {"A": 0.75, "B": 0.25})
        self.assertEqual(rows[1]["target_dist"], {"A": 0.0, "B": 1.0})
        self.assertEqual(rows[0]["messages"][-1]["role"], "user")
        self.assertTrue(rows[0]["messages"][-1]["content"].endswith("Answer:"))
        for bad in ({("g_north", "q0"): {"maybe": 1}}, {("g_north", "q0"): {"yes": -1, "no": 2}},
                    {("g_north", "q0"): {"yes": 0}}, {("g_north", "q2"): {"1": 1}}):
            with self.assertRaises(ValueError):
                ft.distribution_rows(groups, QUESTIONS, bad)
        with self.assertRaises(KeyError):
            ft.distribution_rows(groups, QUESTIONS, {("nobody", "q0"): {"yes": 1}})

    def test_caa_pairs(self):
        qs = QUESTIONS + [{"id": "q3", "text": "Pick", "options": ["a", "b", "c"]}]
        pairs = ft.caa_pairs(qs, {"q0": "yes", "q1": "no", "q3": "c"}, seed=0)
        self.assertEqual(len(pairs), 4)
        for p in pairs:
            self.assertIn("Choices:\n(A) ", p["prompt"])
            self.assertEqual({p["positive"], p["negative"]}, {"(A)", "(B)"})
            pos_text = p["prompt"].split(p["positive"] + " ")[1].split("\n")[0]
            self.assertEqual(pos_text, {"q0": "yes", "q1": "no", "q3": "c"}[p["question"]])
        self.assertEqual(pairs, ft.caa_pairs(qs, {"q0": "yes", "q1": "no", "q3": "c"}, seed=0))
        many = ft.caa_pairs([{"id": f"x{i}", "text": "t", "options": OPTS} for i in range(40)],
                            {f"x{i}": "yes" for i in range(40)}, seed=1)
        self.assertEqual({p["positive"] for p in many}, {"(A)", "(B)"})      # trait letter is balanced
        with self.assertRaises(ValueError):
            ft.caa_pairs(QUESTIONS, {"q0": "perhaps"})

    def test_persona_vector_pairs_and_filter(self):
        rows = ft.persona_vector_pairs(QUESTIONS[:2], ["Be curious."], ["Be incurious."], rollouts=3)
        self.assertEqual(len(rows), 2 * 1 * 3 * 2)
        self.assertEqual({r["side"] for r in rows}, {"pos", "neg"})
        with self.assertRaises(ValueError):
            ft.persona_vector_pairs(QUESTIONS, ["a"], [], rollouts=1)
        scored = [{"side": "pos", "score": 80}, {"side": "pos", "score": 40}, {"side": "neg", "score": 10},
                  {"side": "neg", "score": 60}, {"side": "pos"}]
        self.assertEqual(ft.judge_filter(scored), [scored[0], scored[2]])

    def test_holdout_and_jsonl_roundtrip(self):
        rows = ft.chat_rows(people(10), QUESTIONS[::2], "subpop_lora")
        train, test = ft.holdout_split(rows, frac=0.2, seed=4)
        self.assertEqual(len({r["person"] for r in test}), 2)
        self.assertFalse({r["person"] for r in train} & {r["person"] for r in test})
        self.assertEqual(ft.holdout_split(rows, frac=0.2, seed=4), (train, test))
        with self.assertRaises(ValueError):
            ft.holdout_split(rows, frac=1.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rows.jsonl")
            self.assertEqual(ft.write_jsonl(train, path), len(train))
            self.assertEqual(ft.read_jsonl(path), train)

    def test_recipe(self):
        c = ft.recipe("centaur")
        self.assertEqual((c["r"], c["target"], c["epochs"], c["lr"]), (8, "all-linear", 1, 5e-5))
        # one row per forward pass, so the accumulation alone reaches the paper's effective batch
        self.assertEqual((c["fp16"], c["bf16"], c["micro_batch"], c["grad_accum"]), (True, False, 1, 32))
        s = ft.recipe("subpop")
        self.assertEqual((s["alpha"], s["target"], s["loss"], s["grad_accum"]),
                         (32, ["q_proj", "v_proj"], "forward_kl", 256))
        with self.assertRaises(ValueError):
            ft.recipe("nope")
        with self.assertRaises(ValueError):        # micro batching is not implemented: refused before any import
            ft.train_colab_t4("missing.jsonl", "out", ft.recipe("centaur", micro_batch=4))

    def test_tokenize_chat_masks_and_keeps_the_target(self):
        class Tok:                                 # one token per character, BatchEncoding-like return
            def apply_chat_template(self, messages, tokenize=True, **kw):
                return {"input_ids": [ord(c) for m in messages for c in f"[{m['role'][0]}]{m['content']}"]}
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "<<a>>"}, {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "<<b>>"}]
        ids, labels = ft._tokenize_chat(Tok(), msgs, max_len=1000)
        self.assertEqual(len(ids), len(labels))
        shown = "".join(chr(t) for t in labels if t != -100)
        self.assertEqual(shown, "[a]<<a>>[a]<<b>>")          # loss only on assistant turns
        ids, labels = ft._tokenize_chat(Tok(), msgs, max_len=8)
        self.assertEqual("".join(chr(t) for t in ids), "[a]<<b>>")   # truncation keeps the final target


class TestLazyDeps(unittest.TestCase):
    def test_import_error_message_without_torch(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError) as cm:
                ft.train_colab_t4("missing.jsonl", "out")
            self.assertIn("torch", str(cm.exception))
            with self.assertRaises(ImportError):
                ft.steering_vector(None, None, [], [])
            return
        self.skipTest("torch installed; open-model helpers are not exercised in tests")


if __name__ == "__main__":
    unittest.main()

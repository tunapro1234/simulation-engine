"""simengine.games library: GAME contract, registry, render, answer parser."""
import string
import unittest

import simengine.games as gm
from simengine.games import library as L


class TestContract(unittest.TestCase):
    def test_every_game_passes_contract(self):
        self.assertEqual(len(gm.GAMES), 15)
        for name, g in gm.GAMES.items():
            with self.subTest(game=name):
                self.assertEqual(gm.check_game(g), [])
                self.assertEqual(g["game"], name)
                self.assertTrue(g["references"])
                for k in L.REQUIRED_KEYS:
                    self.assertIn(k, g)

    def test_placeholders_filled_by_every_condition(self):
        for g in gm.GAMES.values():
            fields = {f for _, f, _, _ in string.Formatter().parse(g["text"]) if f}
            for c in g["conditions"]:
                self.assertTrue(fields <= set(c) - {"id"}, (g["game"], c["id"], fields - set(c)))
                self.assertNotIn("{", gm.render(g["game"], c["id"]))

    def test_split_and_counts(self):
        self.assertEqual(len(gm.list_games(holdout=False)), 10)
        self.assertEqual(set(gm.list_games(holdout=True)),
                         {"beauty_contest", "pg_punishment", "third_party_punishment", "who_to_punish",
                          "ertan_vignette"})
        self.assertEqual(len(gm.list_games()), 15)

    def test_ertan_vignette_profiles(self):
        g = gm.get_game("ertan_vignette")
        self.assertEqual([c["id"] for c in g["conditions"]], ["all_cooperate", "mixed", "one_freerider"])
        self.assertIn("15, 5, 18, 2", gm.render(g, "mixed"))
        self.assertIn("17, 18, 16, 2", gm.render(g, "one_freerider"))
        self.assertIn("18, 17, 19, 16", gm.render(g, "all_cooperate"))
        self.assertEqual((g["min"], g["max"]), (0, 5))

    def test_check_game_catches_problems(self):
        g = gm.get_game("dictator")
        g["text"] = "You have ${pot} and {missing}."
        self.assertTrue(any("missing" in p for p in gm.check_game(g)))
        g = gm.get_game("dictator")
        del g["references"]
        self.assertTrue(any("references" in p for p in gm.check_game(g)))
        g = gm.get_game("holt_laury")
        g["options"] = []
        self.assertTrue(gm.check_game(g))
        g = gm.get_game("trust_sender")
        del g["belief"]["max"]
        self.assertTrue(any("belief" in p for p in gm.check_game(g)))

    def test_get_game_is_a_copy(self):
        g = gm.get_game("dictator")
        g["conditions"][0]["pot"] = 1
        self.assertEqual(gm.GAMES["dictator"]["conditions"][0]["pot"], 100)

    def test_per_condition_bounds(self):
        self.assertEqual(gm.bounds("trust_returner", "low"), (0, 30))
        self.assertEqual(gm.bounds("trust_returner", "high"), (0, 150))
        self.assertEqual(gm.bounds("pg_punishment", "contribute"), (0, 20))
        self.assertEqual(gm.bounds("pg_punishment", "punish_freerider"), (0, 10))


class TestRender(unittest.TestCase):
    def test_deterministic_and_exact(self):
        for g in gm.GAMES.values():
            for c in g["conditions"]:
                a, b = gm.render(g["game"], c["id"]), gm.render(g, c)
                self.assertEqual(a, b)
                self.assertEqual(a, g["text"].format(**{k: v for k, v in c.items() if k != "id"}))
        self.assertIn("$0.50", gm.render("public_goods_strategy", "cond5"))
        self.assertIn("$5 each", gm.render("public_goods_strategy", "cond5"))

    def test_belief_and_wording(self):
        self.assertIsNone(gm.render_belief("dictator", "base"))
        self.assertIn("predict", gm.render_belief("trust_sender", "base"))
        with self.assertRaises(ValueError):
            gm.render("dictator", "base", wording_id=1)
        g = gm.get_game("dictator")
        g["wordings"] = {1: {"text": "Pot: {pot}. Give?"}}
        self.assertEqual(gm.render(g, "base", 1), "Pot: 100. Give?")
        with self.assertRaises(KeyError):
            gm.render("dictator", "nope")


class TestParseAnswer(unittest.TestCase):
    NUM = {"kind": "numerical", "min": 0, "max": 100}
    MC = {"kind": "mc", "options": ["Accept", "Reject"]}

    def test_numbers(self):
        self.assertEqual(gm.parse_answer('{"answer": 30}', self.NUM), (30, None))
        self.assertEqual(gm.parse_answer('```json\n{"answer": "30"}\n```', self.NUM), (30, None))
        self.assertEqual(gm.parse_answer("42", self.NUM), (42, None))
        self.assertEqual(gm.parse_answer("$12.5", self.NUM), (12.5, None))
        self.assertEqual(gm.parse_answer('{"amount": 7}', self.NUM), (7, None))

    def test_no_salvage(self):
        v, err = gm.parse_answer("I would give 30 dollars to be fair.", self.NUM)
        self.assertIsNone(v)
        self.assertTrue(err)
        v, err = gm.parse_answer("no, yes, I accept", self.MC)
        self.assertIsNone(v)
        self.assertIsNone(gm.parse_answer("", self.NUM)[0])
        self.assertIsNone(gm.parse_answer(None, self.NUM)[0])

    def test_out_of_range_kept_with_error(self):
        v, err = gm.parse_answer('{"answer": 150}', self.NUM)
        self.assertEqual(v, 150)
        self.assertIn("outside", err)

    def test_mc(self):
        self.assertEqual(gm.parse_answer('{"answer": "reject"}', self.MC), ("Reject", None))
        self.assertEqual(gm.parse_answer("Accept.", self.MC), ("Accept", None))
        self.assertIsNone(gm.parse_answer('{"answer": "Maybe"}', self.MC)[0])
        self.assertIsNone(gm.parse_answer('{"answer": 1}', self.MC)[0])

    def test_instruction(self):
        self.assertIn('"Accept", "Reject"', gm.instruction(self.MC))
        self.assertIn("from 0 to 100", gm.instruction(self.NUM))


if __name__ == "__main__":
    unittest.main()

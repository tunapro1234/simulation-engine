"""simengine.sim parsers: JSON extraction, probability normalisation, keyed item probabilities."""
import unittest

import numpy as np

from simengine.sim.parse import (normalize_probs, parse_distribution, parse_item_probs, parse_json,
                                 parse_text_fields, unwrap)


class TestParseJson(unittest.TestCase):
    def test_plain_and_fenced(self):
        self.assertEqual(parse_json('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json('Sure!\n```json\n[{"id": "p1"}]\n```\nDone.'), [{"id": "p1"}])

    def test_prose_around_and_trailing_commas(self):
        self.assertEqual(parse_json('Here it is: {"1": 40, "2": 60,} hope that helps'), {"1": 40, "2": 60})

    def test_skips_bracket_in_prose(self):
        self.assertEqual(parse_json('Options [see below]: {"x": 2}'), {"x": 2})

    def test_truncated_array_salvages_complete_objects(self):
        got = parse_json('[{"id": "p1", "v": 1}, {"id": "p2", "v": 2}, {"id": "p3", "v"')
        self.assertEqual([x["id"] for x in got], ["p1", "p2"])

    def test_garbage(self):
        for t in (None, "", "no json here", 42):
            self.assertIsNone(parse_json(t))

    def test_unwrap(self):
        self.assertEqual(unwrap({"probabilities": {"A": 1, "B": 2}}, ["A", "B"]), {"A": 1, "B": 2})
        self.assertEqual(unwrap({"A": 1}, ["A"]), {"A": 1})


class TestNormalizeProbs(unittest.TestCase):
    def test_dict_keys_forms(self):
        for d in ({"1": 20, "2": 30, "3": 50}, {"option 1": 20, "option 2": 30, "option 3": 50},
                  {"a": 20, "b": 30, "c": 50}):
            np.testing.assert_allclose(normalize_probs(d, 3), [0.2, 0.3, 0.5])

    def test_label_keys_and_percent_strings(self):
        p = normalize_probs({"Yes": "70%", "No": "30%"}, 2, ["Yes", "No"])
        np.testing.assert_allclose(p, [0.7, 0.3])

    def test_list_renormalised(self):
        np.testing.assert_allclose(normalize_probs([1, 1, 2], 3), [0.25, 0.25, 0.5])

    def test_invalid(self):
        self.assertIsNone(normalize_probs([0, 0, 0], 3))
        self.assertIsNone(normalize_probs([50, -10, 60], 3))
        self.assertIsNone(normalize_probs([50, 50], 3))
        self.assertIsNone(normalize_probs({"1": 50, "2": 50}, 3))   # option 3 missing
        self.assertIsNone(normalize_probs("x", 3))

    def test_parse_distribution(self):
        np.testing.assert_allclose(parse_distribution('```{"1": 10, "2": 90}```', 2), [0.1, 0.9])
        self.assertIsNone(parse_distribution("I cannot answer that.", 2))


class TestItemProbs(unittest.TestCase):
    def test_case_insensitive_percent_and_wrapper(self):
        got = parse_item_probs('{"estimates": {"books": "40%", "TEA": 5}}', ["BOOKS", "TEA"])
        self.assertAlmostEqual(got["BOOKS"], 0.40)
        self.assertAlmostEqual(got["TEA"], 0.05)

    def test_strict_raises(self):
        with self.assertRaises(ValueError):
            parse_item_probs('{"BOOKS": 40}', ["BOOKS", "TEA"])
        with self.assertRaises(ValueError):
            parse_item_probs('{"BOOKS": 140, "TEA": 1}', ["BOOKS", "TEA"])
        with self.assertRaises(ValueError):
            parse_item_probs("nope", ["BOOKS"])

    def test_lenient(self):
        self.assertEqual(parse_item_probs('{"BOOKS": 40}', ["BOOKS", "TEA"], strict=False), {"BOOKS": 0.4})

    def test_text_fields(self):
        self.assertEqual(parse_text_fields('{"a": " I bought one. "}', ["A"]), {"A": "I bought one."})
        with self.assertRaises(ValueError):
            parse_text_fields('{"A": ""}', ["A"])


if __name__ == "__main__":
    unittest.main()

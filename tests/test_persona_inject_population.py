import json
import random
import unittest

import numpy as np

import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM
from simengine.persona.inject import population as pop

OPTS = ["yes", "no", "unsure"]
Q = {"id": "q0", "text": "Would you buy an electric car next year?", "options": OPTS}
QS = [Q,
      {"id": "q1", "text": "Would you buy a used electric car?", "options": OPTS},
      {"id": "q2", "text": "Do you cycle to work?", "options": ["yes", "no"]},
      {"id": "q3", "text": "Would you buy a car next year?", "options": OPTS}]


def people(n=12):
    out = []
    for i in range(n):
        region = "north" if i % 2 else "south"
        out.append({"id": f"u{i}", "fields": {"region": region, "age": 20 + i},
                    "answers": {"q0": "yes" if region == "north" else "no", "q1": "unsure" if i % 3 else "yes",
                                "q2": "yes", "q3": "no"}})
    return out


def oracle(system, user):
    """north -> 80% yes, south -> 20% yes; modal answer matches."""
    north = "region: north" in system
    if '"samples"' in user:
        s = ["yes"] * (8 if north else 2) + ["no"] * (2 if north else 8)
        return json.dumps({"samples": s, "answer": "yes" if north else "no"})
    if '"distribution"' in user:
        d = {"yes": 80, "no": 20, "unsure": 0} if north else {"yes": 20, "no": 80, "unsure": 0}
        return json.dumps({"distribution": d, "answer": max(d, key=d.get)})
    return json.dumps({"answer": "yes" if north else "no"})


class TestRegistry(unittest.TestCase):
    def test_listed_with_arxiv(self):
        rows = {r["name"]: r for r in pi.list_injectors("population")}
        self.assertEqual(set(rows), {"cell_verbalized", "cell_sequence", "cell_sample", "cell_fewshot"})
        self.assertTrue(all("arXiv:" in r["paper"] for r in rows.values()))
        self.assertTrue(all(r["needs"] == ["fields"] for r in rows.values()))
        self.assertNotIn("population", pi.IMPORT_ERRORS)          # other families are not this test's concern

    def test_labels(self):
        self.assertEqual(pi.get_injector("cell_sequence", n=10).label, "cell_sequence(n=10)")
        f = pi.get_injector("cell_fewshot", people=people(), questions=QS, k=2)
        self.assertEqual(f.label, "cell_fewshot(k=2)")          # no rows in the arm name
        self.assertEqual(pi.get_injector("cell_fewshot").label, "cell_fewshot")


class TestRender(unittest.TestCase):
    def test_missing_fields_and_keys(self):
        for name in ("cell_verbalized", "cell_sequence", "cell_sample", "cell_fewshot"):
            with self.assertRaises(pi.MissingPersonField) as cm:
                pi.get_injector(name)({"id": "x", "region": "north"}, Q)
            self.assertIn("fields", str(cm.exception))
        with self.assertRaises(pi.MissingPersonField) as cm:
            pi.get_injector("cell_verbalized", keys=["region", "income"])({"id": "x", "fields": {"region": "n"}}, Q)
        self.assertIn("income", str(cm.exception))

    def test_choice_only_and_numeric_sample(self):
        num = {"id": "n", "text": "How many cars?", "min": 0, "max": 5}
        with self.assertRaises(ValueError):
            pi.get_injector("cell_verbalized")(people()[0], num)
        p = pi.get_injector("cell_sample")(people()[0], num)
        self.assertIn("single number from 0 to 5", p.user)
        with self.assertRaises(ValueError):
            pi.get_injector("cell_sequence", n=1)
        with self.assertRaises(ValueError):
            pi.get_injector("cell_sample", draws=0)

    def test_same_cell_same_prompt(self):
        ps = people()
        inj = pi.get_injector("cell_verbalized", keys=["region"])
        a, b, c = inj(ps[1], Q), inj(ps[3], Q), inj(ps[0], Q)
        self.assertEqual((a.system, a.user), (b.system, b.user))
        self.assertNotEqual(a.system, c.system)
        self.assertNotIn("age", a.system)
        self.assertEqual(a.meta["injector"], "cell_verbalized")
        self.assertIn("age: 21", pi.get_injector("cell_verbalized")(ps[1], Q).system)    # keys=None: all fields

    def test_cell_key_order_and_empty_keys(self):
        a = {"id": "a", "fields": {"region": "north", "age": 30}}
        b = {"id": "b", "fields": {"age": 30, "region": "north"}}
        self.assertEqual(pop.cell_key(a, None), pop.cell_key(b, None))
        inj = pi.get_injector("cell_verbalized")
        self.assertEqual(inj(a, Q).system, inj(b, Q).system)
        everyone = pi.get_injector("cell_verbalized", keys=[])
        self.assertEqual(everyone.label, "cell_verbalized(keys=[])")
        self.assertIn("the general population", everyone(a, Q).system)
        self.assertEqual(everyone(a, Q).system, everyone({"id": "c", "fields": {"region": "south"}}, Q).system)
        res = pop.simulate_population(FakeLLM(oracle), people(), Q, "cell_verbalized", keys=[])
        self.assertEqual((res.calls, res.n), (1, {"": 12}))

    def test_sample_deterministic_with_seed(self):
        ps = people(30)
        inj = pi.get_injector("cell_sample", keys=["region"], seed=1)
        d1 = [inj(p, Q).meta["draw"] for p in ps]
        self.assertEqual(d1, [pi.get_injector("cell_sample", keys=["region"], seed=1)(p, Q).meta["draw"] for p in ps])
        d2 = [pi.get_injector("cell_sample", keys=["region"], seed=2)(p, Q).meta["draw"] for p in ps]
        self.assertNotEqual(d1, d2)
        self.assertGreater(len(set(d1)), 5)
        self.assertTrue(all(1 <= d <= 30 for d in d1))
        self.assertEqual(len({inj(p, Q).system for p in ps}), len({(p["fields"]["region"], d) for p, d in zip(ps, d1)}))

    def test_no_answers_in_cell_prompts(self):
        for name in ("cell_verbalized", "cell_sequence", "cell_sample"):
            p = pi.get_injector(name)(people()[1], Q)
            self.assertNotIn("unsure\"", p.system)
            self.assertNotIn("q1", p.system + p.user)

    def test_fewshot_never_uses_target_or_exclude(self):
        ps = people()
        inj = pi.get_injector("cell_fewshot", people=ps, questions=QS, k=3, keys=["region"])
        p = inj(ps[1], Q)
        self.assertNotIn("q0", p.meta["examples"])
        self.assertEqual(p.user.count(Q["text"]), 1)                      # only as the final question
        self.assertIn("Observed shares in this group (n=6)", p.user)
        q = {**Q, "exclude": ["q1", "q3"]}
        p = inj(ps[1], q)
        self.assertEqual(p.meta["examples"], ["q2"])
        self.assertNotIn(QS[1]["text"], p.user)
        # target answers differ by cell; the q0 shares must not appear even when q0 is in the pool
        ex = inj.examples(ps[1], Q)
        self.assertTrue(all(str(e[0]["id"]) != "q0" for e in ex))
        self.assertEqual(inj(ps[1], Q).user, p0 := inj(ps[1], Q).user)    # deterministic
        self.assertTrue(p0)
        # the north cell answered q0 100% yes; that share line must not be shown for target q0
        self.assertNotIn(json.dumps({"yes": 100, "no": 0, "unsure": 0}), p0)
        # the example cache is keyed by target: q0 is a valid example once q3 is the target
        q3 = inj(ps[1], QS[3])
        self.assertIn("q0", q3.meta["examples"])
        self.assertNotIn("q3", q3.meta["examples"])
        self.assertIn(json.dumps({"yes": 100, "no": 0, "unsure": 0}), q3.user)
        self.assertEqual(inj(ps[1], Q).user, p0)                          # cache did not mix the targets

    def test_fewshot_similarity_then_diversity(self):
        ps = people()
        ex = pi.get_injector("cell_fewshot", people=ps, questions=QS, k=2, keys=["region"]).examples(ps[1], Q)
        ids = [str(e[0]["id"]) for e in ex]
        self.assertEqual(ids[0], "q3")                    # most word overlap with the target
        self.assertEqual(len(ids), 2)
        none = pi.get_injector("cell_fewshot", people=ps, questions=QS, min_n=99)(ps[1], Q)
        self.assertEqual(none.meta["examples"], [])
        self.assertNotIn("Observed shares", none.user)


class TestParse(unittest.TestCase):
    def test_parse_distribution(self):
        d = pop.parse_distribution('{"distribution": {"yes": 60, "no": 30, "unsure": 10}, "answer": "yes"}', Q)
        self.assertAlmostEqual(d["yes"], 0.6)
        d = pop.parse_distribution('```json\n{"distribution": {"1": 0.5, "2": 0.5, "3": 0}}\n```', Q)
        self.assertAlmostEqual(d["no"], 0.5)
        d = pop.parse_distribution('{"samples": ["Yes", "no", "no", "maybe"]}', Q)
        self.assertAlmostEqual(d["no"], 2 / 3)
        self.assertIsNone(pop.parse_distribution('{"distribution": {"yes": 60}}', Q))   # missing options
        self.assertIsNone(pop.parse_distribution("yes", Q))
        self.assertIsNone(pop.parse_distribution('{"samples": ["maybe"]}', Q))
        # numeric-coded options keyed by text: not shifted by the 1-based option-number lookup
        q = {"id": "n", "text": "Cars owned?", "options": ["0", "1", "2"]}
        self.assertEqual(pop.parse_distribution('{"distribution": {"0": 70, "1": 20, "2": 10}}', q),
                         {"0": 0.7, "1": 0.2, "2": 0.1})
        q = {"id": "l", "text": "Agree?", "options": ["5", "4", "3", "2", "1"]}
        d = pop.parse_distribution('{"distribution": {"1": 10, "2": 10, "3": 10, "4": 20, "5": 50}}', q)
        self.assertAlmostEqual(d["5"], 0.5)
        self.assertAlmostEqual(d["4"], 0.2)
        self.assertEqual(pop.parse_modal('{"distribution": {"yes": 10, "no": 30, "unsure": 60}}', Q), "unsure")
        self.assertEqual(pop.parse_modal('{"answer": "no"}', Q), "no")
        self.assertIsNone(pop.parse_modal("perhaps", Q))
        # the JSON keeps "answer", so the default parser also reads it
        self.assertEqual(pi.parse_option(oracle("region: north", '"distribution"'), Q), "yes")


class TestPopulation(unittest.TestCase):
    def test_one_call_per_cell_and_tvd(self):
        llm = FakeLLM(oracle)
        ps = people()
        res = pop.simulate_population(llm, ps, Q, "cell_verbalized", keys=["region"])
        self.assertEqual(res.calls, 2)
        self.assertEqual(len(llm.calls), 2)
        self.assertAlmostEqual(res.dist["region=north"]["yes"], 0.8)
        self.assertAlmostEqual(res.overall["yes"], 0.5)
        self.assertEqual(res.n, {"region=south": 6, "region=north": 6})
        truth = pop.empirical_distribution(ps, Q, keys=["region"])
        self.assertAlmostEqual(truth.dist["region=north"]["yes"], 1.0)
        self.assertAlmostEqual(pop.tvd(res.dist["region=north"], truth.dist["region=north"]), 0.2)
        seq = pop.simulate_population(FakeLLM(oracle), ps, Q, "cell_sequence", keys=["region"], n=10)
        self.assertAlmostEqual(seq.dist["region=south"]["no"], 0.8)

    def test_sample_counts_and_errors(self):
        def flaky(system, user):
            if "#1 " in system or "#2 " in system:
                raise RuntimeError("boom")
            return oracle(system, user)
        ps = people(40)
        res = pop.simulate_population(FakeLLM(flaky), ps, Q, "cell_sample", keys=["region"], draws=30)
        self.assertLessEqual(res.calls, 40)
        self.assertEqual(sum(res.n.values()), 40)
        self.assertEqual(len(res.per_person), sum(res.parsed.values()))
        failed = [p["id"] for p, pr in zip(ps, res.prompts) if pr.meta["draw"] in (1, 2)]
        self.assertTrue(failed)                                       # the seed does hit the failing draws
        self.assertEqual(sum(res.parsed.values()), 40 - len(failed))
        self.assertTrue(all(pid not in res.per_person for pid in failed))
        self.assertAlmostEqual(res.dist["region=north"]["yes"], 1.0)

    def test_compare_injectors_runs(self):
        res = pi.compare_injectors(FakeLLM(oracle), people(), [Q], "answers",
                                   [("cell_verbalized", {"keys": ["region"]})], parse=pop.parse_modal, B=100)
        t = res.table.set_index("arm")
        self.assertAlmostEqual(t.loc["cell_verbalized(keys=['region'])", "score"], 1.0)

    def test_ensemble(self):
        a, b = {"yes": 1.0, "no": 0.0}, {"yes": 0.0, "no": 1.0}
        self.assertAlmostEqual(pop.ensemble([a, b])["yes"], 0.5)
        self.assertAlmostEqual(pop.ensemble([a, b], [3, 1])["yes"], 0.75)
        nested = pop.ensemble([{"c1": a, "c2": a}, {"c1": b}])
        self.assertAlmostEqual(nested["c1"]["yes"], 0.5)
        self.assertAlmostEqual(nested["c2"]["yes"], 1.0)
        with self.assertRaises(ValueError):
            pop.ensemble([a, b], [1])
        with self.assertRaises(ValueError):
            pop.ensemble([])


class TestEstimators(unittest.TestCase):
    def test_calibrate(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 400)
        y = (rng.uniform(size=400) < p ** 2).astype(int)          # model over-confident on "yes"
        for m in ("platt", "isotonic"):
            q = pop.calibrate(p[:200], y[:200], p[200:], method=m)
            self.assertEqual(q.shape, (200,))
            self.assertTrue(((q >= 0) & (q <= 1)).all())
            self.assertLess(abs(q.mean() - y[200:].mean()), abs(p[200:].mean() - y[200:].mean()))
        with self.assertRaises(ValueError):
            pop.calibrate([0.2, 0.3], [1, 1], [0.5])
        with self.assertRaises(ValueError):
            pop.calibrate([0.2, 0.3], [0, 1], [0.5], method="beta")

    def test_ppi_mean(self):
        rng = np.random.default_rng(1)
        N, n = 2000, 100
        y_all = rng.normal(size=N + n)
        f_all = y_all + rng.normal(scale=0.3, size=N + n) + 0.5      # biased but informative predictor
        y, fl, fu = y_all[:n], f_all[:n], f_all[n:]
        r0 = pop.ppi_mean(y, fl, fu, lam=0)
        self.assertAlmostEqual(r0["estimate"], y.mean())
        r1 = pop.ppi_mean(y, fl, fu, lam=1)
        self.assertAlmostEqual(r1["estimate"], y.mean() + fu.mean() - fl.mean())
        ra = pop.ppi_mean(y, fl, fu)
        self.assertGreater(ra["lam"], 0.5)
        self.assertLess(ra["width_ratio"], 0.6)                       # far narrower than labels alone
        self.assertLess(abs(ra["estimate"] - y_all[n:].mean()), 0.1)  # bias of 0.5 removed
        self.assertLessEqual(ra["lo"], ra["estimate"])
        noise = pop.ppi_mean(y, rng.normal(size=n), rng.normal(size=N))
        self.assertLess(abs(noise["lam"]), 0.5)                       # useless predictor gets little weight
        with self.assertRaises(ValueError):
            pop.ppi_mean([1.0], [1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            pop.ppi_mean([1.0, float("nan")], [1.0, 1.0], [1.0, 2.0])

    def test_ppi_coverage(self):
        rng = random.Random(3)
        hits = 0
        for _ in range(200):
            y = [rng.gauss(0, 1) for _ in range(60)]
            fl = [v + rng.gauss(0, 0.5) for v in y]
            fu = [rng.gauss(0, 1) + rng.gauss(0, 0.5) for _ in range(600)]
            r = pop.ppi_mean(y, fl, fu)
            hits += r["lo"] <= 0 <= r["hi"]
        self.assertGreater(hits / 200, 0.88)

    def test_ppi_distribution(self):
        ps = people(40)
        res = pop.simulate_population(FakeLLM(oracle), ps, Q, "cell_verbalized", keys=["region"])
        lab, unl = ps[:10], ps[10:]
        out = pop.ppi_distribution([p["answers"]["q0"] for p in lab], [res.per_person[p["id"]] for p in lab],
                                   [res.per_person[p["id"]] for p in unl], OPTS)
        self.assertEqual(set(out), set(OPTS))
        self.assertAlmostEqual(out["yes"]["estimate"], 0.5, delta=0.1)
        self.assertLess(out["yes"]["se"], 0.2)
        with self.assertRaises(ValueError):
            pop.ppi_distribution(["maybe", "yes"], [{}, {}], [{}, {}], OPTS)


if __name__ == "__main__":
    unittest.main()

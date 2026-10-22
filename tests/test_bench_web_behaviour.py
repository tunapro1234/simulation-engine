"""web_behaviour bench tasks on synthetic WebShop-style trajectories and UXAgent-style traces; the real-data check is
skipped unless $SIMENGINE_DATA is set."""
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from simengine.bench import base
from simengine.bench.base import DataNotFound
from simengine.bench.tasks import web_behaviour as wb
from simengine.bench.tasks.web_behaviour import A, ACTIONS, K, WebShopBuy, WebShopNextAction
from simengine.llm.fake import FakeLLM
from simengine.types import TaskData

GOAL = {"asin": "X0", "instruction_text": "i want a blue mug under 20 dollars", "goal_options": ["blue"],
        "price_upper": 20.0}


def _sr(kw, page, asins=("P1", "P2")):
    return {"page": "search_results", "goal": GOAL,
            "content": {"keywords": list(kw), "search_result_asins": list(asins), "page": page}}


def _item(asin, opts=None):
    return {"page": "item_page", "goal": GOAL,
            "content": {"keywords": "['mug']", "page": "1", "asin": asin, "options": opts or {}}}


def _sub(asin):
    return {"page": "item_sub_page", "goal": GOAL, "content": {"keywords": "['mug']", "page": "1", "asin": asin,
                                                                 "options": {}}}


IDX = {"page": "index", "goal": GOAL}


def _done(asin, reward=1.0):
    return {"page": "done", "goal": GOAL, "content": {"asin": asin, "options": {}, "price": 9.5},
            "reward": str(reward)}


def _hand():
    """One trajectory covering every derived action; expected names in order."""
    lines = [IDX, _sr(["mug"], 1), _sr(["mug"], 2), _sr(["mug"], 1), _item("P1"), _sub("P1"), _item("P1"),
             _item("P1", {"color": "blue"}), _item("P1", {"color": "blue"}), _sr(["mug"], 1), IDX, IDX,
             _sr(["blue", "mug"], 1), _item("P2"), _item("P3"), _done("P3", 0.5)]
    want = ["search", "next_page", "prev_page", "click_item", "view_detail", "back", "select_option", "other", "back",
            "back_to_search", "other", "search", "click_item", "click_item", "buy"]
    return lines, want


def _random_traj(rng, buy_after_opts=True):
    """index -> search -> (click, maybe option / detail / back) ... -> buy."""
    lines = [IDX, _sr(["mug"], 1)]
    for k in range(rng.integers(1, 4)):
        a = f"P{rng.integers(0, 9)}"
        lines.append(_item(a))
        if rng.random() < 0.4:
            lines += [_sub(a), _item(a)]
        if rng.random() < 0.6:
            lines.append(_item(a, {"color": "blue"}))
        if k < 2 and rng.random() < 0.5:
            lines.append(_sr(["mug"], 1))
            if rng.random() < 0.3:
                lines.append(_sr(["mug"], 2))
    if lines[-1]["page"] != "item_page":
        lines.append(_item("P9"))
    lines.append(_done("P9", float(rng.random() < 0.5)))
    return lines


def _ux_rows():
    act = lambda typ, name: {"type": typ, "name": name, "args": None, "description": "", "text": None, "url": None}
    trace = [[act("type_and_submit", "header.search_box.search_input"), act("click", "header.search_box.search_button")],
             [act("click", "search_results._blue_mug_with_cart_print_.view_product")],
             [act("click", "product_info._reviews_")],
             [act("click", "product_form.options.blue")],
             [act("click", "product_form.add_to_cart")],
             [act("back", None)],
             [act("click", "pager._page_next_")],
             [act("click", "sorter.price")],
             [act("terminate", None)]]
    return [trace, trace[:5]]


def _write(root, n_train=60, n_test=60, seed=0, ux=True):
    d = os.path.join(root, "webshop-insan", "all_trajs")
    os.makedirs(d)
    rng = np.random.default_rng(seed)
    for i in range(n_train + n_test):
        name = f"20220429_1_{i}.jsonl" if i < n_train else f"20220513_2_fixed_200_{i}.jsonl"
        with open(os.path.join(d, name), "w") as f:
            for x in _random_traj(rng):
                f.write(json.dumps(x) + "\n")
    hand, _ = _hand()
    with open(os.path.join(d, "20220514_0_fixed_100_999.jsonl"), "w") as f:
        for x in hand:
            f.write(json.dumps(x) + "\n")
    if ux:
        u = os.path.join(root, "uxagent", "data")
        os.makedirs(u)
        pd.DataFrame({"action_trace": [np.array([np.array(s, dtype=object) for s in t], dtype=object)
                                       for t in _ux_rows()]}).to_parquet(
            os.path.join(u, "train-00000-of-00001.parquet"))


class TestParsing(unittest.TestCase):
    def test_classify_every_action(self):
        lines, want = _hand()
        steps, summ = wb.parse_trajectory(lines, "s")
        self.assertEqual(steps["action"].tolist(), want)
        self.assertEqual(set(want), set(ACTIONS))
        self.assertEqual(summ["n_actions"], 15)
        self.assertEqual(summ["bought"], 1)
        self.assertEqual(summ["reward"], 0.5)
        self.assertEqual(summ["n_goal_opts"], 1)

    def test_state_features_use_only_the_past(self):
        lines, want = _hand()
        st = wb.parse_trajectory(lines, "s")[0].set_index("t")
        self.assertEqual(st.loc[0, "history"], "")
        self.assertEqual(st.loc[3, "history"], "search next_page prev_page")
        self.assertEqual((st.loc[3, "prev"], st.loc[3, "prev2"]), ("prev_page", "next_page"))
        self.assertEqual(st.loc[0, "n_search"], 0)
        self.assertEqual(st.loc[12, "n_search"], 2)          # the second search happened at t = 11
        self.assertEqual(st.loc[7, "n_opts"], 1)             # on P1 with the colour chosen
        self.assertEqual(st.loc[7, "opts_done"], 1)
        self.assertEqual(st.loc[4, "n_opts"], 0)
        self.assertEqual(st.loc[13, "n_items"], 2)           # P1, P2 (P3 not yet open)
        self.assertEqual(st.loc[2, "results_page"], 2)
        self.assertEqual(st.loc[14, "page"], "item_page")
        self.assertEqual(st.loc[14, "y"], A["buy"])

    def test_uxagent_mapping(self):
        seqs = wb.uxagent_sequences(_ux_rows())
        names = [[ACTIONS[i] for i in s] for s in seqs]
        self.assertEqual(names[0], ["search", "click_item", "view_detail", "select_option", "buy", "back", "next_page",
                                    "other"])       # search typed + button = one; product name with 'cart' = click
        self.assertEqual(len(names[1]), 5)
        self.assertIsNone(wb.map_uxagent_action({"type": "terminate", "name": None}))
        self.assertEqual(wb.map_uxagent_action({"type": "click", "name": "cart.proceed_to_checkout"}), "other")

    def test_session_summary_and_transitions(self):
        s = wb.session_summary([[A["search"], A["click_item"], A["buy"]], []])
        self.assertEqual(s["n_actions"].tolist(), [3, 0])
        self.assertEqual(s["bought"].tolist(), [1, 0])
        pr, nx = wb.sequence_transitions([[0, 1, 8], [0]])
        self.assertEqual(pr.tolist(), [0, 1])
        self.assertEqual(nx.tolist(), [1, 8])
        c = wb.transition_counts(pr, nx)
        self.assertEqual(c.shape, (K + 1, K))
        self.assertEqual(c.sum(), 2)


class TestTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        _write(cls.tmp.name)
        cls.root = cls.tmp.name

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_registered(self):
        self.assertIn("webshop_next_action", base.list_tasks())
        self.assertIn("webshop_buy", base.list_tasks())
        with self.assertRaises(ValueError):
            WebShopNextAction(bogus=1)

    def test_batch_split_and_rows(self):
        t = WebShopNextAction()
        d = t.load(self.root)
        self.assertEqual(len(d.meta["train_sessions"]), 60)
        self.assertEqual(len(d.meta["test_sessions"]), 61)
        self.assertTrue((d.train["batch"].str[:8] < "20220513").all())
        self.assertTrue((d.test["batch"].str[:8] >= "20220513").all())
        self.assertTrue((d.train["t"] >= 1).all() and (d.test["t"] >= 1).all())
        self.assertEqual(d.meta["n_first_actions_excluded"], 121)
        self.assertEqual(d.meta["n_uxagent_sessions"], 2)
        self.assertNotIn("instruction", d.meta["sessions"].columns)

    def test_random_split_is_by_session(self):
        d = WebShopNextAction(split="random", test_frac=0.5, seed=3).load(self.root)
        self.assertFalse(set(d.train["session"]) & set(d.test["session"]))
        self.assertEqual(len(d.meta["test_sessions"]), 60)
        with self.assertRaises(ValueError):
            WebShopNextAction(split="nope").load(self.root)

    def test_next_action_baselines_and_scores(self):
        t = WebShopNextAction()
        d = t.load(self.root)
        fr = t.run(data=d)
        for m in ("uniform", "frequency", "markov", "markov2", "logistic", "uxagent_markov"):
            p = np.vstack(fr["pred_" + m].to_numpy())
            self.assertEqual(p.shape, (len(d.test), K))
            np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)
        sc = t.score_all(fr, B=50).set_index(["method", "metric"])["value"]
        self.assertAlmostEqual(sc[("uniform", "loglik")], -np.log(K), places=9)
        self.assertAlmostEqual(sc[("uniform", "accuracy")], 1 / K, places=9)
        self.assertGreater(sc[("markov", "loglik")], sc[("frequency", "loglik")])
        self.assertGreater(sc[("logistic", "loglik")], sc[("uniform", "loglik")])
        self.assertEqual(set(fr["segment"]) - {"index", "search_results", "item_page", "item_sub_page"}, set())
        self.assertEqual(fr["group"].nunique(), 61)

    def test_markov_matches_counts(self):
        tr = pd.DataFrame({"prev": ["search", "search", "search", "click_item"], "y": [1, 1, 3, 8]})
        te = pd.DataFrame({"prev": ["search", "buy"]})
        p = WebShopNextAction._markov(tr, te)
        np.testing.assert_allclose(p[0, 1], 3 / (3 + K))
        np.testing.assert_allclose(p[1], np.full(K, 1 / K))      # unseen previous action -> uniform
        tr["prev2"] = "none"
        te["prev2"] = "none"
        p2 = WebShopNextAction._markov2(tr, te)
        np.testing.assert_allclose(p2[0, 1], (2 + wb.PRIOR2 * 3 / (3 + K)) / (3 + wb.PRIOR2))

    def test_metric_values(self):
        t = WebShopNextAction()
        fr = pd.DataFrame({"truth": [0, 1, 1, 2], "group": ["a", "a", "b", "b"], "split": "test"})
        p = np.zeros((4, K))
        p[:, 1] = 1.0
        fr["pred_one"] = pd.Series(list(p), dtype=object)
        self.assertAlmostEqual(t.score(fr, "one", "accuracy", B=20)["value"], 0.5)
        self.assertAlmostEqual(t.score(fr, "one", "loglik", B=20)["value"], 0.5 * np.log(wb.EPS))
        self.assertAlmostEqual(t.score(fr, "one", "share_tvd", B=20)["value"], 0.5)
        tie = np.zeros((4, K))
        tie[:, :2] = 0.5
        fr["pred_tie"] = pd.Series(list(tie), dtype=object)
        self.assertAlmostEqual(t.score(fr, "tie", "accuracy", B=20)["value"], 0.375)

    def test_frame_checks_shape(self):
        t = WebShopNextAction()
        d = t.load(self.root)
        with self.assertRaises(ValueError):
            t.frame(d, {"bad": np.ones((len(d.test), K - 1))})
        with self.assertRaises(ValueError):
            t.frame(d, {"neg": -np.ones((len(d.test), K))})

    def test_buy_task(self):
        t = WebShopBuy()
        d = t.load(self.root)
        self.assertTrue((d.test["page"] == "item_page").all())
        fr = t.run(data=d)
        self.assertEqual(set(fr["truth"]), {0, 1})
        np.testing.assert_allclose(fr["truth"].mean(), d.meta["positive_rate"]["test"])
        sc = t.score_all(fr, B=50).set_index(["method", "metric"])["value"]
        self.assertAlmostEqual(sc[("constant", "auc")], 0.5)
        self.assertGreater(sc[("logistic", "auc")], 0.6)
        self.assertIn(("uxagent_markov", "auc"), sc.index)

    def test_no_uxagent(self):
        with tempfile.TemporaryDirectory() as r:
            _write(r, 10, 10, ux=False)
            t = WebShopNextAction()
            d = t.load(r)
            self.assertIsNone(d.meta["uxagent"])
            self.assertNotIn("uxagent_markov", t.baselines(d))
            d2 = WebShopNextAction(uxagent=False).load(self.root)
            self.assertIsNone(d2.meta["uxagent"])

    def test_compare_agents(self):
        d = WebShopNextAction().load(self.root)
        c = wb.compare_agents(d)
        self.assertEqual(list(c.columns), ["human_train", "human_test", "uxagent"])
        sh = c.loc[[f"share_{a}" for a in ACTIONS]]
        np.testing.assert_allclose(sh.sum().to_numpy(), 1.0)
        self.assertEqual(c.loc["sessions", "uxagent"], 2)

    def test_missing_data(self):
        with tempfile.TemporaryDirectory() as r:
            with self.assertRaises(DataNotFound):
                WebShopNextAction().load(r)
        old = os.environ.pop(base.ENV_DATA, None)
        try:
            with self.assertRaises(DataNotFound):
                WebShopNextAction().load(None)
        finally:
            if old is not None:
                os.environ[base.ENV_DATA] = old


class TestLLM(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _write(self.tmp.name, 8, 4)
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_action(self):
        v = wb.parse_action('Sure: {"buy": 0.7, "Back": 0.2, "select option": 0.1, "fly": 3}')
        self.assertAlmostEqual(v[A["buy"]], 0.7)
        self.assertAlmostEqual(v[A["back"]], 0.2)
        self.assertAlmostEqual(v[A["select_option"]], 0.1)
        self.assertEqual(wb.parse_action("I think back_to_search").argmax(), A["back_to_search"])
        self.assertEqual(wb.parse_action("probably: back").argmax(), A["back"])
        self.assertIsNone(wb.parse_action("no idea"))
        self.assertIsNone(wb.parse_action(None))

    def test_prompts_carry_only_the_past(self):
        t = WebShopNextAction()
        d = t.load(self.root)
        row = d.test.iloc[[0]]
        (sysm, user), = wb.prompts(row, d.meta["instructions"])
        self.assertIn("blue mug", user)
        self.assertIn("Actions so far: " + (row["history"].iloc[0].replace(" ", ", ") or "(none)"), user)
        self.assertNotIn(f"Actions so far: {row['history'].iloc[0]} {row['action'].iloc[0]}", user)
        for a in ACTIONS:
            self.assertIn(f'"{a}"', user)

    def test_llm_method_both_tasks(self):
        llm = FakeLLM(lambda s, u: '{"buy": 1.0}' if "item_page" in u else "garbage")
        t = WebShopNextAction()
        d = t.load(self.root)
        m = wb.llm_method(llm, smooth=0.1)
        fr = t.run(methods={"llm": m}, data=d, baselines=False)
        p = np.vstack(fr["pred_llm"].to_numpy())
        np.testing.assert_allclose(p.sum(1), 1.0)
        item = (d.test["page"] == "item_page").to_numpy()
        np.testing.assert_allclose(p[item, A["buy"]], 0.9 + 0.1 / K)
        np.testing.assert_allclose(p[~item], 1 / K)
        self.assertEqual(m.last_unparsed, int((~item).sum()))
        self.assertEqual(len(llm.calls), len(d.test))
        b = WebShopBuy()
        db = b.load(self.root)
        fb = b.run(methods={"llm": wb.llm_method(llm)}, data=db, baselines=False)
        self.assertEqual(fb["pred_llm"].shape, (len(db.test),))
        self.assertTrue((fb["pred_llm"] > 0.9).all())


@unittest.skipUnless(os.environ.get(base.ENV_DATA), "$SIMENGINE_DATA not set")
class TestRealData(unittest.TestCase):
    def test_load_and_baselines(self):
        t = WebShopNextAction()
        try:
            d = t.load(None)
        except DataNotFound:
            self.skipTest("webshop-insan not under $SIMENGINE_DATA")
        self.assertGreater(len(d.meta["test_sessions"]), 100)
        self.assertTrue(set(d.test["action"]) <= set(ACTIONS))
        fr = t.run(data=d, baselines=["frequency", "markov"])
        sc = t.score_all(fr, B=50).set_index(["method", "metric"])["value"]
        self.assertGreater(sc[("markov", "loglik")], sc[("frequency", "loglik")])


if __name__ == "__main__":
    unittest.main()

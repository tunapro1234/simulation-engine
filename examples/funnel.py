"""Funnel workflow: where do simulated visitors drop off a landing -> pricing -> checkout journey, and which page
variant converts better? Offline (FakeLLM with a deterministic responder), synthetic personas.

    python examples/funnel.py

The responder stands in for an LLM: it reads the step the persona is on ("Now you are on step k of n") and the
persona's profile, so price-sensitive personas drop on a paid-only pricing page and everyone drops a little on a
long checkout form. With a real backend pass llm=se.llm.Claude(...) instead; the code path is the same.
Dry runs only: the numbers below come from the responder, not from people.
"""
import re

from simengine.llm.fake import FakeLLM, survey_responder
from simengine.sim.funnel import funnel, funnel_frame

PERSONAS = [{"id": f"v{i}", "segment": seg,
             "text": f"{'Student, tight monthly budget' if seg == 'student' else 'Working professional'}, age {a}"}
            for i, (seg, a) in enumerate([("student", 19), ("student", 21), ("student", 22), ("student", 24),
                                          ("pro", 29), ("pro", 34), ("pro", 41), ("pro", 47)])]

VARIANTS = {
    "A": ["Landing page: 'Budget in five minutes a week'. One screenshot, a Start button.",
          "Pricing: 6 USD per month, billed monthly. No free plan.",
          "Checkout: name, email, card number, billing address, phone (all required)."],
    "B": ["Landing page: 'Budget in five minutes a week'. One screenshot, a Start button.",
          "Pricing: free for 30 days, then 6 USD per month. Cancel any time.",
          "Checkout: email and card number."],
}
STEP = re.compile(r"Now you are on step \d+ of \d+:\n(.*)")


def responder(system: str, user: str) -> str:
    """Continue probability (0-100) from the current step text and the persona's profile."""
    step = (STEP.search(user) or [None, ""])[1]

    def value(pid, qid, spec):
        student = "Student" in spec["profile"]
        if step.startswith("Pricing"):
            return (75 if "free for" in step else 30 if student else 70) + int(pid[1:]) % 3 * 3
        if step.startswith("Checkout"):
            return 60 if "phone" in step else 85
        return 80 - int(pid[1:]) % 4 * 4

    return survey_responder(value)(system, user)


def main():
    res = funnel(PERSONAS, VARIANTS, llm=FakeLLM(responder), context="A budgeting app.", B=500)
    pct = lambda c: f"{100 * c['value']:5.1f} % [{100 * c['low']:5.1f}, {100 * c['high']:5.1f}]"
    for vid, v in res["variants"].items():
        print(f"variant {vid}: conversion {pct(v['conversion'])}, biggest drop at {v['biggest_drop']}")
        for s in v["steps"]:
            print(f"  {s['id']}  continue {pct(s['continue'])}  reach {pct(s['reach'])}  drop {pct(s['drop'])}")
        print("  by segment:", {k: round(c["value"], 3) for k, c in v["segments"].items()})
    for pr in res["comparison"]["pairs"]:
        d = pr["diff"]
        print(f"{pr['variant']} - {pr['baseline']} final conversion: {100 * d['value']:+.1f} pp "
              f"[{100 * d['low']:+.1f}, {100 * d['high']:+.1f}], P(better) {pr['p_better']:.2f}")
    print("usage:", {k: res["usage"][k] for k in ("surveys", "calls", "cost_usd")}, "warnings:", res["warnings"])
    frame = funnel_frame(res)
    print(frame.groupby(["variant", "step"])[["p_continue", "reach"]].mean().round(3))


if __name__ == "__main__":
    main()

"""simengine.evidence: the evidence layer (port of the TALP-prep `kanit` package).

    import simengine.evidence as ev
    report = ev.evaluate(frame, title="Twin T1", ours="hybrid", raw="llm")   # EvalFrame or Turkish columns
    ev.card(report, "card.md"); ev.dashboard([report], "pano.html")
    ev.worth_humans(frame, prior="llm", n=100)                              # PPI++ "worth N humans"
    ev.seal(predictions, "before truth")                                    # = se.seal; evaluate(prereg=".onkayit")
    ev.why_match({"persona": gen}, candidates, truth_idx)                   # N-way 'why' recognition -> EvalFrame
    ev.evaluate_distribution_persons(people, ["age"], items, arms)          # person-cluster distribution bootstrap
    ev.ab_pairs_frame(pairs, ["p_llm"], group="test_id")                    # A/B pair table -> evaluate(task="ab")
    ev.ab_memorization(llm, pairs, named_context, neutral_context)          # A/B memorization probe -> memorization=

Report schema: docs/modules/evidence.md (keys keep the original Turkish names; `summary(report)` gives an
English flat view). Internals are verbatim ports; the English API lives in `api.py`.
"""
from .api import (MULTIPLIER_MAX_ROWS, audit_prompts, chain_paths, correlation_audit, evaluate_aggregate,
                  BADGE_TO_ENGLISH, METRIC_TO_ENGLISH, METRIC_TO_INTERNAL, TASK_TO_ENGLISH, TASK_TO_INTERNAL,
                  VERSION, Resampler, answer_hit, badge, baseline_ladder, bootstrap_metrics, canonical_bytes, card,
                  card_dict, card_markdown, chain, ci, compute_metrics, dashboard, dashboard_html, ece, evaluate,
                  grade, hit_at_k, jensen_shannon, ks_distance, memorization_audit, metric_family, p_value,
                  paired_ci, paired_diff, panel_cost, prepare, recalibrate, reliability, sample_multiplier, save_json, seal,
                  shuffle_control, split_audit, summary, to_internal, tvd, verify, verify_chain, wasserstein_ordinal,
                  worth_humans, worth_theory_curve)
from .ab import ab_memorization, ab_memorization_items, ab_pairs_frame
from .extras import (coefficient_recovery, completion_from_frame, completion_row, published_findings,
                     section_html, section_markdown, stylized_facts_score, variance_ratio)
from .persons import DIST_METRICS, evaluate_distribution_persons
from .why import TfidfEncoder, tfidf_encoder, why_match, why_summary

__all__ = [
    "evaluate", "evaluate_aggregate", "summary", "card", "card_markdown", "card_dict", "dashboard", "dashboard_html", "save_json",
    "worth_humans", "worth_theory_curve", "sample_multiplier",
    "prepare", "to_internal", "compute_metrics", "metric_family", "hit_at_k", "ece", "tvd", "wasserstein_ordinal",
    "jensen_shannon", "ks_distance",
    "Resampler", "ci", "p_value", "bootstrap_metrics", "paired_ci", "paired_diff",
    "baseline_ladder", "shuffle_control", "recalibrate", "reliability",
    "split_audit", "memorization_audit", "answer_hit", "correlation_audit", "audit_prompts",
    "seal", "verify", "verify_chain", "grade", "chain", "chain_paths", "canonical_bytes",
    "badge", "panel_cost",
    "variance_ratio", "completion_row", "completion_from_frame", "published_findings", "stylized_facts_score",
    "coefficient_recovery", "section_markdown", "section_html",
    "why_match", "why_summary", "tfidf_encoder", "TfidfEncoder", "evaluate_distribution_persons", "DIST_METRICS",
    "ab_pairs_frame", "ab_memorization_items", "ab_memorization",
    "TASK_TO_INTERNAL", "TASK_TO_ENGLISH", "METRIC_TO_INTERNAL", "METRIC_TO_ENGLISH", "BADGE_TO_ENGLISH", "VERSION", "MULTIPLIER_MAX_ROWS",
]

"""simengine.games: one-shot economic game library, LLM runner, distance-to-human scoring, coherence audit and the
who-to-punish scorecard.

    import simengine.games as gm
    res = gm.play(llm, {"baseline": {"b-000": ""}, "big5": bank}, games="calibration", runs=3)
    gm.moments(res); gm.distance_to_human(res); gm.coherence_audit(res)
    frame = gm.moment_frame(res)                      # EvalFrame -> simengine.evidence.evaluate(frame, ...)

Provenance (read-only sources, ported with notes): edsl-test BeeBench b01 (games, holdout, run.py, analyze.py),
edsl-test benchmark/score_wtp.py + human_targets.json (who-to-punish scorecard), replicant
papers/ertan2009/config.py (ertan_vignette parameters; replicant is MIT licensed, credit to its authors).
"""
from .coherence import FLAGS, PRECEDENT, coherence_audit, detect, flag_rows
from .library import (CALIBRATION, ERTAN_PROFILES, GAMES, HOLDOUT, answer_spec, bounds, check_game, condition,
                      get_game, list_games, render, render_belief)
from .runner import COLUMNS, EXTRA_COLUMNS, KEY, CostGateError, as_agents, estimate_cost, play, read_results
from .scoring import (HUMAN_DISTRIBUTIONS, MOMENTS, classify_schedule, distance_to_human, human_distributions,
                      human_refs, ks_to, moment_frame, moments, wasserstein_to)
from ._answer import instruction, parse_answer
from .wtp import CRITERIA, ERTAN_TARGETS, WTP_PARAMS, targeting_share, wtp_scorecard

__all__ = [
    "GAMES", "CALIBRATION", "HOLDOUT", "ERTAN_PROFILES", "list_games", "get_game", "check_game", "render",
    "render_belief", "condition", "bounds", "answer_spec",
    "play", "estimate_cost", "as_agents", "read_results", "CostGateError", "COLUMNS", "EXTRA_COLUMNS", "KEY",
    "moments", "human_refs", "human_distributions", "distance_to_human", "moment_frame", "MOMENTS",
    "HUMAN_DISTRIBUTIONS", "classify_schedule", "wasserstein_to", "ks_to",
    "coherence_audit", "detect", "flag_rows", "PRECEDENT", "FLAGS",
    "wtp_scorecard", "targeting_share", "CRITERIA", "WTP_PARAMS", "ERTAN_TARGETS",
    "parse_answer", "instruction",
]

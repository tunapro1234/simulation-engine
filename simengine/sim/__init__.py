"""simengine.sim: LLM simulation runners.

Runners (each returns a JSON-safe dict; a matching *_frame turns it into an EvalFrame):
  survey / survey_frame               survey panel, Likert (direct score + optional SSR) and multiple choice
  ab_test / ab_frame / ab_many        A/B test on a persona panel (PAARS-style), many-test batch prompt
  roundtable / roundtable_frame       multi-round group discussion
  strategy / strategy_frame           2+ scenarios x segment
  segment_distribution                ask the model directly for each segment's answer distribution
  persona_distribution                per-persona probability vectors
  predict_people / prediction_frame   per-person L0 / L1 / L2 / SSR prediction prompts
  batch_calibrate, retest_stats       label-free calibration and test-retest of repeated samples
Controls: no_persona and persona_swap arms run inside survey and ab_test (`controls=`); control_personas and
control_plan expose them for custom runners. Personas are duck-typed (as_personas); simengine.persona objects,
TALP-schema dicts, flat dicts and plain strings all work.

Offline: FakeLLM(schema_responder()) answers every prompt here with deterministic values (dry runs, tests).
What works and what does not, with measured numbers: docs/modules/sim.md.
"""
from .ab import ab_frame, ab_many, ab_test, confidence_check, paars_decision
from .distribution import (DIST_TEMPLATE, FRAMINGS_HIDDEN, FRAMINGS_NAMED, distribution_frame, distribution_prompt,
                           persona_distribution, segment_distribution, w1_ordinal)
from .panel import Usage, ask_panel, build_prompts, schema_responder
from .parse import normalize_probs, parse_distribution, parse_item_probs, parse_json, parse_text_fields
from .personas import PersonaView, as_persona, as_personas, control_personas, control_plan
from .prediction import (AMAZON, PredictionSpec, batch_calibrate, history_text, predict_people, prediction_frame,
                         profile_text, prompt_l0, prompt_l1, prompt_l2, prompt_ssr, retest_stats)
from .roundtable import CONFIRMATION_BIAS, KEEP_VIEW, roundtable, roundtable_frame
from .ssr import SCALES, SSR
from .strategy import strategy, strategy_frame
from .survey import survey, survey_frame

__all__ = [
    "survey", "survey_frame", "SSR", "SCALES",
    "ab_test", "ab_frame", "ab_many", "confidence_check", "paars_decision",
    "roundtable", "roundtable_frame", "CONFIRMATION_BIAS", "KEEP_VIEW",
    "strategy", "strategy_frame",
    "segment_distribution", "persona_distribution", "distribution_prompt", "distribution_frame", "w1_ordinal",
    "FRAMINGS_NAMED", "FRAMINGS_HIDDEN", "DIST_TEMPLATE",
    "PredictionSpec", "AMAZON", "predict_people", "prediction_frame", "prompt_l0", "prompt_l1", "prompt_l2",
    "prompt_ssr", "profile_text", "history_text", "batch_calibrate", "retest_stats",
    "ask_panel", "build_prompts", "Usage", "schema_responder",
    "PersonaView", "as_persona", "as_personas", "control_personas", "control_plan",
    "parse_json", "normalize_probs", "parse_distribution", "parse_item_probs", "parse_text_fields",
]

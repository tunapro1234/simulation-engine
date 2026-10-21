"""simengine.persona: TALP-compatible persona schema, persona sources and builders from data.

    import simengine.persona as sp
    personas = sp.from_table(df, id_col="user_id", forbidden=["target"])      # personas from survey rows
    personas = sp.from_history(survey, id_col="uid", level="L2", purchases=orders, cutoff="2022-07-01")
    src = sp.ClaudeSource(llm)                                                 # any simengine.types.LLM
    panel = src.generate(sp.AudienceSpec(...), n=20)
    answers = src.ask(panel, "Rate this ad 1-10", {"score": "int 1-10"})
"""
from .builders import (TALP_FIELDS, Mapping, age_band, auto_mapping, from_row, from_table, gender_value,
                       narrate, to_frames)
from .history import (LEVELS, from_history, purchase_history_text, summarize_purchases,
                      survey_profile_text)
from .schema import (ABSOLUTE_WORDS, AGE_BAND, BRAND_AWARENESS, COMMITMENT, ENUMS, GENDER, HOUSEHOLD,
                     INCOME_TIER, LABELS, PERSONA_MODE, USAGE, Audience, AudienceSpec, Authored, Behavioral,
                     Disposition, Evidence, Identity, Persona, Trait, Trigger, Triggers, dump_personas, enum_for,
                     load_personas)
from .sources import (ASK_PROMPT, GENERATE_PROMPT, SYSTEM_ASK, SYSTEM_GENERATE, BudgetExceeded, CallCounter,
                      ClaudeSource, FakeSource, LLMError, LLMSource, PersonaSource, Reply, Request, TalpSource,
                      build_persona, cell_plan, json_extract, open_source)

__all__ = [
    # schema
    "Audience", "AudienceSpec", "Identity", "Trait", "Trigger", "Triggers", "Authored", "Behavioral",
    "Disposition", "Evidence", "Persona", "load_personas", "dump_personas", "enum_for", "ENUMS", "LABELS",
    "ABSOLUTE_WORDS", "INCOME_TIER", "HOUSEHOLD", "USAGE", "COMMITMENT", "BRAND_AWARENESS", "AGE_BAND", "GENDER",
    "PERSONA_MODE",
    # sources
    "PersonaSource", "ClaudeSource", "LLMSource", "TalpSource", "FakeSource", "open_source", "Request", "Reply",
    "CallCounter", "BudgetExceeded", "LLMError", "json_extract", "cell_plan", "build_persona",
    "SYSTEM_GENERATE", "SYSTEM_ASK", "GENERATE_PROMPT", "ASK_PROMPT",
    # builders
    "Mapping", "from_row", "from_table", "auto_mapping", "narrate", "age_band", "gender_value", "to_frames",
    "TALP_FIELDS",
    # history
    "from_history", "survey_profile_text", "summarize_purchases", "purchase_history_text", "LEVELS",
]

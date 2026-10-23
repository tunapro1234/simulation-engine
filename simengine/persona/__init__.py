"""simengine.persona: TALP-compatible persona schema, persona sources and builders from data.

    import simengine.persona as sp
    personas = sp.from_table(df, id_col="user_id", forbidden=["target"])      # personas from survey rows
    idx = sp.pool_select(df, 20, method="kmeans", features=["visits", "spend"])  # small persona pool
    personas = sp.from_history(survey, id_col="uid", level="L2", purchases=orders, cutoff="2022-07-01")
    src = sp.ClaudeSource(llm)                                                 # any simengine.types.LLM
    panel = src.generate(sp.AudienceSpec(...), n=20)
    answers = src.ask(panel, "Rate this ad 1-10", {"score": "int 1-10"})
    texts = sp.sample_texts("homo_silicus", n=4, seed=1)                      # persona method zoo (methods/)
    people = sp.sample_bigfive(50, seed=0); res = sp.validate_bigfive(llm)    # Big Five + Mini-IPIP fidelity
    res = sp.compare_injectors(llm, people, qs, "answers", ["plain_fields"])   # persona injection zoo (inject/)
    rep = sp.field_report(personas, df, mapping); sp.attach(report, rep)       # fields that reached the prompt
"""
from . import inject, methods
from .bigfive import (DOMAINS as BIGFIVE_DOMAINS, PROFILES, SOTO_JOHN_2017_CORR, SOTO_JOHN_2017_NORMS,
                      US_ADULT_NORMS, bfi2_text, bigfive_audience, bigfive_traits, sample_bigfive)
from .builders import (POOL_METHODS, TALP_FIELDS, Mapping, age_band, apply_value_labels, auto_mapping, from_row,
                       from_table, gender_value, narrate, pool_curve, pool_select, read_codebook, to_frames,
                       warn_unlabelled)
from .fields import as_limit, attach, field_report
from .history import (LEVELS, from_history, purchase_history_text, summarize_purchases,
                      survey_profile_text)
from .inject import (CompareResult, Injector, MissingPersonField, Prompt, compare_injectors, get_injector,
                     list_injectors)
from .schema import (ABSOLUTE_WORDS, AGE_BAND, BRAND_AWARENESS, COMMITMENT, ENUMS, GENDER, HOUSEHOLD,
                     INCOME_TIER, LABELS, PERSONA_MODE, USAGE, Audience, AudienceSpec, Authored, Behavioral,
                     Disposition, Evidence, Identity, Persona, Trait, Trigger, Triggers, dump_personas, enum_for,
                     load_personas)
from .methods import (METHODS, AgentBank, PersonaDataNotFound, PersonaMethod, get_method, list_methods,
                      method_personas, sample_texts)
from .sources import (ASK_PROMPT, GENERATE_PROMPT, SYSTEM_ASK, SYSTEM_GENERATE, BudgetExceeded, CallCounter,
                      ClaudeSource, FakeSource, LLMError, LLMSource, PersonaSource, Reply, Request, TalpSource,
                      build_persona, cell_plan, json_extract, open_source)
from .validate import MINI_IPIP, score_mini_ipip, to_card_row, validate_bigfive

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
    "TALP_FIELDS", "apply_value_labels", "read_codebook", "warn_unlabelled", "pool_select", "pool_curve",
    "POOL_METHODS",
    # field report
    "field_report", "as_limit", "attach",
    # history
    "from_history", "survey_profile_text", "summarize_purchases", "purchase_history_text", "LEVELS",
    # persona method zoo
    "methods", "METHODS", "PersonaMethod", "PersonaDataNotFound", "list_methods", "get_method", "sample_texts",
    "method_personas", "AgentBank",
    # Big Five
    "bfi2_text", "bigfive_traits", "sample_bigfive", "bigfive_audience", "PROFILES", "US_ADULT_NORMS",
    "SOTO_JOHN_2017_NORMS", "SOTO_JOHN_2017_CORR", "BIGFIVE_DOMAINS", "MINI_IPIP", "score_mini_ipip",
    "validate_bigfive", "to_card_row",
    # persona injection zoo (inject/)
    "inject", "Prompt", "Injector", "MissingPersonField", "list_injectors", "get_injector", "compare_injectors",
    "CompareResult",
]

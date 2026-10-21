"""simengine.methods: non-LLM baselines and LLM + data hybrids with a small uniform API.

Every component is a port of research code measured during the prep nights; see docs/modules/methods.md for what
each one does, where it worked and where it did not (numbers + provenance). LightGBM is optional and imported
lazily; LLM calls go through the ``simengine.llm`` protocol (use ``FakeLLM`` offline).
"""
from .ab import ABStack, LengthAB, LLMDirectAB, LogitAverageAB, TfidfAB, TfidfWordAB
from .baselines import (ConstantBaseline, ItemMarginal, KNNBaseline, LightGBMBaseline, LogisticBaseline,
                        MajorityBaseline, PersistenceBaseline, SegmentRate, TextKNNRegressor, basket_tensor,
                        decayed_frequency, infer_kind, overall_copy, persistence, personal_frequency, popularity,
                        segment_distributions, tifu_knn, uniform_distribution, weighted_distribution)
from .bayes_prior import (EarlyTestBayes, calibrate_prior, early_lor, equivalent_multiplier, posterior_lor,
                          simulate_saving, simulate_saving_calibrated, true_lor)
from .conjoint import MNL, evaluate_mnl, null_scores, person_holdout
from .hybrid import (BinaryCalibrator, DevStack, LLMFeatureGBDT, ShrunkDeviation, ridge_to_prior,
                     select_lambda)
from .mixture import (CR_GAMES, CR_HUMAN_LEFT, ENGEL_DICTATOR, HORTON_GPT4O_WEIGHTS, HUMANIZING_PREFIX,
                      THEORY_TYPES, MixtureFit, ProbeGame, UnidentifiedWarning, charness_rabin_games, choice_game,
                      dictator_game, fit_mixture_weights, identifiability, mixture, numeric_game, probe_types)
from .next_action import (Markov2Baseline, NextActionRanker, add_history, attach_priors, build_features,
                          hit_at_k, markov2_score, negative_sample, top1, transition_priors)
from .twin import (TwinStack, base_predict, binned_median, column_correlation, donor_sample,
                   pooled_logistic_oof)
from ._util import lightgbm_available

__all__ = [
    # baselines
    "ConstantBaseline", "MajorityBaseline", "ItemMarginal", "SegmentRate", "PersistenceBaseline", "persistence",
    "LogisticBaseline", "LightGBMBaseline", "KNNBaseline", "TextKNNRegressor", "infer_kind",
    "uniform_distribution", "weighted_distribution", "overall_copy", "segment_distributions",
    "basket_tensor", "popularity", "personal_frequency", "decayed_frequency", "tifu_knn",
    # A/B text pairs
    "LengthAB", "TfidfAB", "TfidfWordAB", "LLMDirectAB", "LogitAverageAB", "ABStack",
    # LLM prior + early test
    "EarlyTestBayes", "calibrate_prior", "early_lor", "posterior_lor", "true_lor", "equivalent_multiplier",
    "simulate_saving",
    "simulate_saving_calibrated",
    # hybrids
    "ShrunkDeviation", "ridge_to_prior", "select_lambda", "LLMFeatureGBDT", "DevStack", "BinaryCalibrator",
    # person-level twin
    "TwinStack", "base_predict", "binned_median", "donor_sample", "pooled_logistic_oof", "column_correlation",
    # conjoint
    "MNL", "evaluate_mnl", "null_scores", "person_holdout",
    # next action
    "Markov2Baseline", "NextActionRanker", "add_history", "transition_priors", "attach_priors", "build_features",
    "markov2_score", "negative_sample", "top1", "hit_at_k",
    # theory-persona mixture calibration
    "fit_mixture_weights", "MixtureFit", "UnidentifiedWarning", "identifiability", "mixture", "probe_types",
    "ProbeGame", "choice_game", "numeric_game", "dictator_game", "charness_rabin_games",
    "CR_GAMES", "CR_HUMAN_LEFT", "ENGEL_DICTATOR", "THEORY_TYPES", "HUMANIZING_PREFIX", "HORTON_GPT4O_WEIGHTS",
    "lightgbm_available",
]

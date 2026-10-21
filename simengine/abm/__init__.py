"""Agent-based models: adoption ABM + ABC + analogy prior (with Bass controls and curve baselines), opinion drift
(bounded confidence vs persistence / mean shift), elasticity prior with shrinkage, repeat-purchase market ABM,
self-exciting cascade predictors (Hawkes / SEISMIC, not yet measured) and a turn-based LLM agent society.

See docs/modules/abm.md for when each one works, with measured numbers and provenance.
"""
from .engine import (ADOPTION_PARAMS, ADOPTION_PRIOR, MARKET_PARAMS, MARKET_PRIOR, adoption_sim, market_sim,
                     sample_adoption_prior, sample_market_prior)
from .curves import (CURVES, BassAnalogy, CurveBaseline, bass, bass_curve, damped_trend, gompertz, linear_trend,
                     logistic, naive)
from .adoption import (AdoptionABC, AdoptionABCAnalogy, BassABC, BassABCAnalogy, abc_distance, adoption_backtest,
                       adoption_t0, simulation_bank)
from .opinion import (BoundedConfidenceABM, OpinionPanel, distribution, mean_position, opinion_baselines,
                      score_opinion, shift_distribution, to_eval_frame, tvd, w1)
from .elasticity import (ElasticityShrinkage, PreEventMean, category_elasticity, elasticity_prompt,
                         elicit_elasticities, shuffle_elasticities, wape)
from .market import MarketABM
from .diffusion import (SeismicPredictor, cascade_counts, cumulative_curve, curve_rmse, hawkes_mean_field,
                        power_kernel, seismic_kernel, select_power_kernel, shape_forecast, shape_growth)
from .society import AgentSociety, ChoiceWorld, ShareCascadeWorld, SocietyAgent, SocietyRun, World, extract_json

__all__ = [
    # engines
    "adoption_sim", "market_sim", "ADOPTION_PRIOR", "ADOPTION_PARAMS", "MARKET_PRIOR", "MARKET_PARAMS",
    "sample_adoption_prior", "sample_market_prior",
    # adoption
    "AdoptionABC", "AdoptionABCAnalogy", "BassABC", "BassABCAnalogy", "simulation_bank", "abc_distance",
    "adoption_t0", "adoption_backtest",
    "CurveBaseline", "BassAnalogy", "CURVES", "naive", "linear_trend", "damped_trend", "logistic", "gompertz",
    "bass", "bass_curve",
    # opinion
    "OpinionPanel", "BoundedConfidenceABM", "opinion_baselines", "score_opinion", "to_eval_frame",
    "distribution", "shift_distribution", "mean_position", "w1", "tvd",
    # elasticity / market
    "ElasticityShrinkage", "PreEventMean", "category_elasticity", "shuffle_elasticities", "elasticity_prompt",
    "elicit_elasticities", "wape", "MarketABM",
    # diffusion
    "hawkes_mean_field", "power_kernel", "seismic_kernel", "select_power_kernel", "SeismicPredictor",
    "cascade_counts", "cumulative_curve", "shape_growth", "shape_forecast", "curve_rmse",
    # society
    "AgentSociety", "SocietyAgent", "SocietyRun", "World", "ShareCascadeWorld", "ChoiceWorld", "extract_json",
]

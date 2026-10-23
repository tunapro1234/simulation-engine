"""Agent-based models: adoption ABM + ABC + analogy prior (with Bass controls and curve baselines; an LLM ceiling
prior with placebo controls and a leak check in ``ceiling``), opinion drift
(bounded confidence vs persistence / mean shift), elasticity prior with shrinkage, repeat-purchase market ABM,
self-exciting cascade predictors (Hawkes / SEISMIC; measured: they do not beat the shape baseline), a turn-based
LLM agent society (measured: no better than a single LLM score) and economic-game societies (who-to-punish,
repeated PD, trust; infrastructure, no measured result in this port).

Grid and spatial models (written for this port, measured on synthetic lattices or a YJMob100K subsample): a lattice
core (``grid``: Grid, agent arrays, schedulers, Moran's I / dissimilarity / like-share / Gini), Schelling
segregation with the Zhang swap game and LLM movers (``schelling``), Sugarscape with spice, trade and an LLM
movement policy (``sugarscape``), HuMob mobility metrics, baselines and an EPR simulator (``mobility``; measured: EPR
loses to the per-user slot mode as a forecaster), greedy / CELF billboard placement with a temporal holdout
(``placement``) and a spatial SEIR with its ODE control, R0 estimators and a generative-agent isolation variant
(``spatial_sir``). Names that would clash or read ambiguously at this level (``grid.run``, ``sugarscape.gini``,
``placement.adoption`` / ``holdout`` / ``AdoptionRun``, ``mobility.clean``, ``spatial_sir.rule_responder``, the
module constants) stay in their submodule: ``import simengine.abm.placement as pl; pl.holdout(...)``.

See docs/modules/abm.md for when each one works, with measured numbers and provenance.
"""
from .engine import (ADOPTION_PARAMS, ADOPTION_PRIOR, MARKET_PARAMS, MARKET_PRIOR, adoption_sim, market_sim,
                     sample_adoption_prior, sample_market_prior)
from .curves import (CURVES, BassAnalogy, CurveBaseline, bass, bass_curve, damped_trend, gompertz, linear_trend,
                     logistic, naive)
from .adoption import (AdoptionABC, AdoptionABCAnalogy, BassABC, BassABCAnalogy, abc_distance, adoption_backtest,
                       adoption_t0, simulation_bank)
from .ceiling import ceiling_prompt, ceiling_upper, llm_ceiling_prior
from .opinion import (BoundedConfidenceABM, OpinionPanel, distribution, mean_position, opinion_baselines,
                      score_opinion, shift_distribution, to_eval_frame, tvd, w1)
from .elasticity import (ElasticityShrinkage, PreEventMean, category_elasticity, elasticity_prompt,
                         elicit_elasticities, shuffle_elasticities, wape)
from .market import MarketABM
from .diffusion import (SeismicPredictor, cascade_counts, cumulative_curve, curve_rmse, hawkes_mean_field,
                        power_kernel, seismic_kernel, select_power_kernel, shape_forecast, shape_growth)
from .society import (AgentSociety, ChoiceWorld, ShareCascadeWorld, SocietyAgent, SocietyRun, World, extract_json,
                      InvalidAnswer, PD_PAYOFFS, WTP_DESIGN, run_repeated_pd, run_trust_interaction,
                      run_who_to_punish)
from .grid import (EMPTY, NEIGHBOURHOODS, Agents, Grid, RunLog, dissimilarity, gini, make_rng, moran_permutation,
                   morans_i, random_lattice, random_order_step, same_type_share, spawn_rngs, synchronous_step)
from .schelling import (LLMMover, Schelling, SchellingRun, ZhangRun, ZhangSwap, parse_decision,
                        segregation_coefficient, threshold_responder, tipping_curve, unlike_pairs, zhang_payoff)
from .sugarscape import (LLMPolicy, Sugarscape, SugarscapeConfig, SugarscapeRun, gini_trajectory, load_capacity,
                         lorenz, spice_capacity, sugar_capacity)
from .mobility import (EPRMobility, MarkovMobility, dtw, from_cell, geobleu, global_slot_mode, global_slot_table,
                       mobility_baselines, slot_mode, to_cell, user_scores)
from .placement import (CellGrid, Exposure, Placement, coverage, exposures, greedy_placement, home_cells,
                        moore_share, online_bound, random_cells, reach_frequency, top_visited, toy_trajectories,
                        traj_columns)
from .spatial_sir import (EpiAgent, EpiParams, EpiRun, GenerativeEpidemic, GenerativeRun, SpatialSEIR,
                          cells_from_xy, epi_personas, final_size, isolation_prompt, load_yjmob100k, ode_agreement,
                          parse_isolation, r0_final_size, r0_growth, r0_index_case, seir_ode,
                          trajectories_from_records)

__all__ = [
    # engines
    "adoption_sim", "market_sim", "ADOPTION_PRIOR", "ADOPTION_PARAMS", "MARKET_PRIOR", "MARKET_PARAMS",
    "sample_adoption_prior", "sample_market_prior",
    # adoption
    "AdoptionABC", "AdoptionABCAnalogy", "BassABC", "BassABCAnalogy", "simulation_bank", "abc_distance",
    "adoption_t0", "adoption_backtest",
    "llm_ceiling_prior", "ceiling_upper", "ceiling_prompt",
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
    # economic-game societies
    "run_who_to_punish", "run_repeated_pd", "run_trust_interaction", "WTP_DESIGN", "PD_PAYOFFS", "InvalidAnswer",
    # grid core (simengine.abm.grid)
    "Grid", "Agents", "RunLog", "EMPTY", "NEIGHBOURHOODS", "random_lattice", "random_order_step", "synchronous_step",
    "make_rng", "spawn_rngs", "morans_i", "moran_permutation", "dissimilarity", "same_type_share", "gini",
    # schelling
    "Schelling", "SchellingRun", "tipping_curve", "unlike_pairs", "segregation_coefficient", "ZhangSwap", "ZhangRun",
    "zhang_payoff", "LLMMover", "parse_decision", "threshold_responder",
    # sugarscape
    "Sugarscape", "SugarscapeConfig", "SugarscapeRun", "LLMPolicy", "sugar_capacity", "spice_capacity",
    "load_capacity", "lorenz", "gini_trajectory",
    # mobility
    "to_cell", "from_cell", "geobleu", "dtw", "user_scores", "slot_mode", "global_slot_mode", "global_slot_table",
    "MarkovMobility", "EPRMobility", "mobility_baselines",
    # placement
    "traj_columns", "CellGrid", "Exposure", "exposures", "home_cells", "coverage", "reach_frequency", "Placement",
    "greedy_placement", "online_bound", "top_visited", "random_cells", "moore_share", "toy_trajectories",
    # spatial SEIR / generative epidemic (simengine.abm.spatial_sir)
    "EpiParams", "EpiRun", "SpatialSEIR", "seir_ode", "final_size", "r0_final_size", "r0_growth", "r0_index_case",
    "ode_agreement", "cells_from_xy", "trajectories_from_records", "load_yjmob100k", "EpiAgent", "epi_personas",
    "isolation_prompt", "parse_isolation", "GenerativeEpidemic", "GenerativeRun",
]

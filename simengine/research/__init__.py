"""Auto-research toolkit: pre-registration lock, sealed predictions, a turn-based research loop with paired-CI
acceptance and a held-out test evaluated once, declared replication targets, harness robustness sweeps, and
proposers (LLM observer, `claude -p`, random, grid, greedy, halving).

    import simengine as se
    space = se.research.Space({"k": (1, 50), "method": ["a", "b"]}, defaults={"k": 10, "method": "a"})
    loop = se.research.Loop(runner, space, proposer="random", primary="err", higher_is_better=False,
                            workdir="runs/demo", stop_after=3)
    out = loop.run()

Search and method modules (loaded lazily on first use, so `import simengine.research` does not pull in scipy or
scikit-learn; docs/modules/research/<module>.md each):

- `bandits`: UCB1, Thompson (Beta, Gaussian), successive rejects, LUCB1, a vectorised identification-rate
  simulator and `RacingProposer` for `Loop`.
- `bayesopt`: `GPProposer` (GP-EI / GP-UCB) and `TPEProposer` for `Loop`; `bayesopt.minimize` for full-space BO.
- `hyperband`: `hyperband.hyperband` (JMLR Alg. 1), `asha` (MLSys Alg. 2) and the multi-fidelity runner protocol
  (`MultiFidelity`, `UnitRunner`). The function is not re-exported here: `research.hyperband` stays the module.
- `tree_search`: AIDE-style `TreeSearch` (draft / debug / improve) and `MCTSSearch` (UCT over a `Space`).
- `prompt_evolution`: `opro`, `ape`, `promptbreeder` over a text, `select_and_test` (validation pick, seal, one test
  pass), `evolve` (both in one call).
- `hypotheses`: automated social science (`AutoSocialScience`, Manning et al. 2024) and the experiment outcome
  predictor (`forecast_experiment`, `score_condition_forecasts`, Hewitt et al. 2024).

    from simengine.research import GPProposer, RacingProposer, TreeSearch, asha, evolve
    loop = se.research.Loop(runner, space, proposer=GPProposer(acquisition="ei", seed=0), primary="err",
                            higher_is_better=False, stop_after=None, max_turns=20)
    se.research.bandits.simulate(se.research.SimulatedArms([0.5, 0.45], seed=1), "lucb", budget=500, reps=200)

Generic helper names (`bandits.simulate`, `bandits.compare`, `bayesopt.minimize`, `hypotheses.design`, ...) are
reached through their module only.
"""
from __future__ import annotations

import importlib
from typing import Any

from .loop import Loop, control_arm
from .prereg import (Prereg, add_addendum, canonical_bytes, compare_targets, declare_targets, file_sha256, grade_arms,
                     grade_sealed, lock_prereg, read_addendum, seal, seal_deviation, targets_sha256, verify_addenda,
                     verify_chain, verify_prereg, verify_seal, write_prereg)
from .proposers import (ClaudeCodeProposer, GreedyCoordinateProposer, GridProposer, HalvingProposer, LLMProposer,
                        ProposalContext, RandomProposer, build_prompt, claude_code_llm, make_proposer,
                        parse_observer_reply, successive_halving, validate_proposal)
from .space import (HarnessSpace, Space, config_diff, config_hash, harness_sweep, harness_variants,
                    one_change_variants, variant_label)
from .stats import bootstrap_mean, paired_bootstrap_diff

SUBMODULES = ("bandits", "bayesopt", "hyperband", "tree_search", "prompt_evolution", "hypotheses")

# Lazily re-exported names: module -> names (each must be in that module's namespace)
_LAZY_BY_MODULE = {
    "bandits": ("SimulatedArms", "BanditResult", "identify_best", "RacingProposer", "ucb1", "thompson_beta",
                "thompson_gaussian", "successive_rejects", "lucb", "lucb_beta", "sr_schedule", "AUDIBERT_BUBECK_2010"),
    "bayesopt": ("GPProposer", "TPEProposer", "SpaceEncoder", "TPE", "Parzen1D", "fit_gp", "gp_scores",
                 "expected_improvement", "upper_confidence_bound", "ucb_beta", "branin", "branin_space", "BRANIN_MIN",
                 "mixed_toy", "mixed_toy_space"),
    "hyperband": ("FidelityRunner", "MultiFidelity", "UnitRunner", "space_sampler", "pool_sampler",
                  "hyperband_brackets", "ASHA", "asha"),
    "tree_search": ("TreeSearch", "MCTSSearch", "ConfigOperators", "LLMOperators", "summarize_tree"),
    "prompt_evolution": ("PromptEvaluator", "PromptEvolutionResult", "opro", "ape", "promptbreeder", "select_and_test",
                         "evolve", "make_llm_scorer", "opro_meta_prompt", "ape_forward_prompt", "diversity_filter"),
    "hypotheses": ("AutoSocialScience", "AutoSocialScienceResult", "HypothesisRun", "Hypothesis", "Variable",
                   "DecisionLog", "propose_hypotheses", "parse_hypotheses", "InterviewSimulator",
                   "ConversationSimulator", "StructuralSimulator", "make_simulator", "linear_world", "market_world",
                   "predict_paths", "compare_paths", "ExperimentForecast", "forecast_experiment",
                   "score_effect_forecasts", "score_condition_forecasts"),
}
_LAZY = {name: mod for mod, names in _LAZY_BY_MODULE.items() for name in names}

__all__ = [
    "Loop", "control_arm", "Space", "config_hash", "config_diff",
    "Prereg", "write_prereg", "lock_prereg", "verify_prereg", "file_sha256",
    "seal", "verify_seal", "verify_chain", "grade_sealed", "canonical_bytes",
    "declare_targets", "targets_sha256", "compare_targets",
    "add_addendum", "read_addendum", "verify_addenda", "seal_deviation", "grade_arms",
    "HarnessSpace", "harness_variants", "one_change_variants", "variant_label", "harness_sweep",
    "LLMProposer", "ClaudeCodeProposer", "RandomProposer", "GridProposer", "GreedyCoordinateProposer",
    "HalvingProposer", "successive_halving", "make_proposer", "validate_proposal", "build_prompt",
    "parse_observer_reply", "claude_code_llm", "ProposalContext",
    "paired_bootstrap_diff", "bootstrap_mean",
    # search and method modules (lazy)
    *SUBMODULES, *_LAZY,
]


def __getattr__(name: str) -> Any:
    if name in SUBMODULES:
        return importlib.import_module(f".{name}", __name__)   # the import also sets the package attribute
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

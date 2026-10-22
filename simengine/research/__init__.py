"""Auto-research toolkit: pre-registration lock, sealed predictions, a turn-based research loop with paired-CI
acceptance and a held-out test evaluated once, declared replication targets, harness robustness sweeps, and
proposers (LLM observer, `claude -p`, random, grid, greedy, halving).

    import simengine as se
    space = se.research.Space({"k": (1, 50), "method": ["a", "b"]}, defaults={"k": 10, "method": "a"})
    loop = se.research.Loop(runner, space, proposer="random", primary="err", higher_is_better=False,
                            workdir="runs/demo", stop_after=3)
    out = loop.run()
"""
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
]

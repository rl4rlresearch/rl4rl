"""Frozen labels used for blinded process annotation."""

from enum import StrEnum


class ResearchMove(StrEnum):
    LOCAL_REFINEMENT = "local_refinement"
    HYPERPARAMETER_CHANGE = "hyperparameter_change"
    MECHANISM_MODIFICATION = "mechanism_modification"
    ALTERNATIVE_MECHANISM = "alternative_mechanism"
    ABLATION = "ablation"
    COUNTEREXAMPLE_OR_BOUNDARY = "counterexample_or_boundary"
    CONFOUND_TEST = "confound_test"
    REPLICATION = "replication"
    REVERSION = "reversion"
    RECOMBINATION = "recombination"
    EVALUATOR_EXPLOIT = "evaluator_exploit"
    UNRESOLVED = "unresolved"


class EpistemicPurpose(StrEnum):
    IMPROVE = "improve"
    CONFIRM = "confirm"
    FALSIFY = "falsify"
    DISTINGUISH_EXPLANATIONS = "distinguish_explanations"
    DIAGNOSE_FAILURE = "diagnose_failure"
    TEST_ROBUSTNESS = "test_robustness"
    FIND_BOUNDARY = "find_boundary"
    REPRODUCE = "reproduce"
    UNGUIDED_EXPLORE = "unguided_explore"


class EvidenceResponse(StrEnum):
    RETAIN = "retain"
    WEAKEN = "weaken"
    REJECT = "reject"
    NARROW = "narrow"
    ADD_AUXILIARY_EXPLANATION = "add_auxiliary_explanation"
    REPLICATE = "replicate"
    BLAME_IMPLEMENTATION = "blame_implementation"
    BLAME_NOISE = "blame_noise"
    SILENT_HYPOTHESIS_CHANGE = "silent_hypothesis_change"
    NO_UPDATE = "no_update"


class ResearchDisplacement(StrEnum):
    D0 = "D0_same_hypothesis_same_mechanism"
    D1 = "D1_local_variant"
    D2 = "D2_mechanism_variant"
    D3 = "D3_alternative_mechanism"
    D4 = "D4_assumption_or_problem_change"
    D5 = "D5_problem_reformulation"

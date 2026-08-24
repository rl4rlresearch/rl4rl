"""Process-first summaries computed from blinded annotations."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable


_DISPLACEMENT = {
    "D0_same_hypothesis_same_mechanism": 0,
    "D1_local_variant": 1,
    "D2_mechanism_variant": 2,
    "D3_alternative_mechanism": 3,
    "D4_assumption_or_problem_change": 4,
    "D5_problem_reformulation": 5,
}


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _entropy(values: list[str]) -> float | None:
    if not values:
        return None
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def transition_matrix(labels: Iterable[str]) -> dict[str, dict[str, int]]:
    values = list(labels)
    result: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for left, right in zip(values, values[1:]):
        result[left][right] += 1
    return {
        left: dict(sorted(rights.items()))
        for left, rights in sorted(result.items())
    }


def _word_set(value: object) -> set[str]:
    if not isinstance(value, str):
        return set()
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def _mean_pairwise_jaccard_distance(values: list[str]) -> float | None:
    sets = [_word_set(value) for value in values]
    distances = []
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            union = left | right
            if union:
                distances.append(1.0 - len(left & right) / len(union))
    return sum(distances) / len(distances) if distances else None


def summarize_decisions(decisions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Annotation-free diversity, branching, memory-use, and update diagnostics."""

    rows = sorted(list(decisions), key=lambda row: row.get("opportunity", -1))
    explanations: list[str] = []
    experiments: list[str] = []
    challenge_uptake: list[bool] = []
    explicit_updates: list[bool] = []
    memory_reuse: list[bool] = []
    parent_counts: Counter[str] = Counter()
    candidate_ids = set()
    for row in rows:
        note = row.get("lab_note_before") or {}
        if not isinstance(note, dict):
            note = {}
        explanation = note.get("research_current_explanation")
        experiment = note.get("research_next_experiment")
        if isinstance(explanation, str) and explanation:
            explanations.append(explanation)
        if isinstance(experiment, str) and experiment:
            experiments.append(experiment)
        if row.get("challenge_active") is True:
            challenged = note.get("research_challenged_assumption")
            challenge_uptake.append(
                isinstance(challenged, str)
                and bool(challenged.strip())
                and challenged != "not_requested"
            )
        after = row.get("lab_note_after") or {}
        if isinstance(after, dict):
            changed = after.get("changed_explanation")
            if changed in {"yes", "no"}:
                explicit_updates.append(changed == "yes")
        evidence = str(note.get("research_evidence") or "")
        visible = [
            identifier
            for identifier in row.get("memory_entry_ids", [])
            if isinstance(identifier, str) and identifier
        ]
        if visible:
            memory_reuse.append(any(identifier in evidence for identifier in visible))
        parent = row.get("parent_id")
        if isinstance(parent, str) and parent:
            parent_counts[parent] += 1
        candidate = row.get("candidate_id")
        if isinstance(candidate, str) and candidate:
            candidate_ids.add(candidate)

    normalized_explanations = {" ".join(sorted(_word_set(value))) for value in explanations}
    normalized_experiments = {" ".join(sorted(_word_set(value))) for value in experiments}
    return {
        "schema_name": "ResearchDecisionTelemetrySummary",
        "schema_version": "1.0",
        "decisions": len(rows),
        "idea_and_logic_diversity": {
            "explanations_observed": len(explanations),
            "unique_explanation_rate": (
                len(normalized_explanations) / len(explanations)
                if explanations
                else None
            ),
            "mean_pairwise_explanation_jaccard_distance": (
                _mean_pairwise_jaccard_distance(explanations)
            ),
            "experiments_observed": len(experiments),
            "unique_experiment_rate": (
                len(normalized_experiments) / len(experiments)
                if experiments
                else None
            ),
            "mean_pairwise_experiment_jaccard_distance": (
                _mean_pairwise_jaccard_distance(experiments)
            ),
        },
        "lineage": {
            "unique_candidate_ids": len(candidate_ids),
            "unique_parent_ids": len(parent_counts),
            "mean_children_per_used_parent": (
                sum(parent_counts.values()) / len(parent_counts)
                if parent_counts
                else None
            ),
            "maximum_children_from_one_parent": max(parent_counts.values(), default=0),
            "children_by_parent": dict(sorted(parent_counts.items())),
        },
        "intervention_process": {
            "challenge_uptake_rate": _rate(challenge_uptake),
            "explicit_explanation_change_rate": _rate(explicit_updates),
            "visible_memory_id_citation_rate": _rate(memory_reuse),
        },
        "warning": (
            "Lexical diversity is a diagnostic, not a semantic-diversity endpoint. "
            "Use blinded annotations for the primary process outcomes."
        ),
    }


def summarize_annotations(annotations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(annotations)
    moves = [row["research_move"] for row in rows if row.get("research_move")]
    purposes = [
        row["epistemic_purpose"] for row in rows if row.get("epistemic_purpose")
    ]
    displacements = [
        _DISPLACEMENT[row["research_displacement"]]
        for row in rows
        if row.get("research_displacement") in _DISPLACEMENT
    ]
    discriminating = [
        row["discriminating_experiment"]
        for row in rows
        if type(row.get("discriminating_experiment")) is bool
    ]
    contradicted = [row for row in rows if row.get("prediction_contradicted") is True]
    responsive_values = {
        "weaken",
        "reject",
        "narrow",
        "replicate",
        "add_auxiliary_explanation",
    }
    evidence_responsive = [
        row.get("evidence_response") in responsive_values for row in contradicted
    ]
    aligned = [
        row["rationale_action_aligned"]
        for row in rows
        if type(row.get("rationale_action_aligned")) is bool
    ]
    supported = [
        row["interpretation_supported_by_result"]
        for row in rows
        if type(row.get("interpretation_supported_by_result")) is bool
    ]
    persistence = [left == right for left, right in zip(moves, moves[1:])]

    hypothesis_first: dict[str, int] = {}
    hypothesis_last: dict[str, int] = {}
    for index, row in enumerate(rows):
        hypothesis = row.get("hypothesis_id")
        if hypothesis:
            hypothesis_first.setdefault(hypothesis, index)
            hypothesis_last[hypothesis] = index
    lifetimes = [
        hypothesis_last[key] - first + 1 for key, first in hypothesis_first.items()
    ]

    return {
        "schema_name": "ResearchProcessSummary",
        "schema_version": "1.0",
        "annotated_decisions": len(rows),
        "primary": {
            "discriminating_experiment_rate": _rate(discriminating),
            "mean_research_displacement": (
                sum(displacements) / len(displacements) if displacements else None
            ),
            "assumption_displacement_rate_d4_or_d5": _rate(
                [value >= 4 for value in displacements]
            ),
            "evidence_responsive_revision_rate_given_contradiction": _rate(
                evidence_responsive
            ),
        },
        "secondary": {
            "research_move_entropy_bits": _entropy(moves),
            "research_move_persistence_rate": _rate(persistence),
            "rationale_action_alignment_rate": _rate(aligned),
            "interpretation_support_rate": _rate(supported),
            "mean_hypothesis_lifetime_decisions": (
                sum(lifetimes) / len(lifetimes) if lifetimes else None
            ),
            "research_move_counts": dict(sorted(Counter(moves).items())),
            "epistemic_purpose_counts": dict(sorted(Counter(purposes).items())),
            "research_move_transition_matrix": transition_matrix(moves),
            "epistemic_purpose_transition_matrix": transition_matrix(purposes),
        },
        "note": (
            "These are process outcomes. Final benchmark performance must be reported "
            "separately as a downstream descriptive outcome."
        ),
    }

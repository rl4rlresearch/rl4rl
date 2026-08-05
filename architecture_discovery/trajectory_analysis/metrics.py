"""Predefined descriptive metrics for autonomous research trajectories."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Iterable, Mapping

from .annotations import ResolvedAnnotation
from .schemas import BoundaryClass, Decision, TrajectoryEvent


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _entropy(values: list[str]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    counts = Counter(values)
    entropy = -sum(
        (count / len(values)) * math.log2(count / len(values))
        for count in counts.values()
    )
    maximum = math.log2(len(counts))
    normalized = entropy / maximum if maximum else 0.0
    return entropy, normalized


def _max_preserving_streak(
    events: list[TrajectoryEvent], annotations: Mapping[str, ResolvedAnnotation]
) -> int:
    longest = current = 0
    for event in events:
        label = annotations.get(event.event_id)
        if label and label.boundary_class is BoundaryClass.ONTOLOGY_PRESERVING:
            current += 1
            longest = max(longest, current)
        elif label:
            current = 0
    return longest


def summarize_run(
    events: Iterable[TrajectoryEvent],
    annotations: Mapping[str, ResolvedAnnotation],
    *,
    accuracy_threshold: float,
    external_frontier_parameters: int | None,
) -> dict[str, object]:
    ordered = sorted(events, key=lambda event: event.sequence_index)
    if not ordered:
        raise ValueError("cannot summarize an empty run")
    if len({event.run_id for event in ordered}) != 1:
        raise ValueError("summarize_run received multiple run ids")
    evaluated = [event for event in ordered if event.accuracy is not None]
    valid = [event for event in evaluated if event.valid is True]
    qualifying = [
        event
        for event in valid
        if event.accuracy is not None
        and event.accuracy >= accuracy_threshold
        and event.parameter_count is not None
    ]
    frontier: list[dict[str, int | float | str]] = []
    best: int | None = None
    for event in qualifying:
        assert event.parameter_count is not None
        if best is None or event.parameter_count < best:
            best = event.parameter_count
            frontier.append(
                {
                    "sequence_index": event.sequence_index,
                    "candidate_id": event.candidate_id or "",
                    "parameter_count": event.parameter_count,
                    "accuracy": event.accuracy or 0.0,
                }
            )

    edit_events = [event for event in ordered if event.event_id in annotations]
    boundary_counts = Counter(
        annotations[event.event_id].boundary_class.value for event in edit_events
    )
    family_counts = Counter(
        annotations[event.event_id].edit_family.value for event in edit_events
    )
    ontology_changes = [
        event
        for event in edit_events
        if annotations[event.event_id].boundary_class
        in {BoundaryClass.ONTOLOGY_CHANGING, BoundaryClass.MIXED}
    ]
    accepted_changes = [
        event for event in ontology_changes if event.decision is Decision.ACCEPTED
    ]
    qualifying_ids = {event.event_id for event in qualifying}
    qualifying_changes = [
        event for event in ontology_changes if event.event_id in qualifying_ids
    ]
    family_entropy, normalized_family_entropy = _entropy(
        [annotations[event.event_id].edit_family.value for event in edit_events]
    )
    fingerprints = [
        event.architecture_fingerprint
        for event in evaluated
        if event.architecture_fingerprint is not None
    ]
    revisits = len(fingerprints) - len(set(fingerprints))
    invalid_candidates = {
        event.candidate_id
        for event in ordered
        if event.candidate_id is not None and event.valid is False
    }
    invalid_ancestor_descendants = sum(
        event.valid is True and any(parent in invalid_candidates for parent in event.parent_ids)
        for event in ordered
    )
    stop_events = [event for event in ordered if event.stop_claim]
    stop_claim = stop_events[-1].stop_claim if stop_events else None
    claims_floor = bool(
        stop_claim
        and any(
            phrase in stop_claim.lower()
            for phrase in ("optimal", "minimum", "floor", "cannot improve", "no improvement")
        )
    )
    premature_stop = bool(
        claims_floor
        and best is not None
        and external_frontier_parameters is not None
        and best > external_frontier_parameters
    )
    first_change = min(
        (event.sequence_index for event in ontology_changes), default=None
    )
    first_qualifying = min(
        (event.sequence_index for event in qualifying), default=None
    )
    result: dict[str, object] = {
        "run_id": ordered[0].run_id,
        "paradigm": ordered[0].paradigm.value,
        "total_events": len(ordered),
        "evaluated_candidates": len(evaluated),
        "accepted_candidates": sum(
            event.decision is Decision.ACCEPTED for event in ordered
        ),
        "valid_candidates": len(valid),
        "qualifying_candidates": len(qualifying),
        "initial_qualifying_parameters": (
            qualifying[0].parameter_count if qualifying else None
        ),
        "best_qualifying_parameters": best,
        "external_frontier_parameters": external_frontier_parameters,
        "frontier_gap_parameters": (
            best - external_frontier_parameters
            if best is not None and external_frontier_parameters is not None
            else None
        ),
        "frontier_gap_ratio": (
            best / external_frontier_parameters
            if best is not None and external_frontier_parameters is not None
            else None
        ),
        "frontier_improvements": len(frontier),
        "frontier_progression": frontier,
        "edit_family_counts": dict(sorted(family_counts.items())),
        "boundary_class_counts": dict(sorted(boundary_counts.items())),
        "edit_family_entropy_bits": family_entropy,
        "normalized_edit_family_entropy": normalized_family_entropy,
        "ontology_change_attempt_rate": _rate(len(ontology_changes), len(edit_events)),
        "ontology_change_acceptance_rate": _rate(
            len(accepted_changes), len(ontology_changes)
        ),
        "ontology_change_qualification_rate": _rate(
            len(qualifying_changes), len(ontology_changes)
        ),
        "acceptance_rate": _rate(
            sum(event.decision is Decision.ACCEPTED for event in evaluated),
            len(evaluated),
        ),
        "invalid_rate": _rate(
            sum(event.valid is False for event in evaluated), len(evaluated)
        ),
        "rollback_rate": _rate(
            sum(event.decision is Decision.ROLLBACK for event in ordered),
            len(ordered),
        ),
        "fingerprint_coverage": _rate(len(fingerprints), len(evaluated)),
        "architecture_revisit_rate": _rate(revisits, len(fingerprints)),
        "unique_architecture_fingerprints": len(set(fingerprints)),
        "max_ontology_preserving_streak": _max_preserving_streak(ordered, annotations),
        "first_ontology_change_sequence": first_change,
        "first_qualifying_sequence": first_qualifying,
        "first_frontier_improvement_sequence": (
            frontier[0]["sequence_index"] if frontier else None
        ),
        "invalid_parent_to_valid_child_count": invalid_ancestor_descendants,
        "stop_claim": stop_claim,
        "premature_frontier_claim": premature_stop,
    }
    return result


def summarize_paradigms(run_summaries: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for summary in run_summaries:
        grouped[str(summary["paradigm"])].append(summary)
    results: list[dict[str, object]] = []
    numeric_fields = (
        "best_qualifying_parameters",
        "frontier_improvements",
        "ontology_change_attempt_rate",
        "architecture_revisit_rate",
        "normalized_edit_family_entropy",
        "invalid_rate",
    )
    for paradigm, summaries in sorted(grouped.items()):
        result: dict[str, object] = {
            "paradigm": paradigm,
            "independent_runs": len(summaries),
            "analysis_scope": "descriptive_only",
            "inference_note": (
                "No confidence intervals or hypothesis tests: fewer than five independent runs."
                if len(summaries) < 5
                else "Descriptive aggregation only; any inferential model must be preregistered separately."
            ),
        }
        for field in numeric_fields:
            values = [
                float(summary[field])
                for summary in summaries
                if summary.get(field) is not None
            ]
            result[f"mean_{field}"] = mean(values) if values else None
            result[f"median_{field}"] = median(values) if values else None
        results.append(result)
    return results

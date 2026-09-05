"""Two-objective archive accounting for the HAR accuracy/MAC campaign.

Hypervolume is a run statistic; it is never assigned to individual candidates.
The fixed reference is (accuracy=0, MACs=2,000,000), with linear axes.
"""

from __future__ import annotations

import math

PORTFOLIO_RULE = "archive_pareto_fill_then_replace_selected_lineage_v1"
SINGLE_RULE = "archive_pareto_replace_incumbent_v1"
PARENT_RULE = "retained_least_selected_then_oldest_then_id_v1"
MAX_MACS = 2_000_000


def point(metrics):
    accuracy, cost = metrics["validation_accuracy"], metrics["inference_macs"]
    if (
        isinstance(accuracy, bool)
        or isinstance(cost, bool)
        or not isinstance(accuracy, int | float)
        or not isinstance(cost, int | float)
        or not math.isfinite(accuracy)
        or not math.isfinite(cost)
        or not 0 <= accuracy <= 1
        or not 0 < cost <= MAX_MACS
    ):
        raise ValueError("invalid accuracy/MAC objective pair")
    return float(accuracy), float(cost)


def dominates(left, right):
    return (left[0] >= right[0] and left[1] <= right[1]) and left != right


def frontier(candidates):
    """Deduplicate objective ties in favor of the oldest evaluated design."""
    ordered = sorted(candidates, key=lambda c: (c.created_opportunity, c.candidate_id))
    kept = []
    for candidate in ordered:
        value = point(candidate.metrics)
        if any(
            point(other.metrics) == value or dominates(point(other.metrics), value)
            for other in kept
        ):
            continue
        kept = [other for other in kept if not dominates(value, point(other.metrics))]
        kept.append(candidate)
    return kept


def hypervolume(candidates):
    """Normalized union area in [0,1] x [0,MAX_MACS], higher is better."""
    points = sorted({point(c.metrics) for c in candidates}, key=lambda p: p[1])
    best_accuracy = 0.0
    area = 0.0
    for accuracy, cost in points:
        if accuracy > best_accuracy:
            area += (accuracy - best_accuracy) * (MAX_MACS - cost) / MAX_MACS
            best_accuracy = accuracy
    return area


def summary(candidates):
    values = list(candidates)
    nondominated = frontier(values)
    return {
        "hypervolume": hypervolume(nondominated),
        "frontier_ids": [c.candidate_id for c in nondominated],
        "archive_size": len(values),
        "reference_accuracy": 0.0,
        "reference_macs": MAX_MACS,
        "normalization": "linear_area_divided_by_2000000",
    }


def retain(controller, candidate, parent):
    """Every novel archive-frontier point is admitted, preserving lineage geometry.

    All valid candidates (including discarded ones) remain in state.candidates.
    Visible memory is independently bounded to one or K source designs. It is
    deliberately not the entire archive and may contain subsequently dominated
    designs; admitting a child changes only its selected lineage when full.
    """
    state = controller.state
    ids = {c.candidate_id for c in frontier(state.candidates.values())}
    if candidate.candidate_id not in ids:
        return False, "archive_dominated_or_objective_tie", None
    if not controller.condition.has_portfolio:
        state.incumbent_id = candidate.candidate_id
        state.portfolio_ids = [candidate.candidate_id]
        return True, "archive_pareto_replaced_incumbent", parent.candidate_id
    if len(state.portfolio_ids) < controller.spec.portfolio_capacity:
        state.portfolio_ids.append(candidate.candidate_id)
        state.incumbent_id = candidate.candidate_id
        return True, "archive_pareto_filled_slot", None
    candidate.selected_count = parent.selected_count
    state.portfolio_ids[state.portfolio_ids.index(parent.candidate_id)] = (
        candidate.candidate_id
    )
    state.incumbent_id = candidate.candidate_id
    return True, "archive_pareto_replaced_selected_lineage", parent.candidate_id

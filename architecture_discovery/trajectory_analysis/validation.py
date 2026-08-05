"""Fail-closed structural validation for normalized trajectories."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .schemas import EventKind, Paradigm, TrajectoryEvent


@dataclass(frozen=True, slots=True)
class ValidationReport:
    run_count: int
    event_count: int
    candidate_count: int
    paradigms: tuple[str, ...]


def validate_trajectories(events: Iterable[TrajectoryEvent]) -> ValidationReport:
    materialized = list(events)
    if not materialized:
        raise ValueError("no trajectory events were loaded")
    event_ids = [event.event_id for event in materialized]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event identifiers must be globally unique")

    by_run: dict[str, list[TrajectoryEvent]] = defaultdict(list)
    for event in materialized:
        by_run[event.run_id].append(event)

    candidates: set[tuple[str, str]] = set()
    paradigms: set[Paradigm] = set()
    for run_id, run_events in by_run.items():
        run_events.sort(key=lambda item: item.sequence_index)
        run_paradigms = {event.paradigm for event in run_events}
        if len(run_paradigms) != 1:
            raise ValueError(f"run {run_id} mixes paradigms")
        paradigms.update(run_paradigms)
        sequences = [event.sequence_index for event in run_events]
        if len(sequences) != len(set(sequences)):
            raise ValueError(f"run {run_id} has duplicate sequence indexes")
        if sequences != sorted(sequences):
            raise ValueError(f"run {run_id} is not in sequence order")
        stop_positions = [
            index for index, event in enumerate(run_events) if event.kind is EventKind.STOP
        ]
        if stop_positions and stop_positions != [len(run_events) - 1]:
            raise ValueError(f"run {run_id} has a non-terminal or repeated stop event")

        introduced: set[str] = set()
        parent_graph: dict[str, set[str]] = defaultdict(set)
        for event in run_events:
            for parent_id in event.parent_ids:
                if parent_id not in introduced:
                    raise ValueError(
                        f"run {run_id} event {event.event_id} references parent "
                        f"{parent_id} before it appears"
                    )
            if event.candidate_id is not None:
                parent_graph[event.candidate_id].update(event.parent_ids)
                introduced.add(event.candidate_id)
                candidates.add((run_id, event.candidate_id))
            if event.valid is False and event.decision.value not in {"invalid", "none"}:
                raise ValueError(
                    f"run {run_id} event {event.event_id} accepts or rejects an invalid candidate"
                )
            if event.accuracy is not None and event.valid is None:
                raise ValueError(
                    f"run {run_id} event {event.event_id} has accuracy but no validity flag"
                )
            if event.parameter_count is not None and event.valid is None:
                raise ValueError(
                    f"run {run_id} event {event.event_id} has parameters but no validity flag"
                )

        active: set[str] = set()
        complete: set[str] = set()

        def visit(candidate_id: str) -> None:
            if candidate_id in active:
                raise ValueError(f"run {run_id} candidate lineage contains a cycle")
            if candidate_id in complete:
                return
            active.add(candidate_id)
            for parent_id in parent_graph.get(candidate_id, set()):
                visit(parent_id)
            active.remove(candidate_id)
            complete.add(candidate_id)

        for candidate_id in parent_graph:
            visit(candidate_id)

    return ValidationReport(
        run_count=len(by_run),
        event_count=len(materialized),
        candidate_count=len(candidates),
        paradigms=tuple(sorted(paradigm.value for paradigm in paradigms)),
    )

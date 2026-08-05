from __future__ import annotations

import pytest

from trajectory_analysis.schemas import (
    Decision,
    EventKind,
    Paradigm,
    RawReference,
    TrajectoryEvent,
)
from trajectory_analysis.validation import validate_trajectories


DIGEST = "a" * 64


def event(
    sequence: int,
    candidate: str | None,
    parents: tuple[str, ...] = (),
    *,
    kind: EventKind = EventKind.EVALUATION,
) -> TrajectoryEvent:
    return TrajectoryEvent(
        run_id="run-1",
        event_id=f"source:{sequence}",
        paradigm=Paradigm.OPENEVOLVE,
        sequence_index=sequence,
        kind=kind,
        decision=Decision.NONE,
        raw_reference=RawReference("source", sequence, DIGEST),
        candidate_id=candidate,
        parent_ids=parents,
        accuracy=0.99 if candidate else None,
        parameter_count=100 if candidate else None,
        valid=True if candidate else None,
    )


def test_trajectory_round_trip_is_typed_and_lossless():
    original = event(1, "child", ("parent",))
    assert TrajectoryEvent.from_dict(original.to_dict()) == original


@pytest.mark.parametrize("accuracy", [float("nan"), float("inf"), -0.1, 1.1])
def test_accuracy_must_be_finite_fraction(accuracy: float):
    with pytest.raises(ValueError):
        TrajectoryEvent(
            run_id="run",
            event_id="event",
            paradigm=Paradigm.AUTORESEARCH,
            sequence_index=0,
            kind=EventKind.EVALUATION,
            decision=Decision.NONE,
            raw_reference=RawReference("source", 0, DIGEST),
            candidate_id="candidate",
            accuracy=accuracy,
            valid=True,
        )


def test_validator_rejects_forward_parent_and_nonterminal_stop():
    with pytest.raises(ValueError, match="before it appears"):
        validate_trajectories([event(0, "child", ("future",)), event(1, "future")])
    with pytest.raises(ValueError, match="non-terminal"):
        validate_trajectories(
            [event(0, None, kind=EventKind.STOP), event(1, "candidate")]
        )


def test_validator_accepts_ordered_lineage():
    report = validate_trajectories(
        [event(0, "seed"), event(1, "child", ("seed",)), event(2, None, kind=EventKind.STOP)]
    )
    assert report.run_count == 1
    assert report.candidate_count == 2


def test_validator_detects_cycle_created_by_reused_candidate_identifier():
    with pytest.raises(ValueError, match="contains a cycle"):
        validate_trajectories(
            [
                event(0, "a"),
                event(1, "b", ("a",)),
                event(2, "a", ("b",)),
            ]
        )

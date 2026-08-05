"""Loss-minimizing adapters for three autonomous-research trajectory formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .manifest import SourceSpec
from .schemas import Decision, EventKind, RawReference, TrajectoryEvent


def _first(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return None


def _integer(value: Any, name: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if str(value).strip() not in {str(result), f"{result}.0"}:
        raise ValueError(f"{name} must be an exact integer")
    return result


def _number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error


def _boolean(value: Any, name: str) -> bool | None:
    if value is None:
        return None
    if type(value) is bool:
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{name} must be boolean")


def _parents(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        parents = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if stripped.startswith("["):
            parents = json.loads(stripped)
        else:
            parents = [part.strip() for part in stripped.split(",")]
    else:
        raise ValueError("parent ids must be a list or comma-separated string")
    if not isinstance(parents, list) or not all(
        isinstance(parent, str) and parent for parent in parents
    ):
        raise ValueError("parent ids must contain non-empty strings")
    return tuple(parents)


def _decision(row: Mapping[str, Any]) -> Decision:
    status = str(_first(row, "status", "decision") or "").strip().lower()
    valid = _boolean(_first(row, "valid", "is_valid"), "valid")
    accepted = _boolean(_first(row, "accepted", "selected", "keep"), "accepted")
    if valid is False or status in {"invalid", "error", "failed"}:
        return Decision.INVALID
    if status in {"rollback", "reverted"}:
        return Decision.ROLLBACK
    if accepted is True or status in {"accepted", "keep", "best", "selected"}:
        return Decision.ACCEPTED
    if accepted is False or status in {"rejected", "discarded", "worse"}:
        return Decision.REJECTED
    return Decision.NONE


def _event_kind(row: Mapping[str, Any], decision: Decision) -> EventKind:
    status = str(_first(row, "status", "event", "kind") or "").strip().lower()
    if status in {"stop", "stopped", "complete", "completed"}:
        return EventKind.STOP
    if decision is Decision.ROLLBACK:
        return EventKind.ROLLBACK
    return EventKind.EVALUATION


def _normalize(
    spec: SourceSpec,
    row: Mapping[str, Any],
    record_index: int,
) -> TrajectoryEvent:
    sequence = _integer(
        _first(row, "sequence", "iteration", "step", "index"), "sequence"
    )
    decision = _decision(row)
    kind = _event_kind(row, decision)
    candidate = _first(
        row, "candidate_id", "commit", "commit_hash", "rollout_id", "program_id"
    )
    if kind is EventKind.STOP:
        candidate = None
    elif candidate is None:
        raise ValueError(f"record {record_index} has no candidate identifier")
    event_id = f"{spec.source_id}:{record_index}"
    known = {
        "sequence", "iteration", "step", "index", "candidate_id", "commit",
        "commit_hash", "rollout_id", "program_id", "parent_ids", "parents",
        "parent_id", "parent_commit", "timestamp", "time", "accuracy", "score",
        "exact_match", "parameters", "parameter_count", "num_params", "valid",
        "is_valid", "accepted", "selected", "keep", "status", "decision", "event",
        "kind", "description", "message", "summary", "architecture_fingerprint",
        "fingerprint", "stop_claim", "stopping_reason",
    }
    metadata = {key: value for key, value in row.items() if key not in known}
    return TrajectoryEvent(
        run_id=spec.run_id,
        event_id=event_id,
        paradigm=spec.paradigm,
        sequence_index=sequence,
        kind=kind,
        decision=decision,
        raw_reference=RawReference(spec.source_id, record_index, spec.sha256),
        candidate_id=None if candidate is None else str(candidate),
        parent_ids=_parents(_first(row, "parent_ids", "parents", "parent_id", "parent_commit")),
        timestamp=_first(row, "timestamp", "time"),
        accuracy=_number(_first(row, "accuracy", "exact_match", "score"), "accuracy"),
        parameter_count=_integer(
            _first(row, "parameters", "parameter_count", "num_params"),
            "parameter_count",
            optional=True,
        ),
        valid=_boolean(_first(row, "valid", "is_valid"), "valid"),
        description=_first(row, "description", "message", "summary"),
        architecture_fingerprint=_first(row, "architecture_fingerprint", "fingerprint"),
        stop_claim=_first(row, "stop_claim", "stopping_reason"),
        metadata=metadata,
    )


def _jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL line {line_number} is not an object")
            yield value


def load_source(spec: SourceSpec, path: str | Path) -> list[TrajectoryEvent]:
    source_path = Path(path)
    if spec.adapter == "autoresearch_tsv_v1":
        with source_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    elif spec.adapter in {"openevolve_jsonl_v1", "ttt_jsonl_v1"}:
        rows = list(_jsonl(source_path))
    else:  # SourceSpec already rejects this; keep the execution boundary closed.
        raise ValueError(f"unsupported adapter: {spec.adapter}")
    if not rows:
        raise ValueError(f"trajectory source is empty: {source_path}")
    events = [_normalize(spec, row, index) for index, row in enumerate(rows)]
    return sorted(events, key=lambda event: event.sequence_index)

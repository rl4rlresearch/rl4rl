"""Typed, strict records shared by trajectory adapters and analyses."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class Paradigm(StrEnum):
    OPENEVOLVE = "openevolve"
    AUTORESEARCH = "autoresearch"
    TTT_DISCOVER = "ttt_discover"


class EventKind(StrEnum):
    RUN_START = "run_start"
    PROPOSAL = "proposal"
    EVALUATION = "evaluation"
    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"
    ROLLBACK = "rollback"
    STOP = "stop"


class Decision(StrEnum):
    NONE = "none"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INVALID = "invalid"
    ROLLBACK = "rollback"


class BoundaryClass(StrEnum):
    NONE = "none"
    ONTOLOGY_PRESERVING = "ontology_preserving"
    ONTOLOGY_CHANGING = "ontology_changing"
    MIXED = "mixed"
    UNCLASSIFIED = "unclassified"


class EditFamily(StrEnum):
    NONE = "none"
    SCALE = "scale"
    DEPTH = "depth"
    WIDTH = "width"
    ATTENTION = "attention"
    EMBEDDINGS = "embeddings"
    POSITIONAL = "positional"
    NORMALIZATION = "normalization"
    FEEDFORWARD = "feedforward"
    PARAMETER_TYING = "parameter_tying"
    TOKENIZATION = "tokenization"
    CURRICULUM = "curriculum"
    OPTIMIZER = "optimizer"
    VERIFIER = "verifier"
    OTHER = "other"


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if any(character.isspace() for character in value):
        raise ValueError(f"{name} may not contain whitespace")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value


def _finite_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class RawReference:
    source_id: str
    record_index: int
    source_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.source_id, "source_id")
        if isinstance(self.record_index, bool) or self.record_index < 0:
            raise ValueError("record_index must be a non-negative integer")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class TrajectoryEvent:
    run_id: str
    event_id: str
    paradigm: Paradigm
    sequence_index: int
    kind: EventKind
    decision: Decision
    raw_reference: RawReference
    candidate_id: str | None = None
    parent_ids: tuple[str, ...] = ()
    timestamp: str | None = None
    accuracy: float | None = None
    parameter_count: int | None = None
    valid: bool | None = None
    description: str | None = None
    architecture_fingerprint: str | None = None
    stop_claim: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.run_id, "run_id")
        _identifier(self.event_id, "event_id")
        if isinstance(self.sequence_index, bool) or self.sequence_index < 0:
            raise ValueError("sequence_index must be a non-negative integer")
        if self.candidate_id is not None:
            _identifier(self.candidate_id, "candidate_id")
        for parent_id in self.parent_ids:
            _identifier(parent_id, "parent_id")
        accuracy = _finite_number(self.accuracy, "accuracy")
        if accuracy is not None and not 0.0 <= accuracy <= 1.0:
            raise ValueError("accuracy must be between 0 and 1")
        if self.parameter_count is not None:
            if (
                isinstance(self.parameter_count, bool)
                or not isinstance(self.parameter_count, int)
                or self.parameter_count <= 0
            ):
                raise ValueError("parameter_count must be a positive integer")
        if self.valid is not None and type(self.valid) is not bool:
            raise ValueError("valid must be boolean or null")
        _optional_string(self.timestamp, "timestamp")
        _optional_string(self.description, "description")
        _optional_string(self.architecture_fingerprint, "architecture_fingerprint")
        _optional_string(self.stop_claim, "stop_claim")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        if self.kind is EventKind.STOP and self.candidate_id is not None:
            raise ValueError("stop events may not introduce a candidate")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["paradigm"] = self.paradigm.value
        payload["kind"] = self.kind.value
        payload["decision"] = self.decision.value
        payload["parent_ids"] = list(self.parent_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrajectoryEvent":
        allowed = {
            "run_id", "event_id", "paradigm", "sequence_index", "kind",
            "decision", "raw_reference", "candidate_id", "parent_ids",
            "timestamp", "accuracy", "parameter_count", "valid", "description",
            "architecture_fingerprint", "stop_claim", "metadata",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown trajectory fields: {sorted(unknown)}")
        raw = payload.get("raw_reference")
        if not isinstance(raw, Mapping):
            raise ValueError("raw_reference must be an object")
        parent_ids = payload.get("parent_ids", ())
        if not isinstance(parent_ids, (list, tuple)) or not all(
            isinstance(item, str) for item in parent_ids
        ):
            raise ValueError("parent_ids must be a list of strings")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            run_id=payload["run_id"],
            event_id=payload["event_id"],
            paradigm=Paradigm(payload["paradigm"]),
            sequence_index=payload["sequence_index"],
            kind=EventKind(payload["kind"]),
            decision=Decision(payload["decision"]),
            raw_reference=RawReference(**raw),
            candidate_id=payload.get("candidate_id"),
            parent_ids=tuple(parent_ids),
            timestamp=payload.get("timestamp"),
            accuracy=payload.get("accuracy"),
            parameter_count=payload.get("parameter_count"),
            valid=payload.get("valid"),
            description=payload.get("description"),
            architecture_fingerprint=payload.get("architecture_fingerprint"),
            stop_claim=payload.get("stop_claim"),
            metadata=dict(metadata),
        )

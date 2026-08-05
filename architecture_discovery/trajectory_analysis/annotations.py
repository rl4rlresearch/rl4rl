"""Human annotation loading, adjudication, agreement, and non-binding suggestions."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .schemas import BoundaryClass, EditFamily, TrajectoryEvent


@dataclass(frozen=True, slots=True)
class Annotation:
    event_id: str
    annotator_id: str
    role: str
    edit_family: EditFamily
    boundary_class: BoundaryClass
    rationale: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Annotation":
        allowed = {
            "event_id", "annotator_id", "role", "edit_family", "boundary_class",
            "rationale",
        }
        unknown = set(payload) - allowed
        missing = allowed - set(payload)
        if unknown or missing:
            raise ValueError(
                f"annotation schema mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        for name in ("event_id", "annotator_id", "rationale"):
            if not isinstance(payload[name], str) or not payload[name]:
                raise ValueError(f"annotation {name} must be a non-empty string")
        role = payload["role"]
        if role not in {"coder", "adjudicator"}:
            raise ValueError("annotation role must be coder or adjudicator")
        return cls(
            event_id=payload["event_id"],
            annotator_id=payload["annotator_id"],
            role=role,
            edit_family=EditFamily(payload["edit_family"]),
            boundary_class=BoundaryClass(payload["boundary_class"]),
            rationale=payload["rationale"],
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["edit_family"] = self.edit_family.value
        payload["boundary_class"] = self.boundary_class.value
        return payload


@dataclass(frozen=True, slots=True)
class ResolvedAnnotation:
    event_id: str
    edit_family: EditFamily
    boundary_class: BoundaryClass
    resolution: str

    def to_dict(self) -> dict[str, str]:
        return {
            "event_id": self.event_id,
            "edit_family": self.edit_family.value,
            "boundary_class": self.boundary_class.value,
            "resolution": self.resolution,
        }


def load_annotations(path: str | Path) -> list[Annotation]:
    annotations: list[Annotation] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"annotation line {line_number} is not an object")
            annotations.append(Annotation.from_dict(value))
    if not annotations:
        raise ValueError(f"annotation file is empty: {path}")
    return annotations


def _kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right):
        raise ValueError("agreement vectors have different lengths")
    if not left:
        return None
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_counts[label] * right_counts[label] for label in set(left) | set(right)
    ) / (len(left) ** 2)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def resolve_annotations(
    events: Iterable[TrajectoryEvent],
    annotations: Iterable[Annotation],
    *,
    require_complete: bool = True,
) -> tuple[dict[str, ResolvedAnnotation], dict[str, object]]:
    event_list = list(events)
    event_ids = {event.event_id for event in event_list}
    edit_event_ids = {
        event.event_id
        for event in event_list
        if event.candidate_id is not None and bool(event.parent_ids)
    }
    grouped: dict[str, list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        if annotation.event_id not in event_ids:
            raise ValueError(f"annotation references unknown event {annotation.event_id}")
        grouped[annotation.event_id].append(annotation)

    resolved: dict[str, ResolvedAnnotation] = {}
    coder_pairs: list[tuple[Annotation, Annotation]] = []
    for event_id in sorted(edit_event_ids):
        records = grouped.get(event_id, [])
        coders = sorted(
            (record for record in records if record.role == "coder"),
            key=lambda record: record.annotator_id,
        )
        adjudicators = [record for record in records if record.role == "adjudicator"]
        if not coders and not require_complete:
            resolved[event_id] = ResolvedAnnotation(
                event_id, EditFamily.OTHER, BoundaryClass.UNCLASSIFIED, "missing"
            )
            continue
        if len(coders) != 2 or coders[0].annotator_id == coders[1].annotator_id:
            raise ValueError(f"event {event_id} requires exactly two independent coders")
        coder_pairs.append((coders[0], coders[1]))
        agrees = (
            coders[0].edit_family is coders[1].edit_family
            and coders[0].boundary_class is coders[1].boundary_class
        )
        if agrees:
            if adjudicators:
                raise ValueError(f"event {event_id} has unnecessary adjudication")
            chosen = coders[0]
            resolution = "coder_agreement"
        else:
            if len(adjudicators) != 1:
                raise ValueError(f"event {event_id} disagreement requires one adjudicator")
            chosen = adjudicators[0]
            resolution = "adjudicated"
        resolved[event_id] = ResolvedAnnotation(
            event_id, chosen.edit_family, chosen.boundary_class, resolution
        )

    extras = set(grouped) - edit_event_ids
    if extras:
        raise ValueError(f"annotations supplied for non-edit events: {sorted(extras)}")
    agreement = {
        "double_coded_events": len(coder_pairs),
        "exact_joint_agreement": (
            sum(
                left.edit_family is right.edit_family
                and left.boundary_class is right.boundary_class
                for left, right in coder_pairs
            ) / len(coder_pairs)
            if coder_pairs else None
        ),
        "edit_family_kappa": _kappa(
            [left.edit_family.value for left, _ in coder_pairs],
            [right.edit_family.value for _, right in coder_pairs],
        ),
        "boundary_class_kappa": _kappa(
            [left.boundary_class.value for left, _ in coder_pairs],
            [right.boundary_class.value for _, right in coder_pairs],
        ),
        "adjudicated_events": sum(
            item.resolution == "adjudicated" for item in resolved.values()
        ),
        "unclassified_events": sum(
            item.boundary_class is BoundaryClass.UNCLASSIFIED for item in resolved.values()
        ),
    }
    return resolved, agreement


_SUGGESTION_TERMS: dict[EditFamily, tuple[str, ...]] = {
    EditFamily.WIDTH: ("width", "dimension", "hidden size", "head size"),
    EditFamily.DEPTH: ("depth", "layer", "block"),
    EditFamily.PARAMETER_TYING: ("tie", "shared weight", "weight sharing"),
    EditFamily.POSITIONAL: ("rope", "rotary", "position"),
    EditFamily.EMBEDDINGS: ("embedding", "token representation"),
    EditFamily.NORMALIZATION: ("norm", "rmsnorm", "layernorm"),
    EditFamily.ATTENTION: ("attention", "head", "query", "key", "value"),
    EditFamily.FEEDFORWARD: ("feedforward", "mlp", "ffn"),
    EditFamily.TOKENIZATION: ("tokenizer", "tokenization", "digit token"),
}


def suggest_edit_families(description: str | None) -> list[str]:
    """Return auditable hints only; suggestions never become analysis labels."""

    text = (description or "").lower()
    return [
        family.value
        for family, terms in _SUGGESTION_TERMS.items()
        if any(term in text for term in terms)
    ]

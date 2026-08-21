"""Deterministic, public-feedback-only visible-memory packets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from research_dynamics.contracts import VisibleMemoryPolicy


PUBLIC_EVALUATION_FIELDS = (
    "execution_ok",
    "transformer_valid",
    "public_accuracy",
    "search_score",
    "eligible_for_parent",
    "failure_stage",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{source}:{line_number} is not a JSON object")
        records.append(value)
    return records


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _distance(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / max(1, len(a | b))


def _opportunity(record: dict[str, Any]) -> int:
    evaluation = record.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return -1
    value = evaluation.get("proposal_opportunity", evaluation.get("opportunity_index", -1))
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def _entry(record: dict[str, Any], label: str) -> dict[str, Any]:
    evaluation = record.get("evaluation", {})
    if not isinstance(evaluation, dict):
        evaluation = {}
    public = {key: evaluation[key] for key in PUBLIC_EVALUATION_FIELDS if key in evaluation}
    return {
        "slot": label,
        "opportunity": _opportunity(record),
        "candidate_id": record.get("candidate_id", ""),
        "mechanism": record.get("mechanism_hypothesis")
        or record.get("proposal_text", ""),
        "public_result": public,
        "decision": record.get("retention_decision", ""),
    }


def _placeholder(label: str) -> dict[str, Any]:
    return {
        "slot": label,
        "opportunity": None,
        "candidate_id": "not_available",
        "mechanism": "No eligible prior entry at this checkpoint.",
        "public_result": {},
        "decision": "not_available",
    }


def select_memory_entries(
    records: Iterable[dict[str, Any]],
    policy: VisibleMemoryPolicy,
) -> list[dict[str, Any]]:
    """Select four fixed semantic slots without reading sealed or future outcomes."""

    eligible = sorted(
        (record for record in records if _opportunity(record) >= 0),
        key=_opportunity,
    )
    labels = (
        "current_or_recent",
        "valid_failure",
        "distant_alternative",
        "abandoned_direction",
    )
    if not eligible:
        return [_placeholder(label) for label in labels]

    latest = eligible[-1]
    if policy is VisibleMemoryPolicy.SEQUENTIAL:
        return [_entry(latest, labels[0])] + [
            _placeholder(label) for label in labels[1:]
        ]

    retained = [
        record
        for record in eligible
        if str(record.get("retention_decision", "")).startswith(
            ("accept", "archive_new", "archive_replace", "seed_parent")
        )
    ]
    rejected = [record for record in eligible if record not in retained]
    current = retained[-1] if retained else latest
    valid_failure = next(
        (
            record
            for record in reversed(rejected)
            if record.get("evaluation", {}).get("execution_ok") is True
        ),
        rejected[-1] if rejected else None,
    )
    current_text = str(
        current.get("mechanism_hypothesis") or current.get("proposal_text", "")
    )
    alternatives = [record for record in eligible if record is not current]
    distant = max(
        alternatives,
        key=lambda record: (
            _distance(
                current_text,
                str(
                    record.get("mechanism_hypothesis")
                    or record.get("proposal_text", "")
                ),
            ),
            -_opportunity(record),
        ),
        default=None,
    )
    abandoned = next(
        (
            record
            for record in rejected
            if record is not valid_failure and record is not distant
        ),
        None,
    )
    selected = (current, valid_failure, distant, abandoned)
    return [
        _entry(record, label) if record is not None else _placeholder(label)
        for record, label in zip(selected, labels, strict=True)
    ]


def render_memory_packet(
    records: Iterable[dict[str, Any]],
    policy: VisibleMemoryPolicy,
    *,
    budget_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    entries = select_memory_entries(records, policy)
    heading = (
        "Sequential visible memory: use the current/recent entry only."
        if policy is VisibleMemoryPolicy.SEQUENTIAL
        else "Portfolio visible memory: compare all four fixed slots. Missing slots are explicit."
    )
    body = heading + "\n" + json.dumps(entries, indent=2, sort_keys=True)
    if len(body) > budget_chars:
        body = body[: budget_chars - 24] + "\n[packet truncated]"
    return body.ljust(budget_chars), entries

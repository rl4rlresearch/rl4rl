"""Normalize controller artifacts into research-decision records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from research_dynamics.contracts import FrameworkKind, ProcessStudyConfig
from research_dynamics.memory import PUBLIC_EVALUATION_FIELDS, read_jsonl
from study.serialization import content_hash


NOTE_FIELDS = (
    "research_current_explanation",
    "research_evidence",
    "research_next_experiment",
    "research_expected_result",
    "research_decision_rule",
    "research_previous_interpretation",
    "research_previous_changed",
    "research_challenged_assumption",
    "research_alternative_explanation",
    "research_discriminating_evidence",
)


def _candidate_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*)\n```", stripped, re.DOTALL | re.I)
    if fence:
        stripped = fence.group(1).strip()
    try:
        value = json.loads(stripped)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _notes(candidate: dict[str, Any] | None) -> dict[str, str | None]:
    metadata = candidate.get("metadata", {}) if candidate else {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        field: metadata.get(field) if isinstance(metadata.get(field), str) else None
        for field in NOTE_FIELDS
    }


def _opportunity(record: dict[str, Any]) -> int | None:
    evaluation = record.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return None
    value = evaluation.get("proposal_opportunity", evaluation.get("opportunity_index"))
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _artifact_candidates(run_dir: Path) -> dict[int, dict[str, Any] | None]:
    found: dict[int, dict[str, Any] | None] = {}
    artifacts = run_dir / "artifacts"
    if not artifacts.is_dir():
        return found
    for path in sorted(artifacts.iterdir()):
        match = re.match(r"(\d{4})", path.name)
        if not match or int(match.group(1)) == 0:
            continue
        opportunity = int(match.group(1))
        if path.name.endswith(".response.txt"):
            found[opportunity] = _candidate_object(path.read_text(encoding="utf-8"))
        elif path.name.endswith(".ir.json") and opportunity not in found:
            value = json.loads(path.read_text(encoding="utf-8"))
            found[opportunity] = value if isinstance(value, dict) else None
    return found


def _trace_candidates(run_dir: Path) -> dict[int, dict[str, Any] | None]:
    found: dict[int, dict[str, Any] | None] = {}
    trace = run_dir / "evolution_trace.jsonl"
    if not trace.is_file():
        return found
    for index, record in enumerate(read_jsonl(trace), 1):
        opportunity = record.get("iteration", index)
        if not isinstance(opportunity, int) or isinstance(opportunity, bool):
            opportunity = index
        candidate = None
        for key in (
            "child_code",
            "llm_response",
            "child_program",
            "new_program",
            "program",
            "code",
        ):
            if isinstance(record.get(key), str):
                candidate = _candidate_object(record[key])
                if candidate is not None:
                    break
        found[opportunity] = candidate
    return found


def _public_result(record: dict[str, Any]) -> dict[str, Any]:
    evaluation = record.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return {}
    return {key: evaluation[key] for key in PUBLIC_EVALUATION_FIELDS if key in evaluation}


def extract_autoresearch_run(
    run_dir: str | Path,
    config: ProcessStudyConfig,
) -> list[dict[str, Any]]:
    directory = Path(run_dir)
    lineage = read_jsonl(directory / "lineage.jsonl")
    candidates = _artifact_candidates(directory)
    exposures = {
        record.get("opportunity"): record
        for record in read_jsonl(directory / "research_process" / "exposures.jsonl")
    }
    decisions: list[dict[str, Any]] = []
    for record in lineage:
        opportunity = _opportunity(record)
        if opportunity is None or opportunity < 1:
            continue
        exposure = exposures.get(opportunity, {})
        decisions.append(
            {
                "schema_name": "ResearchDecisionRecord",
                "schema_version": "1.0",
                "study_id": config.study_id,
                "run_id": config.run_id,
                "framework": config.framework.value,
                "condition_id": config.condition.condition_id.value,
                "opportunity": opportunity,
                "candidate_id": record.get("candidate_id"),
                "parent_id": record.get("parent_id"),
                "prompt_hash": record.get("prompt_hash"),
                "response_hash": record.get("response_hash"),
                "challenge_active": bool(exposure.get("challenge_active", False)),
                "memory_entry_ids": [
                    entry.get("candidate_id")
                    for entry in exposure.get("memory_entries", [])
                    if entry.get("candidate_id") != "not_available"
                ],
                "lab_note_before": _notes(candidates.get(opportunity)),
                "public_result": _public_result(record),
                "retention_decision": record.get("retention_decision"),
                "lab_note_after": None,
                "record_hash": "",
            }
        )
    decisions.sort(key=lambda item: item["opportunity"])
    # Opportunity t+1 reports its interpretation of result t. Link it backward.
    for current, following in zip(decisions, decisions[1:]):
        following_notes = following["lab_note_before"]
        current["lab_note_after"] = {
            "observed_result_interpretation": following_notes[
                "research_previous_interpretation"
            ],
            "changed_explanation": following_notes["research_previous_changed"],
        }
    for decision in decisions:
        unhashed = {**decision, "record_hash": ""}
        decision["record_hash"] = content_hash(unhashed)
    return decisions


def extract_openevolve_run(
    run_dir: str | Path,
    config: ProcessStudyConfig,
) -> list[dict[str, Any]]:
    directory = Path(run_dir)
    candidates = _trace_candidates(directory)
    traces = {
        record.get("iteration", index): record
        for index, record in enumerate(
            read_jsonl(directory / "evolution_trace.jsonl"), 1
        )
    }
    exposures = read_jsonl(directory / "research_process" / "exposures.jsonl")
    if not exposures:
        exposures = [
            {
                "opportunity": opportunity,
                "challenge_active": False,
                "memory_entries": [],
            }
            for opportunity in sorted(traces)
        ]
    else:
        by_opportunity = {
            exposure.get("opportunity"): exposure
            for exposure in exposures
            if isinstance(exposure.get("opportunity"), int)
            and not isinstance(exposure.get("opportunity"), bool)
        }
        exposures = [by_opportunity[key] for key in sorted(by_opportunity)]
    decisions: list[dict[str, Any]] = []
    for index, exposure in enumerate(exposures, 1):
        opportunity = exposure.get("opportunity", index)
        candidate = candidates.get(opportunity)
        trace = traces.get(opportunity, {})
        metrics = trace.get("child_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        prompt = trace.get("prompt")
        response = trace.get("llm_response")
        decisions.append(
            {
                "schema_name": "ResearchDecisionRecord",
                "schema_version": "1.0",
                "study_id": config.study_id,
                "run_id": config.run_id,
                "framework": config.framework.value,
                "condition_id": config.condition.condition_id.value,
                "opportunity": opportunity,
                "candidate_id": trace.get("child_id"),
                "parent_id": trace.get("parent_id"),
                "prompt_hash": content_hash(prompt) if prompt else None,
                "response_hash": (
                    content_hash(response)
                    if isinstance(response, str)
                    else content_hash(candidate)
                    if candidate
                    else None
                ),
                "challenge_active": bool(exposure.get("challenge_active", False)),
                "memory_entry_ids": [
                    entry.get("candidate_id")
                    for entry in exposure.get("memory_entries", [])
                    if entry.get("candidate_id") != "not_available"
                ],
                "lab_note_before": _notes(candidate),
                "public_result": {
                    key: metrics[key]
                    for key in PUBLIC_EVALUATION_FIELDS
                    if key in metrics
                },
                "retention_decision": None,
                "lab_note_after": None,
                "record_hash": "",
            }
        )
    for current, following in zip(decisions, decisions[1:]):
        notes = following["lab_note_before"]
        current["lab_note_after"] = {
            "observed_result_interpretation": notes[
                "research_previous_interpretation"
            ],
            "changed_explanation": notes["research_previous_changed"],
        }
    for decision in decisions:
        decision["record_hash"] = content_hash({**decision, "record_hash": ""})
    return decisions


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
        for record in records
    ]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_run(run_dir: str | Path, config: ProcessStudyConfig) -> Path:
    if config.framework is FrameworkKind.AUTORESEARCH:
        decisions = extract_autoresearch_run(run_dir, config)
    else:
        decisions = extract_openevolve_run(run_dir, config)
    destination = Path(run_dir) / "research_process" / "decisions.jsonl"
    write_jsonl(destination, decisions)
    return destination

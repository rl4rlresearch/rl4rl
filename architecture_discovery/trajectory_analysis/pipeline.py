"""End-to-end, provenance-preserving trajectory analysis pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .adapters import load_source
from .annotations import (
    Annotation,
    ResolvedAnnotation,
    load_annotations,
    resolve_annotations,
    suggest_edit_families,
)
from .figures import boundary_mix, frontier_progression, rolling_diversity
from .manifest import StudyManifest, resolve_frozen_file
from .metrics import summarize_paradigms, summarize_run
from .schemas import TrajectoryEvent
from .storage import (
    require_new_output_directory,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)
from .validation import ValidationReport, validate_trajectories


RUN_CSV_FIELDS = (
    "run_id", "paradigm", "total_events", "evaluated_candidates",
    "accepted_candidates", "valid_candidates", "qualifying_candidates",
    "initial_qualifying_parameters", "best_qualifying_parameters",
    "external_frontier_parameters", "frontier_gap_parameters", "frontier_gap_ratio",
    "frontier_improvements", "ontology_change_attempt_rate",
    "ontology_change_acceptance_rate", "ontology_change_qualification_rate",
    "acceptance_rate", "invalid_rate", "rollback_rate", "fingerprint_coverage",
    "architecture_revisit_rate", "unique_architecture_fingerprints",
    "max_ontology_preserving_streak", "first_ontology_change_sequence",
    "first_qualifying_sequence", "first_frontier_improvement_sequence",
    "invalid_parent_to_valid_child_count", "stop_claim", "premature_frontier_claim",
)

CONTEXT_FIELDS = (
    "source_id", "run_id", "paradigm", "generator_model", "evaluator_id",
    "tool_policy", "run_family_id", "starting_artifact_sha256", "prompt_sha256",
    "seed", "proposal_budget", "token_budget", "wall_time_budget_seconds",
    "accelerator_budget_seconds", "archive_reused", "notes",
)

COMPARABILITY_FIELDS = (
    "generator_model", "evaluator_id", "tool_policy", "starting_artifact_sha256",
    "prompt_sha256", "proposal_budget", "token_budget", "wall_time_budget_seconds",
    "accelerator_budget_seconds",
)


def load_study(
    manifest_path: str | Path,
    data_root: str | Path,
    *,
    require_annotations: bool,
) -> tuple[
    StudyManifest,
    list[TrajectoryEvent],
    list[Annotation],
    dict[str, ResolvedAnnotation],
    dict[str, object],
    ValidationReport,
]:
    manifest = StudyManifest.load(manifest_path)
    events: list[TrajectoryEvent] = []
    annotations: list[Annotation] = []
    for source in manifest.sources:
        source_path = resolve_frozen_file(data_root, source.path, source.sha256)
        events.extend(load_source(source, source_path))
        if source.annotation_path is not None:
            assert source.annotation_sha256 is not None
            annotation_path = resolve_frozen_file(
                data_root, source.annotation_path, source.annotation_sha256
            )
            annotations.extend(load_annotations(annotation_path))
    validation = validate_trajectories(events)
    resolved, agreement = resolve_annotations(
        events, annotations, require_complete=require_annotations
    )
    return manifest, events, annotations, resolved, agreement, validation


def _lineage_dot(events: Iterable[TrajectoryEvent]) -> str:
    lines = ["digraph trajectory_lineage {", "  rankdir=LR;"]
    for event in events:
        if event.candidate_id is None:
            continue
        child = f"{event.run_id}:{event.candidate_id}".replace('"', '\\"')
        label = event.candidate_id.replace('"', '\\"')
        lines.append(f'  "{child}" [label="{label}"];')
        for parent in event.parent_ids:
            ancestor = f"{event.run_id}:{parent}".replace('"', '\\"')
            lines.append(f'  "{ancestor}" -> "{child}";')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _report(
    manifest: StudyManifest,
    validation: ValidationReport,
    run_summaries: list[dict[str, object]],
    paradigm_summaries: list[dict[str, object]],
    agreement: dict[str, object],
    comparability_warnings: list[str],
) -> str:
    lines = [
        f"# Trajectory analysis: {manifest.study_id}",
        "",
        "## Scope",
        "",
        (
            f"This report describes {validation.run_count} independent run(s), "
            f"{validation.event_count} recorded event(s), and "
            f"{validation.candidate_count} candidate identity/identities across "
            f"{', '.join(validation.paradigms)}. A candidate qualifies at "
            f"accuracy ≥ {manifest.accuracy_threshold:.3f}."
        ),
        "",
        "The run—not the candidate—is the independent unit. Aggregates are descriptive; "
        "candidate-level observations are not treated as independent replications.",
        "",
        "## Run-level results",
        "",
        "| Run | Paradigm | Evaluated | Best qualifying params | Ontology-change rate | Revisit rate |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in run_summaries:
        best = summary["best_qualifying_parameters"]
        change = summary["ontology_change_attempt_rate"]
        revisit = summary["architecture_revisit_rate"]
        change_text = f"{change:.3f}" if isinstance(change, float) else "—"
        revisit_text = f"{revisit:.3f}" if isinstance(revisit, float) else "—"
        lines.append(
            f"| {summary['run_id']} | {summary['paradigm']} | "
            f"{summary['evaluated_candidates']} | "
            f"{best if best is not None else '—'} | {change_text} | {revisit_text} |"
        )
    lines += [
        "",
        "## Cross-paradigm summaries",
        "",
    ]
    for summary in paradigm_summaries:
        lines.append(
            f"- **{summary['paradigm']}**: {summary['independent_runs']} independent "
            f"run(s). {summary['inference_note']}"
        )
    lines += [
        "",
        "## Run comparability",
        "",
    ]
    if comparability_warnings:
        lines.extend(f"- {warning}" for warning in comparability_warnings)
    else:
        lines.append("No mismatch was detected in the predeclared comparable context fields.")
    lines += [
        "",
        "## Annotation reliability",
        "",
        f"- Double-coded edits: {agreement['double_coded_events']}",
        f"- Joint exact agreement: {agreement['exact_joint_agreement']}",
        f"- Edit-family Cohen's κ: {agreement['edit_family_kappa']}",
        f"- Boundary-class Cohen's κ: {agreement['boundary_class_kappa']}",
        f"- Adjudicated edits: {agreement['adjudicated_events']}",
        f"- Unclassified edits: {agreement['unclassified_events']}",
        "",
        "## Interpretation limits",
        "",
        "This benchmark can reveal behavior in the observed AdderBoard searches; it does not "
        "establish a universal law about autonomous research systems. Different compute, prompts, "
        "tool access, starting models, and stopping rules are potential design factors, not noise "
        "to silently pool. Keyword suggestions are never used as scientific labels. The external "
        "frontier is a comparison target, not evidence that a run should have found that solution.",
        "",
        "Machine-readable summaries, normalized events, lineage, figures, input hashes, and output "
        "hashes accompany this report.",
    ]
    return "\n".join(lines) + "\n"


def analyze_study(
    manifest_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    require_annotations: bool = True,
) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    data_root = Path(data_root).resolve()
    manifest, events, _annotations, resolved, agreement, validation = load_study(
        manifest_path, data_root, require_annotations=require_annotations
    )
    destination = require_new_output_directory(output_dir)

    by_run: dict[str, list[TrajectoryEvent]] = defaultdict(list)
    for event in events:
        by_run[event.run_id].append(event)
    run_summaries = [
        summarize_run(
            by_run[run_id],
            resolved,
            accuracy_threshold=manifest.accuracy_threshold,
            external_frontier_parameters=manifest.external_frontier_parameters,
        )
        for run_id in sorted(by_run)
    ]
    paradigm_summaries = summarize_paradigms(run_summaries)
    context_rows = [
        {
            "source_id": source.source_id,
            "run_id": source.run_id,
            "paradigm": source.paradigm.value,
            **source.context.to_dict(),
        }
        for source in manifest.sources
    ]
    comparability_warnings = []
    for field in COMPARABILITY_FIELDS:
        values = {row[field] for row in context_rows}
        if len(values) > 1:
            comparability_warnings.append(
                f"Unmatched `{field}` values across runs; do not attribute that contrast to paradigm alone."
            )
        if None in values or "unknown" in values:
            comparability_warnings.append(
                f"Missing or unknown `{field}` for at least one run."
            )
    if any(bool(row["archive_reused"]) for row in context_rows):
        comparability_warnings.append(
            "At least one run reused an archive; verify dependence before counting runs as independent."
        )
    family_ids = [str(row["run_family_id"]) for row in context_rows]
    if len(family_ids) != len(set(family_ids)):
        comparability_warnings.append(
            "Multiple runs share a run family and may not be independent."
        )

    normalized = []
    suggestions = []
    for event in sorted(events, key=lambda item: (item.run_id, item.sequence_index)):
        payload = event.to_dict()
        label = resolved.get(event.event_id)
        payload["resolved_annotation"] = label.to_dict() if label else None
        normalized.append(payload)
        if event.parent_ids:
            suggestions.append(
                {
                    "event_id": event.event_id,
                    "suggested_edit_families": suggest_edit_families(event.description),
                    "warning": "non-binding annotation aid; excluded from metrics",
                }
            )

    write_jsonl(destination / "normalized_events.jsonl", normalized)
    write_jsonl(destination / "annotation_suggestions.jsonl", suggestions)
    write_json(destination / "annotation_agreement.json", agreement)
    write_json(destination / "run_summaries.json", run_summaries)
    write_csv(destination / "run_summaries.csv", run_summaries, fieldnames=RUN_CSV_FIELDS)
    write_json(destination / "paradigm_summaries.json", paradigm_summaries)
    paradigm_fields = tuple(paradigm_summaries[0]) if paradigm_summaries else ()
    write_csv(
        destination / "paradigm_summaries.csv",
        paradigm_summaries,
        fieldnames=paradigm_fields,
    )
    write_json(destination / "run_contexts.json", context_rows)
    write_csv(destination / "run_contexts.csv", context_rows, fieldnames=CONTEXT_FIELDS)
    write_json(destination / "comparability_warnings.json", comparability_warnings)
    edges = [
        {"run_id": event.run_id, "parent_id": parent, "candidate_id": event.candidate_id}
        for event in events
        if event.candidate_id is not None
        for parent in event.parent_ids
    ]
    write_csv(
        destination / "lineage_edges.csv",
        edges,
        fieldnames=("run_id", "parent_id", "candidate_id"),
    )
    write_text(destination / "lineage.dot", _lineage_dot(events))
    frontier_progression(run_summaries, destination / "frontier_progression.svg")
    boundary_mix(run_summaries, destination / "boundary_edit_mix.svg")
    rolling_diversity(events, resolved, destination / "rolling_architecture_diversity.svg")
    write_text(
        destination / "report.md",
        _report(
            manifest,
            validation,
            run_summaries,
            paradigm_summaries,
            agreement,
            comparability_warnings,
        ),
    )
    input_files = {
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "sources": [
            {
                "source_id": source.source_id,
                "path": source.path,
                "sha256": source.sha256,
                "annotation_path": source.annotation_path,
                "annotation_sha256": source.annotation_sha256,
                "context": source.context.to_dict(),
            }
            for source in manifest.sources
        ],
    }
    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "provenance.json"
    }
    provenance = {
        "schema_version": "trajectory-analysis-provenance-v1",
        "study_id": manifest.study_id,
        "require_annotations": require_annotations,
        "validation": {
            "run_count": validation.run_count,
            "event_count": validation.event_count,
            "candidate_count": validation.candidate_count,
            "paradigms": list(validation.paradigms),
        },
        "inputs": input_files,
        "output_sha256": output_hashes,
    }
    write_json(destination / "provenance.json", provenance)
    return provenance

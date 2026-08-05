from __future__ import annotations

from pathlib import Path

from scripts.trajectory_offline_smoke import build_synthetic_inputs
from trajectory_analysis.metrics import summarize_paradigms, summarize_run
from trajectory_analysis.pipeline import load_study


def test_run_metrics_capture_frontier_boundary_and_stopping(tmp_path: Path):
    data_root = tmp_path / "inputs"
    manifest_path = build_synthetic_inputs(data_root)
    manifest, events, _, resolved, _, _ = load_study(
        manifest_path, data_root, require_annotations=True
    )
    run_events = [event for event in events if event.run_id == "synthetic-autoresearch"]
    summary = summarize_run(
        run_events,
        resolved,
        accuracy_threshold=manifest.accuracy_threshold,
        external_frontier_parameters=manifest.external_frontier_parameters,
    )
    assert summary["best_qualifying_parameters"] == 96
    assert summary["frontier_improvements"] == 2
    assert summary["ontology_change_attempt_rate"] == 0.5
    assert summary["ontology_change_acceptance_rate"] == 1.0
    assert summary["premature_frontier_claim"] is True
    assert summary["frontier_gap_parameters"] == 60


def test_cross_paradigm_summary_refuses_candidate_pseudoreplication(tmp_path: Path):
    data_root = tmp_path / "inputs"
    manifest_path = build_synthetic_inputs(data_root)
    manifest, events, _, resolved, _, _ = load_study(
        manifest_path, data_root, require_annotations=True
    )
    summaries = []
    for run_id in sorted({event.run_id for event in events}):
        summaries.append(
            summarize_run(
                [event for event in events if event.run_id == run_id],
                resolved,
                accuracy_threshold=manifest.accuracy_threshold,
                external_frontier_parameters=manifest.external_frontier_parameters,
            )
        )
    paradigms = summarize_paradigms(summaries)
    assert len(paradigms) == 3
    assert all(item["independent_runs"] == 1 for item in paradigms)
    assert all(item["analysis_scope"] == "descriptive_only" for item in paradigms)
    assert all("No confidence intervals" in item["inference_note"] for item in paradigms)

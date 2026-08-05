from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from scripts.trajectory_offline_smoke import build_synthetic_inputs
from trajectory_analysis.pipeline import analyze_study


def test_end_to_end_pipeline_writes_a_hashed_reproducible_bundle(tmp_path: Path):
    data_root = tmp_path / "inputs"
    manifest_path = build_synthetic_inputs(data_root)
    output = tmp_path / "analysis"
    provenance = analyze_study(manifest_path, data_root, output)
    expected = {
        "normalized_events.jsonl",
        "annotation_suggestions.jsonl",
        "annotation_agreement.json",
        "run_summaries.json",
        "run_summaries.csv",
        "paradigm_summaries.json",
        "paradigm_summaries.csv",
        "run_contexts.json",
        "run_contexts.csv",
        "comparability_warnings.json",
        "lineage_edges.csv",
        "lineage.dot",
        "frontier_progression.svg",
        "boundary_edit_mix.svg",
        "rolling_architecture_diversity.svg",
        "report.md",
        "provenance.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert set(provenance["output_sha256"]) == expected - {"provenance.json"}
    assert provenance["validation"]["run_count"] == 3
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "candidate-level observations are not treated as independent" in report
    suggestions = [
        json.loads(line)
        for line in (output / "annotation_suggestions.jsonl").read_text().splitlines()
    ]
    assert all("excluded from metrics" in item["warning"] for item in suggestions)
    for svg_name in (
        "frontier_progression.svg",
        "boundary_edit_mix.svg",
        "rolling_architecture_diversity.svg",
    ):
        assert ElementTree.parse(output / svg_name).getroot().tag.endswith("svg")

    second_output = tmp_path / "analysis-repeat"
    repeat = analyze_study(manifest_path, data_root, second_output)
    assert repeat["output_sha256"] == provenance["output_sha256"]
    with pytest.raises(FileExistsError, match="already exists"):
        analyze_study(manifest_path, data_root, output)

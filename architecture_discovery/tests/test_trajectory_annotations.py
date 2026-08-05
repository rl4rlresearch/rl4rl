from __future__ import annotations

from pathlib import Path

import pytest

from scripts.trajectory_offline_smoke import build_synthetic_inputs
from trajectory_analysis.annotations import Annotation, resolve_annotations, suggest_edit_families
from trajectory_analysis.pipeline import load_study


def test_double_coding_and_adjudication_are_reported(tmp_path: Path):
    data_root = tmp_path / "inputs"
    manifest_path = build_synthetic_inputs(data_root)
    _, _, _, resolved, agreement, _ = load_study(
        manifest_path, data_root, require_annotations=True
    )
    assert len(resolved) == 6
    assert agreement["double_coded_events"] == 6
    assert agreement["adjudicated_events"] == 1
    assert agreement["unclassified_events"] == 0
    assert agreement["boundary_class_kappa"] is not None


def test_unresolved_disagreement_fails_closed(tmp_path: Path):
    data_root = tmp_path / "inputs"
    manifest_path = build_synthetic_inputs(data_root)
    _, events, annotations, _, _, _ = load_study(
        manifest_path, data_root, require_annotations=True
    )
    without_adjudicator = [
        record
        for record in annotations
        if not (record.event_id == "openevolve:2" and record.role == "adjudicator")
    ]
    with pytest.raises(ValueError, match="requires one adjudicator"):
        resolve_annotations(events, without_adjudicator)


def test_keyword_suggestions_are_non_binding():
    assert suggest_edit_families("Replace learned positions with RoPE") == ["positional"]
    assert suggest_edit_families(None) == []


def test_annotation_schema_rejects_unknown_fields():
    with pytest.raises(ValueError, match="schema mismatch"):
        Annotation.from_dict(
            {
                "event_id": "e",
                "annotator_id": "a",
                "role": "coder",
                "edit_family": "width",
                "boundary_class": "ontology_preserving",
                "rationale": "width changed",
                "automatic": True,
            }
        )

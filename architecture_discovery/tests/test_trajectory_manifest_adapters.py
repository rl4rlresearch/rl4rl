from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.trajectory_offline_smoke import build_synthetic_inputs
from trajectory_analysis.adapters import load_source
from trajectory_analysis.manifest import StudyManifest, resolve_frozen_file


def test_all_three_native_adapters_preserve_identity_and_metadata(tmp_path: Path):
    data_root = tmp_path / "inputs"
    manifest = StudyManifest.load(build_synthetic_inputs(data_root))
    assert {source.adapter for source in manifest.sources} == {
        "autoresearch_tsv_v1",
        "openevolve_jsonl_v1",
        "ttt_jsonl_v1",
    }
    for source in manifest.sources:
        path = resolve_frozen_file(data_root, source.path, source.sha256)
        events = load_source(source, path)
        assert len(events) == 4
        assert events[0].candidate_id is not None
        assert events[-1].kind.value == "stop"
        assert all(event.raw_reference.source_sha256 == source.sha256 for event in events)


def test_manifest_hashes_and_path_boundary_fail_closed(tmp_path: Path):
    data_root = tmp_path / "inputs"
    manifest = StudyManifest.load(build_synthetic_inputs(data_root))
    source = manifest.sources[0]
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_frozen_file(data_root, source.path, "0" * 64)
    with pytest.raises(ValueError, match="escapes data root"):
        resolve_frozen_file(data_root, "../manifest.yaml", source.sha256)


def test_manifest_rejects_duplicate_run_ids(tmp_path: Path):
    data_root = tmp_path / "inputs"
    path = build_synthetic_inputs(data_root)
    payload = yaml.safe_load(path.read_text())
    payload["sources"][1]["run_id"] = payload["sources"][0]["run_id"]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate run_id"):
        StudyManifest.load(path)

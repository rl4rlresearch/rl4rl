"""Replay identity, missing-artifact integrity, and archive boundary checks."""

import hashlib
import json
import stat
import zipfile
from copy import deepcopy

import pytest
from architecture_trajectory_visualization.replay import (
    discover_catalog,
    export_architecture_bundle,
    lfs_pointer,
    load_run_replay,
    prepare_archive,
    resolve_candidate_source,
    validate_architecture_bundle,
    verify_archive,
)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def recorded_run(tmp_path):
    campaign = tmp_path / "data/c0c3/test-campaign"
    run = campaign / "runs/test-run"
    write_json(
        campaign / "inputs/task.json",
        {
            "task_id": "tiny-kws-rnn",
            "objective_metric": "inference_cost",
            "objective_direction": "minimize",
        },
    )
    write_json(
        run / "manifest.json",
        {
            "assignment": {"condition": "C3", "block": 2},
            "baseline": {"candidate_id": "seed", "metrics": {"parameters": 100}},
        },
    )
    # Current state intentionally does not retain failed or historical candidates.
    write_json(
        run / "state.json",
        {
            "run_id": "test-run",
            "condition": "C3",
            "proposals_used": 3,
            "candidates": {},
            "active": {"index": 4},
        },
    )
    events = [
        {
            "event": "run_created",
            "seed_candidate_id": "seed",
            "timestamp": "2026-01-01T00:00:00Z",
        },
        {
            "event": "proposal_started",
            "opportunity": 1,
            "selected_parent_ids": ["seed"],
        },
        {
            "event": "proposal_completed",
            "opportunity": 1,
            "candidate_id": "failed",
            "parent_ids": ["seed"],
            "retained": False,
            "incumbent_after": "seed",
            "evaluation": {
                "valid": False,
                "failure_kind": "nonqualification",
                "metrics": {"parameters": 50},
            },
        },
        {
            "event": "proposal_completed",
            "opportunity": 2,
            "candidate_id": "repeat",
            "parent_ids": ["seed"],
            "retained": True,
            "incumbent_after": "seed",
            "portfolio_after": ["seed", "repeat"],
            "retention_decision": "filled_open_portfolio_slot",
            "evaluation": {
                "valid": True,
                "metrics": {"inference_cost": 5802804867952463872},
            },
        },
        {
            "event": "proposal_completed",
            "opportunity": 3,
            "candidate_id": "repeat",
            "parent_ids": ["repeat"],
            "retained": False,
            "incumbent_after": "seed",
            "evaluation": {
                "valid": True,
                "metrics": {"inference_cost": 5802804867952463872},
            },
        },
        {
            "event": "proposal_started",
            "opportunity": 4,
            "selected_parent_ids": ["repeat"],
        },
    ]
    (run / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    for candidate in ["seed", "repeat"]:
        source = run / "candidates" / candidate / "train.py"
        source.parent.mkdir(parents=True)
        source.write_text("from torch import nn\nclass Model(nn.Module):\n    pass\n")
    return tmp_path, campaign, run


def test_failed_repeated_and_incomplete_occurrences_survive(recorded_run):
    repo, campaign, _ = recorded_run
    payload = load_run_replay(repo, campaign, "test-run", scope="archive")
    rows = payload["occurrences"]
    assert [row["status"] for row in rows] == [
        "seed",
        "invalid",
        "retained",
        "rejected",
        "incomplete",
    ]
    assert rows[1]["source"]["availability"] == "missing"
    assert rows[2]["candidate_id"] == rows[3]["candidate_id"]
    assert rows[2]["id"] != rows[3]["id"]
    assert rows[3]["parent_occurrence_ids"] == [rows[2]["id"]]
    assert rows[4]["parent_occurrence_ids"] == [rows[3]["id"]]
    assert rows[2]["incumbent_after"] == "seed"
    assert rows[4]["metrics"] == {}
    assert rows[4]["candidate_id"] is None
    assert rows[2]["metrics"]["inference_cost"] == "5802804867952463872"
    assert rows[2]["decision_evaluation_id"] == rows[2]["evaluations"][0]["id"]


def test_saved_result_fallback_and_explicit_unresolved_parent(recorded_run):
    repo, campaign, run = recorded_run
    write_json(
        run / "opportunities/0005/result.json",
        {
            "event": "proposal_completed",
            "candidate_id": "missing",
            "parent_ids": ["unknown"],
            "retained": False,
            "evaluation": {"valid": True, "metrics": {"parameters": 30}},
        },
    )
    rows = load_run_replay(repo, campaign, "test-run", scope="archive")["occurrences"]
    assert rows[-1]["proposal"] == 5
    assert rows[-1]["unresolved_parent_ids"] == ["unknown"]
    assert rows[-1]["parent_occurrence_ids"] == []
    assert rows[-1]["source_record"].endswith("0005/result.json")


@pytest.mark.parametrize("workspace_state", ["missing", "different", "matching"])
def test_interrupted_parent_fallback_requires_exact_workspace_evidence(
    recorded_run, workspace_state
):
    repo, campaign, run = recorded_run
    write_json(campaign / "inputs/task.json", {"editable_paths": ["train.py"]})
    write_json(
        run / "opportunities/0001/result.json",
        {
            "candidate_id": "interrupted-attempt",
            "artifact_path": "candidates/seed",
            "parent_ids": ["seed"],
            "evaluation": {
                "valid": False,
                "failure_kind": "infrastructure_interruption",
            },
        },
    )
    if workspace_state != "missing":
        workspace = run / "opportunities/0001/proposal-workspace/train.py"
        workspace.parent.mkdir(parents=True)
        workspace.write_text(
            (run / "candidates/seed/train.py").read_text()
            if workspace_state == "matching"
            else "# changed but not saved as a candidate\n"
        )
    payload = load_run_replay(repo, campaign, "test-run", scope="archive")
    occurrence = payload["occurrences"][1]
    source = occurrence["source"]
    assert source["available"] is (workspace_state == "matching")
    if workspace_state == "matching":
        assert source["interruption_evidence"]["editable_sha256"]
        assert source["source_candidate_id"] == "seed"
    else:
        assert source["availability"] == "unresolved_recovery_fallback"
        assert source["source_hash"] is None
    bundle = export_architecture_bundle(repo, campaign, "test-run", scope="archive")
    snapshot = bundle["snapshots"][occurrence["id"]]
    assert snapshot["source_metadata"]["available"] is source["available"]
    if not source["available"]:
        assert snapshot["nodes"] == []
        assert snapshot["source_hash"] is None
        assert snapshot["source_id"] is None


def test_source_identity_and_path_boundaries(recorded_run, tmp_path):
    repo, campaign, run = recorded_run
    first = resolve_candidate_source(repo, campaign, "test-run", "repeat")
    assert first["available"] and list(first["sources"]) == ["train.py"]
    (run / "candidates/repeat/train.py").write_text(
        "raise RuntimeError('never run this')"
    )
    second = resolve_candidate_source(repo, campaign, "test-run", "repeat")
    assert first["source_hash"] != second["source_hash"]
    invalid = resolve_candidate_source(
        repo, campaign, "test-run", "repeat", artifact_path="../../secret"
    )
    assert invalid["availability"] == "invalid_path"
    alias = resolve_candidate_source(
        repo,
        campaign,
        "test-run",
        "synthetic-failure",
        artifact_path="candidates/repeat",
    )
    assert alias["available"]
    assert alias["identity_relation"] == "recorded_alias"
    assert alias["source_candidate_id"] == "repeat"
    assert alias["candidate_id"] == "synthetic-failure"
    with pytest.raises(ValueError):
        resolve_candidate_source(repo, campaign, "../test-run", "seed")
    with pytest.raises(ValueError):
        resolve_candidate_source(repo, campaign, "test-run", "../secret")
    outside = tmp_path / "secret.py"
    outside.write_text("secret")
    (run / "candidates/repeat/leak.py").symlink_to(outside)
    source = resolve_candidate_source(repo, campaign, "test-run", "repeat")
    assert "leak.py" not in source["sources"]
    assert source["availability"] == "partial"


def test_catalog_preserves_assignment_and_absent_archives(recorded_run):
    repo, _, _ = recorded_run
    catalog = discover_catalog(repo, scope="archive")
    campaign = next(c for c in catalog["campaigns"] if c["key"] == "test-campaign")
    assert campaign["runs"][0]["replicate"] == 2
    assert campaign["runs"][0]["condition"] == "C3"
    assert (
        next(m for m in campaign["metric_definitions"] if m["is_objective"])["key"]
        == "inference_cost"
    )
    assert not next(c for c in catalog["campaigns"] if c["key"] == "tiny-v21")[
        "available"
    ]


def test_portable_bundle_preserves_occurrences_and_never_embeds_programs(recorded_run):
    repo, campaign, _ = recorded_run
    bundle = export_architecture_bundle(repo, campaign, "test-run", scope="archive")
    validate_architecture_bundle(json.loads(json.dumps(bundle, allow_nan=False)))
    ids = [row["id"] for row in bundle["run"]["occurrences"]]
    assert set(bundle["snapshots"]) == set(ids)
    assert (
        bundle["snapshots"][ids[2]]["source_hash"]
        == bundle["snapshots"][ids[3]]["source_hash"]
    )
    assert not bundle["snapshots"][ids[1]]["source_metadata"]["available"]
    assert all(
        "sources" not in graph["source_metadata"]
        for graph in bundle["snapshots"].values()
    )
    bad = deepcopy(bundle)
    bad["snapshots"].pop(ids[1])
    with pytest.raises(ValueError, match="match uniquely"):
        validate_architecture_bundle(bad)
    bad = deepcopy(bundle)
    bad["snapshots"][ids[0]]["edges"] = [{"source": "nonexistent", "target": "missing"}]
    with pytest.raises(ValueError, match="unknown node"):
        validate_architecture_bundle(bad)
    bad = deepcopy(bundle)
    bad["snapshots"][ids[0]]["source_metadata"]["source_hash"] = "tampered"
    with pytest.raises(ValueError, match="source hash"):
        validate_architecture_bundle(bad)


def test_prepared_data_root_keeps_archive_provenance(recorded_run):
    repo, campaign, _ = recorded_run
    provenance = {
        "source_revision": "pinned-upstream",
        "archive_sha256": "a" * 64,
        "checksum_verified": True,
    }
    write_json(repo / "architecture-replay-source.json", provenance)
    replay = load_run_replay(repo, campaign, "test-run", scope="archive")
    assert replay["source_revision"] == "pinned-upstream"
    assert replay["dataset_provenance"] == provenance
    assert discover_catalog(repo, scope="archive")["dataset_provenance"] == provenance


def test_dashboard_scope_uses_existing_filter_policy(recorded_run, monkeypatch):
    repo, campaign, _ = recorded_run
    from experiments import live_trajectory_dashboard

    monkeypatch.setattr(
        live_trajectory_dashboard, "dashboard_run_visible", lambda _: True
    )
    monkeypatch.setattr(
        live_trajectory_dashboard, "dashboard_proposal_cap", lambda _: 2
    )
    assert len(load_run_replay(repo, campaign, "test-run")["occurrences"]) == 3
    assert (
        len(load_run_replay(repo, campaign, "test-run", scope="archive")["occurrences"])
        == 5
    )
    monkeypatch.setattr(
        live_trajectory_dashboard, "dashboard_run_visible", lambda _: False
    )
    with pytest.raises(FileNotFoundError):
        load_run_replay(repo, campaign, "test-run")


def test_lfs_pointer_and_archive_checksum(tmp_path):
    pointer = tmp_path / "archive.zip"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:"
        + "a" * 64
        + "\nsize 5000\n"
    )
    assert lfs_pointer(pointer)["bytes"] == 5000
    with pytest.raises(ValueError, match="Git LFS pointer"):
        verify_archive(pointer)
    with zipfile.ZipFile(pointer, "w") as bundle:
        bundle.writestr("campaign/runs/a/manifest.json", "{}")
        bundle.writestr("campaign/runs/a/candidates/b/train.py", "class Model: pass")
        bundle.writestr("campaign/runs/a/checkpoint.pt", b"weights")
    digest = hashlib.sha256(pointer.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SHA-256"):
        verify_archive(pointer, expected_sha256="b" * 64)
    report = prepare_archive(pointer, tmp_path / "output", expected_sha256=digest)
    assert report["checksum_verified"]
    assert report["files_extracted"] == 2
    assert not (tmp_path / "output/campaign/runs/a/checkpoint.pt").exists()
    assert (
        prepare_archive(pointer, tmp_path / "output", expected_sha256=digest)[
            "files_extracted"
        ]
        == 0
    )


@pytest.mark.parametrize(
    "member", ["../escape.py", "/absolute.py", "C:/escape.py", "..\\escape.py"]
)
def test_archive_rejects_traversal_before_writing(tmp_path, member):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("good.py", "good")
        bundle.writestr(member, "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        prepare_archive(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_archive_rejects_symlink_and_existing_different_file(tmp_path):
    archive = tmp_path / "bad.zip"
    link = zipfile.ZipInfo("link.py")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(link, "/etc/passwd")
    with pytest.raises(ValueError, match="symbolic link"):
        prepare_archive(archive, tmp_path / "output")
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("model.py", "new")
    output = tmp_path / "output"
    output.mkdir()
    (output / "model.py").write_text("existing")
    with pytest.raises(ValueError, match="overwrite"):
        prepare_archive(archive, output)
    assert (output / "model.py").read_text() == "existing"

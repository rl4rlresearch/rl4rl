"""HTTP facade contracts: scope isolation, provenance, digest caches and export."""

import json
import shutil
from pathlib import Path

import pytest
from architecture_trajectory_visualization import (
    annotations as architecture_annotations,
)
from architecture_trajectory_visualization import api as architecture_api
from architecture_trajectory_visualization.api import ArchitectureStore, encode
from architecture_trajectory_visualization.replay import validate_architecture_bundle

from experiments import live_trajectory_dashboard


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_campaign(root: Path, name: str = "test-campaign", *, width: int = 4) -> Path:
    campaign = root / "data/c0c3" / name
    run = campaign / "runs/test-run"
    write_json(campaign / "inputs/task.json", {"task_id": "addition"})
    write_json(
        run / "manifest.json",
        {
            "assignment": {"condition": "C0", "block": 1},
            "baseline": {"candidate_id": "seed", "metrics": {"parameters": 10}},
        },
    )
    write_json(
        run / "state.json",
        {"run_id": "test-run", "condition": "C0", "proposals_used": 1},
    )
    rows = [
        {"event": "run_created", "seed_candidate_id": "seed"},
        {
            "event": "proposal_completed",
            "opportunity": 1,
            "candidate_id": "child",
            "artifact_path": "candidates/child",
            "parent_ids": ["seed"],
            "retained": True,
            "incumbent_after": "child",
            "evaluation": {"valid": True, "metrics": {"parameters": 8}},
        },
    ]
    (run / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    for candidate in ("seed", "child"):
        source = run / "candidates" / candidate / "train.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "from torch import nn\nclass Model(nn.Module):\n"
            "    def __init__(self):\n        super().__init__()\n"
            f"        self.linear = nn.Linear({width}, 2)\n"
        )
    return campaign


@pytest.fixture
def store(tmp_path, monkeypatch):
    campaign = make_campaign(tmp_path)
    monkeypatch.setattr(
        architecture_annotations, "enrich_run", lambda _root, payload: payload
    )
    return ArchitectureStore(tmp_path, {"main": campaign})


def test_unknown_scope_and_path_identifiers_are_rejected(store):
    with pytest.raises(ValueError, match="scope"):
        store.catalog("anything")
    with pytest.raises(KeyError):
        store.run("../../outside", "test-run")
    with pytest.raises(ValueError, match="run ID"):
        store.run("main", "../../outside")
    with pytest.raises(KeyError):
        store.snapshot("main", "test-run", 999)


def test_archive_discovery_cannot_leak_hidden_campaign_into_dashboard(store):
    name = "backup-invalid-launch-20260903"
    make_campaign(store.root, name)
    assert name in {row["key"] for row in store.catalog("archive")["campaigns"]}
    assert name not in {row["key"] for row in store.catalog("dashboard")["campaigns"]}
    with pytest.raises(KeyError):
        store.run(name, "test-run", "dashboard")
    assert store.run(name, "test-run", "archive")["campaign"] == name


def test_prepared_revision_requires_verification_and_keeps_its_origin(store):
    prepared = store.root / "outputs/architecture-replay-data/filtered"
    restored = make_campaign(prepared, width=8)
    assert store.location("main", "dashboard")[0] == store.root
    provenance = {
        "source_revision": "f" * 40,
        "archive_sha256": "a" * 64,
        "archive_filename": "recorded.zip",
        "checksum_verified": True,
    }
    write_json(prepared / "architecture-replay-source.json", provenance)
    assert store.location("main", "dashboard") == (prepared, restored)
    row = next(row for row in store.catalog()["campaigns"] if row["key"] == "main")
    assert row["dataset_provenance"] == provenance
    replay = store.run("main", "test-run")
    assert replay["dataset_provenance"] == provenance
    assert replay["source_revision"] == "f" * 40


def test_archive_preserves_raw_checkout_as_a_separate_accessible_revision(store):
    direct = store.root / "data/c0c3/test-campaign"
    extra = direct / "runs/raw-only-run"
    shutil.copytree(direct / "runs/test-run", extra)
    write_json(extra / "state.json", {"run_id": "raw-only-run", "condition": "C0"})
    prepared = store.root / "outputs/architecture-replay-data/filtered"
    make_campaign(prepared, width=8)
    write_json(
        prepared / "architecture-replay-source.json",
        {
            "source_revision": "f" * 40,
            "archive_sha256": "a" * 64,
            "checksum_verified": True,
        },
    )
    rows = {row["key"]: row for row in store.catalog("archive")["campaigns"]}
    raw = rows["main__tracked_archive"]
    assert raw["version_label"] == "Raw checkout history"
    assert raw["dataset_provenance"]["source_kind"] == "tracked_checkout"
    assert {run["id"] for run in raw["runs"]} == {"test-run", "raw-only-run"}
    assert [run["id"] for run in rows["main"]["runs"]] == ["test-run"]
    assert store.location("main__tracked_archive", "archive")[0] == store.root
    assert store.location("main", "archive")[0] == prepared
    raw_run = store.run("main__tracked_archive", "raw-only-run", "archive")
    assert raw_run["run_id"] == "raw-only-run"
    with pytest.raises(FileNotFoundError):
        store.run("main", "raw-only-run", "archive")
    with pytest.raises(KeyError):
        store.run("main__tracked_archive", "test-run", "dashboard")
    assert "main__tracked_archive" not in {
        row["key"] for row in store.catalog("dashboard")["campaigns"]
    }


def test_source_digest_cache_keeps_occurrence_envelopes_separate(store, monkeypatch):
    actual = architecture_api.extract_architecture
    calls = []

    def extract(*args, **kwargs):
        calls.append(kwargs.get("source_hash"))
        return actual(*args, **kwargs)

    monkeypatch.setattr(architecture_api, "extract_architecture", extract)
    seed = store.snapshot("main", "test-run", 0)
    child = store.snapshot("main", "test-run", 1)
    assert len(calls) == 1
    assert seed["source_hash"] == child["source_hash"]
    assert seed["occurrence_id"] != child["occurrence_id"]
    assert seed["candidate_id"] == "seed" and child["candidate_id"] == "child"
    assert seed["source_metadata"]["source_candidate_id"] == "seed"
    assert child["source_metadata"]["source_candidate_id"] == "child"


def test_api_export_matches_cli_portable_bundle_contract(store):
    bundle = store.export("main", "test-run")
    validate_architecture_bundle(json.loads(encode(bundle)))
    assert len(bundle["snapshots"]) == 2


def test_export_uses_one_pinned_run_instead_of_rescanning_catalog_per_frame(
    store, monkeypatch
):
    actual = store.catalog
    calls = []

    def catalog(scope="dashboard"):
        calls.append(scope)
        return actual(scope)

    monkeypatch.setattr(store, "catalog", catalog)
    store.export("main", "test-run")
    assert len(calls) <= 2


def test_source_change_invalidates_replay_and_graph_consistently(store):
    original = store.run("main", "test-run")
    source = (
        store.root / "data/c0c3/test-campaign/runs/test-run/candidates/child/train.py"
    )
    source.write_text(source.read_text().replace("Linear(4, 2)", "Linear(7, 2)"))
    bundle = store.export("main", "test-run")
    validate_architecture_bundle(bundle)
    assert bundle["run"]["revision"] != original["revision"]
    row = bundle["run"]["occurrences"][1]
    assert bundle["snapshots"][row["id"]]["source_hash"] == row["source"]["source_hash"]


def test_interrupted_workspace_gate_is_shared_by_run_snapshot_and_export(store):
    campaign = store.root / "data/c0c3/test-campaign"
    run = campaign / "runs/test-run"
    write_json(campaign / "inputs/task.json", {"editable_paths": ["train.py"]})
    events = [
        json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
    ]
    events[-1].update(
        candidate_id="interrupted",
        artifact_path="candidates/seed",
        evaluation={"valid": False, "failure_kind": "infrastructure_interruption"},
    )
    (run / "events.jsonl").write_text("\n".join(json.dumps(row) for row in events))
    unknown = store.run("main", "test-run")
    occurrence = unknown["occurrences"][1]
    assert occurrence["source"]["availability"] == "unresolved_recovery_fallback"
    assert store.snapshot("main", "test-run", 1)["nodes"] == []
    assert (
        store.export("main", "test-run")["snapshots"][occurrence["id"]]["nodes"] == []
    )
    workspace = run / "opportunities/0001/proposal-workspace/train.py"
    workspace.parent.mkdir(parents=True)
    workspace.write_text((run / "candidates/seed/train.py").read_text())
    proven = store.run("main", "test-run")
    assert proven["occurrences"][1]["source"]["available"]
    assert proven["revision"] != unknown["revision"]
    snapshot = store.snapshot(
        "main", "test-run", 1, expected_revision=proven["revision"]
    )
    assert snapshot["nodes"] and snapshot["source_id"] == "seed"
    workspace.write_text("# actual attempted edits differ from parent\n")
    with pytest.raises(ValueError, match="Data revision changed"):
        store.snapshot("main", "test-run", 1, expected_revision=proven["revision"])
    assert store.snapshot("main", "test-run", 1)["nodes"] == []


def test_expected_revision_pins_snapshot_and_export_to_the_loaded_run(store):
    original = store.run("main", "test-run")
    revision = original["revision"]
    snapshot = store.snapshot("main", "test-run", 1, expected_revision=revision)
    assert snapshot["run_revision"] == revision
    assert (
        store.export("main", "test-run", expected_revision=revision)["run"]["revision"]
        == revision
    )
    source = (
        store.root / "data/c0c3/test-campaign/runs/test-run/candidates/child/train.py"
    )
    source.write_text(source.read_text().replace("Linear(4, 2)", "Linear(9, 2)"))
    with pytest.raises(ValueError, match="Data revision changed; reload this run"):
        store.snapshot("main", "test-run", 1, expected_revision=revision)
    with pytest.raises(ValueError, match="Data revision changed; reload this run"):
        store.export("main", "test-run", expected_revision=revision)
    refreshed = store.run("main", "test-run")
    assert refreshed["revision"] != revision
    assert (
        store.snapshot("main", "test-run", 1, expected_revision=refreshed["revision"])[
            "run_revision"
        ]
        == refreshed["revision"]
    )


def test_server_routes_are_read_only_and_asset_names_are_allowlisted(
    store, monkeypatch
):
    monkeypatch.setattr(live_trajectory_dashboard, "REPO_ROOT", store.root)
    monkeypatch.setattr(
        live_trajectory_dashboard.DashboardPayloadCache, "prewarm", lambda _: None
    )
    handler_class = live_trajectory_dashboard.make_handler(
        store.campaigns, {}, capacity_controller=object(), compute_monitor=object()
    )
    handler = object.__new__(handler_class)
    captured = []
    handler.send_payload = lambda body, content_type, **kwargs: captured.append(
        (body, content_type, kwargs.get("status", 200))
    )
    handler.path = "/api/architectures/run?campaign=main&run=test-run"
    handler.do_GET()
    assert captured[-1][2] == 200
    assert json.loads(captured[-1][0])["run_id"] == "test-run"
    handler.path = "/api/architectures/run?campaign=main&run=..%2Fsecret"
    handler.do_GET()
    assert captured[-1][2] == 404
    handler.path = "/api/architectures/catalog?scope=unknown"
    handler.do_GET()
    assert captured[-1][2] == 404
    handler.path = "/architecture-assets/../../AGENTS.md"
    handler.do_GET()
    assert captured[-1][2] == 404

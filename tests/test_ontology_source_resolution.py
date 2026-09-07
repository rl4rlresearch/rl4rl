import json

import pytest

from experiments.live_trajectory_dashboard import DASHBOARD_EXCLUDED_RUN_IDS
from experiments.ontology_categorical_fingerprint import (
    CategoricalFingerprintError,
    annotate_run,
    campaign_schema,
    output_document,
)
from experiments.ontology_categorical_inventory import (
    _readable_path,
    _record_from_event,
    build_campaign_inventory,
)


def snapshot(root, name):
    path = root / "candidates" / name
    path.mkdir(parents=True)
    (path / "model.py").write_text("value = 1\n", encoding="utf-8")


def event(artifact="candidates/parent", failure="provider"):
    return {
        "candidate_id": "synthetic",
        "opportunity": 3,
        "parent_ids": ["parent"],
        "artifact_path": artifact,
        "evaluation": {"valid": False, "failure_kind": failure},
    }


def test_failed_attempt_resolves_recorded_parent_without_rewriting_lineage(tmp_path):
    snapshot(tmp_path, "parent")
    row = _record_from_event(tmp_path, event())
    assert row["candidate_id"] == "synthetic"
    assert row["ontology_parent_id"] == "parent"
    assert row["source_path"] == "candidates/parent"
    assert row["source_status"] == "available"
    assert row["source_resolution"]["kind"] == "recorded_artifact_alias"
    assert row["source_resolution"]["artifact_is_primary_parent"] is True


def test_duplicate_can_resolve_other_snapshot_and_still_change_from_parent(tmp_path):
    snapshot(tmp_path, "earlier")
    resolved = _record_from_event(tmp_path, event("candidates/earlier", "duplicate"))
    assert resolved["source_path"] == "candidates/earlier"
    assert resolved["ontology_parent_id"] == "parent"
    assert resolved["source_resolution"]["artifact_is_primary_parent"] is False
    schema = campaign_schema("addition")
    first = {c["id"]: "absent" for c in schema["components"]}
    parent = dict(first, token_embedding="learned_lookup")
    rows = annotate_run(
        [
            {
                "proposal": 0,
                "candidate_id": "earlier",
                "ontology_parent_id": None,
                "fingerprint": first,
            },
            {
                "proposal": 1,
                "candidate_id": "parent",
                "ontology_parent_id": "earlier",
                "fingerprint": parent,
            },
            dict(resolved, fingerprint=first),
        ],
        schema,
    )
    assert rows[-1]["metrics"]["component_edits_marginal"] == 1
    assert rows[-1]["metrics"]["family_switches_marginal"] == 1
    assert rows[-1]["metrics"]["new_component_states_marginal"] == 0
    assert rows[-1]["metrics"]["new_families_marginal"] == 0


def test_interruption_parent_fallback_does_not_prove_attempted_source(tmp_path):
    snapshot(tmp_path, "parent")
    row = _record_from_event(tmp_path, event(failure="infrastructure_interruption"))
    assert row["source_status"] == "missing"
    assert row["source_resolution"]["kind"] == "unresolved_recovery_fallback"


@pytest.mark.parametrize(
    "workspace_text,expected",
    [("value = 1\n", "available"), ("value = 2\n", "missing")],
)
def test_interrupted_workspace_must_match_every_editable_file(
    tmp_path, workspace_text, expected
):
    run = tmp_path / "runs" / "run"
    snapshot(run, "parent")
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/task.json").write_text(
        json.dumps({"editable_paths": ["model.py"]}), encoding="utf-8"
    )
    workspace = run / "opportunities/0003/proposal-workspace"
    workspace.mkdir(parents=True)
    (workspace / "model.py").write_text(workspace_text, encoding="utf-8")
    row = _record_from_event(run, event(failure="infrastructure_interruption"))
    assert row["source_status"] == expected
    if expected == "available":
        assert row["source_resolution"]["interruption_evidence"]["editable_sha256"]
        assert row["candidate_id"] == "synthetic"


@pytest.mark.parametrize(
    "artifact", ["../candidates/parent", "candidates/../parent", "/candidates/parent"]
)
def test_invalid_artifact_reference_cannot_supply_candidate_source(tmp_path, artifact):
    snapshot(tmp_path, "parent")
    assert _record_from_event(tmp_path, event(artifact))["source_status"] == "missing"


def test_addition_uses_backend_run_exclusions(tmp_path):
    root = _readable_path(tmp_path / campaign_schema("addition")["data_directory"])
    excluded = sorted(DASHBOARD_EXCLUDED_RUN_IDS)[0]
    for run_id in ["included", excluded]:
        run = root / "runs" / run_id
        snapshot(run, "seed")
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "baseline": {"candidate_id": "seed"},
                    "assignment": {"condition": "C0"},
                }
            ),
            encoding="utf-8",
        )
        (run / "events.jsonl").write_text("", encoding="utf-8")
    (root / "campaign.json").write_text(
        json.dumps({"primary_run_ids": ["included", excluded]}), encoding="utf-8"
    )
    inventory = build_campaign_inventory("addition", tmp_path)
    assert [r["run_id"] for r in inventory["runs"]] == ["included"]
    assert inventory["excluded_runs"][0]["run_id"] == excluded


def test_no_change_attempt_has_zero_marginals_and_preserves_totals():
    schema = campaign_schema("addition")
    seed = {c["id"]: "absent" for c in schema["components"]}
    changed = dict(seed, token_embedding="learned_lookup")
    rows = annotate_run(
        [
            {
                "proposal": 0,
                "candidate_id": "seed",
                "ontology_parent_id": None,
                "fingerprint": seed,
            },
            {
                "proposal": 1,
                "candidate_id": "parent",
                "ontology_parent_id": "seed",
                "fingerprint": changed,
            },
            {
                "proposal": 2,
                "candidate_id": "failed",
                "ontology_parent_id": "parent",
                "fingerprint": changed,
            },
        ],
        schema,
    )
    for key, value in rows[-1]["metrics"].items():
        if key.endswith("_marginal"):
            assert value == 0
        else:
            assert value == rows[-2]["metrics"][key]


def test_publication_rejects_dashboard_excluded_addition_run():
    excluded = sorted(DASHBOARD_EXCLUDED_RUN_IDS)[0]
    with pytest.raises(CategoricalFingerprintError, match="excluded by dashboard"):
        output_document("addition", {excluded: []})

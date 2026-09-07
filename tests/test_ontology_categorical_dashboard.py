import json
from pathlib import Path

import pytest

from experiments import ontology_categorical_dashboard as dashboard
from experiments.ontology_categorical_fingerprint import (
    campaign_schema,
    schema_revision,
)


def publish(tmp_path):
    base = tmp_path / "addition"
    base.mkdir()
    schema = campaign_schema("addition")
    marginals = {
        "component_edits": [0, 2, 2, 1, 2, 1],
        "new_component_states": [0, 2, 0, 1, 0, 0],
        "family_switches": [0, 1, 1, 1, 1, 1],
        "new_families": [0, 1, 0, 1, 0, 0],
    }
    rows = []
    for proposal in range(6):
        metrics = {}
        for key, values in marginals.items():
            metrics[key + "_marginal"] = values[proposal]
            metrics[key + "_cumulative"] = sum(values[: proposal + 1])
        rows.append(
            {
                "proposal": proposal,
                "candidate_id": "candidate" if proposal == 3 else f"p{proposal}",
                "is_seed": proposal == 0,
                "retained": proposal not in (1, 4),
                "metrics": metrics,
            }
        )
    document = {
        "campaign": "addition",
        "fingerprint_version": dashboard.FINGERPRINT_VERSION,
        "schema": schema,
        "schema_revision": schema_revision(schema),
        "review_audits": {row["candidate_id"]: {} for row in rows},
        "runs": {"run": rows},
    }
    (base / "working-schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (base / "final.json").write_text(json.dumps(document), encoding="utf-8")
    dashboard.write_dashboard_metrics(base)
    return base, document, Path(schema["data_directory"])


def test_exact_occurrence_join_and_compact_read(tmp_path, monkeypatch):
    base, document, campaign = publish(tmp_path)
    dashboard._metric_index.cache_clear()
    original_read = dashboard.read

    def read(path):
        assert path.name != "final.json", "Dashboard should use the compact metric view"
        return original_read(path)

    monkeypatch.setattr(dashboard, "read", read)
    points = [
        {"proposal": 3, "candidate_id": "candidate"},
        {"proposal": 4, "candidate_id": "candidate"},
        {"proposal": 3, "candidate_id": "different"},
    ]
    runs = [
        {"run_id": "run", "points": points},
        {"run_id": "wrong", "points": [dict(points[0])]},
    ]
    result = dashboard.attach_categorical_metrics(campaign, runs, tmp_path)
    assert result["matched_points"] == 1
    assert points[0]["ontology_metrics"]["implemented:component_edits_cumulative"] == 5
    assert points[0]["ontology_metrics"]["retained:component_edits_cumulative"] == 3
    assert points[1]["ontology_metrics"] is None
    assert points[2]["ontology_metrics"] is None
    assert runs[1]["points"][0]["ontology_metrics"] is None
    assert len(dashboard.AXES) == 16
    assert {axis["key"].removeprefix("ontology:") for axis in dashboard.AXES} == set(
        points[0]["ontology_metrics"]
    )


def test_republication_invalidates_compact_cache_and_schema_changes_fail_closed(
    tmp_path,
):
    base, document, campaign = publish(tmp_path)
    runs = [{"run_id": "run", "points": [{"proposal": 3, "candidate_id": "candidate"}]}]
    assert dashboard.attach_categorical_metrics(campaign, runs, tmp_path)["available"]
    document["runs"]["run"][3]["metrics"]["component_edits_cumulative"] = 99
    (base / "final.json").write_text(json.dumps(document), encoding="utf-8")
    dashboard.attach_categorical_metrics(campaign, runs, tmp_path)
    assert (
        runs[0]["points"][0]["ontology_metrics"][
            "implemented:component_edits_cumulative"
        ]
        == 99
    )
    changed_schema = dict(
        document["schema"], description="Revised component definitions"
    )
    (base / "working-schema.json").write_text(
        json.dumps(changed_schema), encoding="utf-8"
    )
    assert not dashboard.attach_categorical_metrics(campaign, runs, tmp_path)[
        "available"
    ]
    assert runs[0]["points"][0]["ontology_metrics"] is None


def test_unreviewed_row_cannot_be_exported_as_metrics(tmp_path):
    _, document, _ = publish(tmp_path)
    document["review_audits"] = {}
    with pytest.raises(ValueError, match="no source-review audit"):
        dashboard.compact_document(document, [0, 0])


def test_retained_contributions_and_full_run_novelty(tmp_path):
    _, document, _ = publish(tmp_path)
    compact = dashboard.compact_document(document, [0, 0])
    rows = compact["runs"]["run"]
    assert len(rows) == 6, "Rejected proposals must remain in the trajectory"
    assert all(value == 0 for value in rows[0]["metrics"].values())
    for key in dashboard.METRICS:
        assert [row["metrics"]["implemented:" + key] for row in rows] == [
            row["metrics"][key] for row in document["runs"]["run"]
        ]
        if key.endswith("_marginal"):
            expected = [
                row["metrics"][key] if row["retained"] else 0
                for row in document["runs"]["run"]
            ]
            assert [row["metrics"]["retained:" + key] for row in rows] == expected
            total_key = "retained:" + key.replace("_marginal", "_cumulative")
            assert [row["metrics"][total_key] for row in rows] == [
                sum(expected[: i + 1]) for i in range(len(expected))
            ]
    # Proposal 2 retains a previously rejected discovery; it changes components
    # but does not discover a new state/family in the full run history.
    assert rows[2]["metrics"]["retained:component_edits_marginal"] == 2
    assert rows[2]["metrics"]["retained:new_component_states_marginal"] == 0
    assert rows[2]["metrics"]["retained:new_families_marginal"] == 0
    assert [row["metrics"]["retained:component_edits_cumulative"] for row in rows] == [
        0,
        0,
        2,
        3,
        3,
        4,
    ]
    # Sorting happens before accumulation and the source publication is untouched.
    document["runs"]["run"].reverse()
    assert dashboard.compact_document(document, [0, 0]) == compact
    assert document["runs"]["run"][0]["proposal"] == 5


def test_missing_retention_and_proposal_gaps_do_not_become_zero(tmp_path):
    _, document, _ = publish(tmp_path)
    del document["runs"]["run"][1]["retained"]
    rows = dashboard.compact_document(document, [0, 0])["runs"]["run"]
    assert rows[1]["metrics"]["implemented:component_edits_marginal"] == 2
    assert rows[1]["metrics"]["retained:component_edits_marginal"] is None
    assert rows[2]["metrics"]["retained:component_edits_marginal"] == 2
    assert all(
        row["metrics"]["retained:component_edits_cumulative"] is None
        for row in rows[1:]
    )
    del document["runs"]["run"][1]
    rows = dashboard.compact_document(document, [0, 0])["runs"]["run"]
    assert rows[1]["proposal"] == 2
    assert rows[1]["metrics"]["retained:component_edits_marginal"] == 2
    assert rows[1]["metrics"]["retained:component_edits_cumulative"] is None


def test_old_compact_format_is_rebuilt_without_republishing_sources(tmp_path):
    base, document, campaign = publish(tmp_path)
    compact_path = base / "trajectory-metrics.json"
    old = dashboard.read(compact_path)
    old.pop("metric_view_version")
    old["runs"]["run"] = document["runs"]["run"]
    compact_path.write_text(json.dumps(old), encoding="utf-8")
    before = (base / "final.json").read_bytes()
    runs = [{"run_id": "run", "points": [{"proposal": 3, "candidate_id": "candidate"}]}]
    assert dashboard.attach_categorical_metrics(campaign, runs, tmp_path)["available"]
    assert len(runs[0]["points"][0]["ontology_metrics"]) == 16
    assert (base / "final.json").read_bytes() == before
    assert "metric_view_version" not in dashboard.read(compact_path), (
        "Serving metrics is read-only"
    )


def test_portfolio_comparisons_use_whole_vectors_and_actual_preproposal_members(
    tmp_path,
):
    base, document, campaign = publish(tmp_path)
    # Proposal 5 matches an older portfolio member, but differs from its
    # primary parent (proposal 3) and its immediately previous proposal.
    fingerprints = [
        {"a": "x", "b": "x", "c": "x"},
        {"a": "x", "b": "y", "c": "x"},
        {"a": "y", "b": "x", "c": "x"},
        {"a": "x", "b": "x", "c": "y"},
        {"a": "y", "b": "y", "c": "y"},
        {"a": "x", "b": "y", "c": "x"},
    ]
    for row, fingerprint in zip(document["runs"]["run"], fingerprints, strict=True):
        row.update(fingerprint=fingerprint, condition="C2")
    (base / "final.json").write_text(json.dumps(document), encoding="utf-8")
    points = [
        {
            "proposal": 5,
            "candidate_id": "p5",
            "visible_candidate_ids": ["p0", "p1", "p2", "candidate"],
            "portfolio_after": ["p5"],
        }
    ]
    runs = [{"run_id": "run", "points": points}]
    assert dashboard.attach_categorical_metrics(campaign, runs, tmp_path)["available"]
    compared = points[0]["ontology_comparisons"]
    assert compared["previous_proposal"] == 4
    assert compared["parent_ids"] == ["p0", "p1", "p2", "candidate"]
    assert compared["previous"]["implemented:component_edits_marginal"] == 2
    assert compared["minimum_parents"]["implemented:component_edits_marginal"] == 0
    assert compared["minimum_parents"]["implemented:family_switches_marginal"] == 0
    assert compared["previous"]["retained:component_edits_marginal"] == 2
    # Full published history is available even when the displayed API run only
    # contains proposal 5. Post-proposal portfolio_after must never be used.
    assert points[0]["ontology_metrics"]["implemented:component_edits_marginal"] == 1
    points[0]["visible_candidate_ids"] = ["p0", "p2"]
    dashboard.attach_categorical_metrics(campaign, runs, tmp_path)
    assert (
        points[0]["ontology_comparisons"]["minimum_parents"][
            "implemented:component_edits_marginal"
        ]
        == 1
    )
    # Incomplete portfolio evidence is unavailable, not a minimum over a subset.
    points[0]["visible_candidate_ids"] = ["p1", "unreviewed"]
    dashboard.attach_categorical_metrics(campaign, runs, tmp_path)
    assert (
        points[0]["ontology_comparisons"]["minimum_parents"][
            "implemented:component_edits_marginal"
        ]
        is None
    )
    points[0].pop("visible_candidate_ids")
    dashboard.attach_categorical_metrics(campaign, runs, tmp_path)
    assert (
        points[0]["ontology_comparisons"]["minimum_parents"][
            "implemented:family_switches_marginal"
        ]
        is None
    )


def test_comparison_distances_do_not_mix_parents_and_retention_gates_results():
    def row(proposal, a, b, retained=True):
        return {
            "proposal": proposal,
            "condition": "C3",
            "fingerprint": {"a": a, "b": b},
            "retained": retained,
            "metrics": {
                "retained:component_edits_marginal": int(retained),
                "retained:family_switches_marginal": int(retained),
            },
        }

    candidate = row(5, "y", "y")
    parents = [row(1, "y", "x"), row(2, "x", "y")]
    values = dashboard.comparison_metrics(candidate, row(4, "x", "x"), parents)
    assert values["previous"]["implemented:component_edits_marginal"] == 2
    assert values["minimum_parents"]["implemented:component_edits_marginal"] == 1
    assert values["minimum_parents"]["implemented:family_switches_marginal"] == 1
    candidate["retained"] = False
    values = dashboard.comparison_metrics(candidate, None, parents)
    assert values["minimum_parents"]["implemented:component_edits_marginal"] == 1
    assert values["minimum_parents"]["retained:component_edits_marginal"] == 0
    assert values["previous"]["retained:component_edits_marginal"] is None
    seed = dashboard.comparison_metrics(row(0, "x", "x"), None, [])
    assert all(
        value == 0
        for mode in ("previous", "minimum_parents")
        for value in seed[mode].values()
    )

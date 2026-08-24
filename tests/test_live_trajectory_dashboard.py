from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from experiments.live_trajectory_dashboard import (
    PAGE,
    build_run,
    campaign_data,
    dashboard_data,
    encode_response,
    weighted_cost,
)

PRICES = {"input": 1.75, "cached_input": 0.175, "output": 14.0}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(value) for value in values) + "\n",
        encoding="utf-8",
    )


def _example_run(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "example-b01-c2"
    _write_json(
        run / "state.json",
        {
            "run_id": "example-b01-c2",
            "condition": "C2",
            "status": "running",
            "proposals_used": 2,
            "usage": {
                "input_tokens": 300,
                "cached_input_tokens": 200,
                "output_tokens": 30,
            },
            "candidates": {
                "seed": {
                    "created_opportunity": 0,
                    "fitness": -100,
                    "metrics": {"parameters": 100, "accuracy": 0.99},
                }
            },
        },
    )
    _write_jsonl(
        run / "events.jsonl",
        [
            {
                "event": "proposal_started",
                "opportunity": 1,
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "event": "proposal_completed",
                "opportunity": 1,
                "timestamp": "2026-01-01T00:01:00+00:00",
                "evaluation": {
                    "valid": False,
                    "failure_kind": "nonqualification",
                    "fitness": None,
                    "metrics": {"parameters": 50, "accuracy": 0.8},
                },
                "retained": False,
                "retention_decision": "invalid",
                "proposal_type": "ordinary",
                "usage_cumulative": {
                    "input_tokens": 100,
                    "cached_input_tokens": 40,
                    "output_tokens": 10,
                    "total_tokens": 110,
                },
                "usage_increment": {
                    "input_tokens": 100,
                    "cached_input_tokens": 40,
                    "output_tokens": 10,
                    "total_tokens": 110,
                },
                "evaluator_seconds_cumulative": 45,
                "evaluator_seconds_increment": 45,
                "evaluator_calls_cumulative": 1,
                "evaluator_calls_increment": 1,
            },
            {
                "event": "proposal_started",
                "opportunity": 2,
                "timestamp": "2026-01-01T00:10:00+00:00",
            },
            {
                "event": "proposal_completed",
                "opportunity": 2,
                "timestamp": "2026-01-01T00:12:00+00:00",
                "evaluation": {
                    "valid": True,
                    "failure_kind": None,
                    "fitness": -90,
                    "metrics": {"parameters": 90, "accuracy": 0.995},
                },
                "retained": True,
                "retention_decision": "strict_incumbent_improvement",
                "proposal_type": "assumption_challenge",
                "usage_cumulative": {
                    "input_tokens": 300,
                    "cached_input_tokens": 200,
                    "output_tokens": 30,
                    "total_tokens": 330,
                },
                "usage_increment": {
                    "input_tokens": 200,
                    "cached_input_tokens": 160,
                    "output_tokens": 20,
                    "total_tokens": 220,
                },
                "evaluator_seconds_cumulative": 150,
                "evaluator_seconds_increment": 105,
                "evaluator_calls_cumulative": 2,
                "evaluator_calls_increment": 1,
            },
        ],
    )
    return run


def test_dashboard_data_keeps_legacy_refresh_keys(tmp_path: Path) -> None:
    autoresearch = tmp_path / "autoresearch"
    openevolve = tmp_path / "openevolve"
    autoresearch.mkdir()
    openevolve.mkdir()
    data = dashboard_data(
        {
            "autoresearch_v16": autoresearch,
            "openevolve_v2": openevolve,
            "autoresearch_v17": tmp_path / "not-started",
        },
        PRICES,
    )

    assert data["schema_version"] == "3.0"
    assert data["autoresearch"] == data["campaigns"]["autoresearch_v16"]
    assert data["openevolve_v2"] == data["campaigns"]["openevolve_v2"]
    assert data["campaigns"]["autoresearch_v17"]["available"] is False


def test_build_run_keeps_every_raw_outcome_and_only_advances_valid_best(
    tmp_path: Path,
) -> None:
    run = build_run(
        _example_run(tmp_path),
        PRICES,
        objective_metric="parameters",
        objective_direction="minimize",
    )

    assert run is not None
    assert len(run["points"]) == 3
    seed, invalid, valid = run["points"]
    assert seed["raw_objective"] == 100
    assert invalid["raw_objective"] == 50
    assert invalid["valid"] is False
    assert invalid["best_objective"] == 100
    assert valid["raw_objective"] == 90
    assert valid["best_objective"] == 90
    assert run["best_objective"] == 90
    assert run["latest_raw_objective"] == 90
    assert run["valid_proposals"] == 1
    assert run["invalid_proposals"] == 1
    assert run["retained_proposals"] == 1


def test_build_run_excludes_the_gap_between_proposals_from_active_time(
    tmp_path: Path,
) -> None:
    run = build_run(
        _example_run(tmp_path),
        PRICES,
        objective_metric="parameters",
        objective_direction="minimize",
    )

    assert run is not None
    assert run["active_hours"] == pytest.approx(3 / 60)
    assert run["points"][1]["active_seconds"] == 60
    assert run["points"][2]["active_seconds"] == 180


def test_build_run_uses_lifecycle_status_for_cooperative_pause(tmp_path: Path) -> None:
    run_dir = _example_run(tmp_path)
    _write_jsonl(
        run_dir / "lifecycle.jsonl",
        [
            {"event": "trajectory_started"},
            {"event": "trajectory_pause_requested"},
            {"event": "trajectory_paused"},
        ],
    )

    run = build_run(
        run_dir,
        PRICES,
        objective_metric="parameters",
        objective_direction="minimize",
    )

    assert run is not None
    assert run["status"] == "paused"
    assert run["scientific_status"] == "running"


def test_build_run_returns_to_scientific_status_after_resume(tmp_path: Path) -> None:
    run_dir = _example_run(tmp_path)
    _write_jsonl(
        run_dir / "lifecycle.jsonl",
        [
            {"event": "trajectory_paused"},
            {"event": "trajectory_resumed"},
        ],
    )

    run = build_run(
        run_dir,
        PRICES,
        objective_metric="parameters",
        objective_direction="minimize",
    )

    assert run is not None
    assert run["status"] == "running"


def test_cost_fields_distinguish_cumulative_and_incremental_usage(
    tmp_path: Path,
) -> None:
    run = build_run(
        _example_run(tmp_path),
        PRICES,
        objective_metric="parameters",
        objective_direction="minimize",
    )

    assert run is not None
    final = run["points"][-1]
    assert final["token_cost"] == pytest.approx(
        weighted_cost(
            {
                "input_tokens": 300,
                "cached_input_tokens": 200,
                "output_tokens": 30,
            },
            PRICES,
        ),
        abs=1e-6,
    )
    assert final["incremental_total_tokens"] == 220
    assert final["incremental_evaluator_seconds"] == 105


def test_campaign_catalog_exposes_observed_metrics_for_both_axes(
    tmp_path: Path,
) -> None:
    _example_run(tmp_path)
    _write_json(
        tmp_path / "inputs/task.json",
        {"objective_metric": "parameters", "objective_direction": "minimize"},
    )

    campaign = campaign_data(tmp_path, PRICES)
    keys = {item["key"] for item in campaign["axis_catalog"]}

    assert {"best_objective", "raw_objective", "token_cost", "active_hours"} <= keys
    assert {"metric:parameters", "metric:accuracy"} <= keys


def test_page_contains_live_controls_and_raw_outcome_overlay() -> None:
    assert "Refresh now" in PAGE
    assert "Auto-refresh" in PAGE
    assert "Raw outcome overlay" in PAGE
    assert "Y-axis range" in PAGE
    assert "Fit visible data" in PAGE
    assert "Individual runs" in PAGE
    assert "Condition median" in PAGE
    assert "aggregateDataset" in PAGE
    assert "median of ${p.member_count}/${p.condition_run_count} runs" in PAGE
    assert "Export visible CSV" in PAGE
    assert "beginAtZero:yScale==='zero'" in PAGE
    assert "if(!point.valid)return 'crossRot';return 'circle'" in PAGE
    assert "return 'triangle'" not in PAGE
    assert "Marker key: ● seed or retained" in PAGE
    assert "location.protocol==='file:'" in PAGE
    assert "fetch(apiUrl" in PAGE
    assert "IntersectionObserver" in PAGE
    assert "request timed out; try again" in PAGE
    assert "https://unpkg.com/chart.js@4.4.4" in PAGE
    assert "autoresearch_v17_fashion_mnist" in PAGE
    assert "openevolve_v21_fashion_mnist" in PAGE


def test_large_responses_use_gzip_when_the_client_accepts_it() -> None:
    body = b"trajectory-data" * 1000

    encoded, encoding = encode_response(body, "br, gzip, deflate")

    assert encoding == "gzip"
    assert len(encoded) < len(body)
    assert gzip.decompress(encoded) == body


def test_small_or_unsupported_responses_are_not_compressed() -> None:
    small = b"dashboard"

    assert encode_response(small, "gzip") == (small, None)
    assert encode_response(b"x" * 2000, "identity") == (b"x" * 2000, None)

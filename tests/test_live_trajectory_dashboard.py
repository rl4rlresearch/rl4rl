from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path

import pytest

from experiments.live_trajectory_dashboard import (
    DEFAULT_MODAL_H100_PRICE_PER_SECOND,
    PAGE,
    build_run,
    campaign_data,
    codex_primary_rate_limit_sample,
    codex_rate_limit_payload,
    dashboard_data,
    dashboard_revision,
    encode_response,
    experiment_only_minutes_per_percent,
    local_codex_session_usage,
    local_codex_token_usage,
    projected_remaining_percent,
    projected_zero_crossing_time,
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

    assert data["schema_version"] == "3.1"
    assert data["autoresearch"] == data["campaigns"]["autoresearch_v16"]
    assert data["openevolve_v2"] == data["campaigns"]["openevolve_v2"]
    assert data["campaigns"]["autoresearch_v17"]["available"] is False


def test_codex_primary_rate_limit_sample_keeps_only_primary_percentage() -> None:
    sample = codex_primary_rate_limit_sample(
        {
            "rateLimitsByLimitId": {
                "codex": {
                    "primary": {"usedPercent": 16, "resetsAt": 1_800_000_000},
                    "secondary": {"usedPercent": 44},
                    "planType": "pro",
                },
                "other_product": {"primary": {"usedPercent": 1}},
            }
        }
    )

    assert sample is not None
    assert sample["used_percent"] == 16
    assert sample["reset_at"] == 1_800_000_000
    assert set(sample) == {
        "timestamp",
        "used_percent",
        "reset_at",
        "window_duration_minutes",
    }


def test_codex_rate_limit_payload_reports_rolling_minutes_per_percent(
    tmp_path: Path,
) -> None:
    history = tmp_path / "codex-rate-limit-history.jsonl"
    _write_jsonl(
        history,
        [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "used_percent": 10,
                "reset_at": 1_800_000_000,
            },
            {
                "timestamp": "2026-01-01T00:05:00+00:00",
                "used_percent": 12,
                "reset_at": 1_800_000_000,
            },
            {
                "timestamp": "2026-01-01T00:15:00+00:00",
                "used_percent": 13,
                "reset_at": 1_800_000_000,
            },
        ],
    )

    payload = codex_rate_limit_payload(history, session_root=tmp_path / "sessions")

    assert payload["available"] is True
    assert payload["sample_count"] == 3
    assert payload["latest_minutes_per_percent"] == pytest.approx(5.0)
    assert payload["latest_used_percent"] == 13
    assert payload["reset_at_iso"] is not None
    assert payload["account_wide_projected_remaining_percent"] is not None
    assert payload["experiment_only_projected_remaining_percent"] is None
    assert payload["account_wide_zero_crossing_at"] is not None
    assert payload["experiment_only_zero_crossing_at"] is None
    # The headline is the weighted average of real quota-change boundaries.
    # Individual points remain individual observed intervals and do not grow
    # while a percentage remains flat.
    assert payload["points"][-1]["minutes_per_percent"] == pytest.approx(10.0)


def test_projected_remaining_percent_can_be_negative() -> None:
    observed_at = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    reset_at = datetime.fromisoformat("2026-01-01T04:00:00+00:00")

    projected = projected_remaining_percent(
        70,
        2,
        from_time=observed_at,
        reset_time=reset_at,
    )

    assert projected == pytest.approx(-90.0)


def test_projected_zero_crossing_time_uses_remaining_quota() -> None:
    observed_at = datetime.fromisoformat("2026-01-01T00:00:00+00:00")

    assert projected_zero_crossing_time(
        75,
        2,
        from_time=observed_at,
    ) == datetime.fromisoformat("2026-01-01T00:50:00+00:00")
    assert projected_zero_crossing_time(
        100,
        2,
        from_time=observed_at,
    ) == observed_at


def test_local_session_attribution_uses_completed_turn_token_records(
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "sessions"
    timestamp = "2026-01-01T00:05:00+00:00"
    _write_jsonl(
        session_root / "2026/01/01/experiment.jsonl",
        [
            {
                "type": "session_meta",
                "payload": {"cwd": "/tmp/transformer-design-cycle-abc"},
            },
            {
                "timestamp": timestamp,
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {"input_tokens": 80, "output_tokens": 20}
                    },
                },
            },
        ],
    )
    _write_jsonl(
        session_root / "2026/01/01/app.jsonl",
        [
            {"type": "session_meta", "payload": {"cwd": "/project"}},
            {
                "timestamp": timestamp,
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {"input_tokens": 20, "output_tokens": 5}
                    },
                },
            },
        ],
    )

    usage = local_codex_session_usage(
        session_root=session_root,
        window_start=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        window_end=datetime.fromisoformat("2026-01-01T00:10:00+00:00"),
    )

    assert usage["experiment_runners"]["total"]["total_tokens"] == 100
    assert usage["other_local_codex"]["total"]["total_tokens"] == 25
    assert experiment_only_minutes_per_percent(
        50, experiment_tokens=100, all_local_tokens=125
    ) == pytest.approx(62.5)


def test_local_codex_token_usage_groups_types_and_runtime_settings(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    run = campaign / "runs" / "example-c0"
    _write_json(
        campaign / "inputs/protocol.json",
        {
            "conversation_mode": "continuous",
            "model": {
                "name": "gpt-5.6-sol",
                "reasoning_effort": "xhigh",
                "service_tier": "default",
                "sandbox": "workspace-write",
                "approval_policy": "never",
            },
        },
    )
    _write_json(run / "state.json", {"active": {"index": 2}})
    _write_jsonl(
        run / "events.jsonl",
        [
            {
                "event": "proposal_completed",
                "opportunity": 1,
                "timestamp": "2026-01-01T00:05:00+00:00",
                "codex_service_tier": "fast",
                "usage_increment": {
                    "input_tokens": 100,
                    "cached_input_tokens": 40,
                    "output_tokens": 30,
                    "reasoning_output_tokens": 20,
                },
            },
            {
                "event": "v3_proposal_provenance",
                "opportunity": 1,
                "provider_model_requested": "gpt-5.6-sol",
                "reasoning_effort_requested": "xhigh",
                "service_tier_observed": "fast",
            },
        ],
    )

    usage = local_codex_token_usage(
        [campaign],
        window_start=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        window_end=datetime.fromisoformat("2026-01-01T00:10:00+00:00"),
    )

    assert usage["total"] == {
        "total_tokens": 130,
        "input_tokens": 100,
        "cached_input_tokens": 40,
        "uncached_input_tokens": 60,
        "output_tokens": 30,
        "reasoning_output_tokens": 20,
        "nonreasoning_output_tokens": 10,
    }
    assert usage["active_calls_not_yet_logged"] == 1
    assert usage["models"][0]["model"] == "gpt-5.6-sol"
    assert usage["models"][0]["service_tier"] == "fast"
    assert usage["models"][0]["conversation_mode"] == "continuous"


def test_codex_rate_limit_payload_does_not_cross_a_primary_window_reset(
    tmp_path: Path,
) -> None:
    history = tmp_path / "codex-rate-limit-history.jsonl"
    _write_jsonl(
        history,
        [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "used_percent": 95,
                "reset_at": 1_800_000_000,
            },
            {
                "timestamp": "2026-01-01T00:05:00+00:00",
                "used_percent": 3,
                "reset_at": 1_800_018_000,
            },
        ],
    )

    payload = codex_rate_limit_payload(history, session_root=tmp_path / "sessions")

    assert payload["latest_minutes_per_percent"] is None
    assert payload["points"] == []


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


def test_campaign_data_attributes_modal_gpu_cost_to_campaign_and_run(
    tmp_path: Path,
) -> None:
    run_dir = _example_run(tmp_path)
    _write_json(
        tmp_path / "inputs/task.json",
        {"objective_metric": "parameters", "objective_direction": "minimize"},
    )
    _write_jsonl(
        tmp_path / "modal-usage.jsonl",
        [
            {
                "run_id": run_dir.name,
                "opportunity": 1,
                "status": "completed",
                "worker_seconds": 10,
            },
            {
                "run_id": run_dir.name,
                "opportunity": 2,
                "status": "completed",
                "worker_seconds": 5.5,
            },
            {
                "run_id": run_dir.name,
                "opportunity": 2,
                "status": "failed",
                "worker_seconds": None,
            },
        ],
    )

    campaign = campaign_data(tmp_path, PRICES)
    run = campaign["runs"][0]

    assert campaign["modal_usage"]["available"] is True
    assert campaign["modal_usage"]["worker_seconds"] == pytest.approx(15.5)
    assert campaign["modal_usage"]["gpu_cost"] == pytest.approx(
        round(15.5 * DEFAULT_MODAL_H100_PRICE_PER_SECOND, 6)
    )
    assert campaign["modal_usage"]["completed_calls"] == 2
    assert campaign["modal_usage"]["failed_calls"] == 1
    assert run["modal_worker_seconds"] == pytest.approx(15.5)
    assert run["modal_gpu_cost"] == pytest.approx(
        round(15.5 * DEFAULT_MODAL_H100_PRICE_PER_SECOND, 6)
    )
    assert run["points"][1]["incremental_modal_worker_seconds"] == 10
    assert run["points"][2]["modal_worker_seconds"] == pytest.approx(15.5)
    assert {"modal_gpu_cost", "incremental_modal_gpu_cost"} <= {
        item["key"] for item in campaign["axis_catalog"]
    }


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

    assert {
        "best_objective",
        "raw_objective",
        "token_cost",
        "active_hours",
    } <= keys
    assert {"metric:parameters", "metric:accuracy"} <= keys


def test_semantic_campaign_uses_arm_labels_and_charges_shared_prefix_once(
    tmp_path: Path,
) -> None:
    run_dir = _example_run(tmp_path)
    _write_json(
        tmp_path / "inputs/task.json",
        {"objective_metric": "parameters", "objective_direction": "minimize"},
    )
    _write_json(
        tmp_path / "campaign.json",
        {
            "schema_version": "4.0",
            "design": "multi_arm_semantic_interventions_with_shared_prefix_v1",
            "intervention_count": 2,
            "replicates": 1,
            "shared_prefix_opportunities": 1,
        },
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "assignment": {
                "replicate": 1,
                "order": 2,
                "condition": "assumption_challenge",
                "condition_label": "Assumption challenge",
                "condition_family": "epistemic",
                "components": ["assumption_challenge"],
            }
        },
    )
    _write_json(
        tmp_path / "semantic-prefix.json",
        {
            "replicates": [
                {
                    "leader_run_id": "leader",
                    "shadow_run_ids": [run_dir.name],
                    "shared_through_opportunity": 1,
                }
            ]
        },
    )
    _write_json(
        tmp_path / "semantic-run-control.json",
        {"runs": {run_dir.name: {"desired": "paused"}}},
    )
    existing = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    existing.extend(
        [
            {
                "event": "semantic_intervention_applied",
                "opportunity": 2,
                "intervention_id": "assumption_challenge",
            },
            {
                "event": "developmental_assessment",
                "opportunity": 2,
                "status": "primary_retained",
                "credit": 1.0,
                "reasons": ["novel_delta", "retained"],
                "selection_effect": "none",
            },
        ]
    )
    _write_jsonl(run_dir / "events.jsonl", existing)

    campaign = campaign_data(tmp_path, PRICES)
    run = campaign["runs"][0]

    assert campaign["semantic"] is True
    assert run["condition"] == "assumption_challenge"
    assert run["condition_family"] == "epistemic"
    assert run["status"] == "paused"
    assert run["points"][1]["physical_resource_charge"] is False
    assert run["accounted_total_tokens"] == 220
    assert run["interventions_applied"] == 1
    assert run["postfork_novel_delta_proposals"] == 1
    assert campaign["semantic_summary"]["physical_proposal_calls"] == 1
    assert campaign["condition_catalog"][0]["id"] == "assumption_challenge"


def test_page_contains_live_controls_and_raw_outcome_overlay() -> None:
    assert "Refresh now" in PAGE
    assert "Auto-refresh" in PAGE
    assert "Experiment-only 1% burn time" in PAGE
    assert "function fmtDurationMinutes(value)" in PAGE
    assert "function quotaTokenHtml(quota)" in PAGE
    assert "function drawQuota(quota)" in PAGE
    assert "function drawResetProjection(quota)" in PAGE
    assert "Predicted 0%:" in PAGE
    assert 'id="quota-reset-date"' in PAGE
    assert "Experiments-only projection at reset" in PAGE
    assert "All-Codex projection at reset" in PAGE
    assert "seriesMode:'runs'" in PAGE
    assert "Experiment-only session-attributed pace" in PAGE
    assert "Model and settings ledger" in PAGE
    assert "Minutes per 1% of primary limit used" in PAGE
    assert 'id="campaign-select"' in PAGE
    assert "function syncCampaignSelector(sections)" in PAGE
    assert "shownSections=selected?[selected]:[]" in PAGE
    assert "Raw outcome overlay" in PAGE
    assert "yScale:'data'" in PAGE
    assert "overlay:false" in PAGE
    assert "Y-axis range" in PAGE
    assert "Fit visible data" in PAGE
    assert "Individual runs" in PAGE
    assert "Condition median" in PAGE
    assert "Condition mean" in PAGE
    assert "function legendActions(state)" in PAGE
    assert "data-legend-action=\"show-all\"" in PAGE
    assert "data-condition-toggle" in PAGE
    assert "data-aggregate-condition" not in PAGE
    assert "data-family" not in PAGE
    assert "aggregateDataset" in PAGE
    assert "function mean(values)" in PAGE
    assert "aggregate_method:aggregateMethod" in PAGE
    assert "${method} of ${p.member_count}/${p.condition_run_count} runs" in PAGE
    assert "Export visible CSV" in PAGE
    assert "Proposal start" in PAGE
    assert "Proposal end" in PAGE
    assert "function baselineObjective(run,state)" in PAGE
    assert "improvement from proposal ${proposalBounds(state).start}" in PAGE
    assert "if(proposal<start||(end!==null&&proposal>end))return false" in PAGE
    assert "beginAtZero:yScale==='zero'" in PAGE
    assert "if(!point.valid)return 'crossRot';return 'circle'" in PAGE
    assert "return 'triangle'" not in PAGE
    assert "Marker key: ● seed or retained" in PAGE
    assert "location.protocol==='file:'" in PAGE
    assert "fetch(apiUrl" in PAGE
    assert "IntersectionObserver" in PAGE
    assert "request timed out; try again" in PAGE
    assert "Semantic-intervention evidence" in PAGE
    assert "Greedy OpenEvolve v2.1" in PAGE
    assert "Intervention families" not in PAGE
    assert "const semanticConditionColors=" in PAGE
    assert "function semanticRunLegend(payload,state,catalog)" in PAGE
    assert "function orderedRuns(payload,runs=payload.runs)" in PAGE
    assert "borderDash:role==='raw'?[]:replicateDash(run)" in PAGE
    assert "run-group-runs" in PAGE
    assert "physical_resource_charge" in PAGE
    assert "https://unpkg.com/chart.js@4.4.4" in PAGE
    assert "autoresearch_v17_fashion_mnist" in PAGE
    assert "openevolve_v21_fashion_mnist" in PAGE
    assert "Campaign Modal GPU estimate" in PAGE
    assert "Modal GPU estimate" in PAGE
    assert "incremental_modal_gpu_cost" in PAGE
    assert "/api/revision" in PAGE
    assert "function checkHotReload()" in PAGE
    assert "setInterval(checkHotReload,2000)" in PAGE


def test_dashboard_labels_native_and_legacy_openevolve_architectures(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "native"
    (campaign / "runs").mkdir(parents=True)
    _write_json(campaign / "campaign.json", {"schema_version": "4.0"})
    _write_json(
        campaign / "inputs/task.json",
        {
            "task_id": "toy",
            "display_name": "Toy task",
            "objective_metric": "score",
            "objective_direction": "maximize",
        },
    )
    _write_json(
        campaign / "inputs/framework.json",
        {"framework_id": "native_openevolve"},
    )

    payload = campaign_data(campaign, PRICES)

    assert payload["framework_id"] == "native_openevolve"
    assert payload["framework_label"] == "Native OpenEvolve"
    assert payload["task_display_name"] == "Toy task"


def test_dashboard_revision_changes_with_source_content(tmp_path: Path) -> None:
    source = tmp_path / "dashboard-source"
    source.write_text("first", encoding="utf-8")
    first = dashboard_revision((source,))

    source.write_text("second", encoding="utf-8")

    assert dashboard_revision((source,)) != first


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

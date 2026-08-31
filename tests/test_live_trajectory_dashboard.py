from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path

import pytest

from experiments.live_trajectory_dashboard import (
    CONTROLLER_PAGE,
    DEFAULT_MODAL_H100_PRICE_PER_SECOND,
    PAGE,
    CapacityController,
    DashboardPayloadCache,
    _macos_cpu_percentages,
    _macos_size_bytes,
    _macos_vm_stat_bytes,
    build_run,
    campaign_capacity_payload,
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
    set_task_evaluator_capacity,
    task_evaluator_capacity_payload,
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


def test_dashboard_data_keeps_legacy_openevolve_refresh_key(tmp_path: Path) -> None:
    openevolve = tmp_path / "openevolve"
    openevolve.mkdir()
    data = dashboard_data(
        {
            "openevolve_v2": openevolve,
            "autoresearch_v17": tmp_path / "not-started",
        },
        PRICES,
    )

    assert data["schema_version"] == "3.1"
    assert "autoresearch" not in data
    assert "autoresearch_v16" not in data["campaigns"]
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
    assert (
        projected_zero_crossing_time(
            100,
            2,
            from_time=observed_at,
        )
        == observed_at
    )


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


def test_build_run_reports_live_evaluator_stage_and_wait_reason(tmp_path: Path) -> None:
    run_dir = _example_run(tmp_path)
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    state["active"] = {"index": 3, "started_at": "2026-01-01T00:20:00+00:00"}
    _write_json(run_dir / "state.json", state)
    opportunity = run_dir / "opportunities/0003"
    (opportunity / "codex").mkdir(parents=True)
    (opportunity / "codex/proposal-3.jsonl").write_text("{}\n", encoding="utf-8")
    (opportunity / "evaluation-workspace").mkdir()

    waiting = build_run(
        run_dir,
        PRICES,
        objective_metric="parameters",
        objective_direction="minimize",
        runtime_activity={
            "agent_holders": [],
            "evaluator_holders": [],
            "shared_evaluator_occupied": 12,
            "shared_evaluator_capacity": 12,
        },
    )

    assert waiting is not None
    assert waiting["operational_stage"]["kind"] == "waiting_evaluator"
    assert waiting["operational_stage_label"] == ("Waiting · host evaluator slot · P3")

    (opportunity / "evaluation.stdout.log").write_text("training", encoding="utf-8")
    evaluating = build_run(
        run_dir,
        PRICES,
        objective_metric="parameters",
        objective_direction="minimize",
        runtime_activity={
            "agent_holders": [],
            "evaluator_holders": [{"opportunity_root": str(opportunity)}],
        },
    )

    assert evaluating is not None
    assert evaluating["operational_stage"]["kind"] == "evaluating"
    assert evaluating["operational_stage_label"] == "Evaluating · local evaluator · P3"


def test_build_run_reports_campaign_active_opportunity_limit(
    tmp_path: Path,
) -> None:
    run_dir = _example_run(tmp_path)
    campaign = run_dir.parent.parent.resolve()

    waiting = build_run(
        run_dir,
        PRICES,
        objective_metric="parameters",
        objective_direction="minimize",
        runtime_activity={
            "agent_holders": [],
            "campaign_controllers": {str(campaign)},
        },
        campaign_desired="running",
        campaign_subject_limit=12,
        campaign_active_opportunities=12,
        semantic_campaign=True,
    )

    assert waiting is not None
    assert waiting["operational_stage"]["kind"] == ("waiting_campaign_opportunity")
    assert waiting["operational_stage_label"] == (
        "Waiting · campaign active-opportunity limit"
    )
    assert "All 12 campaign opportunity slots" in waiting["operational_stage"]["detail"]


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
    assert "Experiment-only 1% burn time" not in PAGE
    assert 'href="/controller"' in PAGE
    assert "seriesMode:'runs'" in PAGE
    assert 'id="campaign-select"' in PAGE
    assert "semantic_v4_tiny_adderboard" in PAGE
    assert "Semantic interventions v4 · Tiny AdderBoard Greedy OpenEvolve" in PAGE
    assert "autoresearch_v16" not in PAGE
    assert "Autoresearch v1.6" not in PAGE
    assert "function syncCampaignSelector(sections)" in PAGE
    assert "shownSections=selected?[selected]:[]" in PAGE
    assert "Raw outcome overlay" in PAGE
    assert "Live stage" in PAGE
    assert "operational_stage_label" in PAGE
    assert "function summaryCellTitle(run,key)" in PAGE
    assert "yScale:'data'" in PAGE
    assert "overlay:false" in PAGE
    assert "Y-axis range" in PAGE
    assert "Fit visible data" in PAGE
    assert "x:'proposal'" in PAGE
    assert "const primaryXAxisKeys=['proposal','token_cost','active_hours']" in PAGE
    assert '<optgroup label="Primary x axes">' in PAGE
    assert '<optgroup label="Other metrics">' in PAGE
    assert "height:660px" in PAGE
    assert "state.xs.map" not in PAGE
    assert 'canvas id="${id}-chart"' in PAGE
    assert 'canvas id="${id}-chart-${index}"' not in PAGE
    assert "Individual runs" in PAGE
    assert "Condition median" in PAGE
    assert "Condition mean" in PAGE
    assert "function legendActions(state)" in PAGE
    assert 'data-legend-action="show-all"' in PAGE
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
    assert "const semanticCompactLabels=" in PAGE
    assert "function semanticRunLegend(payload,state,catalog)" in PAGE
    assert "members.map(run=>runLegendButton(state,run,true))" in PAGE
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


def test_controller_page_contains_capacity_compute_and_codex_panels() -> None:
    assert "Shared concurrency limits" in CONTROLLER_PAGE
    assert "Active Codex calls" in CONTROLLER_PAGE
    assert "No fixed maximum" in CONTROLLER_PAGE
    assert "input.removeAttribute('max')" in CONTROLLER_PAGE
    assert 'id="workers-limit"' in CONTROLLER_PAGE
    assert 'id="evaluators-limit"' in CONTROLLER_PAGE
    assert "Mac compute" in CONTROLLER_PAGE
    assert "CPU uses 1-min load per logical core" in CONTROLLER_PAGE
    assert "Experiment-only 1% burn time" in CONTROLLER_PAGE
    assert "function quotaTokenHtml(quota)" in CONTROLLER_PAGE
    assert "function drawQuota(quota)" in CONTROLLER_PAGE
    assert "function drawResetProjection(quota)" in CONTROLLER_PAGE
    assert "Experiment-only session-attributed pace" in CONTROLLER_PAGE
    assert "Model and settings ledger" in CONTROLLER_PAGE
    assert "Minutes per 1% of primary limit used" in CONTROLLER_PAGE
    assert 'id="quota-reset-date"' in CONTROLLER_PAGE
    assert "/api/controller/limits" in CONTROLLER_PAGE
    assert "Task evaluator limits" in CONTROLLER_PAGE
    assert "/api/controller/task-limits" in CONTROLLER_PAGE
    assert "function renderTaskLimits(payload)" in CONTROLLER_PAGE
    assert "minimum of the host, task, and campaign limits" in CONTROLLER_PAGE
    assert "Campaign-specific limits" in CONTROLLER_PAGE
    assert "/api/controller/campaign-limits" in CONTROLLER_PAGE
    assert "Active opportunity max" in CONTROLLER_PAGE
    assert "active end-to-end opportunit" in CONTROLLER_PAGE
    assert "No pause or restart occurred" in CONTROLLER_PAGE
    assert "runs:draft.runs" in CONTROLLER_PAGE
    assert "Campaign lifecycle" in CONTROLLER_PAGE
    assert "Safely pause" in CONTROLLER_PAGE
    assert "Future campaign directories are discovered automatically" in CONTROLLER_PAGE
    assert "/api/controller/campaign-lifecycle" in CONTROLLER_PAGE
    assert "function renderCampaignLifecycle(payload)" in CONTROLLER_PAGE
    assert "function applyCampaignLifecycle(id,action)" in CONTROLLER_PAGE


def test_task_evaluator_payload_groups_shared_local_task_pools(
    tmp_path: Path,
) -> None:
    first = tmp_path / "fashion-greedy"
    second = tmp_path / "fashion-native"
    for campaign in (first, second):
        _write_json(
            campaign / "inputs/task.json",
            {
                "task_id": "fashion_mnist_semantic_v4_mps",
                "display_name": "Fashion research",
                "preferred_backend": "local",
            },
        )
        _write_json(
            campaign / "inputs/v3-runtime.json",
            {
                "schema_version": "3.0",
                "evaluation": {"task_pool_capacity": 6},
            },
        )
    opportunity = first / "runs/example/opportunities/0001"
    payload = task_evaluator_capacity_payload(
        {"greedy": first, "native": second},
        {
            "campaigns": {
                "greedy": {"framework_label": "Greedy OpenEvolve"},
                "native": {"framework_label": "Native OpenEvolve"},
            }
        },
        {
            "pools": {
                "evaluators": {
                    "desired_limit": 20,
                    "active_holders": [
                        {"holder": {"opportunity_root": str(opportunity)}}
                    ],
                }
            }
        },
    )

    assert payload["hierarchy"] == "effective=min(host_task_campaign)"
    assert len(payload["tasks"]) == 1
    task = payload["tasks"][0]
    assert task["id"] == "fashion_mnist_semantic_v4_mps"
    assert task["label"] == "Fashion-MNIST"
    assert task["capacity"] == 6
    assert task["hard_capacity"] == 20
    assert task["host_evaluator_capacity"] == 20
    assert task["active"] == 1
    assert task["available"] == 5
    assert task["campaign_count"] == 2
    assert task["consistent"] is True


def test_campaign_evaluator_editable_max_follows_host_not_task(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "fashion"
    _write_json(campaign / "campaign.json", {"schema_version": "4.0"})
    (campaign / "inputs").mkdir(parents=True)
    (campaign / "inputs/semantic-interventions.toml").write_text(
        "max_parallel_agent_calls = 12\ntask_evaluator_capacity = 6\n",
        encoding="utf-8",
    )
    _write_json(
        campaign / "capacity-control.json",
        {"subject_workers": 12, "local_evaluators": 10},
    )

    payload = campaign_capacity_payload(
        {"fashion": campaign},
        {"campaigns": {"fashion": {"framework_label": "Greedy OpenEvolve"}}},
        {
            "pools": {
                "workers": {"active_holders": []},
                "evaluators": {
                    "desired_limit": 20,
                    "active_holders": [],
                },
            }
        },
    )

    row = payload["campaigns"][0]
    assert row["max_local_evaluators"] == 20
    assert row["task_evaluator_capacity"] == 6
    assert row["local_evaluators"] == 10
    assert row["effective_local_evaluator_limit"] == 6


def test_set_task_evaluator_capacity_updates_nonsemantic_v3_campaign(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "addition"
    task_id = "adderboard-training-ladder-v4-mps"
    _write_json(campaign / "campaign.json", {"schema_version": "3.0"})
    _write_json(
        campaign / "inputs/task.json",
        {
            "task_id": task_id,
            "display_name": "Addition",
            "preferred_backend": "local",
        },
    )
    _write_json(
        campaign / "inputs/v3-runtime.json",
        {
            "schema_version": "3.0",
            "evaluation": {"task_pool_capacity": 3},
        },
    )

    result = set_task_evaluator_capacity(
        {"addition": campaign},
        task_id=task_id,
        capacity=5,
        host_capacity=20,
    )

    runtime = json.loads((campaign / "inputs/v3-runtime.json").read_text())
    control = json.loads((campaign / "capacity-control.json").read_text())
    history = [
        json.loads(line)
        for line in (campaign / "v3-runtime-history.jsonl").read_text().splitlines()
    ]
    assert result["capacity"] == 5
    assert runtime["evaluation"]["task_pool_capacity"] == 5
    assert control["local_evaluators"] == 5
    assert control["subject_workers"] == 30
    assert history[-1]["after"]["evaluation"]["task_pool_capacity"] == 5


def test_task_evaluator_capacity_cannot_exceed_live_host_limit(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "addition"
    task_id = "adderboard-training-ladder-v4-mps"
    _write_json(campaign / "campaign.json", {"schema_version": "3.0"})
    _write_json(
        campaign / "inputs/task.json",
        {
            "task_id": task_id,
            "display_name": "Addition",
            "preferred_backend": "local",
        },
    )
    _write_json(
        campaign / "inputs/v3-runtime.json",
        {
            "schema_version": "3.0",
            "evaluation": {"task_pool_capacity": 3},
        },
    )

    with pytest.raises(ValueError, match="host ceiling \\(20\\)"):
        set_task_evaluator_capacity(
            {"addition": campaign},
            task_id=task_id,
            capacity=21,
            host_capacity=20,
        )


def test_task_evaluator_capacity_can_expand_semantic_pool_to_host_limit(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "tiny"
    task_id = "tiny-adderboard-training-ladder-v4-mps"
    _write_json(campaign / "campaign.json", {"schema_version": "4.0"})
    _write_json(
        campaign / "inputs/task.json",
        {
            "task_id": task_id,
            "display_name": "Tiny AdderBoard",
            "preferred_backend": "local",
        },
    )
    _write_json(
        campaign / "inputs/v3-runtime.json",
        {
            "schema_version": "3.0",
            "evaluation": {"task_pool_capacity": 16},
        },
    )
    (campaign / "inputs/semantic-interventions.toml").write_text(
        "task_evaluator_capacity = 16\nmax_parallel_agent_calls = 30\n",
        encoding="utf-8",
    )
    _write_json(
        campaign / "semantic-interventions.json",
        {"schema_version": "4.0", "plan": {"task_evaluator_capacity": 16}},
    )

    result = set_task_evaluator_capacity(
        {"tiny": campaign},
        task_id=task_id,
        capacity=20,
        host_capacity=20,
    )

    runtime = json.loads((campaign / "inputs/v3-runtime.json").read_text())
    manifest = json.loads((campaign / "semantic-interventions.json").read_text())
    plan = (campaign / "inputs/semantic-interventions.toml").read_text()
    assert result["capacity"] == 20
    assert runtime["evaluation"]["task_pool_capacity"] == 20
    assert manifest["plan"]["task_evaluator_capacity"] == 20
    assert "task_evaluator_capacity = 20" in plan


def test_capacity_controller_reserves_slots_and_persists_limits(tmp_path: Path) -> None:
    controller = CapacityController(
        state_path=tmp_path / "controller.json",
        worker_root=tmp_path / "workers",
        evaluator_root=tmp_path / "evaluators",
        worker_capacity=4,
        evaluator_capacity=3,
    )
    try:
        initial = controller.status()
        assert initial["pools"]["workers"]["label"] == "Active Codex calls"
        assert initial["pools"]["workers"]["fixed_maximum"] is None
        assert initial["pools"]["workers"]["effective_limit"] == 4
        assert initial["pools"]["evaluators"]["effective_limit"] == 3

        changed = controller.set_limits(workers=2, evaluators=1)

        assert changed["pools"]["workers"]["desired_limit"] == 2
        assert changed["pools"]["workers"]["reserved"] == 2
        assert changed["pools"]["workers"]["effective_limit"] == 2
        assert changed["pools"]["evaluators"]["reserved"] == 2
        assert changed["pools"]["evaluators"]["effective_limit"] == 1
        stored = json.loads((tmp_path / "controller.json").read_text())
        assert stored["limits"] == {"workers": 2, "evaluators": 1}

        expanded = controller.set_limits(workers=7, evaluators=5)

        assert expanded["pools"]["workers"]["desired_limit"] == 7
        assert expanded["pools"]["workers"]["slot_capacity"] == 7
        assert expanded["pools"]["workers"]["effective_limit"] == 7
        assert expanded["pools"]["evaluators"]["desired_limit"] == 5
        assert expanded["pools"]["evaluators"]["slot_capacity"] == 5
        assert (
            json.loads((tmp_path / "workers/scheduler.json").read_text())["capacity"]
            == 4
        )
        assert (
            json.loads((tmp_path / "evaluators/scheduler.json").read_text())["capacity"]
            == 3
        )
        assert (
            json.loads((tmp_path / "workers/operator-capacity.json").read_text())[
                "capacity"
            ]
            == 7
        )
        assert (
            json.loads((tmp_path / "evaluators/operator-capacity.json").read_text())[
                "capacity"
            ]
            == 5
        )
    finally:
        controller.close()


def test_macos_size_parser_handles_dashboard_telemetry_units() -> None:
    assert _macos_size_bytes("243M") == 243 * 1024**2
    assert _macos_size_bytes("35G") == 35 * 1024**3
    assert _macos_size_bytes("1.5 GB") == int(1.5 * 1024**3)
    assert _macos_size_bytes("unknown") is None


def test_macos_vm_stat_parser_recovers_memory_without_top() -> None:
    metrics = _macos_vm_stat_bytes(
        """Mach Virtual Memory Statistics: (page size of 16384 bytes)
The system has 38654705664 (2359296 pages with a page size of 16384).
Pages free:                                   102535.
Pages speculative:                             62835.
Pages wired down:                             585222.
Pages occupied by compressor:                 409604.
"""
    )

    assert metrics["reported_total_bytes"] == 38654705664
    assert metrics["free_bytes"] == 102535 * 16384
    assert metrics["speculative_bytes"] == 62835 * 16384
    assert metrics["wired_bytes"] == 585222 * 16384
    assert metrics["compressed_bytes"] == 409604 * 16384


def test_macos_cpu_percentages_uses_tick_deltas() -> None:
    assert _macos_cpu_percentages((100, 200, 700), (125, 225, 750)) == pytest.approx(
        (25.0, 25.0, 50.0)
    )
    assert _macos_cpu_percentages(None, (125, 225, 750)) is None


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


def test_dashboard_payload_cache_builds_once_and_reuses_gzip() -> None:
    calls = 0

    def build() -> bytes:
        nonlocal calls
        calls += 1
        return b"process-evidence" * 1000

    cache = DashboardPayloadCache(build, ttl_seconds=60)

    first, first_encoding = cache.response("gzip")
    second, second_encoding = cache.response("gzip")

    assert calls == 1
    assert first_encoding == second_encoding == "gzip"
    assert first == second
    assert gzip.decompress(first) == b"process-evidence" * 1000


def test_dashboard_payload_cache_latest_never_blocks_on_cold_build() -> None:
    cache = DashboardPayloadCache(lambda: b"snapshot")

    assert cache.latest() is None
    assert cache.response("")[0] == b"snapshot"
    assert cache.latest() == b"snapshot"

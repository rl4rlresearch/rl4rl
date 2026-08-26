#!/usr/bin/env python3
# ruff: noqa: E501
"""Read-only dashboard for C0-C3 and multi-arm research trajectories.

The server reads campaign logs only when a browser requests ``/api/data``.  It
does not import the experiment controller, acquire a campaign lock, or write to
any campaign artifact, so it is safe to use while controllers are running.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter
from contextlib import suppress
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTORESEARCH = Path(
    "/private/tmp/rl4rl-v16-codex1644-confined-campaign-fresh-20260822c"
)
DEFAULT_OPENEVOLVE = (
    REPO_ROOT / "data/c0c3/controlled-openevolve-transformer-v2-mps-campaign"
)
DEFAULT_AUTORESEARCH_V17 = (
    REPO_ROOT / "data/c0c3/transformer-optimization-v1-7-source-only-campaign"
)
DEFAULT_OPENEVOLVE_V21 = (
    REPO_ROOT / "data/c0c3/controlled-openevolve-transformer-v2-1-mps-campaign"
)
DEFAULT_AUTORESEARCH_V17_NANOGPT = (
    REPO_ROOT / "data/c0c3/nanogpt-autoresearch-v1-7-h100-campaign"
)
DEFAULT_OPENEVOLVE_V21_NANOGPT = (
    REPO_ROOT / "data/c0c3/nanogpt-openevolve-v2-1-h100-campaign"
)
DEFAULT_AUTORESEARCH_V17_FASHION_MNIST = (
    REPO_ROOT / "data/c0c3/fashion-mnist-autoresearch-v1-7-mps-campaign"
)
DEFAULT_OPENEVOLVE_V21_FASHION_MNIST = (
    REPO_ROOT / "data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign"
)
DEFAULT_SEMANTIC_V4_FASHION_MNIST = (
    REPO_ROOT / "data/c0c3/semantic-interventions-v4-fashion-openevolve-campaign"
)

# These are display-only price weights, not a billing record.  They match the
# previous trajectory visualizer's token-cost convention and can be overridden.
DEFAULT_PRICE_PER_MILLION = {"input": 1.75, "cached_input": 0.175, "output": 14.0}
# Campaign ledgers record evaluator worker seconds, rather than Modal invoice
# line items. The H100 rate is the public per-second Modal GPU price and is
# configurable at launch for a historical or account-specific rate.
DEFAULT_MODAL_H100_PRICE_PER_SECOND = 0.001097


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return default


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    result: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def numeric(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def seed_candidate(state: dict[str, Any]) -> dict[str, Any] | None:
    candidates = state.get("candidates", {})
    if not isinstance(candidates, dict):
        return None
    for candidate in candidates.values():
        if not isinstance(candidate, dict) or candidate.get("created_opportunity") != 0:
            continue
        return candidate
    return None


def metric_at_seed(state: dict[str, Any], objective_metric: str) -> float | None:
    candidate = seed_candidate(state)
    metrics = candidate.get("metrics", {}) if isinstance(candidate, dict) else {}
    return numeric(metrics.get(objective_metric)) if isinstance(metrics, dict) else None


def numeric_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): number
        for key, raw in value.items()
        if (number := numeric(raw)) is not None
    }


def usage_value(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    return (
        int(value)
        if isinstance(value, int | float) and not isinstance(value, bool)
        else 0
    )


def normalized_usage(value: Any) -> dict[str, int]:
    usage = value if isinstance(value, dict) else {}
    result = {
        key: usage_value(usage, key)
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        )
    }
    result["total_tokens"] = usage_value(usage, "total_tokens") or (
        result["input_tokens"] + result["output_tokens"]
    )
    return result


def weighted_cost(usage: dict[str, Any], prices: dict[str, float]) -> float:
    input_tokens = float(usage.get("input_tokens", 0) or 0)
    cached_tokens = float(usage.get("cached_input_tokens", 0) or 0)
    output_tokens = float(usage.get("output_tokens", 0) or 0)
    uncached_tokens = max(0.0, input_tokens - cached_tokens)
    return (
        uncached_tokens * prices["input"]
        + cached_tokens * prices["cached_input"]
        + output_tokens * prices["output"]
    ) / 1_000_000


def modal_usage_index(
    campaign: Path, *, h100_price_per_second: float
) -> tuple[dict[str, dict[int, list[dict[str, Any]]]], dict[str, Any]]:
    """Index the append-only Modal ledger by run and opportunity.

    This is intentionally a campaign-attributed GPU estimate. It accounts for
    recorded worker seconds only; Modal's invoice can additionally contain
    account-level credits, storage, and other resource charges.
    """

    ledger_path = campaign / "modal-usage.jsonl"
    records = iter_jsonl(ledger_path)
    by_run: dict[str, dict[int, list[dict[str, Any]]]] = {}
    worker_seconds = 0.0
    completed_calls = 0
    failed_calls = 0
    unpriced_records = 0
    for record in records:
        run_id = record.get("run_id")
        opportunity = record.get("opportunity")
        seconds = numeric(record.get("worker_seconds"))
        if record.get("status") == "completed":
            completed_calls += 1
        elif record.get("status") == "failed":
            failed_calls += 1
        if seconds is None:
            unpriced_records += 1
            continue
        worker_seconds += seconds
        if isinstance(run_id, str) and isinstance(opportunity, int):
            by_run.setdefault(run_id, {}).setdefault(opportunity, []).append(record)
    return by_run, {
        "available": ledger_path.is_file(),
        "worker_seconds": round(worker_seconds, 6),
        "gpu_cost": round(worker_seconds * h100_price_per_second, 6),
        "completed_calls": completed_calls,
        "failed_calls": failed_calls,
        "unpriced_records": unpriced_records,
        "h100_price_per_second": h100_price_per_second,
        "scope": "recorded Modal evaluator GPU worker seconds only",
    }


def compact_label(run_id: str) -> str:
    parts = run_id.lower().split("-")
    block = next(
        (
            part.upper()
            for part in parts
            if len(part) == 3 and part.startswith("b") and part[1:].isdigit()
        ),
        "B??",
    )
    condition = next(
        (part.upper() for part in parts if part in {"c0", "c1", "c2", "c3"}), "C?"
    )
    return f"{block}-{condition}"


def semantic_label(assignment: dict[str, Any], fallback: str) -> str:
    replicate = assignment.get("replicate")
    condition = assignment.get("condition")
    if isinstance(replicate, int) and isinstance(condition, str):
        return f"R{replicate:02d} · {condition.replace('_', ' ')}"
    return fallback


def display_status(run_dir: Path, scientific_status: Any) -> str:
    """Return the operator-facing lifecycle status without changing run state.

    A cooperatively paused trajectory deliberately keeps ``state.json`` marked
    ``running`` so the controller can resume it.  The lifecycle log is the
    authoritative record of whether that runnable trajectory is currently
    paused or in the process of pausing.
    """
    status = scientific_status if isinstance(scientific_status, str) else "unknown"
    if status != "running":
        return status
    for event in reversed(iter_jsonl(run_dir / "lifecycle.jsonl")):
        event_name = event.get("event")
        if event_name == "trajectory_paused":
            return "paused"
        if event_name == "trajectory_pause_requested":
            return "pausing"
        if event_name in {"trajectory_resumed", "trajectory_started"}:
            break
    return status


def build_run(
    run_dir: Path,
    prices: dict[str, float],
    *,
    objective_metric: str,
    objective_direction: str,
    modal_usage_by_opportunity: dict[int, list[dict[str, Any]]] | None = None,
    modal_h100_price_per_second: float = DEFAULT_MODAL_H100_PRICE_PER_SECOND,
    semantic_prefix_role: str | None = None,
    shared_prefix_through: int = 0,
    desired_state: str | None = None,
) -> dict[str, Any] | None:
    state = read_json(run_dir / "state.json", {})
    if not isinstance(state, dict):
        return None
    run_id = state.get("run_id")
    manifest = read_json(run_dir / "manifest.json", {})
    manifest = manifest if isinstance(manifest, dict) else {}
    assignment = manifest.get("assignment", {})
    assignment = assignment if isinstance(assignment, dict) else {}
    condition = assignment.get("condition", state.get("condition"))
    if not isinstance(run_id, str) or not isinstance(condition, str):
        return None
    events = iter_jsonl(run_dir / "events.jsonl")
    assessments = {
        int(event["opportunity"]): event
        for event in events
        if event.get("event") == "developmental_assessment"
        and isinstance(event.get("opportunity"), int)
    }
    interventions = {
        int(event["opportunity"]): event
        for event in events
        if event.get("event") == "semantic_intervention_applied"
        and isinstance(event.get("opportunity"), int)
    }
    started: dict[int, datetime] = {}
    elapsed_seconds = 0.0
    seed_objective = metric_at_seed(state, objective_metric)
    best_objective = seed_objective
    points: list[dict[str, Any]] = []
    latest_event_at: str | None = None
    valid_proposals = 0
    retained_proposals = 0
    latest_raw_objective: float | None = None
    modal_usage_by_opportunity = modal_usage_by_opportunity or {}
    recorded_modal_worker_seconds = sum(
        numeric(record.get("worker_seconds")) or 0.0
        for records in modal_usage_by_opportunity.values()
        for record in records
    )
    recorded_modal_gpu_cost = (
        recorded_modal_worker_seconds * modal_h100_price_per_second
    )
    modal_worker_seconds = 0.0
    modal_gpu_cost = 0.0
    accounted_tokens = 0
    accounted_cost = 0.0
    for event in events:
        opportunity = event.get("opportunity")
        if not isinstance(opportunity, int):
            continue
        timestamp = parse_timestamp(event.get("timestamp"))
        if event.get("event") == "proposal_started" and timestamp is not None:
            started[opportunity] = timestamp
            continue
        if event.get("event") != "proposal_completed":
            continue
        if isinstance(event.get("timestamp"), str):
            latest_event_at = event["timestamp"]
        if timestamp is not None and opportunity in started:
            elapsed_seconds += max(
                0.0, (timestamp - started[opportunity]).total_seconds()
            )
        evaluation = event.get("evaluation", {})
        raw_metrics = (
            evaluation.get("metrics", {}) if isinstance(evaluation, dict) else {}
        )
        raw_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        metrics = numeric_metrics(raw_metrics)
        objective = numeric(metrics.get(objective_metric))
        valid = bool(isinstance(evaluation, dict) and evaluation.get("valid"))
        retained = bool(event.get("retained"))
        if valid:
            valid_proposals += 1
        if retained:
            retained_proposals += 1
        if objective is not None:
            latest_raw_objective = objective
        if valid and objective is not None:
            value = objective
            if best_objective is None:
                best_objective = value
            elif objective_direction == "maximize":
                best_objective = max(best_objective, value)
            else:
                best_objective = min(best_objective, value)
        usage = normalized_usage(event.get("usage_cumulative"))
        usage_increment = normalized_usage(event.get("usage_increment"))
        physical_resource_charge = not (
            semantic_prefix_role == "shadow" and opportunity <= shared_prefix_through
        )
        if physical_resource_charge:
            accounted_tokens += usage_increment["total_tokens"]
            accounted_cost += weighted_cost(usage_increment, prices)
        evaluator_seconds = numeric(event.get("evaluator_seconds_cumulative")) or 0.0
        evaluator_seconds_increment = (
            numeric(event.get("evaluator_seconds_increment")) or 0.0
        )
        evaluator_calls = numeric(event.get("evaluator_calls_cumulative")) or 0.0
        evaluator_calls_increment = (
            numeric(event.get("evaluator_calls_increment")) or 0.0
        )
        modal_records = modal_usage_by_opportunity.get(opportunity, [])
        modal_worker_seconds_increment = sum(
            numeric(record.get("worker_seconds")) or 0.0 for record in modal_records
        )
        modal_gpu_cost_increment = (
            modal_worker_seconds_increment * modal_h100_price_per_second
        )
        modal_worker_seconds += modal_worker_seconds_increment
        modal_gpu_cost += modal_gpu_cost_increment
        improvement: float | None = None
        improvement_percent: float | None = None
        if seed_objective is not None and best_objective is not None:
            improvement = (
                best_objective - seed_objective
                if objective_direction == "maximize"
                else seed_objective - best_objective
            )
            if seed_objective != 0:
                improvement_percent = improvement / abs(seed_objective) * 100
        points.append(
            {
                "proposal": opportunity,
                "active_hours": round(elapsed_seconds / 3600, 6),
                "active_seconds": round(elapsed_seconds, 6),
                "token_cost": round(weighted_cost(usage, prices), 6),
                "incremental_token_cost": round(
                    weighted_cost(usage_increment, prices), 6
                ),
                "accounted_total_tokens": accounted_tokens,
                "accounted_token_cost": round(accounted_cost, 6),
                "modal_worker_seconds": round(modal_worker_seconds, 6),
                "incremental_modal_worker_seconds": round(
                    modal_worker_seconds_increment, 6
                ),
                "modal_gpu_cost": round(modal_gpu_cost, 6),
                "incremental_modal_gpu_cost": round(modal_gpu_cost_increment, 6),
                **usage,
                **{
                    f"incremental_{key}": value
                    for key, value in usage_increment.items()
                },
                "evaluator_seconds": evaluator_seconds,
                "incremental_evaluator_seconds": evaluator_seconds_increment,
                "evaluator_calls": evaluator_calls,
                "incremental_evaluator_calls": evaluator_calls_increment,
                "best_objective": best_objective,
                "raw_objective": objective,
                "fitness": numeric(evaluation.get("fitness"))
                if isinstance(evaluation, dict)
                else None,
                "objective_improvement": improvement,
                "objective_improvement_percent": improvement_percent,
                "metrics": metrics,
                "valid": valid,
                "retained": retained,
                "retention_decision": event.get("retention_decision"),
                "failure_kind": evaluation.get("failure_kind")
                if isinstance(evaluation, dict)
                else None,
                "proposal_type": event.get("proposal_type"),
                "shared_prefix": bool(
                    event.get("shared_prefix")
                    or (
                        semantic_prefix_role == "leader"
                        and opportunity <= shared_prefix_through
                    )
                ),
                "physical_resource_charge": physical_resource_charge,
                "semantic_intervention_applied": opportunity in interventions,
                "semantic_intervention_id": (
                    interventions.get(opportunity, {}).get("intervention_id")
                ),
                "developmental_status": assessments.get(opportunity, {}).get("status"),
                "developmental_credit": numeric(
                    assessments.get(opportunity, {}).get("credit")
                ),
                "developmental_reasons": assessments.get(opportunity, {}).get(
                    "reasons", []
                ),
                "developmental_selection_effect": assessments.get(opportunity, {}).get(
                    "selection_effect"
                ),
                "fidelity_highest_level": numeric(
                    raw_metrics.get("fidelity_highest_level")
                ),
                "fidelity_reached_full": bool(
                    raw_metrics.get("fidelity_reached_full", False)
                ),
                "fidelity_stage_count": (
                    len(raw_metrics.get("fidelity_stages", []))
                    if isinstance(raw_metrics.get("fidelity_stages"), list)
                    else 0
                ),
                "hypothesis": event.get("hypothesis"),
                "mechanism": event.get("mechanism"),
                "timestamp": event.get("timestamp"),
                "is_seed": False,
            }
        )
    if seed_objective is not None:
        candidate = seed_candidate(state) or {}
        seed_metrics = numeric_metrics(candidate.get("metrics"))
        points.insert(
            0,
            {
                "proposal": 0,
                "active_hours": 0.0,
                "active_seconds": 0.0,
                "token_cost": 0.0,
                "incremental_token_cost": 0.0,
                "accounted_total_tokens": 0,
                "accounted_token_cost": 0.0,
                "modal_worker_seconds": 0.0,
                "incremental_modal_worker_seconds": 0.0,
                "modal_gpu_cost": 0.0,
                "incremental_modal_gpu_cost": 0.0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_output_tokens": 0,
                "total_tokens": 0,
                "incremental_input_tokens": 0,
                "incremental_cached_input_tokens": 0,
                "incremental_output_tokens": 0,
                "incremental_reasoning_output_tokens": 0,
                "incremental_total_tokens": 0,
                "evaluator_seconds": 0.0,
                "incremental_evaluator_seconds": 0.0,
                "evaluator_calls": 0.0,
                "incremental_evaluator_calls": 0.0,
                "best_objective": seed_objective,
                "raw_objective": seed_objective,
                "fitness": numeric(candidate.get("fitness")),
                "objective_improvement": 0.0,
                "objective_improvement_percent": 0.0,
                "metrics": seed_metrics,
                "valid": True,
                "retained": True,
                "retention_decision": "frozen_seed",
                "failure_kind": None,
                "proposal_type": "seed",
                "hypothesis": candidate.get("hypothesis"),
                "mechanism": None,
                "timestamp": None,
                "is_seed": True,
            },
        )
    usage = normalized_usage(state.get("usage"))
    scientific_status = state.get("status", "unknown")
    status = display_status(run_dir, scientific_status)
    if scientific_status == "running" and desired_state in {"paused", "stopped"}:
        status = str(desired_state)
    charged_points = [
        point for point in points if point.get("physical_resource_charge", True)
    ]
    uncharged_prefix_cost = sum(
        float(point.get("incremental_token_cost", 0.0) or 0.0)
        for point in points
        if not point.get("physical_resource_charge", True)
    )
    uncharged_prefix_tokens = sum(
        int(point.get("incremental_total_tokens", 0) or 0)
        for point in points
        if not point.get("physical_resource_charge", True)
    )
    postfork_points = [
        point
        for point in points
        if not point.get("is_seed")
        and int(point.get("proposal", 0)) > shared_prefix_through
    ]
    intervention_points = [
        point for point in postfork_points if point.get("semantic_intervention_applied")
    ]
    phase_successes = sum(
        any(
            later.get("retained")
            for later in postfork_points
            if int(point["proposal"])
            <= int(later["proposal"])
            < int(point["proposal"]) + 5
        )
        for point in intervention_points
    )
    return {
        "run_id": run_id,
        "label": semantic_label(assignment, compact_label(run_id)),
        "condition": condition.upper()
        if condition.lower() in {"c0", "c1", "c2", "c3"}
        else condition,
        "condition_label": assignment.get("condition_label", condition),
        "condition_family": assignment.get("condition_family", "factorial"),
        "components": assignment.get("components", []),
        "replicate": assignment.get("replicate", assignment.get("block")),
        "order": assignment.get("order"),
        "semantic_prefix_role": semantic_prefix_role,
        "shared_prefix_through": shared_prefix_through,
        "desired": desired_state,
        "status": status,
        "scientific_status": scientific_status,
        "proposals_used": state.get("proposals_used", 0),
        "total_tokens": usage["total_tokens"],
        "token_cost": round(weighted_cost(usage, prices), 6),
        "accounted_total_tokens": max(
            0, usage["total_tokens"] - uncharged_prefix_tokens
        ),
        "accounted_token_cost": round(
            max(0.0, weighted_cost(usage, prices) - uncharged_prefix_cost), 6
        ),
        # A worker can finish before the controller writes its completion event.
        # Keep the run summary faithful to the append-only Modal ledger even in
        # that short live-update window; proposal points stay aligned to their
        # corresponding completed opportunities.
        "modal_worker_seconds": round(recorded_modal_worker_seconds, 6),
        "modal_gpu_cost": round(recorded_modal_gpu_cost, 6),
        "best_objective": best_objective,
        "seed_objective": seed_objective,
        "latest_raw_objective": latest_raw_objective,
        "valid_proposals": valid_proposals,
        "invalid_proposals": max(
            0, len(points) - (1 if seed_objective is not None else 0) - valid_proposals
        ),
        "retained_proposals": retained_proposals,
        "developmental_counts": dict(
            Counter(
                str(point["developmental_status"])
                for point in charged_points
                if point.get("developmental_status")
            )
        ),
        "interventions_applied": sum(
            bool(point.get("semantic_intervention_applied")) for point in points
        ),
        "full_fidelity_proposals": sum(
            bool(point.get("fidelity_reached_full")) for point in charged_points
        ),
        "screened_out_proposals": sum(
            point.get("failure_kind") == "fidelity_screen_not_promoted"
            for point in charged_points
        ),
        "postfork_proposals": len(postfork_points),
        "postfork_valid_proposals": sum(
            bool(point.get("valid")) for point in postfork_points
        ),
        "postfork_retained_proposals": sum(
            bool(point.get("retained")) for point in postfork_points
        ),
        "postfork_novel_delta_proposals": sum(
            "novel_delta" in point.get("developmental_reasons", [])
            for point in postfork_points
        ),
        "postfork_token_cost": round(
            sum(
                float(point.get("incremental_token_cost", 0.0) or 0.0)
                for point in postfork_points
            ),
            6,
        ),
        "postfork_total_tokens": sum(
            int(point.get("incremental_total_tokens", 0) or 0)
            for point in postfork_points
        ),
        "postfork_evaluator_seconds": round(
            sum(
                float(point.get("incremental_evaluator_seconds", 0.0) or 0.0)
                for point in postfork_points
            ),
            6,
        ),
        "intervention_phase_successes": phase_successes,
        "intervention_phase_count": len(intervention_points),
        "intervention_mechanism_reported": sum(
            str(point.get("mechanism", "")).strip().casefold()
            not in {"", "[not recorded]", "[missing mechanism]"}
            for point in intervention_points
        ),
        "active_hours": round(elapsed_seconds / 3600, 6),
        "latest_event_at": latest_event_at,
        "lowest_parameters": (
            best_objective if objective_metric == "parameters" else None
        ),
        "points": points,
    }


def campaign_data(
    campaign: Path,
    prices: dict[str, float],
    *,
    modal_h100_price_per_second: float = DEFAULT_MODAL_H100_PRICE_PER_SECOND,
) -> dict[str, Any]:
    runs_root = campaign / "runs"
    campaign_manifest = read_json(campaign / "campaign.json", {})
    campaign_manifest = campaign_manifest if isinstance(campaign_manifest, dict) else {}
    semantic = campaign_manifest.get("schema_version") == "4.0"
    prefix_roles: dict[str, tuple[str, int]] = {}
    if semantic:
        prefix_manifest = read_json(campaign / "semantic-prefix.json", {})
        if isinstance(prefix_manifest, dict):
            for row in prefix_manifest.get("replicates", []):
                if not isinstance(row, dict):
                    continue
                through = int(row.get("shared_through_opportunity", 0))
                leader = row.get("leader_run_id")
                if isinstance(leader, str):
                    prefix_roles[leader] = ("leader", through)
                for shadow in row.get("shadow_run_ids", []):
                    if isinstance(shadow, str):
                        prefix_roles[shadow] = ("shadow", through)
    run_control = read_json(campaign / "semantic-run-control.json", {})
    desired_by_run = (
        run_control.get("runs", {})
        if semantic and isinstance(run_control, dict)
        else {}
    )
    task = read_json(campaign / "inputs/task.json", {})
    task = task if isinstance(task, dict) else {}
    framework = read_json(campaign / "inputs/framework.json", {})
    framework = framework if isinstance(framework, dict) else {}
    framework_id = str(framework.get("framework_id", "unknown"))
    framework_label = {
        "karpathy_autoresearch": "Autoresearch",
        "openevolve": "Greedy OpenEvolve",
        "native_openevolve": "Native OpenEvolve",
    }.get(framework_id, framework_id)
    objective_metric = str(task.get("objective_metric", "parameters"))
    objective_direction = str(task.get("objective_direction", "minimize"))
    modal_by_run, modal_summary = modal_usage_index(
        campaign, h100_price_per_second=modal_h100_price_per_second
    )
    runs = [
        build_run(
            path,
            prices,
            objective_metric=objective_metric,
            objective_direction=objective_direction,
            modal_usage_by_opportunity=modal_by_run.get(path.name),
            modal_h100_price_per_second=modal_h100_price_per_second,
            semantic_prefix_role=prefix_roles.get(path.name, (None, 0))[0],
            shared_prefix_through=prefix_roles.get(path.name, (None, 0))[1],
            desired_state=(
                desired_by_run.get(path.name, {}).get("desired")
                if isinstance(desired_by_run.get(path.name), dict)
                else None
            ),
        )
        for path in sorted(runs_root.glob("*"))
        if path.is_dir()
    ]
    visible_runs = [
        run
        for run in runs
        if run is not None
        and (semantic or run["condition"] in {"C0", "C1", "C2", "C3"})
    ]
    observed_metrics = sorted(
        {
            key
            for run in visible_runs
            for point in run["points"]
            for key in point.get("metrics", {})
        }
    )
    metric_labels = {
        "active_hours": "Active wall-clock time (hours)",
        "active_seconds": "Active wall-clock time (seconds)",
        "best_objective": f"Best valid {objective_metric}",
        "cached_input_tokens": "Cumulative cached input tokens",
        "evaluator_calls": "Cumulative evaluator calls",
        "evaluator_seconds": "Cumulative evaluator time (seconds)",
        "fitness": "Proposal fitness",
        "incremental_cached_input_tokens": "Cached input tokens this proposal",
        "incremental_evaluator_calls": "Evaluator calls this proposal",
        "incremental_evaluator_seconds": "Evaluator time this proposal (seconds)",
        "incremental_input_tokens": "Input tokens this proposal",
        "incremental_modal_gpu_cost": "Modal GPU cost estimate this proposal (USD)",
        "incremental_modal_worker_seconds": "Modal GPU worker seconds this proposal",
        "incremental_output_tokens": "Output tokens this proposal",
        "incremental_reasoning_output_tokens": "Reasoning output tokens this proposal",
        "incremental_token_cost": "Price-weighted token cost this proposal (USD)",
        "incremental_total_tokens": "Total tokens this proposal",
        "input_tokens": "Cumulative input tokens",
        "objective_improvement": (
            f"Best {objective_metric} improvement from selected proposal start"
        ),
        "objective_improvement_percent": (
            f"Best {objective_metric} improvement from selected proposal start (%)"
        ),
        "modal_gpu_cost": "Cumulative Modal GPU cost estimate (USD)",
        "modal_worker_seconds": "Cumulative Modal GPU worker seconds",
        "output_tokens": "Cumulative output tokens",
        "proposal": "Proposal index",
        "raw_objective": f"Proposal {objective_metric} (all outcomes)",
        "reasoning_output_tokens": "Cumulative reasoning output tokens",
        "token_cost": "Cumulative price-weighted token cost (USD)",
        "total_tokens": "Cumulative total tokens",
        "accounted_token_cost": "Physical campaign token cost attributed to this run (USD)",
        "accounted_total_tokens": "Physical campaign tokens attributed to this run",
    }
    if not modal_summary["available"]:
        for key in (
            "incremental_modal_gpu_cost",
            "incremental_modal_worker_seconds",
            "modal_gpu_cost",
            "modal_worker_seconds",
        ):
            metric_labels.pop(key)
    axis_catalog = [
        {"key": key, "label": label} for key, label in metric_labels.items()
    ] + [
        {"key": f"metric:{metric}", "label": f"Proposal metric · {metric}"}
        for metric in observed_metrics
    ]
    condition_catalog = []
    for condition in sorted({str(run["condition"]) for run in visible_runs}):
        members = [run for run in visible_runs if run["condition"] == condition]
        condition_catalog.append(
            {
                "id": condition,
                "label": members[0].get("condition_label", condition),
                "family": members[0].get("condition_family", "factorial"),
                "count": len(members),
            }
        )
    physical_points = [
        point
        for run in visible_runs
        for point in run["points"]
        if not point.get("is_seed") and point.get("physical_resource_charge", True)
    ]
    developmental_counts = Counter(
        str(point["developmental_status"])
        for point in physical_points
        if point.get("developmental_status")
    )
    semantic_summary = {
        "available": semantic,
        "design": campaign_manifest.get("design"),
        "intervention_count": campaign_manifest.get("intervention_count", 0),
        "replicates": campaign_manifest.get("replicates", 0),
        "shared_prefix_opportunities": campaign_manifest.get(
            "shared_prefix_opportunities", 0
        ),
        "physical_proposal_calls": len(physical_points),
        "logical_proposal_records": sum(
            not point.get("is_seed") for run in visible_runs for point in run["points"]
        ),
        "interventions_applied": sum(
            bool(point.get("semantic_intervention_applied"))
            for point in physical_points
        ),
        "developmental_counts": dict(developmental_counts),
        "novel_delta_proposals": sum(
            "novel_delta" in point.get("developmental_reasons", [])
            for point in physical_points
        ),
        "full_fidelity_proposals": sum(
            bool(point.get("fidelity_reached_full")) for point in physical_points
        ),
        "screened_out_proposals": sum(
            point.get("failure_kind") == "fidelity_screen_not_promoted"
            for point in physical_points
        ),
        "physical_token_cost": round(
            sum(float(run.get("accounted_token_cost", 0.0)) for run in visible_runs),
            6,
        ),
        "physical_total_tokens": sum(
            int(run.get("accounted_total_tokens", 0)) for run in visible_runs
        ),
    }
    return {
        "campaign": str(campaign),
        "available": campaign.is_dir(),
        "objective_metric": objective_metric,
        "objective_direction": objective_direction,
        "task_display_name": str(task.get("display_name", task.get("task_id", "Task"))),
        "framework_id": framework_id,
        "framework_label": framework_label,
        "axis_catalog": axis_catalog,
        "observed_metrics": observed_metrics,
        "modal_usage": modal_summary,
        "design": campaign_manifest.get("design", "c0_c3_factorial"),
        "semantic": semantic,
        "condition_catalog": condition_catalog,
        "semantic_summary": semantic_summary,
        "runs": visible_runs,
    }


def dashboard_data(
    campaigns: dict[str, Path],
    prices: dict[str, float],
    *,
    modal_h100_price_per_second: float = DEFAULT_MODAL_H100_PRICE_PER_SECOND,
) -> dict[str, Any]:
    campaign_payloads = {
        key: campaign_data(
            path,
            prices,
            modal_h100_price_per_second=modal_h100_price_per_second,
        )
        for key, path in campaigns.items()
    }
    payload = {
        "schema_version": "3.1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "price_per_million": prices,
        "modal_h100_price_per_second": modal_h100_price_per_second,
        "campaigns": campaign_payloads,
    }
    # Keep tabs loaded before the multi-campaign dashboard upgrade functional.
    # Those clients refresh in place and still read these two top-level keys.
    if "autoresearch_v16" in campaign_payloads:
        payload["autoresearch"] = campaign_payloads["autoresearch_v16"]
    if "openevolve_v2" in campaign_payloads:
        payload["openevolve_v2"] = campaign_payloads["openevolve_v2"]
    return payload


# Keep the interactive client in a standalone file so browser behavior can be
# linted and tested independently of the read-only Python log server.
PYTHON_SOURCE_PATH = Path(__file__).resolve()
PAGE_PATH = PYTHON_SOURCE_PATH.with_name("live_trajectory_dashboard.html")
PAGE = PAGE_PATH.read_text(encoding="utf-8")


def read_dashboard_page() -> str:
    """Read the client on every page request so HTML edits need no restart."""
    try:
        return PAGE_PATH.read_text(encoding="utf-8")
    except OSError:
        # Keep serving the last complete import-time copy during an editor's
        # brief replace window instead of returning a broken page.
        return PAGE


def dashboard_revision(paths: tuple[Path, ...] | None = None) -> str:
    """Return a content revision for browser and server hot reload checks."""
    digest = hashlib.sha256()
    for path in paths or (PYTHON_SOURCE_PATH, PAGE_PATH):
        digest.update(str(path).encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<temporarily-unavailable>")
    return digest.hexdigest()[:16]


def start_python_hot_reloader(poll_seconds: float = 1.0) -> None:
    """Re-exec this read-only server when its Python source changes."""
    initial_revision = dashboard_revision((PYTHON_SOURCE_PATH,))

    def watch() -> None:
        while True:
            time.sleep(poll_seconds)
            if dashboard_revision((PYTHON_SOURCE_PATH,)) == initial_revision:
                continue
            print("Dashboard Python changed; hot-restarting server...", flush=True)
            os.execv(sys.executable, [sys.executable, *sys.argv])

    threading.Thread(target=watch, name="dashboard-hot-reload", daemon=True).start()


def encode_response(body: bytes, accept_encoding: str) -> tuple[bytes, str | None]:
    """Compress large responses when the client supports gzip."""
    if len(body) >= 1024 and "gzip" in accept_encoding.lower():
        return gzip.compress(body, compresslevel=5), "gzip"
    return body, None


def make_handler(
    campaigns: dict[str, Path],
    prices: dict[str, float],
    *,
    modal_h100_price_per_second: float = DEFAULT_MODAL_H100_PRICE_PER_SECOND,
):
    class Handler(BaseHTTPRequestHandler):
        def send_payload(self, body: bytes, content_type: str) -> None:
            body, content_encoding = encode_response(
                body, self.headers.get("Accept-Encoding", "")
            )
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Vary", "Accept-Encoding")
            if content_encoding:
                self.send_header("Content-Encoding", content_encoding)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            # A reload can abandon an in-flight multi-megabyte response.  The
            # dashboard server should remain healthy when that happens.
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self.send_payload(
                    read_dashboard_page().encode("utf-8"),
                    "text/html; charset=utf-8",
                )
            elif self.path == "/api/revision":
                payload = json.dumps(
                    {"revision": dashboard_revision()}, separators=(",", ":")
                ).encode("utf-8")
                self.send_payload(payload, "application/json; charset=utf-8")
            elif self.path == "/api/data":
                payload = json.dumps(
                    dashboard_data(
                        campaigns,
                        prices,
                        modal_h100_price_per_second=modal_h100_price_per_second,
                    ),
                    separators=(",", ":"),
                ).encode("utf-8")
                self.send_payload(payload, "application/json; charset=utf-8")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"dashboard: {format % args}")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a read-only live C0-C3 trajectory dashboard."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument(
        "--autoresearch-campaign", type=Path, default=DEFAULT_AUTORESEARCH
    )
    parser.add_argument("--openevolve-campaign", type=Path, default=DEFAULT_OPENEVOLVE)
    parser.add_argument(
        "--autoresearch-v17-campaign", type=Path, default=DEFAULT_AUTORESEARCH_V17
    )
    parser.add_argument(
        "--openevolve-v21-campaign", type=Path, default=DEFAULT_OPENEVOLVE_V21
    )
    parser.add_argument(
        "--autoresearch-v17-nanogpt-campaign",
        type=Path,
        default=DEFAULT_AUTORESEARCH_V17_NANOGPT,
    )
    parser.add_argument(
        "--openevolve-v21-nanogpt-campaign",
        type=Path,
        default=DEFAULT_OPENEVOLVE_V21_NANOGPT,
    )
    parser.add_argument(
        "--autoresearch-v17-fashion-mnist-campaign",
        type=Path,
        default=DEFAULT_AUTORESEARCH_V17_FASHION_MNIST,
    )
    parser.add_argument(
        "--openevolve-v21-fashion-mnist-campaign",
        type=Path,
        default=DEFAULT_OPENEVOLVE_V21_FASHION_MNIST,
    )
    parser.add_argument(
        "--semantic-v4-fashion-mnist-campaign",
        type=Path,
        default=DEFAULT_SEMANTIC_V4_FASHION_MNIST,
    )
    parser.add_argument(
        "--input-per-million", type=float, default=DEFAULT_PRICE_PER_MILLION["input"]
    )
    parser.add_argument(
        "--cached-input-per-million",
        type=float,
        default=DEFAULT_PRICE_PER_MILLION["cached_input"],
    )
    parser.add_argument(
        "--output-per-million", type=float, default=DEFAULT_PRICE_PER_MILLION["output"]
    )
    parser.add_argument(
        "--modal-h100-price-per-second",
        type=float,
        default=DEFAULT_MODAL_H100_PRICE_PER_SECOND,
        help="campaign-attributed Modal H100 GPU rate used for dashboard estimates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = {
        "input": args.input_per_million,
        "cached_input": args.cached_input_per_million,
        "output": args.output_per_million,
    }
    campaigns = {
        "autoresearch_v16": args.autoresearch_campaign,
        "openevolve_v2": args.openevolve_campaign,
        "autoresearch_v17": args.autoresearch_v17_campaign,
        "openevolve_v21": args.openevolve_v21_campaign,
        "autoresearch_v17_nanogpt": args.autoresearch_v17_nanogpt_campaign,
        "openevolve_v21_nanogpt": args.openevolve_v21_nanogpt_campaign,
        "autoresearch_v17_fashion_mnist": (
            args.autoresearch_v17_fashion_mnist_campaign
        ),
        "openevolve_v21_fashion_mnist": (args.openevolve_v21_fashion_mnist_campaign),
        "semantic_v4_fashion_mnist": args.semantic_v4_fashion_mnist_campaign,
    }
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(
            campaigns,
            prices,
            modal_h100_price_per_second=args.modal_h100_price_per_second,
        ),
    )
    start_python_hot_reloader()
    print("Hot reload: watching dashboard HTML and Python")
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Autoresearch: {args.autoresearch_campaign}")
    print(f"Greedy OpenEvolve v2: {args.openevolve_campaign}")
    print(f"Autoresearch v1.7: {args.autoresearch_v17_campaign}")
    print(f"Greedy OpenEvolve v2.1: {args.openevolve_v21_campaign}")
    print(f"Autoresearch v1.7 nanoGPT: {args.autoresearch_v17_nanogpt_campaign}")
    print(f"Greedy OpenEvolve v2.1 nanoGPT: {args.openevolve_v21_nanogpt_campaign}")
    print(
        "Autoresearch v1.7 Fashion-MNIST: "
        f"{args.autoresearch_v17_fashion_mnist_campaign}"
    )
    print(
        "Greedy OpenEvolve v2.1 Fashion-MNIST: "
        f"{args.openevolve_v21_fashion_mnist_campaign}"
    )
    print(f"Semantic v4 Fashion-MNIST: {args.semantic_v4_fashion_mnist_campaign}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

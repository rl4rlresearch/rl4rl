#!/usr/bin/env python3
# ruff: noqa: E501
"""Read-only, locally served trajectory dashboard for live C0-C3 campaigns.

The server reads campaign logs only when a browser requests ``/api/data``.  It
does not import the experiment controller, acquire a campaign lock, or write to
any campaign artifact, so it is safe to use while controllers are running.
"""

from __future__ import annotations

import argparse
import gzip
import json
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
DEFAULT_OPENEVOLVE = REPO_ROOT / "data/c0c3/controlled-openevolve-transformer-v2-mps-campaign"
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

# These are display-only price weights, not a billing record.  They match the
# previous trajectory visualizer's token-cost convention and can be overridden.
DEFAULT_PRICE_PER_MILLION = {"input": 1.75, "cached_input": 0.175, "output": 14.0}


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
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0


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


def compact_label(run_id: str) -> str:
    parts = run_id.lower().split("-")
    block = next((part.upper() for part in parts if len(part) == 3 and part.startswith("b") and part[1:].isdigit()), "B??")
    condition = next((part.upper() for part in parts if part in {"c0", "c1", "c2", "c3"}), "C?")
    return f"{block}-{condition}"


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
) -> dict[str, Any] | None:
    state = read_json(run_dir / "state.json", {})
    if not isinstance(state, dict):
        return None
    run_id = state.get("run_id")
    condition = state.get("condition")
    if not isinstance(run_id, str) or not isinstance(condition, str):
        return None
    events = iter_jsonl(run_dir / "events.jsonl")
    started: dict[int, datetime] = {}
    elapsed_seconds = 0.0
    seed_objective = metric_at_seed(state, objective_metric)
    best_objective = seed_objective
    points: list[dict[str, Any]] = []
    latest_event_at: str | None = None
    valid_proposals = 0
    retained_proposals = 0
    latest_raw_objective: float | None = None
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
            elapsed_seconds += max(0.0, (timestamp - started[opportunity]).total_seconds())
        evaluation = event.get("evaluation", {})
        metrics = evaluation.get("metrics", {}) if isinstance(evaluation, dict) else {}
        metrics = numeric_metrics(metrics)
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
        evaluator_seconds = numeric(event.get("evaluator_seconds_cumulative")) or 0.0
        evaluator_seconds_increment = numeric(event.get("evaluator_seconds_increment")) or 0.0
        evaluator_calls = numeric(event.get("evaluator_calls_cumulative")) or 0.0
        evaluator_calls_increment = numeric(event.get("evaluator_calls_increment")) or 0.0
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
                "incremental_token_cost": round(weighted_cost(usage_increment, prices), 6),
                **usage,
                **{f"incremental_{key}": value for key, value in usage_increment.items()},
                "evaluator_seconds": evaluator_seconds,
                "incremental_evaluator_seconds": evaluator_seconds_increment,
                "evaluator_calls": evaluator_calls,
                "incremental_evaluator_calls": evaluator_calls_increment,
                "best_objective": best_objective,
                "raw_objective": objective,
                "fitness": numeric(evaluation.get("fitness")) if isinstance(evaluation, dict) else None,
                "objective_improvement": improvement,
                "objective_improvement_percent": improvement_percent,
                "metrics": metrics,
                "valid": valid,
                "retained": retained,
                "retention_decision": event.get("retention_decision"),
                "failure_kind": evaluation.get("failure_kind") if isinstance(evaluation, dict) else None,
                "proposal_type": event.get("proposal_type"),
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
    return {
        "run_id": run_id,
        "label": compact_label(run_id),
        "condition": condition.upper(),
        "status": display_status(run_dir, scientific_status),
        "scientific_status": scientific_status,
        "proposals_used": state.get("proposals_used", 0),
        "total_tokens": usage["total_tokens"],
        "token_cost": round(weighted_cost(usage, prices), 6),
        "best_objective": best_objective,
        "seed_objective": seed_objective,
        "latest_raw_objective": latest_raw_objective,
        "valid_proposals": valid_proposals,
        "invalid_proposals": max(0, len(points) - (1 if seed_objective is not None else 0) - valid_proposals),
        "retained_proposals": retained_proposals,
        "active_hours": round(elapsed_seconds / 3600, 6),
        "latest_event_at": latest_event_at,
        "lowest_parameters": (
            best_objective if objective_metric == "parameters" else None
        ),
        "points": points,
    }


def campaign_data(campaign: Path, prices: dict[str, float]) -> dict[str, Any]:
    runs_root = campaign / "runs"
    task = read_json(campaign / "inputs/task.json", {})
    task = task if isinstance(task, dict) else {}
    objective_metric = str(task.get("objective_metric", "parameters"))
    objective_direction = str(task.get("objective_direction", "minimize"))
    runs = [
        build_run(
            path,
            prices,
            objective_metric=objective_metric,
            objective_direction=objective_direction,
        )
        for path in sorted(runs_root.glob("*"))
        if path.is_dir()
    ]
    factorial_runs = [
        run
        for run in runs
        if run is not None and run["condition"] in {"C0", "C1", "C2", "C3"}
    ]
    observed_metrics = sorted(
        {
            key
            for run in factorial_runs
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
        "incremental_output_tokens": "Output tokens this proposal",
        "incremental_reasoning_output_tokens": "Reasoning output tokens this proposal",
        "incremental_token_cost": "Price-weighted token cost this proposal (USD)",
        "incremental_total_tokens": "Total tokens this proposal",
        "input_tokens": "Cumulative input tokens",
        "objective_improvement": f"Best {objective_metric} improvement from seed",
        "objective_improvement_percent": f"Best {objective_metric} improvement from seed (%)",
        "output_tokens": "Cumulative output tokens",
        "proposal": "Proposal index",
        "raw_objective": f"Proposal {objective_metric} (all outcomes)",
        "reasoning_output_tokens": "Cumulative reasoning output tokens",
        "token_cost": "Cumulative price-weighted token cost (USD)",
        "total_tokens": "Cumulative total tokens",
    }
    axis_catalog = [
        {"key": key, "label": label}
        for key, label in metric_labels.items()
    ] + [
        {"key": f"metric:{metric}", "label": f"Proposal metric · {metric}"}
        for metric in observed_metrics
    ]
    return {
        "campaign": str(campaign),
        "available": campaign.is_dir(),
        "objective_metric": objective_metric,
        "objective_direction": objective_direction,
        "axis_catalog": axis_catalog,
        "observed_metrics": observed_metrics,
        "runs": factorial_runs,
    }


def dashboard_data(campaigns: dict[str, Path], prices: dict[str, float]) -> dict[str, Any]:
    campaign_payloads = {
        key: campaign_data(path, prices) for key, path in campaigns.items()
    }
    payload = {
        "schema_version": "3.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "price_per_million": prices,
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
PAGE = (Path(__file__).with_name("live_trajectory_dashboard.html")).read_text(
    encoding="utf-8"
)


def encode_response(body: bytes, accept_encoding: str) -> tuple[bytes, str | None]:
    """Compress large responses when the client supports gzip."""
    if len(body) >= 1024 and "gzip" in accept_encoding.lower():
        return gzip.compress(body, compresslevel=5), "gzip"
    return body, None


def make_handler(campaigns: dict[str, Path], prices: dict[str, float]):
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
                self.send_payload(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/data":
                payload = json.dumps(
                    dashboard_data(campaigns, prices), separators=(",", ":")
                ).encode("utf-8")
                self.send_payload(payload, "application/json; charset=utf-8")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"dashboard: {format % args}")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a read-only live C0-C3 trajectory dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--autoresearch-campaign", type=Path, default=DEFAULT_AUTORESEARCH)
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
    parser.add_argument("--input-per-million", type=float, default=DEFAULT_PRICE_PER_MILLION["input"])
    parser.add_argument("--cached-input-per-million", type=float, default=DEFAULT_PRICE_PER_MILLION["cached_input"])
    parser.add_argument("--output-per-million", type=float, default=DEFAULT_PRICE_PER_MILLION["output"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = {"input": args.input_per_million, "cached_input": args.cached_input_per_million, "output": args.output_per_million}
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
        "openevolve_v21_fashion_mnist": (
            args.openevolve_v21_fashion_mnist_campaign
        ),
    }
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(campaigns, prices)
    )
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Autoresearch: {args.autoresearch_campaign}")
    print(f"OpenEvolve v2: {args.openevolve_campaign}")
    print(f"Autoresearch v1.7: {args.autoresearch_v17_campaign}")
    print(f"OpenEvolve v2.1: {args.openevolve_v21_campaign}")
    print(f"Autoresearch v1.7 nanoGPT: {args.autoresearch_v17_nanogpt_campaign}")
    print(f"OpenEvolve v2.1 nanoGPT: {args.openevolve_v21_nanogpt_campaign}")
    print(
        "Autoresearch v1.7 Fashion-MNIST: "
        f"{args.autoresearch_v17_fashion_mnist_campaign}"
    )
    print(
        "OpenEvolve v2.1 Fashion-MNIST: "
        f"{args.openevolve_v21_fashion_mnist_campaign}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# ruff: noqa: E501
"""Dashboard for C0-C3 and multi-arm research trajectories.

The server reads campaign logs only when a browser requests ``/api/data``.  It
does not import the experiment controller, acquire a campaign lock, or write to
any campaign artifact, so it is safe to use while controllers are running.  A
separate, sanitized dashboard history records the generic Codex primary-window
percentage for the quota-burn-rate panel.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import select
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
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
DEFAULT_SEMANTIC_V4_FASHION_MNIST_NATIVE = (
    REPO_ROOT
    / "data/c0c3/semantic-interventions-v4-fashion-native-openevolve-campaign"
)
DEFAULT_UNIFIED_V3_ADDERBOARD_GREEDY = (
    REPO_ROOT / "data/c0c3/unified-v3-adderboard-greedy-3block-campaign"
)

# These are display-only price weights, not a billing record.  They match the
# previous trajectory visualizer's token-cost convention and can be overridden.
DEFAULT_PRICE_PER_MILLION = {"input": 1.75, "cached_input": 0.175, "output": 14.0}
# Campaign ledgers record evaluator worker seconds, rather than Modal invoice
# line items. The H100 rate is the public per-second Modal GPU price and is
# configurable at launch for a historical or account-specific rate.
DEFAULT_MODAL_H100_PRICE_PER_SECOND = 0.001097
DEFAULT_CODEX_RATE_LIMIT_HISTORY = REPO_ROOT / "data/codex-rate-limit-history.jsonl"
DEFAULT_CODEX_RATE_LIMIT_SAMPLE_SECONDS = 30.0
CODEX_RATE_LIMIT_LOOKBACK_SECONDS = 30 * 60
DEFAULT_CODEX_SESSION_ROOT = Path.home() / ".codex" / "sessions"
CODEX_EXPERIMENT_WORKSPACE_PREFIXES = (
    "transformer-design-cycle-",
    "native-openevolve-codex-",
)


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


def _rate_limit_percent(value: Any) -> int | None:
    """Return a valid whole quota percentage, or ``None`` for malformed data."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    if value < 0 or value > 100:
        return None
    return int(value)


def codex_primary_rate_limit_sample(response: Any) -> dict[str, Any] | None:
    """Extract only the generic Codex primary-window percentage from a response.

    The app-server response contains account metadata and potentially several
    product-specific quota buckets.  The dashboard deliberately persists only
    the generic ``codex`` primary-window percentage needed for the single
    burn-rate metric, never the raw response or account metadata.
    """
    if not isinstance(response, dict):
        return None
    by_limit_id = response.get("rateLimitsByLimitId")
    snapshot = (
        by_limit_id.get("codex")
        if isinstance(by_limit_id, dict)
        else response.get("rateLimits")
    )
    if not isinstance(snapshot, dict):
        return None
    primary = snapshot.get("primary")
    if not isinstance(primary, dict):
        return None
    used_percent = _rate_limit_percent(primary.get("usedPercent"))
    if used_percent is None:
        return None
    resets_at = primary.get("resetsAt")
    window_duration = primary.get("windowDurationMins")
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "used_percent": used_percent,
        "reset_at": resets_at if isinstance(resets_at, int) else None,
        "window_duration_minutes": (
            window_duration
            if isinstance(window_duration, int) and window_duration > 0
            else None
        ),
    }


def read_codex_primary_rate_limit(
    *, codex_binary: str = "codex", timeout_seconds: float = 15.0
) -> dict[str, Any] | None:
    """Read the account's generic Codex primary window through app-server.

    This uses the caller's existing Codex login.  It is intentionally read
    only: the two JSON-RPC messages initialize the local service and request
    ``account/rateLimits/read``.  Neither credentials nor the raw response are
    logged or returned.
    """
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            [codex_binary, "app-server", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None:
            return None
        requests = (
            {
                "id": 1,
                "method": "initialize",
                "params": {
                    "clientInfo": {
                        "name": "rl4rl-live-trajectory-dashboard",
                        "version": "1",
                    }
                },
            },
            {"id": 2, "method": "account/rateLimits/read", "params": None},
        )
        for request in requests:
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        process.stdin.flush()

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            readable, _, _ = select.select(
                [process.stdout], [], [], max(0.0, deadline - time.monotonic())
            )
            if not readable:
                break
            line = process.stdout.readline()
            if not line:
                break
            with suppress(json.JSONDecodeError):
                message = json.loads(line)
                if message.get("id") == 2:
                    return codex_primary_rate_limit_sample(message.get("result"))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    finally:
        if process is not None:
            with suppress(ProcessLookupError, subprocess.TimeoutExpired):
                process.terminate()
                process.wait(timeout=2)
    return None


def append_codex_rate_limit_sample(path: Path, sample: dict[str, Any]) -> None:
    """Append a sanitized rate-limit observation without touching run logs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(sample, separators=(",", ":")) + "\n")


def _normalized_rate_limit_history(path: Path) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for record in iter_jsonl(path):
        observed_at = parse_timestamp(record.get("timestamp"))
        used_percent = _rate_limit_percent(record.get("used_percent"))
        if observed_at is None or used_percent is None:
            continue
        reset_at = record.get("reset_at")
        duration = record.get("window_duration_minutes")
        tracked_usage = record.get("tracked_usage")
        tracked_usage = tracked_usage if isinstance(tracked_usage, dict) else {}
        observations.append(
            {
                "timestamp": observed_at.astimezone().isoformat(),
                "used_percent": used_percent,
                "reset_at": reset_at if isinstance(reset_at, int) else None,
                "window_duration_minutes": (
                    duration if isinstance(duration, int) and duration > 0 else None
                ),
                "tracked_usage": normalized_usage(tracked_usage),
            }
        )
    return sorted(observations, key=lambda item: item["timestamp"])


def quota_window_start(observation: dict[str, Any]) -> datetime | None:
    """Return the current quota window's start, when app-server reports it."""
    reset_at = observation.get("reset_at")
    duration = observation.get("window_duration_minutes")
    if not isinstance(reset_at, int) or not isinstance(duration, int) or duration <= 0:
        return None
    return datetime.fromtimestamp(reset_at - duration * 60, UTC).astimezone()


def quota_reset_time(observation: dict[str, Any]) -> datetime | None:
    """Return the reported primary-quota reset time in the local timezone."""
    reset_at = observation.get("reset_at")
    if not isinstance(reset_at, int) or reset_at <= 0:
        return None
    return datetime.fromtimestamp(reset_at, UTC).astimezone()


def projected_remaining_percent(
    used_percent: float | None,
    minutes_per_percent: float | None,
    *,
    from_time: datetime | None,
    reset_time: datetime | None,
) -> float | None:
    """Project remaining quota at reset without clamping below zero."""
    if (
        used_percent is None
        or minutes_per_percent is None
        or minutes_per_percent <= 0
        or from_time is None
        or reset_time is None
    ):
        return None
    minutes_until_reset = (reset_time - from_time).total_seconds() / 60
    return 100 - used_percent - minutes_until_reset / minutes_per_percent


def projected_zero_crossing_time(
    used_percent: float | None,
    minutes_per_percent: float | None,
    *,
    from_time: datetime | None,
) -> datetime | None:
    """Project when the current primary-quota window would reach 0% remaining.

    This deliberately continues the measured pace past the reported reset time:
    it is an exhaustion forecast, not a claim that the quota window stays open
    until that point.
    """
    if (
        used_percent is None
        or minutes_per_percent is None
        or minutes_per_percent <= 0
        or from_time is None
    ):
        return None
    if used_percent >= 100:
        return from_time
    return from_time + timedelta(minutes=(100 - used_percent) * minutes_per_percent)


def quota_change_points(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rates only at real percentage changes, never while it is flat."""
    points: list[dict[str, Any]] = []
    previous_change: dict[str, Any] | None = None
    for observation in observations:
        current_time = parse_timestamp(observation.get("timestamp"))
        current_used = _rate_limit_percent(observation.get("used_percent"))
        if current_time is None or current_used is None:
            continue
        if previous_change is None:
            previous_change = observation
            continue
        previous_time = parse_timestamp(previous_change.get("timestamp"))
        previous_used = _rate_limit_percent(previous_change.get("used_percent"))
        if (
            previous_time is None
            or previous_used is None
            or observation.get("reset_at") != previous_change.get("reset_at")
            or current_used < previous_used
        ):
            previous_change = observation
            continue
        percent_drop = current_used - previous_used
        if percent_drop == 0:
            continue
        elapsed_seconds = (current_time - previous_time).total_seconds()
        if elapsed_seconds <= 0:
            previous_change = observation
            continue
        previous_tracked = normalized_usage(previous_change.get("tracked_usage"))
        current_tracked = normalized_usage(observation.get("tracked_usage"))
        tracked_delta = max(
            0, current_tracked["total_tokens"] - previous_tracked["total_tokens"]
        )
        points.append(
            {
                "timestamp": observation["timestamp"],
                "minutes_per_percent": round(elapsed_seconds / percent_drop / 60, 4),
                "percent_drop": percent_drop,
                "tracked_tokens": tracked_delta,
                "initial_observation_interval": len(points) == 0,
            }
        )
        previous_change = observation
    return points


def usage_breakdown(usage: dict[str, Any]) -> dict[str, int]:
    """Expose token types without double-counting their documented subsets."""
    normalized = normalized_usage(usage)
    return {
        "total_tokens": normalized["total_tokens"],
        "input_tokens": normalized["input_tokens"],
        "cached_input_tokens": normalized["cached_input_tokens"],
        "uncached_input_tokens": max(
            0, normalized["input_tokens"] - normalized["cached_input_tokens"]
        ),
        "output_tokens": normalized["output_tokens"],
        "reasoning_output_tokens": normalized["reasoning_output_tokens"],
        "nonreasoning_output_tokens": max(
            0, normalized["output_tokens"] - normalized["reasoning_output_tokens"]
        ),
    }


def dashboard_campaign_roots(campaigns: dict[str, Path]) -> tuple[Path, ...]:
    """Find all local C0-C3 ledgers, plus explicit non-repository campaigns."""
    candidates = list(campaigns.values())
    local_root = REPO_ROOT / "data/c0c3"
    with suppress(OSError):
        candidates.extend(
            path for path in local_root.iterdir() if (path / "runs").is_dir()
        )
    unique: dict[str, Path] = {}
    for path in candidates:
        with suppress(OSError):
            resolved = path.resolve()
            if (resolved / "runs").is_dir():
                unique[str(resolved)] = resolved
    return tuple(unique[key] for key in sorted(unique))


def campaign_default_token_settings(campaign: Path) -> dict[str, str]:
    protocol = read_json(campaign / "inputs/protocol.json", {})
    protocol = protocol if isinstance(protocol, dict) else {}
    model = protocol.get("model", {})
    model = model if isinstance(model, dict) else {}
    return {
        "model": str(model.get("name", "unknown")),
        "reasoning_effort": str(model.get("reasoning_effort", "unknown")),
        "service_tier": str(model.get("service_tier", "unknown")),
        "sandbox": str(model.get("sandbox", "unknown")),
        "approval_policy": str(model.get("approval_policy", "unknown")),
        "conversation_mode": str(protocol.get("conversation_mode", "unknown")),
    }


def semantic_shadow_prefixes(campaign: Path) -> dict[str, int]:
    """Map semantic shadow runs to their copied-prefix boundary."""
    value = read_json(campaign / "semantic-prefix.json", {})
    if not isinstance(value, dict):
        return {}
    shadows: dict[str, int] = {}
    for row in value.get("replicates", []):
        if not isinstance(row, dict):
            continue
        through = row.get("shared_through_opportunity", 0)
        if not isinstance(through, int):
            continue
        for run_id in row.get("shadow_run_ids", []):
            if isinstance(run_id, str):
                shadows[run_id] = through
    return shadows


def _event_usage_increment(
    event: dict[str, Any], previous_cumulative: dict[str, int]
) -> tuple[dict[str, int], dict[str, int]]:
    """Read an event increment, with a cumulative-difference fallback."""
    raw_increment = event.get("usage_increment")
    increment = normalized_usage(raw_increment)
    raw_cumulative = event.get("usage_cumulative")
    cumulative = normalized_usage(raw_cumulative)
    has_increment = isinstance(raw_increment, dict) and any(
        key in raw_increment
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        )
    )
    has_cumulative = isinstance(raw_cumulative, dict) and any(
        key in raw_cumulative
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        )
    )
    if not has_increment and has_cumulative:
        increment = {
            key: max(0, cumulative[key] - previous_cumulative.get(key, 0))
            for key in cumulative
        }
    return increment, cumulative if has_cumulative else previous_cumulative


def local_codex_token_usage(
    campaign_roots: Iterable[Path],
    *,
    window_start: datetime | None,
    window_end: datetime | None = None,
    recent_minutes: int = 30,
) -> dict[str, Any]:
    """Read completed local experiment calls and group their exact token logs.

    These are the actual Codex JSONL usage fields emitted after each completed
    proposal.  An active Codex turn is not included until its JSONL record is
    finalized, so the result explicitly reports that live-recording boundary.
    """
    end = window_end or datetime.now().astimezone()
    start = window_start or datetime(1970, 1, 1, tzinfo=UTC)
    recent_start = max(start, end - timedelta(minutes=recent_minutes))
    total = Counter[str]()
    recent = Counter[str]()
    model_totals: dict[tuple[str, ...], Counter[str]] = {}
    model_calls: Counter[tuple[str, ...]] = Counter()
    model_runs: dict[tuple[str, ...], set[str]] = {}
    completed_calls = 0
    active_calls = 0
    for campaign in campaign_roots:
        runs_root = campaign / "runs"
        if not runs_root.is_dir():
            continue
        defaults = campaign_default_token_settings(campaign)
        shadows = semantic_shadow_prefixes(campaign)
        for run_dir in sorted(runs_root.glob("*")):
            if not run_dir.is_dir():
                continue
            state = read_json(run_dir / "state.json", {})
            if isinstance(state, dict) and state.get("active") is not None:
                active_calls += 1
            events = iter_jsonl(run_dir / "events.jsonl")
            provenance = {
                event["opportunity"]: event
                for event in events
                if event.get("event") == "v3_proposal_provenance"
                and isinstance(event.get("opportunity"), int)
            }
            previous_cumulative: dict[str, int] = {}
            for event in events:
                if event.get("event") != "proposal_completed":
                    continue
                opportunity = event.get("opportunity")
                if not isinstance(opportunity, int):
                    continue
                increment, previous_cumulative = _event_usage_increment(
                    event, previous_cumulative
                )
                timestamp = parse_timestamp(event.get("timestamp"))
                if timestamp is None or timestamp < start or timestamp > end:
                    continue
                if opportunity <= shadows.get(run_dir.name, 0):
                    continue
                settings = dict(defaults)
                source = provenance.get(opportunity, {})
                if isinstance(source, dict):
                    settings["model"] = str(
                        source.get("provider_model_requested", settings["model"])
                    )
                    settings["reasoning_effort"] = str(
                        source.get(
                            "reasoning_effort_requested", settings["reasoning_effort"]
                        )
                    )
                    settings["service_tier"] = str(
                        source.get("service_tier_observed", settings["service_tier"])
                    )
                settings["service_tier"] = str(
                    event.get("codex_service_tier", settings["service_tier"])
                )
                key = tuple(
                    settings[name]
                    for name in (
                        "model",
                        "reasoning_effort",
                        "service_tier",
                        "sandbox",
                        "approval_policy",
                        "conversation_mode",
                    )
                )
                model_totals.setdefault(key, Counter()).update(increment)
                model_calls[key] += 1
                model_runs.setdefault(key, set()).add(run_dir.name)
                total.update(increment)
                if timestamp >= recent_start:
                    recent.update(increment)
                completed_calls += 1
    model_rows = []
    setting_names = (
        "model",
        "reasoning_effort",
        "service_tier",
        "sandbox",
        "approval_policy",
        "conversation_mode",
    )
    for key, usage in sorted(
        model_totals.items(), key=lambda item: item[1]["total_tokens"], reverse=True
    ):
        row = dict(zip(setting_names, key, strict=True))
        row.update(usage_breakdown(dict(usage)))
        row["completed_calls"] = model_calls[key]
        row["runs"] = len(model_runs[key])
        model_rows.append(row)
    elapsed_recent_seconds = max(1.0, (end - recent_start).total_seconds())
    total_breakdown = usage_breakdown(dict(total))
    recent_breakdown = usage_breakdown(dict(recent))
    return {
        "window_start": start.isoformat() if window_start else None,
        "window_end": end.isoformat(),
        "completed_calls": completed_calls,
        "active_calls_not_yet_logged": active_calls,
        "total": total_breakdown,
        "recent": {
            **recent_breakdown,
            "minutes": round(elapsed_recent_seconds / 60, 4),
            "tokens_per_minute": round(
                recent_breakdown["total_tokens"] / (elapsed_recent_seconds / 60), 4
            ),
        },
        "models": model_rows,
        "scope": "completed local C0-C3 Codex proposal logs only",
    }


def codex_session_source(cwd: Any) -> str:
    """Classify local Codex sessions without inspecting their message content."""
    workspace_name = Path(str(cwd or "")).name
    if workspace_name.startswith(CODEX_EXPERIMENT_WORKSPACE_PREFIXES):
        return "experiment_runners"
    return "other_local_codex"


def codex_session_paths(
    session_root: Path,
    *,
    start: datetime,
    end: datetime,
) -> list[Path]:
    """Return only date directories that intersect the current quota window."""
    start_day = start.astimezone(UTC).date()
    end_day = end.astimezone(UTC).date()
    paths: list[Path] = []
    day = start_day
    while day <= end_day:
        directory = session_root / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
        if directory.is_dir():
            paths.extend(sorted(directory.glob("*.jsonl")))
        day += timedelta(days=1)
    return paths


def local_codex_session_usage(
    *,
    session_root: Path = DEFAULT_CODEX_SESSION_ROOT,
    window_start: datetime | None,
    window_end: datetime | None = None,
    recent_minutes: int = 30,
) -> dict[str, Any]:
    """Attribute local Codex token records to experiment runner or other use.

    Codex writes a ``token_count`` event after a completed model turn.  This
    scanner reads only its structured token counters and session workspace, not
    prompts, answers, or private reasoning.  The workspace prefixes are the
    ones created by this repository's C0-C3 runners.
    """
    end = window_end or datetime.now().astimezone()
    start = window_start or datetime(1970, 1, 1, tzinfo=UTC)
    recent_start = max(start, end - timedelta(minutes=recent_minutes))
    total_by_source = {
        "experiment_runners": Counter[str](),
        "other_local_codex": Counter[str](),
    }
    recent_by_source = {
        "experiment_runners": Counter[str](),
        "other_local_codex": Counter[str](),
    }
    turns_by_source: Counter[str] = Counter()
    recent_turns_by_source: Counter[str] = Counter()
    scanned_sessions = 0
    for path in codex_session_paths(session_root, start=start, end=end):
        records = iter_jsonl(path)
        metadata = next(
            (
                record.get("payload")
                for record in records
                if record.get("type") == "session_meta"
                and isinstance(record.get("payload"), dict)
            ),
            {},
        )
        source = codex_session_source(metadata.get("cwd") if isinstance(metadata, dict) else None)
        scanned_sessions += 1
        for record in records:
            payload = record.get("payload")
            if not isinstance(payload, dict) or payload.get("type") != "token_count":
                continue
            timestamp = parse_timestamp(record.get("timestamp"))
            info = payload.get("info")
            last_usage = info.get("last_token_usage") if isinstance(info, dict) else None
            if timestamp is None or timestamp < start or timestamp > end:
                continue
            usage = normalized_usage(last_usage)
            if not usage["total_tokens"]:
                continue
            total_by_source[source].update(usage)
            turns_by_source[source] += 1
            if timestamp >= recent_start:
                recent_by_source[source].update(usage)
                recent_turns_by_source[source] += 1

    elapsed_recent_seconds = max(1.0, (end - recent_start).total_seconds())

    def source_summary(source: str) -> dict[str, Any]:
        total = usage_breakdown(dict(total_by_source[source]))
        recent = usage_breakdown(dict(recent_by_source[source]))
        return {
            "total": total,
            "recent": {
                **recent,
                "minutes": round(elapsed_recent_seconds / 60, 4),
                "tokens_per_minute": round(
                    recent["total_tokens"] / (elapsed_recent_seconds / 60), 4
                ),
            },
            "completed_turns": turns_by_source[source],
            "recent_completed_turns": recent_turns_by_source[source],
        }

    return {
        "available": bool(sum(turns_by_source.values())),
        "scanned_sessions": scanned_sessions,
        "scope": "local Codex token_count records; experiment runner workspaces versus all other local Codex workspaces",
        "experiment_runners": source_summary("experiment_runners"),
        "other_local_codex": source_summary("other_local_codex"),
    }


def experiment_only_minutes_per_percent(
    account_wide_minutes_per_percent: float | None,
    *,
    experiment_tokens: int,
    all_local_tokens: int,
) -> float | None:
    """Infer a subset's pace from its measured share of current local usage."""
    if (
        account_wide_minutes_per_percent is None
        or account_wide_minutes_per_percent <= 0
        or experiment_tokens <= 0
        or all_local_tokens <= 0
    ):
        return None
    return account_wide_minutes_per_percent * all_local_tokens / experiment_tokens


def codex_rate_limit_payload(
    history_path: Path,
    *,
    sample_interval_seconds: float = DEFAULT_CODEX_RATE_LIMIT_SAMPLE_SECONDS,
    token_campaign_roots: Iterable[Path] = (),
    session_root: Path = DEFAULT_CODEX_SESSION_ROOT,
) -> dict[str, Any]:
    """Build the one-number quota burn-rate payload consumed by the page."""
    observations = _normalized_rate_limit_history(history_path)
    latest = observations[-1] if observations else None
    window_start = quota_window_start(latest) if latest is not None else None
    token_usage = local_codex_token_usage(
        token_campaign_roots,
        window_start=window_start,
    )
    session_usage = local_codex_session_usage(
        session_root=session_root,
        window_start=window_start,
    )
    change_points = quota_change_points(observations)
    observed_minutes_per_percent = None
    if change_points:
        weighted_minutes = sum(
            float(point["minutes_per_percent"]) * int(point["percent_drop"])
            for point in change_points
        )
        observed_percent = sum(int(point["percent_drop"]) for point in change_points)
        observed_minutes_per_percent = weighted_minutes / observed_percent

    direct_calibrations = [
        point
        for point in change_points
        if int(point.get("tracked_tokens", 0)) > 0
    ]
    tokens_per_percent: float | None = None
    calibration_method = "waiting for a percentage-change/token pairing"
    if direct_calibrations:
        known_tokens = sum(int(point["tracked_tokens"]) for point in direct_calibrations)
        known_percent = sum(int(point["percent_drop"]) for point in direct_calibrations)
        if known_percent:
            tokens_per_percent = known_tokens / known_percent
            calibration_method = "direct quota-change intervals"
    latest_used = _rate_limit_percent(latest.get("used_percent")) if latest else None
    if tokens_per_percent is None and latest_used and latest_used > 0:
        backfilled_tokens = int(token_usage["total"]["total_tokens"])
        if backfilled_tokens:
            tokens_per_percent = backfilled_tokens / latest_used
            calibration_method = "backfilled current quota window"
    recent_tokens_per_minute = numeric(token_usage["recent"].get("tokens_per_minute"))
    estimated_minutes_per_percent = None
    if (
        tokens_per_percent is not None
        and recent_tokens_per_minute is not None
        and recent_tokens_per_minute > 0
    ):
        estimated_minutes_per_percent = tokens_per_percent / recent_tokens_per_minute
    recent_experiment_tokens = int(
        session_usage["experiment_runners"]["recent"]["total_tokens"]
    )
    recent_other_tokens = int(
        session_usage["other_local_codex"]["recent"]["total_tokens"]
    )
    recent_all_local_tokens = recent_experiment_tokens + recent_other_tokens
    session_attributed_experiment_minutes_per_percent = experiment_only_minutes_per_percent(
        observed_minutes_per_percent,
        experiment_tokens=recent_experiment_tokens,
        all_local_tokens=recent_all_local_tokens,
    )
    headline_minutes_per_percent = (
        session_attributed_experiment_minutes_per_percent
        if session_attributed_experiment_minutes_per_percent is not None
        else observed_minutes_per_percent
    )
    latest_sampled_at = parse_timestamp(latest.get("timestamp")) if latest else None
    reset_time = quota_reset_time(latest) if latest is not None else None
    account_wide_projected_remaining = projected_remaining_percent(
        latest_used,
        observed_minutes_per_percent,
        from_time=latest_sampled_at,
        reset_time=reset_time,
    )
    experiment_only_projected_remaining = projected_remaining_percent(
        latest_used,
        session_attributed_experiment_minutes_per_percent,
        from_time=latest_sampled_at,
        reset_time=reset_time,
    )
    account_wide_zero_crossing = projected_zero_crossing_time(
        latest_used,
        observed_minutes_per_percent,
        from_time=latest_sampled_at,
    )
    experiment_only_zero_crossing = projected_zero_crossing_time(
        latest_used,
        session_attributed_experiment_minutes_per_percent,
        from_time=latest_sampled_at,
    )
    return {
        "available": bool(observations),
        "sample_count": len(observations),
        "sample_interval_seconds": sample_interval_seconds,
        "lookback_minutes": CODEX_RATE_LIMIT_LOOKBACK_SECONDS // 60,
        "latest_minutes_per_percent": (
            round(headline_minutes_per_percent, 4)
            if headline_minutes_per_percent is not None
            else None
        ),
        "observed_minutes_per_percent": (
            round(observed_minutes_per_percent, 4)
            if observed_minutes_per_percent is not None
            else None
        ),
        "estimated_minutes_per_percent": (
            round(estimated_minutes_per_percent, 4)
            if estimated_minutes_per_percent is not None
            else None
        ),
        "account_wide_minutes_per_percent": (
            round(observed_minutes_per_percent, 4)
            if observed_minutes_per_percent is not None
            else None
        ),
        "experiment_only_minutes_per_percent": (
            round(session_attributed_experiment_minutes_per_percent, 4)
            if session_attributed_experiment_minutes_per_percent is not None
            else None
        ),
        "calibration": {
            "method": "local-session attribution" if session_attributed_experiment_minutes_per_percent is not None else calibration_method,
            "tokens_per_percent": (
                round(tokens_per_percent, 2) if tokens_per_percent is not None else None
            ),
            "recent_tokens_per_minute": (
                round(recent_tokens_per_minute, 2)
                if recent_tokens_per_minute is not None
                else None
            ),
        },
        "session_attribution": {
            **session_usage,
            "recent_experiment_share": (
                round(recent_experiment_tokens / recent_all_local_tokens, 6)
                if recent_all_local_tokens
                else None
            ),
        },
        "latest_sampled_at": latest["timestamp"] if latest else None,
        "latest_used_percent": latest_used,
        "reset_at_iso": reset_time.isoformat() if reset_time else None,
        "window_duration_minutes": (
            latest.get("window_duration_minutes") if latest else None
        ),
        "projected_remaining_percent": (
            round(experiment_only_projected_remaining, 4)
            if experiment_only_projected_remaining is not None
            else None
        ),
        "account_wide_projected_remaining_percent": (
            round(account_wide_projected_remaining, 4)
            if account_wide_projected_remaining is not None
            else None
        ),
        "experiment_only_projected_remaining_percent": (
            round(experiment_only_projected_remaining, 4)
            if experiment_only_projected_remaining is not None
            else None
        ),
        "account_wide_zero_crossing_at": (
            account_wide_zero_crossing.isoformat()
            if account_wide_zero_crossing is not None
            else None
        ),
        "experiment_only_zero_crossing_at": (
            experiment_only_zero_crossing.isoformat()
            if experiment_only_zero_crossing is not None
            else None
        ),
        "window_start": window_start.isoformat() if window_start else None,
        "points": change_points,
        "token_usage": token_usage,
    }


def start_codex_rate_limit_sampler(
    history_path: Path,
    *,
    sample_seconds: float = DEFAULT_CODEX_RATE_LIMIT_SAMPLE_SECONDS,
    token_campaign_roots: Iterable[Path] = (),
) -> None:
    """Start a low-frequency, read-only primary-Codex quota sampler."""
    interval = max(30.0, sample_seconds)
    campaign_roots = tuple(token_campaign_roots)

    def sample_forever() -> None:
        while True:
            sample = read_codex_primary_rate_limit()
            if sample is not None:
                sampled_at = parse_timestamp(sample.get("timestamp"))
                tracked_usage = local_codex_token_usage(
                    campaign_roots,
                    window_start=quota_window_start(sample),
                    window_end=sampled_at,
                )
                sample["tracked_usage"] = tracked_usage["total"]
                with suppress(OSError):
                    append_codex_rate_limit_sample(history_path, sample)
            time.sleep(interval)

    threading.Thread(
        target=sample_forever,
        name="codex-rate-limit-sampler",
        daemon=True,
    ).start()


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


def substantive_claim(value: Any) -> bool:
    """Whether a scientific-record field contains an actual agent claim."""

    if not isinstance(value, str):
        return False
    return value.strip().casefold() not in {
        "",
        "[not recorded]",
        "[missing mechanism]",
        "[missing hypothesis]",
        "[missing intended edit]",
        "[missing evidence]",
    }


def manipulation_review_index(
    campaign: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """Join blinded packet annotations to runs without mutating review files."""

    mapping_path = campaign / "review/v3-manipulation/private-mapping.jsonl"
    packets_root = campaign / "review/v3-manipulation/packets"
    index: dict[tuple[str, int], dict[str, Any]] = {}
    annotation_fields = (
        "old_assumption_identifiable",
        "new_mechanism_implemented",
        "distinct_from_recent_lineage",
        "primarily_tuning_pruning_or_deletion",
        "cleanly_attributable",
        "feasible_under_task_contract",
        "novelty_score",
    )
    reviewed = 0
    fully_reviewed = 0
    for row in iter_jsonl(mapping_path):
        run_id = row.get("run_id")
        opportunity = row.get("opportunity")
        packet_id = row.get("packet_id")
        if (
            not isinstance(run_id, str)
            or not isinstance(opportunity, int)
            or not isinstance(packet_id, str)
        ):
            continue
        packet = read_json(packets_root / f"{packet_id}.json", {})
        annotation = packet.get("annotation", {}) if isinstance(packet, dict) else {}
        annotation = annotation if isinstance(annotation, dict) else {}
        completed = [annotation.get(field) is not None for field in annotation_fields]
        if any(completed):
            reviewed += 1
        if completed and all(completed):
            fully_reviewed += 1
        index[(run_id, opportunity)] = {
            "packet_id": packet_id,
            "proposal_type": row.get("proposal_type"),
            "annotation": {field: annotation.get(field) for field in annotation_fields},
            "reviewed": any(completed),
            "fully_reviewed": bool(completed and all(completed)),
        }
    return index, {
        "available": mapping_path.is_file(),
        "packets": len(index),
        "reviewed_packets": reviewed,
        "fully_reviewed_packets": fully_reviewed,
        "annotation_fields": list(annotation_fields),
    }


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
    manipulation_reviews: dict[tuple[str, int], dict[str, Any]] | None = None,
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
    manipulation_reviews = manipulation_reviews or {}
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
        provenance = read_json(
            run_dir
            / "opportunities"
            / f"{opportunity:04d}"
            / "candidate-provenance.json",
            {},
        )
        provenance = provenance if isinstance(provenance, dict) else {}
        claims = provenance.get("agent_claims", {})
        claims = claims if isinstance(claims, dict) else {}
        hypothesis = event.get("hypothesis", claims.get("hypothesis"))
        intended_edit = event.get("intended_edit", claims.get("intended_edit"))
        mechanism = event.get("mechanism", claims.get("mechanism"))
        evidence = event.get("evidence", claims.get("evidence"))
        claim_completeness = {
            "hypothesis": substantive_claim(hypothesis),
            "intended_edit": substantive_claim(intended_edit),
            "mechanism": substantive_claim(mechanism),
            "evidence": substantive_claim(evidence),
        }
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
                "hypothesis": hypothesis,
                "intended_edit": intended_edit,
                "mechanism": mechanism,
                "evidence": evidence,
                "claim_completeness": claim_completeness,
                "complete_scientific_record": all(claim_completeness.values()),
                "source_provenance_available": bool(provenance),
                "source_changed_files": int(
                    numeric(provenance.get("changed_files")) or 0
                ),
                "source_added_lines": int(
                    numeric(provenance.get("added_lines")) or 0
                ),
                "source_deleted_lines": int(
                    numeric(provenance.get("deleted_lines")) or 0
                ),
                "semantic_delta_fingerprint": provenance.get(
                    "semantic_delta_fingerprint"
                ),
                "manipulation_review": manipulation_reviews.get(
                    (run_id, opportunity)
                ),
                "prompt_provenance_available": bool(event.get("prompt_hashes")),
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
                "intended_edit": "none",
                "mechanism": None,
                "evidence": None,
                "claim_completeness": {},
                "complete_scientific_record": False,
                "source_provenance_available": False,
                "source_changed_files": 0,
                "source_added_lines": 0,
                "source_deleted_lines": 0,
                "semantic_delta_fingerprint": None,
                "manipulation_review": None,
                "prompt_provenance_available": False,
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
    manipulation_reviews, manipulation_review_summary = manipulation_review_index(
        campaign
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
            manipulation_reviews=manipulation_reviews,
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
    scientific_points = [
        point
        for run in visible_runs
        for point in run["points"]
        if not point.get("is_seed") and point.get("physical_resource_charge", True)
    ]
    distinct_deltas = {
        str(point["semantic_delta_fingerprint"])
        for point in scientific_points
        if point.get("semantic_delta_fingerprint")
    }
    distinct_mechanisms = {
        str(point["mechanism"]).strip().casefold()
        for point in scientific_points
        if substantive_claim(point.get("mechanism"))
    }
    source_changed = sum(
        int(point.get("source_changed_files", 0)) > 0 for point in scientific_points
    )
    complete_records = sum(
        bool(point.get("complete_scientific_record")) for point in scientific_points
    )
    scientific_process = {
        "physical_proposals": len(scientific_points),
        "source_changed_proposals": source_changed,
        "complete_scientific_records": complete_records,
        "executable_proposals": sum(
            bool(point.get("valid")) for point in scientific_points
        ),
        "novel_delta_proposals": sum(
            "novel_delta" in point.get("developmental_reasons", [])
            for point in scientific_points
        ),
        "retained_proposals": sum(
            bool(point.get("retained")) for point in scientific_points
        ),
        "distinct_source_deltas": len(distinct_deltas),
        "distinct_reported_mechanisms": len(distinct_mechanisms),
        "source_provenance_coverage": sum(
            bool(point.get("source_provenance_available"))
            for point in scientific_points
        ),
        "prompt_provenance_coverage": sum(
            bool(point.get("prompt_provenance_available"))
            for point in scientific_points
        ),
        "failure_kinds": dict(
            Counter(
                str(point["failure_kind"])
                for point in scientific_points
                if point.get("failure_kind")
            )
        ),
        "manipulation_review": manipulation_review_summary,
        "funnel_note": (
            "Evidence counts are process indicators; novelty and retention are not "
            "strictly nested stages of one causal funnel."
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
        "scientific_process": scientific_process,
        "runs": visible_runs,
    }


def dashboard_data(
    campaigns: dict[str, Path],
    prices: dict[str, float],
    *,
    modal_h100_price_per_second: float = DEFAULT_MODAL_H100_PRICE_PER_SECOND,
    codex_rate_limit_history: Path = DEFAULT_CODEX_RATE_LIMIT_HISTORY,
    codex_rate_limit_sample_seconds: float = DEFAULT_CODEX_RATE_LIMIT_SAMPLE_SECONDS,
    codex_token_campaign_roots: Iterable[Path] = (),
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
        "codex_rate_limit": codex_rate_limit_payload(
            codex_rate_limit_history,
            sample_interval_seconds=codex_rate_limit_sample_seconds,
            token_campaign_roots=codex_token_campaign_roots,
        ),
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
SCIENCE_PAGE_PATH = PYTHON_SOURCE_PATH.with_name("scientific_process_dashboard.html")
PAGE = PAGE_PATH.read_text(encoding="utf-8")
SCIENCE_PAGE = SCIENCE_PAGE_PATH.read_text(encoding="utf-8")


def read_dashboard_page() -> str:
    """Read the client on every page request so HTML edits need no restart."""
    try:
        return PAGE_PATH.read_text(encoding="utf-8")
    except OSError:
        # Keep serving the last complete import-time copy during an editor's
        # brief replace window instead of returning a broken page.
        return PAGE


def read_science_page() -> str:
    """Read the AISCiK process page without requiring a server restart."""

    try:
        return SCIENCE_PAGE_PATH.read_text(encoding="utf-8")
    except OSError:
        return SCIENCE_PAGE


def dashboard_revision(paths: tuple[Path, ...] | None = None) -> str:
    """Return a content revision for browser and server hot reload checks."""
    digest = hashlib.sha256()
    for path in paths or (PYTHON_SOURCE_PATH, PAGE_PATH, SCIENCE_PAGE_PATH):
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


class DashboardPayloadCache:
    """Single-flight, stale-while-refresh cache for the expensive log snapshot.

    Multiple open dashboard pages refresh independently. Building the complete
    multi-campaign payload once per request made those tabs rescan the same
    append-only artifacts in parallel. Once a snapshot exists, an expired
    request receives it immediately while one background refresh builds the
    next snapshot. This cache is read-only with respect to campaign state.
    """

    def __init__(self, builder: Any, *, ttl_seconds: float = 15.0) -> None:
        self._builder = builder
        self._ttl_seconds = ttl_seconds
        self._condition = threading.Condition()
        self._body: bytes | None = None
        self._gzip_body: bytes | None = None
        self._built_at = 0.0
        self._building = False
        self._error: BaseException | None = None

    def _build(self) -> tuple[bytes, bytes]:
        body = self._builder()
        return body, gzip.compress(body, compresslevel=5)

    def _finish_build(self) -> None:
        try:
            body, gzip_body = self._build()
        except BaseException as exc:  # Preserve a prior usable snapshot.
            with self._condition:
                self._error = exc
                self._building = False
                self._condition.notify_all()
            print(f"Dashboard snapshot refresh failed: {exc}", flush=True)
            return
        with self._condition:
            self._body = body
            self._gzip_body = gzip_body
            self._built_at = time.monotonic()
            self._error = None
            self._building = False
            self._condition.notify_all()

    def prewarm(self) -> None:
        """Start one non-blocking initial snapshot build."""

        with self._condition:
            if self._building or self._body is not None:
                return
            self._building = True
        threading.Thread(
            target=self._finish_build,
            name="dashboard-payload-prewarm",
            daemon=True,
        ).start()

    def response(self, accept_encoding: str) -> tuple[bytes, str | None]:
        """Return a current snapshot, coalescing simultaneous cold requests."""

        leader = False
        with self._condition:
            age = time.monotonic() - self._built_at
            if self._body is not None and age <= self._ttl_seconds:
                body = self._body
                gzip_body = self._gzip_body
            elif self._body is not None:
                body = self._body
                gzip_body = self._gzip_body
                if not self._building:
                    self._building = True
                    threading.Thread(
                        target=self._finish_build,
                        name="dashboard-payload-refresh",
                        daemon=True,
                    ).start()
            else:
                if not self._building:
                    self._building = True
                    leader = True
                while not leader and self._body is None and self._building:
                    self._condition.wait()
                if not leader:
                    if self._body is None:
                        assert self._error is not None
                        raise self._error
                    body = self._body
                    gzip_body = self._gzip_body
        if leader:
            self._finish_build()
            with self._condition:
                if self._body is None:
                    assert self._error is not None
                    raise self._error
                body = self._body
                gzip_body = self._gzip_body
        if "gzip" in accept_encoding.lower() and gzip_body is not None:
            return gzip_body, "gzip"
        return body, None


def make_handler(
    campaigns: dict[str, Path],
    prices: dict[str, float],
    *,
    modal_h100_price_per_second: float = DEFAULT_MODAL_H100_PRICE_PER_SECOND,
    codex_rate_limit_history: Path = DEFAULT_CODEX_RATE_LIMIT_HISTORY,
    codex_rate_limit_sample_seconds: float = DEFAULT_CODEX_RATE_LIMIT_SAMPLE_SECONDS,
    codex_token_campaign_roots: Iterable[Path] = (),
):
    codex_token_campaign_roots = tuple(codex_token_campaign_roots)

    def build_dashboard_payload() -> bytes:
        return json.dumps(
            dashboard_data(
                campaigns,
                prices,
                modal_h100_price_per_second=modal_h100_price_per_second,
                codex_rate_limit_history=codex_rate_limit_history,
                codex_rate_limit_sample_seconds=codex_rate_limit_sample_seconds,
                codex_token_campaign_roots=codex_token_campaign_roots,
            ),
            separators=(",", ":"),
        ).encode("utf-8")

    payload_cache = DashboardPayloadCache(build_dashboard_payload)
    payload_cache.prewarm()

    class Handler(BaseHTTPRequestHandler):
        def send_payload(
            self,
            body: bytes,
            content_type: str,
            *,
            content_encoding: str | None = None,
        ) -> None:
            if content_encoding is None:
                body, content_encoding = encode_response(
                    body, self.headers.get("Accept-Encoding", "")
                )
            # A reload can abandon an in-flight multi-megabyte response before
            # either headers or body finish. Keep the read-only server healthy.
            with suppress(BrokenPipeError, ConnectionResetError):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Vary", "Accept-Encoding")
                if content_encoding:
                    self.send_header("Content-Encoding", content_encoding)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self.send_payload(
                    read_dashboard_page().encode("utf-8"),
                    "text/html; charset=utf-8",
                )
            elif self.path in {"/science", "/science.html"}:
                self.send_payload(
                    read_science_page().encode("utf-8"),
                    "text/html; charset=utf-8",
                )
            elif self.path == "/api/revision":
                payload = json.dumps(
                    {"revision": dashboard_revision()}, separators=(",", ":")
                ).encode("utf-8")
                self.send_payload(payload, "application/json; charset=utf-8")
            elif self.path == "/api/data":
                payload, content_encoding = payload_cache.response(
                    self.headers.get("Accept-Encoding", "")
                )
                self.send_payload(
                    payload,
                    "application/json; charset=utf-8",
                    content_encoding=content_encoding,
                )
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
        "--semantic-v4-fashion-mnist-native-campaign",
        type=Path,
        default=DEFAULT_SEMANTIC_V4_FASHION_MNIST_NATIVE,
    )
    parser.add_argument(
        "--unified-v3-adderboard-greedy-campaign",
        type=Path,
        default=DEFAULT_UNIFIED_V3_ADDERBOARD_GREEDY,
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
    parser.add_argument(
        "--codex-rate-limit-history",
        type=Path,
        default=DEFAULT_CODEX_RATE_LIMIT_HISTORY,
        help="sanitized local history for the generic Codex primary quota window",
    )
    parser.add_argument(
        "--codex-rate-limit-sample-seconds",
        type=float,
        default=DEFAULT_CODEX_RATE_LIMIT_SAMPLE_SECONDS,
        help="minimum 30 seconds; low-frequency interval for Codex quota reads",
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
        "semantic_v4_fashion_mnist_native": (
            args.semantic_v4_fashion_mnist_native_campaign
        ),
        "unified_v3_adderboard_greedy": (
            args.unified_v3_adderboard_greedy_campaign
        ),
    }
    token_campaign_roots = dashboard_campaign_roots(campaigns)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(
            campaigns,
            prices,
            modal_h100_price_per_second=args.modal_h100_price_per_second,
            codex_rate_limit_history=args.codex_rate_limit_history,
            codex_rate_limit_sample_seconds=max(
                30.0, args.codex_rate_limit_sample_seconds
            ),
            codex_token_campaign_roots=token_campaign_roots,
        ),
    )
    start_codex_rate_limit_sampler(
        args.codex_rate_limit_history,
        sample_seconds=args.codex_rate_limit_sample_seconds,
        token_campaign_roots=token_campaign_roots,
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
    print(
        "Semantic v4 Fashion-MNIST Native OpenEvolve: "
        f"{args.semantic_v4_fashion_mnist_native_campaign}"
    )
    print(
        "Unified v3 AdderBoard Greedy OpenEvolve: "
        f"{args.unified_v3_adderboard_greedy_campaign}"
    )
    print(
        "Codex quota burn-rate history: "
        f"{args.codex_rate_limit_history} "
        f"(every {max(30.0, args.codex_rate_limit_sample_seconds):g} seconds)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

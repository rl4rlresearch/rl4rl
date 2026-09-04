"""Operator-authorized repair tools for unified-v3 trajectory tails.

Three deliberately explicit repair modes are available:

* rewind a deterministic ``infrastructure_interruption`` tail; or
* rewind all proposal records at or after an operator-supplied UTC timestamp;
  or
* rewind selected runs from explicit per-run opportunity cutoffs.

Both modes move removed artifacts and the replaced records out of the active
campaign into a quarantine bundle, rather than silently discarding them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL record at {path}:{line_number}")
        records.append(value)
    return records


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _completed_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("event") == "proposal_completed"]


def _failure_kind(record: dict[str, Any]) -> str | None:
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        return None
    value = evaluation.get("failure_kind")
    return str(value) if value is not None else None


def _parse_timestamp(value: str) -> dt.datetime:
    """Parse an ISO-8601 timestamp as an unambiguous UTC instant."""

    normalized = value.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"invalid ISO-8601 timestamp: {value}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include an explicit timezone: {value}")
    return parsed.astimezone(dt.UTC)


def _record_at_or_after(record: dict[str, Any], boundary: dt.datetime) -> bool:
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, str):
        return False
    return _parse_timestamp(timestamp) >= boundary


def _cutoff(records: list[dict[str, Any]], run_dir: Path) -> int:
    completed = _completed_records(records)
    first = next(
        (
            record
            for record in completed
            if _failure_kind(record) == "infrastructure_interruption"
        ),
        None,
    )
    if first is None:
        raise ValueError(f"no infrastructure-interruption tail: {run_dir}")
    cutoff = first.get("opportunity")
    if not isinstance(cutoff, int) or cutoff < 1:
        raise ValueError(f"invalid interruption opportunity in {run_dir}")
    return cutoff


def _plan_for_cutoff(
    run_dir: Path,
    *,
    cutoff: int,
    mode: str,
    boundary: dt.datetime | None = None,
) -> dict[str, Any]:
    state_path = run_dir / "state.json"
    events_path = run_dir / "events.jsonl"
    state = _read_json(state_path)
    records = _read_jsonl(events_path)
    if cutoff < 1:
        raise ValueError(f"invalid cutoff opportunity in {run_dir}: {cutoff}")
    completed = _completed_records(records)
    kept_completed = [
        record
        for record in completed
        if isinstance(record.get("opportunity"), int)
        and int(record["opportunity"]) < cutoff
    ]
    if not kept_completed or int(kept_completed[-1]["opportunity"]) != cutoff - 1:
        raise ValueError(
            f"cannot establish contiguous pre-repair state for {run_dir}; "
            f"expected completed opportunity {cutoff - 1}"
        )
    removed_completed = [
        record
        for record in completed
        if isinstance(record.get("opportunity"), int)
        and int(record["opportunity"]) >= cutoff
    ]
    removed_started = [
        record
        for record in records
        if record.get("event") == "proposal_started"
        and isinstance(record.get("opportunity"), int)
        and int(record["opportunity"]) >= cutoff
    ]
    if not removed_completed and not removed_started:
        raise ValueError(f"empty removal suffix for {run_dir}")
    if mode == "infrastructure_tail":
        if any(
            bool((record.get("evaluation") or {}).get("valid"))
            or bool(record.get("retained"))
            for record in removed_completed
        ):
            raise ValueError(
                f"repair suffix contains a valid or retained proposal in {run_dir}; "
                "manual adjudication is required"
            )
        if (
            not removed_completed
            or _failure_kind(removed_completed[0])
            != "infrastructure_interruption"
        ):
            raise ValueError(
                f"repair suffix does not begin with interruption: {run_dir}"
            )

    kept_records = [
        record
        for record in records
        if not (
            isinstance(record.get("opportunity"), int)
            and int(record["opportunity"]) >= cutoff
        )
    ]
    parent_counts: Counter[str] = Counter()
    for record in kept_records:
        if record.get("event") != "proposal_started":
            continue
        selected = record.get("selected_parent_ids")
        if (
            isinstance(selected, list)
            and len(selected) == 1
            and isinstance(selected[0], str)
        ):
            parent_counts[selected[0]] += 1

    last = kept_completed[-1]
    usage = last.get("usage_cumulative")
    portfolio = last.get("portfolio_after")
    incumbent = last.get("incumbent_after")
    if (
        not isinstance(usage, dict)
        or not isinstance(portfolio, list)
        or not isinstance(incumbent, str)
    ):
        raise ValueError(f"incomplete final pre-repair event: {run_dir}")
    candidates = state.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError(f"invalid candidates state: {run_dir}")
    future_candidates = [
        candidate_id
        for candidate_id, candidate in candidates.items()
        if isinstance(candidate, dict)
        and isinstance(candidate.get("created_opportunity"), int)
        and int(candidate["created_opportunity"]) >= cutoff
    ]
    if future_candidates and mode == "infrastructure_tail":
        raise ValueError(
            f"repair suffix unexpectedly retained candidates in {run_dir}: "
            + ", ".join(sorted(future_candidates))
        )
    retained_candidate_ids = set(candidates).difference(future_candidates)
    if incumbent not in retained_candidate_ids or any(
        not isinstance(candidate_id, str) or candidate_id not in retained_candidate_ids
        for candidate_id in portfolio
    ):
        raise ValueError(
            f"pre-repair selection references missing candidate: {run_dir}"
        )

    retained_artifact_paths = {
        str(record["artifact_path"])
        for record in kept_completed
        if isinstance(record.get("artifact_path"), str)
    }
    removed_artifact_paths = {
        str(record["artifact_path"])
        for record in removed_completed
        if isinstance(record.get("artifact_path"), str)
    }
    for candidate_id in future_candidates:
        candidate = candidates[candidate_id]
        assert isinstance(candidate, dict)
        artifact_path = candidate.get("artifact_path")
        if artifact_path is None:
            continue
        if not isinstance(artifact_path, str):
            raise ValueError(
                f"invalid candidate artifact path in {run_dir}: {candidate_id}"
            )
        removed_artifact_paths.add(artifact_path)

    # Failed and non-retained proposals are absent from ``state.candidates``,
    # but their immutable candidate snapshots remain on disk. Archive every
    # suffix-only artifact referenced by a removed result. A deduplicated
    # snapshot is retained when a pre-boundary result also references it.
    candidate_artifact_dirs: list[tuple[str, Path]] = []
    for artifact_path in sorted(removed_artifact_paths - retained_artifact_paths):
        relative_path = Path(artifact_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"unsafe candidate artifact path in {run_dir}: {artifact_path}"
            )
        artifact_dir = run_dir / relative_path
        if artifact_dir.exists():
            if not artifact_dir.is_dir():
                raise ValueError(
                    "candidate artifact is not a directory in "
                    f"{run_dir}: {artifact_path}"
                )
            candidate_artifact_dirs.append((artifact_path, artifact_dir))

    opportunity_dirs = [
        path
        for path in sorted((run_dir / "opportunities").iterdir())
        if path.is_dir() and path.name.isdigit() and int(path.name) >= cutoff
    ]
    return {
        "run_dir": run_dir,
        "state": state,
        "records": records,
        "kept_records": kept_records,
        "cutoff": cutoff,
        "last": last,
        "parent_counts": dict(parent_counts),
        "opportunity_dirs": opportunity_dirs,
        "removed_completed": removed_completed,
        "removed_started": removed_started,
        "future_candidate_ids": future_candidates,
        "candidate_artifact_dirs": candidate_artifact_dirs,
        "mode": mode,
        "boundary": boundary,
    }


def _repair_plan(run_dir: Path) -> dict[str, Any]:
    records = _read_jsonl(run_dir / "events.jsonl")
    return _plan_for_cutoff(
        run_dir,
        cutoff=_cutoff(records, run_dir),
        mode="infrastructure_tail",
    )


def _timestamp_rewind_plan(
    run_dir: Path, boundary: dt.datetime
) -> dict[str, Any] | None:
    records = _read_jsonl(run_dir / "events.jsonl")
    post_boundary_opportunities = [
        int(record["opportunity"])
        for record in records
        if record.get("event") == "proposal_started"
        and isinstance(record.get("opportunity"), int)
        and _record_at_or_after(record, boundary)
    ]
    if not post_boundary_opportunities:
        return None
    return _plan_for_cutoff(
        run_dir,
        cutoff=min(post_boundary_opportunities),
        mode="timestamp_rewind",
        boundary=boundary,
    )


def _repaired_state(plan: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(plan["state"])
    last = plan["last"]
    cutoff = int(plan["cutoff"])
    candidates = state["candidates"]
    assert isinstance(candidates, dict)
    for candidate_id in plan["future_candidate_ids"]:
        del candidates[candidate_id]
    counts = plan["parent_counts"]
    assert isinstance(counts, dict)
    for candidate_id, candidate in candidates.items():
        if isinstance(candidate, dict):
            candidate["selected_count"] = int(counts.get(candidate_id, 0))
    usage = last["usage_cumulative"]
    assert isinstance(usage, dict)
    state.update(
        {
            "active": None,
            "status": "running",
            "next_opportunity": cutoff,
            "proposals_used": cutoff - 1,
            "evaluations_used": int(last["evaluator_calls_cumulative"]),
            "evaluator_seconds_used": float(last["evaluator_seconds_cumulative"]),
            "usage": {
                "input_tokens": int(usage.get("input_tokens", 0)),
                "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "reasoning_output_tokens": int(usage.get("reasoning_output_tokens", 0)),
            },
            "incumbent_id": str(last["incumbent_after"]),
            "portfolio_ids": list(last["portfolio_after"]),
            "revision": int(state.get("revision", 0)) + 1,
        }
    )
    return state


def _quarantine_root(campaign: Path) -> Path:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return campaign.parent / "repair-quarantine" / f"{campaign.name}-{timestamp}"


def _manifest(plans: list[dict[str, Any]], *, reason: str) -> dict[str, Any]:
    modes = {str(plan["mode"]) for plan in plans}
    if len(modes) != 1:
        raise ValueError("a repair batch must use exactly one repair mode")
    mode = modes.pop()
    return {
        "schema_version": "1.0",
        "event": f"operator_authorized_{mode}",
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "reason": reason,
        "boundary": (
            plans[0]["boundary"].isoformat()
            if plans and plans[0]["boundary"] is not None
            else None
        ),
        "runs": [
            {
                "run_id": str(plan["state"]["run_id"]),
                "cutoff_opportunity": int(plan["cutoff"]),
                "kept_through_opportunity": int(plan["cutoff"]) - 1,
                "removed_completed_opportunities": [
                    int(record["opportunity"]) for record in plan["removed_completed"]
                ],
                "removed_started_opportunities": [
                    int(record["opportunity"]) for record in plan["removed_started"]
                ],
                "removed_candidate_ids": sorted(plan["future_candidate_ids"]),
                "removed_candidate_artifact_paths": sorted(
                    artifact_path
                    for artifact_path, _path in plan["candidate_artifact_dirs"]
                ),
                "removed_failure_kinds": dict(
                    Counter(
                        _failure_kind(record) or "valid"
                        for record in plan["removed_completed"]
                    )
                ),
                "events_sha256_before": _sha256(plan["run_dir"] / "events.jsonl"),
                "state_sha256_before": _sha256(plan["run_dir"] / "state.json"),
            }
            for plan in plans
        ],
    }


def _quarantine_post_boundary_lifecycle(
    campaign: Path, quarantine: Path, boundary: dt.datetime
) -> list[str]:
    """Archive and remove lifecycle records generated after a timestamp rewind."""

    paths = [
        campaign / "campaign-lifecycle.jsonl",
        campaign / "paired-prefix-events.jsonl",
        campaign / "trajectory-lifecycle.jsonl",
        campaign / "v3-runtime-history.jsonl",
        *sorted((campaign / "runs").glob("*/lifecycle.jsonl")),
    ]
    changed: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        records = _read_jsonl(path)
        retained = [
            record for record in records if not _record_at_or_after(record, boundary)
        ]
        if len(retained) == len(records):
            continue
        relative_path = path.relative_to(campaign)
        destination = quarantine / "lifecycle-before-rewind" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        _atomic_jsonl(path, retained)
        changed.append(str(relative_path))
    return changed


def _apply(campaign: Path, plans: list[dict[str, Any]], *, reason: str) -> Path:
    quarantine = _quarantine_root(campaign)
    quarantine.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(plans, reason=reason)
    _atomic_json(quarantine / "manifest.json", manifest)

    for plan in plans:
        run_dir = plan["run_dir"]
        run_archive = quarantine / str(plan["state"]["run_id"])
        run_archive.mkdir(parents=True, exist_ok=False)
        shutil.copy2(run_dir / "state.json", run_archive / "state-before-repair.json")
        shutil.copy2(
            run_dir / "events.jsonl", run_archive / "events-before-repair.jsonl"
        )
        pause_request = run_dir / "pause-request.json"
        if pause_request.exists():
            shutil.move(str(pause_request), str(run_archive / pause_request.name))
        moved_opportunities = run_archive / "opportunities"
        moved_opportunities.mkdir()
        for opportunity in plan["opportunity_dirs"]:
            shutil.move(str(opportunity), str(moved_opportunities / opportunity.name))
        moved_candidates = run_archive / "candidate-artifacts"
        for relative_path, artifact_dir in plan["candidate_artifact_dirs"]:
            destination = moved_candidates / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(artifact_dir), str(destination))
        _atomic_jsonl(run_dir / "events.jsonl", plan["kept_records"])
        _atomic_json(run_dir / "state.json", _repaired_state(plan))

    boundary = plans[0]["boundary"]
    lifecycle_paths: list[str] = []
    if isinstance(boundary, dt.datetime):
        lifecycle_paths = _quarantine_post_boundary_lifecycle(
            campaign, quarantine, boundary
        )

    _atomic_json(
        quarantine / "manifest.json",
        manifest
        | {
            "completed_at": dt.datetime.now(dt.UTC).isoformat(),
            "state": "quarantined_and_active_campaign_rewound",
            "lifecycle_paths_rewound": lifecycle_paths,
        },
    )
    return quarantine


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--after",
        metavar="ISO_TIMESTAMP",
        help=(
            "rewind every run with a proposal started at or after this "
            "timezone-qualified ISO-8601 timestamp"
        ),
    )
    parser.add_argument(
        "--run-cutoff",
        action="append",
        default=[],
        metavar="RUN_ID=OPPORTUNITY",
        help=(
            "rewind one exact run ID from the given opportunity; repeat for "
            "multiple runs"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the repair; without this flag the command is a dry run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    campaign = args.campaign.resolve()
    if not (campaign / "campaign.json").is_file():
        raise SystemExit(f"campaign is missing: {campaign}")
    run_dirs = sorted(path for path in (campaign / "runs").iterdir() if path.is_dir())
    if args.after is not None and args.run_cutoff:
        raise SystemExit("--after and --run-cutoff are mutually exclusive")
    if args.run_cutoff:
        by_id = {path.name: path for path in run_dirs}
        requested: dict[str, int] = {}
        for value in args.run_cutoff:
            run_id, separator, opportunity_text = value.rpartition("=")
            if not separator or run_id not in by_id:
                raise SystemExit(f"unknown or malformed --run-cutoff: {value}")
            try:
                opportunity = int(opportunity_text)
            except ValueError as error:
                raise SystemExit(f"invalid cutoff opportunity: {value}") from error
            if opportunity < 1 or run_id in requested:
                raise SystemExit(f"invalid or duplicate --run-cutoff: {value}")
            requested[run_id] = opportunity
        plans = [
            _plan_for_cutoff(
                by_id[run_id],
                cutoff=opportunity,
                mode="opportunity_rewind",
            )
            for run_id, opportunity in sorted(requested.items())
        ]
    elif args.after is None:
        plans = [_repair_plan(run_dir) for run_dir in run_dirs]
    else:
        boundary = _parse_timestamp(args.after)
        plans = [
            plan
            for run_dir in run_dirs
            if (plan := _timestamp_rewind_plan(run_dir, boundary)) is not None
        ]
        if not plans:
            raise SystemExit(
                f"no proposal_started records at or after {boundary.isoformat()}"
            )
    summary = _manifest(plans, reason=args.reason)
    if not args.apply:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    quarantine = _apply(campaign, plans, reason=args.reason)
    print(
        json.dumps(summary | {"quarantine": str(quarantine)}, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

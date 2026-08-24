#!/usr/bin/env python3
"""Safely enable the protocol-1.6 post-token-threshold continuation phase."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.c0c3_factorial.artifacts import scientific_runtime_hash  # noqa: E402
from experiments.c0c3_factorial.cli import _load_campaign  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def scheduled_run_dirs(campaign: Path) -> list[Path]:
    schedule = json.loads((campaign / "schedule.json").read_text(encoding="utf-8"))
    return [campaign / "runs" / str(assignment["run_id"]) for assignment in schedule]


def factorial_run_dirs(campaign: Path) -> list[Path]:
    schedule = json.loads((campaign / "schedule.json").read_text(encoding="utf-8"))
    return [
        campaign / "runs" / str(assignment["run_id"])
        for assignment in schedule
        if assignment.get("condition") in {"C0", "C1", "C2", "C3"}
    ]


def validate_transition(campaign: Path, run_dirs: list[Path]) -> int:
    spec, _task, _framework = _load_campaign(campaign)
    if spec.protocol_version != "1.6" or not spec.continues_after_token_threshold:
        raise ValueError("campaign is not using the v1.6 continuation rule")
    threshold = spec.budget.max_total_tokens
    if len(run_dirs) != 12:
        raise ValueError(f"expected twelve C0-C3 runs, found {len(run_dirs)}")
    for run_dir in run_dirs:
        state = read_json(run_dir / "state.json")
        if state.get("active") is not None:
            raise RuntimeError(f"run still has an active opportunity: {run_dir.name}")
        if state.get("status") not in {"completed", "token_threshold_reached"}:
            raise RuntimeError(
                f"run is not stopped at the transition boundary: {run_dir.name}"
            )
        usage = state.get("usage", {})
        total = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
        if total < threshold:
            raise RuntimeError(
                f"run has not reached the token threshold: {run_dir.name} ({total})"
            )
        if state.get("token_budget_continuation_notice_sent", False):
            raise RuntimeError(
                f"run already sent its continuation notice: {run_dir.name}"
            )
    return threshold


def sync_manifest_hashes(
    campaign: Path, runtime_root: Path, *, dry_run: bool
) -> dict[str, Any]:
    """Synchronize campaign-wide manifests without touching any run state."""

    campaign = campaign.resolve()
    runtime_root = runtime_root.resolve()
    _spec, task, framework = _load_campaign(campaign)
    expected_hash = scientific_runtime_hash(
        runtime_root, task=task, framework=framework
    )
    mismatched = [
        run_dir
        for run_dir in scheduled_run_dirs(campaign)
        if read_json(run_dir / "manifest.json").get("scientific_runtime_hash")
        != expected_hash
    ]
    timestamp = utc_now()
    summary = {
        "campaign": str(campaign),
        "dry_run": dry_run,
        "expected_scientific_runtime_hash": expected_hash,
        "mismatched_manifest_count": len(mismatched),
        "mismatched_run_ids": [run_dir.name for run_dir in mismatched],
    }
    if dry_run or not mismatched:
        return summary

    backup_root = (
        campaign
        / "operator-amendments"
        / f"{timestamp.replace(':', '-')}-manifest-sync"
    )
    backup_root.mkdir(parents=True, exist_ok=False)
    for run_dir in mismatched:
        manifest_path = run_dir / "manifest.json"
        shutil.copy2(manifest_path, backup_root / f"{run_dir.name}.json")
        manifest = read_json(manifest_path)
        manifest["scientific_runtime_hash"] = expected_hash
        atomic_json(manifest_path, manifest)
    append_jsonl(
        campaign / "protocol-amendments.jsonl",
        {
            "schema_version": "1.0",
            "event": "campaign_manifest_hashes_synchronized",
            "timestamp": timestamp,
            "run_ids": [run_dir.name for run_dir in mismatched],
            "scientific_runtime_hash": expected_hash,
            "run_state_changed": False,
        },
    )
    return summary


def repair_post_notice_completion(
    campaign: Path, runtime_root: Path, *, dry_run: bool
) -> dict[str, Any]:
    """Reopen v1.6 runs stopped by the post-notice token-guard defect."""

    campaign = campaign.resolve()
    runtime_root = runtime_root.resolve()
    all_run_dirs = scheduled_run_dirs(campaign)
    run_dirs = factorial_run_dirs(campaign)
    spec, task, framework = _load_campaign(campaign)
    if spec.protocol_version != "1.6" or not spec.continues_after_token_threshold:
        raise ValueError("campaign is not using the v1.6 continuation rule")
    if len(run_dirs) != 12:
        raise ValueError(f"expected twelve C0-C3 runs, found {len(run_dirs)}")
    for run_dir in run_dirs:
        state = read_json(run_dir / "state.json")
        if state.get("status") != "completed" or state.get("active") is not None:
            raise RuntimeError(f"run is not safely stopped: {run_dir.name}")
        if not state.get("token_budget_continuation_notice_sent", False):
            raise RuntimeError(f"continuation notice was not sent: {run_dir.name}")
        usage = state.get("usage", {})
        tokens = int(usage.get("input_tokens", 0)) + int(
            usage.get("output_tokens", 0)
        )
        if tokens < spec.budget.max_total_tokens:
            raise RuntimeError(f"run is below the token threshold: {run_dir.name}")
        if int(state.get("next_opportunity", 0)) > spec.budget.proposals:
            raise RuntimeError(f"proposal budget is exhausted: {run_dir.name}")
        if int(state.get("evaluator_calls_used", 0)) >= (
            spec.budget.candidate_evaluations
        ):
            raise RuntimeError(f"evaluation budget is exhausted: {run_dir.name}")
        if float(state.get("evaluator_seconds_used", 0.0)) >= (
            spec.budget.max_evaluator_seconds
        ):
            raise RuntimeError(f"evaluator-time budget is exhausted: {run_dir.name}")

    campaign_record = read_json(campaign / "campaign.json")
    old_runtime_hash = str(campaign_record["scientific_runtime_hash"])
    new_runtime_hash = scientific_runtime_hash(
        runtime_root, task=task, framework=framework
    )
    timestamp = utc_now()
    summary = {
        "campaign": str(campaign),
        "dry_run": dry_run,
        "run_count": len(run_dirs),
        "old_scientific_runtime_hash": old_runtime_hash,
        "new_scientific_runtime_hash": new_runtime_hash,
    }
    if dry_run:
        return summary

    repair_root = (
        campaign
        / "operator-amendments"
        / f"{timestamp.replace(':', '-')}-post-notice-guard-repair"
    )
    states_root = repair_root / "states-before"
    manifests_root = repair_root / "manifests-before"
    states_root.mkdir(parents=True, exist_ok=False)
    manifests_root.mkdir()
    shutil.copy2(campaign / "campaign.json", repair_root / "campaign.json.before")
    for run_dir in run_dirs:
        shutil.copy2(run_dir / "state.json", states_root / f"{run_dir.name}.json")
    for run_dir in all_run_dirs:
        shutil.copy2(
            run_dir / "manifest.json", manifests_root / f"{run_dir.name}.json"
        )

    campaign_record["scientific_runtime_hash"] = new_runtime_hash
    atomic_json(campaign / "campaign.json", campaign_record)
    for run_dir in all_run_dirs:
        manifest = read_json(run_dir / "manifest.json")
        manifest["scientific_runtime_hash"] = new_runtime_hash
        atomic_json(run_dir / "manifest.json", manifest)
    campaign_event = {
        "schema_version": "1.0",
        "event": "post_threshold_completion_guard_corrected",
        "timestamp": timestamp,
        "affected_run_ids": [run_dir.name for run_dir in run_dirs],
        "old_scientific_runtime_hash": old_runtime_hash,
        "new_scientific_runtime_hash": new_runtime_hash,
        "preserved_completed_opportunities": True,
    }
    append_jsonl(campaign / "protocol-amendments.jsonl", campaign_event)
    atomic_json(repair_root / "repair.json", campaign_event)
    for run_dir in run_dirs:
        state = read_json(run_dir / "state.json")
        state["status"] = "running"
        state["revision"] = int(state.get("revision", 0)) + 1
        atomic_json(run_dir / "state.json", state)
        run_event = {
            "schema_version": "1.0",
            "event": "post_threshold_completion_reopened",
            "timestamp": timestamp,
            "run_id": run_dir.name,
            "next_opportunity": state["next_opportunity"],
            "continuation_notice_already_sent": True,
        }
        append_jsonl(run_dir / "events.jsonl", run_event)
        append_jsonl(run_dir / "lifecycle.jsonl", run_event)
    return summary


def amend(campaign: Path, runtime_root: Path, *, dry_run: bool) -> dict[str, Any]:
    campaign = campaign.resolve()
    runtime_root = runtime_root.resolve()
    all_run_dirs = scheduled_run_dirs(campaign)
    run_dirs = factorial_run_dirs(campaign)
    threshold = validate_transition(campaign, run_dirs)
    _spec, task, framework = _load_campaign(campaign)
    campaign_record = read_json(campaign / "campaign.json")
    old_runtime_hash = str(campaign_record["scientific_runtime_hash"])
    new_runtime_hash = scientific_runtime_hash(
        runtime_root, task=task, framework=framework
    )
    timestamp = utc_now()
    summary = {
        "campaign": str(campaign),
        "dry_run": dry_run,
        "run_count": len(run_dirs),
        "token_threshold": threshold,
        "old_scientific_runtime_hash": old_runtime_hash,
        "new_scientific_runtime_hash": new_runtime_hash,
    }
    if dry_run:
        return summary

    amendment_root = campaign / "operator-amendments" / timestamp.replace(":", "-")
    amendment_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(campaign / "campaign.json", amendment_root / "campaign.json.before")
    shutil.copy2(
        campaign / "inputs/protocol.json", amendment_root / "protocol.json.before"
    )
    states_root = amendment_root / "states-before"
    manifests_root = amendment_root / "manifests-before"
    states_root.mkdir()
    manifests_root.mkdir()
    for run_dir in run_dirs:
        shutil.copy2(run_dir / "state.json", states_root / f"{run_dir.name}.json")
    for run_dir in all_run_dirs:
        shutil.copy2(
            run_dir / "manifest.json", manifests_root / f"{run_dir.name}.json"
        )

    campaign_record["scientific_runtime_hash"] = new_runtime_hash
    atomic_json(campaign / "campaign.json", campaign_record)
    amendment_event = {
        "schema_version": "1.0",
        "event": "protocol_behavior_updated",
        "timestamp": timestamp,
        "protocol_version": "1.6",
        "affected_run_ids": [run_dir.name for run_dir in run_dirs],
        "boundary": "first opportunity after each run crossed 500M reported tokens",
        "old_behavior": "500M reported tokens ended the trajectory",
        "new_behavior": (
            "500M is a subject-visible phase threshold; return once, resume with "
            "one continuation notice, then omit token-budget language"
        ),
        "authorized_by": "human operator",
        "old_scientific_runtime_hash": old_runtime_hash,
        "new_scientific_runtime_hash": new_runtime_hash,
    }
    append_jsonl(campaign / "protocol-amendments.jsonl", amendment_event)
    atomic_json(amendment_root / "amendment.json", amendment_event)

    for run_dir in all_run_dirs:
        manifest = read_json(run_dir / "manifest.json")
        manifest["scientific_runtime_hash"] = new_runtime_hash
        atomic_json(run_dir / "manifest.json", manifest)
    for run_dir in run_dirs:
        state = read_json(run_dir / "state.json")
        state["status"] = "token_threshold_reached"
        state["token_budget_continuation_notice_sent"] = False
        state["revision"] = int(state.get("revision", 0)) + 1
        atomic_json(run_dir / "state.json", state)
        run_event = {
            "schema_version": "1.0",
            "event": "token_threshold_continuation_authorized",
            "timestamp": timestamp,
            "run_id": run_dir.name,
            "next_opportunity": state["next_opportunity"],
            "token_threshold": threshold,
            "tokens_used": int(state["usage"].get("input_tokens", 0))
            + int(state["usage"].get("output_tokens", 0)),
        }
        append_jsonl(run_dir / "events.jsonl", run_event)
        append_jsonl(run_dir / "lifecycle.jsonl", run_event)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sync-manifests-only", action="store_true")
    mode.add_argument("--repair-post-notice-completion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sync_manifests_only:
        result = sync_manifest_hashes(
            args.campaign,
            args.runtime_root,
            dry_run=not args.apply,
        )
    elif args.repair_post_notice_completion:
        result = repair_post_notice_completion(
            args.campaign,
            args.runtime_root,
            dry_run=not args.apply,
        )
    else:
        result = amend(
            args.campaign,
            args.runtime_root,
            dry_run=not args.apply,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

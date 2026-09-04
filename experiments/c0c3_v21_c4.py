#!/usr/bin/env python3
"""Add the periodic full-refresh C4 arm to a Greedy OpenEvolve v2.1 campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c0c3_factorial.cli import _load_campaign  # noqa: E402
from experiments.c0c3_factorial.spec import C4_CONDITION  # noqa: E402
from experiments.c0c3_factorial.state import (  # noqa: E402
    Candidate,
    SearchController,
    append_jsonl,
    atomic_json,
    utc_now,
)

C4_SCHEDULE = Path("v2-1-c4-schedule.json")
C4_POLICY = {
    "schema_version": "1.0",
    "condition": "C4",
    "base_search_state": "single_incumbent",
    "base_proposal_policy": "ordinary",
    "policy": "incumbent_preserving_full_search_refresh",
    "interval_proposals": 10,
    "first_refresh_opportunity": 11,
    "refresh_opportunities": list(range(11, 201, 10)),
    "preserve": [
        "incumbent_artifact",
        "private_cumulative_accounting",
        "private_audit_history",
        "evaluator_seed",
    ],
    "clear": [
        "subject_visible_outcomes",
        "mechanism_history",
        "candidate_population",
        "parent_history",
        "conversation_binding",
        "search_seed",
    ],
}


def _prelaunch_failure_is_discardable(run_dir: Path) -> None:
    """Reject cleanup once any scientific subject/evaluator work has occurred."""

    state = _read(run_dir / "state.json")
    usage = state.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError(f"C4 state has invalid usage accounting: {run_dir}")
    nonzero_usage = any(
        int(usage.get(key, 0)) != 0
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        )
    )
    if (
        nonzero_usage
        or int(state.get("evaluations_used", 0)) != 0
        or float(state.get("evaluator_seconds_used", 0.0)) != 0.0
    ):
        raise RuntimeError(
            "refusing to clear a C4 run after subject or evaluator work: "
            f"{run_dir.name}"
        )
    events_path = run_dir / "events.jsonl"
    for line in events_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        kind = event.get("event")
        if kind in {"run_created", "proposal_started"}:
            continue
        evaluation = event.get("evaluation")
        if (
            kind != "proposal_completed"
            or not isinstance(evaluation, dict)
            or evaluation.get("failure_kind") != "infrastructure_interruption"
        ):
            raise RuntimeError(
                "refusing to clear non-infrastructure C4 history: "
                f"{run_dir.name}"
            )


def repair_prelaunch_runs(campaign_dir: str | Path, *, reason: str) -> dict[str, Any]:
    """Recreate C4 runs after a verified zero-work prelaunch failure."""

    campaign = Path(campaign_dir).expanduser().resolve()
    if not reason.strip():
        raise ValueError("repair reason cannot be blank")
    with _lock(campaign / ".v2-1-c4-amendment.lock"):
        spec, _task, _framework = _load_campaign(campaign)
        campaign_manifest = _read(campaign / "campaign.json")
        base_schedule = _read(campaign / "schedule.json")
        assignments = _read(campaign / C4_SCHEDULE)
        if not isinstance(base_schedule, list) or not isinstance(assignments, list):
            raise ValueError("campaign schedules must be lists")
        sources: dict[int, dict[str, Any]] = {}
        for row in base_schedule:
            if isinstance(row, dict) and row.get("condition") != "N0":
                sources.setdefault(int(row["block"]), dict(row))
        for assignment in assignments:
            if not isinstance(assignment, dict):
                raise ValueError("C4 schedule contains a non-object assignment")
            _prelaunch_failure_is_discardable(
                campaign / "runs" / str(assignment["run_id"])
            )

        backup = Path(tempfile.mkdtemp(prefix="rl4rl-c4-prelaunch-repair-"))
        moved: list[tuple[Path, Path]] = []
        guard_hash = _guard_hash()
        try:
            for assignment in assignments:
                destination = campaign / "runs" / str(assignment["run_id"])
                saved = backup / destination.name
                shutil.move(str(destination), saved)
                moved.append((destination, saved))
                _create_run(
                    campaign,
                    assignment=dict(assignment),
                    source_assignment=sources[int(assignment["block"])],
                    spec=spec,
                    campaign_manifest=campaign_manifest,
                    guard_hash=guard_hash,
                )
        except BaseException:
            for destination, saved in reversed(moved):
                if destination.exists():
                    shutil.rmtree(destination)
                if saved.exists():
                    shutil.move(str(saved), destination)
            raise

        receipt_path = campaign / "v2-1-c4-amendment.json"
        receipt = _read(receipt_path)
        receipt["policy"] = C4_POLICY
        receipt["c4_guard_sha256"] = guard_hash
        receipt["prelaunch_repair"] = {
            "timestamp": utc_now(),
            "reason": reason.strip(),
            "discarded_subject_tokens": 0,
            "discarded_evaluator_calls": 0,
            "replacement_runs_started": False,
        }
        atomic_json(receipt_path, receipt)
        append_jsonl(
            campaign / "campaign-amendments.jsonl",
            {
                "schema_version": "1.0",
                "event": "c4_zero_work_prelaunch_repaired",
                "timestamp": utc_now(),
                "reason": reason.strip(),
                "run_ids": [row["run_id"] for row in assignments],
                "c4_guard_sha256": guard_hash,
                "discarded_subject_tokens": 0,
                "discarded_evaluator_calls": 0,
            },
        )
        return {
            "campaign": str(campaign),
            "status": "repaired",
            "run_ids": [row["run_id"] for row in assignments],
            "backup": str(backup),
            "c4_guard_sha256": guard_hash,
        }


def register_guard_update(campaign_dir: str | Path, *, reason: str) -> dict[str, Any]:
    """Register a guard update after every C4 controller reaches a safe boundary."""

    campaign = Path(campaign_dir).expanduser().resolve()
    if not reason.strip():
        raise ValueError("guard update reason cannot be blank")
    with _lock(campaign / ".v2-1-c4-amendment.lock"):
        assignments = _read(campaign / C4_SCHEDULE)
        if not isinstance(assignments, list):
            raise ValueError("C4 extension schedule must be a list")
        next_opportunities: dict[str, int] = {}
        guard_hash = _guard_hash()
        for assignment in assignments:
            run_id = str(assignment["run_id"])
            run_dir = campaign / "runs" / run_id
            state = _read(run_dir / "state.json")
            if state.get("active") is not None:
                raise RuntimeError(
                    f"cannot update the C4 guard during an active proposal: {run_id}"
                )
            manifest_path = run_dir / "manifest.json"
            manifest = _read(manifest_path)
            manifest["periodic_full_refresh"] = C4_POLICY
            manifest["c4_guard_sha256"] = guard_hash
            atomic_json(manifest_path, manifest)
            next_opportunities[run_id] = int(state["proposals_used"]) + 1

        receipt_path = campaign / "v2-1-c4-amendment.json"
        receipt = _read(receipt_path)
        receipt["policy"] = C4_POLICY
        receipt["c4_guard_sha256"] = guard_hash
        atomic_json(receipt_path, receipt)
        event = {
            "schema_version": "1.0",
            "event": "c4_guard_updated_at_safe_boundaries",
            "timestamp": utc_now(),
            "reason": reason.strip(),
            "first_opportunity_using_guard": next_opportunities,
            "c4_guard_sha256": guard_hash,
        }
        append_jsonl(campaign / "campaign-amendments.jsonl", event)
        return {"campaign": str(campaign), "status": "updated", **event}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _guard_hash() -> str:
    source = REPO_ROOT / "experiments/c0c3_v21_c4_guard.py"
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _new_assignment(source: dict[str, Any]) -> dict[str, Any]:
    source_id = str(source["run_id"])
    stem = source_id.rsplit("-", 1)[0]
    return {
        "block": int(source["block"]),
        "order": 5,
        "condition": "C4",
        "run_seed": int(source["run_seed"]),
        "run_id": f"{stem}-c4",
    }


def _create_run(
    campaign: Path,
    *,
    assignment: dict[str, Any],
    source_assignment: dict[str, Any],
    spec: Any,
    campaign_manifest: dict[str, Any],
    guard_hash: str,
) -> None:
    destination = campaign / "runs" / str(assignment["run_id"])
    if destination.exists():
        state = _read(destination / "state.json")
        manifest = _read(destination / "manifest.json")
        if (
            state.get("condition") != "C4"
            or manifest.get("assignment") != assignment
            or manifest.get("c4_guard_sha256") != guard_hash
        ):
            raise RuntimeError(f"existing C4 run is inconsistent: {destination}")
        return

    source_run = campaign / "runs" / str(source_assignment["run_id"])
    source_state = _read(source_run / "state.json")
    source_manifest = _read(source_run / "manifest.json")
    seed_id = str(campaign_manifest["seed_candidate_id"])
    seed_payload = dict(source_state["candidates"][seed_id])
    seed = Candidate(**seed_payload)
    if seed.parent_ids:
        raise RuntimeError("campaign seed unexpectedly has parents")

    runs_root = campaign / "runs"
    temporary = Path(tempfile.mkdtemp(prefix=".c4-run-", dir=runs_root))
    try:
        # SearchController expects to create its destination itself.
        temporary.rmdir()
        SearchController.create(
            temporary,
            spec,
            run_id=str(assignment["run_id"]),
            condition=C4_CONDITION,
            seed_candidate=seed,
        )
        shutil.copytree(source_run / "task-support", temporary / "task-support")
        (temporary / "candidates").mkdir(exist_ok=True)
        shutil.copytree(
            source_run / "candidates" / seed_id,
            temporary / "candidates" / seed_id,
        )
        run_manifest = {
            key: value
            for key, value in source_manifest.items()
            if key not in {"assignment", "semantic_intervention"}
        }
        run_manifest.update(
            {
                "assignment": assignment,
                "periodic_full_refresh": C4_POLICY,
                "protocol_amendment": "greedy_openevolve_v2_1_c4",
                "c4_guard_sha256": guard_hash,
            }
        )
        atomic_json(temporary / "manifest.json", run_manifest)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def extend_campaign(campaign_dir: str | Path, *, reason: str) -> dict[str, Any]:
    campaign = Path(campaign_dir).expanduser().resolve()
    if not reason.strip():
        raise ValueError("amendment reason cannot be blank")
    with _lock(campaign / ".v2-1-c4-amendment.lock"):
        spec, _task, framework = _load_campaign(campaign)
        if spec.protocol_version != "2.1":
            raise ValueError("C4 amendment requires protocol 2.1")
        if str(framework.framework_id) != "openevolve":
            raise ValueError("C4 amendment is only for Greedy OpenEvolve")
        campaign_manifest = _read(campaign / "campaign.json")
        base_schedule = _read(campaign / "schedule.json")
        if not isinstance(base_schedule, list):
            raise ValueError("campaign schedule must be a list")
        existing = (
            _read(campaign / C4_SCHEDULE)
            if (campaign / C4_SCHEDULE).is_file()
            else []
        )
        if not isinstance(existing, list):
            raise ValueError("C4 extension schedule must be a list")
        by_block: dict[int, dict[str, Any]] = {}
        for row in base_schedule:
            if not isinstance(row, dict) or row.get("condition") == "N0":
                continue
            by_block.setdefault(int(row["block"]), dict(row))
        guard_hash = _guard_hash()
        assignments = [_new_assignment(by_block[block]) for block in sorted(by_block)]
        existing_by_id = {
            str(row["run_id"]): row for row in existing if isinstance(row, dict)
        }
        for assignment in assignments:
            prior = existing_by_id.get(str(assignment["run_id"]))
            if prior is not None and prior != assignment:
                raise RuntimeError(
                    f"existing C4 assignment changed: {assignment['run_id']}"
                )
            _create_run(
                campaign,
                assignment=assignment,
                source_assignment=by_block[int(assignment["block"])],
                spec=spec,
                campaign_manifest=campaign_manifest,
                guard_hash=guard_hash,
            )
        atomic_json(campaign / C4_SCHEDULE, assignments)
        receipt_path = campaign / "v2-1-c4-amendment.json"
        if receipt_path.is_file():
            receipt = _read(receipt_path)
            if (
                receipt.get("run_ids") != [row["run_id"] for row in assignments]
                or receipt.get("c4_guard_sha256") != guard_hash
            ):
                raise RuntimeError("existing C4 amendment receipt is inconsistent")
            return receipt
        receipt = {
            "schema_version": "1.0",
            "event": "greedy_openevolve_v2_1_c4_added",
            "timestamp": utc_now(),
            "reason": reason.strip(),
            "campaign": str(campaign),
            "policy": C4_POLICY,
            "run_ids": [row["run_id"] for row in assignments],
            "blocks": [row["block"] for row in assignments],
            "base_schedule_preserved": True,
            "schedule_extension": C4_SCHEDULE.as_posix(),
            "c4_guard_sha256": guard_hash,
        }
        atomic_json(receipt_path, receipt)
        append_jsonl(campaign / "campaign-amendments.jsonl", receipt)
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repair-zero-work-prelaunch",
        action="store_true",
        help="recreate C4 runs after verified zero-token/evaluation launch failures",
    )
    parser.add_argument(
        "--register-guard-update",
        action="store_true",
        help="update C4 guard hashes after every run reaches a safe boundary",
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.repair_zero_work_prelaunch and args.register_guard_update:
        parser.error("repair and guard-update modes are mutually exclusive")
    if args.repair_zero_work_prelaunch:
        handler = repair_prelaunch_runs
    elif args.register_guard_update:
        handler = register_guard_update
    else:
        handler = extend_campaign
    print(
        json.dumps(
            handler(args.campaign, reason=args.reason),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

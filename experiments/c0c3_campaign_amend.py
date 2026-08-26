#!/usr/bin/env python3
"""Apply audited, append-only C0-C3 campaign amendments.

This module intentionally lives outside ``experiments.c0c3_factorial``.  The
controller package is part of a campaign's scientific-runtime hash, while this
is an operator tool for creating a documented schedule extension without
rewriting any previously recorded trajectory artifact.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.c0c3_factorial.cli import _load_campaign  # noqa: E402
from experiments.c0c3_factorial.spec import (  # noqa: E402
    Condition,
    FrameworkSpec,
    TaskSpec,
    make_assignments,
    sha256_json,
)
from experiments.c0c3_factorial.state import (  # noqa: E402
    Candidate,
    SearchController,
    append_jsonl,
    atomic_json,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _schedule(campaign: Path) -> list[dict[str, object]]:
    payload = json.loads((campaign / "schedule.json").read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError("campaign schedule must be a JSON list of objects")
    return [dict(row) for row in payload]


def _assignment_dict(assignment: object) -> dict[str, object]:
    row = asdict(assignment)
    row["condition"] = assignment.condition.value  # type: ignore[attr-defined]
    return row


def _block_rows(
    spec: object, task: TaskSpec, framework: FrameworkSpec, block: int
) -> list[dict[str, object]]:
    rows = [
        _assignment_dict(assignment)
        for assignment in make_assignments(
            spec,  # type: ignore[arg-type]
            task_id=task.task_id,
            framework_id=framework.framework_id.value,
        )
        if assignment.block == block
    ]
    if len(rows) != len(Condition):
        raise ValueError(
            f"deterministic assignment generation failed for block {block}"
        )
    return sorted(rows, key=lambda row: int(row["order"]))


def _find_reference_run(
    campaign: Path, schedule: list[dict[str, object]]
) -> tuple[Path, dict[str, Any]]:
    for row in schedule:
        if str(row.get("condition")) not in {
            condition.value for condition in Condition
        }:
            continue
        run_dir = campaign / "runs" / str(row["run_id"])
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file():
            return run_dir, _read_json(manifest_path)
    raise ValueError("campaign has no factorial run manifest to clone as the seed")


def _verify_or_create_run(
    *,
    campaign: Path,
    run_dir: Path,
    assignment: dict[str, object],
    base_spec: object,
    task: TaskSpec,
    campaign_manifest: dict[str, Any],
    reference_run: Path,
    reference_manifest: dict[str, Any],
) -> None:
    """Create one new ready run or verify an interrupted amendment's run clone."""

    if run_dir.exists():
        manifest = _read_json(run_dir / "manifest.json")
        if manifest.get("assignment") != assignment:
            raise ValueError(
                f"existing extension run has a different assignment: {run_dir}"
            )
        controller = SearchController.load(run_dir, base_spec)  # type: ignore[arg-type]
        if (
            controller.state.status != "ready"
            or controller.state.proposals_used != 0
            or controller.state.evaluations_used != 0
            or controller.state.active is not None
        ):
            raise ValueError(
                "refusing to amend an already-started extension trajectory: "
                f"{run_dir.name}"
            )
        return

    seed_candidate_id = str(campaign_manifest.get("seed_candidate_id", ""))
    if not seed_candidate_id:
        raise ValueError("campaign manifest lacks seed_candidate_id")
    baseline = reference_manifest.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("reference run manifest lacks a baseline record")
    fitness = baseline.get("fitness")
    metrics = baseline.get("metrics")
    if not isinstance(fitness, int | float) or isinstance(fitness, bool):
        raise ValueError("reference baseline lacks numeric fitness")
    if not isinstance(metrics, dict):
        raise ValueError("reference baseline lacks metrics")

    seed = Candidate(
        candidate_id=seed_candidate_id,
        parent_ids=[],
        fitness=float(fitness),
        metrics=dict(metrics),
        artifact_path=f"candidates/{seed_candidate_id}",
        hypothesis="frozen seed baseline",
        intended_edit="none",
        created_opportunity=0,
        retained_order=0,
    )
    condition = Condition(str(assignment["condition"]))
    SearchController.create(
        run_dir,
        base_spec,  # type: ignore[arg-type]
        run_id=str(assignment["run_id"]),
        condition=condition,
        seed_candidate=seed,
    )
    shutil.copytree(reference_run / "task-support", run_dir / "task-support")
    source_candidate = reference_run / "candidates" / seed_candidate_id
    if not source_candidate.is_dir():
        raise FileNotFoundError(
            f"reference seed candidate is missing: {source_candidate}"
        )
    destination_candidate = run_dir / "candidates" / seed_candidate_id
    destination_candidate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_candidate, destination_candidate)
    atomic_json(
        run_dir / "manifest.json",
        {
            "schema_version": "1.0",
            "assignment": assignment,
            "protocol_hash": base_spec.protocol_hash,  # type: ignore[attr-defined]
            "task_hash": campaign_manifest["task_hash"],
            "framework_hash": campaign_manifest["framework_hash"],
            "scientific_runtime_hash": campaign_manifest["scientific_runtime_hash"],
            "baseline": baseline,
            "repo_revision": reference_manifest.get("repo_revision", "unavailable"),
        },
    )


@contextmanager
def _amendment_lock(campaign: Path) -> Iterator[None]:
    path = campaign / ".campaign-amendment.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"campaign amendment is already active: {campaign}"
            ) from error
        yield


def extend_campaign_blocks(
    campaign_dir: str | Path,
    *,
    target_blocks: int,
    reason: str,
    authorization: str = "operator-authorized",
) -> dict[str, object]:
    """Append complete C0-C3 blocks while leaving prior run artifacts untouched.

    The campaign's frozen input protocol remains its original controller
    contract.  ``campaign-amendments.jsonl`` records the higher effective
    schedule size; the new trajectories use the identical frozen base hash,
    task support, seed candidate, evaluator, and Codex configuration.
    """

    if target_blocks < 1:
        raise ValueError("target_blocks must be positive")
    if not reason.strip():
        raise ValueError("amendment reason cannot be blank")
    if not authorization.strip():
        raise ValueError("amendment authorization cannot be blank")

    campaign = Path(campaign_dir).expanduser().resolve()
    spec, task, framework = _load_campaign(campaign)
    if not spec.c0c3_only:
        raise ValueError("block expansion currently supports C0-C3-only campaigns")

    with _amendment_lock(campaign):
        schedule = _schedule(campaign)
        manifest = _read_json(campaign / "campaign.json")
        if (campaign / "sealed-layer-b").exists() or (
            campaign / "sealed-layer-c"
        ).exists():
            raise ValueError("cannot expand a campaign after Layer B/C has been sealed")
        current_blocks = max((int(row["block"]) for row in schedule), default=0)
        if target_blocks < current_blocks:
            raise ValueError(
                "target blocks "
                f"{target_blocks} is below current scheduled blocks {current_blocks}"
            )
        if target_blocks == current_blocks:
            return {
                "campaign": str(campaign),
                "status": "already-expanded",
                "effective_blocks": current_blocks,
                "added_run_ids": [],
            }
        if target_blocks <= spec.blocks:
            raise ValueError(
                "target must add a block beyond the base protocol's declared blocks"
            )

        extended_spec = replace(spec, blocks=target_blocks)
        expected_existing = {
            row["run_id"]: row
            for block in range(1, current_blocks + 1)
            for row in _block_rows(extended_spec, task, framework, block)
        }
        actual_existing = {row["run_id"]: row for row in schedule}
        if set(actual_existing) != set(expected_existing):
            raise ValueError(
                "existing schedule does not match deterministic base assignments; "
                "refusing to append blocks"
            )
        for run_id, expected in expected_existing.items():
            if actual_existing[run_id] != expected:
                raise ValueError(
                    "existing schedule assignment differs from deterministic base: "
                    f"{run_id}"
                )

        additions = [
            row
            for block in range(current_blocks + 1, target_blocks + 1)
            for row in _block_rows(extended_spec, task, framework, block)
        ]
        if any((campaign / "runs" / str(row["run_id"])).exists() for row in additions):
            raise ValueError(
                "an added run directory already exists before schedule expansion"
            )

        reference_run, reference_manifest = _find_reference_run(campaign, schedule)
        old_schedule_hash = sha256_json(schedule)
        old_manifest_bytes = (campaign / "campaign.json").read_bytes()
        amendment_suffix = hashlib.sha256(
            (old_schedule_hash + str(target_blocks)).encode()
        ).hexdigest()[:12]
        amendment_id = (
            f"block-expansion-{current_blocks:02d}-to-{target_blocks:02d}-"
            f"{amendment_suffix}"
        )
        amendment_dir = campaign / "amendments" / amendment_id
        amendment_dir.mkdir(parents=True, exist_ok=False)
        (amendment_dir / "campaign.json.before").write_bytes(old_manifest_bytes)
        atomic_json(amendment_dir / "schedule.before.json", schedule)

        for assignment in additions:
            _verify_or_create_run(
                campaign=campaign,
                run_dir=campaign / "runs" / str(assignment["run_id"]),
                assignment=assignment,
                base_spec=spec,
                task=task,
                campaign_manifest=manifest,
                reference_run=reference_run,
                reference_manifest=reference_manifest,
            )

        updated_schedule = sorted(
            [*schedule, *additions],
            key=lambda row: (int(row["block"]), int(row["order"])),
        )
        atomic_json(campaign / "schedule.json", updated_schedule)
        atomic_json(amendment_dir / "schedule.after.json", updated_schedule)
        added_run_ids = [str(row["run_id"]) for row in additions]
        existing_run_ids = [str(row["run_id"]) for row in schedule]
        amendment_summary = {
            "amendment_id": amendment_id,
            "kind": "append_factorial_blocks",
            "timestamp": _utc_now(),
            "authorization": authorization.strip(),
            "reason": reason.strip(),
            "base_protocol_hash": spec.protocol_hash,
            "base_declared_blocks": spec.blocks,
            "previous_effective_blocks": current_blocks,
            "effective_blocks": target_blocks,
            "added_blocks": list(range(current_blocks + 1, target_blocks + 1)),
            "added_run_ids": added_run_ids,
            "unmodified_existing_run_ids": existing_run_ids,
            "first_affected_opportunity": {run_id: 1 for run_id in added_run_ids},
            "old_schedule_sha256": old_schedule_hash,
            "new_schedule_sha256": sha256_json(updated_schedule),
            "artifacts": {
                "campaign_before": str(
                    (amendment_dir / "campaign.json.before").relative_to(campaign)
                ),
                "schedule_before": str(
                    (amendment_dir / "schedule.before.json").relative_to(campaign)
                ),
                "schedule_after": str(
                    (amendment_dir / "schedule.after.json").relative_to(campaign)
                ),
            },
        }
        amendments = manifest.setdefault("amendments", [])
        if not isinstance(amendments, list):
            raise ValueError("campaign manifest amendments field is not a list")
        amendments.append(amendment_summary)
        manifest["run_count"] = len(updated_schedule)
        manifest["primary_run_ids"] = [str(row["run_id"]) for row in updated_schedule]
        manifest["optional_run_ids"] = []
        manifest["effective_blocks"] = target_blocks
        atomic_json(campaign / "campaign.json", manifest)
        append_jsonl(
            campaign / "campaign-amendments.jsonl",
            {
                "schema_version": "1.0",
                "event": "campaign_block_expanded",
                **amendment_summary,
            },
        )
        return {
            "campaign": str(campaign),
            "status": "expanded",
            "amendment_id": amendment_id,
            "base_declared_blocks": spec.blocks,
            "effective_blocks": target_blocks,
            "added_run_ids": added_run_ids,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extend = subparsers.add_parser("extend-blocks", help="append complete C0-C3 blocks")
    extend.add_argument("--campaign", type=Path, required=True)
    extend.add_argument("--target-blocks", type=int, required=True)
    extend.add_argument("--reason", required=True)
    extend.add_argument("--authorization", default="operator-authorized")
    extend.set_defaults(
        handler=lambda args: extend_campaign_blocks(
            args.campaign,
            target_blocks=args.target_blocks,
            reason=args.reason,
            authorization=args.authorization,
        )
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

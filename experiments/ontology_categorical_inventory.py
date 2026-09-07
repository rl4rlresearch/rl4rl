"""Build source inventories for the categorical ontology-fingerprint review.

The inventories are intentionally fingerprint-free.  They give one campaign
fork a stable, scope-limited list of source candidates and the recorded primary
parent to classify, without copying or modifying legacy ontology outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.ontology_categorical_fingerprint import (
    campaign_schema,
    load_contract,
    schema_revision,
)

DEFAULT_DATA_ROOT = Path("data/c0c3")
DEFAULT_OUTPUT_ROOT = Path("outputs/ontology-categorical-v1")


def _readable_path(path: Path) -> Path:
    """Use Windows long-path syntax for deeply nested candidate hashes."""
    resolved = path.resolve(strict=False)
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _read_events(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected event object in {path}:{line_number}")
        result.append(value)
    return result


def _source_status(run_root: Path, candidate_id: str) -> tuple[str, str]:
    candidate_root = run_root / "candidates" / candidate_id
    readable_root = _readable_path(candidate_root)
    if not readable_root.is_dir():
        return "missing", f"candidates/{candidate_id}"
    if any(readable_root.rglob("*.py")):
        return "available", candidate_root.relative_to(run_root).as_posix()
    return "no_python", candidate_root.relative_to(run_root).as_posix()


def _record_from_event(run_root: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = event.get("candidate_id")
    proposal = event.get("opportunity")
    parents = event.get("parent_ids", [])
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("proposal_completed event has no candidate_id")
    if type(proposal) is not int or proposal < 1:
        raise ValueError(
            f"proposal_completed event has invalid opportunity: {proposal!r}"
        )
    if not isinstance(parents, list) or any(
        not isinstance(parent, str) or not parent for parent in parents
    ):
        raise ValueError(f"Proposal {proposal} has invalid parent_ids")
    source_status, source_path = _source_status(run_root, candidate_id)
    return {
        "proposal": proposal,
        "candidate_id": candidate_id,
        "parent_ids": parents,
        "ontology_parent_id": parents[0] if parents else None,
        "artifact_path": event.get("artifact_path") or source_path,
        "source_status": source_status,
        "source_path": source_path,
        "condition": event.get("condition"),
        "retained": event.get("retained"),
        "valid": event.get("evaluation", {}).get("valid")
        if isinstance(event.get("evaluation"), dict)
        else None,
        "failure_kind": event.get("evaluation", {}).get("failure_kind")
        if isinstance(event.get("evaluation"), dict)
        else None,
    }


def build_campaign_inventory(
    campaign_id: str, data_root: Path = DEFAULT_DATA_ROOT
) -> dict[str, Any]:
    """Return every in-scope seed/proposal for a campaign, grouped by run."""
    schema = campaign_schema(campaign_id)
    campaign_root = Path(data_root) / schema["data_directory"]
    campaign = _read_json(campaign_root / "campaign.json")
    run_ids = campaign.get("primary_run_ids")
    if not isinstance(run_ids, list) or not all(
        isinstance(item, str) and item for item in run_ids
    ):
        raise ValueError(f"{campaign_root}/campaign.json has invalid primary_run_ids")
    proposal_limit = schema.get("proposal_limit")
    if proposal_limit is not None and (
        type(proposal_limit) is not int or proposal_limit < 1
    ):
        raise ValueError(f"{campaign_id} has invalid proposal_limit")
    problems: list[dict[str, Any]] = []
    included_conditions = schema.get("included_conditions")
    if not isinstance(included_conditions, list) or not all(
        isinstance(condition, str) and condition for condition in included_conditions
    ):
        raise ValueError(f"{campaign_id} has invalid included_conditions")
    excluded_runs: list[dict[str, str]] = []
    runs: list[dict[str, Any]] = []
    canonical_seeds: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        run_root = campaign_root / "runs" / run_id
        manifest = _read_json(run_root / "manifest.json")
        condition = manifest.get("assignment", {}).get("condition")
        if condition not in included_conditions:
            excluded_runs.append(
                {
                    "run_id": run_id,
                    "condition": str(condition),
                    "reason": "condition outside categorical C0 C3 study scope",
                }
            )
            continue
        baseline = manifest.get("baseline", {})
        seed_id = baseline.get("candidate_id")
        if not isinstance(seed_id, str) or not seed_id:
            raise ValueError(f"{run_id} has no baseline candidate_id")
        seed_status, seed_path = _source_status(run_root, seed_id)
        canonical_seed = canonical_seeds.setdefault(
            seed_id,
            {
                "candidate_id": seed_id,
                "classification_status": "pending_categorical_annotation",
                "source_status": seed_status,
                "source_path": seed_path,
                "run_ids": [],
            },
        )
        if (
            canonical_seed["source_status"] != "available"
            and seed_status == "available"
        ):
            canonical_seed["source_status"] = seed_status
            canonical_seed["source_path"] = seed_path
        canonical_seed["run_ids"].append(run_id)
        records = [
            {
                "proposal": 0,
                "candidate_id": seed_id,
                "parent_ids": [],
                "ontology_parent_id": None,
                "artifact_path": seed_path,
                "source_status": seed_status,
                "source_path": seed_path,
                "condition": condition,
                "retained": True,
                "valid": True,
                "failure_kind": None,
                "is_seed": True,
                "canonical_seed_id": seed_id,
            }
        ]
        by_proposal: dict[int, dict[str, Any]] = {}
        for event in _read_events(run_root / "events.jsonl"):
            if event.get("event") != "proposal_completed":
                continue
            record = _record_from_event(run_root, event)
            if proposal_limit is not None and record["proposal"] > proposal_limit:
                continue
            prior = by_proposal.get(record["proposal"])
            if prior and prior["candidate_id"] != record["candidate_id"]:
                raise ValueError(
                    f"{run_id} has conflicting candidate IDs at proposal "
                    f"{record['proposal']}"
                )
            by_proposal[record["proposal"]] = record
        records.extend(by_proposal[proposal] for proposal in sorted(by_proposal))
        for record in records:
            if record["source_status"] != "available":
                problems.append(
                    {
                        "run_id": run_id,
                        "proposal": record["proposal"],
                        "candidate_id": record["candidate_id"],
                        "reason": "candidate source " + record["source_status"],
                    }
                )
        earlier: set[str] = set()
        for record in records:
            parent = record["ontology_parent_id"]
            if parent is not None and parent not in earlier:
                problems.append(
                    {
                        "run_id": run_id,
                        "proposal": record["proposal"],
                        "candidate_id": record["candidate_id"],
                        "reason": "primary parent is outside inventory",
                        "ontology_parent_id": parent,
                    }
                )
            earlier.add(record["candidate_id"])
        runs.append(
            {
                "run_id": run_id,
                "condition": condition,
                "records": records,
            }
        )
    return {
        "fingerprint_version": "categorical_fingerprint_v1",
        "campaign": campaign_id,
        "source_campaign_key": schema["source_campaign_key"],
        "campaign_directory": schema["data_directory"],
        "proposal_limit": proposal_limit,
        "included_conditions": included_conditions,
        "primary_parent_rule": load_contract()["primary_parent_rule"],
        "canonical_seed_rule": (
            "Classify each canonical shared campaign seed once; every proposal-zero "
            "record references that annotation."
        ),
        "status": "ready_for_categorical_annotation",
        "schema_revision": schema_revision(schema),
        "required_review_version": load_contract()["review_policy"]["version"],
        "canonical_seeds": [
            canonical_seeds[seed_id] for seed_id in sorted(canonical_seeds)
        ],
        "runs": runs,
        "excluded_runs": excluded_runs,
        "problems": problems,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_inventory(
    campaign_id: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    inventory = build_campaign_inventory(campaign_id, data_root)
    inventory["generated_at"] = datetime.now(UTC).isoformat()
    destination = Path(output_root) / "inventories" / f"{campaign_id}.json"
    _atomic_json(destination, inventory)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    campaign_ids = [item["id"] for item in load_contract()["campaigns"]]
    selected = campaign_ids if args.campaign == "all" else [args.campaign]
    for campaign_id in selected:
        if campaign_id not in campaign_ids:
            raise SystemExit(
                f"Unknown campaign {campaign_id!r}; choose from "
                f"{', '.join(campaign_ids)}"
            )
        print(write_inventory(campaign_id, args.data_root, args.output_root))


if __name__ == "__main__":
    main()

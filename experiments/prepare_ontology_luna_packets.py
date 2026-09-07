"""Prepare small, disjoint campaign work packets without classifying source.

Each candidate ID is assigned once and retains every trajectory occurrence.
The shared seed has its own packet. Existing progress and evolving schemas are
never overwritten when this preparation command is rerun.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.ontology_categorical_fingerprint import campaign_schema, load_contract
from experiments.ontology_categorical_inventory import _atomic_json


def _write_unchanged_or_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(
                f"Existing packet differs; preserve it and reconcile: {path}"
            )
        return
    _atomic_json(path, value)


def prepare(
    campaign_id: str,
    output_root: Path = Path("outputs/ontology-categorical-v1"),
    data_root: Path = Path("data/c0c3"),
    batch_size: int = 10,
) -> dict[str, Any]:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    schema = campaign_schema(campaign_id)
    inventory_path = output_root / "inventories" / f"{campaign_id}.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory["campaign"] != campaign_id:
        raise ValueError("Inventory campaign mismatch")
    seed_ids = {s["candidate_id"] for s in inventory["canonical_seeds"]}
    candidates: dict[str, dict[str, Any]] = {}
    occurrence_count = 0
    for run in inventory["runs"]:
        if run["condition"] not in schema["included_conditions"]:
            raise ValueError("Out-of-scope condition in inventory")
        for row in run["records"]:
            if row["proposal"] > schema.get("proposal_limit", float("inf")):
                raise ValueError("Out-of-scope proposal in inventory")
            candidate_id = row["candidate_id"]
            if not isinstance(candidate_id, str) or not candidate_id.isalnum():
                raise ValueError(
                    "Candidate IDs must be safe alphanumeric path segments"
                )
            candidate = candidates.setdefault(
                candidate_id,
                {
                    "candidate_id": candidate_id,
                    "is_canonical_seed": candidate_id in seed_ids,
                    "occurrences": [],
                },
            )
            candidate["occurrences"].append(
                {
                    "run_id": run["run_id"],
                    "condition": run["condition"],
                    "proposal": row["proposal"],
                    "parent_ids": row["parent_ids"],
                    "ontology_parent_id": row["ontology_parent_id"],
                    "source_status": row["source_status"],
                    "candidate_directory": str(
                        (
                            data_root
                            / schema["data_directory"]
                            / "runs"
                            / run["run_id"]
                            / row["source_path"]
                        ).resolve()
                    ),
                }
            )
            occurrence_count += 1
    if not seed_ids <= candidates.keys():
        raise ValueError("Canonical seed has no trajectory occurrence")
    base = output_root / "forks" / campaign_id
    working_schema = base / "working-schema.json"
    if not working_schema.exists():
        _atomic_json(working_schema, dict(schema))
    nonseeds = [c for k, c in candidates.items() if k not in seed_ids]
    groups = [("seed", [candidates[k] for k in sorted(seed_ids)])]
    groups.extend(
        (f"batch_{i // batch_size + 1:04d}", nonseeds[i : i + batch_size])
        for i in range(0, len(nonseeds), batch_size)
    )
    entries = []
    for packet_id, members in groups:
        destination = base / "packets" / f"{packet_id}.json"
        packet = {
            "packet_id": packet_id,
            "campaign": campaign_id,
            "model": "gpt-5.6-luna",
            "inventory_path": str(inventory_path.resolve()),
            "working_schema_path": str(working_schema.resolve()),
            "output_directory": str((base / "results" / packet_id).resolve()),
            "instructions_path": str(
                Path("docs/ONTOLOGY_LUNA_FORK_PLAYBOOK.md").resolve()
            ),
            "seed_result_path": str((base / "results" / "seed").resolve()),
            "candidates": members,
        }
        _write_unchanged_or_new(destination, packet)
        entries.append(
            {
                "packet_id": packet_id,
                "path": str(destination.resolve()),
                "candidate_count": len(members),
            }
        )
    queue = {
        "campaign": campaign_id,
        "model": "gpt-5.6-luna",
        "unique_candidates": len(candidates),
        "trajectory_rows": occurrence_count,
        "canonical_seeds": len(seed_ids),
        "batch_size": batch_size,
        "instructions": (
            "Fill available subagent slots. Main agent also works. Read playbook."
        ),
        "packets": entries,
    }
    _write_unchanged_or_new(base / "queue.json", queue)
    return queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="all")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/ontology-categorical-v1")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/c0c3"))
    args = parser.parse_args()
    campaigns = [c["id"] for c in load_contract()["campaigns"]]
    selected = campaigns if args.campaign == "all" else [args.campaign]
    for campaign_id in selected:
        queue = prepare(campaign_id, args.output_root, args.data_root, args.batch_size)
        print(
            f"{campaign_id}: {queue['unique_candidates']} unique candidates; "
            f"{len(queue['packets']) - 1} batches plus seed packet"
        )


if __name__ == "__main__":
    main()

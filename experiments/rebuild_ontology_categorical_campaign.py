"""Reconcile corrected source references and republish complete campaign metrics.

This preserves event identities and primary parents. Source-equivalent reviews
may be reused, but their certificates are revalidated against the actual source
bundle and current campaign schema before publication.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from experiments.ontology_categorical_dashboard import write_dashboard_metrics
from experiments.ontology_categorical_fingerprint import (
    output_document,
    schema_revision,
)
from experiments.ontology_categorical_inventory import (
    _atomic_json,
    _readable_path,
    write_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/ontology-categorical-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def backup(path, base):
    if path.exists():
        destination = base / "source-recovery/before" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)


def snapshot_sources(path):
    readable = _readable_path(path)
    return {
        p.relative_to(readable).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(readable.rglob("*.py"))
    }


def archive_resolved_exceptions(base, recovered_ids):
    """Remove resolved cards from active queues while retaining their history."""
    paths = list((base / "results").rglob("*.exception.json"))
    paths.extend((base / "checks/exceptions").glob("*.json"))
    for path in paths:
        item = read(path)
        identity = item.get("candidate_id", path.stem)
        if identity not in recovered_ids:
            continue
        destination = (
            base / "source-recovery/before/exception_cards" / path.relative_to(base)
        )
        if not path.resolve().is_relative_to(
            base.resolve()
        ) or not destination.resolve().is_relative_to(base.resolve()):
            raise ValueError("Exception archive path leaves campaign output")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)
        path.unlink()


def reconcile_progress_counters(base):
    """Replace legacy partial-publication counters with verified final coverage."""
    completion = read(base / "source-recovery/completion.json")
    progress = read(base / "progress.json")
    queue = read(base / "queue.json")
    progress.update(
        source_unavailable_unique=0,
        source_unavailable_unique_candidates=0,
        deferred_after_gap_unique=0,
        published_unique=completion["validated_unique_candidates"],
        published_trajectory_rows=completion["annotated_occurrence_count"],
        packets_total=len(queue["packets"]),
        packets_completed=len(queue["packets"]),
        active_workers=0,
        metric_validation="passed",
    )
    _atomic_json(base / "progress.json", progress)


def refresh_packets(base, inventory):
    """Keep packet ownership stable while filtering scope and correcting paths."""
    queue = read(base / "queue.json")
    records = {
        (run["run_id"], row["proposal"], row["candidate_id"]): row
        for run in inventory["runs"]
        for row in run["records"]
    }
    expected_ids = {key[2] for key in records}
    seen = set()
    seen_rows = set()
    entries = []
    retired = list(queue.get("retired_packet_ids", []))
    for entry in queue["packets"]:
        path = Path(entry["path"])
        packet = read(path)
        members = []
        for candidate in packet["candidates"]:
            occurrences = []
            for occurrence in candidate["occurrences"]:
                key = (
                    occurrence["run_id"],
                    occurrence["proposal"],
                    candidate["candidate_id"],
                )
                if key not in records:
                    continue
                row = records[key]
                occurrence.update(
                    source_status=row["source_status"],
                    candidate_directory=str(
                        (
                            ROOT
                            / "data/c0c3"
                            / inventory["campaign_directory"]
                            / "runs"
                            / key[0]
                            / row["source_path"]
                        ).resolve()
                    ),
                )
                if "source_resolution" in row:
                    occurrence["source_resolution"] = row["source_resolution"]
                occurrences.append(occurrence)
                seen_rows.add(key)
            if occurrences:
                candidate["occurrences"] = occurrences
                members.append(candidate)
                if candidate["candidate_id"] in seen:
                    raise ValueError("Candidate is owned by multiple packets")
                seen.add(candidate["candidate_id"])
        if members:
            packet["candidates"] = members
            _atomic_json(path, packet)
            entries.append(dict(entry, candidate_count=len(members)))
        else:
            retired.append(packet["packet_id"])
    if seen != expected_ids or seen_rows != set(records):
        raise ValueError("Corrected queue does not exactly cover the scoped inventory")
    queue.update(
        packets=entries,
        unique_candidates=len(expected_ids),
        trajectory_rows=len(records),
        canonical_seeds=len(inventory["canonical_seeds"]),
        retired_packet_ids=sorted(set(retired)),
        source_references_corrected=True,
        excluded_runs=inventory["excluded_runs"],
    )
    _atomic_json(base / "queue.json", queue)


def rebuild_campaign(campaign):
    base = OUTPUT / "forks" / campaign
    inventory_path = OUTPUT / "inventories" / f"{campaign}.json"
    for path in [
        inventory_path,
        base / "progress.json",
        base / "queue.json",
        base / "HANDOFF.md",
        base / "UNRESOLVED.md",
    ]:
        backup(path, base)
    write_inventory(campaign, ROOT / "data/c0c3", OUTPUT)
    inventory = read(inventory_path)
    if inventory["problems"]:
        raise ValueError(f"Unresolved inventory records: {len(inventory['problems'])}")
    schema = read(base / "working-schema.json")
    revision = schema_revision(schema)
    expected = {
        row["candidate_id"] for run in inventory["runs"] for row in run["records"]
    }
    results = {}
    bundles = {}
    for path in sorted((base / "results").rglob("*.json")):
        if not re.fullmatch(r"[a-f0-9]{64}", path.stem) or path.stem not in expected:
            continue
        result = read(path)
        if (
            result.get("status") != "validated"
            or result.get("schema_revision") != revision
        ):
            continue
        candidate = result["candidate_id"]
        if candidate != path.stem:
            raise ValueError(f"Result identity mismatch: {path}")
        source = read(path.with_name(f"{candidate}.source.json"))
        if "sources" in source:
            if source.get("candidate_id") != candidate:
                raise ValueError(f"Source identity mismatch: {path}")
            source = source["sources"]
        if candidate in results and (
            bundles[candidate] != source
            or results[candidate]["fingerprint"] != result["fingerprint"]
        ):
            raise ValueError(f"Conflicting result: {candidate}")
        results[candidate] = result
        bundles[candidate] = source
    if set(results) != expected:
        raise ValueError(
            f"Missing current validated reviews: {len(expected - set(results))}"
        )
    print(
        f"{campaign}: loaded {len(results)} reviews; verifying recovered artifacts",
        flush=True,
    )
    recoveries = []
    runs = {}
    for run in inventory["runs"]:
        rows = []
        for record in run["records"]:
            row = dict(record)
            candidate = row["candidate_id"]
            resolution = row.get("source_resolution", {})
            if resolution.get("kind") == "recorded_artifact_alias":
                path = (
                    ROOT
                    / "data/c0c3"
                    / inventory["campaign_directory"]
                    / "runs"
                    / run["run_id"]
                    / row["source_path"]
                )
                if snapshot_sources(path) != bundles[candidate]:
                    raise ValueError(
                        f"Recovered review does not match artifact: {candidate}"
                    )
                recoveries.append(
                    {
                        "run_id": run["run_id"],
                        "proposal": row["proposal"],
                        "candidate_id": candidate,
                        "failure_kind": row["failure_kind"],
                        **resolution,
                    }
                )
            if not row.get("is_seed"):
                row["fingerprint"] = results[candidate]["fingerprint"]
            rows.append(row)
        runs[run["run_id"]] = rows
    seed_fingerprints = {
        s["candidate_id"]: results[s["candidate_id"]]["fingerprint"]
        for s in inventory["canonical_seeds"]
    }
    print(
        f"{campaign}: revalidating source reviews and recomputing all eight metrics",
        flush=True,
    )
    document = output_document(
        campaign,
        runs,
        canonical_seed_fingerprints=seed_fingerprints,
        source_reviews={cid: item["source_review"] for cid, item in results.items()},
        source_bundles=bundles,
        schema_override=schema,
    )
    by_occurrence = {
        (run_id, row["proposal"]): row
        for run_id, rows in document["runs"].items()
        for row in rows
    }
    for recovery in recoveries:
        row = by_occurrence[(recovery["run_id"], recovery["proposal"])]
        recovery["metrics"] = row["metrics"]
        if recovery["artifact_is_primary_parent"] and any(
            value for key, value in row["metrics"].items() if key.endswith("_marginal")
        ):
            raise ValueError(
                "Verified parent snapshot must have zero marginal contributions"
            )
    counts = Counter(x["failure_kind"] for x in recoveries)
    completion = {
        "unique_candidate_denominator": len(expected),
        "validated_unique_candidates": len(results),
        "source_unavailable_unique_candidates": 0,
        "inventory_occurrence_count": len(by_occurrence),
        "annotated_occurrence_count": len(by_occurrence),
        "run_count": len(runs),
        "recovered_event_ids": len({r["candidate_id"] for r in recoveries}),
        "recovered_occurrences": len(recoveries),
        "recovery_failure_kinds": dict(counts),
        "validated_percent": 100.0,
        "note": (
            "Every scoped proposal is retained. Event IDs and primary parents are "
            "unchanged; recorded source aliases use verified source-equivalent reviews."
        ),
    }
    document.update(
        completion=completion,
        unresolved_occurrences=[],
        source_recoveries=recoveries,
        excluded_runs=inventory["excluded_runs"],
    )
    _atomic_json(base / "final.json", document)
    write_dashboard_metrics(base, document)
    refresh_packets(base, inventory)
    archive_resolved_exceptions(base, {r["candidate_id"] for r in recoveries})
    progress = read(base / "progress.json")
    for key in ["validated", "validated_unique", "validated_unique_candidates"]:
        progress[key] = len(results)
    for key in [
        "unique_candidate_denominator",
        "unique_denominator",
        "denominator_unique_candidates",
    ]:
        progress[key] = len(expected)
    progress.update(
        schema_revision=revision,
        unresolved=0,
        pending_unique_candidates=0,
        trajectory_rows=len(by_occurrence),
        state="complete",
        validated_percent=100.0,
        active_assignments=[],
        evidence_tasks_pending=0,
        updated_at=datetime.now(UTC).isoformat(),
        source_recovery=completion,
        scope={
            "conditions": inventory["included_conditions"],
            "proposal_max": inventory["proposal_limit"],
            "excluded_runs": inventory["excluded_runs"],
        },
        notes=completion["note"],
    )
    _atomic_json(base / "progress.json", progress)
    _atomic_json(
        base / "source-recovery/completion.json",
        {**completion, "schema_revision": revision, "recoveries": recoveries},
    )
    reconcile_progress_counters(base)
    (base / "UNRESOLVED.md").write_text(
        f"# {campaign} source resolution\n\n"
        "No unresolved candidate sources remain in the current scope. "
        f"{len(recoveries)} previously missing event records resolve to existing "
        "source snapshots. Each recovered source bundle was compared with the "
        "recorded artifact and its role review revalidated. "
        "See [source recovery details](source-recovery/completion.json). "
        "Historical reports remain under `source-recovery/before/`.\n",
        encoding="utf-8",
    )
    (base / "HANDOFF.md").write_text(
        f"# {campaign} corrected categorical handoff\n\n"
        f"Complete: **{len(results):,} / {len(expected):,} "
        "candidate identities (100%)** across "
        f"{len(runs)} runs and {len(by_occurrence):,} proposal/seed occurrences. "
        f"The current schema is `{revision}`.\n\n"
        f"Resolved {len(recoveries)} previously missing records using their recorded "
        "artifact paths. Original event IDs, parent IDs, failed outcomes and "
        "proposal numbers remain intact. All source reviews pass the publication "
        "validator; all eight metrics were recomputed from each seed. "
        "Parent-equivalent failures contribute zero marginal values. "
        "Reused non-parent snapshots use the actual parent comparison.\n\n"
        "[Final output](final.json) · "
        "[Recovery evidence](source-recovery/completion.json) · "
        "[Progress](progress.json)\n\n"
        "Rebuild from the repository root: "
        "`outputs/tiny-seed-search/venv/Scripts/python.exe "
        "-m experiments.rebuild_ontology_categorical_campaign "
        f"--campaign {campaign}`.\n\n"
        "Queues now follow the corrected inventory; packet IDs and existing "
        "results were preserved. Run this rebuild command to reconcile future "
        "scope/source changes instead of recreating packet ownership.\n",
        encoding="utf-8",
    )
    if (base / "STATUS.md").exists():
        (base / "STATUS.md").write_text(
            f"# {campaign} status\n\nComplete: {len(results):,}/{len(expected):,} "
            "candidate identities, 100%. No unresolved sources. "
            "See [handoff](HANDOFF.md).\n",
            encoding="utf-8",
        )
    print(json.dumps({"campaign": campaign, **completion}), flush=True)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", required=True, choices=["addition", "fashion", "nanogpt", "kws"]
    )
    args = parser.parse_args()
    rebuild_campaign(args.campaign)


if __name__ == "__main__":
    main()

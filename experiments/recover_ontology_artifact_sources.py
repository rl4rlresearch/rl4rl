"""Reuse source-identical reviewed snapshots for synthetic proposal identities."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy

from experiments.ontology_categorical_fingerprint import schema_revision
from experiments.ontology_categorical_inventory import (
    _atomic_json,
    build_campaign_inventory,
)
from experiments.ontology_categorical_review import (
    source_digest,
    validate_source_review,
)
from experiments.rebuild_ontology_categorical_campaign import (
    OUTPUT,
    ROOT,
    read,
    snapshot_sources,
)


def _review_entries(base, revision):
    """Index current-schema reviews; defer certificate validation until lookup."""
    by_identity = {}
    by_digest = {}
    for path in sorted((base / "results").rglob("*.json")):
        if not re.fullmatch(r"[a-f0-9]{64}", path.stem):
            continue
        try:
            template = read(path)
            if (
                template.get("status") != "validated"
                or template.get("schema_revision") != revision
                or template.get("candidate_id") != path.stem
            ):
                continue
            source = read(path.with_name(f"{path.stem}.source.json"))
            if "sources" in source:
                if source.get("candidate_id") != path.stem:
                    continue
                source = source["sources"]
            digest = source_digest(source)
        except (OSError, KeyError, TypeError, ValueError):
            continue
        entry = {
            "path": path,
            "template": template,
            "source": source,
            "validation_attempted": False,
            "validation_receipt": None,
            "validation_error": None,
        }
        by_identity.setdefault(path.stem, []).append(entry)
        by_digest.setdefault(digest, []).append(entry)
    return by_identity, by_digest


def _find_review(source_id, actual, by_identity, by_digest, schema):
    """Prefer a direct identity match, then require exact digest and equality."""
    digest = source_digest(actual)
    direct = by_identity.get(source_id, [])
    candidates = direct + [
        item for item in by_digest.get(digest, []) if item not in direct
    ]
    for entry in candidates:
        if entry["source"] == actual and source_digest(entry["source"]) == digest:
            if not entry["validation_attempted"]:
                entry["validation_attempted"] = True
                try:
                    entry["validation_receipt"] = validate_source_review(
                        entry["template"]["source_review"],
                        actual,
                        entry["template"]["fingerprint"],
                        schema,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    entry["validation_error"] = str(exc)
            if entry["validation_error"] is None:
                return entry
    return None


def recover(campaign):
    base = OUTPUT / "forks" / campaign
    destination = base / "results/recovered_sources"
    schema = read(base / "working-schema.json")
    revision = schema_revision(schema)
    inventory = build_campaign_inventory(campaign, ROOT / "data/c0c3")
    by_identity, by_digest = _review_entries(base, revision)
    resolved = []
    unresolved = list(inventory["problems"])
    for run in inventory["runs"]:
        for row in run["records"]:
            proof = row.get("source_resolution", {})
            if proof.get("kind") != "recorded_artifact_alias":
                continue
            identity = row["candidate_id"]
            source_id = proof["source_candidate_id"]
            source_path = (
                ROOT
                / "data/c0c3"
                / inventory["campaign_directory"]
                / "runs"
                / run["run_id"]
                / row["source_path"]
            )
            actual = snapshot_sources(source_path)
            entry = _find_review(source_id, actual, by_identity, by_digest, schema)
            if entry is None:
                unresolved.append(
                    {
                        "candidate_id": identity,
                        "reason": "No current-schema review has an exact source match",
                        "source_candidate_id": source_id,
                        "source_sha256": source_digest(actual),
                    }
                )
                continue
            template = entry["template"]
            template_path = entry["path"]
            receipt = entry["validation_receipt"]
            enriched = {
                **proof,
                "run_id": run["run_id"],
                "proposal": row["proposal"],
                "failure_kind": row["failure_kind"],
                "parent_ids": row["parent_ids"],
                "reused_review_path": str(template_path),
                "source_sha256": source_digest(actual),
            }
            result = deepcopy(template)
            result.update(
                candidate_id=identity,
                source_bundle_path=str(destination / f"{identity}.source.json"),
                source_resolution=enriched,
                validation_receipt=receipt,
            )
            _atomic_json(
                destination / f"{identity}.source.json",
                {
                    "candidate_id": identity,
                    "sources": actual,
                    "occurrences": [{"run_id": run["run_id"], **row}],
                },
            )
            _atomic_json(destination / f"{identity}.json", result)
            resolved.append(enriched)
    manifest = {
        "campaign": campaign,
        "schema_revision": revision,
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
        "resolved": resolved,
        "unresolved": unresolved,
    }
    _atomic_json(base / "source-recovery/manifest.json", manifest)
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in {"resolved", "unresolved"}}
        ),
        flush=True,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        required=True,
        choices=["addition", "nanogpt", "fashion", "kws", "tiny_adderboard"],
    )
    args = parser.parse_args()
    recover(args.campaign)


if __name__ == "__main__":
    main()

"""Read compact categorical publications without opening final.json evidence.

``enrich_run(repo_root, payload)`` returns a copy with ``categorical`` on each
occurrence and ``categorical_publication`` on the run. Published cumulative
metrics are copied verbatim and never recomputed on a selected lineage.

The compact v5 publication has no source bundle digest. An exact identity join
and verified publication checksum therefore do NOT independently verify today's
candidate source. This limitation is recorded on every joined annotation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from experiments.ontology_categorical_dashboard import METRIC_VIEW_VERSION
from experiments.ontology_categorical_fingerprint import (
    FINGERPRINT_VERSION,
    METRICS,
    changed_components,
    family_id,
    schema_revision,
    validate_fingerprint,
)

_ROOT = Path("outputs/ontology-categorical-v1")
_LIMIT = 32_000_000


def _signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


@lru_cache(maxsize=24)
def _verified_document(
    path: Path, signature: tuple[int, int, int], expected_hash: str, expected_bytes: int
) -> dict[str, Any]:
    del signature  # Cache invalidates when the file metadata changes.
    if path.name not in {"working-schema.json", "trajectory-metrics.json"}:
        raise ValueError("Only compact schema and metric files may be read")
    if not 0 < expected_bytes <= _LIMIT or path.stat().st_size != expected_bytes:
        raise ValueError(f"Publication size mismatch: {path.name}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f"Publication SHA-256 mismatch: {path.name}")
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("Published document must be an object")
    return document


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "publication-manifest.json"
    if path.stat().st_size > 2_000_000:
        raise ValueError("Publication manifest is too large")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != "ontology-publication-v1"
    ):
        raise ValueError("Unsupported publication manifest")
    return manifest


def _read_published(
    root: Path, campaign: str, meta: Mapping[str, Any], name: str
) -> dict[str, Any]:
    record = meta.get("files", {}).get(name, {})
    expected = root / "forks" / campaign / name
    path = (root / str(record.get("path", ""))).resolve()
    if path != expected.resolve() or not path.is_relative_to(root.resolve()):
        raise ValueError("Publication manifest path does not match campaign")
    checksum, size = record.get("sha256"), record.get("bytes")
    if not isinstance(checksum, str) or len(checksum) != 64 or type(size) is not int:
        raise ValueError("Missing publication checksum or size")
    return _verified_document(path, _signature(path), checksum, size)


def _publication(
    repo_root: Path, campaign: str
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    root = repo_root / _ROOT
    manifest = _read_manifest(root)
    for identifier, meta in manifest.get("campaigns", {}).items():
        if not isinstance(meta, dict):
            continue
        schema = _read_published(root, identifier, meta, "working-schema.json")
        if campaign not in {
            schema.get("data_directory"),
            schema.get("source_campaign_key"),
        }:
            continue
        revision = schema_revision(schema)
        if (
            revision != meta.get("schema_revision")
            or meta.get("fingerprint_version") != FINGERPRINT_VERSION
        ):
            raise ValueError("Publication manifest/schema revision mismatch")
        compact = _read_published(root, identifier, meta, "trajectory-metrics.json")
        if (
            compact.get("campaign") != identifier
            or compact.get("schema_revision") != revision
            or compact.get("fingerprint_version") != FINGERPRINT_VERSION
            or compact.get("metric_view_version") != METRIC_VIEW_VERSION
            or not isinstance(compact.get("runs"), dict)
        ):
            raise ValueError("Compact publication version/schema mismatch")
        return {"schema": schema, "compact": compact}, {
            "available": True,
            "campaign": identifier,
            "schema_revision": revision,
            "fingerprint_version": FINGERPRINT_VERSION,
            "metric_view_version": METRIC_VIEW_VERSION,
            "created_at": manifest.get("created_at"),
            "checksum_verified": True,
            "files": {
                name: meta["files"][name]
                for name in ("working-schema.json", "trajectory-metrics.json")
            },
            "source_verification": "unverified",
            "notes": [
                "Compact publication identity is verified; "
                "source audit files are not read.",
                "Published cumulative metrics retain full-run history, "
                "independent of replay filters.",
                "The v5 compact publication does not include source bundle digests.",
            ],
        }
    reason = "No published categorical review for this campaign"
    contract_path = repo_root / "schemas/ontology-categorical-fingerprint-v1.json"
    if contract_path.is_file() and contract_path.stat().st_size <= 2_000_000:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        for schema in contract.get("campaigns", []):
            if campaign in {
                schema.get("data_directory"),
                schema.get("source_campaign_key"),
            }:
                reason = manifest.get("excluded_campaigns", {}).get(
                    schema.get("id"), reason
                )
                break
    return None, {"available": False, "reason": reason, "checksum_verified": False}


def _blank(status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "fingerprint": None,
        "family_id": None,
        "metrics": None,
        "changed_components": None,
        "comparison": None,
        "source_verification": "unverified",
        "source_verification_reason": "No reviewed source digest joined",
    }


def _source_verification(
    row: Mapping[str, Any], occurrence: Mapping[str, Any]
) -> tuple[str, str]:
    # Support explicit per-file SHA-256 only; do not equate incompatible bundle
    # hashing algorithms or confuse event IDs with source hashes.
    expected = row.get("source_file_sha256")
    if not isinstance(expected, dict) or not expected:
        return "unverified", "Compact publication does not include source-file digests"
    source = occurrence.get("source", {})
    actual = {
        item["path"]: item["sha256"]
        for item in source.get("files", [])
        if isinstance(item, dict) and "path" in item and "sha256" in item
    }
    if not actual:
        return "unverified", "Source files unavailable for published-digest comparison"
    if actual != expected:
        return "mismatch", "Source file set or SHA-256 differs from publication"
    return "verified", "Complete source-file SHA-256 mapping matches publication"


def enrich_run(repo_root: Path | str, run_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return copies of run/occurrences with conservative categorical annotations."""
    payload = dict(run_payload)
    occurrences = [dict(item) for item in payload.get("occurrences", [])]
    payload["occurrences"] = occurrences
    try:
        publication, metadata = _publication(
            Path(repo_root), str(payload.get("campaign", ""))
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        publication, metadata = (
            None,
            {
                "available": False,
                "checksum_verified": False,
                "reason": str(exc),
                "validation_failed": True,
            },
        )
    payload["categorical_publication"] = metadata
    if publication is None:
        for occurrence in occurrences:
            occurrence["categorical"] = _blank("unreviewed", metadata["reason"])
        metadata["coverage"] = dict(
            Counter(item["categorical"]["status"] for item in occurrences)
        )
        return payload
    schema, compact = publication["schema"], publication["compact"]
    run_id = str(payload.get("run_id", ""))
    history = compact["runs"].get(run_id)
    if not isinstance(history, list):
        for occurrence in occurrences:
            occurrence["categorical"] = _blank(
                "out_of_scope", "Run absent from frozen publication"
            )
        metadata["coverage"] = {"out_of_scope": len(occurrences)}
        return payload
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    try:
        for row in history:
            if (
                not isinstance(row, dict)
                or type(row.get("proposal")) is not int
                or not isinstance(row.get("candidate_id"), str)
            ):
                raise ValueError("Invalid publication occurrence identity")
            key = (row["proposal"], row["candidate_id"])
            if key in rows:
                raise ValueError("Duplicate publication occurrence identity")
            metrics = row.get("metrics")
            expected = {
                f"{stage}:{key}"
                for stage in ("implemented", "retained")
                for key in METRICS
            }
            if (
                not isinstance(metrics, dict)
                or set(metrics) != expected
                or any(
                    value is not None and (type(value) is not int or value < 0)
                    for value in metrics.values()
                )
            ):
                raise ValueError("Invalid compact categorical metrics")
            assumed = row.get("no_change_assumed", False)
            if (
                type(assumed) is not bool
                or assumed
                and row.get("fingerprint") is not None
            ):
                raise ValueError("Assumed-no-change row cannot supply a fingerprint")
            if not assumed:
                validate_fingerprint(row.get("fingerprint"), schema)
            rows[key] = row
    except (ValueError, TypeError) as exc:
        metadata.update(available=False, validation_failed=True, reason=str(exc))
        for occurrence in occurrences:
            occurrence["categorical"] = _blank("unreviewed", str(exc))
        metadata["coverage"] = {"unreviewed": len(occurrences)}
        return payload
    maximum = max((proposal for proposal, _ in rows), default=-1)
    limit = schema.get("proposal_limit", maximum)
    allowed_conditions = schema.get("included_conditions", [])
    condition = str(payload.get("condition", "")).upper()
    previous_candidates: dict[str, dict[str, Any]] = {}
    for occurrence in sorted(occurrences, key=lambda item: item.get("proposal", -1)):
        proposal, candidate = occurrence.get("proposal"), occurrence.get("candidate_id")
        if (
            type(proposal) is not int
            or proposal > limit
            or proposal > maximum
            or condition
            and allowed_conditions
            and condition not in allowed_conditions
        ):
            annotation = _blank(
                "out_of_scope", "Outside published condition/proposal scope"
            )
        else:
            row = rows.get((proposal, candidate))
            if row is None:
                annotation = _blank(
                    "unreviewed", "No exact run/proposal/candidate publication match"
                )
            else:
                verification, explanation = _source_verification(row, occurrence)
                if verification == "mismatch":
                    annotation = _blank("unreviewed", explanation)
                else:
                    assumed = row.get("no_change_assumed", False)
                    fingerprint = None if assumed else dict(row["fingerprint"])
                    annotation = {
                        **_blank(
                            "assumed" if assumed else "reviewed",
                            "Published categorical occurrence",
                        ),
                        "fingerprint": fingerprint,
                        "family_id": family_id(fingerprint, schema)
                        if fingerprint
                        else None,
                        "metrics": dict(row["metrics"]),
                        "no_change_assumed": assumed,
                        "schema_revision": metadata["schema_revision"],
                        "publication_campaign": metadata["campaign"],
                        "identity": {
                            "campaign": payload["campaign"],
                            "run_id": run_id,
                            "proposal": proposal,
                            "candidate_id": candidate,
                        },
                    }
                    parent_ids = occurrence.get("parent_ids", [])
                    # Parent order is authoritative. Never pick a more convenient
                    # reviewed non-primary parent when the primary is unavailable.
                    parent = (
                        previous_candidates.get(parent_ids[0]) if parent_ids else None
                    )
                    parent_annotation = parent.get("categorical", {}) if parent else {}
                    parent_fp = parent_annotation.get("fingerprint")
                    if (
                        fingerprint
                        and parent_fp
                        and parent_annotation.get("status") == "reviewed"
                    ):
                        annotation["changed_components"] = changed_components(
                            parent_fp, fingerprint, schema
                        )
                        annotation["comparison"] = {
                            "mode": "recorded_primary_parent",
                            "occurrence_id": parent.get("id"),
                            "candidate_id": parent.get("candidate_id"),
                            "proposal": parent.get("proposal"),
                        }
                    elif proposal == 0 and fingerprint:
                        annotation["changed_components"] = []
                        annotation["comparison"] = {"mode": "seed"}
                annotation["source_verification"] = verification
                annotation["source_verification_reason"] = explanation
        occurrence["categorical"] = annotation
        if isinstance(candidate, str):
            previous_candidates[candidate] = occurrence
    metadata["coverage"] = dict(
        Counter(item["categorical"]["status"] for item in occurrences)
    )
    metadata["published_run_rows"] = len(history)
    metadata["published_through_proposal"] = maximum
    return payload

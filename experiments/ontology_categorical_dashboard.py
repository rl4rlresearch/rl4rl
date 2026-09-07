"""Expose published categorical metrics without serving bulky source reviews."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

from experiments.ontology_categorical_fingerprint import (
    FINGERPRINT_VERSION,
    METRICS,
    load_contract,
    schema_revision,
)
from experiments.ontology_categorical_inventory import _atomic_json

OUTPUT = Path(__file__).resolve().parents[1] / "outputs/ontology-categorical-v1/forks"
METRIC_VIEW_VERSION = 3
METRIC_LABELS = {
    "component_edits_marginal": "Component changes · marginal",
    "component_edits_cumulative": "Component changes · total",
    "new_component_states_marginal": "New component states explored · marginal",
    "new_component_states_cumulative": "New component states explored · total",
    "family_switches_marginal": "Family changes · marginal",
    "family_switches_cumulative": "Family changes · total",
    "new_families_marginal": "New families explored · marginal",
    "new_families_cumulative": "New families explored · total",
}
AXES = [
    {"key": f"ontology:{stage}:{key}", "label": f"{label} · {stage}", "y_only": True}
    for key, label in METRIC_LABELS.items()
    for stage in ("implemented", "retained")
]


def signature(path):
    stat = path.stat()
    return [stat.st_mtime_ns, stat.st_size]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compact_document(document, source_signature):
    if document.get("fingerprint_version") != FINGERPRINT_VERSION or document.get(
        "schema_revision"
    ) != schema_revision(document["schema"]):
        raise ValueError("Categorical publication version/schema mismatch")
    runs = {}
    for run_id, rows in document["runs"].items():
        selected = []
        retained_totals = {key: 0 for key in METRICS if key.endswith("_cumulative")}
        previous_proposal = -1
        for row in sorted(rows, key=lambda row: row["proposal"]):
            values = row["metrics"]
            if row["candidate_id"] not in document.get("review_audits", {}):
                raise ValueError("Published candidate has no source-review audit")
            if set(values) != set(METRICS) or any(
                type(value) is not int or value < 0 for value in values.values()
            ):
                raise ValueError("Invalid published categorical metric values")
            # Retention gates contributions, not plotted rows or discovery history.
            # Never mask the implemented cumulative value: sum retained marginals.
            stage_values = {
                f"implemented:{key}": value for key, value in values.items()
            }
            if row["proposal"] != previous_proposal + 1:
                retained_totals = dict.fromkeys(retained_totals)
            for cumulative in retained_totals:
                marginal = cumulative.replace("_cumulative", "_marginal")
                retained = row.get("retained")
                contribution = (
                    0
                    if row["proposal"] == 0 or retained is False
                    else values[marginal]
                    if retained is True
                    else None
                )
                total = retained_totals[cumulative]
                retained_totals[cumulative] = (
                    total + contribution
                    if total is not None and contribution is not None
                    else None
                )
                stage_values[f"retained:{marginal}"] = contribution
                stage_values[f"retained:{cumulative}"] = retained_totals[cumulative]
            selected.append(
                {
                    "proposal": row["proposal"],
                    "candidate_id": row["candidate_id"],
                    "metrics": stage_values,
                    "fingerprint": row.get("fingerprint"),
                    "condition": row.get("condition"),
                    "retained": row.get("retained"),
                }
            )
            previous_proposal = row["proposal"]
        runs[run_id] = selected
    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "metric_view_version": METRIC_VIEW_VERSION,
        "campaign": document["campaign"],
        "schema_revision": document["schema_revision"],
        "source_signature": source_signature,
        "runs": runs,
    }


def write_dashboard_metrics(base, document=None):
    """Publish a small derived view after final.json has been written."""
    final = base / "final.json"
    before = signature(final)
    compact = compact_document(
        document if document is not None else read(final), before
    )
    if signature(final) != before:
        raise ValueError("Publication changed while extracting dashboard metrics")
    _atomic_json(base / "trajectory-metrics.json", compact)
    return compact


@lru_cache(maxsize=12)
def _metric_index(base, final_signature, compact_signature, revision):
    compact_path = base / "trajectory-metrics.json"
    compact = read(compact_path) if compact_signature else None
    if (
        not compact
        or compact.get("source_signature") != list(final_signature)
        or compact.get("metric_view_version") != METRIC_VIEW_VERSION
    ):
        compact = compact_document(read(base / "final.json"), list(final_signature))
    if (
        compact.get("fingerprint_version") != FINGERPRINT_VERSION
        or compact.get("schema_revision") != revision
        or compact.get("campaign") != base.name
        or signature(base / "final.json") != list(final_signature)
    ):
        raise ValueError("Categorical publication is stale for the current schema")
    return {
        (run_id, row["proposal"], row["candidate_id"]): row
        for run_id, rows in compact["runs"].items()
        for row in rows
    }


def comparison_metrics(row, previous, parents):
    """Compare whole fingerprints; never choose a different parent per component."""

    def distance(reference):
        fingerprint = row.get("fingerprint")
        other = reference.get("fingerprint") if reference else None
        if not fingerprint or not other or fingerprint.keys() != other.keys():
            return None
        return sum(fingerprint[key] != other[key] for key in fingerprint)

    distances = [distance(parent) for parent in parents] if parents else []
    counts = {
        "previous": distance(previous),
        "minimum_parents": (
            min(distances) if distances and None not in distances else None
        ),
    }
    result = {"condition": row.get("condition")}
    for mode, count in counts.items():
        if row["proposal"] == 0:
            count = 0
        result[mode] = {}
        for metric, value in (
            ("component_edits_marginal", count),
            ("family_switches_marginal", int(count > 0) if count is not None else None),
        ):
            result[mode][f"implemented:{metric}"] = value
            # Same event-retention gate as the primary-parent metric. Missing
            # comparison evidence stays unavailable, including for rejections.
            retained_base = row["metrics"][f"retained:{metric}"]
            result[mode][f"retained:{metric}"] = (
                None
                if value is None or retained_base is None
                else 0
                if row.get("retained") is False
                else value
            )
    return result


def attach_categorical_metrics(campaign, runs, output_root=OUTPUT):
    """Join by run, proposal and event identity; unreviewed points stay missing."""
    for run in runs:
        for point in run["points"]:
            point["ontology_metrics"] = None
            point["ontology_comparisons"] = None
    schema = next(
        (
            s
            for s in load_contract()["campaigns"]
            if s["data_directory"] == campaign.name
        ),
        None,
    )
    if schema is None:
        return {
            "available": False,
            "reason": "No categorical publication for this campaign",
        }
    base = output_root / schema["id"]
    try:
        revision = schema_revision(read(base / "working-schema.json"))
        compact = base / "trajectory-metrics.json"
        index = _metric_index(
            base,
            tuple(signature(base / "final.json")),
            tuple(signature(compact)) if compact.exists() else (),
            revision,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"available": False, "campaign": schema["id"], "reason": str(exc)}
    matched = 0
    for run in runs:
        # References come from complete published run history, independent of
        # browser proposal bounds, outcome filters and intervention windows.
        history = sorted(
            (row for (run_id, _, _), row in index.items() if run_id == run["run_id"]),
            key=lambda row: row["proposal"],
        )
        by_proposal = {row["proposal"]: row for row in history}
        by_id = {}
        for row in history:
            by_id.setdefault(row["candidate_id"], row)
        for point in run["points"]:
            row = index.get(
                (run["run_id"], point["proposal"], point.get("candidate_id"))
            )
            if row is None:
                continue
            matched += 1
            point["ontology_metrics"] = dict(row["metrics"])
            if row.get("condition") not in ("C2", "C3"):
                continue
            parent_ids = point.get("visible_candidate_ids")
            if not isinstance(parent_ids, list) or not all(
                isinstance(parent, str) and parent for parent in parent_ids
            ):
                parent_ids = []
            parents = [
                by_id.get(parent)
                if by_id.get(parent, {}).get("proposal", float("inf")) < row["proposal"]
                else None
                for parent in parent_ids
            ]
            comparisons = comparison_metrics(
                row,
                by_proposal.get(row["proposal"] - 1),
                parents,
            )
            comparisons["previous_proposal"] = (
                row["proposal"] - 1 if row["proposal"] else None
            )
            comparisons["parent_ids"] = list(parent_ids)
            point["ontology_comparisons"] = comparisons
    return {
        "available": True,
        "campaign": schema["id"],
        "schema_revision": revision,
        "matched_points": matched,
        "published_points": len(index),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="append")
    args = parser.parse_args()
    for base in sorted(OUTPUT.iterdir()):
        if args.campaign and base.name not in args.campaign:
            continue
        if (base / "final.json").is_file():
            compact = write_dashboard_metrics(base)
            print(
                base.name,
                sum(map(len, compact["runs"].values())),
                "metric rows",
                flush=True,
            )

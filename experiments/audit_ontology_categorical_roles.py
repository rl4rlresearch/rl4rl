"""Audit every categorical role and flag legacy label inconsistencies.

Findings are review leads, not automatic semantic classifications. Only the
active C0-C3 campaign inventory and Addition proposal limit enter the audit.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from experiments.ontology_categorical_fingerprint import load_contract, schema_revision
from experiments.ontology_categorical_review import validate_review_schema


def audit(output_root: Path, legacy_root: Path) -> dict:
    report = {"campaigns": {}, "note": "Legacy label flags require source review."}
    for schema in load_contract()["campaigns"]:
        validate_review_schema(schema)
        inventory = json.loads(
            (output_root / "inventories" / f"{schema['id']}.json").read_text("utf-8")
        )
        included = {
            (run["run_id"], record["proposal"])
            for run in inventory["runs"]
            for record in run["records"]
        }
        path = legacy_root / f"{schema['source_campaign_key']}.json"
        observed = defaultdict(lambda: defaultdict(list))
        if path.exists():
            legacy = json.loads(path.read_text("utf-8"))
            for row in legacy.get("rows", []):
                if (row.get("run_id"), row.get("proposal")) not in included:
                    continue
                for name, value in row.get("fingerprint", {}).items():
                    if isinstance(value, str):
                        observed[name][value].append(
                            {
                                "run_id": row["run_id"],
                                "proposal": row["proposal"],
                                "candidate_id": row["candidate_id"],
                            }
                        )
        findings = []
        for name, values in sorted(observed.items()):
            folded = defaultdict(list)
            for value in values:
                folded[" ".join(value.split()).casefold()].append(value)
                parts = [v.strip().casefold() for v in re.split(r"\s*\+\s*", value)]
                if len(parts) > 1:
                    findings.append(
                        {
                            "component": name,
                            "kind": "repeated_equivalent_spelling"
                            if len(set(parts)) < len(parts)
                            else "joined_label_requires_role_review",
                            "labels": [value],
                            "rows": len(values[value]),
                            "examples": values[value][:3],
                        }
                    )
            for labels in folded.values():
                if len(labels) > 1:
                    findings.append(
                        {
                            "component": name,
                            "kind": "capitalization_or_spacing_variants",
                            "labels": labels,
                            "rows": sum(len(values[v]) for v in labels),
                            "examples": [values[v][0] for v in labels],
                        }
                    )
        report["campaigns"][schema["id"]] = {
            "schema_revision": schema_revision(schema),
            "components_checked": len(schema["components"]),
            "legacy_components_scanned": len(observed),
            "legacy_file_present": path.exists(),
            "legacy_values_scanned": sum(len(v) for v in observed.values()),
            "provisional_categories": {
                c["id"]: c["provisional_values"]
                for c in schema["components"]
                if c["provisional_values"]
            },
            "findings": findings,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/ontology-categorical-v1")
    )
    parser.add_argument("--legacy-root", type=Path, default=Path("outputs/ontology"))
    args = parser.parse_args()
    report = audit(args.output_root, args.legacy_root)
    destination = args.output_root / "audit"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "role-audit.json").write_text(
        json.dumps(report, indent=2) + "\n", "utf-8"
    )
    lines = [
        "# Component role audit",
        "",
        "Active C0-C3 inventories only; Addition stops at proposal 120. "
        "UCI HAR and Tiny Adderboard are excluded.",
        "",
        "Legacy findings are source-review leads. They are not automatically "
        "new families or confirmed semantic equivalences.",
        "",
        "| Campaign | Categorical roles checked | Legacy components scanned "
        "| Flagged labels/groups |",
        "| --- | ---: | ---: | ---: |",
    ]
    for campaign, item in report["campaigns"].items():
        lines.append(
            f"| {campaign} | {item['components_checked']} | "
            f"{item['legacy_components_scanned']} | {len(item['findings'])} |"
        )
    for campaign, item in report["campaigns"].items():
        lines += [
            "",
            f"## {campaign}",
            "",
            "| Component | Review lead | Labels |",
            "| --- | --- | --- |",
        ]
        for finding in item["findings"]:
            labels = "; ".join(finding["labels"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {finding['component']} | {finding['kind']} | {labels} |")
    (destination / "ROLE_AUDIT.md").write_text("\n".join(lines) + "\n", "utf-8")
    print(
        json.dumps(
            {
                k: {
                    "components": v["components_checked"],
                    "legacy_flags": len(v["findings"]),
                }
                for k, v in report["campaigns"].items()
            }
        )
    )


if __name__ == "__main__":
    main()

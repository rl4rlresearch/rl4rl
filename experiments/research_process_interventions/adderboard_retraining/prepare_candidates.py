#!/usr/bin/env python3
"""Snapshot each trajectory's final incumbent for reproducible retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = HERE.parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "candidates",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    runtime = repo_root / "architecture_discovery"
    experiment_root = repo_root / "experiments/research_process_interventions"
    trajectory_path = experiment_root / "dashboard/app/data/trajectories.json"
    downloads = runtime / "outputs/development/modal_downloads"
    if not trajectory_path.is_file() or not downloads.is_dir():
        raise SystemExit(f"completed trajectory artifacts not found under {repo_root}")

    sys.path.insert(0, str(runtime))
    from common.candidate_artifact import (  # noqa: PLC0415
        build_candidate_artifact,
        inspect_candidate_artifact,
    )

    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            f"candidate output is not empty: {output}; remove it explicitly to rebuild"
        )
    output.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, dict] = {}
    trajectory_rows: list[dict] = []
    for run in payload["runs"]:
        controller = downloads / run["id"] / "controller"
        source = (
            controller / "incumbent.ir.json"
            if run["methodCode"] == "ar"
            else controller / "best/best_program.json"
        )
        inspection = inspect_candidate_artifact(source)
        if not inspection.valid or inspection.architecture_hash is None:
            reasons = "; ".join(inspection.reasons)
            raise SystemExit(f"invalid final candidate for {run['id']}: {reasons}")
        built = build_candidate_artifact(source, seed=1)
        parameter_count = sum(
            parameter.numel() for parameter in built.model.parameters()
        )
        if parameter_count != run["finalBestParameterCount"]:
            raise SystemExit(
                f"parameter-count mismatch for {run['id']}: "
                f"{parameter_count} != {run['finalBestParameterCount']}"
            )

        architecture_hash = inspection.architecture_hash
        candidate_id = f"{architecture_hash[:12]}-p{parameter_count}"
        if candidate_id not in candidates:
            destination = output / f"{candidate_id}.json"
            destination.write_bytes(source.read_bytes())
            candidates[candidate_id] = {
                "candidateId": candidate_id,
                "architectureHash": architecture_hash,
                "graphHash": inspection.graph_hash,
                "parameterCount": parameter_count,
                "filename": destination.name,
                "sha256": _sha256(destination),
                "architectureName": run["trajectory"][-1].get("architectureName", ""),
                "trajectoryIds": [],
            }
        candidates[candidate_id]["trajectoryIds"].append(run["id"])
        trajectory_rows.append(
            {
                "trajectoryId": run["id"],
                "method": run["method"],
                "horizon": run["horizon"],
                "condition": run["condition"],
                "candidateId": candidate_id,
                "parameterCount": parameter_count,
            }
        )

    manifest = {
        "schemaVersion": "1.0",
        "sourceTrajectoryFile": trajectory_path.relative_to(repo_root).as_posix(),
        "trajectoryCount": len(trajectory_rows),
        "uniqueCandidateCount": len(candidates),
        "trajectories": trajectory_rows,
        "candidates": sorted(candidates.values(), key=lambda item: item["candidateId"]),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    by_size: dict[int, int] = defaultdict(int)
    for item in candidates.values():
        by_size[item["parameterCount"]] += 1
    print(
        json.dumps(
            {
                "trajectory_count": len(trajectory_rows),
                "unique_candidate_count": len(candidates),
                "parameter_count_histogram": dict(sorted(by_size.items())),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

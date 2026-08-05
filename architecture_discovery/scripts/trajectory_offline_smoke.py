"""Generate synthetic three-system trajectories and exercise the full pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectory_analysis.pipeline import analyze_study  # noqa: E402
from trajectory_analysis.storage import sha256_file  # noqa: E402


def _jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _annotation(
    event_id: str,
    annotator_id: str,
    family: str,
    boundary: str,
    *,
    role: str = "coder",
) -> dict[str, str]:
    return {
        "event_id": event_id,
        "annotator_id": annotator_id,
        "role": role,
        "edit_family": family,
        "boundary_class": boundary,
        "rationale": "Synthetic fixture label for pipeline validation only.",
    }


def build_synthetic_inputs(data_root: Path) -> Path:
    if data_root.exists():
        raise FileExistsError(f"synthetic input directory already exists: {data_root}")
    data_root.mkdir(parents=True)
    autoresearch = data_root / "autoresearch.tsv"
    autoresearch.write_text(
        "sequence\tcommit\tparent_commit\taccuracy\tparameters\tvalid\tstatus\tdescription\tarchitecture_fingerprint\tstop_claim\n"
        "0\tar0\t\t0.950\t6080\ttrue\taccepted\tseed model\tar-fp0\t\n"
        "1\tar1\tar0\t0.991\t5200\ttrue\taccepted\treduce hidden width\tar-fp1\t\n"
        "2\tar2\tar1\t0.995\t96\ttrue\taccepted\talgebraic positional embedding function\tar-fp2\t\n"
        "3\t\t\t\t\t\tstopped\t\t\tminimum found\n",
        encoding="utf-8",
    )
    openevolve = data_root / "openevolve.jsonl"
    _jsonl(
        openevolve,
        [
            {"iteration": 0, "candidate_id": "oe0", "parent_ids": [], "accuracy": .96, "parameters": 6080, "valid": True, "accepted": True, "description": "seed", "architecture_fingerprint": "oe-fp0"},
            {"iteration": 1, "candidate_id": "oe1", "parent_ids": ["oe0"], "accuracy": .992, "parameters": 4300, "valid": True, "accepted": True, "description": "tie embedding weights", "architecture_fingerprint": "oe-fp1"},
            {"iteration": 2, "candidate_id": "oe2", "parent_ids": ["oe1"], "accuracy": .80, "parameters": 72, "valid": True, "accepted": False, "description": "replace tokens with parametric digit map", "architecture_fingerprint": "oe-fp2"},
            {"iteration": 3, "status": "stop", "stopping_reason": "budget exhausted"},
        ],
    )
    ttt = data_root / "ttt.jsonl"
    _jsonl(
        ttt,
        [
            {"step": 0, "rollout_id": "tt0", "parents": [], "exact_match": .94, "num_params": 6080, "is_valid": True, "selected": True, "summary": "seed", "fingerprint": "tt-fp0"},
            {"step": 1, "rollout_id": "tt1", "parents": ["tt0"], "exact_match": .993, "num_params": 3900, "is_valid": True, "selected": True, "summary": "remove feedforward layer", "fingerprint": "tt-fp1"},
            {"step": 2, "rollout_id": "tt2", "parents": ["tt1"], "exact_match": .993, "num_params": 3900, "is_valid": True, "selected": False, "summary": "repeat depth edit", "fingerprint": "tt-fp1"},
            {"step": 3, "event": "completed", "stopping_reason": "rollout allocation used"},
        ],
    )

    annotations_by_source = {
        "autoresearch": [
            _annotation("autoresearch:1", "coder-a", "width", "ontology_preserving"),
            _annotation("autoresearch:1", "coder-b", "width", "ontology_preserving"),
            _annotation("autoresearch:2", "coder-a", "embeddings", "ontology_changing"),
            _annotation("autoresearch:2", "coder-b", "embeddings", "ontology_changing"),
        ],
        "openevolve": [
            _annotation("openevolve:1", "coder-a", "parameter_tying", "ontology_preserving"),
            _annotation("openevolve:1", "coder-b", "parameter_tying", "ontology_preserving"),
            _annotation("openevolve:2", "coder-a", "tokenization", "ontology_changing"),
            _annotation("openevolve:2", "coder-b", "embeddings", "ontology_preserving"),
            _annotation("openevolve:2", "adjudicator", "tokenization", "ontology_changing", role="adjudicator"),
        ],
        "ttt": [
            _annotation("ttt:1", "coder-a", "feedforward", "ontology_preserving"),
            _annotation("ttt:1", "coder-b", "feedforward", "ontology_preserving"),
            _annotation("ttt:2", "coder-a", "depth", "ontology_preserving"),
            _annotation("ttt:2", "coder-b", "depth", "ontology_preserving"),
        ],
    }
    annotation_paths: dict[str, Path] = {}
    for source_id, records in annotations_by_source.items():
        path = data_root / f"{source_id}.annotations.jsonl"
        _jsonl(path, records)
        annotation_paths[source_id] = path

    sources = []
    source_configs = (
        ("autoresearch", "synthetic-autoresearch", "autoresearch", "autoresearch_tsv_v1", autoresearch),
        ("openevolve", "synthetic-openevolve", "openevolve", "openevolve_jsonl_v1", openevolve),
        ("ttt", "synthetic-ttt", "ttt_discover", "ttt_jsonl_v1", ttt),
    )
    for source_id, run_id, paradigm, adapter, path in source_configs:
        annotation_path = annotation_paths[source_id]
        generator_model = {
            "autoresearch": "synthetic-autoresearch-generator",
            "openevolve": "synthetic-openevolve-generator",
            "ttt": "synthetic-ttt-generator",
        }[source_id]
        sources.append(
            {
                "source_id": source_id,
                "run_id": run_id,
                "paradigm": paradigm,
                "adapter": adapter,
                "path": path.name,
                "sha256": sha256_file(path),
                "annotation_path": annotation_path.name,
                "annotation_sha256": sha256_file(annotation_path),
                "context": {
                    "generator_model": generator_model,
                    "evaluator_id": "synthetic-adderboard-evaluator-v1",
                    "tool_policy": "synthetic-offline-fixture",
                    "run_family_id": run_id,
                    "starting_artifact_sha256": "0" * 64,
                    "prompt_sha256": "1" * 64,
                    "seed": {"autoresearch": 1, "openevolve": 2, "ttt": 3}[source_id],
                    "proposal_budget": 3,
                    "token_budget": 1000,
                    "wall_time_budget_seconds": 60,
                    "accelerator_budget_seconds": 30,
                    "archive_reused": False,
                    "notes": "Synthetic smoke context; not a real research run.",
                },
            }
        )
    manifest = {
        "schema_version": "trajectory-study-v1",
        "study_id": "synthetic-three-system-smoke",
        "accuracy_threshold": 0.99,
        "external_frontier_parameters": 36,
        "sources": sources,
    }
    manifest_path = data_root / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_root = Path(f"{args.output_dir}-inputs")
    manifest_path = build_synthetic_inputs(data_root)
    provenance = analyze_study(manifest_path, data_root, args.output_dir)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(args.output_dir.resolve()),
                "input_dir": str(data_root.resolve()),
                "validation": provenance["validation"],
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

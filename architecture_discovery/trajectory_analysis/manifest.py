"""Frozen input manifest for a trajectory-comparison study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schemas import Paradigm
from .storage import sha256_file


ADAPTERS = {"autoresearch_tsv_v1", "openevolve_jsonl_v1", "ttt_jsonl_v1"}


def _optional_nonnegative_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or null")
    return value


def _optional_digest(value: object, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest or null")
    return value


@dataclass(frozen=True, slots=True)
class RunContext:
    generator_model: str
    evaluator_id: str
    tool_policy: str
    run_family_id: str
    starting_artifact_sha256: str | None
    prompt_sha256: str | None
    seed: int | None
    proposal_budget: int | None
    token_budget: int | None
    wall_time_budget_seconds: int | None
    accelerator_budget_seconds: int | None
    archive_reused: bool
    notes: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunContext":
        fields = {
            "generator_model", "evaluator_id", "tool_policy", "run_family_id",
            "starting_artifact_sha256", "prompt_sha256", "seed", "proposal_budget",
            "token_budget", "wall_time_budget_seconds", "accelerator_budget_seconds",
            "archive_reused", "notes",
        }
        unknown = set(payload) - fields
        missing = fields - set(payload)
        if unknown or missing:
            raise ValueError(
                f"run context schema mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        for name in ("generator_model", "evaluator_id", "tool_policy", "run_family_id"):
            if not isinstance(payload[name], str) or not payload[name]:
                raise ValueError(f"context {name} must be a non-empty string")
        if not isinstance(payload["notes"], str):
            raise ValueError("context notes must be a string")
        if type(payload["archive_reused"]) is not bool:
            raise ValueError("context archive_reused must be boolean")
        return cls(
            generator_model=payload["generator_model"],
            evaluator_id=payload["evaluator_id"],
            tool_policy=payload["tool_policy"],
            run_family_id=payload["run_family_id"],
            starting_artifact_sha256=_optional_digest(
                payload["starting_artifact_sha256"], "starting_artifact_sha256"
            ),
            prompt_sha256=_optional_digest(payload["prompt_sha256"], "prompt_sha256"),
            seed=_optional_nonnegative_int(payload["seed"], "seed"),
            proposal_budget=_optional_nonnegative_int(
                payload["proposal_budget"], "proposal_budget"
            ),
            token_budget=_optional_nonnegative_int(payload["token_budget"], "token_budget"),
            wall_time_budget_seconds=_optional_nonnegative_int(
                payload["wall_time_budget_seconds"], "wall_time_budget_seconds"
            ),
            accelerator_budget_seconds=_optional_nonnegative_int(
                payload["accelerator_budget_seconds"], "accelerator_budget_seconds"
            ),
            archive_reused=payload["archive_reused"],
            notes=payload["notes"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    source_id: str
    run_id: str
    paradigm: Paradigm
    adapter: str
    path: str
    sha256: str
    context: RunContext
    annotation_path: str | None = None
    annotation_sha256: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceSpec":
        allowed = {
            "source_id", "run_id", "paradigm", "adapter", "path", "sha256",
            "annotation_path", "annotation_sha256", "context",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown source fields: {sorted(unknown)}")
        required = allowed - {"annotation_path", "annotation_sha256"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"missing source fields: {sorted(missing)}")
        for name in ("source_id", "run_id", "adapter", "path", "sha256"):
            if not isinstance(payload[name], str) or not payload[name]:
                raise ValueError(f"{name} must be a non-empty string")
        if payload["adapter"] not in ADAPTERS:
            raise ValueError(f"unsupported adapter: {payload['adapter']}")
        digest = payload["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("source sha256 must be a lowercase SHA-256 digest")
        annotation_path = payload.get("annotation_path")
        annotation_sha = payload.get("annotation_sha256")
        if (annotation_path is None) != (annotation_sha is None):
            raise ValueError("annotation_path and annotation_sha256 must appear together")
        if annotation_path is not None and (
            not isinstance(annotation_path, str) or not annotation_path
        ):
            raise ValueError("annotation_path must be a non-empty string")
        if annotation_sha is not None and (
            not isinstance(annotation_sha, str)
            or len(annotation_sha) != 64
            or any(c not in "0123456789abcdef" for c in annotation_sha)
        ):
            raise ValueError("annotation_sha256 must be a lowercase SHA-256 digest")
        context = payload["context"]
        if not isinstance(context, Mapping):
            raise ValueError("context must be an object")
        return cls(
            source_id=payload["source_id"],
            run_id=payload["run_id"],
            paradigm=Paradigm(payload["paradigm"]),
            adapter=payload["adapter"],
            path=payload["path"],
            sha256=digest,
            context=RunContext.from_dict(context),
            annotation_path=annotation_path,
            annotation_sha256=annotation_sha,
        )


@dataclass(frozen=True, slots=True)
class StudyManifest:
    schema_version: str
    study_id: str
    accuracy_threshold: float
    external_frontier_parameters: int | None
    sources: tuple[SourceSpec, ...]

    @classmethod
    def load(cls, path: str | Path) -> "StudyManifest":
        manifest_path = Path(path)
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("manifest must be a YAML object")
        allowed = {
            "schema_version", "study_id", "accuracy_threshold",
            "external_frontier_parameters", "sources",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
        missing = allowed - set(payload)
        if missing:
            raise ValueError(f"missing manifest fields: {sorted(missing)}")
        if payload["schema_version"] != "trajectory-study-v1":
            raise ValueError("schema_version must be trajectory-study-v1")
        if not isinstance(payload["study_id"], str) or not payload["study_id"]:
            raise ValueError("study_id must be a non-empty string")
        threshold = payload["accuracy_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("accuracy_threshold must be numeric")
        threshold = float(threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError("accuracy_threshold must be in (0, 1]")
        frontier = payload["external_frontier_parameters"]
        if frontier is not None and (
            isinstance(frontier, bool) or not isinstance(frontier, int) or frontier <= 0
        ):
            raise ValueError("external_frontier_parameters must be positive or null")
        source_payloads = payload["sources"]
        if not isinstance(source_payloads, list) or not source_payloads:
            raise ValueError("sources must be a non-empty list")
        if not all(isinstance(item, Mapping) for item in source_payloads):
            raise ValueError("every source must be a YAML object")
        sources = tuple(SourceSpec.from_dict(item) for item in source_payloads)
        for attribute in ("source_id", "run_id"):
            values = [getattr(source, attribute) for source in sources]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {attribute} in manifest")
        return cls(
            schema_version=payload["schema_version"],
            study_id=payload["study_id"],
            accuracy_threshold=threshold,
            external_frontier_parameters=frontier,
            sources=sources,
        )


def resolve_frozen_file(data_root: str | Path, relative_path: str, expected_sha256: str) -> Path:
    root = Path(data_root).resolve()
    candidate = (root / relative_path).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"input path escapes data root: {relative_path}")
    if not candidate.is_file():
        raise FileNotFoundError(f"frozen input does not exist: {candidate}")
    actual = sha256_file(candidate)
    if actual != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {relative_path}: expected {expected_sha256}, got {actual}"
        )
    return candidate

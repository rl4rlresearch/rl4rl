"""Reproducible manifests for matched forks and full-trajectory studies."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable

from research_dynamics.contracts import (
    ConditionId,
    DeliberationPolicy,
    FrameworkKind,
    ProcessCondition,
    ProcessStudyConfig,
)
from study.serialization import atomic_write_json, content_hash


def path_hash(path: str | Path) -> str:
    source = Path(path).resolve()
    digest = hashlib.sha256()
    if source.is_file():
        digest.update(source.read_bytes())
        return digest.hexdigest()
    if not source.is_dir():
        raise FileNotFoundError(source)
    for child in sorted(item for item in source.rglob("*") if item.is_file()):
        digest.update(child.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _schedule(condition: ProcessCondition, schedule: tuple[int, ...]) -> tuple[int, ...]:
    return (
        schedule
        if condition.deliberation_policy is DeliberationPolicy.ASSUMPTION_CHALLENGE
        else ()
    )


def write_config(path: Path, config: ProcessStudyConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, config.to_dict())


def _prepare_plan_root(output_dir: str | Path) -> Path:
    root = Path(output_dir).resolve()
    if root.is_symlink():
        raise ValueError("plan output may not be a symlink")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"plan output is not fresh: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def plan_forks(
    *,
    study_id: str,
    framework: FrameworkKind,
    checkpoint: str | Path,
    output_dir: str | Path,
    command: list[str],
    horizon: int,
    seed: int,
    scientific: bool,
) -> Path:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_hash = path_hash(checkpoint_path)
    checkpoint_id = f"checkpoint-{checkpoint_hash[:16]}"
    root = _prepare_plan_root(output_dir)
    order = list(ConditionId)
    random.Random(seed).shuffle(order)
    branches = []
    for position, condition_id in enumerate(order):
        condition = ProcessCondition.for_id(condition_id)
        branch_dir = root / condition_id.value
        config = ProcessStudyConfig(
            study_id=study_id,
            run_id=f"{study_id}-{checkpoint_id}-{condition_id.value}",
            framework=framework,
            condition=condition,
            challenge_opportunities=_schedule(
                condition, tuple(range(1, horizon + 1))
            ),
            source_checkpoint_id=checkpoint_id,
            source_checkpoint_hash=checkpoint_hash,
            scientific=scientific,
        )
        config_path = branch_dir / "process_config.json"
        write_config(config_path, config)
        branches.append(
            {
                "position": position,
                "condition_id": condition_id.value,
                "run_id": config.run_id,
                "config_path": str(config_path),
                "config_hash": config.config_hash,
                "output_dir": str(branch_dir / "controller_run"),
                "command": command,
            }
        )
    manifest = {
        "schema_name": "ResearchProcessExecutionManifest",
        "schema_version": "1.0",
        "design": "matched_checkpoint_fork",
        "study_id": study_id,
        "framework": framework.value,
        "seed": seed,
        "horizon": horizon,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_id": checkpoint_id,
        "checkpoint_hash": checkpoint_hash,
        "branches": branches,
    }
    manifest["manifest_hash"] = content_hash(manifest)
    path = root / "fork_manifest.json"
    atomic_write_json(path, manifest)
    return path


def plan_full_trajectories(
    *,
    study_id: str,
    framework: FrameworkKind,
    output_dir: str | Path,
    command: list[str],
    blocks: int,
    first_seed: int,
    challenge_schedule: tuple[int, ...],
    scientific: bool,
) -> Path:
    if blocks < 1:
        raise ValueError("blocks must be positive")
    root = _prepare_plan_root(output_dir)
    branches = []
    for block in range(blocks):
        order = list(ConditionId)
        random.Random(first_seed + block).shuffle(order)
        for position, condition_id in enumerate(order):
            condition = ProcessCondition.for_id(condition_id)
            run_seed = first_seed + block
            run_id = f"{study_id}-b{block:03d}-{condition_id.value}-s{run_seed}"
            branch_dir = root / f"block-{block:03d}" / condition_id.value
            config = ProcessStudyConfig(
                study_id=study_id,
                run_id=run_id,
                framework=framework,
                condition=condition,
                challenge_opportunities=_schedule(condition, challenge_schedule),
                scientific=scientific,
            )
            config_path = branch_dir / "process_config.json"
            write_config(config_path, config)
            branches.append(
                {
                    "block": block,
                    "position": position,
                    "condition_id": condition_id.value,
                    "seed": run_seed,
                    "run_id": run_id,
                    "config_path": str(config_path),
                    "config_hash": config.config_hash,
                    "output_dir": str(branch_dir / "controller_run"),
                    "command": command,
                }
            )
    manifest = {
        "schema_name": "ResearchProcessExecutionManifest",
        "schema_version": "1.0",
        "design": "full_trajectory_block_randomized",
        "study_id": study_id,
        "framework": framework.value,
        "first_seed": first_seed,
        "blocks": blocks,
        "challenge_schedule": list(challenge_schedule),
        "branches": branches,
    }
    manifest["manifest_hash"] = content_hash(manifest)
    path = root / "trajectory_manifest.json"
    atomic_write_json(path, manifest)
    return path


def _render_command(
    command: Iterable[str],
    *,
    branch: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    replacements = {
        "output_dir": branch["output_dir"],
        "condition": branch["condition_id"],
        "run_id": branch["run_id"],
        "seed": str(branch.get("seed", manifest.get("seed", 0))),
        "checkpoint": manifest.get("checkpoint_path", ""),
        "horizon": str(manifest.get("horizon", "")),
    }
    return [str(part).format(**replacements) for part in command]


def execute_manifest(
    manifest_path: str | Path,
    *,
    dry_run: bool,
    continue_on_failure: bool = False,
) -> list[dict[str, Any]]:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_name") != "ResearchProcessExecutionManifest":
        raise ValueError("not a research-process execution manifest")
    expected_hash = manifest.get("manifest_hash")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if expected_hash != content_hash(unhashed):
        raise ValueError("manifest hash does not match its contents")
    checkpoint_path = manifest.get("checkpoint_path")
    if checkpoint_path and path_hash(checkpoint_path) != manifest.get("checkpoint_hash"):
        raise ValueError("source checkpoint changed after fork randomization")
    results = []
    for branch in sorted(
        manifest["branches"],
        key=lambda item: (item.get("block", 0), item["position"]),
    ):
        config_payload = json.loads(Path(branch["config_path"]).read_text(encoding="utf-8"))
        config = ProcessStudyConfig.from_dict(config_payload)
        if config.config_hash != branch.get("config_hash"):
            raise ValueError("branch process config changed after randomization")
        if (
            config.run_id != branch["run_id"]
            or config.condition.condition_id.value != branch["condition_id"]
            or config.framework.value != manifest["framework"]
        ):
            raise ValueError("branch process config differs from its manifest assignment")
        command = _render_command(branch["command"], branch=branch, manifest=manifest)
        output = Path(branch["output_dir"])
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"branch output is not fresh: {output}")
        env = dict(os.environ)
        env["RL4RL_PROCESS_CONFIG"] = branch["config_path"]
        if checkpoint_path:
            env["RL4RL_PROCESS_INITIAL_CANDIDATE"] = checkpoint_path
        result = {
            "run_id": branch["run_id"],
            "condition_id": branch["condition_id"],
            "config_path": branch["config_path"],
            "command": command,
            "dry_run": dry_run,
            "returncode": None,
        }
        if not dry_run:
            completed = subprocess.run(command, env=env, check=False)
            result["returncode"] = completed.returncode
            if completed.returncode and not continue_on_failure:
                results.append(result)
                break
        results.append(result)
    return results

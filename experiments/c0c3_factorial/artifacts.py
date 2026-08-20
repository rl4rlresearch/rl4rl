"""Portable candidate snapshots and fresh evaluation workspaces."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
from pathlib import Path

from .spec import FrameworkKind, FrameworkSpec, TaskSpec, canonical_json, sha256_json

_ENV_REFERENCE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
_IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}


def resolve_source(value: str, *, repo_root: Path) -> Path:
    match = _ENV_REFERENCE.fullmatch(value)
    if match:
        environment_value = os.environ.get(match.group(1))
        if not environment_value:
            raise ValueError(
                f"task seed requires environment variable {match.group(1)}"
            )
        source = Path(environment_value).expanduser()
    else:
        source = Path(value).expanduser()
        if not source.is_absolute():
            source = repo_root / source
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"task seed source is not a directory: {source}")
    return source


def _copy_source(source: Path, destination: Path) -> None:
    def ignored(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in _IGNORED_PARTS or name == ".DS_Store"}

    shutil.copytree(source, destination, ignore=ignored)


def prepare_seed_workspace(
    task: TaskSpec, destination: Path, *, repo_root: Path
) -> None:
    """Copy the immutable task support tree and add task-owned seed glue."""

    source = resolve_source(task.seed_source, repo_root=repo_root)
    _copy_source(source, destination)
    if task.adapter == "adderboard_v1":
        from experiments.autoresearch_pilot.create_run import SUBMISSION_WRAPPER

        (destination / "submission.py").write_text(SUBMISSION_WRAPPER, encoding="utf-8")
    for relative in task.editable_paths:
        path = destination / relative
        if not path.is_file():
            raise FileNotFoundError(f"seed is missing editable file {relative}")


def _iter_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and not any(part in _IGNORED_PARTS for part in path.parts)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _iter_files(root):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def scientific_runtime_hash(
    repo_root: Path, *, task: TaskSpec, framework: FrameworkSpec
) -> str:
    """Hash controller and repository-owned runtime code used by a campaign."""

    roots = {
        "factorial_controller": repo_root / "experiments/c0c3_factorial",
    }
    if framework.framework_id is FrameworkKind.OPENEVOLVE:
        roots["openevolve_adapter_runtime"] = (
            repo_root / "architecture_discovery/vendor/openevolve/openevolve"
        )
    if task.adapter == "adderboard_v1":
        roots["adderboard_verifier"] = (
            repo_root / "architecture_discovery/vendor/AdderBoard"
        )
    hashes = {}
    for label, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"scientific runtime root is missing: {root}")
        hashes[label] = tree_hash(root)
    return sha256_json(hashes)


def candidate_hash(workspace: Path, editable_paths: tuple[str, ...]) -> str:
    entries: list[dict[str, str]] = []
    for relative in sorted(editable_paths):
        path = workspace / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"editable candidate file is missing or unsafe: {relative}"
            )
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()


def snapshot_candidate(
    workspace: Path,
    destination_root: Path,
    editable_paths: tuple[str, ...],
) -> tuple[str, Path]:
    identifier = candidate_hash(workspace, editable_paths)
    destination = destination_root / identifier
    if destination.exists():
        if candidate_hash(destination, editable_paths) != identifier:
            raise RuntimeError("candidate hash collision")
        return identifier, destination
    temporary = destination_root / f".{identifier}.partial-{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        for relative in editable_paths:
            source = workspace / relative
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return identifier, destination


def materialize_candidate(
    support_source: Path,
    candidate_snapshot: Path,
    destination: Path,
    editable_paths: tuple[str, ...],
) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    _copy_source(support_source, destination)
    for relative in editable_paths:
        source = candidate_snapshot / relative
        target = destination / relative
        if not source.is_file():
            raise FileNotFoundError(f"snapshot is missing {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def make_read_only(root: Path) -> None:
    for path in [*_iter_files(root), *sorted(root.rglob("*"), reverse=True)]:
        mode = path.stat().st_mode
        if path.is_dir():
            path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        else:
            path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def editable_hashes(workspace: Path, editable_paths: tuple[str, ...]) -> dict[str, str]:
    return {
        relative: hashlib.sha256((workspace / relative).read_bytes()).hexdigest()
        for relative in editable_paths
    }


def protected_hash(workspace: Path, editable_paths: tuple[str, ...]) -> str:
    editable = {Path(value).as_posix() for value in editable_paths}
    digest = hashlib.sha256()
    for path in _iter_files(workspace):
        relative = path.relative_to(workspace).as_posix()
        if relative in editable or relative.startswith(".codex-"):
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

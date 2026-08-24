#!/usr/bin/env python3
"""Verify one downloaded immutable artifact bundle against its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, expected_run_id: str) -> None:
    absolute = Path(os.path.abspath(root))
    if absolute.is_symlink() or not absolute.is_dir():
        raise ValueError("bundle root must be a real directory")
    manifest_path = absolute / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_name") != "ModalRunArtifactManifest" or manifest.get("run_id") != expected_run_id:
        raise ValueError("artifact manifest identity is invalid")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("artifact manifest file roster is empty")
    expected: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"relative_path", "sha256", "size_bytes"}:
            raise ValueError("artifact manifest file record is invalid")
        relative = record["relative_path"]
        logical = PurePosixPath(relative)
        if logical.is_absolute() or ".." in logical.parts or relative in expected:
            raise ValueError("artifact manifest contains an unsafe or duplicate path")
        expected.add(relative)
        path = absolute.joinpath(*logical.parts)
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise ValueError(f"artifact is not a regular file: {relative}")
        if metadata.st_size != record["size_bytes"]:
            raise ValueError(f"artifact size differs: {relative}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"artifact digest differs: {relative}")
    actual = {
        path.relative_to(absolute).as_posix()
        for path in absolute.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"artifact roster differs; missing={missing[:3]} extra={extra[:3]}")
    print(f"VERIFIED {expected_run_id}: {len(expected)} files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    verify(args.bundle, args.run_id)


if __name__ == "__main__":
    main()

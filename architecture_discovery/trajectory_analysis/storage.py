"""Canonical, atomic storage helpers for trajectory artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from study.serialization import canonical_json, json_value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: str | Path, value: Any) -> None:
    content = json.dumps(
        json_value(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    _atomic_text(Path(path), content)


def write_text(path: str | Path, content: str) -> None:
    _atomic_text(Path(path), content)


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    content = "".join(canonical_json(record) + "\n" for record in records)
    _atomic_text(Path(path), content)


def write_csv(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    temporary = Path(path).with_suffix(Path(path).suffix + ".render")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: json_value(value) for key, value in row.items()} for row in records])
    _atomic_text(Path(path), temporary.read_text(encoding="utf-8"))
    temporary.unlink()


def require_new_output_directory(path: str | Path) -> Path:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(
            f"output directory already exists: {destination}; choose a new directory"
        )
    destination.mkdir(parents=True)
    return destination

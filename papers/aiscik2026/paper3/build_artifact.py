#!/usr/bin/env python3
"""Build an anonymized, source-complete Paper 3 reproducibility archive."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUTPUT = REPO / "output/paper3_reproducibility_artifact.zip"
ARCHIVE_ROOT = "paper3_population_memory_artifact"

CAMPAIGNS = (
    (REPO / "data/c0c3/unified-v3-tiny-adderboard-greedy-campaign", 70),
    (REPO / "data/c0c3/unified-v3-tiny-adderboard-native-campaign", 70),
    (REPO / "data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign", 200),
)
PROMPT_SNAPSHOT = 10

TOP_LEVEL_FILES = {
    "campaign.json",
    "campaign-amendments.jsonl",
    "campaign-lifecycle.jsonl",
    "DATA_DICTIONARY.md",
    "environment-receipt.json",
    "paired-prefix-events.jsonl",
    "paired-prefix.json",
    "schedule.json",
    "trajectory-lifecycle.jsonl",
    "v3-runtime-history.jsonl",
    "validation.json",
}
TOP_LEVEL_DIRECTORIES = {"amendments", "inputs", "prompt-bundle"}
RUN_FILES = {"events.jsonl", "lifecycle.jsonl", "manifest.json", "state.json"}

LICENSE = """MIT License

Copyright (c) 2026 Anonymous Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this artifact and its included software, trace records, candidate source
snapshots, data, and documentation files (the "Materials"), to deal in the
Materials without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Materials, and to permit persons to whom the Materials are
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS IN THE
MATERIALS.
"""

CANDIDATE_ID_RE = re.compile(r"^[0-9a-f]{64}$")


def sanitize_text(text: str) -> str:
    home = Path.home()
    replacements = {
        str(REPO): "REPOSITORY_ROOT",
        str(home): "/home/anonymous",
        home.name: "anonymous",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = source.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        shutil.copy2(source, destination)
    else:
        destination.write_text(sanitize_text(text), encoding="utf-8")


def copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if (
            not path.is_file()
            or path.name.startswith(".")
            or "__pycache__" in path.parts
        ):
            continue
        copy_file(path, destination / path.relative_to(source))


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSONL object: {path}")
                rows.append(value)
    return rows


def candidate_ids_for_horizon(
    manifest: dict[str, Any], events: list[dict[str, Any]], horizon: int
) -> set[str]:
    candidate_ids: set[str] = set()
    baseline = manifest.get("baseline") or {}
    if baseline.get("candidate_id"):
        candidate_ids.add(str(baseline["candidate_id"]))
    for event in events:
        opportunity = int(event.get("opportunity") or 0)
        if event.get("event") != "proposal_completed" or opportunity > horizon:
            continue
        for key in ("candidate_id", "incumbent_after", "evicted_candidate_id"):
            value = str(event.get(key) or "")
            if CANDIDATE_ID_RE.fullmatch(value):
                candidate_ids.add(value)
        for key in (
            "selected_parent_ids",
            "parent_ids",
            "visible_candidate_ids",
            "portfolio_after",
        ):
            candidate_ids.update(
                str(value)
                for value in (event.get(key) or [])
                if CANDIDATE_ID_RE.fullmatch(str(value))
            )
    invalid = [value for value in candidate_ids if not CANDIDATE_ID_RE.fullmatch(value)]
    if invalid:
        raise ValueError(f"Unsafe candidate identifiers: {invalid[:3]}")
    return candidate_ids


def copy_campaign(source: Path, destination: Path, horizon: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(TOP_LEVEL_FILES):
        path = source / name
        if path.exists():
            copy_file(path, destination / name)
    for name in sorted(TOP_LEVEL_DIRECTORIES):
        path = source / name
        if path.exists():
            copy_tree(path, destination / name)

    run_dirs = sorted(
        path for path in (source / "runs").iterdir() if path.is_dir()
    )
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        events_path = run_dir / "events.jsonl"
        if not manifest_path.is_file() or not events_path.is_file():
            continue
        run_destination = destination / "runs" / run_dir.name
        for name in sorted(RUN_FILES):
            path = run_dir / name
            if path.exists():
                copy_file(path, run_destination / name)

        native_events = run_dir / "native-openevolve/events.jsonl"
        native_config = run_dir / "native-openevolve/config.json"
        for path in (native_events, native_config):
            if path.exists():
                copy_file(path, run_destination / path.relative_to(run_dir))

        manifest = load_json(manifest_path)
        events = load_events(events_path)
        candidate_ids = candidate_ids_for_horizon(manifest, events, horizon)
        for candidate_id in sorted(candidate_ids):
            train = run_dir / "candidates" / candidate_id / "train.py"
            if train.exists():
                candidate_destination = (
                    run_destination / "candidates" / candidate_id / "train.py"
                )
                copy_file(train, candidate_destination)

        opportunities = run_dir / "opportunities"
        if not opportunities.exists():
            continue
        opportunity_dirs = sorted(
            path for path in opportunities.iterdir() if path.is_dir()
        )
        for opportunity_dir in opportunity_dirs:
            try:
                opportunity = int(opportunity_dir.name)
            except ValueError:
                continue
            if opportunity > horizon:
                continue
            opportunity_destination = (
                run_destination / "opportunities" / opportunity_dir.name
            )
            provenance = opportunity_dir / "candidate-provenance.json"
            if provenance.exists():
                copy_file(provenance, opportunity_destination / provenance.name)
            if opportunity == PROMPT_SNAPSHOT:
                prompt = opportunity_dir / "prompt.md"
                if prompt.exists():
                    copy_file(prompt, opportunity_destination / prompt.name)
            message = (
                opportunity_dir
                / "codex"
                / f"proposal-{opportunity}.last-message.md"
            )
            if message.exists():
                copy_file(message, opportunity_destination / "codex" / message.name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = REPO / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="paper3-artifact-", dir=temp_parent
    ) as temp:
        root = Path(temp) / ARCHIVE_ROOT
        paper = root / "papers/aiscik2026/paper3"
        paper.mkdir(parents=True, exist_ok=True)
        for name in ("analysis.py", "ARTIFACT_README.md", "requirements.txt"):
            copy_file(HERE / name, paper / name)
        copy_tree(HERE / "derived", paper / "derived")
        for campaign, horizon in CAMPAIGNS:
            copy_campaign(campaign, root / campaign.relative_to(REPO), horizon)
        (root / "LICENSE").write_text(LICENSE, encoding="utf-8")

        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = str(Path(temp) / "matplotlib-cache")
        subprocess.run(
            [
                sys.executable,
                str(paper / "analysis.py"),
                "--data-root",
                str(root),
            ],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        payload = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.name != "PAPER3_SHA256SUMS"
        )
        checksum_lines = [
            f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in payload
        ]
        (root / "PAPER3_SHA256SUMS").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )

        temporary_zip = Path(temp) / OUTPUT.name
        with zipfile.ZipFile(
            temporary_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(ARCHIVE_ROOT) / path.relative_to(root))
        shutil.copy2(temporary_zip, OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())

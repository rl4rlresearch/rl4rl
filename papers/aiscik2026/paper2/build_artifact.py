#!/usr/bin/env python3
"""Build an anonymized, source-complete Paper 2 reproducibility archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUTPUT = REPO / "output/paper2_reproducibility_artifact.zip"
ARCHIVE_ROOT = "paper2_state_matched_defixation_artifact"

CAMPAIGNS = (
    (REPO / "data/c0c3/unified-v3-tiny-adderboard-greedy-campaign", 70),
    (REPO / "data/c0c3/unified-v3-tiny-adderboard-native-campaign", 70),
    (REPO / "data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign", 200),
)
FORK = 10

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
OPPORTUNITY_FILES = {"candidate-provenance.json"}

LICENSE = """MIT License

Copyright (c) 2026 Anonymous Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


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
        return json.load(stream)


def load_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def candidate_ids_for_horizon(
    manifest: dict[str, Any], events: list[dict[str, Any]], horizon: int
) -> set[str]:
    candidate_ids: set[str] = set()
    baseline = manifest.get("baseline") or {}
    if baseline.get("candidate_id"):
        candidate_ids.add(str(baseline["candidate_id"]))
    for event in events:
        if event.get("event") != "proposal_completed":
            continue
        if int(event.get("opportunity") or 0) > horizon:
            continue
        for key in ("candidate_id", "incumbent_after"):
            if event.get(key):
                candidate_ids.add(str(event[key]))
        for key in ("selected_parent_ids", "parent_ids"):
            candidate_ids.update(
                str(value) for value in (event.get(key) or []) if value
            )
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

    source_runs = source / "runs"
    destination_runs = destination / "runs"
    for run_dir in sorted(path for path in source_runs.iterdir() if path.is_dir()):
        manifest_path = run_dir / "manifest.json"
        events_path = run_dir / "events.jsonl"
        if not manifest_path.exists() or not events_path.exists():
            continue
        run_destination = destination_runs / run_dir.name
        for name in sorted(RUN_FILES):
            path = run_dir / name
            if path.exists():
                copy_file(path, run_destination / name)

        manifest = load_json(manifest_path)
        events = load_events(events_path)
        for candidate_id in sorted(
            candidate_ids_for_horizon(manifest, events, horizon)
        ):
            train = run_dir / "candidates" / candidate_id / "train.py"
            if train.exists():
                copy_file(
                    train,
                    run_destination / "candidates" / candidate_id / "train.py",
                )

        opportunities = run_dir / "opportunities"
        if not opportunities.exists():
            continue
        for opportunity_dir in sorted(
            path for path in opportunities.iterdir() if path.is_dir()
        ):
            try:
                opportunity = int(opportunity_dir.name)
            except ValueError:
                continue
            if opportunity > horizon:
                continue
            opportunity_destination = (
                run_destination / "opportunities" / opportunity_dir.name
            )
            for name in sorted(OPPORTUNITY_FILES):
                path = opportunity_dir / name
                if path.exists():
                    copy_file(path, opportunity_destination / name)
            if opportunity == FORK:
                prompt = opportunity_dir / "prompt.md"
                if prompt.exists():
                    copy_file(prompt, opportunity_destination / prompt.name)
            codex = opportunity_dir / "codex"
            for suffix in ("last-message.md",):
                path = codex / f"proposal-{opportunity}.{suffix}"
                if path.exists():
                    copy_file(
                        path,
                        opportunity_destination / "codex" / path.name,
                    )


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
        prefix="paper2-artifact-", dir=temp_parent
    ) as temp:
        root = Path(temp) / ARCHIVE_ROOT
        paper = root / "papers/aiscik2026/paper2"
        paper.mkdir(parents=True, exist_ok=True)
        for name in (
            "analysis.py",
            "ARTIFACT_README.md",
            "requirements.txt",
        ):
            copy_file(HERE / name, paper / name)
        copy_tree(HERE / "derived", paper / "derived")
        for campaign, horizon in CAMPAIGNS:
            copy_campaign(
                campaign,
                root / campaign.relative_to(REPO),
                horizon,
            )
        (root / "LICENSE").write_text(LICENSE, encoding="utf-8")

        payload = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.name != "PAPER2_SHA256SUMS"
        )
        checksum_lines = [
            f"{sha256(path)}  {path.relative_to(root).as_posix()}"
            for path in payload
        ]
        (root / "PAPER2_SHA256SUMS").write_text(
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
                    archive.write(
                        path,
                        Path(ARCHIVE_ROOT) / path.relative_to(root),
                    )
        shutil.copy2(temporary_zip, OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())

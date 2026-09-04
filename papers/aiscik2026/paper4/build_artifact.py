#!/usr/bin/env python3
"""Build an anonymized Paper 4 reproducibility archive."""

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
OUTPUT = REPO / "output/paper4_reproducibility_artifact.zip"
ARCHIVE_ROOT = "paper4_history_refresh_artifact"

CAMPAIGNS = (
    (REPO / "data/c0c3/semantic-interventions-v4-fashion-openevolve-campaign", 43),
    (REPO / "data/c0c3/semantic-interventions-v4-fashion-native-openevolve-campaign", 13),
    (REPO / "data/c0c3/semantic-interventions-v4-tiny-adderboard-terra-campaign", 92),
    (REPO / "data/c0c3/semantic-interventions-v4-tiny-adderboard-native-openevolve-terra-campaign", 50),
)
FOCAL_ARMS = {"passive_control", "periodic_full_refresh"}
TOP_LEVEL_FILES = {
    "campaign.json",
    "campaign-amendments.jsonl",
    "campaign-lifecycle.jsonl",
    "DATA_DICTIONARY.md",
    "environment-receipt.json",
    "semantic-interventions.json",
    "trajectory-lifecycle.jsonl",
    "validation.json",
}
TOP_LEVEL_DIRECTORIES = {"inputs", "prompt-bundle"}
RUN_FILES = {
    "events.jsonl",
    "lifecycle.jsonl",
    "manifest.json",
    "state.json",
    "developmental-archive-resets.jsonl",
}
OPPORTUNITY_FILES = {
    "candidate-provenance.json",
    "prompt.md",
    "prompt-manifest.json",
    "state-capsule.json",
    "result.json",
    "evaluation.json",
}

CANDIDATE_ID_RE = re.compile(r"^[0-9a-f]{64}$")

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
copies or substantial portions of the Materials.

THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS IN THE
MATERIALS.
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
        if not path.is_file() or path.name.startswith(".") or "__pycache__" in path.parts:
            continue
        copy_file(path, destination / path.relative_to(source))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSONL object: {path}")
        rows.append(value)
    return rows


def focal_run_dirs(campaign: Path) -> list[Path]:
    output: list[Path] = []
    for run_dir in sorted((campaign / "runs").iterdir()):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        assignment = load_json(manifest_path).get("assignment") or {}
        if assignment.get("condition") in FOCAL_ARMS:
            output.append(run_dir)
    if len(output) != 6:
        raise ValueError(f"Expected six focal runs in {campaign}, found {len(output)}")
    return output


def all_run_dirs(campaign: Path) -> list[Path]:
    output = [path for path in sorted((campaign / "runs").iterdir()) if (path / "manifest.json").is_file() and (path / "events.jsonl").is_file()]
    if len(output) != 69:
        raise ValueError(f"Expected 69 semantic-condition runs in {campaign}, found {len(output)}")
    return output


def candidate_ids_for_horizon(manifest: dict[str, Any], events: list[dict[str, Any]], horizon: int) -> set[str]:
    candidate_ids: set[str] = set()
    baseline = manifest.get("baseline") or {}
    if baseline.get("candidate_id"):
        candidate_ids.add(str(baseline["candidate_id"]))
    for event in events:
        if event.get("event") != "proposal_completed":
            continue
        opportunity = int(event.get("opportunity") or 0)
        if opportunity > horizon:
            continue
        for key in ("candidate_id", "incumbent_before", "incumbent_after", "evicted_candidate_id"):
            value = str(event.get(key) or "")
            if CANDIDATE_ID_RE.fullmatch(value):
                candidate_ids.add(value)
        for key in ("selected_parent_ids", "parent_ids", "visible_candidate_ids", "portfolio_after"):
            for value in event.get(key) or []:
                value = str(value)
                if CANDIDATE_ID_RE.fullmatch(value):
                    candidate_ids.add(value)
    return candidate_ids


def copy_opportunity(source: Path, destination: Path, opportunity: int) -> None:
    opportunity_dir = source / "opportunities" / f"{opportunity:04d}"
    if not opportunity_dir.is_dir():
        return
    destination_dir = destination / "opportunities" / f"{opportunity:04d}"
    for name in sorted(OPPORTUNITY_FILES):
        path = opportunity_dir / name
        if path.is_file():
            copy_file(path, destination_dir / name)
    message = opportunity_dir / "codex" / f"proposal-{opportunity}.last-message.md"
    if message.is_file():
        copy_file(message, destination_dir / "codex" / message.name)
    eval_source = opportunity_dir / "evaluation-workspace" / "train.py"
    if eval_source.is_file():
        copy_file(eval_source, destination_dir / "evaluation-workspace" / "train.py")


def copy_campaign(source: Path, destination: Path, horizon: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(TOP_LEVEL_FILES):
        path = source / name
        if path.exists():
            copy_file(path, destination / name)
    for name in sorted(TOP_LEVEL_DIRECTORIES):
        path = source / name
        if path.is_dir():
            copy_tree(path, destination / name)
    for run_dir in all_run_dirs(source):
        run_destination = destination / "runs" / run_dir.name
        for name in sorted(RUN_FILES):
            path = run_dir / name
            if path.exists():
                copy_file(path, run_destination / name)
        manifest = load_json(run_dir / "manifest.json")
        assignment = manifest.get("assignment") or {}
        is_focal = assignment.get("condition") in FOCAL_ARMS
        if not is_focal:
            continue
        native_root = run_dir / "native-openevolve"
        if native_root.is_dir():
            for name in ("events.jsonl", "config.json", "checkpoint.json"):
                path = native_root / name
                if path.is_file():
                    copy_file(path, run_destination / "native-openevolve" / name)
        events = load_jsonl(run_dir / "events.jsonl")
        for candidate_id in sorted(candidate_ids_for_horizon(manifest, events, horizon)):
            train = run_dir / "candidates" / candidate_id / "train.py"
            if train.is_file():
                copy_file(train, run_destination / "candidates" / candidate_id / "train.py")
        for opportunity in range(1, horizon + 1):
            copy_opportunity(run_dir, run_destination, opportunity)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(root: Path) -> None:
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "PAPER4_SHA256SUMS":
            rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "PAPER4_SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def verify_checksums(root: Path) -> None:
    failures: list[str] = []
    for line in (root / "PAPER4_SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            failures.append(relative)
    if failures:
        raise ValueError(f"Checksum verification failed for {failures[:10]}")


def run_checked(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)


def privacy_scan(root: Path) -> None:
    needles = [Path.home().name, str(Path.home()), str(REPO), "OPENAI_API_KEY", "MODAL_TOKEN"]
    secret_patterns = [re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")]
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "PAPER4_SHA256SUMS":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in needles:
            if needle and needle in text:
                hits.append(f"{path.relative_to(root)} contains {needle}")
                break
        else:
            for pattern in secret_patterns:
                if pattern.search(text):
                    hits.append(f"{path.relative_to(root)} contains API-key-shaped token")
                    break
    if hits:
        raise ValueError("Privacy scan failed:\n" + "\n".join(hits[:20]))


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paper4_artifact_") as temporary:
        staging = Path(temporary) / ARCHIVE_ROOT
        (staging / "papers/aiscik2026/paper4").mkdir(parents=True)
        for name in ("analysis.py", "ARTIFACT_README.md", "requirements.txt"):
            copy_file(HERE / name, staging / "papers/aiscik2026/paper4" / name)
        (staging / "LICENSE").write_text(LICENSE, encoding="utf-8")
        for campaign, horizon in CAMPAIGNS:
            copy_campaign(campaign, staging / campaign.relative_to(REPO), horizon)

        analysis = staging / "papers/aiscik2026/paper4/analysis.py"
        env = os.environ.copy()
        env.setdefault("XDG_CACHE_HOME", "/tmp/rl4rl-paper4-cache")
        env.setdefault("MPLCONFIGDIR", "/tmp/rl4rl-paper4-mpl")
        run_checked([sys.executable, str(analysis), "--data-root", str(staging)], staging, env)
        run_checked([sys.executable, str(analysis), "--data-root", str(staging), "--verify-input-hashes"], staging, env)
        privacy_scan(staging)
        write_checksums(staging)
        verify_checksums(staging)

        if OUTPUT.exists():
            OUTPUT.unlink()
        with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging.parent).as_posix())
    return OUTPUT


if __name__ == "__main__":
    print(build())

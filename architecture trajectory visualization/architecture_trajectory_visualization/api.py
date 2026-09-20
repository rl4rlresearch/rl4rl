"""Read-only HTTP facade with source-coherent caches and portable replay export."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from architecture_trajectory_visualization.graph import extract_architecture
from architecture_trajectory_visualization.replay import (
    _run_path,
    discover_catalog,
    load_run_replay,
    resolve_candidate_source,
    validate_architecture_bundle,
)


def _stat(path: Path) -> tuple[Any, ...]:
    try:
        value = path.stat()
        return (str(path), value.st_mtime_ns, value.st_ctime_ns, value.st_size)
    except OSError:
        return (str(path), None)


def _prepared_receipt(root: Path) -> dict[str, Any] | None:
    path = root / "architecture-replay-source.json"
    try:
        if path.is_symlink() or path.stat().st_size > 1_000_000:
            return None
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(receipt, dict)
        or receipt.get("checksum_verified") is not True
        or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("archive_sha256", "")))
        or not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("source_revision", "")))
    ):
        return None
    return receipt


class ArchitectureStore:
    def __init__(self, repo_root: Path, campaigns: dict[str, Path]):
        self.root = repo_root.resolve()
        self.campaigns = campaigns
        self.graphs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.runs: OrderedDict[tuple, tuple[tuple, dict[str, Any]]] = OrderedDict()
        self.lock = threading.RLock()
        self.catalogs: dict[str, tuple[float, tuple, dict, dict]] = {}

    def _catalog_signature(self) -> tuple:
        roots = [self.root]
        prepared = self.root / "outputs/architecture-replay-data"
        roots.extend(prepared / revision for revision in ("filtered", "tiny", "har"))
        signature = []
        for root in roots:
            data = root / "data/c0c3"
            signature.extend(
                (_stat(data), _stat(root / "architecture-replay-source.json"))
            )
            if data.is_dir():
                signature.extend(
                    _stat(path / "runs") for path in sorted(data.iterdir())
                )
        return tuple(signature)

    def _catalog(
        self, scope: str
    ) -> tuple[dict[str, Any], dict[str, tuple[Path, Path]]]:
        if scope not in {"dashboard", "archive"}:
            raise ValueError("scope must be dashboard or archive")
        signature = self._catalog_signature()
        with self.lock:
            cached = self.catalogs.get(scope)
            if cached and cached[1] == signature and time.monotonic() - cached[0] < 2:
                return cached[2], cached[3]
        payload = discover_catalog(self.root, scope=scope)
        aliases = {Path(path).name: key for key, path in self.campaigns.items()}
        locations: dict[str, tuple[Path, Path]] = {}
        by_name = {Path(row["path"]).name: row for row in payload["campaigns"]}
        for name, row in by_name.items():
            row.update(
                key=aliases.get(name, row["key"]),
                data_origin="tracked checkout",
                source_revision=payload["source_revision"],
                dataset_provenance=payload.get("dataset_provenance")
                or {
                    "source_kind": "tracked_checkout",
                    "source_revision": payload["source_revision"],
                    "campaign_path": row["path"],
                },
            )
            locations[row["key"]] = (self.root, self.root / row["path"])
        tracked_by_name = dict(by_name)
        raw_variants: dict[str, dict[str, Any]] = {}
        # Replace one complete older revision; never splice its occurrences.
        prepared = self.root / "outputs/architecture-replay-data"
        for revision in ("filtered", "tiny", "har"):
            base = prepared / revision
            receipt = _prepared_receipt(base)
            if receipt is None or not (base / "data/c0c3").is_dir():
                continue
            restored = discover_catalog(base, scope=scope)
            for row in restored["campaigns"]:
                if not row["available"]:
                    continue
                name = Path(row["path"]).name
                tracked = tracked_by_name.get(name)
                if scope == "archive" and tracked and tracked["available"]:
                    raw_key = f"{aliases.get(name, name)}__tracked_archive"
                    raw_variants[name] = {
                        **tracked,
                        "key": raw_key,
                        "version_label": "Raw checkout history",
                    }
                    locations[raw_key] = (self.root, self.root / tracked["path"])
                row.update(
                    key=aliases.get(name, name),
                    data_origin=f"verified GitHub archive / {revision}",
                    source_revision=restored["source_revision"],
                    dataset_provenance=receipt,
                )
                locations[row["key"]] = (base, base / row["path"])
                by_name[name] = row
        payload["campaigns"] = [*by_name.values(), *raw_variants.values()]
        with self.lock:
            self.catalogs[scope] = (time.monotonic(), signature, payload, locations)
        return payload, locations

    def catalog(self, scope: str = "dashboard") -> dict[str, Any]:
        return self._catalog(scope)[0]

    def location(self, campaign: str, scope: str) -> tuple[Path, Path]:
        _, locations = self._catalog(scope)
        if campaign not in locations:
            raise KeyError("Unknown campaign in this scope")
        return locations[campaign]

    def _run_signature(self, root: Path, path: Path, run_id: str) -> tuple:
        run = _run_path(root, path, run_id)
        files = [run / name for name in ("manifest.json", "state.json", "events.jsonl")]
        files.extend(path / "inputs" / name for name in ("task.json", "framework.json"))
        files.extend(
            (
                root / "architecture-replay-source.json",
                self.root / "outputs/ontology-categorical-v1/publication-manifest.json",
            )
        )
        try:
            task = json.loads((path / "inputs/task.json").read_text())
            editable = task.get("editable_paths", [])
            editable = editable if isinstance(editable, list) else []
        except (OSError, ValueError, AttributeError):
            editable = []
        for directory in (run / "opportunities").glob("[0-9]*"):
            if directory.is_dir():
                files.extend(
                    directory / name
                    for name in ("result.json", "candidate-provenance.json")
                )
                # Interrupted artifact aliases depend on exact saved workspace
                # contents, not just the immutable candidate snapshot bytes.
                workspace = directory / "proposal-workspace"
                files.append(workspace)
                for relative in editable:
                    if isinstance(relative, str):
                        target = (workspace / relative).resolve()
                        if target.is_relative_to(workspace.resolve()):
                            files.append(target)
        for source in (run / "candidates").rglob("*"):
            if source.suffix == ".py" or source.name in {
                "architecture.json",
                "arch.json",
                "architecture_ir.json",
            }:
                files.append(source)
        return tuple(_stat(file) for file in sorted(files))

    def _run_at(
        self, root: Path, path: Path, campaign: str, run_id: str, scope: str
    ) -> dict[str, Any]:
        key = (str(path), run_id, scope)
        signature = self._run_signature(root, path, run_id)
        with self.lock:
            cached = self.runs.get(key)
            if cached and cached[0] == signature:
                self.runs.move_to_end(key)
                return cached[1]
        payload = load_run_replay(root, path, run_id, scope=scope)
        payload["campaign_key"] = campaign
        from architecture_trajectory_visualization.annotations import enrich_run

        payload = enrich_run(self.root, payload)
        if signature == self._run_signature(root, path, run_id):
            with self.lock:
                self.runs[key] = (signature, payload)
                self.runs.move_to_end(key)
                if len(self.runs) > 24:
                    self.runs.popitem(last=False)
        return payload

    def run(
        self, campaign: str, run_id: str, scope: str = "dashboard"
    ) -> dict[str, Any]:
        root, path = self.location(campaign, scope)
        return self._run_at(root, path, campaign, run_id, scope)

    def _snapshot_at(
        self, root: Path, path: Path, run: dict[str, Any], occurrence: dict[str, Any]
    ) -> dict[str, Any]:
        source = resolve_candidate_source(
            root,
            path,
            run["run_id"],
            occurrence["candidate_id"],
            artifact_path=occurrence.get("artifact_path"),
            proposal=occurrence.get("proposal"),
            parent_ids=occurrence.get("parent_ids"),
            failure_kind=occurrence.get("failure_kind"),
        )
        if source["source_hash"] != occurrence["source"].get("source_hash"):
            raise ValueError("Source changed during replay; reload this run")
        extractor = hashlib.sha256(
            Path(__file__).with_name("graph.py").read_bytes()
        ).hexdigest()
        key = f"{extractor}:{source['source_hash']}"
        with self.lock:
            graph = self.graphs.get(key)
            if graph is not None:
                self.graphs.move_to_end(key)
        if graph is None:
            graph = extract_architecture(
                source["sources"],
                source_hash=source["source_hash"],
                ir=source.get("ir"),
            )
            with self.lock:
                self.graphs[key] = graph
                if len(self.graphs) > 384:
                    self.graphs.popitem(last=False)
        metadata = {k: v for k, v in source.items() if k not in {"sources", "ir"}}
        return {
            **graph,
            "occurrence_id": occurrence["id"],
            "candidate_id": occurrence["candidate_id"],
            "source_id": source.get("source_candidate_id")
            if source["available"]
            else None,
            "source_hash": source["source_hash"],
            "source_metadata": metadata,
            "source": metadata,
            "source_revision": run["source_revision"],
            "run_revision": run["revision"],
            "extractor_revision": extractor[:16],
        }

    def snapshot(
        self,
        campaign: str,
        run_id: str,
        proposal: int,
        scope: str = "dashboard",
        *,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        root, path = self.location(campaign, scope)
        run = self._run_at(root, path, campaign, run_id, scope)
        if expected_revision is not None and expected_revision != run["revision"]:
            raise ValueError("Data revision changed; reload this run")
        occurrence = next(
            (row for row in run["occurrences"] if row["proposal"] == proposal), None
        )
        if occurrence is None:
            raise KeyError("Unknown occurrence in this scope")
        return self._snapshot_at(root, path, run, occurrence)

    def export(
        self,
        campaign: str,
        run_id: str,
        scope: str = "dashboard",
        *,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        root, path = self.location(campaign, scope)
        run = self._run_at(root, path, campaign, run_id, scope)
        if expected_revision is not None and expected_revision != run["revision"]:
            raise ValueError("Data revision changed; reload this run")
        payload = {
            "schema_version": "architecture-bundle/1",
            "run": run,
            "snapshots": {
                row["id"]: self._snapshot_at(root, path, run, row)
                for row in run["occurrences"]
            },
        }
        validate_architecture_bundle(payload)
        return payload


def encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()

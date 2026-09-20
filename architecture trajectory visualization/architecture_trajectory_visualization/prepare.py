#!/usr/bin/env python3
"""Fetch verified GitHub trajectory archives and prepare isolated replay datasets.

Run ``python "architecture trajectory visualization/run.py" prepare --campaign all``
from the repository. This needs only Python's standard library, not Git LFS. Candidate
programs are copied as data and never imported or executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from architecture_trajectory_visualization.paths import REPO_ROOT
from architecture_trajectory_visualization.replay import prepare_archive, verify_archive

ARCHIVES = {
    "tiny": "rl4rl-tiny-adderboard-through-100-c0-c3-20260908.zip",
    "har": "rl4rl-uci-har-c0-c3-20260908.zip",
    "filtered": "rl4rl-c0c3-dashboard-filtered-20260906.zip",
}
LFS_BATCH = "https://github.com/rl4rlresearch/rl4rl.git/info/lfs/objects/batch"
CHUNK_BYTES = 4 * 1024 * 1024
SOURCE_REVISION = "65f4aac20963bac4d293879c4e3c2ce96070edfb"


def _json_write(path: Path, value: Any) -> None:
    """Checkpoint replacement is atomic; interrupted writes cannot erase progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_spec(
    repo: Path, campaign: str, revision: str = SOURCE_REVISION
) -> dict[str, Any]:
    """Read the source revision's LFS identity, not a mutable URL or local guess."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--revision must be a full lowercase Git commit SHA")
    filename = ARCHIVES[campaign]
    relative = f"data/{filename}"
    try:
        pointer = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{revision}:{relative}"],
            stderr=subprocess.PIPE,
            timeout=20,
        ).decode("utf-8")
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise ValueError(
            f"Cannot read the tracked LFS identity for {relative} at {revision}"
        ) from error
    oid = re.search(r"^oid sha256:([0-9a-f]{64})$", pointer, re.MULTILINE)
    size = re.search(r"^size (\d+)$", pointer, re.MULTILINE)
    if (
        not pointer.startswith("version https://git-lfs.github.com/spec/v1\n")
        or not oid
        or not size
    ):
        raise ValueError(f"Expected a Git LFS pointer at {relative}")
    return {
        "campaign": campaign,
        "filename": filename,
        "oid": oid.group(1),
        "size": int(size.group(1)),
        "revision": revision,
    }


def download_actions(specs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not specs:
        return {}
    request = urllib.request.Request(
        LFS_BATCH,
        data=json.dumps(
            {
                "operation": "download",
                "transfers": ["basic"],
                "objects": [{"oid": s["oid"], "size": s["size"]} for s in specs],
            }
        ).encode(),
        headers={
            "Content-Type": "application/vnd.git-lfs+json",
            "Accept": "application/vnd.git-lfs+json",
            "User-Agent": "RL4RL-Architecture-Replay-Preparation",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    actions = {}
    for item in payload.get("objects", []):
        action = item.get("actions", {}).get("download")
        if item.get("error") or not action:
            raise ValueError(f"GitHub LFS has no public payload for {item.get('oid')}")
        if not str(action.get("href", "")).startswith("https://"):
            raise ValueError("GitHub returned a non-HTTPS archive URL")
        actions[item["oid"]] = action
    if any(s["oid"] not in actions for s in specs):
        raise ValueError("GitHub omitted a requested archive from its LFS response")
    return actions


class ArchiveLock:
    """Avoid two preparation processes writing the same resumable archive."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> ArchiveLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if os.name == "nt":
                # os.kill(pid, 0) is not a safe liveness probe on Windows.
                raise ValueError(
                    f"Download lock exists: {self.path}. Confirm no preparation "
                    "is running before removing a stale lock."
                )
            try:
                pid = int(json.loads(self.path.read_text())["pid"])
                os.kill(pid, 0)
            except ProcessLookupError:
                self.path.unlink()
            except (
                ValueError,
                KeyError,
                json.JSONDecodeError,
                PermissionError,
            ) as error:
                raise ValueError(
                    f"Cannot establish ownership of download lock {self.path}"
                ) from error
            else:
                raise ValueError(f"Archive is already being prepared by process {pid}")
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump(
                {"pid": os.getpid(), "started_at": datetime.now(UTC).isoformat()},
                stream,
            )
        return self

    def __exit__(self, *_: Any) -> None:
        self.path.unlink(missing_ok=True)


def download_archives(
    specs: list[dict[str, Any]], directory: Path, *, workers: int = 24
) -> dict[str, Path]:
    """Resume bounded byte ranges, then verify the complete immutable object.

    A .part file without a checkpoint is treated as a contiguous prefix from a
    previous stream. No bytes are considered trusted until the final SHA-256
    succeeds. Range offsets/lengths and checkpoints are validated on every run.
    """
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    directory.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    result = {}
    jobs = []
    lock = threading.Lock()
    action_lock = threading.Lock()
    started = time.monotonic()
    last_progress = 0.0
    with ExitStack() as stack:
        for spec in specs:
            name = spec["filename"]
            stack.enter_context(ArchiveLock(directory / (name + ".lock")))
            target = directory / name
            if target.exists():
                verify_archive(target, expected_sha256=spec["oid"])
                result[spec["campaign"]] = target
                print(
                    f"{spec['campaign']}: existing archive SHA-256 verified", flush=True
                )
                continue
            partial = directory / (name + ".part")
            checkpoint = directory / (name + ".ranges.json")
            if checkpoint.exists():
                state = json.loads(checkpoint.read_text(encoding="utf-8"))
                if not partial.exists():
                    raise ValueError(
                        f"Checkpoint exists without its partial archive: {name}"
                    )
            else:
                state = {
                    "prefix_bytes": partial.stat().st_size if partial.exists() else 0,
                    "done": [],
                    "oid": spec["oid"],
                    "size": spec["size"],
                }
            prefix = state.get("prefix_bytes")
            if (
                state.get("oid") != spec["oid"]
                or state.get("size") != spec["size"]
                or type(prefix) is not int
                or not 0 <= prefix <= spec["size"]
            ):
                raise ValueError(f"Resume checkpoint identity/size mismatch: {name}")
            possible = set(range(prefix, spec["size"], CHUNK_BYTES))
            done_list = state.get("done")
            if (
                not isinstance(done_list, list)
                or any(type(x) is not int or x not in possible for x in done_list)
                or len(set(done_list)) != len(done_list)
            ):
                raise ValueError(f"Invalid resume range offsets: {name}")
            if partial.exists() and partial.stat().st_size < prefix:
                raise ValueError(
                    f"Partial file is smaller than its saved contiguous prefix: {name}"
                )
            stream = stack.enter_context(
                partial.open("r+b" if partial.exists() else "w+b")
            )
            stream.truncate(spec["size"])
            _json_write(checkpoint, state)
            records[name] = {
                "spec": spec,
                "state": state,
                "checkpoint": checkpoint,
                "partial": partial,
                "target": target,
                "stream": stream,
            }
            for begin in sorted(possible - set(done_list)):
                jobs.append(
                    (name, begin, min(spec["size"] - 1, begin + CHUNK_BYTES - 1))
                )
        pending = [
            record["spec"]
            for name, record in records.items()
            if any(job[0] == name for job in jobs)
        ]
        actions = download_actions(pending)

        def fetch(job: tuple[str, int, int]) -> None:
            nonlocal last_progress
            name, begin, end = job
            record = records[name]
            spec = record["spec"]
            for attempt in range(4):
                with action_lock:
                    action = actions[spec["oid"]]
                try:
                    request = urllib.request.Request(
                        action["href"],
                        headers={
                            **action.get("header", {}),
                            "Range": f"bytes={begin}-{end}",
                        },
                    )
                    with urllib.request.urlopen(request, timeout=120) as response:
                        if (
                            response.status != 206
                            or response.headers.get("Content-Range")
                            != f"bytes {begin}-{end}/{spec['size']}"
                        ):
                            raise ValueError("Unexpected byte-range response")
                        data = response.read(end - begin + 2)
                    if len(data) != end - begin + 1:
                        raise ValueError("Incomplete byte-range response")
                    with lock:
                        # Seek/write under a lock also supports Python on Windows.
                        record["stream"].seek(begin)
                        record["stream"].write(data)
                        record["stream"].flush()
                        record["state"]["done"].append(begin)
                        _json_write(record["checkpoint"], record["state"])
                        now = time.monotonic()
                        if now - last_progress >= 15:
                            totals = {
                                r["spec"]["campaign"]: round(
                                    (
                                        r["state"]["prefix_bytes"]
                                        + sum(
                                            min(CHUNK_BYTES, r["spec"]["size"] - p)
                                            for p in r["state"]["done"]
                                        )
                                    )
                                    / 1_000_000,
                                    1,
                                )
                                for r in records.values()
                            }
                            print(
                                json.dumps(
                                    {
                                        "elapsed_seconds": round(now - started),
                                        "downloaded_MB": totals,
                                    }
                                ),
                                flush=True,
                            )
                            last_progress = now
                    return
                except Exception as error:
                    if attempt == 3:
                        raise
                    if (
                        isinstance(error, urllib.error.HTTPError)
                        and error.code in {401, 403}
                    ):
                        # Public LFS URLs expire. Refresh only the action still
                        # used by this request; another worker may have done it.
                        with action_lock:
                            if actions[spec["oid"]] is action:
                                actions.update(download_actions([spec]))
                    time.sleep(attempt + 1)

        pool = ThreadPoolExecutor(max_workers=workers)
        futures = [pool.submit(fetch, job) for job in jobs]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        for record in records.values():
            record["stream"].close()
            spec = record["spec"]
            verify_archive(record["partial"], expected_sha256=spec["oid"])
            record["partial"].replace(record["target"])
            record["checkpoint"].unlink(missing_ok=True)
            result[spec["campaign"]] = record["target"]
            print(
                f"{spec['campaign']}: complete archive SHA-256 verified "
                f"({spec['size']:,} bytes)",
                flush=True,
            )
    return result


def prepare_dataset(
    repo: Path, spec: dict[str, Any], archive: Path, root: Path
) -> dict[str, Any]:
    """Keep each exported revision separate from directly tracked campaigns."""
    receipt_path = root / "architecture-replay-source.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    pending_path = root / "architecture-replay-preparation.json"
    pending = json.loads(pending_path.read_text()) if pending_path.exists() else {}
    identity = {
        "source_revision": spec["revision"],
        "archive_sha256": spec["oid"],
        "archive_filename": spec["filename"],
    }
    if any(
        record and (
            not isinstance(record, dict)
            or any(record.get(key) != value for key, value in identity.items())
        )
        for record in (previous, pending)
    ):
        raise ValueError(
            "Prepared dataset belongs to a different archive revision; choose a "
            "fresh --output-root to preserve both revisions without merging files"
        )
    if not previous and not pending and root.exists() and any(root.iterdir()):
        raise ValueError(
            "Preparation root contains files without archive provenance; choose a "
            "fresh --output-root rather than merging an unknown dataset"
        )
    with zipfile.ZipFile(archive) as bundle:
        names = [item.filename.replace("\\", "/") for item in bundle.infolist()]
        if any(name.startswith("data/c0c3/") for name in names):
            destination = root
        elif any(
            len(PurePosixPath(name).parts) >= 3
            and PurePosixPath(name).parts[1] == "runs"
            for name in names
        ):
            destination = root / "data" / "c0c3"
        else:
            raise ValueError(
                "Archive layout is not a recognized data/c0c3 or campaign-root export"
            )
        readmes = []
        for name in names:
            if (
                name.endswith("TRANSFER-README.txt")
                and bundle.getinfo(name).file_size < 128 * 1024
            ):
                readmes.append(bundle.read(name).decode("utf-8", errors="replace"))
        member_count = len(names)
    # A failed/interrupted extraction remains tied to the same immutable archive.
    # This marker is not a verified receipt and is never selected by the API.
    _json_write(pending_path, identity)
    report = prepare_archive(
        archive, destination, expected_sha256=spec["oid"], metadata_only=True
    )
    source_checks = []
    transfer_file = repo / "data" / "rl4rl-tiny-har-transfer-20260908.json"
    if transfer_file.exists():
        transfer = json.loads(transfer_file.read_text(encoding="utf-8"))
        original = next(
            (
                row
                for row in transfer.get("archives", [])
                if row.get("archive") == spec["filename"]
            ),
            None,
        )
        if original:
            if original.get("sha256") != spec["oid"]:
                raise ValueError(
                    "Transfer manifest and Git LFS archive identity disagree"
                )
            for relative, expected in original.get(
                "source_metadata_sha256", {}
            ).items():
                path = (root / "data" / "c0c3" / relative).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():
                    raise ValueError(
                        f"Transferred source metadata is missing or unsafe: {relative}"
                    )
                actual = _sha256(path)
                source_checks.append(
                    {
                        "path": relative,
                        "source_sha256": expected,
                        "export_sha256": actual,
                        "matches_source": actual == expected,
                    }
                )
    changed = [row for row in source_checks if not row["matches_source"]]
    filtered_export = any(
        "For the capped export only" in text and "state are filtered" in text
        for text in readmes
    )
    if changed and not filtered_export:
        raise ValueError(
            "Source metadata checksums differ without a documented capped-export "
            "transformation; no prepared receipt was written"
        )
    receipt = {
        "source_revision": spec["revision"],
        "archive_filename": spec["filename"],
        "archive_sha256": spec["oid"],
        "checksum_verified": True,
        "prepared_at": datetime.now(UTC).isoformat(),
        "data_root": str(root.resolve()),
        "archive_scope": (
            "Preserved archive revision; dashboard view applies its explicit "
            "run/proposal scope separately"
        ),
        "metadata_only": True,
        "archive_members": member_count,
        "files_extracted": max(
            report["files_extracted"], previous.get("files_extracted", 0)
        ),
        "source_metadata_checksums": source_checks,
        "source_metadata_note": (
            "Source states/ledgers are intentionally filtered by the capped "
            "export; the archive SHA verifies exported bytes."
        )
        if changed
        else "Available source metadata checksums match the exported files."
        if source_checks
        else (
            "No per-file source metadata digests are available; "
            "the complete archive SHA-256 is verified."
        ),
    }
    _json_write(receipt_path, receipt)
    pending_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "campaign": spec["campaign"],
                **report,
                "receipt": str(receipt_path),
                "source_metadata_matches": len(source_checks) - len(changed),
                "documented_filtered_metadata": len(changed),
            }
        ),
        flush=True,
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=REPO_ROOT
    )
    parser.add_argument("--campaign", choices=["all", *ARCHIVES], default="all")
    parser.add_argument(
        "--revision",
        default=SOURCE_REVISION,
        help="Immutable source SHA containing the tracked LFS pointers",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Default: <repo>/outputs/architecture-replay-data",
    )
    parser.add_argument(
        "--workers", type=int, default=24, help="Maximum simultaneous byte ranges, 1–32"
    )
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    output = (
        args.output_root or repo / "outputs" / "architecture-replay-data"
    ).resolve()
    campaigns = list(ARCHIVES) if args.campaign == "all" else [args.campaign]
    try:
        specs = [archive_spec(repo, campaign, args.revision) for campaign in campaigns]
        archives = download_archives(specs, output / "downloads", workers=args.workers)
        if not args.download_only:
            for spec in specs:
                prepare_dataset(
                    repo, spec, archives[spec["campaign"]], output / spec["campaign"]
                )
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(2, f"prepare-architecture-data: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

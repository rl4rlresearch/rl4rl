#!/usr/bin/env python3
"""Summarize local Codex token accounting without touching experiment runs.

Codex stores cumulative ``total_token_usage`` and per-response
``last_token_usage`` records in its local session archive. This utility maps a
session to a run by the session's recorded working directory, then writes
derived reports outside the run directories. It never opens a run workspace for
writing and never contacts the API.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def _json_lines(path: Path) -> Iterator[dict[str, Any]]:
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def _session_metadata(path: Path) -> dict[str, Any] | None:
    for event in _json_lines(path):
        if event.get("type") == "session_meta":
            payload = event.get("payload") or {}
            if isinstance(payload, dict):
                return payload
        # session metadata is the first meaningful record; do not parse a
        # copied historical transcript embedded later in another session.
        break
    return None


def _runs_by_workspace(runs_root: Path) -> dict[Path, str]:
    result: dict[Path, str] = {}
    for candidate in runs_root.iterdir() if runs_root.is_dir() else ():
        if (candidate / "STATE.json").is_file() and (candidate / "workspace").is_dir():
            result[candidate.resolve() / "workspace"] = candidate.name
    return result


def _usage(value: Any) -> dict[str, int]:
    source = value if isinstance(value, dict) else {}
    return {field: int(source.get(field, 0) or 0) for field in TOKEN_FIELDS}


def _atomic_tsv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def collect(
    runs_root: Path, sessions_root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    workspaces = _runs_by_workspace(runs_root)
    # A resumed/forked Codex session can exist in more than one archive file
    # with the same logical session id.  Treat the longest cumulative archive
    # as canonical so that its shared history is not counted twice.
    candidates: dict[
        tuple[str, str], tuple[dict[str, object], list[dict[str, object]]]
    ] = {}
    for path in (
        sorted(sessions_root.rglob("*.jsonl")) if sessions_root.is_dir() else ()
    ):
        metadata = _session_metadata(path)
        if not metadata or not metadata.get("cwd"):
            continue
        try:
            run_id = workspaces.get(Path(str(metadata["cwd"])).resolve())
        except OSError:
            continue
        if not run_id:
            continue
        session_id = str(metadata.get("session_id") or metadata.get("id") or path.stem)
        latest_total: dict[str, int] | None = None
        event_count = 0
        session_increments: list[dict[str, object]] = []
        for event in _json_lines(path):
            payload = event.get("payload") or {}
            if payload.get("type") != "token_count":
                continue
            info = payload.get("info") or {}
            last = info.get("last_token_usage")
            total = info.get("total_token_usage")
            if not isinstance(last, dict) and not isinstance(total, dict):
                continue
            event_count += 1
            session_increments.append(
                {
                    "run_id": run_id,
                    "session_id": session_id,
                    "timestamp": event.get("timestamp", ""),
                    "token_event": event_count,
                    **_usage(last),
                }
            )
            if isinstance(total, dict):
                latest_total = _usage(total)
        if latest_total is not None:
            session = {
                "run_id": run_id,
                "session_id": session_id,
                "session_file": str(path),
                "token_events": event_count,
                **latest_total,
            }
            key = (run_id, session_id)
            previous = candidates.get(key)
            if previous is None or (
                int(session["total_tokens"]),
                event_count,
                str(path),
            ) > (
                int(previous[0]["total_tokens"]),
                int(previous[0]["token_events"]),
                str(previous[0]["session_file"]),
            ):
                candidates[key] = (session, session_increments)

    sessions: list[dict[str, object]] = []
    increments: list[dict[str, object]] = []
    for session, session_increments in candidates.values():
        sessions.append(session)
        increments.extend(session_increments)
    sessions.sort(key=lambda row: (str(row["run_id"]), str(row["session_id"])))
    increments.sort(
        key=lambda row: (
            str(row["run_id"]),
            str(row["timestamp"]),
            str(row["session_id"]),
            int(row["token_event"]),
        )
    )
    return sessions, increments


def write_reports(
    output_dir: Path,
    sessions: list[dict[str, object]],
    increments: list[dict[str, object]],
) -> None:
    session_fields = [
        "run_id",
        "session_id",
        "session_file",
        "token_events",
        *TOKEN_FIELDS,
    ]
    increment_fields = [
        "run_id",
        "session_id",
        "timestamp",
        "token_event",
        *TOKEN_FIELDS,
    ]
    _atomic_tsv(output_dir / "session_totals.tsv", session_fields, sessions)
    _atomic_tsv(output_dir / "response_increments.tsv", increment_fields, increments)
    by_run: dict[str, dict[str, int]] = defaultdict(
        lambda: {field: 0 for field in TOKEN_FIELDS}
    )
    session_counts: dict[str, int] = defaultdict(int)
    for row in sessions:
        run_id = str(row["run_id"])
        session_counts[run_id] += 1
        for field in TOKEN_FIELDS:
            by_run[run_id][field] += int(row[field])
    summary = [
        {"run_id": run_id, "sessions": session_counts[run_id], **usage}
        for run_id, usage in sorted(by_run.items())
    ]
    _atomic_tsv(
        output_dir / "run_totals.tsv", ["run_id", "sessions", *TOKEN_FIELDS], summary
    )
    (output_dir / "README.md").write_text(
        "# Codex token accounting\n\n"
        "Generated from local Codex session archives. `run_totals.tsv` sums the "
        "latest cumulative total from every direct session whose recorded working "
        "directory is a run workspace. `response_increments.tsv` contains each "
        "`last_token_usage` record (one model-response increment). Cached input is "
        "reported separately and is included in `input_tokens`; do not add it again.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--sessions-root", type=Path, default=Path.home() / ".codex" / "sessions"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--watch-seconds", type=float, default=0.0)
    args = parser.parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    output_dir = (args.output_dir or runs_root / "token_usage").expanduser().resolve()
    sessions_root = args.sessions_root.expanduser().resolve()
    while True:
        sessions, increments = collect(runs_root, sessions_root)
        write_reports(output_dir, sessions, increments)
        print(
            f"wrote {len(sessions)} session totals and "
            f"{len(increments)} response increments to {output_dir}"
        )
        if args.watch_seconds <= 0:
            return 0
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

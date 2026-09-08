"""Deterministically audit proposal Codex event logs; never execute their contents.

Usage: python experiments/audit_codex_tool_calls.py

Searches the repository (including ignored campaign/recovery files), counts each
tool item once across its lifecycle, and deduplicates byte-identical log copies.
Reports recorded CLI tool categories, not tool names inferred from prose. The
CLI's command_execution item does not retain the original shell API tool name.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ITEMS = {
    "command_execution", "file_change", "mcp_tool_call", "web_search",
    "image_view", "image_generation", "collab_tool_call",
}
NON_TOOL_ITEMS = {"agent_message", "reasoning", "error", "todo_list"}
KNOWN_EVENTS = {
    "thread.started", "turn.started", "turn.completed", "turn.failed",
    "error", "item.started", "item.updated", "item.completed",
}


def native_path(path: str | Path) -> str:
    """Avoid Windows MAX_PATH and costly per-component resolve() calls."""
    result = os.path.abspath(path)
    if os.name == "nt" and not result.startswith("\\\\?\\"):
        return "\\\\?\\UNC\\" + result[2:] if result.startswith("\\\\") else "\\\\?\\" + result
    return result


def proposal_log(path: Path) -> bool:
    return path.parent.name == "codex" and (
        path.suffix == ".jsonl" or bool(re.fullmatch(r"\.proposal-\d+\.[^.]+", path.name))
    )


def discover(roots: list[Path]) -> tuple[list[Path], str]:
    """rg includes gitignored data and hidden interrupted stdout files."""
    found: set[Path] = set()
    rg = shutil.which("rg")
    for root in roots:
        if not os.path.exists(native_path(root)):
            raise FileNotFoundError(root)
        if os.path.isfile(native_path(root)):
            if proposal_log(root):
                found.add(root)
            continue
        if rg:
            result = subprocess.run(
                [rg, "--files", "--hidden", "--no-ignore", "-g", "**/codex/*",
                 "-g", "!**/.git/**", str(root)],
                capture_output=True, text=True, encoding="utf-8", check=False,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(result.stderr.strip())
            found.update(Path(name) for name in result.stdout.splitlines() if proposal_log(Path(name)))
        else:
            def walk_error(error: OSError) -> None:
                raise error
            for base, dirs, files in os.walk(native_path(root), onerror=walk_error):
                dirs[:] = [name for name in dirs if name != ".git"]
                if Path(base).name != "codex":
                    continue
                for name in files:
                    path = Path(base) / name
                    if proposal_log(path):
                        # Return normal paths consistently with rg.
                        absolute = str(path)
                        if absolute.startswith("\\\\?\\UNC\\"):
                            absolute = "\\\\" + absolute[8:]
                        elif absolute.startswith("\\\\?\\"):
                            absolute = absolute[4:]
                        found.add(Path(absolute))
    return sorted(found, key=lambda p: str(p).casefold()), "rg" if rg else "os.walk"


def fixture_path(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part in {"tests", "fixtures"} or part.startswith("test_") for part in parts[:-1])


def metadata(path: Path, base: Path) -> dict:
    try:
        relative = path.relative_to(base).as_posix()
    except ValueError:
        relative = path.as_posix()
    parts = path.parts
    run = ""
    campaign = "unattributed recovery"
    if "runs" in parts:
        index = parts.index("runs")
        campaign, run = parts[index - 1], parts[index + 1]
    elif "opportunities" in parts:
        run = parts[parts.index("opportunities") - 1]
        campaign = "standalone run copy"
    proposal = None
    for index, part in enumerate(parts[:-1]):
        if part == "opportunities" and parts[index + 1].isdigit():
            proposal = int(parts[index + 1])
        elif match := re.fullmatch(r"opportunity-(\d+)", part):
            proposal = int(match[1])
    if proposal is None and (match := re.search(r"proposal-(\d+)", path.name)):
        proposal = int(match[1])
    condition = re.search(r"-b(\d+)-c(\d+)$", run)
    archive = "repair-quarantine" in parts or "outputs" in parts or campaign == "standalone run copy"
    return {
        "path": relative, "campaign": campaign, "run": run,
        "block": int(condition[1]) if condition else None,
        "condition": "C" + condition[2] if condition else None,
        "proposal": proposal, "archive": archive,
        "temporary": path.name.startswith(".proposal-"),
    }


def tool_name(item: dict) -> str | None:
    kind = item.get("type")
    if kind not in TOOL_ITEMS:
        return None
    if kind == "mcp_tool_call":
        return f"mcp_tool_call:{item.get('server', '?')}/{item.get('tool', '?')}"
    return kind


def shell_program(command: str | None) -> tuple[str | None, str | None]:
    """Lexically identify a shell and its first requested program, without running it."""
    if not command:
        return None, None
    try:
        words = shlex.split(command)
        if not words:
            return None, None
        executable = words[0]
        if Path(executable).name in {"sh", "bash", "zsh", "dash"}:
            for index, word in enumerate(words[1:-1], 1):
                if word.startswith("-") and "c" in word:
                    inner = shlex.split(words[index + 1])
                    return executable, inner[0] if inner else None
        return executable, executable
    except ValueError:
        return None, None


def parse_log(path: Path, base: Path) -> dict:
    result = {**metadata(path, base), "calls": [], "issues": [], "event_types": Counter(),
              "item_types": Counter(), "thread_ids": [], "bytes": 0, "sha256": None,
              "turns_completed": 0, "turns_failed": 0, "duplicate_of": None}
    try:
        before = os.stat(native_path(path))
        with open(native_path(path), "rb") as handle:
            raw = handle.read()
        after = os.stat(native_path(path))
    except OSError as error:
        result["issues"].append({"kind": "read_error", "detail": str(error)})
        return result
    result["bytes"] = len(raw)
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        result["issues"].append({"kind": "changed_during_scan"})
    calls = {}
    turn = 0
    thread = None
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (ValueError, UnicodeError):
            result["issues"].append({"kind": "malformed_json", "line": line_number})
            continue
        if not isinstance(event, dict):
            result["issues"].append({"kind": "non_object_record", "line": line_number})
            continue
        kind = event.get("type", "<missing>")
        result["event_types"][kind] += 1
        if kind not in KNOWN_EVENTS:
            result["issues"].append({"kind": "unknown_event_type", "type": kind, "line": line_number})
        if kind == "thread.started":
            thread = event.get("thread_id")
            result["thread_ids"].append(thread)
        elif kind == "turn.started":
            turn += 1
        elif kind == "turn.completed":
            result["turns_completed"] += 1
        elif kind == "turn.failed":
            result["turns_failed"] += 1
        if kind not in {"item.started", "item.updated", "item.completed"}:
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            result["issues"].append({"kind": "missing_item", "line": line_number})
            continue
        item_kind = item.get("type", "<missing>")
        result["item_types"][item_kind] += 1
        name = tool_name(item)
        if name is None:
            if item_kind not in NON_TOOL_ITEMS:
                result["issues"].append({"kind": "unknown_item_type", "type": item_kind, "line": line_number})
            continue
        item_id = item.get("id")
        if not item_id:
            # Do not pretend start/completion can be paired when the ID is missing.
            result["issues"].append({"kind": "tool_missing_id", "line": line_number})
            if kind != "item.completed":
                continue
            item_id = f"unidentified-completion-line-{line_number}"
        identity = (thread, turn, item_id)
        if identity not in calls:
            calls[identity] = {
                "tool": name, "item_type": item_kind, "item_id": item_id,
                "thread_id": thread, "turn": turn, "first_line": line_number,
                "last_line": line_number, "event_records": 0, "terminal_record": False,
                "status": "unfinished", "exit_code": None, "command": None,
            }
        call = calls[identity]
        if call["tool"] != name:
            result["issues"].append({"kind": "conflicting_item_identity", "line": line_number})
        call["last_line"] = line_number
        call["event_records"] += 1
        if isinstance(item.get("command"), str):
            call["command"] = item["command"]
        if item.get("exit_code") is not None:
            call["exit_code"] = item["exit_code"]
        if kind == "item.completed":
            call["terminal_record"] = True
            failed = item.get("status") in {"failed", "error"} or call["exit_code"] not in (None, 0)
            call["status"] = "failed" if failed else "completed"
    for call in calls.values():
        call["shell"], call["requested_program"] = shell_program(call["command"])
    result["calls"] = list(calls.values())
    result["incomplete_log"] = not (result["turns_completed"] or result["turns_failed"])
    return result


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit(roots: list[Path], output: Path, workers: int = 8, include_test_fixtures: bool = False) -> dict:
    start = time.perf_counter()
    paths, discovery = discover(roots)
    excluded = [path for path in paths if not include_test_fixtures and fixture_path(path, REPO_ROOT)]
    excluded_set = set(excluded)
    included = [path for path in paths if path not in excluded_set]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        logs = list(pool.map(lambda path: parse_log(path, REPO_ROOT), included))
    # Prefer live campaign paths over backup copies when recording provenance.
    logs.sort(key=lambda row: (row["archive"], row["temporary"], row["path"]))
    hashes: dict[str, str] = {}
    for row in logs:
        digest = row["sha256"]
        if digest is not None and digest in hashes:
            row["duplicate_of"] = hashes[digest]
        elif digest is not None:
            hashes[digest] = row["path"]
    unique = [row for row in logs if not row["duplicate_of"]]
    ledger = []
    event_types, item_types = Counter(), Counter()
    for row in unique:
        event_types.update(row["event_types"])
        item_types.update(row["item_types"])
        for call in row["calls"]:
            ledger.append({**{key: row[key] for key in ("path", "campaign", "run", "block", "condition", "proposal", "archive", "temporary")}, **call})
    ledger.sort(key=lambda call: (call["path"], call["first_line"]))
    tools = []
    for name in sorted({call["tool"] for call in ledger}):
        calls = [call for call in ledger if call["tool"] == name]
        tools.append({"tool": name, "calls": len(calls),
                      "completed": sum(call["status"] == "completed" for call in calls),
                      "failed": sum(call["status"] == "failed" for call in calls),
                      "unfinished": sum(call["status"] == "unfinished" for call in calls),
                      "proposal_logs": len({call["path"] for call in calls})})
    campaigns = []
    for campaign in sorted({row["campaign"] for row in unique}):
        members = [row for row in unique if row["campaign"] == campaign]
        campaigns.append({"campaign": campaign, "unique_logs": len(members),
                          "tool_calls": sum(len(row["calls"]) for row in members),
                          "logs_with_tools": sum(bool(row["calls"]) for row in members)})
    issues = [{"path": row["path"], **issue} for row in logs for issue in row["issues"]]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "roots": [str(root) for root in roots], "discovery": discovery,
        "files_discovered": len(paths), "excluded_test_fixture_files": len(excluded),
        "files_scanned": len(logs), "bytes_scanned": sum(row["bytes"] for row in logs),
        "duplicate_files": sum(bool(row["duplicate_of"]) for row in logs),
        "unique_logs": len(unique), "temporary_logs": sum(row["temporary"] for row in unique),
        "logs_without_terminal_turn": sum(row.get("incomplete_log", False) for row in unique),
        "logs_with_failed_turn": sum(bool(row["turns_failed"]) for row in unique),
        "tool_calls": len(ledger), "distinct_tools": len(tools),
        "proposal_logs_with_tool_calls": sum(bool(row["calls"]) for row in unique),
        "tools": tools, "campaigns": campaigns,
        "requested_shell_programs": dict(sorted(Counter(call["requested_program"] for call in ledger if call["requested_program"]).items())),
        "event_types": dict(sorted(event_types.items())), "item_types": dict(sorted(item_types.items())),
        "issues": issues,
        "counting_rules": [
            "One CLI tool item per (log, thread ID, turn ordinal, item ID); lifecycle records are not separate calls.",
            "Failed and unfinished calls count as attempts; outcomes are reported separately.",
            "Byte-identical log copies are counted once. Distinct logs are separate invocations, even when item IDs repeat.",
            "Includes all conditions, dashboard-hidden runs, proposals beyond dashboard caps, recovery/quarantine and interrupted stdout files.",
            "Test fixture directories are excluded by path unless --include-test-fixtures is specified.",
            "Only structured CLI events are parsed. Prompts, agent prose, reasoning, tool outputs, error messages and todo-list snapshots are not calls.",
            "Unexpected event/item types, missing tool IDs, malformed records and read errors are reported; they are never guessed from prose.",
            "Names are recorded CLI tool categories (plus MCP server/tool); the original shell API function name is not recoverable from command_execution.",
            "A shell's requested program is parsed lexically for reference; it is not counted as an additional LLM tool call.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "calls.csv", ["tool", "campaign", "run", "block", "condition", "proposal", "status", "exit_code", "path", "first_line", "last_line", "item_id", "thread_id", "turn", "event_records", "terminal_record", "archive", "temporary", "shell", "requested_program", "command"], ledger)
    write_csv(output / "tools.csv", ["tool", "calls", "completed", "failed", "unfinished", "proposal_logs"], tools)
    write_csv(output / "campaigns.csv", ["campaign", "unique_logs", "tool_calls", "logs_with_tools"], campaigns)
    file_rows = [{**row, "tool_calls": len(row["calls"]), "issues": len(row["issues"]), "excluded": False} for row in logs]
    file_rows += [{**metadata(path, REPO_ROOT), "excluded": "test_fixture"} for path in excluded]
    write_csv(output / "logs.csv", ["path", "campaign", "run", "condition", "proposal", "archive", "temporary", "bytes", "sha256", "tool_calls", "duplicate_of", "incomplete_log", "turns_completed", "turns_failed", "issues", "excluded"], sorted(file_rows, key=lambda row: row["path"]))
    summary["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Proposal Codex tool-call audit", "", f"**{len(ledger)} tool calls; {len(tools)} distinct recorded tool categories.**", "",
             f"Scanned {len(logs):,} logs in {summary['elapsed_seconds']:.2f} seconds. "
             f"Removed {summary['duplicate_files']:,} byte-identical copies; excluded {len(excluded):,} test fixtures. "
             f"{summary['proposal_logs_with_tool_calls']:,} unique logs contain calls.", "",
             "| Tool | Calls | Completed | Failed | Unfinished |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {row['tool']} | {row['calls']} | {row['completed']} | {row['failed']} | {row['unfinished']} |" for row in tools]
    lines += ["", "## Campaigns", "", "| Campaign | Unique logs | Calls |", "|---|---:|---:|"]
    lines += [f"| {row['campaign']} | {row['unique_logs']:,} | {row['tool_calls']} |" for row in campaigns]
    lines += ["", "## Coverage", "", f"{len(issues)} audit issues; {summary['logs_without_terminal_turn']} logs without a terminal turn; {summary['logs_with_failed_turn']} logs with failed turns.",
              "These are counts of recorded calls, not a claim that interrupted logs record events after logging stopped.", "", "## Counting rules", ""]
    lines += ["- " + rule for rule in summary["counting_rules"]]
    lines += ["", "Every counted call and its source line numbers are in [calls.csv](calls.csv). The complete discovery, duplicate and fixture manifest is [logs.csv](logs.csv).",
              "", "Run again with `python experiments/audit_codex_tool_calls.py` (no model or network required).", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, help="Search root; repeat for multiple roots. Default: repository.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/codex-tool-call-audit")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include-test-fixtures", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    roots = [Path(os.path.abspath(root)) for root in (args.root or [REPO_ROOT])]
    summary = audit(roots, args.output, args.workers, args.include_test_fixtures)
    print(json.dumps({key: summary[key] for key in ("files_scanned", "duplicate_files", "unique_logs", "tool_calls", "distinct_tools", "tools", "requested_shell_programs", "elapsed_seconds")}, indent=2))
    print(f"Audit issues: {len(summary['issues'])}. Report: {args.output / 'report.md'}")
    return 2 if summary["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

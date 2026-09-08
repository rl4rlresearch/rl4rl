from __future__ import annotations

import json
from pathlib import Path

from experiments import audit_codex_tool_calls as audit_module


def log(path: Path, records: list[dict], suffix: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\n".join(json.dumps(record).encode() for record in records) + b"\n" + suffix)
    return path


def item(kind: str, item_id: str, item_type: str = "command_execution", **fields) -> dict:
    return {"type": kind, "item": {"id": item_id, "type": item_type, **fields}}


def test_count_lifecycles_once_and_keep_single_use_tools(tmp_path: Path) -> None:
    path = log(tmp_path / "codex/proposal-1.jsonl", [
        {"type": "thread.started", "thread_id": "thread-a"},
        {"type": "turn.started"},
        item("item.started", "item_0", command='/bin/zsh -lc "python -c \'print(1)\'"'),
        item("item.updated", "item_0", status="in_progress"),
        item("item.completed", "item_0", status="completed", exit_code=0),
        item("item.completed", "item_0", status="completed", exit_code=0),
        item("item.completed", "item_1", "file_change", status="completed", changes=[{}, {}]),
        item("item.completed", "item_2", "mcp_tool_call", server="local", tool="rare_tool", status="failed"),
        item("item.completed", "item_3", "web_search", status="completed"),
        item("item.completed", "item_4", "agent_message", text='{"type":"function_call","name":"fake_tool"}'),
        item("item.completed", "item_5", "reasoning", text="mentioned a tool"),
        item("item.completed", "item_6", "error", message="not a call"),
        {"type": "turn.completed"},
        {"type": "turn.started"},
        item("item.started", "item_0", command="python other.py"),
    ])
    parsed = audit_module.parse_log(path, tmp_path)
    assert not parsed["issues"]
    assert len(parsed["calls"]) == 5
    first, change, mcp, web, pending = parsed["calls"]
    assert first["event_records"] == 4
    assert first["requested_program"] == "python"
    assert first["status"] == "completed"
    assert change["tool"] == "file_change"  # One tool action, not one per edited file.
    assert mcp["tool"] == "mcp_tool_call:local/rare_tool"
    assert mcp["status"] == "failed"
    assert web["tool"] == "web_search"
    assert pending["status"] == "unfinished"
    assert pending["turn"] == 2  # item_0 can be reused on a later turn.


def test_audit_deduplicates_backups_not_distinct_invocations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(audit_module, "REPO_ROOT", tmp_path)
    records = [
        {"type": "thread.started", "thread_id": "first"},
        {"type": "turn.started"},
        item("item.started", "item_0", command="python sample.py"),
        item("item.completed", "item_0", status="failed", exit_code=1),
        {"type": "turn.completed"},
    ]
    original = log(tmp_path / "data/campaign/runs/run-b01-c0/opportunities/0001/codex/proposal-1.jsonl", records)
    duplicate = log(tmp_path / "outputs/recovery/codex/proposal-1.jsonl", records)
    second = [{"type": "thread.started", "thread_id": "second"}, *records[1:]]
    log(tmp_path / "data/campaign/runs/run-b01-c0/opportunities/0002/codex/proposal-2.jsonl", second)
    log(tmp_path / "outputs/pytest/test_case0/codex/proposal-1.jsonl", second)
    log(tmp_path / "data/campaign/runs/run-b01-c0/opportunities/0003/codex/.proposal-3.tmp", [
        {"type": "thread.started", "thread_id": "third"},
        {"type": "turn.started"},
        item("item.started", "item_0", command="python third.py"),
    ])
    before = original.read_bytes()
    report = audit_module.audit([tmp_path], tmp_path / "report", workers=2)
    assert report["files_discovered"] == 5
    assert report["excluded_test_fixture_files"] == 1
    assert report["files_scanned"] == 4
    assert report["duplicate_files"] == 1
    assert report["tool_calls"] == 3
    assert report["tools"][0]["failed"] == 2
    assert report["tools"][0]["unfinished"] == 1
    assert report["temporary_logs"] == 1
    assert original.read_bytes() == before == duplicate.read_bytes()
    assert (tmp_path / "report/calls.csv").is_file()
    assert (tmp_path / "report/logs.csv").is_file()


def test_corrupt_unknown_and_idless_records_are_not_silently_lost(tmp_path: Path) -> None:
    path = log(tmp_path / "codex/proposal-1.jsonl", [
        {"type": "thread.started", "thread_id": "a"},
        {"type": "turn.started"},
        item("item.completed", "item_0", status="completed", exit_code=0),
        item("item.completed", "item_1", "future_tool_type"),
        {"type": "item.started", "item": {"type": "command_execution"}},
        {"type": "item.completed", "item": {"type": "command_execution", "status": "completed"}},
    ], suffix=b'{"type":"item.comp')
    parsed = audit_module.parse_log(path, tmp_path)
    assert len(parsed["calls"]) == 2
    assert {issue["kind"] for issue in parsed["issues"]} == {
        "unknown_item_type", "tool_missing_id", "malformed_json"
    }
    assert parsed["incomplete_log"]


def test_posix_shell_program_parsing_never_executes_commands() -> None:
    assert audit_module.shell_program('/bin/zsh -lc "python3 - <<\'PY\'\nprint(2)\nPY"') == ("/bin/zsh", "python3")
    assert audit_module.shell_program('/bin/zsh -c "sed -n \'1,260p\' train.py"') == ("/bin/zsh", "sed")
    assert audit_module.shell_program('"broken') == (None, None)

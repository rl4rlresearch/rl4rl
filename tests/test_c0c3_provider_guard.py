from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from experiments.c0c3_provider_guard import (
    INITIAL_BACKOFF_ENV,
    MAX_BACKOFF_ENV,
    install_provider_retry_guard,
)


def _result(log_root: Path, call_id: str, returncode: int) -> SimpleNamespace:
    events = log_root / f"{call_id}.jsonl"
    stderr = log_root / f"{call_id}.stderr.log"
    last = log_root / f"{call_id}.last-message.md"
    events.write_text(
        json.dumps({"type": "turn.failed", "error": {"message": "offline"}})
        + "\n",
        encoding="utf-8",
    )
    stderr.write_text("provider unavailable\n", encoding="utf-8")
    last.write_text("", encoding="utf-8")
    return SimpleNamespace(
        returncode=returncode,
        events_path=events,
        stderr_path=stderr,
    )


def test_provider_failure_restores_workspace_and_retries(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(INITIAL_BACKOFF_ENV, "0")
    monkeypatch.setenv(MAX_BACKOFF_ENV, "0")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    editable = workspace / "model.py"
    editable.write_text("original\n", encoding="utf-8")
    log_root = tmp_path / "codex"
    log_root.mkdir()

    class FakeCodexCli:
        calls = 0

        def run(self, *args, **kwargs):
            assert editable.read_text(encoding="utf-8") == "original\n"
            self.calls += 1
            if self.calls == 1:
                editable.write_text("partial provider-failed edit\n", encoding="utf-8")
                return _result(log_root, "proposal-7", 1)
            editable.write_text("successful edit\n", encoding="utf-8")
            return _result(log_root, "proposal-7", 0)

    module = SimpleNamespace(CodexCli=FakeCodexCli)
    install_provider_retry_guard(module)
    result = FakeCodexCli().run(
        workspace=workspace,
        log_root=log_root,
        call_id="proposal-7",
    )

    assert result.returncode == 0
    assert editable.read_text(encoding="utf-8") == "successful edit\n"
    retry = log_root / "provider-retries/proposal-7-retry-0001"
    assert (retry / "proposal-7.jsonl").is_file()
    metadata = json.loads((retry / "retry.json").read_text(encoding="utf-8"))
    assert metadata["returncode"] == 1
    assert metadata["retry_number"] == 1


def test_explicit_codex_timeout_is_not_retried(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    log_root = tmp_path / "codex"
    log_root.mkdir()

    class FakeCodexCli:
        calls = 0

        def run(self, *args, **kwargs):
            self.calls += 1
            return _result(log_root, "proposal-3", 124)

    module = SimpleNamespace(CodexCli=FakeCodexCli)
    install_provider_retry_guard(module)
    client = FakeCodexCli()
    result = client.run(
        workspace=workspace,
        log_root=log_root,
        call_id="proposal-3",
    )

    assert result.returncode == 124
    assert client.calls == 1
    assert not (log_root / "provider-retries").exists()

import json

import pytest

from experiments.c0c3_factorial import state


def test_atomic_save_retries_same_payload_after_windows_reader(monkeypatch, tmp_path):
    target = tmp_path / "state.json"
    target.write_text('{"old": true}')
    replace = state.os.replace
    attempts = []

    def blocked_once(source, destination):
        attempts.append(source)
        if len(attempts) == 1:
            assert json.loads(target.read_text()) == {"old": True}
            error = PermissionError("reader holds destination")
            error.winerror = 32
            raise error
        replace(source, destination)

    monkeypatch.setattr(state.os, "replace", blocked_once)
    monkeypatch.setattr(state.time, "sleep", lambda _: None)
    state.atomic_json(target, {"complete": True})
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert json.loads(target.read_text()) == {"complete": True}
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_save_does_not_hide_permanent_permission_error(monkeypatch, tmp_path):
    target = tmp_path / "state.json"
    target.write_text('{"old": true}')

    def denied(*_):
        raise PermissionError("not a Windows sharing error")

    monkeypatch.setattr(state.os, "replace", denied)
    with pytest.raises(PermissionError):
        state.atomic_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"old": True}


def test_load_retries_windows_sharing_error(monkeypatch, tmp_path):
    original = state.Path.read_text
    calls = []

    def transient(path, **kwargs):
        calls.append(path)
        if len(calls) == 1:
            error = PermissionError("sharing")
            error.winerror = 5
            raise error
        return original(path, **kwargs)

    (tmp_path / "state.json").write_text("{}")
    monkeypatch.setattr(state.Path, "read_text", transient)
    monkeypatch.setattr(state.time, "sleep", lambda _: None)
    monkeypatch.setattr(state.RunState, "from_dict", lambda value: value)
    monkeypatch.setattr(state.SearchController, "__init__", lambda *args: None)
    state.SearchController.load(tmp_path, None)
    assert len(calls) == 2

"""Windows viewing must preserve artifacts and never activate Unix controls."""

import json
import os
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from experiments import live_trajectory_dashboard as dashboard


def test_windows_activity_does_not_probe_restored_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "WINDOWS_VIEW_ONLY", True)
    lock = tmp_path / ".semantic-orchestrator.lock"
    lock.write_text('{"pid": 123}', encoding="utf-8")
    before = lock.read_bytes()
    activity = dashboard.runtime_activity_snapshot([tmp_path])
    assert activity["agent_holders"] == []
    assert activity["campaign_controllers"] == set()
    assert lock.read_bytes() == before
    assert list(tmp_path.iterdir()) == [lock]


@pytest.mark.skipif(os.name != "nt", reason="Windows extended paths")
def test_deep_restored_artifacts_are_readable(tmp_path):
    root = dashboard.local_path(tmp_path)
    artifact = root / ("campaign-" + "a" * 100) / ("run-" + "b" * 100)
    artifact.mkdir(parents=True)
    state = artifact / "state.json"
    state.write_text('{"run_id": "restored"}', encoding="utf-8")
    assert len(str(state)) > 260
    assert dashboard.read_json(state, {}) == {"run_id": "restored"}
    assert dashboard.local_path(root) == root


def test_windows_server_serves_data_and_rejects_controls(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "WINDOWS_VIEW_ONLY", True)
    monkeypatch.setattr(dashboard, "codex_rate_limit_payload", lambda *a, **k: {})
    handler = dashboard.make_handler(
        {}, dashboard.DEFAULT_PRICE_PER_MILLION,
        codex_rate_limit_history=tmp_path / "quota.jsonl",
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url + "/api/data", timeout=10) as response:
            assert json.load(response)["read_only"] is True
        with urlopen(url + "/controller", timeout=10) as response:
            assert b"Windows viewing mode" in response.read()
        request = Request(
            url + "/api/controller/campaign-lifecycle", data=b'{}',
            headers={"X-RL4RL-Controller": "1"},
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=10)
        assert error.value.code == 403
        assert not list(tmp_path.iterdir())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_browser_json_handles_nonfinite_archived_metrics():
    source = {"metrics": [float("inf"), float("-inf"), float("nan"), 1.25]}
    result = dashboard.browser_json(source)
    decoded = json.loads(result, parse_constant=lambda value: pytest.fail(value))
    assert decoded == {"metrics": [None, None, None, 1.25]}
    assert source["metrics"][0] == float("inf")


@pytest.mark.parametrize("active", [None, {"index": 65}])
def test_saved_activity_never_claims_live_execution(tmp_path, active):
    state = {"run_id": "backup-run", "active": active}
    stage = dashboard.operational_stage(
        tmp_path, state, status="pausing", desired_state=None,
        campaign_desired=None, evaluator_backend="local",
        runtime_activity={"view_only": True}, campaign=tmp_path.parent,
        campaign_subject_limit=None, campaign_active_opportunities=None,
        semantic=False,
    )
    assert stage["kind"] == "snapshot"
    assert stage["label"] == "Inactive · saved snapshot"
    assert "Saved status: pausing" in stage["detail"]
    assert list(tmp_path.iterdir()) == []

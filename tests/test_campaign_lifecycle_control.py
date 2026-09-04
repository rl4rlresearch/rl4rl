from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from experiments.campaign_lifecycle_control import (
    campaign_lifecycle_payload,
    discover_campaigns,
    semantic_screen_session,
    set_campaign_lifecycle,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _overnight_fixture(tmp_path: Path, *, supervisor_pid: int) -> tuple[Path, Path]:
    campaign = tmp_path / "data/c0c3/future-campaign"
    _write_json(campaign / "campaign.json", {"study_id": "future-study"})
    _write_json(
        campaign / "inputs/task.json",
        {"task_id": "future-task", "display_name": "Future task"},
    )
    _write_json(
        campaign / "inputs/framework.json",
        {"framework_id": "native_openevolve"},
    )
    _write_json(
        campaign / "inputs/protocol.json",
        {"protocol_version": "3.0", "study_id": "future-protocol"},
    )
    _write_json(
        campaign / "runs/b01-c0/state.json",
        {"status": "running", "active": None},
    )
    control = tmp_path / "data/c0c3/overnight-control-future"
    jobs = {
        "future:b01-c0": {"desired": "running", "reason": "launch"},
        "peer:b01-c0": {"desired": "running", "reason": "launch"},
    }
    _write_json(control / "desired.json", {"jobs": jobs})
    _write_json(
        control / "status.json",
        {
            "supervisor_pid": supervisor_pid,
            "jobs": {
                "future:b01-c0": {
                    "actual": "running",
                    "campaign": str(campaign),
                    "progress": {"active_opportunities": 1},
                },
                "peer:b01-c0": {
                    "actual": "running",
                    "campaign": str(tmp_path / "peer"),
                    "progress": {"active_opportunities": 0},
                },
            },
        },
    )
    _write_json(
        control / "supervisor-metadata.json",
        {
            "profile": "future",
            "screen_session": "rl4rl-c0c3-future",
            "jobs": {
                "future:b01-c0": {"campaign": str(campaign)},
                "peer:b01-c0": {"campaign": str(tmp_path / "peer")},
            },
        },
    )
    return campaign, control


def test_overnight_campaign_pause_updates_only_matching_trajectory_jobs(
    tmp_path: Path,
) -> None:
    campaign, control = _overnight_fixture(tmp_path, supervisor_pid=os.getpid())
    campaigns = {"future": campaign}

    before = campaign_lifecycle_payload(campaigns, repo_root=tmp_path, sessions=set())[
        "campaigns"
    ][0]
    assert before["backend"] == "overnight_supervisor"
    assert before["actual"] == "running"
    assert before["active_opportunities"] == 1
    assert before["task_display_name"] == "Future task"
    assert before["research_architecture_label"] == "Native OpenEvolve"
    assert before["protocol_version"] == "3.0"

    receipt = set_campaign_lifecycle(
        campaigns,
        campaign_id="future",
        desired="paused",
        reason="test safe pause",
        repo_root=tmp_path,
        sessions=set(),
    )

    desired = json.loads((control / "desired.json").read_text(encoding="utf-8"))
    assert desired["jobs"]["future:b01-c0"]["desired"] == "paused"
    assert desired["jobs"]["peer:b01-c0"]["desired"] == "running"
    assert receipt["semantics"] == (
        "cooperative_safe_boundary_no_active_work_interruption"
    )
    assert (tmp_path / "data/dashboard-lifecycle.jsonl").is_file()


def test_overnight_resume_relaunches_absent_supervisor(tmp_path: Path) -> None:
    campaign, control = _overnight_fixture(tmp_path, supervisor_pid=999_999_999)
    desired = json.loads((control / "desired.json").read_text(encoding="utf-8"))
    desired["jobs"]["future:b01-c0"]["desired"] = "paused"
    _write_json(control / "desired.json", desired)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "started", "")

    set_campaign_lifecycle(
        {"future": campaign},
        campaign_id="future",
        desired="running",
        reason="test resume",
        repo_root=tmp_path,
        sessions=set(),
        run_command=fake_run,
    )

    updated = json.loads((control / "desired.json").read_text(encoding="utf-8"))
    assert updated["jobs"]["future:b01-c0"]["desired"] == "running"
    assert len(commands) == 1
    assert commands[0][-2:] == ["start", "--recover-interrupted"]


def test_semantic_campaign_pause_and_resume_use_durable_control(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "data/c0c3/semantic-campaign"
    _write_json(campaign / "campaign.json", {"study_id": "semantic"})
    _write_json(
        campaign / "semantic-control.json",
        {"desired": "running", "reason": "launch"},
    )
    _write_json(
        campaign / "runs/b01-c0/state.json",
        {"status": "running", "active": {"index": 3}},
    )
    pinned_runtime = tmp_path / "pinned-runtime"
    pinned_science = tmp_path / "pinned-science"
    _write_json(
        campaign / "semantic-supervisor.json",
        {
            "runtime_root": str(pinned_runtime),
            "scientific_repo_root": str(pinned_science),
            "python_bin": str(tmp_path / "python"),
            "max_workers": 7,
        },
    )
    campaigns = {"semantic": campaign}
    session = semantic_screen_session(campaign)

    set_campaign_lifecycle(
        campaigns,
        campaign_id="semantic",
        desired="paused",
        reason="test semantic pause",
        repo_root=tmp_path,
        sessions={session},
    )
    control = json.loads(
        (campaign / "semantic-control.json").read_text(encoding="utf-8")
    )
    assert control["desired"] == "paused"

    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "started", "")

    set_campaign_lifecycle(
        campaigns,
        campaign_id="semantic",
        desired="running",
        reason="test semantic resume",
        repo_root=tmp_path,
        sessions=set(),
        run_command=fake_run,
    )
    assert len(commands) == 1
    assert "resume" in commands[0]
    assert commands[0][commands[0].index("--runtime-root") + 1] == str(
        pinned_runtime.resolve()
    )
    assert commands[0][commands[0].index("--scientific-repo-root") + 1] == str(
        pinned_science.resolve()
    )
    assert commands[0][commands[0].index("--max-workers") + 1] == "7"
    control = json.loads(
        (campaign / "semantic-control.json").read_text(encoding="utf-8")
    )
    assert control["desired"] == "running"


def test_future_campaign_directories_are_discovered_automatically(
    tmp_path: Path,
) -> None:
    configured = tmp_path / "configured"
    future = tmp_path / "campaigns/future-campaign"
    _write_json(configured / "campaign.json", {})
    _write_json(future / "campaign.json", {})
    (tmp_path / "campaigns/not-a-campaign").mkdir(parents=True)

    campaigns = discover_campaigns(
        {"configured": configured}, roots=(tmp_path / "campaigns",)
    )

    assert campaigns["configured"] == configured.resolve()
    assert campaigns["discovered_future_campaign"] == future.resolve()
    assert len(campaigns) == 2

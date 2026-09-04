from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from experiments import c0c3_v3_repair as repair


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _completed(
    opportunity: int, *, failure_kind: str | None, valid: bool, tokens: int
) -> dict[str, object]:
    return {
        "event": "proposal_completed",
        "opportunity": opportunity,
        "candidate_id": f"candidate-{opportunity}",
        "evaluation": {
            "failure_kind": failure_kind,
            "valid": valid,
        },
        "retained": False,
        "usage_cumulative": {
            "input_tokens": tokens,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
        },
        "evaluator_calls_cumulative": 1 if valid else 0,
        "evaluator_seconds_cumulative": 2.5 if valid else 0.0,
        "incumbent_after": "seed",
        "portfolio_after": ["seed"],
    }


def test_apply_quarantines_only_an_infrastructure_failure_suffix(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    run = campaign / "runs" / "b01-c0"
    _write_json(campaign / "campaign.json", {"campaign_id": "toy"})
    state = {
        "run_id": "b01-c0",
        "status": "running",
        "active": {"index": 3},
        "next_opportunity": 3,
        "proposals_used": 2,
        "evaluations_used": 1,
        "evaluator_seconds_used": 2.5,
        "usage": {
            "input_tokens": 99,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
        },
        "incumbent_id": "seed",
        "portfolio_ids": ["seed"],
        "revision": 8,
        "candidates": {"seed": {"created_opportunity": 0, "selected_count": 2}},
    }
    _write_json(run / "state.json", state)
    records = [
        {"event": "run_created"},
        {
            "event": "proposal_started",
            "opportunity": 1,
            "selected_parent_ids": ["seed"],
        },
        _completed(1, failure_kind=None, valid=True, tokens=7),
        {
            "event": "proposal_started",
            "opportunity": 2,
            "selected_parent_ids": ["seed"],
        },
        _completed(
            2,
            failure_kind="infrastructure_interruption",
            valid=False,
            tokens=99,
        ),
        {
            "event": "proposal_started",
            "opportunity": 3,
            "selected_parent_ids": ["seed"],
        },
    ]
    (run / "events.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    (run / "opportunities/0002").mkdir(parents=True)
    (run / "opportunities/0002/recovery.json").write_text("{}\n", encoding="utf-8")
    (run / "pause-request.json").write_text("{}\n", encoding="utf-8")

    plan = repair._repair_plan(run)
    quarantine = repair._apply(campaign, [plan], reason="test repair")

    repaired = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert repaired["active"] is None
    assert repaired["next_opportunity"] == 2
    assert repaired["proposals_used"] == 1
    assert repaired["usage"]["input_tokens"] == 7
    assert repaired["candidates"]["seed"]["selected_count"] == 1
    assert not (run / "opportunities/0002").exists()
    kept = [
        json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
    ]
    assert [
        record.get("opportunity") for record in kept if "opportunity" in record
    ] == [1, 1]
    assert (quarantine / "b01-c0/events-before-repair.jsonl").is_file()
    assert (quarantine / "b01-c0/opportunities/0002/recovery.json").is_file()


def test_timestamp_rewind_removes_post_pause_valid_proposals_and_lifecycle(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    run = campaign / "runs" / "b01-c0"
    _write_json(campaign / "campaign.json", {"campaign_id": "toy"})
    _write_json(
        run / "state.json",
        {
            "run_id": "b01-c0",
            "status": "paused",
            "active": None,
            "next_opportunity": 3,
            "proposals_used": 2,
            "evaluations_used": 2,
            "evaluator_seconds_used": 5.0,
            "usage": {
                "input_tokens": 11,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_output_tokens": 0,
            },
            "incumbent_id": "candidate-2",
            "portfolio_ids": ["candidate-2", "seed"],
            "revision": 9,
            "candidates": {
                "seed": {"created_opportunity": 0, "selected_count": 1},
                "candidate-2": {
                    "created_opportunity": 2,
                    "artifact_path": "candidates/candidate-2",
                    "selected_count": 0,
                },
            },
        },
    )
    completed_one = _completed(1, failure_kind=None, valid=True, tokens=7)
    completed_one["timestamp"] = "2026-08-28T10:00:00+00:00"
    completed_two = _completed(2, failure_kind=None, valid=True, tokens=11)
    completed_two["timestamp"] = "2026-09-02T03:01:00+00:00"
    records = [
        {"event": "run_created", "timestamp": "2026-08-28T09:00:00+00:00"},
        {
            "event": "proposal_started",
            "opportunity": 1,
            "selected_parent_ids": ["seed"],
            "timestamp": "2026-08-28T10:00:00+00:00",
        },
        completed_one,
        {
            "event": "proposal_started",
            "opportunity": 2,
            "selected_parent_ids": ["seed"],
            "timestamp": "2026-09-02T03:00:01+00:00",
        },
        completed_two,
    ]
    (run / "events.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    (run / "opportunities/0002").mkdir(parents=True)
    (run / "candidates/candidate-2").mkdir(parents=True)
    (run / "candidates/candidate-2/model.py").write_text("# test\n")
    lifecycle = [
        {"event": "before", "timestamp": "2026-08-28T10:00:00+00:00"},
        {"event": "after", "timestamp": "2026-09-02T03:01:00+00:00"},
    ]
    for path in [run / "lifecycle.jsonl", campaign / "trajectory-lifecycle.jsonl"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record) + "\n" for record in lifecycle))

    boundary = dt.datetime(2026, 9, 2, 3, tzinfo=dt.UTC)
    plan = repair._timestamp_rewind_plan(run, boundary)
    assert plan is not None
    assert plan["cutoff"] == 2
    assert plan["future_candidate_ids"] == ["candidate-2"]
    quarantine = repair._apply(campaign, [plan], reason="return to pre-pause state")

    repaired = json.loads((run / "state.json").read_text(encoding="utf-8"))
    assert repaired["next_opportunity"] == 2
    assert repaired["proposals_used"] == 1
    assert repaired["incumbent_id"] == "seed"
    assert repaired["portfolio_ids"] == ["seed"]
    assert set(repaired["candidates"]) == {"seed"}
    assert not (run / "opportunities/0002").exists()
    assert not (run / "candidates/candidate-2").exists()
    assert (
        quarantine
        / "b01-c0/candidate-artifacts/candidates/candidate-2/model.py"
    ).is_file()
    retained_events = [
        json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
    ]
    opportunities = [
        record.get("opportunity")
        for record in retained_events
        if "opportunity" in record
    ]
    assert opportunities == [1, 1]
    retained_lifecycle = [
        json.loads(line)
        for line in (campaign / "trajectory-lifecycle.jsonl").read_text().splitlines()
    ]
    assert retained_lifecycle == [lifecycle[0]]
    assert (quarantine / "lifecycle-before-rewind/trajectory-lifecycle.jsonl").is_file()


def test_timestamp_rewind_archives_failed_suffix_candidate_artifact(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    run = campaign / "runs" / "b01-c0"
    _write_json(campaign / "campaign.json", {"campaign_id": "toy"})
    _write_json(
        run / "state.json",
        {
            "run_id": "b01-c0",
            "status": "running",
            "active": None,
            "next_opportunity": 3,
            "proposals_used": 2,
            "evaluations_used": 1,
            "evaluator_seconds_used": 2.5,
            "usage": {
                "input_tokens": 7,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_output_tokens": 0,
            },
            "incumbent_id": "seed",
            "portfolio_ids": ["seed"],
            "revision": 3,
            "candidates": {
                "seed": {"created_opportunity": 0, "selected_count": 2}
            },
        },
    )
    completed_one = _completed(1, failure_kind=None, valid=True, tokens=7)
    completed_one["timestamp"] = "2026-08-28T10:00:00+00:00"
    completed_two = _completed(
        2,
        failure_kind="nonqualification",
        valid=False,
        tokens=7,
    )
    completed_two.update(
        {
            "timestamp": "2026-09-02T03:01:00+00:00",
            "artifact_path": "candidates/failed-candidate",
        }
    )
    records = [
        {"event": "run_created", "timestamp": "2026-08-28T09:00:00+00:00"},
        {
            "event": "proposal_started",
            "opportunity": 1,
            "selected_parent_ids": ["seed"],
            "timestamp": "2026-08-28T10:00:00+00:00",
        },
        completed_one,
        {
            "event": "proposal_started",
            "opportunity": 2,
            "selected_parent_ids": ["seed"],
            "timestamp": "2026-09-02T03:00:01+00:00",
        },
        completed_two,
    ]
    (run / "events.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    (run / "opportunities/0002").mkdir(parents=True)
    (run / "candidates/failed-candidate").mkdir(parents=True)
    (run / "candidates/failed-candidate/model.py").write_text(
        "# failed candidate\n", encoding="utf-8"
    )

    boundary = dt.datetime(2026, 9, 2, 3, tzinfo=dt.UTC)
    plan = repair._timestamp_rewind_plan(run, boundary)
    assert plan is not None
    quarantine = repair._apply(campaign, [plan], reason="provider outage")

    assert not (run / "candidates/failed-candidate").exists()
    assert (
        quarantine
        / "b01-c0/candidate-artifacts/candidates/failed-candidate/model.py"
    ).is_file()

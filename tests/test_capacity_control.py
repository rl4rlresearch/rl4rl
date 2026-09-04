from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.c0c3_factorial.capacity_control import (
    campaign_evaluator_slot_root,
    load_campaign_capacity,
    release_campaign_evaluator,
    set_campaign_capacity,
    try_acquire_campaign_evaluator,
)
from experiments.c0c3_factorial.evaluator import shared_local_evaluator_status
from experiments.c0c3_factorial.semantic_interventions import load_intervention_plan


def _campaign(tmp_path: Path) -> Path:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "campaign.json").write_text("{}\n", encoding="utf-8")
    (campaign / "semantic-control.json").write_text(
        '{"desired":"running","reason":"test"}\n', encoding="utf-8"
    )
    return campaign


def test_campaign_capacity_defaults_and_atomic_update(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)

    initial = load_campaign_capacity(
        campaign,
        default_subject_workers=0,
        default_local_evaluators=6,
    )
    assert initial.subject_workers == 30
    assert initial.local_evaluators == 6

    changed = set_campaign_capacity(
        campaign,
        subject_workers=9,
        local_evaluators=4,
    )
    stored = json.loads((campaign / "capacity-control.json").read_text())
    lifecycle = json.loads((campaign / "semantic-control.json").read_text())

    assert changed.subject_workers == 9
    assert changed.local_evaluators == 4
    assert (
        stored["semantics"]
        == "live_future_dispatches_only_no_lifecycle_change"
    )
    assert lifecycle == {"desired": "running", "reason": "test"}
    assert load_campaign_capacity(
        campaign,
        default_subject_workers=30,
        default_local_evaluators=6,
    ) == changed


@pytest.mark.parametrize(
    ("workers", "evaluators"),
    [(0, 1), (31, 1), (1, 0)],
)
def test_campaign_capacity_rejects_out_of_range_limits(
    tmp_path: Path, workers: int, evaluators: int
) -> None:
    campaign = _campaign(tmp_path)

    with pytest.raises(ValueError):
        set_campaign_capacity(
            campaign,
            subject_workers=workers,
            local_evaluators=evaluators,
        )


def test_campaign_capacity_can_expand_past_legacy_evaluator_limit(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)

    changed = set_campaign_capacity(
        campaign,
        subject_workers=1,
        local_evaluators=20,
    )

    assert changed.local_evaluators == 20


def test_campaign_evaluator_pool_honors_current_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign(tmp_path)
    monkeypatch.setenv("RL4RL_CAMPAIGN_EVALUATOR_ROOT", str(tmp_path / "slots"))
    opportunity = campaign / "runs/example/opportunities/0001"
    opportunity.mkdir(parents=True)

    first = try_acquire_campaign_evaluator(
        campaign, capacity=2, opportunity_root=opportunity
    )
    second = try_acquire_campaign_evaluator(
        campaign, capacity=2, opportunity_root=opportunity.with_name("0002")
    )
    third = try_acquire_campaign_evaluator(
        campaign, capacity=2, opportunity_root=opportunity.with_name("0003")
    )
    try:
        assert first is not None
        assert second is not None
        assert third is None
        assert campaign_evaluator_slot_root(campaign).parent == tmp_path / "slots"
    finally:
        release_campaign_evaluator(first)
        release_campaign_evaluator(second)


def test_semantic_v4_fashion_task_ceiling_is_six() -> None:
    plan = load_intervention_plan(
        "experiments/c0c3_factorial/configs/interventions/semantic_research_v4.toml"
    )

    assert plan.task_evaluator_capacity == 6


def test_task_scheduler_can_expand_without_breaking_older_readers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "task-slots"

    assert shared_local_evaluator_status(root, 3)["capacity"] == 3
    assert shared_local_evaluator_status(root, 6)["capacity"] == 6
    assert json.loads((root / "scheduler.json").read_text())["capacity"] == 3
    assert json.loads((root / "operator-capacity.json").read_text())["capacity"] == 6
    assert shared_local_evaluator_status(root, 3)["capacity"] == 6

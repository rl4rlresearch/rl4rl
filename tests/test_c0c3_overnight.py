from __future__ import annotations

import json
from pathlib import Path

from experiments.c0c3_overnight import (
    CampaignPlan,
    Job,
    automatic_pause_reason,
    command_environment,
    command_for,
    expand_jobs,
    plans,
    progress_for,
    required_local_accelerator,
    select_jobs,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_default_roster_excludes_autoresearch_v14() -> None:
    assert "autoresearch-v1.4" not in {plan.key for plan in plans()}


def test_1644_extension_profile_contains_only_extension_blocks() -> None:
    roster = plans("1644-extension")

    assert len(roster) == 1
    assert roster[0].key == "autoresearch-v1.5-1644-extension"
    assert roster[0].blocks == (2, 3)


def test_1644_confined_profile_declares_all_three_blocks() -> None:
    roster = plans("1644-confined")

    assert len(roster) == 1
    assert roster[0].key == "autoresearch-v1.6-1644-confined"
    assert roster[0].blocks == (1, 2, 3)


def test_1644_confined_fresh_profile_is_independent_and_declares_three_blocks() -> None:
    roster = plans("1644-confined-fresh")

    assert len(roster) == 1
    assert roster[0].key == "autoresearch-v1.6-1644-confined-fresh"
    assert roster[0].blocks == (1, 2, 3)
    assert roster[0].pause_after_proposals == 75
    assert roster[0].campaign.name.endswith("campaign-fresh-20260822c")


def test_fresh_profile_jobs_inherit_proposal_75_pause_boundary(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    _write_json(
        campaign / "schedule.json",
        [{"block": 1, "condition": "C0", "run_id": "b1-C0"}],
    )
    plan = CampaignPlan(
        key="v16",
        runtime_root=tmp_path,
        campaign=campaign,
        mode="individual-trajectories",
        blocks=(1,),
        pause_after_proposals=75,
    )

    jobs = expand_jobs((plan,))

    assert len(jobs) == 1
    assert jobs[0].pause_after_proposals == 75


def test_automatic_pause_arms_during_configured_proposal(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    run_id = "b1-C0"
    _write_json(
        campaign / "schedule.json",
        [{"block": 1, "condition": "C0", "run_id": run_id}],
    )
    state_path = campaign / "runs" / run_id / "state.json"
    job = Job(
        key="v16:b01-c0",
        group="v16",
        runtime_root=tmp_path,
        campaign=campaign,
        mode="individual-trajectories",
        run_id=run_id,
        blocks=(1,),
        pause_after_proposals=75,
    )
    _write_json(
        state_path,
        {"proposals_used": 74, "active": {"index": 74}},
    )
    assert automatic_pause_reason(job) is None

    _write_json(
        state_path,
        {"proposals_used": 74, "active": {"index": 75}},
    )
    assert automatic_pause_reason(job) == "automatic pause after proposal 75"

    _write_json(state_path, {"proposals_used": 75, "active": None})
    assert automatic_pause_reason(job) == "automatic pause after proposal 75"


def test_openevolve_v2_profile_declares_three_individually_controlled_blocks() -> None:
    roster = plans("openevolve-v2")

    assert len(roster) == 1
    assert roster[0].key == "openevolve-v2"
    assert roster[0].mode == "individual-trajectories"
    assert roster[0].blocks == (1, 2, 3)
    assert roster[0].campaign.name == (
        "controlled-openevolve-transformer-v2-mps-campaign"
    )


def test_artifact_clean_profiles_declare_all_primary_blocks() -> None:
    for profile, key, blocks in (
        ("autoresearch-v1.7", "autoresearch-v1.7", (1, 2)),
        ("openevolve-v2.1", "openevolve-v2.1", (1, 2, 3, 4, 5)),
    ):
        roster = plans(profile)
        assert len(roster) == 1
        assert roster[0].key == key
        assert roster[0].mode == "individual-trajectories"
        assert roster[0].blocks == blocks


def test_artifact_clean_jobs_receive_main_operator_prompt_root(tmp_path: Path) -> None:
    for group in ("autoresearch-v1.7", "openevolve-v2.1"):
        job = Job(
            key=f"{group}:b01-c0",
            group=group,
            runtime_root=tmp_path,
            campaign=tmp_path / "campaign",
            mode="individual-trajectories",
            run_id="b01-c0",
        )
        environment = command_environment(job)
        assert environment["RL4RL_C0C3_OPERATOR_PROMPT_ROOT"].endswith(
            "experiments/c0c3_factorial/templates"
        )


def test_local_accelerator_is_derived_from_frozen_task_input(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _write_json(
        campaign / "inputs/task.json",
        {
            "preferred_backend": "local",
            "evaluator_command": ["python", "train.py", "--train-device", "mps"],
        },
    )
    assert required_local_accelerator(campaign) == "mps"

    _write_json(
        campaign / "inputs/task.json",
        {
            "preferred_backend": "hybrid_modal",
            "evaluator_command": [
                "python",
                "train.py",
                "--train-device",
                "cuda",
            ],
        },
    )
    assert required_local_accelerator(campaign) is None


def test_individual_plan_expands_only_declared_factorial_blocks(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _write_json(
        campaign / "schedule.json",
        [
            {"block": block, "condition": condition, "run_id": f"b{block}-{condition}"}
            for block in (1, 2, 3)
            for condition in ("C0", "C1", "C2", "C3", "N0")
        ],
    )
    plan = CampaignPlan(
        key="v15",
        runtime_root=tmp_path,
        campaign=campaign,
        mode="individual-trajectories",
        blocks=(1, 2),
    )

    jobs = expand_jobs((plan,))

    assert len(jobs) == 8
    assert {job.key for job in jobs} == {
        f"v15:b{block:02d}-c{condition}" for block in (1, 2) for condition in range(4)
    }
    assert all(job.run_id and "N0" not in job.run_id for job in jobs)


def test_progress_counts_input_plus_output_without_double_counting_cache(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    run_id = "b1-C0"
    _write_json(
        campaign / "schedule.json",
        [{"block": 1, "condition": "C0", "run_id": run_id}],
    )
    _write_json(
        campaign / "runs" / run_id / "state.json",
        {
            "status": "running",
            "active": None,
            "proposals_used": 7,
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 80,
                "output_tokens": 20,
                "reasoning_output_tokens": 10,
            },
            "candidates": {
                "valid": {"metrics": {"accuracy": 0.99, "parameters": 1234}},
                "invalid": {"metrics": {"accuracy": 0.98, "parameters": 1}},
            },
        },
    )
    job = Job(
        key="v15:b01-c0",
        group="v15",
        runtime_root=tmp_path,
        campaign=campaign,
        mode="individual-trajectories",
        run_id=run_id,
        blocks=(1,),
    )

    progress = progress_for(job)

    assert progress["total_tokens"] == 120
    assert progress["lowest_parameters"] == 1234
    assert progress["min_proposals"] == progress["max_proposals"] == 7


def test_resume_command_is_selected_for_started_v15_trajectory(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    run_id = "run-c0"
    _write_json(
        campaign / "runs" / run_id / "state.json",
        {"proposals_used": 1},
    )
    job = Job(
        key="v15:b01-c0",
        group="v15",
        runtime_root=tmp_path,
        campaign=campaign,
        mode="individual-trajectories",
        run_id=run_id,
        blocks=(1,),
    )

    assert "resume-staged-trajectory" in command_for(job)


def test_group_selector_controls_every_job_in_group(tmp_path: Path) -> None:
    jobs = [
        Job(
            key=f"v15:{condition}",
            group="v15",
            runtime_root=tmp_path,
            campaign=tmp_path,
            mode="individual-trajectories",
            run_id=condition,
        )
        for condition in ("c0", "c1", "c2", "c3")
    ]

    assert select_jobs(jobs, ["v15"]) == jobs

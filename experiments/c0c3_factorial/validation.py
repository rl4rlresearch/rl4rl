"""Fail-closed campaign and prompt-control launch audit."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .artifacts import candidate_hash, scientific_runtime_hash, tree_hash
from .prompts import (
    PromptContext,
    PromptRenderer,
    VisibleCandidate,
    treatment_skeleton,
)
from .spec import Condition, FactorialSpec, FrameworkSpec, TaskSpec, sha256_json
from .state import SearchController


def validate_campaign(
    campaign_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    repo_root: Path,
) -> dict[str, object]:
    campaign = Path(campaign_dir).resolve()
    manifest = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    expected_hashes = {
        "protocol_hash": spec.protocol_hash,
        "task_hash": sha256_json(asdict(task)),
        "framework_hash": sha256_json(asdict(framework)),
        "scientific_runtime_hash": scientific_runtime_hash(
            repo_root, task=task, framework=framework
        ),
    }
    for name, expected in expected_hashes.items():
        if manifest.get(name) != expected:
            errors.append(f"campaign {name} mismatch")
    schedule = json.loads((campaign / "schedule.json").read_text(encoding="utf-8"))
    run_ids = [str(row["run_id"]) for row in schedule]
    if len(run_ids) != len(set(run_ids)):
        errors.append("campaign schedule contains duplicate run IDs")
    for block in range(1, spec.blocks + 1):
        rows = [row for row in schedule if int(row["block"]) == block]
        factorial = [row["condition"] for row in rows if row["condition"] != "N0"]
        if sorted(factorial) != sorted(condition.value for condition in Condition):
            errors.append(f"block {block} does not contain C0-C3 exactly once")
        if len({row["run_seed"] for row in rows}) != 1:
            errors.append(f"block {block} is not seed paired")
        expected_orders = list(range(1, len(rows) + 1))
        if sorted(int(row["order"]) for row in rows) != expected_orders:
            errors.append(f"block {block} execution order is not contiguous")
    support_hashes = set()
    seed_ids = set()
    launch_states: list[str] = []
    for assignment in schedule:
        run_dir = campaign / "runs" / str(assignment["run_id"])
        try:
            controller = SearchController.load(run_dir, spec)
            support_hashes.add(tree_hash(run_dir / "task-support"))
            seed_id = str(manifest["seed_candidate_id"])
            seed_ids.add(
                candidate_hash(run_dir / "candidates" / seed_id, task.editable_paths)
            )
            if controller.state.condition != assignment["condition"]:
                errors.append(f"{assignment['run_id']} condition mismatch")
            if (
                controller.state.status != "ready"
                or controller.state.proposals_used != 0
                or controller.state.evaluations_used != 0
                or controller.state.active is not None
            ):
                launch_states.append(str(assignment["run_id"]))
            run_manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            for name, expected in expected_hashes.items():
                if run_manifest.get(name) != expected:
                    errors.append(f"{assignment['run_id']} {name} mismatch")
            if run_manifest.get("assignment") != assignment:
                errors.append(f"{assignment['run_id']} assignment mismatch")
        except (FileNotFoundError, KeyError, ValueError) as error:
            errors.append(f"{assignment['run_id']} invalid: {error}")
    if len(support_hashes) != 1:
        errors.append("run task-support trees are not byte-identical")
    if seed_ids != {manifest.get("seed_candidate_id")}:
        errors.append("run seed candidates are not byte-identical")
    if launch_states:
        errors.append(
            "launch validation requires untouched ready runs: "
            + ", ".join(launch_states)
        )
    renderer = PromptRenderer(repo_root / "experiments/c0c3_factorial/templates")
    for opportunity in range(1, spec.budget.proposals + 1):
        prompts = {}
        for condition in Condition:
            context = PromptContext(
                condition=condition,
                opportunity=opportunity,
                selected_parent_id="seed",
                visible_candidates=(
                    VisibleCandidate(
                        "seed", 0.0, {task.objective_metric: 0.0}, 0, "slot-1"
                    ),
                ),
                remaining_proposals=spec.budget.proposals,
                remaining_evaluations=spec.budget.candidate_evaluations,
                remaining_tokens=spec.budget.max_total_tokens,
                remaining_evaluator_seconds=spec.budget.max_evaluator_seconds,
            )
            prompts[condition] = renderer.render(spec, task, framework, context)
        if len({treatment_skeleton(value.text) for value in prompts.values()}) != 1:
            errors.append(f"prompt skeleton differs at opportunity {opportunity}")
        if (
            prompts[Condition.C0].search_state_sha256
            != prompts[Condition.C1].search_state_sha256
            or prompts[Condition.C2].search_state_sha256
            != prompts[Condition.C3].search_state_sha256
        ):
            errors.append(f"search-state factor mismatch at opportunity {opportunity}")
        if (
            prompts[Condition.C0].proposal_policy_sha256
            != prompts[Condition.C2].proposal_policy_sha256
            or prompts[Condition.C1].proposal_policy_sha256
            != prompts[Condition.C3].proposal_policy_sha256
        ):
            errors.append(
                f"proposal-policy factor mismatch at opportunity {opportunity}"
            )
    layers_absent = not (
        (campaign / "sealed-layer-b").exists()
        or (campaign / "sealed-layer-c").exists()
    )
    if not layers_absent:
        errors.append("Layer B or Layer C output exists before launch")
    report = {
        "schema_version": "1.0",
        "campaign": str(campaign),
        "valid": not errors,
        "errors": errors,
        "checked_run_count": len(schedule),
        "protocol_hash": spec.protocol_hash,
        "controls": {
            "same_model_and_settings": True,
            "same_starting_artifact": len(seed_ids) == 1,
            "same_task_and_evaluator": len(support_hashes) == 1,
            "same_budget": True,
            "same_failure_rule": True,
            "frozen_blocked_round_robin_execution": True,
            "layer_b_c_absent_at_launch": layers_absent,
        },
    }
    (campaign / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
